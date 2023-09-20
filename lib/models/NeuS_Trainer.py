import os
import cv2
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from lib.utils.builder import MODEL
from lib.metrics import LossMetric
from lib.metrics.similarity import PSNR, mse2psnr, SSIM
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.utils.recorder import Recorder
from lib.utils.net_utils import init_weights
from lib.utils.etqdm import etqdm
from lib.models.renderers import build_renderer
from lib.models.model_abstraction import ModuleAbstract
from lib.models.tools.camera_net import Focal_Net, Pose_Net
from lib.models.tools.ray_utils import get_rays_multicam, get_rays_at, near_far_from_sphere
from lib.models.tools.viztools import cmap, plot_camera_scene, plot_cameras_track


@MODEL.register_module()
class NeuS_Trainer(ModuleAbstract, nn.Module):

    def __init__(self, cfg, data=None):
        super(NeuS_Trainer, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        self.n_rays = cfg.N_RAYS
        self.include_mask = cfg.DATA_PRESET.INCLUDE_MASK
        self.fx_only = cfg.DATA_PRESET.get('FX_ONLY', False)
        self.normalize_dir = cfg.get('NORMALIZE_DIR', True)
        self.opengl_sys = cfg.DATA_PRESET.get('OPENGL_SYS', False)
        self.get_mesh_set = cfg.DATA_PRESET.get('GET_MESH_SET', False)
        self.eval_ray_size = cfg.get('EVAL_RAY_SIZE', 10000)
        if self.include_mask:
            self.mask_rate = cfg.MASK_RATE
        else:
            self.mask_rate = None

        self.origin = data['origin']
        self.radius = data['radius']
        init_focal = data['focal']
        init_c2w = data['poses']
        self.n_imgs = data['n_imgs']
        self.H = data['H']
        self.W = data['W']
        assert init_focal.shape[0] == 1 or init_focal.shape[0] == 2, f"get wrong focal with size: {init_focal.shape[0]}"

        self.focal_net = Focal_Net(H=self.H,
                                   W=self.W,
                                   req_grad=cfg.LEARN_FOCAL,
                                   fx_only=self.fx_only,
                                   order=cfg.FOCAL_ORDER,
                                   init_focal=init_focal)
        self.pose_net = Pose_Net(num_cams=self.n_imgs,
                                 learn_R=cfg.LEARN_R,
                                 learn_t=cfg.LEARN_T,
                                 pose_mode=cfg.POSE_MODE,
                                 init_c2w=init_c2w)
        self.renderer = build_renderer(cfg.RENDERER)

        self.lambda_fine = cfg.LOSS.get('LAMBDA_FINE', 1.0)
        self.lambda_eikonal = cfg.LOSS.get('LAMBDA_EIKONAL', 0.1)
        self.lambda_mask = cfg.LOSS.get('LAMBDA_MASK', 0.0)
        self.lambda_relight = cfg.LOSS.get('LAMBDA_RELIGHT', 1.0)
        rgb_loss_type = cfg.LOSS.get('RGB_LOSS_TYPE', 'mse')
        assert self.lambda_fine != 0 and self.lambda_eikonal != 0
        if cfg.RENDERER.TYPE == 'Color_NeuS':
            assert self.lambda_relight != 0
        if rgb_loss_type == 'mse':
            self.rgb_loss = torch.nn.MSELoss()
        elif rgb_loss_type == 'l1':
            self.rgb_loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"no such rgb loss type: {rgb_loss_type}")

        self.scale_mats_np = data['scale_mats_np']
        self.object_bbox_min = data['object_bbox_min']
        self.object_bbox_max = data['object_bbox_max']

        self.loss_metric = LossMetric(cfg)
        self.PSNR = PSNR(cfg)
        self.SSIM = SSIM(cfg)

        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} got n_rays: {self.n_rays}")
        logger.info(f"{self.name} got normalize_dir: {self.normalize_dir}")
        logger.info(f"{self.name} got include_mask: {self.include_mask} with mask rate: {self.mask_rate}")
        logger.info(f"{self.name} got focal order: {cfg.FOCAL_ORDER}, learn_focal: {cfg.LEARN_FOCAL}")
        logger.info(f"{self.name} got learn_R: {cfg.LEARN_R}, learn_t: {cfg.LEARN_T}")
        logger.info(f"{self.name} got lambda_fine: {self.lambda_fine}")
        logger.info(f"{self.name} got lambda_eikonal: {self.lambda_eikonal}, lambda_mask: {self.lambda_mask}")
        logger.info(f"{self.name} got lambda_relight: {self.lambda_relight}")
        init_weights(self, pretrained=cfg.PRETRAINED, strict=True)

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def render(self, c2w, focal, image, img_id, step_idx, mask=None, **kwargs):
        device = c2w.device
        if self.mask_rate is not None:
            mask_rate = self.mask_rate[0] + \
                  (self.mask_rate[1] - self.mask_rate[0]) * (step_idx /self.cfg.TRAIN.ITERATIONS)
        else:
            mask_rate = None
        rays_o, rays_d, rgb_gt, mask_select = get_rays_multicam(
            c2w=c2w,
            focal=focal,
            image=image,
            n_rays=self.n_rays,
            normalize=self.normalize_dir,
            mask=mask,
            mask_rate=mask_rate,
            return_mask=True if self.include_mask else False,
            opengl=self.opengl_sys,
        )
        rays_o = (rays_o - self.origin.detach().clone().to(device)).float()
        rays_o = (rays_o / self.radius.detach().clone().to(device)).float()
        near, far = near_far_from_sphere(rays_o, rays_d)
        render_output = self.renderer(rays_o, rays_d, near, far)
        render_output['rgb_map_gt'] = rgb_gt
        render_output['mask'] = mask_select
        return render_output

    def compute_loss(self, render_dict, **kwargs):
        loss = 0
        rgb_gt = render_dict['rgb_map_gt']

        rgb_fine_render = render_dict['color_fine']
        rgb_fine_loss = self.rgb_loss(rgb_fine_render, rgb_gt)
        loss += self.lambda_fine * rgb_fine_loss

        eikonal_loss = render_dict['gradient_error']
        loss += self.lambda_eikonal * eikonal_loss

        if self.lambda_mask != 0:
            mask_loss = F.binary_cross_entropy(render_dict['weight_sum'].squeeze().clip(1e-3, 1.0 - 1e-3),
                                               render_dict['mask'])
            loss += self.lambda_mask * mask_loss

        if self.lambda_relight != 0:
            if self.include_mask:
                mask = render_dict['mask'].unsqueeze(-1)
                if render_dict['delta_relight'].dim() == 3:
                    mask = mask.unsqueeze(-1)
                delta_relight = render_dict['delta_relight'] * mask
            else:
                delta_relight = render_dict['delta_relight']
            relight_loss = F.mse_loss(torch.mean(delta_relight),
                                      torch.tensor(0, device=delta_relight.device, dtype=torch.float32))
            loss += self.lambda_relight * relight_loss

        loss_dict = {
            'loss': loss,
            'rgb_fine_loss': rgb_fine_loss,
            'eikonal_loss': eikonal_loss,
        }

        if self.lambda_mask != 0:
            loss_dict['mask_loss'] = mask_loss

        if self.lambda_relight != 0:
            loss_dict['relight_loss'] = relight_loss

        self.psnr_toshow = mse2psnr(rgb_fine_loss.detach().cpu()).item()

        return loss, loss_dict

    def training_step(self, batch, step_idx, **kwargs):
        images = batch['images']  # [N, 3, H, W] 0 ~ 1
        images = images.permute(0, 2, 3, 1).contiguous()  # [N, H, W, 3] 0 ~ 1
        img_ids = batch['img_ids']  # [N]
        N_images = images.shape[0]
        if self.include_mask:
            masks = batch['masks']
        else:
            masks = None

        focal = self.focal_net()
        c2w = self.pose_net(img_ids)

        render_dict = self.render(c2w=c2w,
                                  focal=focal,
                                  image=images,
                                  img_id=img_ids,
                                  step_idx=step_idx,
                                  mask=masks,
                                  **kwargs)

        loss, loss_dict = self.compute_loss(render_dict)

        self.loss_metric.feed(loss_dict, 1)

        if step_idx % self.cfg.TRAIN.LOG_INTERVAL == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)

            if step_idx % (self.cfg.TRAIN.LOG_INTERVAL * 50) == 0:
                all_poses = self.pose_net(torch.arange(self.n_imgs))
                cams_img = plot_camera_scene(all_poses, None, self.radius.cpu().numpy(), f'Iteration_{step_idx}')
                self.summary.add_image('poses', cams_img, step_idx)
                cam_track = plot_cameras_track(all_poses)
                self.summary.add_image('poses_track', cam_track, step_idx)

        return render_dict, loss_dict

    def on_train_finished(self, recorder: Recorder, epoch_idx):
        comment = f"{self.name}-train-"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def validate_image(self, batch, step_idx, **kwargs):
        logger.info("begin viz a image and save results")
        batch_images = batch['images']  # [N, 3, H, W] 0 ~ 1
        H, W = batch_images.shape[-2], batch_images.shape[-1]
        device = batch_images.device
        batch_size = batch_images.shape[0]
        rand_img_index = torch.randperm(batch_size)[0]
        pose_id = batch['img_ids'][rand_img_index]

        image_gt = batch_images[rand_img_index].permute(1, 2, 0).contiguous()  # [H, W, 3]
        c2w = self.pose_net(pose_id)
        focal = self.focal_net()

        rays_o, rays_d = get_rays_at(c2w, focal, H, W, normalize=self.normalize_dir, opengl=self.opengl_sys)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rays_o_dir = torch.split(rays_o, self.eval_ray_size, dim=0)
        rays_d_dir = torch.split(rays_d, self.eval_ray_size, dim=0)
        rgb_render = []
        depth_render = []

        for rays_o_i, rays_d_i in zip(etqdm(rays_o_dir), rays_d_dir):
            rays_o_i = (rays_o_i - self.origin.detach().clone().to(device)).float()
            rays_o_i = (rays_o_i / self.radius.detach().clone().to(device)).float()
            near_i, far_i = near_far_from_sphere(rays_o_i, rays_d_i)
            render_output_i = self.renderer(rays_o_i, rays_d_i, near_i, far_i)

            rgb_render.append(render_output_i['color_fine'].detach().cpu())
            depth_render.append(render_output_i['depth'].detach().cpu())

        rgb_render = torch.cat(rgb_render, dim=0)
        depth_render = torch.cat(depth_render, dim=0)

        image_list = []
        image_gt = image_gt.detach().cpu()
        image_gt_show = image_gt.mul(255.0).numpy().astype(np.uint8)
        image_list.append(image_gt_show)

        rgb_render = rgb_render.reshape(H, W, 3)
        rgb_render_show = rgb_render.mul(255.0).numpy().astype(np.uint8)
        image_list.append(rgb_render_show)

        depth_render = depth_render.reshape(H, W).numpy()
        depth_img = cmap(depth_render)
        image_list.append(depth_img)

        image_list = np.hstack(image_list)

        exp_path = kwargs.get('exp_path', None)
        if exp_path != None:
            viz_path = os.path.join('./exp', exp_path, 'viz_image')
            if not os.path.exists(viz_path):
                os.mkdir(viz_path)
        else:
            viz_path = './tmp/NeuS_Trainer/imgs'
            os.makedirs(viz_path, exist_ok=True)

        imageio.imwrite(f'{viz_path}/img_{step_idx}.png', image_list)

        self.PSNR.feed(rgb_render, image_gt)
        self.SSIM.feed(rgb_render.permute(2, 0, 1).unsqueeze(0), image_gt.permute(2, 0, 1).unsqueeze(0))

    def validate_mesh(self, step_idx, world_space=False, resolution=64, threshold=0.0, **kwargs):
        device = torch.device("cuda")
        bound_min = torch.tensor(self.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, device, resolution=resolution, threshold=threshold)

        mesh = trimesh.Trimesh(vertices, triangles)

        colors = self.renderer.extract_color(vertices, device)

        if world_space:
            vertices = vertices * self.scale_mats_np[0][0, 0] + self.scale_mats_np[0][:3, 3][None]

        # colors = colors[:, ::-1]  # BGR to RGB
        mesh_color = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)

        exp_path = kwargs.get('exp_path', None)
        if exp_path != None:
            mesh_path = os.path.join('./exp', exp_path, 'meshes')
            if not os.path.exists(mesh_path):
                os.mkdir(mesh_path)
        else:
            mesh_path = './tmp/NeuS_Trainer/meshes'
            os.makedirs(mesh_path, exist_ok=True)

        mesh.export(os.path.join(mesh_path, '{:0>8d}_mesh.ply'.format(step_idx)))
        mesh_color.export(os.path.join(mesh_path, '{:0>8d}_color.ply'.format(step_idx)))

    def validation_step(self, batch, step_idx, **kwargs):
        if step_idx % self.cfg.TRAIN.VIZ_IMAGE_INTERVAL == self.cfg.TRAIN.VIZ_IMAGE_INTERVAL - 1:
            self.validate_image(batch, step_idx, **kwargs)
        if step_idx % self.cfg.TRAIN.VIZ_MESH_INTERVAL == self.cfg.TRAIN.VIZ_MESH_INTERVAL - 1:
            self.validate_mesh(step_idx, world_space=True, resolution=512, threshold=0.0, **kwargs)

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_metric([self.PSNR, self.SSIM], epoch_idx, comment=comment)
        self.PSNR.reset()
        self.SSIM.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        self.validate_mesh(step_idx, world_space=True, resolution=kwargs['recon_res'], threshold=0.0, **kwargs)

    def on_test_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-test"
        recorder.record_metric([self.PSNR, self.SSIM], epoch_idx, comment=comment)
        self.PSNR.reset()
        self.SSIM.reset()

    def format_metric(self, mode="train"):
        if mode == "train":
            return (f"loss: {self.loss_metric.get_loss('loss'):.5f}, psnr: {self.psnr_toshow:.2f}")
        elif mode == "test":
            metric_toshow = [self.PSNR, self.SSIM]
        else:
            metric_toshow = [self.PSNR, self.SSIM]

        return " | ".join([str(me) for me in metric_toshow])

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")