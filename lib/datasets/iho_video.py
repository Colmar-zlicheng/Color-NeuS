import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as tvF
from lib.utils.builder import DATASET
from termcolor import colored
from lib.utils.logger import logger
from lib.utils.read_cameras import read_cameras_binary, read_images_binary, read_points3d_binary
from lib.utils.etqdm import etqdm
from lib.utils.transform import load_K_Rt_from_P


@DATASET.register_module()
class IHO_VIDEO(torch.utils.data.Dataset):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.std = 0.5
        self.data_root = cfg.DATA_ROOT
        self.obj_name = cfg.OBJ_ID
        self.fx_only = cfg.DATA_PRESET.get('FX_ONLY', False)
        self.include_mask = cfg.DATA_PRESET.get('INCLUDE_MASK', False)
        self.opengl_sys = cfg.DATA_PRESET.get('OPENGL_SYS', False)
        radius_ratio = cfg.get('RADIUS_RATIO', 1.5)
        self.data_path = os.path.join(self.data_root, 'IHO_video', self.obj_name)
        self.img_dir = os.path.join(self.data_path, 'obj')
        # self.mask_dir = os.path.join(self.data_path, 'obj_seg')

        camdata = read_cameras_binary(os.path.join(self.data_path, 'colmap/cameras.bin'))
        world_pts = read_points3d_binary(os.path.join(self.data_path, 'colmap/points3D.bin'))
        imdata = read_images_binary(os.path.join(self.data_path, 'colmap/images.bin'))
        imdata = sorted(imdata.items(), reverse=False)

        xyz_world = np.array([world_pts[p_id].xyz for p_id in world_pts])
        origin = xyz_world.mean(0)  # (3,)
        radius = np.percentile(np.sqrt(np.sum((xyz_world - origin), axis=1)**2), 99.9)
        self.origin = torch.tensor(origin, dtype=torch.float32)
        self.radius = torch.tensor(radius * radius_ratio, dtype=torch.float32)
        self.K = np.array([[camdata[1].params[0], 0, camdata[1].params[2]],
                           [0, camdata[1].params[1], camdata[1].params[3]], [0, 0, 1]])
        if self.fx_only:
            self.focal = np.array([(self.K[0, 0] + self.K[1, 1]) / 2], dtype=np.float32)
        else:
            self.focal = np.array([self.K[0, 0], self.K[1, 1]], dtype=np.float32)

        if self.opengl_sys:
            _coord_trans_OpenGL = np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

        self.poses = []
        self.image_paths = []
        self.img_ids = []
        for i, (_, im) in enumerate(imdata):
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            Rt = np.concatenate([R, t], axis=1)  # w2c
            P = self.K @ Rt  # (3, 4)
            _, pose = load_K_Rt_from_P(None, P)
            if self.opengl_sys:
                self.poses.append(torch.tensor(_coord_trans_OpenGL @ pose, dtype=torch.float32))  # c2w
            else:
                self.poses.append(torch.tensor(pose, dtype=torch.float32))  # c2w
            self.img_ids.append(torch.tensor([i]))
            self.image_paths.append(os.path.join(self.img_dir, im.name))

        self.img_ids = torch.cat(self.img_ids, dim=0)
        self.n_imgs = len(self.image_paths)

        self.scale_mats_np = []
        self.scale_mats_np = [np.identity(4) for idx in range(self.n_imgs)]
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0]).reshape((4, 1))
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0]).reshape((4, 1))
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        logger.info(f"{self.name}: {self.obj_name}, Got {colored(self.n_imgs, 'yellow', attrs=['bold'])}")
        logger.info(f"{self.name}: {self.obj_name},  include_mask: {self.include_mask}")

    def __len__(self):
        return self.n_imgs

    def get_image(self, idx):
        path = self.image_paths[idx]
        png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = png[:, :, :3].copy()
        mask = tvF.to_tensor(png[:, :, 3].copy()).squeeze()  # [H, W] 0 or 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tvF.to_tensor(image)
        assert image.shape[0] == 3
        image = tvF.normalize(image, [0.5, 0.5, 0.5], [self.std, self.std, self.std])
        image = image * 0.5 + 0.5  # [3, H, W] 0~1
        return image, mask

    def __getitem__(self, idx):
        sample = {}
        sample['pose'] = self.poses[idx]
        sample['img_id'] = self.img_ids[idx]

        if self.include_mask:
            sample['image'], sample['mask'] = self.get_image(idx)
        else:
            sample['image'], _ = self.get_image(idx)

        return sample

    def get_init_data(self):
        tmp_img, _ = self.get_image(0)
        return {
            'poses': torch.stack(self.poses, dim=0),
            'focal': self.focal,
            'H': tmp_img.shape[-2],
            'W': tmp_img.shape[-1],
            'n_imgs': self.n_imgs,
            'origin': self.origin,
            'radius': self.radius,
            'scale_mats_np': self.scale_mats_np,
            'object_bbox_min': self.object_bbox_min,
            'object_bbox_max': self.object_bbox_max
        }

    def get_all_img(self):
        logger.info("Loading all images ...")
        all_img = []
        if self.include_mask:
            all_mask = []
            for i in etqdm(range(self.n_imgs)):
                img, mask = self.get_image(i)
                all_img.append(img)
                all_mask.append(mask)
            all_mask = torch.stack(all_mask, dim=0)
        else:
            all_mask = None
            for i in etqdm(range(self.n_imgs)):
                img, _ = self.get_image(i)
                all_img.append(img)
        return {'images': torch.stack(all_img, dim=0), 'masks': all_mask, 'img_ids': self.img_ids}

    def get_all_init(self, batch_size):
        self.all_img_dict = self.get_all_img()
        self.batch_size = batch_size

    def get_rand_batch_smaples(self, device):
        sample = {}
        rand_index = torch.randperm(self.n_imgs)
        use_index = rand_index[:self.batch_size]

        sample['images'] = self.all_img_dict['images'][use_index].to(device)
        sample['img_ids'] = self.img_ids[use_index]

        if self.include_mask:
            sample['masks'] = self.all_img_dict['masks'][use_index].to(device)

        return sample