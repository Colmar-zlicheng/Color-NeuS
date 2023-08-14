import os
import cv2
import imageio
import torch
import numpy as np
import torchvision.transforms.functional as tvF
from termcolor import colored
from lib.utils.builder import DATASET
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.transform import load_K_Rt_from_P


@DATASET.register_module()
class BlendedMVS(torch.utils.data.Dataset):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.std = 0.5
        self.data_root = cfg.DATA_ROOT
        self.obj_id = cfg.OBJ_ID
        self.fx_only = cfg.DATA_PRESET.get('FX_ONLY', False)
        self.include_mask = cfg.DATA_PRESET.get('INCLUDE_MASK', True)
        self.opengl_sys = cfg.DATA_PRESET.get('OPENGL_SYS', False)
        self.data_path = os.path.join(self.data_root, 'BlendedMVS', f"bmvs_{self.obj_id}")
        self.img_dir = os.path.join(self.data_path, 'image')
        self.mask_dir = os.path.join(self.data_path, 'mask')
        self.camera_path = os.path.join(self.data_path, 'cameras_sphere.npz')

        images = sorted(os.listdir(self.img_dir))
        masks = sorted(os.listdir(self.mask_dir))
        self.image_paths = []
        self.mask_paths = []
        self.img_ids = []
        for i in range(len(images)):
            self.image_paths.append(os.path.join(self.img_dir, images[i]))
            self.mask_paths.append(os.path.join(self.mask_dir, masks[i]))
            self.img_ids.append(torch.tensor([i]))

        self.img_ids = torch.cat(self.img_ids, dim=0)
        self.n_imgs = len(self.image_paths)

        self.origin = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.radius = torch.tensor(1, dtype=torch.float32)

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

        camera_dict = np.load(self.camera_path)
        self.camera_dict = camera_dict

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_imgs)]
        self.scale_mats_np = []
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_imgs)]

        self.intrinsics_all = []
        self.poses = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(intrinsics)
            if self.opengl_sys:
                pose = _coord_trans_OpenGL @ pose
            self.poses.append(torch.from_numpy(pose).float())

        if self.fx_only:
            self.focal = np.array([self.intrinsics_all[0][0, 0]], dtype=np.float32)
        else:
            self.focal = np.array([self.intrinsics_all[0][0, 0], self.intrinsics_all[0][1, 1]], dtype=np.float32)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(self.camera_path)['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        logger.info(f"{self.name}: bmvs_{self.obj_id}, Got {colored(self.n_imgs, 'yellow', attrs=['bold'])}")
        logger.info(f"{self.name}: bmvs_{self.obj_id}, include_mask: {self.include_mask}")

    def __len__(self):
        return self.n_imgs

    def get_image(self, idx):
        path = self.image_paths[idx]
        png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = png[:, :, :3].copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tvF.to_tensor(image)
        assert image.shape[0] == 3
        image = tvF.normalize(image, [0.5, 0.5, 0.5], [self.std, self.std, self.std])
        image = image * 0.5 + 0.5  # [3, H, W] 0~1

        mask_path = self.mask_paths[idx]
        mask = np.array(imageio.imread(mask_path, as_gray=True), dtype=np.uint8)
        mask = tvF.to_tensor(mask).squeeze()  # [H, W] 0 or 1

        image = image * mask.unsqueeze(0)
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
