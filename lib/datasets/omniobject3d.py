import os
import cv2
import json
import imageio
import torch
import numpy as np
import torchvision.transforms.functional as tvF
from termcolor import colored
from lib.utils.builder import DATASET
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.transform import pose_spherical


@DATASET.register_module()
class OmniObject3D(torch.utils.data.Dataset):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.name = type(self).__name__

        self.cfg = cfg
        self.std = 0.5
        self.data_root = cfg.DATA_ROOT
        self.obj_info = cfg.OBJ_ID
        self.fx_only = cfg.DATA_PRESET.get('FX_ONLY', False)
        self.include_mask = cfg.DATA_PRESET.get('INCLUDE_MASK', True)
        self.opengl_sys = cfg.DATA_PRESET.get('OPENGL_SYS', False)

        self.obj_class = self.obj_info[:-4]
        self.obj_id = self.obj_info[-3:]

        self.data_path = os.path.join(self.data_root, 'OmniObject3D/blender_renders', self.obj_class, self.obj_info,
                                      'render')
        render_poses, focal = self._load_blender_data()

        if self.fx_only:
            self.focal = np.array([focal], dtype=np.float32)
        else:
            self.focal = np.array([focal, focal], dtype=np.float32)

        self.n_imgs = len(self.image_paths)

        self.origin = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.radius = torch.tensor(1, dtype=torch.float32)
        self.scale_mats_np = []
        self.scale_mats_np = [np.identity(4) for idx in range(self.n_imgs)]
        self.object_bbox_min = np.asarray([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.asarray([1.01, 1.01, 1.01])

        logger.info(f"{self.name}: {self.obj_info}, Got {colored(self.n_imgs, 'yellow', attrs=['bold'])}")
        logger.info(f"{self.name}: {self.obj_info}, include_mask: {self.include_mask}")

    def _load_blender_data(self):
        with open(os.path.join(self.data_path, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)

        self.image_paths = []
        self.poses = []
        self.img_ids = []

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

        for i, frame in enumerate(meta['frames']):
            fname = os.path.join(self.data_path, 'images', frame['file_path'].split('/')[-1] + '.png')
            self.image_paths.append(fname)
            pose = np.array(frame['transform_matrix'])
            pose[:, 1:3] *= -1
            if self.opengl_sys:
                pose = _coord_trans_OpenGL @ pose
            self.poses.append(torch.from_numpy(pose).float())
            self.img_ids.append(torch.tensor([i]))

        self.img_ids = torch.cat(self.img_ids, dim=0)

        tmp_img = cv2.imread(self.image_paths[0])
        self.H, self.W = tmp_img.shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                                   0)

        return render_poses, focal

    def __len__(self):
        return self.n_imgs

    def get_image(self, idx):
        path = self.image_paths[idx]
        png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = png[:, :, :3].copy()
        if self.include_mask:
            mask = tvF.to_tensor(png[:, :, 3].copy()).squeeze()  # [H, W] 0 or 1
        else:
            mask = None
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