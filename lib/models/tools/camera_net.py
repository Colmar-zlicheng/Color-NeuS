import torch
import torch.nn as nn
import numpy as np
from lib.utils.transform import aa_to_rotmat, rot6d_to_rotmat


# This code is borrow from https://github.com/ActiveVisionLab/nerfmm.git
class Focal_Net(nn.Module):

    def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None):
        super(Focal_Net, self).__init__()
        self.H = H
        self.W = W
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    if init_focal.shape[0] == 2:
                        coe_x = torch.tensor(np.sqrt(init_focal[0] / float(W)), requires_grad=False).float()
                        coe_y = torch.tensor(np.sqrt(init_focal[1] / float(H)), requires_grad=False).float()
                    else:
                        coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                        coe_y = torch.tensor(np.sqrt(init_focal / float(H)), requires_grad=False).float()

                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                    coe_y = torch.tensor(init_focal / float(H), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

    def forward(self, i=None):  # the i=None is just to enable multi-gpu training
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fx**2 * self.W])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fx * self.W])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
        return fxfy


# This code is borrow from https://github.com/ActiveVisionLab/nerfmm.git and https://github.com/quan-meng/gnerf.git
class Pose_Net(nn.Module):

    def __init__(self, num_cams, learn_R, learn_t, pose_mode='3d', init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(Pose_Net, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        self.pose_mode = pose_mode
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        if self.pose_mode == '3d':
            self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        elif self.pose_mode == '6d':
            self.r = nn.Parameter(torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).repeat(num_cams, 1),
                                  requires_grad=learn_R)  # (N, 6)
        else:
            raise ValueError(f"pose mode must be one of 3d or 6d, but got {self.pose_mode}")
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_ids):
        r = self.r[cam_ids]  # (N, 3) axis-angle
        t = self.t[cam_ids]  # (N, 3) or (N, 6)
        if self.pose_mode == '3d':
            R = aa_to_rotmat(r)  # (N, 3, 3)
        elif self.pose_mode == '6d':
            R = rot6d_to_rotmat(r)  # (N, 3)
        c2w = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (N, 3, 4)
        c2w = convert3x4_4x4(c2w)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_ids]

        return c2w


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device)],
                               dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output
