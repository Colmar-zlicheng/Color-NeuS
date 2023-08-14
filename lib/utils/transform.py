import math
import random
from copy import deepcopy
from re import L
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

from .logger import logger
from .misc import CONST


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = "xyz", **kwargs):
        convention = convention.lower()
        if not (set(convention) == set("xyz") and len(convention) == 3):
            raise ValueError(f"Invalid convention {convention}.")
        if isinstance(rotation, np.ndarray):
            data_type = "numpy"
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = "tensor"
        else:
            raise TypeError("Type of rotation should be torch.Tensor or numpy.ndarray")
        for t in self.transforms:
            if "convention" in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == "numpy":
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis angles shape f{axis_angle.shape}.")
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def rotmat_to_aa(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def aa_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis angles f{axis_angle.shape}.")
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def aa_to_rot6d(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis_angle f{axis_angle.shape}.")
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)


def quat_to_aa(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions f{quaternions.shape}.")
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def rot6d_to_rotmat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def rotmat_to_rot6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def rotmat_to_quat(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions shape f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_quat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d shape f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


trans_t = lambda t: torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(
    phi), np.cos(phi), 0], [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(
    th), 0, np.cos(th), 0], [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def rotmat_interpolate(R_start, R_end, ratio):
    q_start = rotmat_to_quat(R_start)
    q_end = rotmat_to_quat(R_end)
    q_new = slerp(q_start, q_end, ratio)
    return quat_to_rotmat(q_new)


def slerp(q_start, q_end, ratio):
    """Perform quaternion interpolation (slerp) between q_start and q_end."""
    q_start = q_start / torch.norm(q_start)
    q_end = q_end / torch.norm(q_end)

    dot_product = torch.dot(q_start, q_end)
    if dot_product < 0.0:
        q_start = -q_start
        dot_product = -dot_product

    if dot_product > 0.9995:
        result = q_start + ratio * (q_end - q_start)
        result = result / torch.norm(result)
        return result

    theta_0 = torch.acos(dot_product)
    theta = theta_0 * ratio

    q_perp = q_end - dot_product * q_start
    q_perp = q_perp / torch.norm(q_perp)

    result = torch.cos(theta) * q_start + torch.sin(theta) * q_perp
    result = result / torch.norm(result)
    return result


def SE3_interpolate(T_start, T_end, ratio):
    R_start = T_start[:3, :3]
    t_start = T_start[:3, 3]
    t_end = T_end[:3, 3]
    R_end = T_end[:3, :3]
    R_new = rotmat_interpolate(R_start, R_end, ratio)
    t_new = t_start + ratio * (t_end - t_start)
    T_new = torch.eye(4, dtype=torch.float32, device=T_start.device)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    return T_new
