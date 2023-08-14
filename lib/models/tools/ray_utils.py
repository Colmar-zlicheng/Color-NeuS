import torch
import numpy as np
from typing import List
from lib.utils.logger import logger


def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near.squeeze(), far.squeeze()


def get_rays_multicam(c2w,
                      focal,
                      image,
                      n_rays,
                      normalize=False,
                      mask=None,
                      mask_rate=0.9,
                      return_mask=False,
                      opengl=False):
    """
    Generate random n rays at world space from N cameras
        focal: [2]
        c2w: [N, 4, 4]
        image: [N, H, W, 3]
        n_rays: num of sampled rays
        normalize: normalize rays_d
        mask: sample rays in mask with rate (mask_rate)
        opengl: In the OpenGL coordinate system, the z-axis is backward
    """
    assert c2w.dim() == 3 and image.dim() == 4, "this is a multicam implementation"
    device = c2w.device
    N_cam = c2w.shape[0]
    H, W = image.shape[1], image.shape[2]
    if opengl:
        y, z = -1, -1
    else:
        y, z = 1, 1

    # important: indexing='xy'
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
    i = i.to(device)
    j = j.to(device)
    dirs = torch.stack([(i - W * 0.5) / focal[0], y * (j - H * 0.5) / focal[1], z * torch.ones_like(i).to(device)], -1)
    if normalize:
        dirs = dirs / torch.norm(dirs, dim=-1).unsqueeze(-1)  # [H, W, 3]
    dirs = dirs.unsqueeze(0).repeat(N_cam, 1, 1, 1)  # [N, H, W, 3]
    rays_d_all = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3, :3], -1)  # [N, H, W, 3]
    rays_o_all = c2w[:, np.newaxis, np.newaxis, :3, -1].expand(rays_d_all.shape)  # [N, H, W, 3]
    rays_o_all = rays_o_all.reshape(-1, 3)  # [N*H*W, 3]
    rays_d_all = rays_d_all.reshape(-1, 3)  # [N*H*W, 3]

    if mask is None:
        rays_idx_raw = torch.randint(0, H * W, (n_rays,)).to(device)  # [n_rays]
        rays_idx = rays_idx_raw.unsqueeze(-1).repeat(1, 3)
    else:
        mask_all = mask.reshape(-1)  # [N*H*W]
        valide_index = torch.where(mask_all > 0)[0]
        rand_valid_index = torch.randperm(valide_index.shape[0])

        n_rays_in_mask = int(mask_rate * n_rays)
        if n_rays_in_mask > valide_index.shape[0]:
            logger.warning(f"there are less than {n_rays_in_mask} rays in mask!")
            n_rays_in_mask = valide_index.shape[0]
        n_rays_in_bkg = n_rays - n_rays_in_mask
        invalid_index = torch.where(mask_all == 0)[0]
        rand_invalid_index = torch.randperm(invalid_index.shape[0])
        rand_index_mask = rand_valid_index[:n_rays_in_mask]
        rand_index_bkg = rand_invalid_index[:n_rays_in_bkg]
        rays_idx_raw = torch.cat([valide_index[rand_index_mask], invalid_index[rand_index_bkg]], dim=-1)  # [n_rays]
        rays_idx_raw = rays_idx_raw[torch.randperm(rays_idx_raw.shape[0])]  # NOTE: keep rand
        rays_idx = rays_idx_raw.unsqueeze(-1).repeat(1, 3)

    rays_o = torch.gather(rays_o_all, dim=0, index=rays_idx)  # [n_rays, 3]
    rays_d = torch.gather(rays_d_all, dim=0, index=rays_idx)  # [n_rays, 3]
    rgb = torch.gather(image.reshape(-1, 3), dim=0, index=rays_idx)  # [n_rays, 3]

    if return_mask:
        assert mask != None
        mask_select = torch.gather(mask_all, dim=0, index=rays_idx_raw)
        return rays_o, rays_d, rgb, mask_select
    else:
        return rays_o, rays_d, rgb, None


def get_rays_at(c2w, focal, H, W, normalize=False, opengl=False):
    """
    Generate all rays at world space from one camera
        H, W: image shape
        focal: [2]
        c2w: [4, 4]
        normalize: normalize rays_d
        opengl: In the OpenGL coordinate system, the z-axis is backward
    """
    assert c2w.dim() == 2, "this is a sigle camera implementation"
    device = c2w.device
    # important: indexing='xy'
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
    i = i.to(device)
    j = j.to(device)

    if opengl:
        y, z = -1, -1
    else:
        y, z = 1, 1

    dirs = torch.stack([(i - 0.5 * W) / focal[0], y * (j - 0.5 * H) / focal[1], z * torch.ones_like(i).to(device)], -1)
    if normalize:
        dirs = dirs / torch.norm(dirs, dim=-1).unsqueeze(-1)  # follow pixel nerf
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # [H, W, 3]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H, W, 3]
    return rays_o, rays_d


# this code is borrow from https://github.com/Totoro97/NeuS.git
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = weights.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds).to(device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom).to(device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
