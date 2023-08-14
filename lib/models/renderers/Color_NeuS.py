import torch
import torch.nn.functional as F
from lib.utils.builder import RENDERER
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.models.renderers.fields import RelightNetwork
from lib.models.renderers.NeuS import NeuS


@RENDERER.register_module()
class Color_NeuS(NeuS):

    def __init__(self, cfg):
        assert cfg.COLOR.MODE == 'no_view_dir'
        super().__init__(cfg)

        self.relight_network = RelightNetwork(cfg.RELIGHT)

        if self.n_outside > 0:
            logger.warning('This mode (n_outside > 0) has not been tested!')

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    **kwargs):
        batch_size, n_samples = z_vals.shape
        device = rays_d.device

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(device)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        global_color = color_network(pts, gradients, dirs, feature_vector)
        global_color_relight, delta_relight = self.relight_network(
            global_color,
            pts,
            dirs,
            gradients=gradients,
        )
        sampled_color = global_color_relight.reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]).to(device))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) *
                     (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha_global = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        # Render with background

        # Render
        weights_global = alpha_global * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha_global + 1e-7], -1), -1)[:, :-1]

        if background_alpha is not None:
            alpha = alpha_global * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

            # Render
            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1),
                                            -1)[:, :-1]

            weights_sum = weights.sum(dim=-1, keepdim=True)
            color = (sampled_color * weights[:, :, None]).sum(dim=1)
            weights_return = weights
        else:
            weights_sum = weights_global.sum(dim=-1, keepdim=True)
            color = (sampled_color * weights_global[:, :, None]).sum(dim=1)
            weights_return = weights_global

        global_color = (global_color.reshape(batch_size, n_samples, 3) * weights_global[:, :, None]).sum(dim=1)

        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0)**2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'global_color': global_color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights_return,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'delta_relight': delta_relight.reshape(batch_size, n_samples, 3)
        }
