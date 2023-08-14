import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.models.tools.PositionEncoding import get_embedder
from lib.utils.transform import inverse_sigmoid


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):

    def __init__(self, cfg):
        super(SDFNetwork, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        d_in = cfg.get('D_IN', 3)
        d_out = cfg.get('D_OUT', 257)
        d_hidden = cfg.get('D_HIDDEN', 256)
        n_layers = cfg.get('N_LAYERS', 8)
        skip_in = cfg.get('SKIP_IN', [4])
        multires = cfg.get('MULTIRES', 6)
        bias = cfg.get('BIAS', 0.5)
        scale = cfg.get('SCALE', 3.0)
        geometric_init = cfg.get('GEOMETRIC_INIT', True)
        weight_norm = cfg.get('WEIGHT_NORM', True)
        inside_outside = cfg.get('INSIDE_OUTSIDE', False)

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        d_feature = cfg.get('D_FEATURE', 256)
        mode = cfg.get('MODE', 'idr')  # ['idr', 'no_view_dir', 'no_normal']
        d_in = cfg.get('D_IN', 9)
        d_out = cfg.get('D_OUT', 3)
        d_hidden = cfg.get('D_HIDDEN', 256)
        n_layers = cfg.get('N_LAYERS', 4)
        weight_norm = cfg.get('WEIGHT_NORM', True)
        multires_view = cfg.get('MULTIRES_VIEW', 4)
        squeeze_out = cfg.get('SQUEEZE_OUT', True)

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            raise ValueError(f'no such mode: {self.mode}')

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):

    def __init__(self,
                 D=8,
                 W=256,
                 d_in=4,
                 d_in_view=3,
                 multires=10,
                 multires_view=4,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=True):
        super(NeRF, self).__init__()
        self.name = type(self).__name__
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):

    def __init__(self, cfg):
        super(SingleVarianceNetwork, self).__init__()
        init_val = cfg.get('INIT_VAL', 0.3)
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        device = x.device
        return torch.ones([len(x), 1]).to(device) * torch.exp(self.variance * 10.0)


class RelightNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        d_in = cfg.get('D_IN', 6)
        d_out = cfg.get('D_OUT', 3)
        d_hidden = cfg.get('D_HIDDEN', 256)
        self.n_layers = cfg.get('N_LAYERS', 4)
        self.y_in_layer = cfg.get('Y_IN_LAYER', 3)
        multires_view = cfg.get('MULTIRES_VIEW', 4)
        self.include_grad = cfg.get('INCLUDE_GRAD', True)
        self.inv_sigmoid = cfg.get('INV_SIGMOID', True)

        if self.include_grad:
            d_in += 3

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            d_in += (input_ch - 3)

        self.in_layer = nn.Linear(d_in, d_hidden)
        self.rl_mlp = nn.ModuleList()
        for i in range(self.n_layers):
            if (i == self.y_in_layer - 1) and (self.y_in_layer == self.n_layers):
                self.rl_mlp.append(nn.Linear(3 + d_hidden, d_out))
            elif i == self.y_in_layer - 1:
                self.rl_mlp.append(nn.Linear(3 + d_hidden, d_hidden))
            elif i == self.n_layers - 1:
                self.rl_mlp.append(nn.Linear(d_hidden, d_out))
            else:
                self.rl_mlp.append(nn.Linear(d_hidden, d_hidden))

        self.relu = nn.ReLU()

        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} got include_grad: {self.include_grad}")
        logger.info(f"{self.name} got inv_sigmoid: {self.inv_sigmoid}")

    def relight(self, rgb, pts, dirs, gradients):
        """
        rgb: [n, 3]
        pts: [n, 3]
        dirs: [n, 3]
        """
        if self.embedview_fn is not None:
            dirs = self.embedview_fn(dirs)

        input_list = [pts, dirs]
        if self.include_grad:
            input_list.append(gradients)

        drgb = self.in_layer(torch.cat(input_list, dim=-1))

        for i in range(self.n_layers):
            drgb = self.relu(drgb)
            if i == self.y_in_layer - 1:
                drgb = self.rl_mlp[i](torch.cat([rgb, drgb], dim=-1))
            else:
                drgb = self.rl_mlp[i](drgb)

        if self.inv_sigmoid:
            rgb = torch.sigmoid(inverse_sigmoid(rgb) + drgb)
            return rgb, drgb
        else:
            rgb = rgb + torch.sigmoid(drgb) - 0.5
            return torch.clamp(rgb, max=1.0, min=0.0), drgb

    def forward(self, rgb, pts, dirs, gradients):
        """
        rgb: [n, 3]
        pts: [n, 3]
        dirs: [n, 3]
        """
        rgb, drgb = self.relight(rgb, pts, dirs, gradients)
        return rgb, drgb
