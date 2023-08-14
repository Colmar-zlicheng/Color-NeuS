import torch
import numpy as np


# This implementation is borrowed from pixel-nerf (https://github.com/sxyu/pixel-nerf.git)
class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_freqs = cfg.get('NUM_FREQS', 6)
        self.d_in = cfg.get('D_IN', 3)
        self.freq_factor = cfg.get('FREQ_FACTOR', np.pi)
        self.include_input = cfg.get('INCLUDE_INPUT', True)
        self.freqs = self.freq_factor * 2.0**torch.arange(0, self.num_freqs)
        self.d_out = self.num_freqs * 2 * self.d_in

        if self.include_input:
            self.d_out += self.d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer("_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1))
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], -1)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim
