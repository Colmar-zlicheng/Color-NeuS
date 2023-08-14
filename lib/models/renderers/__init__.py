from ...utils.builder import RENDERER, build_from_cfg


def build_renderer(cfg, **kwargs):
    return build_from_cfg(cfg, RENDERER, **kwargs)