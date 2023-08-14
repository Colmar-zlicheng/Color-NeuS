from lib.utils.config import CN

from ..utils.builder import build_dataset
from .dtu import DTU
from .iho_video import IHO_VIDEO
from .omniobject3d import OmniObject3D
from .bmvs import BlendedMVS


def create_dataset(cfg: CN, data_preset: CN):
    """
    Create a dataset instance.
    """
    return build_dataset(cfg, data_preset=data_preset)
