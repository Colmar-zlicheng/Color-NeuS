import os
import time
from typing import Dict, Optional, TypeVar
from torch.utils.tensorboard import SummaryWriter

from .dist_utils import master_only
from .misc import TrainMode


class DDPSummaryWriter(SummaryWriter):

    def __init__(self, log_dir, rank, **kwargs):
        super(DDPSummaryWriter, self).__init__(log_dir, **kwargs)
        self.rank = rank

    @master_only
    def add_scalar(self, tag, value, global_step=None, walltime=None):
        super(DDPSummaryWriter, self).add_scalar(tag, value, global_step=global_step, walltime=walltime)

    @master_only
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        return super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    @master_only
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        return super().add_image(tag, img_tensor, global_step, walltime=walltime, dataformats=dataformats)

    @master_only
    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        return super().add_images(tag, img_tensor, global_step, walltime=walltime, dataformats=dataformats)

    # there are lots of things can be added to tensorboard
