import argparse
import os

import torch

from .utils.logger import logger
from .utils.misc import update_config

_parser = argparse.ArgumentParser(description="MR. Anderson")
"----------------------------- Experiment options -----------------------------"
_parser.add_argument("-c", "--cfg", help="experiment configure file name", type=str, default=None)
_parser.add_argument("--exp_id", default="default", type=str, help="Experiment ID")
_parser.add_argument("-obj", "--obj_id", required=True, type=str, help="Object ID or name")
_parser.add_argument("--resume", help="resume training from exp", type=str, default=None)
_parser.add_argument("--resume_epoch", help="resume from the given epoch", type=int, default=0)
_parser.add_argument("--reload", help="reload checkpoint for test", type=str, default=None)
_parser.add_argument("-b",
                     "--batch_size",
                     help="batch size of exp, will replace bs in cfg file if is given",
                     type=int,
                     default=None)
_parser.add_argument("-rr", "--recon_res", help="reconstruction resolution of testing", type=int, default=512)
_parser.add_argument("-g", "--gpu_id", type=str, default=None, help="override enviroment var CUDA_VISIBLE_DEVICES")
_parser.add_argument("--snapshot", default=5, type=int, help="How often to take a snapshot of the model (0 = never)")


def parse_exp_args():
    arg, custom_arg_string = _parser.parse_known_args()

    if arg.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    arg.n_gpus = torch.cuda.device_count()
    arg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return arg, custom_arg_string


def merge_args(arg, cus_arg):
    for k, v in cus_arg.__dict__.items():
        if v is not None:
            arg.__dict__[k] = v
    return arg
