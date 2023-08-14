# tune multi-threading params
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)

import os
import random
from argparse import Namespace
from mimetypes import init
from time import time

import lib.models
import numpy as np
import warnings
import torch
from termcolor import colored
from lib.datasets import create_dataset
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_perfixes, format_args_cfg
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import SummaryWriter
from lib.utils.config import CN


def main_worker(cfg: CN, arg: Namespace, time_f: float):
    if arg.exp_id != 'default':
        warnings.warn("You shouldn't assign exp_id in test mode")
    cfg_name = arg.cfg.split("/")[-1].split(".")[0]
    exp_id = f"eval_{cfg_name}_{arg.obj_id}"

    recorder = Recorder(exp_id, cfg, rank=0, time_f=time_f, eval_only=True)
    summary = SummaryWriter(log_dir=recorder.tensorboard_path)
    exp_path = f"{recorder.exp_id}_{recorder.timestamp}"

    dataset = create_dataset(cfg.DATASET, data_preset=cfg.DATA_PRESET)
    init_data = dataset.get_init_data()
    model = builder.build_model_init(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN, data=init_data)
    model.setup(summary_writer=summary)
    model = model.cuda()
    # device = torch.device('cuda')

    model.eval()
    logger.info(f"got reconstruction resolution: {arg.recon_res}")
    model(None, 0, "test", exp_path=exp_path, recon_res=arg.recon_res)
    model.on_test_finished(recorder, 0)


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    assert arg.reload is not None, "reload checkpointint path is required"
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu_id

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    logger.info("====> Testing on single GPU (Data Parallel) <====")
    main_worker(cfg, arg, exp_time)