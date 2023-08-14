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
import torch
from termcolor import colored
from lib.datasets import create_dataset
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_perfixes, format_args_cfg
from lib.utils.net_utils import build_optimizer_nerf, clip_gradient, setup_seed
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import SummaryWriter
from lib.utils.config import CN


def main_worker(gpu_id: int, cfg: CN, args: Namespace, time_f: float):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=0, time_f=time_f)
    summary = SummaryWriter(log_dir=recorder.tensorboard_path)
    exp_path = f"{recorder.exp_id}_{recorder.timestamp}"

    dataset = create_dataset(cfg.DATASET, data_preset=cfg.DATA_PRESET)
    init_data = dataset.get_init_data()
    model = builder.build_model_init(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN, data=init_data)
    model.setup(summary_writer=summary)
    model = model.cuda()
    device = torch.device('cuda')

    optimizer, scheduler = build_optimizer_nerf(model, cfg.TRAIN.OPTIMIZE, -1, iterations=cfg.TRAIN.ITERATIONS)

    if arg.resume:
        start_step = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume)
    else:
        start_step = 0

    logger.warning(f"############## start training from {start_step} to {cfg.TRAIN.ITERATIONS} ##############")
    dataset.get_all_init(batch_size=cfg.TRAIN.BATCH_SIZE)
    trainbar = etqdm(range(start_step, cfg.TRAIN.ITERATIONS))
    model.train()
    for step_idx in trainbar:
        optimizer.zero_grad()

        batch = dataset.get_rand_batch_smaples(device=device)

        render_dict, loss_dict = model(batch, step_idx, "train")
        loss = loss_dict["loss"]
        loss.backward()

        if cfg.TRAIN.GRAD_CLIP_ENABLED:
            clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        trainbar.set_description(f"{bar_perfixes['train']}: {model.format_metric('train')}, "
                                 f"lr: { ['{:g}'.format(group['lr']) for group in optimizer.param_groups]}")

        if (step_idx + 1) % cfg.TRAIN.SAVE_INTERVAL == 0:
            recorder.record_checkpoints(model, optimizer, scheduler, step_idx, arg.snapshot * cfg.TRAIN.SAVE_INTERVAL)
            model.on_train_finished(recorder, step_idx)

        if (step_idx % cfg.TRAIN.VIZ_IMAGE_INTERVAL == cfg.TRAIN.VIZ_IMAGE_INTERVAL - 1 \
            or step_idx % cfg.TRAIN.VIZ_MESH_INTERVAL == cfg.TRAIN.VIZ_MESH_INTERVAL - 1):
            print(" ")
            logger.info("do validation and save results")
            model.eval()
            model(batch, step_idx, "val", exp_path=exp_path)
            model.on_val_finished(recorder, step_idx)
            torch.cuda.empty_cache()
            model.train()

    recorder.record_checkpoints(model, optimizer, scheduler, cfg.TRAIN.ITERATIONS, 1)
    model.on_train_finished(recorder, step_idx)
    logger.info(colored('Train finished!'))


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=False)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    assert arg.n_gpus == 1, "only support single gpu"

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    logger.info("====> Begin Training with Single GPU <====")
    main_worker(gpu_id=arg.gpu_id, args=arg, cfg=cfg, time_f=exp_time)