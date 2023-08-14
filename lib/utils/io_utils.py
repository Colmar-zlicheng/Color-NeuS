import datetime
import os
import pickle
import random
import shutil
import sys
import traceback
import warnings
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from .logger import logger
from .misc import RandomState


def save_checkpoint(
    state,
    is_best,
    checkpoint="checkpoint",
    filename="checkpoint.pth",
    snapshot=None,
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state["epoch"] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoint, "checkpoint_{}.pth.tar".format(state["epoch"])),
        )

    if is_best:
        if "score" in state:
            shutil.copyfile(filepath, os.path.join(checkpoint, f"_modelbest_{round(state['score'], 3)}.pth.tar"))
        else:
            shutil.copyfile(filepath, os.path.join(checkpoint, "_modelbest.pth.tar"))


def save_states(state, is_best, checkpoint="checkpoint", foldname="checkpoint", snapshot=None):
    foldname = os.path.join(checkpoint, foldname)
    if not os.path.exists(foldname):
        os.makedirs(foldname)

    # *1 save model
    model = state.pop("model")
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        torch.save(model.module.state_dict(), os.path.join(foldname, f"{type(model.module).__name__}.pth.tar"))
    else:
        torch.save(model.state_dict(), os.path.join(foldname, f"{type(model).__name__}.pth.tar"))

    # *2 save random state
    random_state: RandomState = state.pop("random_state")

    with open(os.path.join(foldname, f"random_state.pkl"), "wb") as f:
        pickle.dump(random_state, f)

    # *3 save train params
    torch.save(state, os.path.join(foldname, f"train_param.pth.tar"))

    if snapshot and state["epoch"] % snapshot == 0:
        shutil.copytree(
            foldname,
            os.path.join(checkpoint, "checkpoint_{}".format(state["epoch"])),
        )

    if is_best:
        if "score" in state:
            shutil.copytree(foldname, os.path.join(checkpoint, f"_modelbest_{round(state['score'], 3)}"))
        else:
            shutil.copytree(foldname, os.path.join(checkpoint, "_modelbest"))


def load_random_state(resume_path: str):
    try:
        with open(resume_path, "rb") as f:
            rs: RandomState = pickle.load(f)
        random.setstate(rs.random_rng_state)
        np.random.set_state(rs.numpy_rng_state)
        torch.set_rng_state(rs.torch_rng_state)
        torch.cuda.set_rng_state(rs.torch_cuda_rng_state)
        torch.cuda.set_rng_state_all(rs.torch_cuda_rng_state_all)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except:
        # traceback.print_exc()
        logger.error(f"Couldn't resume random state from {resume_path}, might cause the experiment irreproducible!")

        # raise ValueError()


def load_train_param(
    optimizer: Union[Dict[str, Optimizer], Optimizer],
    scheduler: Union[Dict[str, _LRScheduler], _LRScheduler],
    resume_path: str,
    map_location=None,
):
    try:
        parameters = torch.load(resume_path, map_location=map_location)
        logger.info(f"resume train parameters from {resume_path}")

        epoch = parameters["epoch"]

        if type(optimizer) is not dict:
            opt_missing_states = optimizer.state_dict().keys() - parameters["optimizer"].keys()
            if len(opt_missing_states) > 0:
                logger.warning(f"Missing keys in optimizer ! : {opt_missing_states}")
            optimizer.load_state_dict(parameters["optimizer"])
        else:
            for key in optimizer.keys():
                opt_missing_states = optimizer[key].state_dict().keys() - parameters["optimizer"][key].keys()
                if len(opt_missing_states) > 0:
                    logger.warning(f"Missing keys in optimizer ! : {opt_missing_states}")
                optimizer[key].load_state_dict(parameters["optimizer"][key])
        if type(scheduler) is not dict:
            scheduler_missing_states = scheduler.state_dict().keys() - parameters["scheduler"].keys()
            if len(scheduler_missing_states) > 0:
                logger.warning(f"Missing keys in scheduler ! : {scheduler_missing_states}")
            scheduler.load_state_dict(parameters["scheduler"])
        else:
            for key in scheduler.keys():
                scheduler_missing_states = scheduler[key].state_dict().keys() - parameters["scheduler"][key].keys()
                if len(scheduler_missing_states) > 0:
                    logger.warning(f"Missing keys in scheduler ! : {scheduler_missing_states}")
                scheduler[key].load_state_dict(parameters["scheduler"][key])
        return epoch
    except:
        traceback.print_exc()
        logger.error(f"Couldn't resume from {resume_path}")
        raise ValueError()


def load_model(model, resume_path: str, startswith=None, strict=True, as_parallel=False, map_location=None):
    try:
        _model = model.module if hasattr(model, "module") else model
        checkpoint = torch.load(os.path.join(resume_path, f"{type(_model).__name__}.pth.tar"),
                                map_location=map_location)
        state_dict = checkpoint
        if as_parallel and (list(checkpoint.keys()) and "module" not in list(checkpoint.keys())[0]):
            state_dict = {"module.{}".format(key): item for key, item in checkpoint.items()}
        elif not as_parallel and list(checkpoint.keys()) and "module" in list(checkpoint.keys())[0]:
            state_dict = {".".join(key.split(".")[1:]): item for key, item in checkpoint.items()}
        # filter out tensors not startswith given keyword and strip keyword out, if startswith is not None:
        if startswith is not None:
            state_dict = {
                ".".join(key.split(".")[1:]): item for key, item in state_dict.items() if key.startswith(startswith)
            }
        logger.info(
            f"resume {type(_model).__name__} checkpoint start with {startswith} from {os.path.join(resume_path, f'{type(_model).__name__}.pth.tar')}"
        )
        missing_states = _model.state_dict().keys() - state_dict.keys()
        if len(missing_states) > 0:
            logger.warning(f"Missing keys in model ! : {missing_states}")
        _model.load_state_dict(state_dict, strict=strict)
    except:
        traceback.print_exc()
        logger.error(f"Couldn't resume from {resume_path}")
        raise ValueError()
