import os
import random
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .logger import logger
from .misc import CONST


def worker_init_fn(worker_id):
    seed = worker_id * int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def recurse_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        recurse_freeze(child)


class NeRF_lr_scheduler(_LRScheduler):

    def __init__(self, optimizer, gamma, decay_steps, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.decay_steps = decay_steps
        super(NeRF_lr_scheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [base_lr * (self.gamma**(self.last_epoch / self.decay_steps)) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [base_lr * (self.gamma**(self.last_epoch / self.decay_steps)) for base_lr in self.base_lrs]


class NeuS_lr_scheduler(_LRScheduler):

    def __init__(self, optimizer, warm_up, alpha, end_iter, last_epoch=-1, verbose=False):
        self.warm_up = warm_up
        self.alpha = alpha
        self.end_iter = end_iter
        super(NeuS_lr_scheduler, self).__init__(optimizer, last_epoch, verbose)

    def _get_lr_neus(self):
        if self.last_epoch < self.warm_up:
            learning_factor = self.last_epoch / self.warm_up
        else:
            progress = (self.last_epoch - self.warm_up) / (self.end_iter - self.warm_up)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - self.alpha) + self.alpha
        return [base_lr * learning_factor for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return self._get_lr_neus()

    def _get_closed_form_lr(self):
        return self._get_lr_neus()


def build_optimizer_nerf(model, cfg, it, **kwargs):
    params = model.parameters()

    # Optimizers
    if cfg.TYPE == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=cfg.LR, alpha=0.99, eps=1e-8)
    elif cfg.TYPE == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg.LR, betas=(0.9, 0.99), eps=1e-8)
    elif cfg.TYPE == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg.LR, momentum=0.)
    else:
        raise NotImplementedError

    # Learning rate anneling
    lr = optimizer.param_groups[0]['lr']
    # create learning reate scheduler
    if cfg.SCHEDULER_TYPE == 'NERF':
        scheduler = NeRF_lr_scheduler(optimizer, cfg.GAMMA, cfg.LRATE_DECAY, it)
    elif cfg.SCHEDULER_TYPE == 'NEUS':
        scheduler = NeuS_lr_scheduler(optimizer, cfg.WARM_UP, cfg.LR_ALPHA, kwargs['iterations'], it)
    else:
        raise ValueError(f"get unexcepted scheduler type: {cfg.SCHEDULER_TYPE}")
    # ensure lr is not decreased again
    optimizer.param_groups[0]['lr'] = lr

    return optimizer, scheduler


def build_optimizer(params: Iterable, cfg) -> Optimizer:
    if cfg.OPTIMIZER in ["Adam", "adam"]:
        return torch.optim.Adam(
            params,
            lr=cfg.LR,
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    elif cfg.OPTIMIZER == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=cfg.LR,
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.01)),
        )

    elif cfg.OPTIMIZER in ["SGD", "sgd"]:
        return torch.optim.SGD(
            params,
            lr=cfg.LR,
            momentum=float(cfg.get("MOMENTUM", 0.0)),
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIMIZER} not be implemented yet")


def build_scheduler(optimizer: Optimizer, cfg):
    scheduler = cfg.SCHEDULER
    lr_decay_step = cfg.get("LR_DECAY_STEP", -1)
    lr_decay_each = cfg.get("LR_DECAY_EACH", -1)
    if lr_decay_step != [-1] and lr_decay_each > 0:
        raise ValueError("lr_decay_step and lr_decay_each shouldn't both be used!")
    tar_scheduler = None

    if lr_decay_each > 0 and scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(0, cfg.EPOCH, lr_decay_each)),
            gamma=cfg.LR_DECAY_GAMMA,
        )
    elif isinstance(lr_decay_step, list) and scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )

    elif scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )

    elif scheduler == "MultiStepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )
    else:
        raise NotImplementedError(f"{scheduler} not yet be implemented")

    return tar_scheduler


def clip_gradient(optimizer, max_norm, norm_type):
    """Clips gradients computed during backpropagation to avoid explosion of gradients.

    Args:
        optimizer (torch.optim.optimizer): optimizer with the gradients to be clipped
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)


def setup_seed(seed, conv_repeatable=True):
    """Setup all the random seeds

    Args:
        seed (int or float): seed value
        conv_repeatable (bool, optional): Whether the conv ops are repeatable (depend on cudnn). Defaults to True.
    """
    logger.warning(f"setup random seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")


### Initialize module parameters with values according to the method ###
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_weights(module: nn.Module, pretrained=None, strict=True):
    if pretrained == "" or pretrained is None:
        logger.warning(f"=> Init {type(module).__name__} weights in its' backbone and head")
        """
        Add init for other modules
        ...
        """
    elif os.path.isfile(pretrained):
        logger.warning(f"=> Loading {type(module).__name__} pretrained model from: {pretrained}")
        # self.load_state_dict(pretrained_state_dict, strict=False)
        checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
        if isinstance(checkpoint, OrderedDict):
            # state_dict = checkpoint
            module.load_state_dict(checkpoint, strict=strict)
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict_old = checkpoint["state_dict"]
            state_dict = OrderedDict()
            # delete 'module.' because it is saved from DataParallel module
            for key in state_dict_old.keys():
                if key.startswith("module."):
                    # state_dict[key[7:]] = state_dict[key]
                    # state_dict.pop(key)
                    state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                else:
                    state_dict[key] = state_dict_old[key]
            module.load_state_dict(state_dict, strict=strict)
        elif isinstance(checkpoint, dict):
            module.load_state_dict(checkpoint, strict=strict)
        logger.warning(f"=> Loading SUCCEEDED")
    else:
        logger.error(f"=> No {type(module).__name__} checkpoints file found in {pretrained}")
        raise FileNotFoundError()
