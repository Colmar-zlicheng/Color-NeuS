import functools
import math
import re
from collections import namedtuple
from enum import Enum
import numpy as np
import yaml
import socket
from termcolor import colored
from contextlib import closing
import collections.abc as abc

bar_perfixes = {
    "train": colored("train", "white", attrs=["bold"]),
    "val": colored("val", "yellow", attrs=["bold"]),
    "test": colored("test", "magenta", attrs=["bold"]),
}

RandomState = namedtuple(
    "RandomState",
    [
        "torch_rng_state",
        "torch_cuda_rng_state",
        "torch_cuda_rng_state_all",
        "numpy_rng_state",
        "random_rng_state",
    ],
)
RandomState.__new__.__default__ = (None,) * len(RandomState._fields)


class TrainMode(Enum):
    TRAIN = 0  # train with augmentation
    VAL = 1  # val with augmentation
    TEST = 2  # test real
    TRAIN_ARTI = 3  # train only augmentation
    VAL_ARTI = 4  # val only augmentation
    TRAIN_REAL = 5  # train only real


bar_perfixes = {
    "train": colored("train", "white", attrs=["bold"]),
    "val": colored("val", "yellow", attrs=["bold"]),
    "test": colored("test", "magenta", attrs=["bold"]),
}


def enable_lower_param(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kw_uppers = {}
        for k, v in kwargs.items():
            kw_uppers[k.upper()] = v
        return func(*args, **kw_uppers)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


class ImmutableClass(type):

    def __call__(cls, *args, **kwargs):
        raise AttributeError("Cannot instantiate this class")

    def __setattr__(cls, name, value):
        raise AttributeError("Cannot modify immutable class")

    def __delattr__(cls, name):
        raise AttributeError("Cannot delete immutable class")


class CONST(metaclass=ImmutableClass):
    PI = math.pi
    INT_MAX = 2**32 - 1
    PYRENDER_EXTRINSIC = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def update_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def format_cfg(cfg, indent_lvl=0):
    indent_width = 2
    INDENT = ' ' * indent_width

    cfg_str = ""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            cfg_str += f"\n{INDENT * indent_lvl} * {colored(k, 'magenta')}: {format_cfg(v, indent_lvl+1)}"
    elif isinstance(cfg, (list, tuple)):
        for elm in cfg:
            cfg_str += f"\n{INDENT * (indent_lvl)} - {format_cfg(elm, indent_lvl+1)}"
        cfg_str += f"\n"
    else:
        cfg_str += f"{cfg}"
    return cfg_str


def format_args_cfg(args, cfg={}):
    args_list = [f" - {colored(name, 'green')}: {getattr(args, name)}" for name in vars(args)]
    arg_str = "\n".join(args_list)
    cfg_str = format_cfg(cfg)
    return arg_str + cfg_str


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def param_size(net):
    # ! treat all parameters to be float
    return sum(p.numel() for p in net.parameters()) * 4 / (1024 * 1024)


def camel_to_snake(camel_input):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{1,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return '_'.join(map(str.lower, words))


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
