from typing import Optional
from tqdm import tqdm
from types import MethodType


def set_description(self, _: str):
    # if rank != 0, output nothing!
    pass


def etqdm(iterable, rank: Optional[int] = None, **kwargs):
    if rank:
        iterable.set_description = MethodType(set_description, iterable)
        return iterable
    else:
        return tqdm(iterable, bar_format="{l_bar}{bar:3}{r_bar}", colour="#ffa500", **kwargs)

