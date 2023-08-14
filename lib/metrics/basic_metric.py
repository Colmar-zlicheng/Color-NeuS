from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class Metric(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.skip = False

    def is_empty(self) -> bool:
        return self.count == 0

    def num_sample(self) -> int:
        return self.count

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def feed(self, **kwargs):
        pass

    @abstractmethod
    def get_measures(self, **kwargs) -> Dict:
        pass


class VisMetric(Metric):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.images = None

    def reset(self):
        del self.images
        self.images = None

    def get_measures(self, **kwargs) -> Dict:
        return {"image": self.images}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="") -> None:
        super().__init__()
        self.reset()
        self.name = name

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_by_mean(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.avg:.4e}"

    def get_measures(self) -> Dict:
        return {self.name + 'avg': self.avg}


class LossMetric(Metric):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._losses: Dict[str, AverageMeter] = {}

    def reset(self):
        self._losses: Dict[str, AverageMeter] = {}
        self.count = 0

    def feed(self, losses, batch_size: int = 1, **kwargs):
        for k, v in losses.items():
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                # if it is not a tensor, what is should be?
                continue

            if k in self._losses:
                self._losses[k].update_by_mean(v.item(), batch_size)
            else:
                self._losses[k] = AverageMeter()
                self._losses[k].update_by_mean(v.item(), batch_size)

        self.count += batch_size

    def get_measures(self, **kwargs) -> Dict:
        measure = {}
        for k, v in self._losses.items():
            measure[k] = v.avg
        return measure

    def get_loss(self, loss_name: str) -> float:
        return self._losses[loss_name].avg
