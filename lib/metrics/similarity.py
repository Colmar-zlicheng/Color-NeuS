import torch
import math
import numpy as np
from typing import Dict
from lib.metrics.basic_metric import AverageMeter, Metric
import kornia.metrics as KM

mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


class PSNR(Metric):

    def __init__(self, cfg, name="") -> None:
        super(PSNR, self).__init__()
        self.cfg = cfg
        self.avg_meter = AverageMeter()
        self.name = "PSNR"
        self.reset()

    def reset(self):
        self.avg_meter.reset()

    def feed(self, img0, img_1, **kwargs):
        mse = ((img0 - img_1)**2).mean()
        psnr = -10 * math.log10(mse)
        self.avg_meter.update(psnr, n=1)
        return psnr

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        avg = self.avg_meter.avg
        measures[f"{self.name}"] = avg
        return measures

    def get_result(self):
        return self.avg_meter.avg

    def __str__(self):
        return f"{self.name}: {self.avg_meter.avg:6.4f}"


class SSIM(Metric):

    def __init__(self, cfg, name="") -> None:
        super(SSIM, self).__init__()
        self.cfg = cfg
        self.avg_meter = AverageMeter()
        self.name = "SSIM"
        self.reset()

    def reset(self):
        self.avg_meter.reset()

    def feed(self, img0, img_1, **kwargs):
        ssim = torch.mean(KM.ssim(img0, img_1, 3)).item()
        self.avg_meter.update(ssim, n=1)
        return ssim

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        avg = self.avg_meter.avg
        measures[f"{self.name}"] = avg
        return measures

    def get_result(self):
        return self.avg_meter.avg

    def __str__(self):
        return f"{self.name}: {self.avg_meter.avg:6.4f}"


class LPIPS(Metric):

    def __init__(self, cfg, name="") -> None:
        super(LPIPS, self).__init__()
        self.cfg = cfg
        self.avg_meter = AverageMeter()
        self.name = "LPIPS"
        self.reset()

    def reset(self):
        self.avg_meter.reset()

    def feed(self, img0, img_1, **kwargs):
        # TODO:
        lpips = 0
        self.avg_meter.update(lpips, n=1)
        return lpips

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        avg = self.avg_meter.avg
        measures[f"{self.name}"] = avg
        return measures

    def get_result(self):
        return self.avg_meter.avg

    def __str__(self):
        return f"{self.name}: {self.avg_meter.avg:6.4f}"