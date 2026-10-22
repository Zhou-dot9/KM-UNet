# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import numpy as np
import math
import random

CSDP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSDP-Net', 'CSDP-Net-main')
if CSDP_ROOT not in sys.path:
    sys.path.insert(0, CSDP_ROOT)
from ssim import SSIM as CSDP_SSIM


class RainfallLoss(nn.Module):
    def __init__(self, omega_t=0.57, alpha=0.25):
        super(RainfallLoss, self).__init__()
        self.omega_t = omega_t
        self.alpha = alpha
    def forward(self, pred, target):
        #######################-----Dynamic quantile weighted loss(DQWL)----##########################
        w0 = 0.57
        loss_overal1 = torch.sum(((pred >= target)).float() * (1 - w0) * abs(pred - target))
        loss_overal2 = torch.sum(((pred < target)).float() * w0 * abs(pred - target))
        wi = self.alpha * torch.exp(target)
        loss_greater = torch.sum(
            ((pred >= target) & (target >= 0.7)).float() * (1 - self.omega_t) * wi * abs(pred - target))
        loss_less = torch.sum(((pred < target) & (target >= 0.7)).float() * self.omega_t * wi * abs(pred - target))
        num_sample = target.numel()
        total_loss = (loss_overal1 + loss_overal2) / num_sample + (loss_greater + loss_less) / num_sample
        return total_loss


class enRainfallLoss(nn.Module):
    def __init__(self, omega_t=0.57, alpha=0.25, gamma=0.1):
        super(enRainfallLoss, self).__init__()
        self.omega_t = omega_t
        self.alpha = alpha
        self.gamma = gamma  # 新增FN惩罚系数

    def forward(self, pred, target):
        w0 = self.omega_t
        loss_overal1 = torch.sum(((pred >= target)).float() * (1 - w0) * abs(pred - target))
        loss_overal2 = torch.sum(((pred < target)).float() * w0 * abs(pred - target))
        wi = self.alpha * torch.exp(target)
        loss_greater = torch.sum(
            ((pred >= target) & (target >= 0.7)).float() * (1 - self.omega_t) * wi * abs(pred - target))
        loss_less = torch.sum(((pred < target) & (target >= 0.7)).float() * self.omega_t * wi * abs(pred - target))
        
        fn_penalty = torch.sum(
            ((target >= 0.7) & (pred < target)).float() *     
            self.gamma *                                        
            (torch.exp(self.alpha * (target - pred)) - 1.0)    
        )

        num_sample = target.numel()
        total_loss = (loss_overal1 + loss_overal2 + loss_greater + loss_less + fn_penalty) / num_sample
        return total_loss



class RAINlOSS(nn.Module):
    def __init__(self):
        super(RAINlOSS, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, x, y):
        loss = self.mse(x, y) + self.mae(x, y)
        return loss



class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None, thresholds=[]):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA
        self.thresholds=thresholds

    def forward(self, input, target):
        input = input.transpose(0,1)
        target = target.transpose(0,1)
        balancing_weights = (1, 1, 2, 5, 10, 30)
        weights = torch.ones_like(input) * balancing_weights[0]
        for i, threshold in enumerate(self.thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        # weights = weights * mask.float()

        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


class HybridLoss(nn.Module):
    """KM-UNet: alpha * (0.55 * MSE + 0.45 * W-MSE) + (1 - alpha) * (1 - SSIM)."""

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIM(data_range=1.0)

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        weight_map = torch.exp(target * 2)
        weighted_loss = ((pred - target).pow(2) * weight_map).mean()

        target_min = target.min().detach()
        target_max = target.max().detach()
        target_norm = (target - target_min) / (target_max - target_min + 1e-8)
        pred_min = pred.min().detach()
        pred_max = pred.max().detach()
        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-8)
        ssim_loss = 1 - self.ssim(pred_norm, target_norm)
        return self.alpha * (0.55 * mse_loss + 0.45 * weighted_loss) + (1 - self.alpha) * ssim_loss


class CSDPOrigLoss(nn.Module):
    """CSDP-Net: (MSE + 0.5 * (1 - SSIM)) / 2."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = CSDP_SSIM()

    def forward(self, pred, target):
        return (self.mse(pred, target) + (1 - self.ssim(pred, target)) * 0.5) / 2
