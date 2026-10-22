# -*- coding: utf-8 -*-
import os
import sys

import torch
import torch.optim as optim
import torch.optim.optimizer
import torchvision

CSDP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSDP-Net', 'CSDP-Net-main')
if CSDP_ROOT not in sys.path:
    sys.path.insert(0, CSDP_ROOT)

from config import config_root
from CSDPNet import CSDPNet
from KM_UNetV3_LAPS import KM_UNetV3
from models.ADNMUNet import create_ADNMUNet
from models.ConvLSTM import create_ConvLSTM
from models.LPTQPN import LPTQPN
from models.Mamba_UNet import Mamba_UNet
from models.SCECA_Net import SCECANet
from models.SmaAt_UNet import SmaAt_UNet
from models.SwinUnet import config, SwinUnet
from models.TrajGRU import create_TrajGRU
from models.TransUnet import create_TransUnet
from models.utils import RainfallLoss as MambaRainfallLoss
from models.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_models(model_name, input_frames, output_frames, frame_interval, dataset):
    if dataset == 'LAPS':
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.8]
    elif dataset == 'Shanghai':
        thresholds = [20, 30, 35, 40]
    
    if model_name == "ConvLSTM":
        model = create_ConvLSTM(output_frames)
        base_LR = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)
        criterion = Weighted_mse_mae(thresholds=thresholds).to(device)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    elif model_name == "TrajGRU":
        base_LR = 1e-4
        model = create_TrajGRU(output_frames)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)
        criterion = Weighted_mse_mae(thresholds=thresholds).to(device)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    elif model_name == "LPTQPN":
        model = LPTQPN(inp_channels=input_frames, out_channels=output_frames)
        base_LR = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_LR)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-9)
        criterion = RAINlOSS().to(device)
        early_stop = 10
        if_early_stop = True
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    elif model_name == "TransUnet":
        model = create_TransUnet(output_frames)
        base_LR = 1e-2
        optimizer = optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
        criterion = RAINlOSS().to(device)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = False
        lr_scheduler = None
        norm_clip = False
        save_epoch = False

    elif model_name == 'SmaATUnet':
        model = SmaAt_UNet(n_channels=input_frames, n_classes=output_frames)
        base_LR = 1e-2
        optimizer = optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
        criterion = RAINlOSS().to(device)
        early_stop = 30
        if_early_stop = True
        ex_lr_scheduler = False
        lr_scheduler = None
        norm_clip = False
        save_epoch = False

    elif model_name == 'SwinUnet':
        model = SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=output_frames)
        base_LR = 1e-2
        optimizer = optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
        criterion = RAINlOSS().to(device)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = False
        lr_scheduler = None
        norm_clip = False
        save_epoch = False

    elif model_name == 'KM-UNet':
        model = KM_UNetV3(num_classes=output_frames)
        base_LR = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_LR, weight_decay=0.05)
        criterion = HybridLoss().to(device)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-4)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    elif model_name == 'CSDPNet':
        model = CSDPNet(seqNum=output_frames, filter_size=5, img_size=256)
        base_LR = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)
        criterion = CSDPOrigLoss().to(device)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    elif model_name == 'SCECA':
        model = SCECANet(in_channels=input_frames, out_channels=output_frames)
        base_LR = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_LR)
        criterion = RAINlOSS().to(device)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    elif model_name == 'MambaUnet':
        model = Mamba_UNet(
            predicted_frames=output_frames,
            input_frames=input_frames,
            c_list=[32, 64, 128, 256, 512, 1024],
            split_att='fc',
            bridge=True,
        )
        base_LR = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)
        criterion = MambaRainfallLoss(0.57, 0.25, 0.5, 1).to(device)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        early_stop = None
        if_early_stop = False
        ex_lr_scheduler = True
        norm_clip = False
        save_epoch = False

    return model, optimizer, criterion, lr_scheduler, if_early_stop, ex_lr_scheduler, norm_clip, save_epoch, base_LR, early_stop
