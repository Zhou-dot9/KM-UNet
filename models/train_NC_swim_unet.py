# 在文件最开头添加以下两行代码（在所有import之前）
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt

import torch

from Swim_Unet import SwinUnet
import torch.nn.functional as F
import numpy as np
import h5py
import csv

import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
# from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix,mean_squared_error
from skimage.metrics import structural_similarity as ssim#for metrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

# from Shanghai import Shanghai,CustomShanghai
import os
# from loss import Weighted_mse_mae
from loss import RAINlOSS

# 导入可视化相关模块
from matplotlib import colors
from functools import partial

# from loss import Weighted_mse_mae
from loss import RAINlOSS
import argparse
import copy
import logging


logger = logging.getLogger(__name__)

import yaml
from yacs.config import CfgNode as CN
# 在程序开头启用异常检测
# torch.autograd.set_detect_anomaly(True)
# 在文件顶部添加
__all__ = ['SwinUnet', 'create_swin_unet', 'get_config']

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 256
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = '/home/ubuntu/zj/KM-UNetV3/other_models/swin_tiny_patch4_window7_224_lite.yaml'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.PRETRAIN_CKPT = '/home/ubuntu/zj/KM-UNetV3/other_models/swin_tiny_patch4_window7_224_lite.yaml'
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 3
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 5
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.FINAL_UPSAMPLE = "expand_first"

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    # print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='/home/ubuntu/zj/KM-UNetV3/other_models/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE",
                    help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# parser.add_argument("--dataset_name", default="datasets")
parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)

# 手动构造参数并加载配置
# 修复方案：通过parser生成完整的args对象，再覆盖必要参数
parser = argparse.ArgumentParser()
# 添加所有原始参数定义（必须与原始代码中的parser参数完全一致）
parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz')
parser.add_argument('--dataset', type=str, default='Synapse')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cfg', type=str, default='/home/ubuntu/zj/KM-UNetV3/other_models/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE")
parser.add_argument("--opts", default=None, nargs='+')
parser.add_argument('--zip', action='store_true')
parser.add_argument('--cache-mode', type=str, default='part')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int)
parser.add_argument('--use-checkpoint', action='store_true')
parser.add_argument('--amp-opt-level', type=str, default='O1')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--throughput', action='store_true')
parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)

# 生成包含所有默认参数的args对象
args = parser.parse_args([])  # 空列表表示不传递任何命令行参数
# 覆盖关键参数
args.cfg = '//home/ubuntu/zj/KM-UNetV3/other_models/swin_tiny_patch4_window7_224_lite.yaml'  # 直接指定配置文件路径
args.num_classes = 3  # 根据任务需求设置

# 加载配置
config = get_config(args)

# 在程序开头启用异常检测
# torch.autograd.set_detect_anomaly(True)

def train(model, train_loader, criterion, optimizer, scheduler, scaler, epoch):

    model.train()
    total_loss = 0.0

    for iter, data in enumerate(train_loader):
        # print(data.shape)#([1, 8, 256, 256])
        data = data.unsqueeze(2)
        input = data[:, :5, :, :, :]
        target = data[:, 5:, :, :, :]
        input, target = input.to('cuda').float(), target.to('cuda').float()

        optimizer.zero_grad()

        with autocast():
            output = model(input)
            loss = criterion(output, target)


        # 反向传播
        # with torch.autograd.detect_anomaly():
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 更严格的裁剪

        # scheduler.step()

        total_loss += loss.item()

        # 打印训练进度
        if iter % 400 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Train Epoch: {epoch} [{iter * len(input)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} LR: {lr:.2e}')

    return total_loss / len(train_loader)


def validate(model, val_loader,criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            data = data.unsqueeze(2)
            input = data[:, :5, :, :, :]  # 5-->16
            target = data[:, 5:, :, :, :]  # 5-->16
            input, target = input.to('cuda').float(), target.to('cuda').float()
            # with autocast():
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def test(model, test_loader,criterion):
    model.eval()
    total_loss = 0.0

    preds = []
    gts = []
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8]
    save_dir = "/home/ubuntu/zj/KM-UNetV3/outputs/other_model_vis_LAPS/swim_unet/pred"
    os.makedirs(save_dir, exist_ok=True)
    gts_dir = "/home/ubuntu/zj/KM-UNetV3/outputs/other_model_vis_LAPS/swim_unet/gts"
    os.makedirs(gts_dir, exist_ok=True)
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            # print(data.shape)
            data = data.unsqueeze(2)
            input = data[:, :5, :, :, :]  # 5-->16
            target = data[:, 5:, :, :, :]  # 5-->16
            input, target = input.to('cuda').half(), target.to('cuda').half()
            # with autocast():
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            total_loss += loss.item()
            # print(output.shape)

            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            preds.append(output)
            gts.append(target)

            output,target=output.squeeze(2),target.squeeze(2)
            # 转换为numpy并确保数值范围
            pred_batch = output.clip(0, 1)  # [B,3,H,W]
            true_batch = target.clip(0, 1)  # [B,3,H,W]
            # 可视化部分

            # print(output.shape)#([2, 3, 256, 256])
            batch_size = pred_batch.shape[0]
            for i in range(batch_size):
                image = pred_batch[i]  # 形状 [3, H, W]
                # print(image.shape)
                for j in range(image.shape[0]):
                    channel_data = image[j]
                    # 确保数据为二维数组
                    if channel_data.ndim != 2:
                        channel_data = channel_data.squeeze()
                    # 跳过无效数据
                    if channel_data.size == 0 or channel_data.shape[0] == 0 or channel_data.shape[1] == 0:
                        continue

                    # 创建figure时指定尺寸
                    plt.figure(figsize=(4, 4))
                    # 应用伪彩色映射（例如 'viridis', 'jet', 'hot'）
                    plt.imshow(channel_data, cmap='viridis', vmin=0, vmax=1)
                    plt.axis('off')

                    # # 创建新的figure并显式指定尺寸
                    # fig = plt.figure(figsize=(4, 4))
                    # plt.imshow(channel_data, cmap='gray')
                    # plt.axis('off')

                    # 构造保存路径
                    save_path = os.path.join(
                        save_dir,
                        f'batch_{iter}_sample_{i}_channel_{j}.png'
                    )
                    # 保存并立即关闭图像
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    # plt.close(fig)  # 必须关闭释放内存
                    plt.close()

            batch_size_gts = true_batch.shape[0]
            for i in range(batch_size_gts):
                image = true_batch[i]  # 形状 [9, H, W]
                for j in range(image.shape[0]):
                    channel_data = image[j]
                    # 确保数据为二维数组
                    if channel_data.ndim != 2:
                        channel_data = channel_data.squeeze()
                    # 跳过无效数据
                    if channel_data.size == 0 or channel_data.shape[0] == 0 or channel_data.shape[1] == 0:
                        continue

                    # 创建figure时指定尺寸
                    plt.figure(figsize=(4, 4))
                    # 应用伪彩色映射（例如 'viridis', 'jet', 'hot'）
                    plt.imshow(channel_data, cmap='viridis', vmin=0, vmax=1)
                    plt.axis('off')

                    ##构造保存路径
                    save_path = os.path.join(
                        gts_dir,
                        f'batch_{iter}_sample_{i}_channel_{j}.png'
                    )
                    # 保存并立即关闭图像
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    # plt.close(fig)  # 必须关闭释放内存
                    plt.close()  # 必须关闭释放内存
        # metrics
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        print(" ")
        csv_path="/home/ubuntu/zj/KM-UNetV3/outputs/log/metrics.csv"
        for threshold in thresholds:
            y_pre = np.where(preds >= threshold, 1, 0)  # preds->normalized_preds
            y_true = np.where(gts >= threshold, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / (
                    ((float(TP) + float(FN)) * ((float(FN) + float(TN)))) + (
                    (float(TP) + float(FP)) * ((float(FP) + float(TN)))))
            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            # print('')
            RMSE = np.sqrt(mean_squared_error(gts, preds))  # gts->y_true,preds->y_pre

            # 计算虚警率 (False Alarm Ratio, FAR)
            FAR = float(FP) / float(TP + FP) if float(TP + FP) != 0 else 0

            # 计算结构相似性 (SSIM)
            SSIM = ssim(gts.reshape(-1), preds.reshape(-1), data_range=1)  # gts->y_true,preds->y_pre
            log_info = f'{threshold}: SSIM: {SSIM:.4f},FAR: {FAR:.4f},CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f},RMSE: {RMSE:.4f}'
            print(log_info)

            # 准备CSV数据行（新增部分）
            row_data = [
                threshold,
                round(SSIM, 4),
                round(FAR, 4),
                round(CSI, 4),
                round(HSS, 4),
                round(POD, 4),
                round(RMSE, 4)
            ]

            # 写入CSV文件（新增部分）
            # with open(csv_path, 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     # 如果是首次写入，添加表头
            #     if f.tell() == 0:
            #         writer.writerow([
            #             'Threshold',
            #             'SSIM',
            #             'FAR',
            #             'CSI',
            #             'HSS',
            #             'POD',
            #             'RMSE'
            #         ])
            #     writer.writerow(row_data)

    # print(f'Test set: Average loss: {total_loss / len(test_loader):.4f}')
    return total_loss / len(test_loader)


def nan_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            raise ValueError(f"NaN in {module}")
    elif isinstance(output, (tuple, list)):
        for out in output:
            if torch.isnan(out).any():
                raise ValueError(f"NaN in {module}")


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7):#alpha=0.7-->0.65
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIM(data_range=1.0).to('cuda')  # 固定归一化范围
    def forward(self, pred, target):
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)

        # 强度敏感损失
        weight_map = torch.exp(target * 2)  # 加强强降雨区域的权重
        weighted_loss = (pred - target).pow(2) * weight_map
        weighted_loss = weighted_loss.mean()

        # 动态归一化到[0,1]
        target_min = target.min().detach()
        target_max = target.max().detach()
        target_norm = (target - target_min) / (target_max - target_min + 1e-8)

        pred_min = pred.min().detach()
        pred_max = pred.max().detach()
        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-8)

        # 计算损失
        ssim_loss = 1 - self.ssim(pred_norm, target_norm)

        return self.alpha * (0.55 * mse_loss + 0.45 * weighted_loss) + (1 - self.alpha) * ssim_loss
        #0.6-->0.55;0.4-->0.45


def main():
    # 配置混合精度训练
    torch.backends.cudnn.benchmark = True
    model = SwinUnet(config, img_size=256, num_classes=3).cuda()

    # 注册钩子到所有子模块
    # for module in model.modules():
    #     module.register_forward_hook(nan_hook)

    base_LR = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)

    criterion = RAINlOSS().cuda()

    # 添加以下代码加载自己数据集
    # -------------------------------------------------------------------------------------
    with h5py.File(r'/home/ubuntu/zj/KM-UNetV3/inputs/merged_data.h5', 'r') as hf:
        data = hf['vil'][:]

    num_samples = int(data.shape[0])  # 696
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    group_size = 8  # 6-->8

    num_train = int(train_ratio * (num_samples - group_size + 1))
    num_val = int(val_ratio * (num_samples - group_size + 1))
    num_test = (num_samples - group_size + 1) - num_train - num_val
    groups = [data[i:i + group_size] for i in range(0, num_samples - group_size)]

    # 划分训练集、验证集和测试集
    train_groups = groups[:num_train]
    valid_groups = groups[num_train:num_val + num_train]
    test_groups = groups[num_train + num_val:]

    train_groups = torch.tensor(np.array(train_groups))
    valid_groups = torch.tensor(np.array(valid_groups))
    test_groups = torch.tensor(np.array(test_groups))

    train_loader = DataLoader(train_groups, 1, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_groups, 1, drop_last=True)
    test_loader = DataLoader(test_groups, 1, drop_last=True)

    # 学习率调度器（根据实际训练步数调整）
    epochs = 60

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=5e-4)

    scaler = GradScaler()
    best_val_loss = 1234

    with open(r'/home/ubuntu/zj/KM-UNetV3/outputs/log/log_swim_unet.csv', 'w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(['epoch', 'train_loss', 'val_loss'])

        for epoch in range(epochs):
            print('Epoch [%d/%d]' % (epoch, epochs))

            train_loss = train(model, train_loader,criterion,optimizer, scheduler, scaler, epoch)

            val_loss = validate(model, val_loader,criterion)

            scheduler.step()

            # 写入日志（新增部分）
            log_writer.writerow([
                epoch,
                round(train_loss, 4),
                round(val_loss, 4),
            ])

            print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), r'/home/ubuntu/zj/KM-UNetV3/outputs/other_model_on_LAPS/model_Best_swim_unet.pth')
                print("=========>>>>>>>>> Saved  best model")
                print('Loss: %.4f' % best_val_loss)
    # #
    #         torch.cuda.empty_cache()

    print("=============================Test=================================")
    # 最终测试
    model_path = r'/home/ubuntu/zj/KM-UNetV3/outputs/other_model_on_LAPS/model_Best_swim_unet.pth'
    model.load_state_dict(torch.load(model_path))
    test_loss = test(model, test_loader,criterion)
    print("Save Picture Finished")


if __name__ == "__main__":
    main()






