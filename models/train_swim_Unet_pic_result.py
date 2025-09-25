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
# from torch.cuda.amp import autocast, GradScaler
# from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, mean_squared_error
from skimage.metrics import structural_similarity as ssim  # for metrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

from Shanghai import Shanghai, CustomShanghai
import math
from metrics import SimplifiedEvaluator
import os

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



# ==================== 可视化配置和函数 ====================
# 上海数据集的颜色配置
PIXEL_SCALE = 90.0
COLOR_MAP = np.array([
    [0, 0, 0, 0],
    [0, 236, 236, 255],
    [1, 160, 246, 255],
    [1, 0, 246, 255],
    [0, 239, 0, 255],
    [0, 200, 0, 255],
    [0, 144, 0, 255],
    [255, 255, 0, 255],
    [231, 192, 0, 255],
    [255, 144, 2, 255],
    [255, 0, 0, 255],
    [166, 0, 0, 255],
    [101, 0, 0, 255],
    [255, 0, 255, 255],
    [153, 85, 201, 255],
    [255, 255, 255, 255]
]) / 255

BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


def gray2color(image, cmap=None, **kwargs):
    if cmap is None:
        cmap = colors.ListedColormap(COLOR_MAP)
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)
    colored_image = cmap(norm(image))
    return colored_image


def vis_res(pred_seq, gt_seq=None, save_path=None, pic_name=None,
            pixel_scale=None, gray2color=None, cmap=None, gap=10,
            input_seq=None, even_index_only=False):
    """
    添加间隙的气象预测结果可视化函数
    """

    # 1. 数据预处理
    def process_seq(seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy()
        seq = seq.squeeze()
        if pixel_scale is not None:
            seq = (seq * pixel_scale).astype(np.uint8)
        return seq

    pred_seq = process_seq(pred_seq)
    if gt_seq is not None:
        gt_seq = process_seq(gt_seq)
    if input_seq is not None:
        input_seq = process_seq(input_seq)

    os.makedirs(save_path, exist_ok=True)

    # 2. 选择索引（如果需要只保存偶数索引）
    def select_indices(seq):
        if even_index_only:
            # 选择偶数索引：0,2,4...
            return seq[::2]  # 修改为从0开始取偶数索引
        return seq

    # 对于输入序列，不使用偶数索引选择
    if input_seq is not None:
        colored_input = gray2color(input_seq, cmap=cmap) if gray2color is not None else input_seq

    # 对于预测和真实值序列，根据参数决定是否选择偶数索引
    pred_seq = select_indices(pred_seq)
    if gt_seq is not None:
        gt_seq = select_indices(gt_seq)

    # 3. 灰度转彩色
    def apply_color(seq):
        if gray2color is not None:
            return np.array([gray2color(seq[i], cmap=cmap)
                             for i in range(len(seq))])
        return seq

    colored_pred = apply_color(pred_seq)
    if gt_seq is not None:
        colored_gt = apply_color(gt_seq)

    # 4. 创建拼接图像（带间隙）
    def create_grid_with_gap(seq, gap_width=gap):
        if len(seq) == 0:
            return None

        h, w, c = seq[0].shape
        gap_image = np.ones((h, gap_width, c), dtype=seq[0].dtype)

        with_gaps = []
        for i, img in enumerate(seq):
            with_gaps.append(img)
            if i < len(seq) - 1:
                with_gaps.append(gap_image)

        return np.concatenate(with_gaps, axis=1)

    # 5. 创建并保存所有序列
    grid_pred = create_grid_with_gap(colored_pred)
    if gt_seq is not None:
        grid_gt = create_grid_with_gap(colored_gt)

    plt.imsave(os.path.join(save_path, f"{pic_name}.png"), grid_pred)
    if gt_seq is not None:
        plt.imsave(os.path.join(save_path, "gt.png"), grid_gt)

    # 6. 如果提供了输入序列，也保存它
    if input_seq is not None:
        grid_input = create_grid_with_gap(colored_input)
        plt.imsave(os.path.join(save_path, "input.png"), grid_input)


# 创建可视化函数的偏函数
color_fn = partial(vis_res,
                   pixel_scale=PIXEL_SCALE,
                   gray2color=gray2color)


# ==================== 原有的训练和验证函数 ====================

def train(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0

    for iter, data in enumerate(train_loader):
        input = data[:, :5, :, :, :]  # 5-->16
        target = data[:, 5:, :, :, :]  # 5-->16
        input, target = input.to('cuda').float(), target.to('cuda').float()

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, target)

        # 普通反向传播
        loss.backward()
        optimizer.step()

        # 可选: 梯度裁剪 (根据需要取消注释)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        # 可选: 学习率调度 (根据需要取消注释)
        # scheduler.step()

        total_loss += loss.item()

        # 打印训练进度
        if iter % 400 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Train Epoch: {epoch} [{iter * len(input)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} LR: {lr:.2e}')

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            # print(data.shape)([1, 25, 1, 256, 256])
            # data = data.squeeze(2)
            input = data[:, :5, :, :, :]  # 5-->16
            target = data[:, 5:, :, :, :]  # 5-->16
            input, target = input.to('cuda').float(), target.to('cuda').float()
            # with autocast():

            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def test(model, test_loader, criterion, save_vis=True,
         vis_save_path="/home/ubuntu/zj/KM-UNetV3/outputs/test_visualization"):
    """
    修改后的测试函数，使用新的可视化方式
    """
    model.eval()
    total_loss = 0.0

    # 初始化评估器
    evaluator = SimplifiedEvaluator(
        seq_len=20,
        value_scale=90,
        thresholds=[20, 30, 35, 40]
    )

    # 创建可视化保存目录
    if save_vis:
        os.makedirs(vis_save_path, exist_ok=True)

    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            # data = data.squeeze(2)
            input = data[:, :5, :, :, :]  # 5-->16
            target = data[:, 5:, :, :, :]  # 5-->16
            input, target = input.to('cuda').float(), target.to('cuda').float()


            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item()

            output = output.data.squeeze(2)
            target = target.data.squeeze(2)
            input = input.data.squeeze(2)
            # 转换为numpy并确保数值范围
            pred_batch = output.float().cpu().numpy().clip(0, 1)  # [B,20,H,W]
            true_batch = target.float().cpu().numpy().clip(0, 1)  # [B,20,H,W]
            input_batch = input.float().cpu().numpy().clip(0, 1)  # [B,5,H,W]

            # 评估当前batch
            evaluator.evaluate(true_batch, pred_batch)

            # 使用新的可视化方法
            if save_vis and iter < 10:  # 只保存前10个batch的可视化结果
                batch_size = pred_batch.shape[0]
                for i in range(batch_size):
                    save_path_batch = os.path.join(vis_save_path, f"batch_{iter}_sample_{i}")
                    color_fn(pred_batch[i],
                             true_batch[i],
                             save_path=save_path_batch,
                             pic_name="prediction",
                             cmap=None,
                             even_index_only=False,  # 上海数据集使用偶数索引
                             input_seq=input_batch[i])

        # 计算最终指标
        metrics = evaluator.done()
        avg_loss = total_loss / len(test_loader)

        # 打印结果
        print(f"\nTest Loss: {avg_loss:.4f}")
        print("Threshold Metrics:")
        for thresh, m in metrics["threshold_metrics"].items():
            print(f"{thresh}mm | CSI: {m['CSI']:.4f}  POD: {m['POD']:.4f}  HSS: {m['HSS']:.4f}")

        print("\nRegression Metrics:")
        print(f"RMSE: {metrics['RMSE']:.2f}  SSIM: {metrics['SSIM']:.4f}")
        print(f"LPIPS: {metrics['LPIPS']:.4f}  FAR: {metrics['FAR']:.4f}")

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
    def __init__(self, alpha=0.7):  # alpha=0.7-->0.65
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
        # 0.6-->0.55;0.4-->0.45

def main():
    # 配置混合精度训练
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    # 模型实例化
    model = SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=20).cuda()

    # 注册钩子到所有子模块
    # for module in model.modules():
    #     module.register_forward_hook(nan_hook)
    base_LR = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
    criterion = RAINlOSS().cuda()

    # ===================== 上海数据集加载 ========================
    # 初始化完整数据集
    full_dataset = Shanghai(
        data_path='/home/ubuntu/zj/KM-UNetV3/inputs/shanghai.h5',
        img_size=256,
        type='train'  # 基础类型设为train
    )
    all_indices = np.arange(len(full_dataset))
    # print(all_indices)
    l = len(full_dataset)

    train_end = math.floor(l * 0.6)
    val_end = math.floor(l * 0.8)

    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:l]

    # 创建划分后的数据集
    train_dataset = CustomShanghai(full_dataset, train_indices)
    val_dataset = CustomShanghai(full_dataset, val_indices)
    test_dataset = CustomShanghai(full_dataset, test_indices)

    # 创建DataLoader（调整batch_size和num_workers）
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # 根据GPU显存调整
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # ===========================================================

    # 学习率调度器（根据实际训练步数调整）
    epochs = 150

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150, eta_min=1e-3)

    best_val_loss = 1234

    with open(r'/home/ubuntu/zj/KM-UNetV3/outputs/log/logswimunet.csv', 'w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(['epoch', 'train_loss', 'val_loss'])

        for epoch in range(epochs):
            print('Epoch [%d/%d]' % (epoch, epochs))

            train_loss = train(model, train_loader, criterion, optimizer, scheduler, epoch)

            val_loss = validate(model, val_loader, criterion)

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
                torch.save(model.state_dict(), r'/home/ubuntu/zj/KM-UNetV3/outputs/other_model_on_shanghai/model_Best_swimunet.pth')
                print("=========>>>>>>>>> Saved  best model")
                print('Loss: %.4f' % best_val_loss)

            torch.cuda.empty_cache()

    print("=============================Test=================================")
    # 最终测试 - 使用新的可视化方法
    model_path = r'/home/ubuntu/zj/KM-UNetV3/outputs/other_model_on_shanghai/model_Best_swimunet.pth'
    model.load_state_dict(torch.load(model_path))

    # 使用新的可视化保存路径
    vis_save_path = "/home/ubuntu/zj/KM-UNetV3/outputs/other_model_vis_shanghai/vis_swimunet"
    test_loss = test(model, test_loader, criterion, save_vis=True, vis_save_path=vis_save_path)
    print("Visualization Finished")


if __name__ == "__main__":
    main()