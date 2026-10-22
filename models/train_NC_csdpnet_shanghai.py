# -*- coding: utf-8 -*-
# 在文件最开头添加以下两行代码（在所有import之前）
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt

import sys
CSDP_ROOT = r'D:/zhoujian/KM-UNet/models/CSDP-Net/CSDP-Net-main'
if CSDP_ROOT not in sys.path:
    sys.path.insert(0, CSDP_ROOT)

import torch

from CSDPNet import CSDPNet
import torch.nn.functional as F
import numpy as np
import h5py
import csv

import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
# from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, mean_squared_error
from skimage.metrics import structural_similarity as ssim  # for metrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

from Shanghai import Shanghai, CustomShanghai
import math
from metrics import SimplifiedEvaluator
import os
import time
from thop import profile
from ssim import SSIM as CSDP_SSIM

# 导入可视化相关模块
from matplotlib import colors
from functools import partial


# 在程序开头启用异常检测
# torch.autograd.set_detect_anomaly(True)

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


def _format_params(n):
    if n >= 1e6:
        return f'{n / 1e6:.3f} M'
    if n >= 1e3:
        return f'{n / 1e3:.3f} K'
    return str(n)


def _format_flops(n):
    if n is None:
        return 'N/A'
    if n >= 1e9:
        return f'{n / 1e9:.3f} G'
    if n >= 1e6:
        return f'{n / 1e6:.3f} M'
    return f'{n:.0f}'


def measure_flops(model, dummy_input):
    """用 thop 统计一次前向 FLOPs；自定义算子失败时返回 None。"""
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(flops)
    except Exception as e:
        print(f'[Efficiency] FLOPs (thop) failed: {e}')
        return None


def print_efficiency(stats):
    print('================ Efficiency ================')
    print(f"Params (total)      : {_format_params(stats['params_total'])} ({stats['params_total']})")
    print(f"Params (trainable)  : {_format_params(stats['params_trainable'])} ({stats['params_trainable']})")
    print(f"FLOPs               : {_format_flops(stats['flops'])}  (input {stats['input_shape']})")
    print(f"Peak GPU memory     : {stats['peak_mem_mb']:.2f} MB")
    print(f"Inference time      : {stats['infer_ms']:.2f} ms / sample")
    print(f"Throughput          : {stats['fps']:.2f} samples/s")
    print(f"Timed samples       : {stats['timed_samples']}")
    print('============================================')


def train(model, train_loader, criterion, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0

    for iter, data in enumerate(train_loader):
        # print(data.shape)#([1, 25, 1, 256, 256])
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


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            input = data[:, :5, :, :, :]
            target = data[:, 5:, :, :, :]
            input, target = input.to('cuda').float(), target.to('cuda').float()
            # with autocast():
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def test(model, test_loader, criterion, save_vis=True,
         vis_save_path="D:/zhoujian/KM-UNet/outputs/other_model_vis_shanghai/vis_csdpnet-shanghai"):
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

    # ---------- 计算效率：参数量 / FLOPs ----------
    sample = test_loader.dataset[0]
    if sample.ndim == 4:
        _, _, h, w = sample.shape
    else:
        _, h, w = sample.shape
    dummy = torch.randn(1, 5, 1, h, w, device='cuda', dtype=torch.float32)
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = measure_flops(model, dummy)

    warmup = 3
    infer_times = []
    timed_samples = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            input = data[:, :5, :, :, :]
            target = data[:, 5:, :, :, :]
            input, target = input.to('cuda').half(), target.to('cuda').half()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast():
                output = model(input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if iter >= warmup:
                infer_times.append(t1 - t0)
                timed_samples += input.shape[0]

            with autocast():
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

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    if infer_times:
        mean_s = float(np.mean(infer_times))
        infer_ms = mean_s * 1000.0 / max(test_loader.batch_size or 1, 1)
        fps = timed_samples / sum(infer_times)
    else:
        infer_ms = 0.0
        fps = 0.0

    print_efficiency({
        'params_total': params_total,
        'params_trainable': params_trainable,
        'flops': flops,
        'input_shape': tuple(dummy.shape),
        'peak_mem_mb': peak_mem_mb,
        'infer_ms': infer_ms,
        'fps': fps,
        'timed_samples': timed_samples,
    })

    return total_loss / len(test_loader)


def nan_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            raise ValueError(f"NaN in {module}")
    elif isinstance(output, (tuple, list)):
        for out in output:
            if torch.isnan(out).any():
                raise ValueError(f"NaN in {module}")


class CSDPOrigLoss(nn.Module):
    """CSDP-Net 原文损失: (MSE + 0.5 * (1 - SSIM)) / 2"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = CSDP_SSIM()

    def forward(self, pred, target):
        return (self.mse(pred, target) + (1 - self.ssim(pred, target)) * 0.5) / 2


def main():
    # 配置混合精度训练
    torch.backends.cudnn.benchmark = True
    # 模型实例化：上海数据为 256×256、5 帧输入、20 帧输出
    model = CSDPNet(seqNum=20, filter_size=5, img_size=256).cuda()

    # 注册钩子到所有子模块
    # for module in model.modules():
    #     module.register_forward_hook(nan_hook)

    base_LR = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)

    criterion = CSDPOrigLoss().cuda()

    # ===================== 上海数据集加载 ========================
    # 初始化完整数据集
    full_dataset = Shanghai(
        data_path='D:/zhoujian/KM-UNet/shanghai.h5',
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

    # CSDP-Net 20 帧解码显存高于 U-Net 类模型，batch=1 约 8.5GB；batch=2 约 17GB
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # ===========================================================

    # 学习率调度器（根据实际训练步数调整）
    epochs = 120

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)

    scaler = GradScaler()
    best_val_loss = 1234

    os.makedirs(r'D:/zhoujian/KM-UNet/outputs/log', exist_ok=True)
    os.makedirs(r'D:/zhoujian/KM-UNet/outputs/other_model_on_shanghai', exist_ok=True)
    ckpt_path = r'D:/zhoujian/KM-UNet/outputs/other_model_on_shanghai/model_Best_csdpnet.pth'

    # with open(r'D:/zhoujian/KM-UNet/outputs/log/log_csdpnet_shanghai.csv', 'w', newline='') as log_file:
    #     log_writer = csv.writer(log_file)
    #     log_writer.writerow(['epoch', 'train_loss', 'val_loss'])

    #     for epoch in range(epochs):
    #         print('Epoch [%d/%d]' % (epoch, epochs))

    #         train_loss = train(model, train_loader, criterion, optimizer, scheduler, scaler, epoch)

    #         val_loss = validate(model, val_loader, criterion)

    #         scheduler.step()

    #         # 写入日志（新增部分）
    #         log_writer.writerow([
    #             epoch,
    #             round(train_loss, 4),
    #             round(val_loss, 4),
    #         ])

    #         print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    #         # 保存最佳模型
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             torch.save(model.state_dict(), ckpt_path)
    #             print("=========>>>>>>>>> Saved  best model")
    #             print('Loss: %.4f' % best_val_loss)
    #         torch.cuda.empty_cache()

    print("=============================Test=================================")
    # 最终测试
    model.load_state_dict(torch.load(ckpt_path))
    vis_save_path = "D:/zhoujian/KM-UNet/outputs/other_model_vis_shanghai/vis_csdpnet-shanghai"
    test_loss = test(model, test_loader, criterion, save_vis=True, vis_save_path=vis_save_path)
    print("Visualization Finished")


if __name__ == "__main__":
    main()
