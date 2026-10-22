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

# from Shanghai import Shanghai,CustomShanghai
import os
import time
from thop import profile
from ssim import SSIM as CSDP_SSIM
# from loss import RAINlOSS


# 在程序开头启用异常检测
# torch.autograd.set_detect_anomaly(True)

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


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            data = data.unsqueeze(2)
            input = data[:, :5, :, :, :]
            target = data[:, 5:, :, :, :]
            input, target = input.to('cuda').float(), target.to('cuda').float()
            # with autocast():
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0

    preds = []
    gts = []
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8]
    save_dir = "D:/zhoujian/KM-UNet/outputs/other_model_vis_LAPS/csdpnet/pred"
    os.makedirs(save_dir, exist_ok=True)
    gts_dir = "D:/zhoujian/KM-UNet/outputs/other_model_vis_LAPS/csdpnet/gts"
    os.makedirs(gts_dir, exist_ok=True)

    # ---------- 计算效率：参数量 / FLOPs ----------
    sample = test_loader.dataset[0]
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
            # print(data.shape)
            data = data.unsqueeze(2)
            input = data[:, :5, :, :, :]
            target = data[:, 5:, :, :, :]
            input, target = input.to('cuda').half(), target.to('cuda').half()
            # with autocast():
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
            # print(output.shape)

            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            # CSDP-Net: [B, T, 1, H, W] -> [B, T, H, W]
            if output.ndim == 5:
                output = np.squeeze(output, axis=2)
                target = np.squeeze(target, axis=2)
            preds.append(output)
            gts.append(target)

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

                    # 构造保存路径
                    save_path = os.path.join(
                        save_dir,
                        f'batch_{iter}_sample_{i}_channel_{j}.png'
                    )
                    # 保存并立即关闭图像
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

            batch_size_gts = true_batch.shape[0]
            for i in range(batch_size_gts):
                image = true_batch[i]  # 形状 [3, H, W]
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
                    plt.close()  # 必须关闭释放内存
        # metrics
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        print(" ")
        csv_path = "D:/zhoujian/KM-UNet/outputs/log/metrics.csv"
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
    torch.backends.cudnn.benchmark = True
    # 模型实例化：LAPS 为 256×256、5 帧输入、3 帧输出
    model = CSDPNet(seqNum=3, filter_size=5, img_size=256).cuda()

    # 注册钩子到所有子模块
    # for module in model.modules():
    #     module.register_forward_hook(nan_hook)

    base_LR = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)

    criterion = CSDPOrigLoss().cuda()

    # 添加以下代码加载自己数据集
    # -------------------------------------------------------------------------------------
    with h5py.File(r'D:/zhoujian/KM-UNet/merged_data.h5', 'r') as hf:
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
    epochs = 120

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)

    scaler = GradScaler()
    best_val_loss = 1234

    os.makedirs(r'D:/zhoujian/KM-UNet/outputs/log', exist_ok=True)
    os.makedirs(r'D:/zhoujian/KM-UNet/outputs/other_model_on_LAPS', exist_ok=True)
    ckpt_path = r'D:/zhoujian/KM-UNet/outputs/other_model_on_LAPS/model_Best_csdpnet.pth'

    # with open(r'D:/zhoujian/KM-UNet/outputs/log/log_csdpnet.csv', 'w', newline='') as log_file:
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
    test_loss = test(model, test_loader, criterion)
    print("Save Picture Finished")


if __name__ == "__main__":
    main()
