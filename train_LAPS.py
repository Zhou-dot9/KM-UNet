# 在文件最开头添加以下两行代码（在所有import之前）
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt

import torch
from KM_UNetV3_LAPS import KM_UNetV3

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


def train(model, train_loader, criterion, optimizer, scheduler, scaler, epoch):

    model.train()
    total_loss = 0.0

    for iter, data in enumerate(train_loader):
        input = data[:, :5, :, :]  # 3-->5
        target = data[:, 5:, :, :]  # 3-->5
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
            input = data[:, :5, :, :]  # 3-->5
            target = data[:, 5:, :, :]  # 3-->5
            input, target = input.to('cuda').float(), target.to('cuda').float()
            # with autocast():
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def test(model, test_loader, criterion, save_vis=True,
         vis_save_path="/home/ubuntu/zj/KM-UNetV3/outputs/test_visualization"):

    model.eval()
    total_loss = 0.0
    preds = []
    gts = []
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8]
    save_dir= "/home/ubuntu/zj/KM-UNetV3/outputs/image_pred"
    # os.makedirs(save_dir, exist_ok=True)
    gts_dir="/home/ubuntu/zj/KM-UNetV3/outputs/image_gts"
    # os.makedirs(gts_dir, exist_ok=True)

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
            # print(data.shape)
            input = data[:, :5, :, :]  # 3-->5
            target = data[:, 5:, :, :]  # 3-->5
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

            # 转换为numpy并确保数值范围
            pred_batch = output  # [B,20,H,W]
            true_batch = target  # [B,20,H,W]
            input_batch = input.float().cpu().numpy()  # [B,5,H,W]

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
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # 如果是首次写入，添加表头
                if f.tell() == 0:
                    writer.writerow([
                        'Threshold',
                        'SSIM',
                        'FAR',
                        'CSI',
                        'HSS',
                        'POD',
                        'RMSE'
                    ])
                writer.writerow(row_data)

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
    # 模型实例化
    model = KM_UNetV3(num_classes=3).cuda()

    model_path = r'/home/ubuntu/zj/KM-UNetV3/outputs/best_model/model_Best_hybrid_loss_0.55.pth'
    model.load_state_dict(torch.load(model_path))

    # 注册钩子到所有子模块
    # for module in model.modules():
    #     module.register_forward_hook(nan_hook)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)#2e-4-->1e-3

    criterion=HybridLoss()

    # 添加以下代码加载自己数据集
    # -------------------------------------------------------------------------------------
    with h5py.File('./inputs/merged_data.h5', 'r') as hf:
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=5e-4)

    scaler = GradScaler()
    best_val_loss = 1234

    with open(r'/home/ubuntu/zj/KM-UNetV3/outputs/log/log.csv', 'w', newline='') as log_file:
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
                torch.save(model.state_dict(), r'/home/ubuntu/zj/KM-UNetV3/outputs/best_model/model_Best.pth')
                print("=========>>>>>>>>> Saved  best model")
                print('Loss: %.4f' % best_val_loss)

            torch.cuda.empty_cache()

    print("=============================Test=================================")
    # 最终测试
    # model_path = r'/home/ubuntu/zj/KM-UNetV3/outputs/best_model/model_Best.pth'
    # model.load_state_dict(torch.load(model_path))
    test_loss = test(model, test_loader,criterion)
    print("Save Picture Finished")


if __name__ == "__main__":
    main()






