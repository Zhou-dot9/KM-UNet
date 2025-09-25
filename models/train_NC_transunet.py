# 在文件最开头添加以下两行代码（在所有import之前）
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt

import torch

from TransUnet import VisionTransformer,CONFIGS
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
    save_dir = "/home/ubuntu/zj/KM-UNetV3/outputs/other_model_vis_LAPS/transunet/pred"
    os.makedirs(save_dir, exist_ok=True)
    gts_dir = "/home/ubuntu/zj/KM-UNetV3/outputs/other_model_vis_LAPS/transunet/gts"
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
    # 模型实例化
    model = VisionTransformer(config=CONFIGS['R50-ViT-B_16_3']).cuda()#LAPS:'R50-ViT-B_16_3'

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

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)

    scaler = GradScaler()
    best_val_loss = 1234

    with open(r'/home/ubuntu/zj/KM-UNetV3/outputs/log/log_transunet.csv', 'w', newline='') as log_file:
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
                torch.save(model.state_dict(), r'/home/ubuntu/zj/KM-UNetV3/outputs/other_model_on_LAPS/model_Best_transunet.pth')
                print("=========>>>>>>>>> Saved  best model")
                print('Loss: %.4f' % best_val_loss)
    # #
    #         torch.cuda.empty_cache()

    print("=============================Test=================================")
    # 最终测试
    model_path = r'/home/ubuntu/zj/KM-UNetV3/outputs/other_model_on_LAPS/model_Best_transunet.pth'
    model.load_state_dict(torch.load(model_path))
    test_loss = test(model, test_loader,criterion)
    print("Save Picture Finished")


if __name__ == "__main__":
    main()






