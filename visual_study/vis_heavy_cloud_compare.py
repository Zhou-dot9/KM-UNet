# -*- coding: utf-8 -*-
"""厚云 / 强回波个例对比：KM-UNet vs CSDP-Net vs SCECA vs TrajGRU（上海）。

按测试集 GT_max ≥35 mm 占比选出厚云个例，同一输入上对比四模型。

用法（在 visual_study 目录）:
    python vis_heavy_cloud_compare.py
"""
import os
import sys
import math
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import torch
from torch.cuda.amp import autocast

ROOT = r'D:/zhoujian/KM-UNet'
CSDP_ROOT = r'D:/zhoujian/KM-UNet/models/CSDP-Net/CSDP-Net-main'
MODELS = os.path.join(ROOT, 'models')
for p in (ROOT, CSDP_ROOT, MODELS):
    if p not in sys.path:
        sys.path.insert(0, p)

from KM_UNetV3_SH import KM_UNetV3
from CSDPNet import CSDPNet
from SCECA_Net import SCECANet
from TrajGru import encoder, forecaster, EF
from Shanghai import Shanghai, CustomShanghai


CKPT_KM = r'D:/zhoujian/KM-UNet/outputs/best_model/model_Best_shanghai_DAGEM_DySample_0.55.pth'
CKPT_CSDP = r'D:/zhoujian/KM-UNet/outputs/other_model_on_shanghai/model_Best_csdpnet.pth'
CKPT_SCECA = r'D:/zhoujian/KM-UNet/outputs/other_model_on_shanghai/model_Best_SCECA.pth'
CKPT_TRAJ = r'D:/zhoujian/KM-UNet/outputs/other_model_on_shanghai/model_Best_trajgru.pth'
DATA = r'D:/zhoujian/KM-UNet/shanghai.h5'
SAVE_ROOT = r'D:/zhoujian/KM-UNet/outputs/visual_study/heavy_cloud_compare'
HEAVY_MM = 35.0
PIXEL_SCALE = 90.0
TOP_K = 8
MIN_HEAVY = 0.03
LEAD_FRAMES = (9, 19)  # t+10 / t+20
MODEL_KEYS = ('km', 'csdp', 'sceca', 'traj')
MODEL_NAMES = {
    'km': 'KM-UNet',
    'csdp': 'CSDP-Net',
    'sceca': 'SCECA',
    'traj': 'TrajGRU',
}

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
]) / 255.0
BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


def gray2color(image):
    cmap = colors.ListedColormap(COLOR_MAP)
    norm = colors.BoundaryNorm(BOUNDS, cmap.N)
    return cmap(norm(image))[..., :3]


def radar_rgb(frame_01):
    mm = np.clip(np.asarray(frame_01) * PIXEL_SCALE, 0, PIXEL_SCALE)
    return gray2color(mm)


def to_numpy(t):
    return t.detach().float().cpu().numpy()


def squeeze_pred(y):
    y = np.clip(np.asarray(y), 0, 1)
    if y.ndim == 4:
        y = y[:, 0]
    return y


def _draw_contour(ax, heavy_mask):
    if heavy_mask.any():
        ax.contour(heavy_mask.astype(float), levels=[0.5], colors='white', linewidths=0.8)


def csi_score(pred01, gt01, thr_mm=HEAVY_MM):
    p = np.asarray(pred01) * PIXEL_SCALE >= thr_mm
    g = np.asarray(gt01) * PIXEL_SCALE >= thr_mm
    hits = np.logical_and(p, g).sum()
    fa = np.logical_and(p, ~g).sum()
    miss = np.logical_and(~p, g).sum()
    den = hits + fa + miss
    return float(hits / den) if den else float('nan')


def rmse_mm(pred01, gt01, mask=None):
    d = (np.asarray(pred01) - np.asarray(gt01)) * PIXEL_SCALE
    if mask is not None:
        if not np.asarray(mask).any():
            return float('nan')
        d = d[np.asarray(mask)]
    return float(np.sqrt(np.mean(d ** 2)))


def error_rgb(pred01, gt01, vmax=20.0):
    err = np.abs(np.asarray(pred01) - np.asarray(gt01)) * PIXEL_SCALE
    norm = np.clip(err / vmax, 0.0, 1.0)
    return cm.hot(norm)[..., :3]


def build_test_dataset():
    full = Shanghai(data_path=DATA, img_size=256, type='train')
    l = len(full)
    val_end = math.floor(l * 0.8)
    indices = np.arange(l)[val_end:l]
    return CustomShanghai(full, indices), indices


def rank_heavy_cases(dataset):
    scores = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample.ndim == 4:
            tgt = sample[5:, 0].numpy()
        else:
            tgt = sample[5:].numpy()
        gt_max = tgt.max(axis=0)
        heavy = (gt_max * PIXEL_SCALE) >= HEAVY_MM
        scores.append((float(heavy.mean()), i))
        if (i + 1) % 50 == 0:
            print(f'  scanned {i + 1}/{len(dataset)}')
    scores.sort(key=lambda x: -x[0])
    picked = [(h, i) for h, i in scores if h >= MIN_HEAVY][:TOP_K]
    return picked or scores[:TOP_K]


def load_models(device):
    km = KM_UNetV3(num_classes=20).to(device)
    km.load_state_dict(torch.load(CKPT_KM, map_location=device))
    km.eval()
    print(f'loaded KM-UNet  {CKPT_KM}')

    csdp = CSDPNet(seqNum=20, filter_size=5, img_size=256).to(device)
    csdp.load_state_dict(torch.load(CKPT_CSDP, map_location=device))
    csdp.eval()
    print(f'loaded CSDP-Net {CKPT_CSDP}')

    sceca = SCECANet(in_channels=5, out_channels=20).to(device)
    sceca.load_state_dict(torch.load(CKPT_SCECA, map_location=device))
    sceca.eval()
    print(f'loaded SCECA    {CKPT_SCECA}')

    traj = EF(encoder, forecaster, 20).to(device)
    traj.load_state_dict(torch.load(CKPT_TRAJ, map_location=device))
    traj.eval()
    print(f'loaded TrajGRU  {CKPT_TRAJ}')
    return {'km': km, 'csdp': csdp, 'sceca': sceca, 'traj': traj}


def predict_all(models, inp_b5hw, device):
    x_hw = torch.from_numpy(inp_b5hw).float().to(device)
    x_c = torch.from_numpy(inp_b5hw[:, :, None]).float().to(device)
    out = {}
    with torch.no_grad(), autocast():
        out['km'] = squeeze_pred(to_numpy(models['km'](x_hw))[0])
        out['csdp'] = squeeze_pred(to_numpy(models['csdp'](x_c))[0])
        out['sceca'] = squeeze_pred(to_numpy(models['sceca'](x_hw))[0])
        y_traj = to_numpy(models['traj'](x_c))[0]
        out['traj'] = squeeze_pred(y_traj)
    return out


def save_panel(save_dir, radar_in, gt, preds, heavy_max, stats):
    os.makedirs(save_dir, exist_ok=True)
    gt_max = gt.max(axis=0)
    pred_max = {k: preds[k].max(axis=0) for k in MODEL_KEYS}

    fig = plt.figure(figsize=(19.5, 12.6))
    gs = fig.add_gridspec(3, 6, left=0.02, right=0.99, top=0.94, bottom=0.055,
                          wspace=0.04, hspace=0.16)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(6)] for r in range(3)]

    row0 = [('Input (last)', radar_rgb(radar_in), False),
            ('GT (max)', radar_rgb(gt_max), True)]
    for k in MODEL_KEYS:
        title = f"{MODEL_NAMES[k]} (max)  CSI@35={stats[f'csi_{k}_max']:.3f}"
        row0.append((title, radar_rgb(pred_max[k]), True))
    for ax, (title, img, contour) in zip(axes[0], row0):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=5)
        ax.axis('off')
        if contour:
            _draw_contour(ax, heavy_max)

    for row_i, t in enumerate(LEAD_FRAMES, start=1):
        cells = [(f'GT  t+{t + 1}', radar_rgb(gt[t]), True)]
        # 第一列空位对齐：用 Input 占位只在 max 行；这里第一列放 blank
        for k in MODEL_KEYS:
            cells.append((f'{MODEL_NAMES[k]}  t+{t + 1}', radar_rgb(preds[k][t]), True))
        axes[row_i][0].axis('off')
        for ax, (title, img, contour) in zip(axes[row_i][1:], cells):
            ax.imshow(img)
            ax.set_title(title, fontsize=10, pad=5)
            ax.axis('off')
            if contour:
                _draw_contour(ax, heavy_max)

    seq = '   '.join(f"{MODEL_NAMES[k]}={stats[f'csi_{k}_seq']:.3f}" for k in MODEL_KEYS)
    fig.text(
        0.5, 0.012,
        f'white contour ≥{HEAVY_MM:.0f} mm   |   seq CSI@35  {seq}',
        ha='center', va='bottom', fontsize=11,
    )
    fig.savefig(os.path.join(save_dir, 'panel.png'), dpi=150)
    plt.close(fig)

    # 误差图：四模型 max |pred-GT|
    fig, axes_e = plt.subplots(1, 4, figsize=(16.5, 4.0))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.06, wspace=0.05)
    for ax, k in zip(axes_e, MODEL_KEYS):
        ax.imshow(error_rgb(pred_max[k], gt_max))
        ax.set_title(
            f"|{MODEL_NAMES[k]}−GT|  RMSE_h={stats[f'rmse_{k}_h']:.1f} mm",
            fontsize=11,
        )
        ax.axis('off')
        _draw_contour(ax, heavy_max)
    fig.savefig(os.path.join(save_dir, 'error_max.png'), dpi=140)
    plt.close(fig)

    plt.imsave(os.path.join(save_dir, 'input_last.png'), radar_rgb(radar_in))
    plt.imsave(os.path.join(save_dir, 'gt_max.png'), radar_rgb(gt_max))
    for k in MODEL_KEYS:
        plt.imsave(os.path.join(save_dir, f'{k}_max.png'), radar_rgb(pred_max[k]))
        plt.imsave(os.path.join(save_dir, f'err_{k}_max.png'), error_rgb(pred_max[k], gt_max))


def save_overview(rows, dataset, preds_cache):
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(18.5, 2.9 * n))
    if n == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.02,
                        wspace=0.03, hspace=0.18)
    titles = ['Input (last)', 'GT (max)'] + [f'{MODEL_NAMES[k]} (max)' for k in MODEL_KEYS]
    for r, row in enumerate(rows):
        sample = dataset[row['local_idx']]
        frames = sample.squeeze().numpy() if torch.is_tensor(sample) else np.asarray(sample).squeeze()
        gt_max = frames[5:].max(axis=0)
        heavy = (gt_max * PIXEL_SCALE) >= HEAVY_MM
        cache = preds_cache[row['local_idx']]
        imgs = [radar_rgb(frames[4]), radar_rgb(gt_max)] + [
            radar_rgb(cache[k].max(axis=0)) for k in MODEL_KEYS
        ]
        for c, (ax, img) in enumerate(zip(axes[r], imgs)):
            ax.imshow(img)
            ax.axis('off')
            if c > 0:
                _draw_contour(ax, heavy)
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        axes[r, 0].set_ylabel(
            f"{row['tag']}\n{row['heavy_ratio']:.2f}",
            fontsize=9, rotation=0, ha='right', va='center', labelpad=22,
        )
    fig.savefig(os.path.join(SAVE_ROOT, 'overview.png'), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  heavy>={HEAVY_MM}mm  top_k={TOP_K}')

    dataset, _ = build_test_dataset()
    print(f'test size={len(dataset)}, ranking heavy-cloud cases...')
    picked = rank_heavy_cases(dataset)
    for h, i in picked:
        print(f'  local_idx={i}  heavy_ratio={h:.4f}')

    models = load_models(device)
    rows, preds_cache = [], {}
    for rank, (heavy_ratio, local_idx) in enumerate(picked):
        sample = dataset[local_idx]
        frames = sample.squeeze(1).numpy() if sample.ndim == 4 else sample.numpy()
        inp = frames[:5][None].astype(np.float32)
        gt = frames[5:].astype(np.float32)
        heavy_max = (gt.max(axis=0) * PIXEL_SCALE) >= HEAVY_MM

        preds = predict_all(models, inp, device)
        preds_cache[local_idx] = preds

        stats = {}
        for k in MODEL_KEYS:
            stats[f'csi_{k}_max'] = csi_score(preds[k].max(0), gt.max(0))
            stats[f'csi_{k}_seq'] = csi_score(preds[k], gt)
            stats[f'rmse_{k}_h'] = rmse_mm(preds[k].max(0), gt.max(0), heavy_max)

        tag = f'case_{rank:02d}'
        save_panel(os.path.join(SAVE_ROOT, tag), frames[4], gt, preds, heavy_max, stats)
        row = {'tag': tag, 'local_idx': int(local_idx), 'heavy_ratio': heavy_ratio, **stats}
        rows.append(row)
        csi_txt = ' '.join(f"{k}={stats[f'csi_{k}_max']:.3f}" for k in MODEL_KEYS)
        print(f'saved {tag}  heavy={heavy_ratio:.3f}  CSI_max {csi_txt}')
        torch.cuda.empty_cache()

    csv_path = os.path.join(SAVE_ROOT, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_overview(rows, dataset, preds_cache)
    print(f'Done. {len(rows)} cases -> {SAVE_ROOT}')


if __name__ == '__main__':
    main()
