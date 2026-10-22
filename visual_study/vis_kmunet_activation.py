# -*- coding: utf-8 -*-
"""KM-UNet 中间特征激活（上海数据集，关注厚云）。

权重: model_Best_shanghai_DAGEM_DySample_0.55.pth
在 enc1/enc2/enc3 的 KANConv 输出上计算通道能量，叠到 GT ≥35 mm 上。
能量图为原始分辨率双线性上采样到 256。默认测试集前 10 个 batch。

用法（在 visual_study 目录）:
    python vis_kmunet_activation.py
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
from torch.utils.data import DataLoader
import torch.nn.functional as F

ROOT = r'D:/zhoujian/KM-UNet'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from KM_UNetV3_SH import KM_UNetV3
from Shanghai import Shanghai, CustomShanghai


CKPT = r'D:/zhoujian/KM-UNet/outputs/best_model/model_Best_shanghai_DAGEM_DySample_0.55.pth'
DATA = r'D:/zhoujian/KM-UNet/shanghai.h5'
SAVE_ROOT = r'D:/zhoujian/KM-UNet/outputs/visual_study/kmunet_activation'
HEAVY_MM = 35.0
PIXEL_SCALE = 90.0
MAX_BATCHES = 10
BATCH_SIZE = 1
NUM_WORKERS = 0
STAGES = ('enc1', 'enc2', 'enc3')

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
    return cmap(norm(image))


def to_numpy(t):
    return t.detach().float().cpu().numpy()


def joint_limits(*maps, p_lo=2.0, p_hi=98.0):
    stacked = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in maps])
    lo = float(np.percentile(stacked, p_lo))
    hi = float(np.percentile(stacked, p_hi))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return lo, hi


def scale01(x, lo, hi):
    return np.clip((np.asarray(x, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def upsample_map(arr, size=256):
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    else:
        t = t.unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=True)
    return t.squeeze().numpy()


def radar_rgb(frame_01):
    mm = np.clip(frame_01 * PIXEL_SCALE, 0, PIXEL_SCALE)
    rgba = gray2color(mm)
    return rgba[..., :3]


def overlay_heat(radar_01, heat_map, heavy_mask, lo=None, hi=None,
                 alpha_bg=0.35, alpha_heavy=0.70):
    base = radar_rgb(radar_01)
    if lo is None or hi is None:
        lo, hi = joint_limits(heat_map)
    heat = cm.jet(scale01(heat_map, lo, hi))[..., :3]
    a = alpha_bg + (alpha_heavy - alpha_bg) * heavy_mask.astype(np.float32)
    a = a[..., None]
    return np.clip((1.0 - a) * base + a * heat, 0.0, 1.0)


def feature_energy(feat_chw):
    return torch.sqrt(feat_chw.float().pow(2).mean(dim=0) + 1e-12).cpu().numpy()


def concentration(resp, heavy):
    resp = np.asarray(resp, dtype=np.float64)
    inside = heavy.astype(bool)
    outside = ~inside
    if (not inside.any()) or (not outside.any()):
        return float('nan')
    return float(resp[inside].mean() / (resp[outside].mean() + 1e-8))


class FeatureCatcher:
    """挂钩 enc*[0] = StableHybridKANConv 输出。"""

    def __init__(self, model):
        self.store = {}
        self.handles = [
            getattr(model, name)[0].register_forward_hook(self._make_hook(name))
            for name in STAGES
        ]

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            self.store[name] = out.detach().float()
        return hook

    def clear(self):
        self.store = {}

    def close(self):
        for h in self.handles:
            h.remove()


def _draw_contour(ax, heavy_mask):
    if heavy_mask.any():
        ax.contour(heavy_mask.astype(float), levels=[0.5], colors='white', linewidths=0.8)


def build_loader():
    full_dataset = Shanghai(data_path=DATA, img_size=256, type='train')
    all_indices = np.arange(len(full_dataset))
    l = len(full_dataset)
    val_end = math.floor(l * 0.8)
    test_dataset = CustomShanghai(full_dataset, all_indices[val_end:l])
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def save_panel(save_dir, radar_in, radar_gt, radar_pred, resp, heavy_mask, stats):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.04,
                        wspace=0.08, hspace=0.18)

    top = [
        ('Input (last)', radar_rgb(radar_in), False),
        ('GT (max, heavy cloud)', radar_rgb(radar_gt), True),
        ('Pred (max)', radar_rgb(radar_pred), False),
    ]
    for ax, (title, img, contour) in zip(axes[0], top):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis('off')
        if contour:
            _draw_contour(ax, heavy_mask)

    native = {'enc1': '256', 'enc2': '128', 'enc3': '64'}
    for ax, stage in zip(axes[1], STAGES):
        energy = resp[stage]
        lo, hi = joint_limits(energy)
        ov = overlay_heat(radar_gt, energy, heavy_mask, lo=lo, hi=hi)
        ax.imshow(ov)
        r = stats[f'R_{stage}']
        ax.set_title(f'{stage} energy  ({native[stage]}→256)  R={r:.2f}', fontsize=12, pad=8)
        ax.axis('off')
        _draw_contour(ax, heavy_mask)
        plt.imsave(os.path.join(save_dir, f'{stage}.png'), ov)

    fig.savefig(os.path.join(save_dir, 'panel.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    plt.imsave(os.path.join(save_dir, 'input_last.png'), radar_rgb(radar_in))
    plt.imsave(os.path.join(save_dir, 'gt_max.png'), radar_rgb(radar_gt))
    plt.imsave(os.path.join(save_dir, 'pred_max.png'), radar_rgb(radar_pred))


def save_summary(rows):
    csv_path = os.path.join(SAVE_ROOT, 'concentration.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r['heavy_ratio'] > 0.005 and np.isfinite(r['R_enc2'])]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    if valid:
        x = np.arange(len(valid))
        w = 0.24
        colors_b = ['#4C78A8', '#F58518', '#54A24B']
        for i, stage in enumerate(STAGES):
            vals = [r[f'R_{stage}'] for r in valid]
            ax.bar(x + (i - 1) * w, vals, w, label=stage, color=colors_b[i])
            ax.axhline(np.mean(vals), color=colors_b[i], ls='--', lw=1)
        ax.axhline(1.0, color='0.5', ls=':', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([r['sample'].replace('_sample_0', '') for r in valid],
                           rotation=45, ha='right', fontsize=8)
        means = {s: np.mean([r[f'R_{s}'] for r in valid]) for s in STAGES}
        ax.set_ylabel('Heavy-cloud concentration R')
        ax.set_title(
            f"R>1: stronger in ≥{HEAVY_MM:.0f} mm    "
            + '  '.join(f'{s}={means[s]:.2f}' for s in STAGES)
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_ROOT, 'summary_R.png'), dpi=150)
    plt.close(fig)
    print(f'summary -> {csv_path}')


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  heavy>={HEAVY_MM}mm  batches={MAX_BATCHES}')

    model = KM_UNetV3(num_classes=20).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f'loaded {CKPT}')

    catcher = FeatureCatcher(model)
    loader = build_loader()
    rows = []
    saved = 0
    with torch.no_grad():
        for bidx, data in enumerate(loader):
            if bidx >= MAX_BATCHES:
                break
            data = data.squeeze(2)
            inp = data[:, :5, :, :].to(device).float()
            tgt = data[:, 5:, :, :].to(device).float()

            catcher.clear()
            with autocast():
                pred = model(inp)

            inp_np = to_numpy(inp)
            tgt_np = to_numpy(tgt)
            pred_np = to_numpy(pred)

            for i in range(inp_np.shape[0]):
                radar_in = inp_np[i, -1]
                radar_gt = tgt_np[i].max(axis=0)
                radar_pred = pred_np[i].max(axis=0)
                heavy_mask = (radar_gt * PIXEL_SCALE) >= HEAVY_MM

                resp, stats = {}, {}
                for stage in STAGES:
                    energy = upsample_map(feature_energy(catcher.store[stage][i]), 256)
                    resp[stage] = energy
                    stats[f'R_{stage}'] = concentration(energy, heavy_mask)

                out_dir = os.path.join(SAVE_ROOT, f'batch_{bidx}_sample_{i}')
                save_panel(out_dir, radar_in, radar_gt, radar_pred, resp, heavy_mask, stats)
                row = {
                    'sample': f'batch_{bidx}_sample_{i}',
                    'heavy_ratio': float(heavy_mask.mean()),
                    **stats,
                }
                rows.append(row)
                saved += 1
                print(
                    f"saved {out_dir}  heavy={row['heavy_ratio']:.3f}  "
                    f"R_enc1={stats['R_enc1']:.3f}  R_enc2={stats['R_enc2']:.3f}  "
                    f"R_enc3={stats['R_enc3']:.3f}"
                )

    catcher.close()
    if rows:
        save_summary(rows)
    print(f'Done. {saved} samples -> {SAVE_ROOT}')


if __name__ == '__main__':
    main()
