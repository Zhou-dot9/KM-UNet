# -*- coding: utf-8 -*-
"""普通卷积 vs KANConv 厚云特征响应对比（上海数据集）。

加载 ablation 上海权重：
  unet_baseline_shanghai / unet_kanconv_shanghai
在 enc1/enc2/enc3 的卷积块输出上计算通道能量图，叠到 GT 厚云（≥35 mm）上。
默认输出测试集前 10 个 batch。

用法（在 visual_study 目录）:
    python vis_kan_vs_conv.py
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
ABLATION = os.path.join(ROOT, 'ablation_model')
for p in (ROOT, ABLATION):
    if p not in sys.path:
        sys.path.insert(0, p)

from Shanghai import Shanghai, CustomShanghai
from unet_baseline import UNetBaseline
from unet_kanconv import UNetKANConv


CKPT_CONV = r'D:/zhoujian/KM-UNet/outputs/ablation/unet_baseline_shanghai/model_Best.pth'
CKPT_KAN = r'D:/zhoujian/KM-UNet/outputs/ablation/unet_kanconv_shanghai/model_Best.pth'
DATA = r'D:/zhoujian/KM-UNet/shanghai.h5'
SAVE_ROOT = r'D:/zhoujian/KM-UNet/outputs/visual_study/kan_vs_conv'
HEAVY_MM = 35.0
PIXEL_SCALE = 90.0
MAX_BATCHES = 10
BATCH_SIZE = 1
NUM_WORKERS = 0
STAGES = ('enc1', 'enc2', 'enc3')
MAIN_STAGE = 'enc2'   # 中层 128×128，更贴对流核；enc3 能量更散，作附录 stages.png

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
    """Conv / KAN 共用同一组分位上下界，颜色可直接对比。"""
    stacked = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in maps])
    lo = float(np.percentile(stacked, p_lo))
    hi = float(np.percentile(stacked, p_hi))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return lo, hi


def scale01(x, lo, hi):
    return np.clip((np.asarray(x, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def norm01(x, p=98.0):
    lo, hi = joint_limits(x, p_lo=2.0, p_hi=p)
    return scale01(x, lo, hi)


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
    """feat: [C,H,W] -> spatial L2 energy [H,W]."""
    return torch.sqrt(feat_chw.float().pow(2).mean(dim=0) + 1e-12).cpu().numpy()


def concentration(resp, heavy):
    resp = np.asarray(resp, dtype=np.float64)
    inside = heavy.astype(bool)
    outside = ~inside
    if (not inside.any()) or (not outside.any()):
        return float('nan')
    return float(resp[inside].mean() / (resp[outside].mean() + 1e-8))


class FeatureCatcher:
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


def load_model(cls, ckpt, device):
    model = cls(num_classes=20).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f'loaded {ckpt}')
    return model


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


def _draw_contour(ax, heavy_mask):
    if heavy_mask.any():
        ax.contour(heavy_mask.astype(float), levels=[0.5], colors='white', linewidths=0.8)


def save_panel(save_dir, radar_in, radar_gt, pred_conv, pred_kan,
               resp_conv, resp_kan, heavy_mask, stats):
    os.makedirs(save_dir, exist_ok=True)
    r_c = stats[f'R_conv_{MAIN_STAGE}']
    r_k = stats[f'R_kan_{MAIN_STAGE}']

    fig, axes = plt.subplots(2, 4, figsize=(17.4, 8.4))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.04,
                        wspace=0.16, hspace=0.22)

    top = [
        ('Input (last)', radar_rgb(radar_in), False),
        ('GT (max, heavy cloud)', radar_rgb(radar_gt), True),
        ('Conv Pred (max)', radar_rgb(pred_conv), False),
        ('KANConv Pred (max)', radar_rgb(pred_kan), False),
    ]
    for ax, (title, img, contour) in zip(axes[0], top):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis('off')
        if contour:
            _draw_contour(ax, heavy_mask)

    e_c = resp_conv[MAIN_STAGE]
    e_k = resp_kan[MAIN_STAGE]
    lo, hi = joint_limits(e_c, e_k)
    conv_ov = overlay_heat(radar_gt, e_c, heavy_mask, lo=lo, hi=hi)
    kan_ov = overlay_heat(radar_gt, e_k, heavy_mask, lo=lo, hi=hi)
    diff = scale01(e_k, lo, hi) - scale01(e_c, lo, hi)

    axes[1, 0].imshow(conv_ov)
    axes[1, 0].set_title(f'Conv energy ({MAIN_STAGE})  R={r_c:.2f}', fontsize=12, pad=8)
    axes[1, 1].imshow(kan_ov)
    axes[1, 1].set_title(f'KANConv energy ({MAIN_STAGE})  R={r_k:.2f}', fontsize=12, pad=8)
    sm = cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=lo, vmax=hi))
    sm.set_array([])
    cbar_e = fig.colorbar(sm, ax=axes[1, 1], fraction=0.046, pad=0.03)
    cbar_e.set_label('energy (shared)', fontsize=10)
    cbar_e.ax.tick_params(labelsize=9)

    im = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    axes[1, 2].set_title(f'KAN − Conv ({MAIN_STAGE})', fontsize=12, pad=8)
    cbar = fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    axes[1, 3].axis('off')
    note = (
        'Shared color scale\n'
        '2–98% of Conv + KAN\n'
        'Same red = same energy\n'
        '\n'
        'R = E(≥35 mm) / E(<35 mm)\n'
        f'  Conv         {r_c:.3f}\n'
        f'  KANConv      {r_k:.3f}\n'
        f'  ΔR           {r_k - r_c:+.3f}\n'
        '\n'
        'White contour  ≥ 35 mm\n'
        'Diff red       KAN stronger'
    )
    axes[1, 3].text(
        0.06, 0.50, note,
        fontsize=12.5, va='center', ha='left', linespacing=1.55,
        family='DejaVu Sans', transform=axes[1, 3].transAxes,
    )
    for ax in axes[1, :3]:
        ax.axis('off')
        _draw_contour(ax, heavy_mask)

    fig.savefig(os.path.join(save_dir, 'panel.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12.2, 12.0))
    for r, stage in enumerate(STAGES):
        lo_s, hi_s = joint_limits(resp_conv[stage], resp_kan[stage])
        axes[r, 0].imshow(overlay_heat(radar_gt, resp_conv[stage], heavy_mask, lo=lo_s, hi=hi_s))
        axes[r, 0].set_title(f'Conv {stage}  R={stats[f"R_conv_{stage}"]:.2f}', fontsize=11)
        axes[r, 1].imshow(overlay_heat(radar_gt, resp_kan[stage], heavy_mask, lo=lo_s, hi=hi_s))
        axes[r, 1].set_title(f'KANConv {stage}  R={stats[f"R_kan_{stage}"]:.2f}', fontsize=11)
        d = scale01(resp_kan[stage], lo_s, hi_s) - scale01(resp_conv[stage], lo_s, hi_s)
        im = axes[r, 2].imshow(d, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
        axes[r, 2].set_title(f'KAN − Conv {stage}', fontsize=11)
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)
        for ax in axes[r]:
            ax.axis('off')
            _draw_contour(ax, heavy_mask)
    fig.suptitle('Encoder-stage feature energy (Conv/KAN share scale per row)  |  ≥35 mm', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'stages.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    plt.imsave(os.path.join(save_dir, 'input_last.png'), radar_rgb(radar_in))
    plt.imsave(os.path.join(save_dir, 'gt_max.png'), radar_rgb(radar_gt))
    plt.imsave(os.path.join(save_dir, 'pred_conv.png'), radar_rgb(pred_conv))
    plt.imsave(os.path.join(save_dir, 'pred_kan.png'), radar_rgb(pred_kan))
    plt.imsave(os.path.join(save_dir, f'{MAIN_STAGE}_conv.png'), conv_ov)
    plt.imsave(os.path.join(save_dir, f'{MAIN_STAGE}_kan.png'), kan_ov)
    plt.imsave(os.path.join(save_dir, f'{MAIN_STAGE}_diff.png'),
               cm.RdBu_r(np.clip((diff + 0.8) / 1.6, 0, 1))[..., :3])


def save_summary(rows):
    csv_path = os.path.join(SAVE_ROOT, 'concentration.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r['heavy_ratio'] > 0.005 and np.isfinite(r[f'R_conv_{MAIN_STAGE}'])]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, stage in zip(axes, STAGES):
        if valid:
            x = np.arange(len(valid))
            ax.bar(x - 0.18, [r[f'R_conv_{stage}'] for r in valid], 0.36, label='Conv', color='#4C78A8')
            ax.bar(x + 0.18, [r[f'R_kan_{stage}'] for r in valid], 0.36, label='KANConv', color='#F58518')
            mean_c = np.mean([r[f'R_conv_{stage}'] for r in valid])
            mean_k = np.mean([r[f'R_kan_{stage}'] for r in valid])
            ax.axhline(mean_c, color='#4C78A8', ls='--', lw=1)
            ax.axhline(mean_k, color='#F58518', ls='--', lw=1)
            ax.set_title(f'{stage}  Conv={mean_c:.2f}  KAN={mean_k:.2f}')
            ax.set_xticks(x)
            ax.set_xticklabels([r['sample'].replace('_sample_0', '') for r in valid],
                               rotation=45, ha='right', fontsize=7)
        else:
            ax.set_title(stage)
        if stage == 'enc1':
            ax.set_ylabel('Heavy-cloud concentration R')
            ax.legend(fontsize=8)
    fig.suptitle(f'R = mean energy inside ≥{HEAVY_MM:.0f} mm / outside', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_ROOT, 'summary_R.png'), dpi=150)
    plt.close(fig)
    print(f'summary -> {csv_path}')


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  heavy>={HEAVY_MM}mm  batches={MAX_BATCHES}')

    model_conv = load_model(UNetBaseline, CKPT_CONV, device)
    model_kan = load_model(UNetKANConv, CKPT_KAN, device)
    catch_conv = FeatureCatcher(model_conv)
    catch_kan = FeatureCatcher(model_kan)

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

            catch_conv.clear()
            catch_kan.clear()
            with autocast():
                pred_c = model_conv(inp)
                pred_k = model_kan(inp)

            inp_np = to_numpy(inp)
            tgt_np = to_numpy(tgt)
            pred_c_np = to_numpy(pred_c)
            pred_k_np = to_numpy(pred_k)

            for i in range(inp_np.shape[0]):
                radar_in = inp_np[i, -1]
                radar_gt = tgt_np[i].max(axis=0)
                heavy_mask = (radar_gt * PIXEL_SCALE) >= HEAVY_MM

                resp_conv, resp_kan, stats = {}, {}, {}
                for stage in STAGES:
                    e_c = upsample_map(feature_energy(catch_conv.store[stage][i]), 256)
                    e_k = upsample_map(feature_energy(catch_kan.store[stage][i]), 256)
                    resp_conv[stage] = e_c
                    resp_kan[stage] = e_k
                    stats[f'R_conv_{stage}'] = concentration(e_c, heavy_mask)
                    stats[f'R_kan_{stage}'] = concentration(e_k, heavy_mask)

                out_dir = os.path.join(SAVE_ROOT, f'batch_{bidx}_sample_{i}')
                save_panel(
                    out_dir, radar_in, radar_gt, pred_c_np[i].max(axis=0),
                    pred_k_np[i].max(axis=0), resp_conv, resp_kan, heavy_mask, stats,
                )
                row = {
                    'sample': f'batch_{bidx}_sample_{i}',
                    'heavy_ratio': float(heavy_mask.mean()),
                    **stats,
                }
                rows.append(row)
                saved += 1
                print(
                    f"saved {out_dir}  heavy={row['heavy_ratio']:.3f}  "
                    f"R_conv={stats[f'R_conv_{MAIN_STAGE}']:.3f}  "
                    f"R_kan={stats[f'R_kan_{MAIN_STAGE}']:.3f}"
                )

    catch_conv.close()
    catch_kan.close()
    if rows:
        save_summary(rows)
    print(f'Done. {saved} samples -> {SAVE_ROOT}')


if __name__ == '__main__':
    main()
