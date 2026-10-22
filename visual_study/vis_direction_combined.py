# -*- coding: utf-8 -*-
"""三方向注意力 + 三方向特征响应（合并图，上海数据集）。

同一套 KM_UNetV3_SH 权重与样本：
  - 中排：DirectionAttention 的 H/W/C 注意力图（与 vis_direction_attention 一致）
  - 下排：height/width/channel 三支 DirectionViM 输出的通道能量（特征响应）

用法（在 visual_study 目录）:
    python vis_direction_combined.py
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

from KM_UNetV3_SH import KM_UNetV3, DirectionAttention
from Shanghai import Shanghai, CustomShanghai


CKPT = r'D:/zhoujian/KM-UNet/outputs/best_model/model_Best_shanghai_DAGEM_DySample_0.55.pth'
DATA = r'D:/zhoujian/KM-UNet/shanghai.h5'
SAVE_ROOT = r'D:/zhoujian/KM-UNet/outputs/visual_study/direction_combined'
STAGE = 'enc3'
HEAVY_MM = 35.0
PIXEL_SCALE = 90.0
MAX_BATCHES = 10
BATCH_SIZE = 1
NUM_WORKERS = 0
MODES = ('height', 'width', 'channel')

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
    return torch.sqrt(feat_chw.float().pow(2).mean(dim=0) + 1e-12).cpu().numpy()


def concentration(resp, heavy):
    resp = np.asarray(resp, dtype=np.float64)
    inside = heavy.astype(bool)
    outside = ~inside
    if (not inside.any()) or (not outside.any()):
        return float('nan')
    return float(resp[inside].mean() / (resp[outside].mean() + 1e-8))


def extract_attn_maps(module, x):
    B, C, H, W = x.shape
    q, k, v = module.qkv(x).chunk(3, dim=1)
    spatial = torch.sigmoid(q * k) * v
    spatial_map = spatial.abs().mean(dim=1)

    if module.mode == 'height':
        pooled = module.pool(x).squeeze(-1)
        profile = pooled.abs().mean(dim=1)
        dir_map = profile.unsqueeze(-1).expand(-1, H, W)
        ch_in = pooled.mean(dim=2)
    elif module.mode == 'width':
        pooled = module.pool(x).squeeze(2)
        profile = pooled.abs().mean(dim=1)
        dir_map = profile.unsqueeze(1).expand(-1, H, W)
        ch_in = pooled.mean(dim=2)
    else:
        pooled = module.pool(x).view(B, C)
        dir_map = x.abs().mean(dim=1)
        ch_in = pooled
        profile = None

    weight = module.fc(ch_in.view(B, C))
    weighted_map = (spatial * weight.view(B, C, 1, 1)).abs().mean(dim=1)
    return {
        'spatial': spatial_map,
        'dir': dir_map,
        'weighted': weighted_map,
        'profile': profile,
        'weight': weight,
    }


class CombinedCatcher:
    """同时抓注意力内部图 + 三支 DirectionViM 输出特征。"""

    def __init__(self, vim_block):
        self.store = {'attn': {}, 'feat': {}, 'fusion': None}
        self.handles = []
        for mode, branch in [
            ('height', vim_block.height_block),
            ('width', vim_block.width_block),
            ('channel', vim_block.channel_block),
        ]:
            self.handles.append(
                branch.attn.register_forward_hook(self._attn_hook(mode))
            )
            self.handles.append(
                branch.register_forward_hook(self._feat_hook(mode))
            )
        self.handles.append(
            vim_block.fusion_gate.register_forward_hook(self._gate_hook)
        )

    def _attn_hook(self, mode):
        def hook(mod, inp, out):
            maps = extract_attn_maps(mod, inp[0])
            self.store['attn'][mode] = {
                k: (v.detach() if torch.is_tensor(v) else v) for k, v in maps.items()
            }
        return hook

    def _feat_hook(self, mode):
        def hook(_mod, _inp, out):
            self.store['feat'][mode] = out.detach().float()
        return hook

    def _gate_hook(self, mod, inp, out):
        self.store['fusion'] = out.detach()

    def clear(self):
        self.store = {'attn': {}, 'feat': {}, 'fusion': None}

    def close(self):
        for h in self.handles:
            h.remove()


def get_vim_block(model, stage):
    if stage == 'enc2':
        return model.enc2[1]
    if stage == 'enc3':
        return model.enc3[1]
    if stage == 'dec2':
        return model.dec2[2]
    if stage == 'dec3':
        return model.dec3[2]
    raise ValueError(f'unknown stage {stage}')


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


def build_attn_combo(catcher, i, heavy_mask):
    maps = {}
    for mode in MODES:
        weighted = to_numpy(catcher.store['attn'][mode]['weighted'][i])
        dirm = to_numpy(catcher.store['attn'][mode]['dir'][i])
        combo = norm01(weighted) * (0.35 + 0.65 * norm01(dirm))
        combo = upsample_map(combo, 256)
        if heavy_mask.any():
            combo = combo * (0.25 + 0.75 * heavy_mask.astype(np.float32))
        maps[mode] = combo
    return maps


def build_feat_energy(catcher, i):
    energies = {}
    for mode in MODES:
        energies[mode] = upsample_map(feature_energy(catcher.store['feat'][mode][i]), 256)
    return energies


def save_panel(save_dir, radar_in, radar_gt, radar_pred,
               attn_maps, feat_maps, heavy_mask, fusion, stats):
    os.makedirs(save_dir, exist_ok=True)

    # 第 4 列贯穿三行放色条，保证前三列与上两行严格对齐
    fig = plt.figure(figsize=(14.0, 12.4))
    gs = fig.add_gridspec(
        3, 4, width_ratios=[1.0, 1.0, 1.0, 0.06],
        left=0.02, right=0.98, top=0.95, bottom=0.05,
        wspace=0.06, hspace=0.16,
    )
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax = fig.add_subplot(gs[:, 3])

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

    attn_titles = {
        'height': 'Height attention',
        'width': 'Width attention',
        'channel': 'Channel attention',
    }
    # 注意力三图各自归一化（与旧脚本一致，强调空间结构）
    for ax, mode in zip(axes[1], MODES):
        blend = overlay_heat(radar_gt, attn_maps[mode], heavy_mask)
        ax.imshow(blend)
        ax.set_title(attn_titles[mode], fontsize=12, pad=8)
        ax.axis('off')
        _draw_contour(ax, heavy_mask)
        plt.imsave(os.path.join(save_dir, f'attn_{mode}.png'), blend)

    # 特征响应三图共用色标，便于比谁更强
    lo, hi = joint_limits(*[feat_maps[m] for m in MODES])
    feat_titles = {
        'height': 'Height response',
        'width': 'Width response',
        'channel': 'Channel response',
    }
    for ax, mode in zip(axes[2], MODES):
        blend = overlay_heat(radar_gt, feat_maps[mode], heavy_mask, lo=lo, hi=hi)
        r = stats[f'R_{mode}']
        ax.imshow(blend)
        ax.set_title(f'{feat_titles[mode]}  R={r:.2f}', fontsize=12, pad=8)
        ax.axis('off')
        _draw_contour(ax, heavy_mask)
        plt.imsave(os.path.join(save_dir, f'resp_{mode}.png'), blend)

    sm = cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=lo, vmax=hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('feature energy (shared)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    note = f'white contour ≥{HEAVY_MM:.0f} mm'
    if fusion is not None:
        fw = fusion.reshape(-1)[:3]
        note = (
            f'Fusion gate  H={fw[0]:.2f}  W={fw[1]:.2f}  C={fw[2]:.2f}   |   '
            + note
            + '   |   mid: attention   bottom: branch feature energy'
        )
    fig.text(0.5, 0.012, note, ha='center', va='bottom', fontsize=11)

    fig.savefig(os.path.join(save_dir, 'panel.png'), dpi=160)
    plt.close(fig)

    plt.imsave(os.path.join(save_dir, 'input_last.png'), radar_rgb(radar_in))
    plt.imsave(os.path.join(save_dir, 'gt_max.png'), radar_rgb(radar_gt))
    plt.imsave(os.path.join(save_dir, 'pred_max.png'), radar_rgb(radar_pred))

    if fusion is not None:
        with open(os.path.join(save_dir, 'fusion_gate.txt'), 'w', encoding='utf-8') as f:
            f.write(f'height={fusion[0]:.6f}\nwidth={fusion[1]:.6f}\nchannel={fusion[2]:.6f}\n')


def save_summary(rows):
    csv_path = os.path.join(SAVE_ROOT, 'concentration.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r['heavy_ratio'] > 0.005 and np.isfinite(r['R_height'])]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    if valid:
        x = np.arange(len(valid))
        w = 0.24
        colors_b = ['#4C78A8', '#F58518', '#54A24B']
        for i, mode in enumerate(MODES):
            vals = [r[f'R_{mode}'] for r in valid]
            ax.bar(x + (i - 1) * w, vals, w, label=mode, color=colors_b[i])
            ax.axhline(np.mean(vals), color=colors_b[i], ls='--', lw=1)
        ax.axhline(1.0, color='0.5', ls=':', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([r['sample'].replace('_sample_0', '') for r in valid],
                           rotation=45, ha='right', fontsize=8)
        means = {m: np.mean([r[f'R_{m}'] for r in valid]) for m in MODES}
        ax.set_ylabel('Heavy-cloud concentration R')
        ax.set_title(
            f"Branch feature energy  R>1 stronger in ≥{HEAVY_MM:.0f} mm    "
            + '  '.join(f'{m[0].upper()}={means[m]:.2f}' for m in MODES)
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_ROOT, 'summary_R.png'), dpi=150)
    plt.close(fig)
    print(f'summary -> {csv_path}')


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  stage={STAGE}  heavy>={HEAVY_MM}mm')

    model = KM_UNetV3(num_classes=20).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f'loaded {CKPT}')

    vim = get_vim_block(model, STAGE)
    assert isinstance(vim.height_block.attn, DirectionAttention)
    catcher = CombinedCatcher(vim)

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

            fusion = catcher.store.get('fusion')
            inp_np = to_numpy(inp)
            tgt_np = to_numpy(tgt)
            pred_np = to_numpy(pred)

            for i in range(inp_np.shape[0]):
                radar_in = inp_np[i, -1]
                radar_gt = tgt_np[i].max(axis=0)
                radar_pred = pred_np[i].max(axis=0)
                heavy_mask = (radar_gt * PIXEL_SCALE) >= HEAVY_MM

                attn_maps = build_attn_combo(catcher, i, heavy_mask)
                feat_maps = build_feat_energy(catcher, i)
                stats = {f'R_{m}': concentration(feat_maps[m], heavy_mask) for m in MODES}

                fw = None
                if fusion is not None:
                    fw = to_numpy(fusion[i]).reshape(-1)

                out_dir = os.path.join(SAVE_ROOT, f'batch_{bidx}_sample_{i}')
                save_panel(
                    out_dir, radar_in, radar_gt, radar_pred,
                    attn_maps, feat_maps, heavy_mask, fw, stats,
                )
                row = {
                    'sample': f'batch_{bidx}_sample_{i}',
                    'heavy_ratio': float(heavy_mask.mean()),
                    **stats,
                }
                if fw is not None:
                    row.update({'gate_H': float(fw[0]), 'gate_W': float(fw[1]), 'gate_C': float(fw[2])})
                rows.append(row)
                saved += 1
                print(
                    f"saved {out_dir}  heavy={row['heavy_ratio']:.3f}  "
                    f"R_H={stats['R_height']:.3f}  R_W={stats['R_width']:.3f}  "
                    f"R_C={stats['R_channel']:.3f}"
                )

    catcher.close()
    if rows:
        save_summary(rows)
    print(f'Done. {saved} samples -> {SAVE_ROOT}')


if __name__ == '__main__':
    main()
