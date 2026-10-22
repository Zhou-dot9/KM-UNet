# -*- coding: utf-8 -*-
"""三方向敏感注意力可视化（上海数据集，关注厚云）。

从 KM_UNetV3_SH 的 EnhancedViMBlock.DirectionAttention 提取
Height / Width / Channel 三支注意力，叠在雷达回波上。
默认用 enc3（深层、更对应厚云核），输出前 10 个 batch。

用法（在 visual_study 目录）:
    python vis_direction_attention.py
"""
import os
import sys
import math

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


# ---------------- 路径与超参 ----------------
CKPT = r'D:/zhoujian/KM-UNet/outputs/best_model/model_Best_shanghai_DAGEM_DySample_0.55.pth'
DATA = r'D:/zhoujian/KM-UNet/shanghai.h5'
SAVE_ROOT = r'D:/zhoujian/KM-UNet/outputs/visual_study/direction_attention'
STAGE = 'enc3'          # enc2=128x128, enc3=64x64（更贴近厚云）
HEAVY_MM = 35.0         # 厚云/强回波阈值（与论文 35 mm 档一致）
PIXEL_SCALE = 90.0
MAX_BATCHES = 10
BATCH_SIZE = 2
NUM_WORKERS = 0

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


def norm01(x, p=98.0):
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 2.0))
    hi = float(np.percentile(x, p))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def upsample_map(arr, size=256):
    t = torch.from_numpy(arr).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    else:
        t = t.unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=True)
    return t.squeeze().numpy()


def radar_rgb(frame_01):
    """frame_01: [H,W] in [0,1] -> RGB uint8-like float."""
    mm = np.clip(frame_01 * PIXEL_SCALE, 0, PIXEL_SCALE)
    rgba = gray2color(mm)
    return rgba[..., :3]


def overlay_attn(radar_01, attn_map, heavy_mask, alpha_bg=0.35, alpha_heavy=0.70):
    """注意力叠在雷达图上；厚云区透明度更高。"""
    base = radar_rgb(radar_01)
    attn = norm01(attn_map)
    heat = cm.jet(attn)[..., :3]
    a = alpha_bg + (alpha_heavy - alpha_bg) * heavy_mask.astype(np.float32)
    a = a[..., None]
    blend = (1.0 - a) * base + a * heat
    return np.clip(blend, 0.0, 1.0)


def extract_attn_maps(module, x):
    """按 DirectionAttention.forward 拆出空间图和方向图。x: [B,C,H,W]."""
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


class AttentionCatcher:
    def __init__(self, vim_block):
        self.store = {}
        self.handles = []
        for mode, branch in [
            ('height', vim_block.height_block),
            ('width', vim_block.width_block),
            ('channel', vim_block.channel_block),
        ]:
            self.handles.append(
                branch.attn.register_forward_hook(self._make_hook(mode))
            )
        self.handles.append(
            vim_block.fusion_gate.register_forward_hook(self._gate_hook)
        )

    def _make_hook(self, mode):
        def hook(mod, inp, out):
            maps = extract_attn_maps(mod, inp[0])
            self.store[mode] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in maps.items()}
        return hook

    def _gate_hook(self, mod, inp, out):
        self.store['fusion'] = out.detach()

    def clear(self):
        self.store = {}

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


def save_panel(save_dir, radar_in, radar_gt, radar_pred, maps, fusion, heavy_mask):
    os.makedirs(save_dir, exist_ok=True)
    titles = [
        ('Input (last)', radar_rgb(radar_in)),
        ('GT (max, heavy cloud)', radar_rgb(radar_gt)),
        ('Pred (max)', radar_rgb(radar_pred)),
    ]
    attn_titles = ['Height attention', 'Width attention', 'Channel attention']
    modes = ['height', 'width', 'channel']

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    for ax, (title, img) in zip(axes[0], titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        if title.startswith('GT'):
            ax.contour(heavy_mask.astype(float), levels=[0.5], colors='white', linewidths=0.8)

    for ax, mode, title in zip(axes[1], modes, attn_titles):
        blend = overlay_attn(radar_gt, maps[mode], heavy_mask)
        ax.imshow(blend)
        ax.contour(heavy_mask.astype(float), levels=[0.5], colors='white', linewidths=0.8)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    if fusion is not None:
        fw = fusion.reshape(-1)[:3]
        fig.suptitle(
            f'Fusion gate  H={fw[0]:.2f}  W={fw[1]:.2f}  C={fw[2]:.2f}   '
            f'white contour: ≥{HEAVY_MM:.0f} mm',
            fontsize=11, y=0.02
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(save_dir, 'panel.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    plt.imsave(os.path.join(save_dir, 'input_last.png'), radar_rgb(radar_in))
    plt.imsave(os.path.join(save_dir, 'gt_max.png'), radar_rgb(radar_gt))
    plt.imsave(os.path.join(save_dir, 'pred_max.png'), radar_rgb(radar_pred))
    for mode in modes:
        blend = overlay_attn(radar_gt, maps[mode], heavy_mask)
        plt.imsave(os.path.join(save_dir, f'{mode}.png'), blend)

    if fusion is not None:
        with open(os.path.join(save_dir, 'fusion_gate.txt'), 'w', encoding='utf-8') as f:
            f.write(f'height={fusion[0]:.6f}\nwidth={fusion[1]:.6f}\nchannel={fusion[2]:.6f}\n')


def build_loader():
    full_dataset = Shanghai(data_path=DATA, img_size=256, type='train')
    all_indices = np.arange(len(full_dataset))
    l = len(full_dataset)
    train_end = math.floor(l * 0.6)
    val_end = math.floor(l * 0.8)
    test_indices = all_indices[val_end:l]
    test_dataset = CustomShanghai(full_dataset, test_indices)
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


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
    catcher = AttentionCatcher(vim)

    loader = build_loader()
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

            bs = inp_np.shape[0]
            for i in range(bs):
                radar_in = inp_np[i, -1]
                radar_gt = tgt_np[i].max(axis=0)
                radar_pred = pred_np[i].max(axis=0)
                heavy_mask = (radar_gt * PIXEL_SCALE) >= HEAVY_MM

                maps = {}
                for mode in ('height', 'width', 'channel'):
                    weighted = to_numpy(catcher.store[mode]['weighted'][i])
                    dirm = to_numpy(catcher.store[mode]['dir'][i])
                    # 方向结构 × 空间注意力，并在厚云区加强
                    combo = norm01(weighted) * (0.35 + 0.65 * norm01(dirm))
                    combo = upsample_map(combo, 256)
                    if heavy_mask.any():
                        combo = combo * (0.25 + 0.75 * heavy_mask.astype(np.float32))
                    maps[mode] = combo

                fw = None
                if fusion is not None:
                    fw = to_numpy(fusion[i]).reshape(-1)

                out_dir = os.path.join(SAVE_ROOT, f'batch_{bidx}_sample_{i}')
                save_panel(out_dir, radar_in, radar_gt, radar_pred, maps, fw, heavy_mask)
                saved += 1
                print(f'saved {out_dir}  heavy_ratio={heavy_mask.mean():.3f}')

    catcher.close()
    print(f'Done. {saved} samples -> {SAVE_ROOT}')


if __name__ == '__main__':
    main()
