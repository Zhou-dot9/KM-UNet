# -*- coding: utf-8 -*-
"""预测误差图 + ≥35 mm / ≥5 mm 边界差异图。

与 vis_heavy_cloud_compare.py 同一套上海厚云个例、同一批四模型权重。

上排：有符号误差 (Pred−GT)，四模型共用 ±vmax 色标。
下排：边界叠图（白=GT，红=预报，黄=重合）以及命中/空报/漏报分区。

用法（在 visual_study 目录）:
    python vis_error_boundary.py
"""
import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from vis_heavy_cloud_compare import (
    HEAVY_MM, PIXEL_SCALE, TOP_K, MODEL_KEYS, MODEL_NAMES, SAVE_ROOT,
    radar_rgb, build_test_dataset, rank_heavy_cases, load_models, predict_all,
    csi_score, rmse_mm, _draw_contour,
)


ERR_VMIN, ERR_VMAX = -20.0, 20.0
ECHO_MM = 5.0
BOUNDARY_WIDTH = 2
SAVE_DIR = os.path.join(os.path.dirname(SAVE_ROOT), 'error_boundary')


def region_mask(frame01, thr_mm):
    mm = np.asarray(frame01) * PIXEL_SCALE
    # ≥0 mm 表示回波外轮廓：像素值 > 0（若含 0 会铺满整幅，没有边界）
    if thr_mm <= 0:
        return mm > 0
    return mm >= thr_mm


def heavy_mask(frame01):
    return region_mask(frame01, HEAVY_MM)


def signed_error_mm(pred01, gt01):
    return (np.asarray(pred01) - np.asarray(gt01)) * PIXEL_SCALE


def mask_boundary(mask, width=BOUNDARY_WIDTH):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    eroded = mask
    for _ in range(width):
        nxt = binary_erosion(eroded)
        if not nxt.any():
            break
        eroded = nxt
    return mask & ~eroded


def overlay_boundary(gt01, pred01, thr_mm=HEAVY_MM):
    """白=GT 边界，红=预报边界，黄=重合。底图为变暗的 GT 雷达。"""
    base = radar_rgb(gt01) * 0.45
    gb = mask_boundary(region_mask(gt01, thr_mm))
    pb = mask_boundary(region_mask(pred01, thr_mm))
    out = base.copy()
    out[gb] = (1.0, 1.0, 1.0)
    out[pb] = (1.0, 0.12, 0.12)
    out[gb & pb] = (1.0, 0.92, 0.15)
    return out


def structure_map(gt01, pred01, thr_mm=HEAVY_MM):
    """绿=命中，红=空报，蓝=漏报。"""
    g = region_mask(gt01, thr_mm)
    p = region_mask(pred01, thr_mm)
    rgb = np.zeros(g.shape + (3,), dtype=np.float32)
    rgb[g & p] = (0.15, 0.78, 0.22)
    rgb[p & ~g] = (0.92, 0.18, 0.16)
    rgb[~p & g] = (0.18, 0.42, 0.95)
    return rgb


def boundary_scores(gt01, pred01, thr_mm=HEAVY_MM):
    g = region_mask(gt01, thr_mm)
    p = region_mask(pred01, thr_mm)
    gb = mask_boundary(g)
    pb = mask_boundary(p)
    hit = np.logical_and(p, g).sum()
    fa = np.logical_and(p, ~g).sum()
    miss = np.logical_and(~p, g).sum()
    den = hit + fa + miss
    csi = float(hit / den) if den else float('nan')
    if (not gb.any()) or (not pb.any()):
        return {'csi': csi, 'msd': float('nan'), 'hd': float('nan')}
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    d_pg = dt_g[pb]
    d_gp = dt_p[gb]
    msd = 0.5 * (float(d_pg.mean()) + float(d_gp.mean()))
    hd = float(max(d_pg.max(), d_gp.max()))
    return {'csi': csi, 'msd': msd, 'hd': hd}


def save_case_figures(out_dir, gt_max, pred_max, stats):
    os.makedirs(out_dir, exist_ok=True)
    heavy = heavy_mask(gt_max)
    err = {k: signed_error_mm(pred_max[k], gt_max) for k in MODEL_KEYS}

    fig = plt.figure(figsize=(17.8, 14.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.07],
                          left=0.03, right=0.97, top=0.94, bottom=0.06,
                          wspace=0.06, hspace=0.22)
    axes0 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes1 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axes2 = [fig.add_subplot(gs[2, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])

    for ax, k in zip(axes0, MODEL_KEYS):
        im = ax.imshow(err[k], cmap='RdBu_r', vmin=ERR_VMIN, vmax=ERR_VMAX)
        ax.set_title(
            f"{MODEL_NAMES[k]}  Pred−GT (full field)\n"
            f"RMSE={stats[k]['rmse_all']:.1f} mm  RMSE_h={stats[k]['rmse_h']:.1f} mm",
            fontsize=10,
        )
        ax.axis('off')
        _draw_contour(ax, heavy)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('error (mm)', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    for ax, k in zip(axes1, MODEL_KEYS):
        ax.imshow(overlay_boundary(gt_max, pred_max[k], HEAVY_MM))
        s = stats[k]
        ax.set_title(
            f"{MODEL_NAMES[k]}  ≥{HEAVY_MM:.0f} mm boundary\n"
            f"MSD={s['msd35']:.1f} px  CSI={s['csi35']:.3f}",
            fontsize=10,
        )
        ax.axis('off')

    for ax, k in zip(axes2, MODEL_KEYS):
        ax.imshow(overlay_boundary(gt_max, pred_max[k], ECHO_MM))
        s = stats[k]
        ax.set_title(
            f"{MODEL_NAMES[k]}  ≥{ECHO_MM:.0f} mm boundary\n"
            f"MSD={s['msd5']:.1f} px  CSI={s['csi5']:.3f}",
            fontsize=10,
        )
        ax.axis('off')

    fig.text(
        0.5, 0.012,
        f'Row1: signed error on the full field, shared ±{ERR_VMAX:.0f} mm '
        f'(red=overestimate, blue=underestimate; white contour ≥{HEAVY_MM:.0f} mm).   '
        'Row2–3: white=GT edge, red=pred edge, yellow=overlap.',
        ha='center', va='bottom', fontsize=10,
    )
    fig.savefig(os.path.join(out_dir, 'error_boundary.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.1))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.10, wspace=0.05)
    for ax, k in zip(axes, MODEL_KEYS):
        ax.imshow(structure_map(gt_max, pred_max[k], HEAVY_MM))
        ax.set_title(f"{MODEL_NAMES[k]}  hit / FA / miss  (≥{HEAVY_MM:.0f} mm)", fontsize=11)
        ax.axis('off')
        _draw_contour(ax, heavy)
    legend = [
        Patch(facecolor=(0.15, 0.78, 0.22), label='Hit'),
        Patch(facecolor=(0.92, 0.18, 0.16), label='False alarm'),
        Patch(facecolor=(0.18, 0.42, 0.95), label='Miss'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=10, frameon=False)
    fig.savefig(os.path.join(out_dir, 'structure_diff.png'), dpi=150)
    plt.close(fig)

    for k in MODEL_KEYS:
        normed = np.clip((err[k] - ERR_VMIN) / (ERR_VMAX - ERR_VMIN), 0, 1)
        plt.imsave(os.path.join(out_dir, f'err_signed_{k}.png'),
                   plt.cm.RdBu_r(normed)[..., :3])
        plt.imsave(os.path.join(out_dir, f'boundary_{k}.png'),
                   overlay_boundary(gt_max, pred_max[k], HEAVY_MM))
        plt.imsave(os.path.join(out_dir, f'boundary5_{k}.png'),
                   overlay_boundary(gt_max, pred_max[k], ECHO_MM))
        plt.imsave(os.path.join(out_dir, f'structure_{k}.png'),
                   structure_map(gt_max, pred_max[k], HEAVY_MM))


def save_overview(rows):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16.5, 2.7 * n))
    if n == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.02,
                        wspace=0.04, hspace=0.18)
    for r, row in enumerate(rows):
        for c, k in enumerate(MODEL_KEYS):
            ax = axes[r, c]
            ax.imshow(plt.imread(os.path.join(SAVE_DIR, row['tag'], f'boundary_{k}.png')))
            ax.axis('off')
            if r == 0:
                ax.set_title(MODEL_NAMES[k], fontsize=11)
        axes[r, 0].set_ylabel(row['tag'], fontsize=9, rotation=0,
                              ha='right', va='center', labelpad=18)
    fig.savefig(os.path.join(SAVE_DIR, 'overview_boundary.png'), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  save -> {SAVE_DIR}')

    dataset, _ = build_test_dataset()
    print(f'test size={len(dataset)}, ranking heavy-cloud cases...')
    picked = rank_heavy_cases(dataset)
    models = load_models(device)

    rows = []
    for rank, (heavy_ratio, local_idx) in enumerate(picked):
        sample = dataset[local_idx]
        frames = sample.squeeze(1).numpy() if sample.ndim == 4 else sample.numpy()
        inp = frames[:5][None].astype(np.float32)
        gt = frames[5:].astype(np.float32)
        gt_max = gt.max(axis=0)
        heavy = heavy_mask(gt_max)

        preds = predict_all(models, inp, device)
        pred_max = {k: preds[k].max(axis=0) for k in MODEL_KEYS}

        stats = {}
        for k in MODEL_KEYS:
            s35 = boundary_scores(gt_max, pred_max[k], HEAVY_MM)
            s5 = boundary_scores(gt_max, pred_max[k], ECHO_MM)
            stats[k] = {
                'csi35': s35['csi'], 'msd35': s35['msd'], 'hd35': s35['hd'],
                'csi5': s5['csi'], 'msd5': s5['msd'], 'hd5': s5['hd'],
                'rmse_h': rmse_mm(pred_max[k], gt_max, heavy),
                'rmse_all': rmse_mm(pred_max[k], gt_max),
            }

        tag = f'case_{rank:02d}'
        save_case_figures(os.path.join(SAVE_DIR, tag), gt_max, pred_max, stats)
        row = {'tag': tag, 'local_idx': int(local_idx), 'heavy_ratio': heavy_ratio}
        for k in MODEL_KEYS:
            row[f'csi35_{k}'] = stats[k]['csi35']
            row[f'msd35_{k}'] = stats[k]['msd35']
            row[f'csi5_{k}'] = stats[k]['csi5']
            row[f'msd5_{k}'] = stats[k]['msd5']
            row[f'rmse_h_{k}'] = stats[k]['rmse_h']
            row[f'rmse_all_{k}'] = stats[k]['rmse_all']
        rows.append(row)
        msd_txt = ' '.join(f"{k}={stats[k]['msd35']:.1f}/{stats[k]['msd5']:.1f}" for k in MODEL_KEYS)
        print(f'saved {tag}  heavy={heavy_ratio:.3f}  MSD35/5 {msd_txt}')
        torch.cuda.empty_cache()

    csv_path = os.path.join(SAVE_DIR, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_overview(rows)
    print(f'Done. {len(rows)} cases -> {SAVE_DIR}')


if __name__ == '__main__':
    main()
