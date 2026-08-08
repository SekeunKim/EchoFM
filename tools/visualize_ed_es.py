"""
ED/ES-anchored periodicity visualization.

Detects end-diastole (ED, largest LV cavity) and end-systole (ES, smallest)
frames from a cavity-area proxy (fraction of dark pixels inside the scan
cone), shows them LARGE with labels, and prints the one number that decides
whether periodicity was learned:

    embed sim(ED, ED')  >  embed sim(ED, ES)   ?
    (same phase, next cycle)   (opposite phase)

Run (inside the SIF, on a GPU node):
  python tools/visualize_ed_es.py --ckpt logs/<run>/checkpoint-000NN.pth \
      --data_path <clips> --out logs/<run>/ed_es_epNN
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from EchoFM import models_mae
from data.dataset import EchoDataset_from_cache_npy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--t_patch_size", type=int, default=4)
    ap.add_argument("--pred_t_dim", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def cavity_curve(imgs):
    """Fraction of dark (cavity) pixels in the VENTRICULAR part of the cone.

    Only the upper 55% of rows (apex side in standard display) is measured:
    the atria fill during ventricular systole, so counting the whole sector
    mixes anti-phase chambers and can invert the ED/ES labels.
    """
    gray = imgs[0].mean(dim=0)  # [T, H, W]
    H = gray.shape[-2]
    cone = gray.max(dim=0).values > 0.05
    cone[int(0.55 * H):, :] = False
    dark = (gray < 0.15) & cone
    area = dark.flatten(1).sum(-1).float() / max(int(cone.sum()), 1)
    a = area.cpu().numpy()
    kernel = np.ones(3) / 3.0
    return np.convolve(a, kernel, mode="same")


def cycle_length(imgs, lo=8, hi=28):
    """Cardiac cycle length in frames from the frame-level pixel-similarity
    lag curve: the reprise peak (first local max of similarity at lag >= lo)."""
    import torch.nn.functional as _F

    x = imgs[0].mean(dim=0)  # [Tf, H, W]
    f = _F.adaptive_avg_pool2d(x.unsqueeze(1), (32, 32)).flatten(1)  # [Tf, 1024]
    f = f - f.mean(dim=-1, keepdim=True)
    f = _F.normalize(f, dim=-1)
    S = (f @ f.T).cpu().numpy()
    Tf = S.shape[0]
    lags = range(1, Tf)
    curve = np.array([np.mean(np.diag(S, k)) for k in lags])
    hi = min(hi, Tf - 2)
    if hi <= lo:
        return None
    seg = curve[lo - 1 : hi]
    L = lo + int(np.argmax(seg))
    # require a genuine reprise: similarity must dip before rising again
    if curve[L - 1] <= curve[: lo - 1].min() + 1e-3:
        return None
    return L


def find_ed_es(area, L=None):
    """ED = cavity max; ED' = ED + cycle length (when known); ES = min between.

    With a cycle-length estimate the second ED is *implied by periodicity*
    instead of independently peak-picked, which is far more robust on noisy
    cavity curves.
    """
    T = len(area)
    if L is not None and 4 < L < T:
        ed1 = int(np.argmax(area[: T - L]))
        ed2 = ed1 + L
        es = ed1 + 1 + int(np.argmin(area[ed1 + 1 : ed2]))
        return [ed1, ed2], es
    # fallback: independent local maxima (previous behaviour)
    peaks = [i for i in range(1, T - 1)
             if area[i] >= area[i - 1] and area[i] >= area[i + 1]]
    peaks = sorted(peaks, key=lambda i: -area[i])
    eds = []
    for p in peaks:
        if all(abs(p - q) >= 8 for q in eds):
            eds.append(p)
        if len(eds) == 2:
            break
    eds = sorted(eds)
    if len(eds) == 2:
        lo_, hi_ = eds
        es = lo_ + int(np.argmin(area[lo_:hi_ + 1]))
    else:
        es = int(np.argmin(area))
    return eds, es


@torch.no_grad()
def embed_clip(model, imgs):
    latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
    cls_stack = torch.stack(model.forward_prj(latent), dim=1)
    z = F.normalize(cls_stack, p=2, dim=-1)
    embed_sim = torch.bmm(z, z.transpose(1, 2))[0].cpu().numpy()
    prior_sim = model.pixel_similarity(imgs)[0].cpu().numpy()
    return embed_sim, prior_sim


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = models_mae.mae_vit_large_patch16(
        num_frames=args.num_frames, t_patch_size=args.t_patch_size,
        pred_t_dim=args.pred_t_dim, sep_pos_embed=True, cls_embed=True,
        norm_pix_loss=True,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"loaded {args.ckpt} (epoch {epoch})")

    ds = EchoDataset_from_cache_npy(
        args.data_path, num_frames=args.num_frames, image_size=224, frame_stride=1
    )
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.num_clips, len(ds)), replace=False)

    grp = args.t_patch_size
    stats, plotted = [], 0
    for clip_i in picks:
        imgs = ds[int(clip_i)].unsqueeze(0).to(device)
        area = cavity_curve(imgs)
        L = cycle_length(imgs)
        eds, es = find_ed_es(area, L)
        if len(eds) < 2:
            continue  # need two cycles for the ED-ED' comparison
        ed1, ed2 = eds
        embed_sim, prior_sim = embed_clip(model, imgs)

        t = lambda f: min(f // grp, embed_sim.shape[0] - 1)
        if len({t(ed1), t(ed2), t(es)}) < 3:
            continue  # phases collapse onto the same temporal token
        e_same = float(embed_sim[t(ed1), t(ed2)])
        e_opp = float(0.5 * (embed_sim[t(ed1), t(es)] + embed_sim[t(ed2), t(es)]))
        p_same = float(prior_sim[t(ed1), t(ed2)])
        p_opp = float(0.5 * (prior_sim[t(ed1), t(es)] + prior_sim[t(ed2), t(es)]))
        ok = e_same > e_opp
        name = os.path.basename(ds.paths[int(clip_i)])
        stats.append({"clip": name, "embed_same": e_same, "embed_opp": e_opp,
                      "pixel_same": p_same, "pixel_opp": p_opp, "ok": bool(ok)})

        if plotted < 5:
            frames = imgs[0].permute(1, 2, 3, 0).cpu().numpy().clip(0, 1)
            fig = plt.figure(figsize=(13, 9.5))
            gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 2.2, 0.9], hspace=0.35)

            ax = fig.add_subplot(gs[0, :])
            ax.plot(area, "-", color="0.4", lw=2)
            ax.plot([ed1, ed2], [area[ed1], area[ed2]], "^", color="crimson",
                    ms=14, label="ED (end-diastole, max cavity)")
            ax.plot([es], [area[es]], "v", color="royalblue", ms=14,
                    label="ES (end-systole, min cavity)")
            ax.set_title("LV cavity size proxy over 32 frames", fontsize=13)
            ax.legend(fontsize=11, loc="best")
            ax.tick_params(labelsize=10)

            for col, (f, label, color) in enumerate(
                [(ed1, f"ED  (frame {ed1})", "crimson"),
                 (es, f"ES  (frame {es})", "royalblue"),
                 (ed2, f"ED' next cycle (frame {ed2})", "crimson")]
            ):
                ax = fig.add_subplot(gs[1, col])
                ax.imshow(frames[f])
                ax.set_title(label, fontsize=14, color=color, fontweight="bold")
                for s in ax.spines.values():
                    s.set_edgecolor(color); s.set_linewidth(4)
                ax.set_xticks([]); ax.set_yticks([])

            ax = fig.add_subplot(gs[2, :])
            ax.axis("off")
            verdict = "O  same phase closer" if ok else "X  NOT learned"
            ax.text(0.5, 0.7,
                    f"embedding:  sim(ED, ED') = {e_same:.3f}   vs   "
                    f"sim(ED, ES) = {e_opp:.3f}    →  {verdict}",
                    ha="center", fontsize=17,
                    color="green" if ok else "red", fontweight="bold")
            ax.text(0.5, 0.2,
                    f"pixel prior:   sim(ED, ED') = {p_same:.3f}   vs   "
                    f"sim(ED, ES) = {p_opp:.3f}",
                    ha="center", fontsize=13, color="0.4")
            fig.suptitle(f"{name}   epoch {epoch}", fontsize=12)
            fig.savefig(os.path.join(args.out, f"ed_es_{plotted:02d}.png"),
                        dpi=110, bbox_inches="tight")
            plt.close(fig)
            plotted += 1

    n_ok = sum(s["ok"] for s in stats)
    summary = {
        "ckpt": args.ckpt, "epoch": int(epoch) if epoch != "?" else -1,
        "clips_with_2_cycles": len(stats), "same_phase_closer": n_ok,
        "frac_ok": n_ok / len(stats) if stats else None,
        "embed_same_mean": float(np.mean([s["embed_same"] for s in stats])) if stats else None,
        "embed_opp_mean": float(np.mean([s["embed_opp"] for s in stats])) if stats else None,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "clips": stats}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {plotted} ED/ES figures to {args.out}")


if __name__ == "__main__":
    main()
