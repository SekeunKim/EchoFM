"""
Polished periodicity figure for the README.

Scans clips, keeps clean B-mode ones (no Doppler color) where the ED/ES
cross-cycle test passes with a large margin, and renders a publication-style
figure: cavity curve + ED/ES/ED' frames + embedding-similarity comparison.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from EchoFM import models_mae
from data.dataset import EchoDataset_from_cache_npy
from tools.visualize_ed_es import cavity_curve, cycle_length, find_ed_es


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_scan", type=int, default=120)
    ap.add_argument("--num_render", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    return ap.parse_args()


@torch.no_grad()
def embed_sim_map(model, imgs):
    latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
    z = F.normalize(torch.stack(model.forward_prj(latent), dim=1), dim=-1)
    return torch.bmm(z, z.transpose(1, 2))[0].cpu().numpy()


def is_bmode(imgs):
    """True when the clip is essentially grayscale (no Doppler color)."""
    x = imgs[0]  # [3, T, H, W]
    return float((x.max(dim=0).values - x.min(dim=0).values).mean()) < 0.01


def render(frames, area, ed1, es, ed2, e_same, e_opp, path):
    fig = plt.figure(figsize=(11.5, 8.2), facecolor="white")
    gs = fig.add_gridspec(
        3, 3, height_ratios=[0.95, 2.6, 0.75], hspace=0.42, wspace=0.06
    )

    ax = fig.add_subplot(gs[0, :])
    t = np.arange(len(area))
    ax.plot(t, area, color="#455a64", lw=2.2, solid_capstyle="round")
    ax.fill_between(t, area, area.min() - 0.01, color="#455a64", alpha=0.08)
    for f, c, m in [(ed1, "#c62828", "^"), (es, "#1565c0", "v"), (ed2, "#c62828", "^")]:
        ax.plot([f], [area[f]], m, color=c, ms=13, zorder=5)
    ax.annotate("ED", (ed1, area[ed1]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, color="#c62828", fontweight="bold")
    ax.annotate("ES", (es, area[es]), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=11, color="#1565c0", fontweight="bold")
    ax.annotate("ED′", (ed2, area[ed2]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, color="#c62828", fontweight="bold")
    ax.set_xlim(-0.5, len(area) - 0.5)
    ax.set_title("ventricular cavity size across one clip (32 frames)",
                 fontsize=11, color="#37474f")
    ax.set_xlabel("frame", fontsize=9, color="#607d8b", labelpad=1)
    ax.tick_params(labelsize=8, colors="#607d8b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b0bec5")

    panels = [
        (ed1, "ED  ·  end-diastole", "#c62828"),
        (es, "ES  ·  end-systole", "#1565c0"),
        (ed2, "ED′  ·  next cycle, same phase", "#c62828"),
    ]
    for i, (f, label, color) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(frames[f])
        ax.set_title(f"{label}\nframe {f}", fontsize=11, color=color, pad=8)
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :])
    ax.barh([1, 0], [e_same, e_opp], height=0.55,
            color=["#2e7d32", "#b0bec5"], alpha=0.9)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["sim(ED, ED′)  same phase", "sim(ED, ES)  opposite phase"],
                       fontsize=11)
    for y, v in [(1, e_same), (0, e_opp)]:
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=11, fontweight="bold",
                color="#2e7d32" if y == 1 else "#546e7a")
    ax.axvline(0, color="#90a4ae", lw=1)
    lo = min(0.0, e_opp) - 0.15
    ax.set_xlim(lo, max(e_same, 0) + 0.18)
    ax.set_title("embedding cosine similarity — same cardiac phase is closer",
                 fontsize=11, color="#37474f")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#607d8b")

    fig.suptitle("EchoFM embeddings are cardiac-phase aware", fontsize=14,
                 fontweight="bold", color="#263238", y=0.985)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = models_mae.mae_vit_large_patch16(
        num_frames=32, t_patch_size=4, pred_t_dim=8,
        sep_pos_embed=True, cls_embed=True, norm_pix_loss=True,
    ).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    ds = EchoDataset_from_cache_npy(
        args.data_path, num_frames=32, image_size=224, frame_stride=1
    )
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.num_scan, len(ds)), replace=False)

    candidates = []
    for clip_i in picks:
        imgs = ds[int(clip_i)].unsqueeze(0).to(device)
        if not is_bmode(imgs.cpu()):
            continue
        area = cavity_curve(imgs)
        L = cycle_length(imgs)
        eds, es = find_ed_es(area, L)
        if len(eds) < 2:
            continue
        ed1, ed2 = eds
        tk = lambda f: min(f // 4, 7)
        if len({tk(ed1), tk(ed2), tk(es)}) < 3:
            continue
        # require visible contraction in the cavity curve
        if area[ed1] - area[es] < 0.03 or area[ed2] - area[es] < 0.03:
            continue
        e = embed_sim_map(model, imgs)
        e_same = float(e[tk(ed1), tk(ed2)])
        e_opp = float(0.5 * (e[tk(ed1), tk(es)] + e[tk(ed2), tk(es)]))
        if e_same <= e_opp + 0.3:
            continue
        # keep the exact sampled window: the dataset draws a random temporal
        # window per __getitem__, so re-loading would desync frames and labels
        candidates.append(
            (e_same - e_opp, int(clip_i), ed1, es, ed2, e_same, e_opp,
             imgs[0].cpu(), area)
        )

    candidates.sort(key=lambda c: -c[0])
    print(f"{len(candidates)} strong B-mode candidates")
    for rank, (gap, clip_i, ed1, es, ed2, e_same, e_opp, imgs_c, area) in enumerate(
        candidates[: args.num_render]
    ):
        frames = [np.clip(imgs_c[:, f].permute(1, 2, 0).numpy(), 0, 1)
                  for f in range(imgs_c.shape[1])]
        path = os.path.join(args.out, f"hero_{rank:02d}_gap{gap:.2f}.png")
        render(frames, area, ed1, es, ed2, e_same, e_opp, path)
        print("wrote", path, os.path.basename(ds.paths[clip_i]))


if __name__ == "__main__":
    main()
