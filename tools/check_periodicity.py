"""
Periodicity diagnostic for EchoFM checkpoints.

Embeds unmasked echo clips with the pretrained encoder + frame projector,
then checks whether the per-frame embedding self-similarity map reproduces
the cardiac-cycle structure visible in pixel space:

  - offdiag_r:        Pearson r between embedding and pixel similarity maps
                      (off-diagonal entries), per clip.
  - phase_contrast:   mean embedding similarity over pixel-prior positive
                      pairs minus over negative pairs. > 0 means the encoder
                      pulls same-phase frames together.
  - lag curve + heatmaps saved as PNGs for visual inspection.

Run (inside the SIF, on a GPU node):
  python tools/check_periodicity.py --ckpt logs/<run>/checkpoint-000NN.pth \
      --data_path /mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips \
      --out logs/<run>/periodicity_epNN
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

from EchoFM import models_mae
from data.dataset import EchoDataset_from_cache_npy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_clips", type=int, default=64)
    ap.add_argument("--num_plot", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--t_patch_size", type=int, default=4)
    ap.add_argument("--pred_t_dim", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def embed_clips(model, imgs):
    latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
    cls_stack = torch.stack(model.forward_prj(latent), dim=1)  # [N, T, D]
    z = F.normalize(cls_stack, p=2, dim=-1)
    embed_sim = torch.bmm(z, z.transpose(1, 2))
    prior_sim = model.pixel_similarity(imgs)
    return embed_sim, prior_sim


def offdiag(x):
    T = x.shape[-1]
    m = ~np.eye(T, dtype=bool)
    return x[m]


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


def phase_contrast(embed_sim, prior_sim):
    """Mean embed similarity over prior positives minus over prior negatives."""
    T = prior_sim.shape[-1]
    eye = np.eye(T, dtype=bool)
    idx = np.arange(T)
    adjacent = np.abs(idx[None, :] - idx[:, None]) <= 1
    row_mean = np.where(eye, 0, prior_sim).sum(-1) / (T - 1)
    pos = (prior_sim > row_mean[:, None]) & ~eye
    neg = (prior_sim <= row_mean[:, None]) & ~adjacent
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    return float(embed_sim[pos].mean() - embed_sim[neg].mean())


def lag_curve(sim):
    T = sim.shape[-1]
    return [float(np.mean([sim[t, t + k] for t in range(T - k)])) for k in range(T)]


def make_match_figure(imgs, prior_sim, embed_sim, clip_name, r, out_path):
    """
    Visual check that the similarity maps match the actual video: frame strip,
    raw cardiac intensity signal, anchor-row similarity curves, and the frames
    the pixel prior / embedding pick as best (same phase) vs worst match.

    imgs: [3, T_frames, H, W] cpu tensor in [0,1]; prior_sim/embed_sim: [T,T] np
    """
    T = prior_sim.shape[0]
    T_frames = imgs.shape[1]
    grp = T_frames // T
    thumbs = [
        np.clip(imgs[:, t * grp + grp // 2].permute(1, 2, 0).numpy(), 0, 1)
        for t in range(T)
    ]
    intensity = imgs.mean(dim=(0, 2, 3)).numpy()

    # anchor = token whose prior row varies most (most phase-informative)
    a = int(np.argmax([np.delete(prior_sim[t], t).var() for t in range(T)]))
    idx = np.arange(T)
    nonadj = np.abs(idx - a) > 1

    def best(row):
        return int(np.argmax(np.where(nonadj, row, -np.inf)))

    def worst(row):
        return int(np.argmin(np.where(nonadj, row, np.inf)))

    bp, be, we = best(prior_sim[a]), best(embed_sim[a]), worst(embed_sim[a])

    fig = plt.figure(figsize=(2.0 * T, 10))
    gs = fig.add_gridspec(4, T, height_ratios=[1.6, 0.9, 0.9, 1.8], hspace=0.5)

    for t in range(T):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(thumbs[t])
        ax.axis("off")
        marks = []
        if t == a:
            marks.append("ANCHOR")
        if t == bp:
            marks.append("pixel*")
        if t == be:
            marks.append("embed*")
        if t == we:
            marks.append("far")
        ax.set_title(f"t{t} " + " ".join(marks), fontsize=8,
                     color="tab:red" if t == a else "black")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.arange(T_frames), intensity, "-", color="tab:gray")
    for t in range(T + 1):
        ax.axvline(t * grp - 0.5, color="0.88", lw=0.6)
    ax.set_xlim(-0.5, T_frames - 0.5)
    ax.set_title("cardiac signal: mean frame intensity (raw 32 frames)", fontsize=9)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(idx, prior_sim[a], "o-", label="pixel prior")
    ax.plot(idx, embed_sim[a], "s-", label="embedding")
    ax.axvline(a, color="k", ls=":", lw=1)
    ax.scatter([bp], [prior_sim[a, bp]], marker="*", s=160, color="tab:blue", zorder=5)
    ax.scatter([be], [embed_sim[a, be]], marker="*", s=160, color="tab:orange", zorder=5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("t token", fontsize=8)
    ax.set_title(f"similarity to anchor t{a}", fontsize=9)

    panels = [
        (a, f"anchor t{a}"),
        (bp, f"pixel best t{bp}"),
        (be, f"embed best t{be}"),
        (we, f"embed worst t{we}"),
    ]
    span = max(T // 4, 1)
    for i, (t, name) in enumerate(panels):
        ax = fig.add_subplot(gs[3, i * span : (i + 1) * span])
        ax.imshow(thumbs[t])
        ax.axis("off")
        ax.set_title(name, fontsize=9)

    fig.suptitle(f"{clip_name}  r={r:.2f}", fontsize=10)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = models_mae.mae_vit_large_patch16(
        num_frames=args.num_frames,
        t_patch_size=args.t_patch_size,
        pred_t_dim=args.pred_t_dim,
        sep_pos_embed=True,
        cls_embed=True,
        norm_pix_loss=True,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing keys:", missing[:8])
    model.eval()

    ds = EchoDataset_from_cache_npy(
        args.data_path, num_frames=args.num_frames, image_size=224, frame_stride=1
    )
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.num_clips, len(ds)), replace=False)

    results, plotted = [], 0
    emb_lags, pix_lags = [], []
    for start in range(0, len(picks), args.batch_size):
        batch_idx = picks[start : start + args.batch_size]
        imgs = torch.stack([ds[int(i)] for i in batch_idx]).to(device)
        embed_sim, prior_sim = embed_clips(model, imgs)
        embed_sim = embed_sim.cpu().numpy()
        prior_sim = prior_sim.cpu().numpy()

        for j, clip_i in enumerate(batch_idx):
            e, p = embed_sim[j], prior_sim[j]
            r = pearson(offdiag(e), offdiag(p))
            pc = phase_contrast(e, p)
            results.append(
                {"clip": os.path.basename(ds.paths[int(clip_i)]),
                 "offdiag_r": r, "phase_contrast": pc}
            )
            emb_lags.append(lag_curve(e))
            pix_lags.append(lag_curve(p))

            if plotted < args.num_plot:
                make_match_figure(
                    imgs[j].cpu(), p, e, results[-1]["clip"], r,
                    os.path.join(args.out, f"match_{plotted:02d}.png"),
                )
                fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
                for ax, m, title in zip(
                    axes, [p, e], ["pixel cycle similarity", "embedding similarity"]
                ):
                    im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel("t token")
                fig.colorbar(im, ax=axes, shrink=0.85)
                fig.suptitle(
                    f"{results[-1]['clip']}  r={r:.2f}  contrast={pc if pc is None else round(pc, 3)}",
                    fontsize=9,
                )
                fig.savefig(os.path.join(args.out, f"simmap_{plotted:02d}.png"), dpi=120)
                plt.close(fig)
                plotted += 1

    rs = [x["offdiag_r"] for x in results]
    pcs = [x["phase_contrast"] for x in results if x["phase_contrast"] is not None]
    summary = {
        "ckpt": args.ckpt,
        "epoch": int(ckpt.get("epoch", -1)),
        "num_clips": len(results),
        "offdiag_r_mean": float(np.mean(rs)),
        "offdiag_r_std": float(np.std(rs)),
        "phase_contrast_mean": float(np.mean(pcs)) if pcs else None,
        "phase_contrast_frac_pos": float(np.mean([p > 0 for p in pcs])) if pcs else None,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "clips": results}, f, indent=2)
    print(json.dumps(summary, indent=2))

    # mean similarity-vs-lag curve: periodicity shows as a dip-then-rise
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(np.mean(pix_lags, axis=0), "o-", label="pixel prior")
    ax.plot(np.mean(emb_lags, axis=0), "s-", label="embedding")
    ax.set_xlabel("temporal lag (tokens)")
    ax.set_ylabel("mean similarity")
    ax.legend()
    ax.set_title("similarity vs temporal lag")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lag_curve.png"), dpi=120)
    print(f"wrote {args.out}/summary.json, lag_curve.png, {plotted} heatmaps")


if __name__ == "__main__":
    main()
