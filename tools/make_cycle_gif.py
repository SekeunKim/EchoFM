"""
Animated periodicity check: for each clip, a looping GIF with
  - the playing echo clip (current frame, red box)
  - the frame the EMBEDDING picks as same-phase best match (green box)
  - the embedding phase trajectory in 2D PCA (dot = current time; a learned
    cycle shows as a loop, a collapsed embedding as a single blob)
  - the current anchor row of embedding vs pixel similarity

Run (inside the SIF, on a GPU node):
  python tools/make_cycle_gif.py --ckpt logs/<run>/checkpoint-000NN.pth \
      --data_path /mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips \
      --out logs/<run>/cycle_gif_epNN
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
from PIL import Image

from EchoFM import models_mae
from data.dataset import EchoDataset_from_cache_npy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_clips", type=int, default=3)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--t_patch_size", type=int, default=4)
    ap.add_argument("--pred_t_dim", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def embed_clip(model, imgs):
    latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
    cls_stack = torch.stack(model.forward_prj(latent), dim=1)  # [1, T, D]
    z = F.normalize(cls_stack, p=2, dim=-1)
    embed_sim = torch.bmm(z, z.transpose(1, 2))[0]
    prior_sim = model.pixel_similarity(imgs)[0]
    return z[0].cpu().numpy(), embed_sim.cpu().numpy(), prior_sim.cpu().numpy()


def render_frame(fig_axes, frames, f, T, grp, z2, embed_sim, prior_sim):
    fig, (ax_vid, ax_match, ax_pca, ax_bar) = fig_axes
    t = f // grp
    idx = np.arange(T)
    nonadj = np.abs(idx - t) > 1
    best_e = int(np.argmax(np.where(nonadj, embed_sim[t], -np.inf)))
    best_p = int(np.argmax(np.where(nonadj, prior_sim[t], -np.inf)))

    for ax in (ax_vid, ax_match, ax_pca, ax_bar):
        ax.clear()

    ax_vid.imshow(frames[f])
    ax_vid.set_title(f"clip frame {f} (token t{t})", fontsize=9, color="tab:red")
    ax_vid.axis("off")

    ax_match.imshow(frames[best_e * grp + grp // 2])
    same = " = pixel pick" if best_e == best_p else f" (pixel pick: t{best_p})"
    ax_match.set_title(
        f"embed same-phase match: t{best_e}{same}", fontsize=9, color="tab:green"
    )
    ax_match.axis("off")

    ax_pca.plot(z2[:, 0], z2[:, 1], "-o", color="0.6", ms=4)
    for i in range(T):
        ax_pca.annotate(f"t{i}", z2[i], fontsize=7, color="0.4")
    ax_pca.plot(z2[t, 0], z2[t, 1], "o", color="tab:red", ms=11)
    ax_pca.plot(z2[best_e, 0], z2[best_e, 1], "o", mfc="none", mec="tab:green",
                ms=15, mew=2)
    ax_pca.set_title("embedding phase trajectory (PCA)", fontsize=9)
    ax_pca.set_xticks([]), ax_pca.set_yticks([])

    w = 0.4
    ax_bar.bar(idx - w / 2, prior_sim[t], width=w, label="pixel", color="tab:blue")
    ax_bar.bar(idx + w / 2, embed_sim[t], width=w, label="embed", color="tab:orange")
    lo = min(prior_sim[t].min(), embed_sim[t].min())
    ax_bar.set_ylim(max(0, lo - 0.05), 1.0)
    ax_bar.axvline(t, color="tab:red", ls=":", lw=1)
    ax_bar.legend(fontsize=7, loc="lower left")
    ax_bar.set_title(f"similarity to current token t{t}", fontsize=9)
    ax_bar.set_xticks(idx)
    ax_bar.tick_params(labelsize=7)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return Image.fromarray(buf)


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
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    ds = EchoDataset_from_cache_npy(
        args.data_path, num_frames=args.num_frames, image_size=224, frame_stride=1
    )
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.num_clips, len(ds)), replace=False)

    T = args.num_frames // args.t_patch_size
    grp = args.num_frames // T
    for k, clip_i in enumerate(picks):
        imgs = ds[int(clip_i)].unsqueeze(0).to(device)
        z, embed_sim, prior_sim = embed_clip(model, imgs)

        zc = z - z.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(zc, full_matrices=False)
        z2 = zc @ vt[:2].T

        frames = [
            np.clip(imgs[0, :, f].permute(1, 2, 0).cpu().numpy(), 0, 1)
            for f in range(args.num_frames)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), dpi=80)
        fig.subplots_adjust(hspace=0.25, wspace=0.15)
        fig_axes = (fig, (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]))
        gif_frames = [
            render_frame(fig_axes, frames, f, T, grp, z2, embed_sim, prior_sim)
            for f in range(args.num_frames)
        ]
        plt.close(fig)

        name = os.path.basename(ds.paths[int(clip_i)]).replace(".npy", "")
        path = os.path.join(args.out, f"cycle_{k:02d}_{name[:12]}.gif")
        gif_frames[0].save(
            path, save_all=True, append_images=gif_frames[1:],
            duration=int(1000 / args.fps), loop=0,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
