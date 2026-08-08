"""
Reconstruction-quality diagnostic for EchoFM checkpoints.

Runs the MAE at the training mask ratio on held-out clips and saves
original / masked-input / reconstruction image grids plus quantitative
metrics (masked-patch MSE in the norm-pix training space, and masked-patch
PSNR in raw pixel space).

Run (inside the SIF, on a GPU node):
  python tools/check_recon.py --ckpt logs/<run>/checkpoint-000NN.pth \
      --data_path /mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips \
      --out logs/<run>/recon_epNN
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from EchoFM import models_mae
from data.dataset import EchoDataset_from_cache_npy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_clips", type=int, default=16)
    ap.add_argument("--num_plot", type=int, default=4)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--t_patch_size", type=int, default=4)
    ap.add_argument("--pred_t_dim", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


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
    model.eval()

    ds = EchoDataset_from_cache_npy(
        args.data_path, num_frames=args.num_frames, image_size=224, frame_stride=1
    )
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.num_clips, len(ds)), replace=False)

    results, plotted = [], 0
    for clip_i in picks:
        imgs = ds[int(clip_i)].unsqueeze(0).to(device)  # [1, 3, T, H, W]
        with torch.no_grad():
            loss, pred, mask, parts = model(imgs, mask_ratio=args.mask_ratio)

            # rebuild pixel-space reconstruction (denormalize per-patch)
            _imgs = torch.index_select(
                imgs, 2,
                torch.linspace(0, imgs.shape[2] - 1, model.pred_t_dim)
                .long().to(imgs.device),
            )
            target = model.patchify(_imgs)  # sets patch_info for unpatchify
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred_pix = pred * (var + 1.0e-6) ** 0.5 + mean

            m = mask.view(1, -1, 1)  # 1 = masked
            # paste: visible patches from the original, masked from prediction
            pasted = target * (1 - m) + pred_pix * m
            recon_vid = model.unpatchify(pasted).clamp(0, 1)
            masked_vid = model.unpatchify(target * (1 - m))
            orig_vid = model.unpatchify(target)

            mse_masked_raw = (((pred_pix - target) ** 2).mean(-1) * mask).sum() / mask.sum()
            psnr = float(10 * torch.log10(1.0 / (mse_masked_raw + 1e-12)))
            results.append(
                {"clip": os.path.basename(ds.paths[int(clip_i)]),
                 "loss_normpix": float(parts["recon"]),
                 "psnr_masked": psnr}
            )

        if plotted < args.num_plot:
            frames = [0, 2, 4, 6]
            fig, axes = plt.subplots(3, len(frames), figsize=(3 * len(frames), 8))
            rows = [("original", orig_vid), ("masked input", masked_vid),
                    ("reconstruction", recon_vid)]
            for r, (name, vid) in enumerate(rows):
                for c, f in enumerate(frames):
                    frame = vid[0, :, f].permute(1, 2, 0).cpu().numpy()
                    axes[r, c].imshow(np.clip(frame, 0, 1))
                    axes[r, c].axis("off")
                    if c == 0:
                        axes[r, c].set_title(name, loc="left", fontsize=10)
            fig.suptitle(
                f"{results[-1]['clip']}  PSNR(masked)={results[-1]['psnr_masked']:.1f} dB",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"recon_{plotted:02d}.png"), dpi=110)
            plt.close(fig)
            plotted += 1

    summary = {
        "ckpt": args.ckpt,
        "epoch": int(ckpt.get("epoch", -1)),
        "num_clips": len(results),
        "loss_normpix_mean": float(np.mean([x["loss_normpix"] for x in results])),
        "psnr_masked_mean": float(np.mean([x["psnr_masked"] for x in results])),
        "psnr_masked_std": float(np.std([x["psnr_masked"] for x in results])),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "clips": results}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}: summary.json + {plotted} recon grids")


if __name__ == "__main__":
    main()
