"""
Measurement probe: does the EchoFM embedding carry B-mode clinical measurements?

Labels each pretrain clip with its study's measurements (LVEF, LVIDd, IVSd, ...)
by joining clip-hash -> source DICOM folder (= STUDY_REF) -> matched_measurements.
Encodes clips to video embeddings, then for each measurement fits a 5-fold
Ridge probe (embedding -> value) and reports:
  - Pearson r (embedding vs GT)
  - within-tolerance rate (|pred - gt| <= reward.py tolerance)  -> "clinical hit rate"

Only apical/A4C-family clips are used for chamber measurements by default.

Run inside the SIF on a GPU node:
  python tools/measurement_probe.py --ckpt logs/<run>/checkpoint-000NN.pth \
      --n 800 --out logs/<run>/measprobe_epNN
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from EchoFM import models_mae
from tools.view_clustering import load_clip, embed, coarse as coarse_view

MEAS_CSV = ("/mnt/weka/wekafs/mm_fm_training/sk1064/code/10_echometricsAI/"
            "applications/echo_agent/data/main/matched_measurements.csv")
HASHMAPS = [
    ("/mnt/weka/wekafs/mm_fm_training/lvfp/data/pretrain_apical2_hashmap.csv",
     "/mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips", ".npy"),
    ("/mnt/weka/wekafs/mm_fm_training/lvfp/data/pretrain_nonapical_hashmap.csv",
     "/mnt/weka/wekafs/mm_fm_training/echo_pretrain_nonapical/clips", ".npz"),
]
# B-mode measurements EchoFM could plausibly carry, with reward.py tolerances
TARGETS = {"LVEF": 5, "LVIDd": 4, "LVIDs": 4, "IVSd": 2, "PWTd": 2,
           "LA_AP": 4, "LA_volume": 15, "LAVI": 6}


def study_of(src):
    parts = src.split("/")
    for i, p in enumerate(parts):
        if p.startswith("LV_Filling"):
            return parts[i + 1] if i + 1 < len(parts) else None
    return None


def _viewmap():
    """src DICOM path -> coarse view (A4C/A2C/PLAX/...), via viewlist predictions."""
    p2v = {}
    with open("/mnt/weka/wekafs/mm_fm_training/lvfp/data/viewlist_predictions.csv") as f:
        for row in csv.DictReader(f):
            if row.get("path"):
                p2v[row["path"]] = coarse_view(row.get("view"))
    return p2v


def build_labeled(n, seed, keep_views=None):
    meas = {}
    with open(MEAS_CSV) as f:
        for row in csv.DictReader(f):
            meas[str(row["STUDY_REF"])] = row
    p2v = _viewmap() if keep_views else None
    items = []
    for hashmap, clipdir, ext in HASHMAPS:
        with open(hashmap) as f:
            for row in csv.DictReader(f):
                st = study_of(row["src"])
                if not (st and st in meas):
                    continue
                if keep_views is not None and p2v.get(row["src"]) not in keep_views:
                    continue
                path = os.path.join(clipdir, row["hash"] + ext)
                if os.path.isfile(path):
                    items.append((path, meas[st]))
    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", default="", help="comma list, e.g. A4C,A2C (empty=all)")
    args = ap.parse_args()
    keep = set(args.views.split(",")) if args.views else None
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models_mae.mae_vit_large_patch16(
        num_frames=32, t_patch_size=4, pred_t_dim=8,
        sep_pos_embed=True, cls_embed=True, norm_pix_loss=True,
    ).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    epoch = ckpt.get("epoch", "?")

    items = build_labeled(args.n, args.seed, keep_views=keep)
    print(f"views={args.views or 'ALL'}", flush=True)
    print(f"encoding {len(items)} labeled clips ...", flush=True)
    X, rows = [], []
    for path, row in items:
        try:
            X.append(embed(model, load_clip(path), device)); rows.append(row)
        except Exception:
            pass
    X = np.stack(X)
    Xs = StandardScaler().fit_transform(X)

    results = {}
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (key, tol) in zip(axes.ravel(), TARGETS.items()):
        yv = np.array([float(r[key]) if r.get(key) not in ("", None) else np.nan
                       for r in rows])
        mask = ~np.isnan(yv)
        if mask.sum() < 50:
            ax.set_title(f"{key}: n={mask.sum()} (skip)"); ax.axis("off")
            results[key] = {"n": int(mask.sum())}
            continue
        Xm, ym = Xs[mask], yv[mask]
        from sklearn.ensemble import GradientBoostingRegressor
        pr_lin = np.zeros_like(ym); pr_nl = np.zeros_like(ym)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xm):
            pr_lin[te] = Ridge(alpha=10.0).fit(Xm[tr], ym[tr]).predict(Xm[te])
            pr_nl[te] = GradientBoostingRegressor(
                n_estimators=150, max_depth=3, subsample=0.8,
                random_state=0).fit(Xm[tr], ym[tr]).predict(Xm[te])
        r_lin = float(np.corrcoef(pr_lin, ym)[0, 1])
        r_nl = float(np.corrcoef(pr_nl, ym)[0, 1])
        preds = pr_nl if r_nl >= r_lin else pr_lin
        r = max(r_lin, r_nl)
        hit = float((np.abs(preds - ym) <= tol).mean())
        results[key] = {"n": int(mask.sum()), "pearson_r_linear": r_lin,
                        "pearson_r_nonlinear": r_nl, "within_tol_rate": hit, "tol": tol}
        ax.scatter(ym, preds, s=8, alpha=0.4)
        lo, hi = ym.min(), ym.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{key}  n={mask.sum()}  r_lin={r_lin:.2f} r_nl={r_nl:.2f}  "
                     f"hit@±{tol}={hit:.0%}", fontsize=10)
        ax.set_xlabel("ground truth"); ax.set_ylabel("probe pred")

    fig.suptitle(f"EchoFM embedding -> B-mode measurement probe (5-fold Ridge) — epoch {epoch}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "measurement_probe.png"), dpi=110, bbox_inches="tight")

    summary = {"epoch": int(epoch) if epoch != "?" else -1, "n_clips": len(rows),
               "targets": results}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
