"""
View-clustering diagnostic: do EchoFM video embeddings separate by echo view?

Builds a view-labeled sample (apical / PLAX / PSAX / subcostal / RV-inflow /
suprasternal) by joining the pretrain cache hashmaps to viewlist predictions,
encodes each clip to a video embedding, projects to 2D (t-SNE), and colors by
view. Reports the silhouette score (higher = views cluster more cleanly).

Run inside the SIF on a GPU node:
  python tools/view_clustering.py --ckpt logs/<run>/checkpoint-000NN.pth \
      --per_view 150 --out logs/<run>/viewclust_epNN
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from EchoFM import models_mae

L = "/mnt/weka/wekafs/mm_fm_training/lvfp/data"
CACHES = [
    (f"{L}/pretrain_apical2_hashmap.csv",
     "/mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips", ".npy"),
    (f"{L}/pretrain_nonapical_hashmap.csv",
     "/mnt/weka/wekafs/mm_fm_training/echo_pretrain_nonapical/clips", ".npz"),
]
VIEWLIST = f"{L}/viewlist_predictions.csv"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # unused; kept for parity
COLORS = {
    "A4C": "#d62728", "A2C": "#e377c2", "A3C": "#ff9896", "A5C": "#c49c94",
    "PLAX": "#1f77b4", "PSAX": "#2ca02c", "subcostal": "#9467bd",
    "RV-inflow": "#ff7f0e", "suprasternal": "#8c564b",
}


def coarse(v):
    """Canonical echo view (strips zoomed/color/level modifiers, keeps the plane)."""
    v = str(v).upper()
    # apical: keep A2C/A3C/A4C/A5C distinct (AP4 = apical-4ch color -> A4C)
    if v.startswith("A2C"):
        return "A2C"
    if v.startswith("A3C"):
        return "A3C"
    if v.startswith("A4C") or v.startswith("AP4"):
        return "A4C"
    if v.startswith("A5C"):
        return "A5C"
    if "PLAX" in v:
        return "PLAX"
    if "PSAX" in v or "SAX" in v:
        return "PSAX"
    if "SUBCOST" in v or v.startswith("SC"):
        return "subcostal"
    if "SUPRA" in v or v.startswith("SSN"):
        return "suprasternal"
    if "RV INF" in v:
        return "RV-inflow"
    return None


def build_labeled(per_view, seed):
    p2v = {}
    with open(VIEWLIST) as f:
        for row in csv.DictReader(f):
            if row.get("path"):
                p2v[row["path"]] = row.get("view")
    by_view = {}
    for hashmap, clipdir, ext in CACHES:
        with open(hashmap) as f:
            for row in csv.DictReader(f):
                v = coarse(p2v.get(row["src"]))
                if v is None:
                    continue
                path = os.path.join(clipdir, row["hash"] + ext)
                by_view.setdefault(v, []).append(path)
    rng = np.random.RandomState(seed)
    items = []
    for v, paths in by_view.items():
        paths = [p for p in paths if os.path.isfile(p)]
        rng.shuffle(paths)
        for p in paths[:per_view]:
            items.append((p, v))
    return items


def load_clip(path, num_frames=32):
    if path.endswith(".npz"):
        with np.load(path) as z:
            a = z["clip"]
    else:
        a = np.load(path, mmap_mode="r")
    T = a.shape[0]
    idx = np.linspace(0, T - 1, num_frames).astype(int) if T >= num_frames \
        else np.arange(num_frames) % T
    clip = np.asarray(a[idx]).astype(np.float32) / 255.0
    clip = clip[..., ::-1]  # BGR -> RGB (matches training channel_order)
    t = torch.from_numpy(np.ascontiguousarray(clip)).permute(3, 0, 1, 2)  # 3,T,H,W
    if t.shape[-2:] != (224, 224):
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear",
                                            align_corners=False)
    return t


@torch.no_grad()
def embed(model, clip, device):
    latent, _, _ = model.forward_encoder(clip[None].to(device), mask_ratio=0.0)
    return latent.mean(dim=1)[0].float().cpu().numpy()  # [1024] video embedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per_view", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models_mae.mae_vit_large_patch16(
        num_frames=32, t_patch_size=4, pred_t_dim=8,
        sep_pos_embed=True, cls_embed=True, norm_pix_loss=True,
    ).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    epoch = ckpt.get("epoch", "?")

    items = build_labeled(args.per_view, args.seed)
    print(f"embedding {len(items)} clips across views ...", flush=True)
    X, y = [], []
    for i, (path, v) in enumerate(items):
        try:
            X.append(embed(model, load_clip(path), device))
            y.append(v)
        except Exception as e:
            print(f"skip {os.path.basename(path)}: {e}", flush=True)
    X = np.stack(X)
    y = np.array(y)

    sil = float(silhouette_score(X, y)) if len(set(y)) > 1 else float("nan")
    Z = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=args.seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    for v in sorted(set(y)):
        m = y == v
        ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7,
                   c=COLORS.get(v, "#777777"), label=f"{v} ({m.sum()})")
    ax.legend(fontsize=10, markerscale=1.5, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"EchoFM video embeddings by view — epoch {epoch}\n"
                 f"t-SNE · silhouette = {sil:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "view_tsne.png"), dpi=120, bbox_inches="tight")

    summary = {"ckpt": args.ckpt, "epoch": int(epoch) if epoch != "?" else -1,
               "n": len(y), "silhouette": sil,
               "per_view": {v: int((y == v).sum()) for v in sorted(set(y))}}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
