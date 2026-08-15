"""
Alternative view-difference visualizations (beyond t-SNE/silhouette):
  1. view x view mean-embedding cosine-similarity heatmap
  2. LDA projection (supervised — best axes that separate views)
  3. linear vs MLP probe accuracy (is view info there but entangled?)

Reuses the labeled-sample builder + encoder from view_clustering.py.
"""
import argparse
import json
import os

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from EchoFM import models_mae
from tools.view_clustering import build_labeled, load_clip, embed, COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per_view", type=int, default=120)
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
    X, y = [], []
    for path, v in items:
        try:
            X.append(embed(model, load_clip(path), device)); y.append(v)
        except Exception:
            pass
    X = np.stack(X); y = np.array(y)
    views = sorted(set(y))
    Xs = StandardScaler().fit_transform(X)

    # 1) view x view mean-embedding cosine similarity
    mu = np.stack([X[y == v].mean(0) for v in views])
    mun = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    S = mun @ mun.T

    # 2) LDA supervised 2D projection
    ncomp = min(2, len(views) - 1)
    Z = LinearDiscriminantAnalysis(n_components=ncomp).fit_transform(Xs, y)
    if Z.shape[1] == 1:
        Z = np.c_[Z, np.zeros(len(Z))]

    # 3) probe accuracies (5-fold); integer-encode labels for sklearn robustness
    from sklearn.preprocessing import LabelEncoder
    yi = LabelEncoder().fit_transform(y)
    lin = cross_val_score(LogisticRegression(max_iter=2000, C=1.0),
                          Xs, yi, cv=5).mean()
    mlp = cross_val_score(MLPClassifier(hidden_layer_sizes=(256,), max_iter=800,
                                        random_state=0),
                          Xs, yi, cv=5).mean()
    chance = 1.0 / len(views)

    fig = plt.figure(figsize=(15, 5.2))
    # heatmap
    ax = fig.add_subplot(1, 3, 1)
    im = ax.imshow(S, vmin=S[~np.eye(len(views), dtype=bool)].min(), vmax=1, cmap="viridis")
    ax.set_xticks(range(len(views))); ax.set_xticklabels(views, rotation=90, fontsize=8)
    ax.set_yticks(range(len(views))); ax.set_yticklabels(views, fontsize=8)
    ax.set_title("view x view mean-embedding cosine\n(bright diagonal = views differ)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    # LDA
    ax = fig.add_subplot(1, 3, 2)
    for v in views:
        m = y == v
        ax.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, c=COLORS.get(v, "#777"), label=v)
    ax.legend(fontsize=7, markerscale=1.3, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("LDA projection (supervised)\nbest axes to separate views", fontsize=10)
    # probe bars
    ax = fig.add_subplot(1, 3, 3)
    bars = ax.bar(["chance", "linear\n(logreg)", "MLP\n(nonlinear)"],
                  [chance, lin, mlp], color=["#bbb", "#4c72b0", "#dd8452"])
    for b, val in zip(bars, [chance, lin, mlp]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("view-classification accuracy (5-fold)")
    ax.set_title("is view info recoverable?\nlinear vs nonlinear probe", fontsize=10)

    fig.suptitle(f"EchoFM view separability — epoch {epoch}  "
                 f"({len(views)} views, {len(y)} clips)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "view_separability.png"), dpi=120, bbox_inches="tight")

    summary = {"epoch": int(epoch) if epoch != "?" else -1, "n": len(y),
               "views": views, "chance": chance,
               "linear_acc": float(lin), "mlp_acc": float(mlp),
               "offdiag_cos_mean": float(S[~np.eye(len(views), dtype=bool)].mean())}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
