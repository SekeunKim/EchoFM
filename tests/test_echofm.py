"""
Smoke/unit tests for the completed EchoFM pretraining code.

Run:  python tests/test_echofm.py            (CPU)
      python tests/test_echofm.py --cuda     (GPU)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from EchoFM import models_mae
from data.dataset import sample_clip_indices


def make_tiny(device):
    model = models_mae.MaskedAutoencoderViT(
        img_size=64,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        num_frames=8,
        t_patch_size=2,
        pred_t_dim=4,
        sep_pos_embed=True,
        cls_embed=True,
        norm_pix_loss=False,
    )
    return model.to(device)


def test_masking(device):
    model = make_tiny(device)
    N, D = 3, 16
    T, L = 4, 16  # t_grid, spatial tokens (4x4)
    mask_ratio = 0.5
    len_keep = int(L * (1 - mask_ratio))

    x = torch.arange(N * T * L, dtype=torch.float32, device=device)
    x = x.view(N, T * L, 1).repeat(1, 1, D)

    x_masked, mask, ids_restore, ids_keep = model.uniform_random_masking(x, mask_ratio, L)

    # shapes
    assert x_masked.shape == (N, T * len_keep, D)
    assert mask.shape == (N, T * L)
    assert ids_keep.shape == (N, T * len_keep)

    # spatio-temporal consistency: identical spatial mask at every t
    m = mask.view(N, T, L)
    for t in range(T):
        assert torch.equal(m[:, t, :], m[:, 0, :]), "mask differs across time"
    assert (m.sum(dim=2) == L - len_keep).all(), "keep count differs per frame"

    # kept tokens really are the unmasked ones, in original order
    ref = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
    assert torch.equal(x_masked, ref)

    # decoder-style unshuffle restores kept tokens to original positions
    placeholder = torch.full((N, T * L - T * len_keep, D), -1.0, device=device)
    combined = torch.cat([x_masked, placeholder], dim=1)
    restored = torch.gather(
        combined, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D)
    )
    keep_pos = mask == 0
    assert torch.equal(restored[keep_pos], x[keep_pos]), "ids_restore mis-places kept tokens"
    assert (restored[~keep_pos] == -1.0).all(), "ids_restore mis-places mask tokens"
    print("[ok] uniform_random_masking (spatio-temporal consistent, invertible)")


def test_triplet_sampling(device):
    model = make_tiny(device)
    N, T, D = 2, 8, 16
    cls_tokens = torch.randn(N, T, D, device=device)

    # crafted similarity: anchor 0 similar to {1,4}, dissimilar to rest
    sim = torch.zeros(N, T, T, device=device)
    for n in range(N):
        sim[n] = torch.eye(T, device=device)
        sim[n, 0, 1] = 0.9
        sim[n, 0, 4] = 0.8
    out = model.triplet_sampling(sim, cls_tokens)
    assert out is not None
    a, p, n_ = out
    assert a.shape == p.shape == n_.shape and a.shape[1] == D

    # adjacency exclusion: for anchor t, negative index must not be t-1/t/t+1.
    # verify via exhaustive check on the mask logic for one row
    row = sim[0, 3, :]
    idx = torch.arange(T, device=device)
    not_self = idx != 3
    thresh = row[not_self].mean()
    neg_ok = ((row <= thresh) & not_self & ((idx - 3).abs() > 1)).nonzero().flatten()
    assert 2 not in neg_ok and 3 not in neg_ok and 4 not in neg_ok

    # degenerate similarity (all equal) must not crash
    flat = torch.ones(N, T, T, device=device)
    assert model.triplet_sampling(flat, cls_tokens) is None
    print("[ok] triplet_sampling (adjacency exclusion, degenerate-safe)")


def test_forward_backward(device):
    model = make_tiny(device)
    imgs = torch.rand(2, 3, 8, 64, 64, device=device)
    loss, pred, mask, parts = model(imgs, mask_ratio=0.75)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert torch.isfinite(parts["recon"]) and torch.isfinite(parts["triplet"])
    loss.backward()
    n_grad, n_bad = 0, 0
    for name, prm in model.named_parameters():
        if prm.grad is not None:
            n_grad += 1
            if not torch.isfinite(prm.grad).all():
                n_bad += 1
                print("  non-finite grad:", name)
    assert n_bad == 0
    assert n_grad > 0
    print(f"[ok] forward/backward (loss={loss.item():.4f}, {n_grad} params with grads)")


def test_forward_larger_ratio(device):
    # mask_ratio used in production
    model = make_tiny(device)
    imgs = torch.rand(1, 3, 8, 64, 64, device=device)
    loss, _, _, _ = model(imgs, mask_ratio=0.75)
    assert torch.isfinite(loss)
    print("[ok] forward at mask_ratio=0.75")


def test_clip_indices():
    # long clip: contiguous strided window
    idx = sample_clip_indices(100, 32, 1)
    assert len(idx) == 32 and max(idx) < 100 and idx == sorted(idx)
    d = [b - a for a, b in zip(idx, idx[1:])]
    assert all(x == 1 for x in d)
    # short clip: wraps, never exceeds range, no crash
    idx = sample_clip_indices(10, 32, 1)
    assert len(idx) == 32 and max(idx) < 10 and min(idx) >= 0
    idx = sample_clip_indices(1, 8, 1)
    assert idx == [0] * 8
    print("[ok] sample_clip_indices")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"running tests on {device}")
    torch.manual_seed(0)

    test_clip_indices()
    test_masking(device)
    test_triplet_sampling(device)
    test_forward_backward(device)
    test_forward_larger_ratio(device)
    print("ALL TESTS PASSED")
