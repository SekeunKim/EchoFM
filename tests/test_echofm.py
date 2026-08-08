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
    N, T = 2, 8

    # crafted prior: anchor 0 similar to {1,4}, dissimilar to rest
    prior = torch.zeros(N, T, T, device=device)
    for n in range(N):
        prior[n] = torch.eye(T, device=device)
        prior[n, 0, 1] = 0.9
        prior[n, 0, 4] = 0.8
    embed_sim = torch.rand(N, T, T, device=device)
    pos_idx, neg_idx, valid = model.triplet_sampling(prior, embed_sim)
    assert pos_idx.shape == neg_idx.shape == valid.shape == (N, T)
    # anchor 0: positives must come from {1, 4}
    assert valid[0, 0]
    assert pos_idx[0, 0].item() in (1, 4)
    # negatives never self or adjacent
    idx = torch.arange(T, device=device)
    for n in range(N):
        for t in range(T):
            if valid[n, t]:
                assert (neg_idx[n, t] - t).abs() > 1

    # hard mining: pos = lowest embed-sim in pos set, neg = highest in neg set
    row = embed_sim[0, 0]
    cand = torch.tensor([1, 4], device=device)
    assert pos_idx[0, 0] == cand[row[cand].argmin()]

    # degenerate prior (all equal): no strictly-above-mean frames -> no valid
    flat = torch.ones(N, T, T, device=device)
    _, _, valid_flat = model.triplet_sampling(flat, embed_sim)
    assert not valid_flat.any()
    print("[ok] triplet_sampling (prior-driven sets, hard mining, degenerate-safe)")


def _periodic_video(N, T_frames, H, period, device):
    """Bright square whose x-position oscillates with the given frame period."""
    imgs = torch.zeros(N, 3, T_frames, H, H, device=device)
    import math

    for f in range(T_frames):
        cx = H // 2 + int((H // 4) * math.sin(2 * math.pi * f / period))
        imgs[:, :, f, H // 2 - 6 : H // 2 + 6, cx - 6 : cx + 6] = 1.0
    return imgs


def test_pixel_similarity_periodicity(device):
    model = make_tiny(device)  # num_frames=8, t_patch_size=2 -> t_grid=4
    imgs = _periodic_video(2, 8, 64, period=4, device=device)
    sim = model.pixel_similarity(imgs)  # [N, 4, 4]
    assert sim.shape == (2, 4, 4)
    # frame period 4 = token period 2: lag-2 tokens same phase, lag-1 opposite
    assert (sim[:, 0, 2] > sim[:, 0, 1] + 0.1).all(), "pixel prior misses periodicity"
    assert (sim[:, 1, 3] > sim[:, 1, 2] + 0.1).all()
    print("[ok] pixel_similarity captures cycle periodicity")


def test_periodic_triplet_loss(device):
    model = make_tiny(device)
    imgs = _periodic_video(2, 8, 64, period=4, device=device)
    cls_stack = torch.randn(2, 4, 64, device=device, requires_grad=True)
    loss, active = model.periodic_triplet_loss(imgs, cls_stack)
    assert torch.isfinite(loss) and loss.item() > 0, "triplet dead at init"
    assert 0.0 < active.item() <= 1.0
    loss.backward()
    assert cls_stack.grad is not None and torch.isfinite(cls_stack.grad).all()
    print(f"[ok] periodic_triplet_loss (loss={loss.item():.4f}, active={active.item():.2f})")


def test_forward_backward(device):
    model = make_tiny(device)
    imgs = torch.rand(2, 3, 8, 64, 64, device=device)
    loss, pred, mask, parts = model(imgs, mask_ratio=0.75)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert torch.isfinite(parts["recon"]) and torch.isfinite(parts["triplet"])
    assert "triplet_active" in parts and torch.isfinite(parts["triplet_active"])
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
    test_pixel_similarity_periodicity(device)
    test_periodic_triplet_loss(device)
    test_forward_backward(device)
    test_forward_larger_ratio(device)
    print("ALL TESTS PASSED")
