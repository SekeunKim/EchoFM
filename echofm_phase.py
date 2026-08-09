"""
EchoFM phase utilities: cardiac cycle-length, ED/ES frame detection, and
embedding-based phase retrieval for echocardiogram clips.

Self-contained (torch + numpy only). The pixel-based functions need no model;
the embedding-based ones take an EchoFM model built from the GitHub repo
(https://github.com/SekeunKim/EchoFM).

Clip format everywhere: float tensor [3, T, H, W] in [0, 1] (or [1, 3, T, H, W]).

Example
-------
    import torch
    from huggingface_hub import hf_hub_download
    from echofm_phase import detect_ed_es, phase_embeddings, same_phase_frame

    result = detect_ed_es(clip, fps=30)          # no model required
    print(result)  # {'ed_frames': [...], 'es_frame': ..., 'cycle_frames': ..., 'hr_bpm': ...}

    z = phase_embeddings(model, clip)            # [T_tokens, D] per-frame embeddings
    match = same_phase_frame(model, clip, frame=result['ed_frames'][0])
"""
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "cavity_curve",
    "cycle_length",
    "detect_ed_es",
    "phase_embeddings",
    "phase_similarity",
    "same_phase_frame",
]


def _as_video(clip):
    """Accept [3,T,H,W] or [1,3,T,H,W]; return [3,T,H,W] float tensor."""
    if clip.dim() == 5:
        clip = clip[0]
    if clip.dim() != 4:
        raise ValueError(f"expected [3,T,H,W] or [1,3,T,H,W], got {tuple(clip.shape)}")
    return clip.float()


def cavity_curve(clip, ventricle_rows=0.55, cone_thresh=0.05, dark_thresh=0.15):
    """
    Ventricular cavity-size proxy per frame: fraction of dark (blood-pool)
    pixels inside the upper part of the scan cone (apex-up display; the lower
    part holds the atria, which move in anti-phase with the ventricles).

    Returns: np.ndarray [T], smoothed with a 3-frame moving average.
    """
    clip = _as_video(clip)
    gray = clip.mean(dim=0)  # [T, H, W]
    H = gray.shape[-2]
    cone = gray.max(dim=0).values > cone_thresh
    cone[int(ventricle_rows * H):, :] = False
    dark = (gray < dark_thresh) & cone
    area = dark.flatten(1).sum(-1).float() / max(int(cone.sum()), 1)
    a = area.cpu().numpy()
    return np.convolve(a, np.ones(3) / 3.0, mode="same")


def cycle_length(clip, min_lag=8, max_lag=None, pool=32):
    """
    Cardiac cycle length in frames, from the reprise peak of the frame-level
    pixel-similarity lag curve (frames one cycle apart look alike again).

    Returns int (frames) or None when no reprise is found (e.g. < 1 cycle).
    """
    clip = _as_video(clip)
    x = clip.mean(dim=0)  # [T, H, W]
    T = x.shape[0]
    max_lag = min(max_lag or T - 4, T - 2)
    if max_lag <= min_lag:
        return None
    f = F.adaptive_avg_pool2d(x.unsqueeze(1), (pool, pool)).flatten(1)
    f = F.normalize(f - f.mean(dim=-1, keepdim=True), dim=-1)
    S = (f @ f.T).cpu().numpy()
    curve = np.array([np.mean(np.diag(S, k)) for k in range(1, T)])
    L = min_lag + int(np.argmax(curve[min_lag - 1 : max_lag]))
    if curve[L - 1] <= curve[: min_lag - 1].min() + 1e-3:
        return None
    return L


def detect_ed_es(clip, fps=None):
    """
    Detect end-diastole (ED, max cavity) and end-systole (ES, min cavity)
    frames. When the cycle length is measurable, the second ED is implied by
    periodicity (ED' = ED + cycle), which is robust to noisy cavity curves.

    Returns dict:
        ed_frames    [ed, ed'] (or [ed] when < 2 cycles in the clip)
        es_frame     int
        cycle_frames int or None
        hr_bpm       float or None (needs fps)
        cavity       np.ndarray [T] cavity proxy curve
    """
    area = cavity_curve(clip)
    T = len(area)
    L = cycle_length(clip)

    if L is not None and 4 < L < T:
        ed1 = int(np.argmax(area[: T - L]))
        ed2 = ed1 + L
        es = ed1 + 1 + int(np.argmin(area[ed1 + 1 : ed2]))
        eds = [ed1, ed2]
    else:
        ed1 = int(np.argmax(area))
        es = int(np.argmin(area))
        eds = [ed1]

    hr = 60.0 * fps / L if (L and fps) else None
    return {
        "ed_frames": eds,
        "es_frame": es,
        "cycle_frames": L,
        "hr_bpm": hr,
        "cavity": area,
    }


@torch.no_grad()
def phase_embeddings(model, clip):
    """
    Per-timestep phase embeddings from EchoFM: encoder without masking, then
    the frame projector. Returns L2-normalized [T_tokens, D] (T_tokens = 8
    for the released 32-frame model; each token covers t_patch_size frames).
    """
    clip = _as_video(clip).unsqueeze(0).to(next(model.parameters()).device)
    latent, _, _ = model.forward_encoder(clip, mask_ratio=0.0)
    z = torch.stack(model.forward_prj(latent), dim=1)[0]
    return F.normalize(z, p=2, dim=-1)


@torch.no_grad()
def phase_similarity(model, clip):
    """Embedding self-similarity map [T_tokens, T_tokens]."""
    z = phase_embeddings(model, clip)
    return z @ z.T


def same_phase_frame(model, clip, frame, exclude_adjacent=1):
    """
    Find the timestep the model considers the same cardiac phase as `frame`
    (from a different part of the clip). Returns dict with the matched token,
    its representative frame index, and the similarity.
    """
    clip = _as_video(clip)
    sim = phase_similarity(model, clip)
    T = sim.shape[0]
    grp = clip.shape[1] // T
    t = min(frame // grp, T - 1)
    idx = torch.arange(T, device=sim.device)
    mask = (idx - t).abs() > exclude_adjacent
    row = sim[t].masked_fill(~mask, float("-inf"))
    best = int(row.argmax())
    return {
        "query_token": t,
        "match_token": best,
        "match_frame": best * grp + grp // 2,
        "similarity": float(sim[t, best]),
    }
