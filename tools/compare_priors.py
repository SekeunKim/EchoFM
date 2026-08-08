"""Alpha sweep: prior features x - alpha * temporal_mean; ED/ES cross-cycle test."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from data.dataset import EchoDataset_from_cache_npy
from tools.visualize_ed_es import cavity_curve, find_ed_es

def prior_feats(imgs, T=8):
    x = imgs[0].mean(dim=0)
    Tf = x.shape[0]
    x = x.view(T, Tf // T, *x.shape[-2:]).mean(1)
    x = F.adaptive_avg_pool2d(x.unsqueeze(0), (32, 32))[0].flatten(1)
    return x - x.mean(dim=-1, keepdim=True)

ds = EchoDataset_from_cache_npy(
    "/mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips",
    num_frames=32, image_size=224, frame_stride=1)
rng = np.random.RandomState(1)   # different clip sample than before
picks = rng.choice(len(ds), size=48, replace=False)

alphas = [0.0, 0.5, 0.8, 1.0]
res = {a: [] for a in alphas}
for i in picks:
    imgs = ds[int(i)].unsqueeze(0)
    area = cavity_curve(imgs); eds, es = find_ed_es(area)
    if len(eds) < 2: continue
    ed1, ed2 = eds
    t = lambda f: min(f // 4, 7)
    if len({t(ed1), t(ed2), t(es)}) < 3: continue
    x = prior_feats(imgs)
    for a in alphas:
        xa = x - a * x.mean(dim=0, keepdim=True)
        xn = F.normalize(xa, dim=-1)
        m = (xn @ xn.T).numpy()
        same = m[t(ed1), t(ed2)]
        opp = 0.5 * (m[t(ed1), t(es)] + m[t(ed2), t(es)])
        res[a].append((float(same), float(opp)))

for a, v in res.items():
    ok = sum(s > o for s, o in v)
    gap = np.mean([s - o for s, o in v])
    med = np.median([s - o for s, o in v])
    print(f"alpha={a}: pass {ok}/{len(v)}  gap mean {gap:+.4f} median {med:+.4f}")
