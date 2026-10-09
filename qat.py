"""Quantization-aware training: full-precision numbers stay backstage, rounded ones perform.

The grid scales are fixed before training begins. At every step the backstage
numbers are snapped onto their grids, the snapped model runs, and its gradients
nudge the backstage copy. When training ends, the snapped codes are what gets
saved.
"""
import numpy as np

from gridquant import BIG, decode, scale_of, top_code, tune_clip
from tiny_lm import Adam, make_windows, loss_and_grads


def _snap(w, s, bits):
    return np.clip(np.round(w / s), -top_code(bits), top_code(bits)).astype(np.int8)


def qat_train(params, words, plan, iters, lr, batch=256, seed=0, history=None):
    """Train with rounding inside the loop; returns codes, scale and bits per table."""
    backstage = {k: v.copy() for k, v in params.items()}
    grids = {}
    for t in BIG:
        cfg = plan.for_tensor(t)
        if cfg is None:
            continue
        b, tune, rows = cfg
        clip = tune_clip(backstage[t], b, rows) if tune else 1.0
        grids[t] = (b, scale_of(backstage[t], b, clip, rows))
    ctx, tgt = make_windows(words)
    rng = np.random.default_rng(seed)
    opt = Adam(lr)
    for it in range(iters):
        staged = dict(backstage)
        for t, (b, s) in grids.items():
            staged[t] = decode(_snap(backstage[t], s, b), s)
        pick = rng.integers(0, len(tgt), batch)
        loss, grads = loss_and_grads(staged, ctx[pick], tgt[pick])
        opt.update(backstage, grads)
        if history is not None and (it % 25 == 0 or it == iters - 1):
            history.append((it, loss))
    return {t: (_snap(backstage[t], s, b), s, b) for t, (b, s) in grids.items()}
