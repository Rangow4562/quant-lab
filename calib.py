"""Calibration data: what the inputs of each table actually look like when the model runs.

AWQ, SmoothQuant and GPTQ all need to see the activations that flow into a layer,
not just its weights. This module runs the model over a slice of the toy language
and hands out those input matrices, one per table.
"""
import numpy as np

import tiny_lm as lm

CALIB_WORDS = 20000


def layer_inputs(params, words=None, cap=4096):
    """Input matrix of every weight table, from a walk over calibration text.

    Returns a dict {table: X} where X has one row per calibration window and one
    column per input channel — the exact matrices the tables multiply.
    """
    if words is None:
        words = lm.datasets()[0][:CALIB_WORDS]
    ctx, _ = lm.make_windows(words[:cap + lm.CTX])
    _, acts = lm.predict(params, ctx)
    return dict(zip(lm.WEIGHTS, acts))


def absmax_per_channel(x):
    """The largest absolute value each input column ever reaches."""
    return np.abs(x).max(axis=0).astype(np.float32)


def absmax_per_feature(x):
    """The same, but shared across the six context positions (for the l1 input).

    l1's input channels are six copies of the 48 embedding features, and a scale
    that differs per position could not be folded into emb — so the six copies
    share one number per feature."""
    return np.abs(x.reshape(len(x), lm.CTX, lm.EMB)).max(axis=(0, 1)).astype(np.float32)
