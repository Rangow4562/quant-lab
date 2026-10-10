"""Seven quantization methods beyond the symmetric grid, from the Cast AI guide.

Each function keeps the same contract as gridquant.encode: numbers go in, codes
and the numbers needed to bring them back come out. The activation-aware methods
(AWQ, SmoothQuant, GPTQ) also take calibration inputs, because what they exploit
is the same fact: the error that matters is the one the model feels.
"""
import numpy as np

import calib
import gridquant as gq
import tiny_lang as tl
import tiny_lm as lm


def with_tables(params, tables):
    """A model with some tables replaced; everything else copied."""
    out = {k: v.copy() for k, v in params.items()}
    out.update(tables)
    return out


# ---------------------------------------------------------------- asymmetric

def encode_asym(w, bits, per_row=False):
    """Numbers -> unsigned codes on a grid stretched over [min, max], plus scale and zero point.

    A table whose numbers all live on one side of zero wastes half the symmetric
    grid on empty space. The zero point Z says which code lands on zero, so the
    grid can start wherever the numbers start — and at b bits there are 2**b
    codes instead of 2**b - 1."""
    top = 2 ** bits - 1
    axis = 1 if per_row else None
    wmin = w.min(axis=axis, keepdims=True)
    wmax = w.max(axis=axis, keepdims=True)
    s = np.where(wmax - wmin == 0, 1.0, (wmax - wmin) / top).astype(np.float32)
    z = np.rint(-wmin / s).astype(np.int16)
    codes = np.clip(np.rint(w / s) + z, 0, top).astype(np.uint8)
    return codes, s, z


def decode_asym(codes, s, z):
    return (codes.astype(np.float32) - z) * s


# ------------------------------------------------------------------- k-means

def kmeans_encode(w, bits, iters=15):
    """Numbers -> the index of their nearest centre, plus the centres themselves.

    Starts from the symmetric grid, then Lloyd's iterations pull each centre to
    the mean of the numbers it owns. Centres stay symmetric around zero: every
    pair (+c, -c) is re-centred on the two mirrored means, and zero keeps its own."""
    m = gq.top_code(bits)
    k = 2 * m + 1
    assign, s = gq.encode(w, bits)
    flat = w.ravel().astype(np.float32)
    assign = assign.ravel()
    centres = np.zeros(k, np.float32)
    for _ in range(iters):
        counts = np.bincount(assign + m, minlength=k)
        mean = np.bincount(assign + m, weights=flat, minlength=k) / np.maximum(counts, 1)
        mean = np.where(counts > 0, mean, (np.arange(k) - m) * float(s))
        centres[m] = mean[m]
        for v in range(1, m + 1):
            c = (mean[m + v] - mean[m - v]) / 2
            centres[m + v], centres[m - v] = c, -c
        d = (flat[:, None] - centres[None, :]) ** 2
        assign = d.argmin(axis=1) - m
    return assign.reshape(w.shape).astype(np.int8), centres


def kmeans_decode(codes, centres):
    m = (len(centres) - 1) // 2
    return centres[codes.astype(np.int64) + m]


# -------------------------------------------------------------- group scales

def encode_grouped(w, bits, group):
    """Numbers -> codes on a grid one step wide per `group` input columns.

    Between one grid for the whole table and one grid per row: each row is sliced
    into blocks of `group` columns and every block gets the step of its own
    entries, so a spike no longer ruins the whole row it lives in."""
    top = gq.top_code(bits)
    rows, cols = w.shape
    g = min(group, cols)
    pad = -cols % g
    wp = np.pad(w, ((0, 0), (0, pad)), mode='edge') if pad else w
    wg = wp.reshape(rows, -1, g)
    s = np.where(np.abs(wg).max(axis=2, keepdims=True) == 0, 1.0,
                 np.abs(wg).max(axis=2, keepdims=True) / top).astype(np.float32)
    codes = np.clip(np.rint(wg / s), -top, top).astype(np.int8)
    return codes, s


def decode_grouped(codes, s, cols):
    return (codes * s).reshape(codes.shape[0], -1)[:, :cols]


# ------------------------------------------------------------------- folding

def fold_all(params, s1, s2, s3):
    """Absorb per-channel scales into the tables; the forward pass is unchanged.

    s1 scales l1's input — one number per embedding feature, shared by all six
    context positions, so it can live inside emb. s2 scales l2's input (l1's
    output), s3 scales l3's input (l2's output). Each layer's input side multiplies
    its own scale and its output side divides the next one's: activations end up
    divided by s, weights multiplied by s, and ReLU does not mind positive scales."""
    p = {k: v.copy() for k, v in params.items()}
    a1 = np.tile(s1, lm.CTX) if s1.size == lm.EMB else s1
    p['emb'] = (p['emb'] / s1).astype(np.float32)
    p['l1'] = (p['l1'] * a1 / s2[:, None]).astype(np.float32)
    p['l1_b'] = (p['l1_b'] / s2).astype(np.float32)
    p['l2'] = (p['l2'] * s2 / s3[:, None]).astype(np.float32)
    p['l2_b'] = (p['l2_b'] / s3).astype(np.float32)
    p['l3'] = (p['l3'] * s3).astype(np.float32)
    return p


def _w_absmax_per_feature(w):
    return np.abs(w).reshape(w.shape[0], lm.CTX, lm.EMB).max(axis=(0, 1))


# ----------------------------------------------------------------------- AWQ

def awq_search(w, x, bits, x_absmax, w_absmax, alphas=None):
    """Search the scale strength alpha; return the per-channel scale it justifies.

    For each alpha the scale s = max|x|^a / max|w|^(1-a) is folded into the weight
    columns, the folded weights are quantized with the ordinary narrowed grids,
    and the drift of the layer's output is measured on the calibration inputs.
    The alpha with the least drift wins — the model itself picks what to protect."""
    if alphas is None:
        alphas = np.linspace(0.0, 0.5, 11)
    target = x @ w.T
    best = None
    for a in alphas:
        s = (np.maximum(x_absmax, 1e-8) ** a) / (np.maximum(w_absmax, 1e-8) ** (1 - a))
        s = (s / s.max()).astype(np.float32)
        s_full = np.tile(s, lm.CTX) if s.size != w.shape[1] else s
        wf = w * s_full
        codes, sc = gq.encode(wf, bits, gq.tune_clip(wf, bits, True), True)
        out = (x / s_full) @ gq.decode(codes, sc).T
        mse = float(((out - target) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, s, float(a))
    return best[1], best[2]


def awq_model(params, bits, cal, alphas=None):
    """Fold an activation-aware scale into every table, then quantize as usual.

    The scales are searched on the original model's activations, folded in once,
    and only then is each table quantized — nothing is quantized twice."""
    s1, a1 = awq_search(params['l1'], cal['l1'], bits,
                        calib.absmax_per_feature(cal['l1']), _w_absmax_per_feature(params['l1']),
                        alphas)
    s2, a2 = awq_search(params['l2'], cal['l2'], bits,
                        calib.absmax_per_channel(cal['l2']), np.abs(params['l2']).max(axis=0),
                        alphas)
    s3, a3 = awq_search(params['l3'], cal['l3'], bits,
                        calib.absmax_per_channel(cal['l3']), np.abs(params['l3']).max(axis=0),
                        alphas)
    folded = fold_all(params, s1, s2, s3)
    tables = {}
    for t in gq.BIG:
        codes, sc = gq.encode(folded[t], bits, gq.tune_clip(folded[t], bits, True), True)
        tables[t] = gq.decode(codes, sc)
    return with_tables(params, tables), (a1, a2, a3)


# --------------------------------------------------------------- SmoothQuant

def smooth_scale(w, x_absmax, alpha=0.5, w_absmax=None):
    """The SmoothQuant per-channel scale: max|x|^a / max|w|^(1-a).

    One knob, alpha, splits the difficulty between the two sides of the matmul:
    alpha near 1 pushes it all onto the weights, alpha 0 leaves the activations
    as they are. Halfway is the default."""
    if w_absmax is None:
        w_absmax = np.abs(w).max(axis=0)
    s = (np.maximum(x_absmax, 1e-8) ** alpha) / (np.maximum(w_absmax, 1e-8) ** (1 - alpha))
    return (s / s.max()).astype(np.float32)


def smooth_model(params, cal, alpha=0.5):
    """Migrate quantization difficulty from activations into the weights."""
    s1 = smooth_scale(params['l1'], calib.absmax_per_feature(cal['l1']), alpha,
                      _w_absmax_per_feature(params['l1']))
    s2 = smooth_scale(params['l2'], calib.absmax_per_channel(cal['l2']), alpha)
    s3 = smooth_scale(params['l3'], calib.absmax_per_channel(cal['l3']), alpha)
    return fold_all(params, s1, s2, s3)


def act_quant(x, bits=8, static=None):
    """Activations -> int8 codes, one scale for the whole matrix.

    static=None measures the range from x itself, every forward pass (dynamic);
    a fixed amax makes it static, chosen once on calibration text."""
    amax = float(np.abs(x).max()) if static is None else float(static)
    s = amax / gq.top_code(bits)
    s = np.float32(s if s > 0 else 1.0)
    codes = np.clip(np.rint(x / s), -gq.top_code(bits), gq.top_code(bits)).astype(np.int8)
    return codes, s


def int8_matmul(xc, xs, wc, ws, bias=None):
    """int8 x int8 -> int32 accumulation -> one scale multiply back to float."""
    acc = xc.astype(np.int32) @ wc.T.astype(np.int32)
    out = acc * (xs * ws).ravel()
    return out + bias if bias is not None else out


def quantize_w8(params, tables=('l1', 'l2', 'l3')):
    """Folded weights -> per-row int8 codes and scales."""
    return {t: gq.encode(params[t], 8, per_row=True) for t in tables}


def predict_w8a8(w8, params, ctx, static=None):
    """Forward pass where every table multiply runs on int8 codes.

    w8 holds the quantized weights; params supplies emb and the biases. static
    fixes the activation ranges (static quantization), None measures each range
    from the running values (dynamic quantization)."""
    x = params['emb'][ctx].reshape(len(ctx), -1)
    for i, name in enumerate(lm.WEIGHTS):
        xc, xs = act_quant(x, static=None if static is None else static[name])
        wc, ws = w8[name]
        x = int8_matmul(xc, xs, wc, ws, params[name + '_b'])
        if i < len(lm.WEIGHTS) - 1:
            x = np.maximum(x, 0.0)
    return x


def val_loss_w8a8(w8, params, words, static=None, cap=20000):
    ctx, tgt = lm.make_windows(words[:cap])
    p = lm.softmax(predict_w8a8(w8, params, ctx, static))
    return float(-np.log(p[np.arange(len(tgt)), tgt] + 1e-12).mean())


def grammar_rate_w8a8(w8, params, static=None, n=1000, seed=3, cap=24):
    """The yardstick, with the W8A8 forward pass in place of the float one."""
    rng = np.random.default_rng(seed)
    window = [tl.word_index[w] for w in lm.SEED_LINE[-lm.CTX:]]
    good = 0
    for _ in range(n):
        words = []
        while len(words) < cap:
            logits = predict_w8a8(w8, params, np.array([window]), static)
            probs = lm.softmax(logits)[0].astype(np.float64)
            w = int(rng.choice(len(tl.VOCAB), p=probs / probs.sum()))
            words.append(tl.VOCAB[w])
            window = window[1:] + [w]
            if tl.VOCAB[w] == '.':
                break
        good += tl.is_grammatical(words)
    return good


# ---------------------------------------------------------------------- GPTQ

def gptq_layer(w, x, bits, damp=0.01):
    """Quantize one table with GPTQ's error compensation.

    H = 2 X^T X says how much each input channel moves the layer's output. From
    the Cholesky factor of its inverse, columns are taken in the order that loses
    least, and every column's rounding error is pushed into the columns still
    waiting — so the layer's output, not the weights themselves, stays close."""
    W = w.astype(np.float64)
    X = x.astype(np.float64)
    n = W.shape[1]
    H = 2.0 * X.T @ X
    d = np.diag(H).copy()
    d[d == 0] = 1.0
    H[np.diag_indices(n)] = d
    H += np.eye(n) * (damp * d.mean())
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T
    s = gq.scale_of(w, bits, per_row=True)
    top = gq.top_code(bits)
    codes = np.zeros(W.shape, np.int8)
    for i in range(n):
        col = W[:, i:i + 1]
        q = np.clip(np.rint(col / s), -top, top)
        codes[:, i] = q[:, 0].astype(np.int8)
        err = (col - q * s) / Hinv[i, i]
        if i + 1 < n:
            W[:, i + 1:] -= err * Hinv[i, i + 1:]
    return codes, s


def gptq_model(params, words, bits, cap=4096):
    """Quantize every table with GPTQ, feeding each layer the inputs the previous,
    already quantized layers actually produce."""
    p = {k: v.copy() for k, v in params.items()}
    codes, sc = gq.encode(p['emb'], bits, per_row=True)
    p['emb'] = gq.decode(codes, sc)
    cal = calib.layer_inputs(p, words, cap)
    for name in lm.WEIGHTS:
        codes, sc = gptq_layer(p[name], cal[name], bits)
        p[name] = gq.decode(codes, sc)
    return p


# ----------------------------------------------------------------------- FP8

FP8_E4M3 = (4, 3)   # four exponent bits, three mantissa: max 448, no infinity
FP8_E5M2 = (5, 2)   # wider range, coarser steps: max 57344, then infinity


def cast_fp8(x, fmt=FP8_E4M3):
    """Round every number to the nearest value the FP8 format can hold (simulation).

    E4M3 spends four bits on the exponent and three on the mantissa; E5M2 trades
    the other way. Nearest-value rounding with ties to even, subnormals below the
    smallest normal, and past the top: saturation (E4M3) or infinity (E5M2)."""
    e_bits, m_bits = fmt
    bias = 2 ** (e_bits - 1) - 1
    emin = 1 - bias
    exp_top = 2 ** e_bits - 1 if fmt == FP8_E4M3 else 2 ** e_bits - 2
    top_mant = 2.0 - 2.0 ** (1 - m_bits) if fmt == FP8_E4M3 else 2.0 - 2.0 ** -m_bits
    max_normal = 2.0 ** (exp_top - bias) * top_mant
    sign = np.where(np.asarray(x, np.float32) < 0, -1.0, 1.0)
    a = np.abs(np.asarray(x, np.float32)).astype(np.float64)
    normal = a >= 2.0 ** emin
    e = np.frexp(a)[1] - 1
    step = 2.0 ** (emin - m_bits)
    sub = np.rint(a / step) * step
    ulp = 2.0 ** (e - m_bits)
    nor = np.rint(a / ulp) * ulp
    out = np.where(normal, nor, sub)
    if fmt == FP8_E4M3:
        out = np.where(out > max_normal, max_normal, out)
    else:
        out = np.where(out > max_normal, np.inf, out)
    return (out * sign).astype(np.float32)


def cast_fp8_weights(params, fmt):
    """Every big table through the FP8 format; biases stay as they are."""
    return {t: cast_fp8(params[t], fmt) for t in gq.BIG}
