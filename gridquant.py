"""Symmetric grid quantization.

Every number of a table is stored as the integer code of its nearest grid point.
The grid is symmetric around zero and holds `2 ** bits - 1` points: 255 codes at
8 bits, 15 at 4 bits, 3 at 2 bits. One scale per table (or one per row) turns
codes back into ordinary numbers.
"""
import os
from dataclasses import dataclass

import numpy as np

from bitpack import pack_codes, unpack_codes

BIG = ('emb', 'l1', 'l2', 'l3')


def top_code(bits):
    """The code furthest from zero: 127 at 8 bits, 7 at 4, 1 at 2."""
    return 2 ** (bits - 1) - 1


def scale_of(w, bits, clip=1.0, per_row=False):
    """Distance between neighbouring grid points."""
    amax = np.abs(w).max(axis=1, keepdims=True) if per_row else np.abs(w).max()
    return np.asarray(amax * clip / top_code(bits), dtype=np.float32)


def encode(w, bits, clip=1.0, per_row=False):
    """Numbers -> integer codes on the grid, plus the scale that built the grid."""
    s = scale_of(w, bits, clip, per_row)
    codes = np.clip(np.round(w / s), -top_code(bits), top_code(bits)).astype(np.int8)
    return codes, s


def decode(codes, s):
    """Codes and scale -> ordinary numbers again."""
    return codes.astype(np.float32) * s


def roundtrip(w, bits, clip=1.0, per_row=False):
    return decode(*encode(w, bits, clip, per_row))


def tune_clip(w, bits, per_row=False):
    """Try 64 grid widths from 5% to 100% of the biggest entry; keep the one that
    warps the table least (one width per row when per_row)."""
    widths = np.linspace(0.05, 1.0, 64)
    if not per_row:
        errs = [((roundtrip(w, bits, c) - w) ** 2).sum() for c in widths]
        return float(widths[int(np.argmin(errs))])
    errs = np.stack([((roundtrip(w, bits, c, True) - w) ** 2).sum(axis=1) for c in widths])
    return widths[errs.argmin(axis=0)][:, None].astype(np.float32)


@dataclass
class Plan:
    """Which tables get how many bits, and which repairs are on.

    `bits` is one number for all of BIG, or a per-table dictionary in which
    tables left out simply keep their ordinary numbers.
    """
    bits: object
    tune: bool = False
    rows: bool = False

    def for_tensor(self, t):
        b = self.bits.get(t) if isinstance(self.bits, dict) else self.bits
        if b is None:
            return None
        return b, self.tune and b < 8, self.rows and b < 8


def shrink(params, plan):
    """Encode every planned table; tables left out of the plan stay ordinary."""
    out = {}
    for t in BIG:
        cfg = plan.for_tensor(t)
        if cfg is None:
            continue
        b, tune, rows = cfg
        clip = tune_clip(params[t], b, rows) if tune else 1.0
        out[t] = (*encode(params[t], b, clip, rows), b)
    return out


def rebuild(params, shrunk):
    """A runnable model: planned tables decoded, everything else copied."""
    out = {k: v.copy() for k, v in params.items()}
    for t, (codes, s, b) in shrunk.items():
        out[t] = decode(codes, s)
    return out


def compress(params, plan):
    return rebuild(params, shrink(params, plan))


def avg_bits(params, plan):
    """Bits per number across the planned tables."""
    used = [t for t in BIG if plan.for_tensor(t)]
    num = sum(params[t].size * plan.for_tensor(t)[0] for t in used)
    den = sum(params[t].size for t in used)
    return num / den


def save_shrunk(path, params, shrunk):
    """Write the shrunk model to one file; returns the file size in bytes."""
    parts = {k: v for k, v in params.items() if k not in shrunk}
    for t, (codes, s, b) in shrunk.items():
        parts[t + '.codes'] = pack_codes(codes, b)
        parts[t + '.scale'] = np.asarray(s)
        parts[t + '.meta'] = np.array([b, *codes.shape])
    np.savez(path, **parts)
    return os.path.getsize(path)


def load_shrunk(path):
    """Read a shrunk file back into a runnable model."""
    parts = dict(np.load(path))
    params = {}
    for k, v in parts.items():
        if k.endswith('.codes'):
            t = k[:-6]
            b, *shape = parts[t + '.meta'].tolist()
            params[t] = decode(unpack_codes(v, b, shape), parts[t + '.scale'])
        elif '.' not in k:
            params[k] = v
    return params
