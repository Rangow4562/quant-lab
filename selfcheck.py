"""Self-checks for the parts. Run:  python selfcheck.py"""
import os
import tempfile

import numpy as np

import gridquant as gq
import tiny_lm as lm
from bitpack import pack_codes, unpack_codes


def test_roundtrip_error_stays_under_half_a_step():
    w = np.random.default_rng(0).normal(0, 1, (20, 30)).astype(np.float32)
    for bits in (8, 4, 3, 2):
        codes, s = gq.encode(w, bits)
        assert np.abs(codes).max() <= gq.top_code(bits)
        assert np.abs(gq.decode(codes, s) - w).max() <= float(s) / 2 + 1e-6


def test_pack_roundtrip_is_exact():
    rng = np.random.default_rng(1)
    for bits in (8, 6, 5, 4, 3, 2):
        top = gq.top_code(bits)
        codes = rng.integers(-top, top + 1, (7, 13)).astype(np.int8)
        packed = pack_codes(codes, bits)
        assert len(packed) == -(-codes.size * bits // 8)
        assert np.array_equal(unpack_codes(packed, bits, codes.shape), codes)


def test_narrow_grid_wins_when_one_entry_is_huge():
    w = np.random.default_rng(2).normal(0, 0.1, (1, 1000)).astype(np.float32)
    w[0, 0] = 3.0
    clip = gq.tune_clip(w, 3)
    assert clip < 0.5
    assert ((gq.roundtrip(w, 3, clip) - w) ** 2).sum() < ((gq.roundtrip(w, 3) - w) ** 2).sum()


def test_file_reproduces_the_same_model():
    params = lm.init_params(3)
    plan = gq.Plan({'emb': 8, 'l1': 3, 'l2': 2, 'l3': 8}, tune=True, rows=True)
    shrunk = gq.shrink(params, plan)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.npz')
        gq.save_shrunk(path, params, shrunk)
        loaded = gq.load_shrunk(path)
    for k, v in gq.rebuild(params, shrunk).items():
        assert np.array_equal(loaded[k], v), k


def test_two_bits_leaves_three_codes():
    w = np.random.default_rng(4).normal(0, 1, (5, 50)).astype(np.float32)
    codes, _ = gq.encode(w, 2, 0.5)
    assert set(np.unique(codes).tolist()) == {-1, 0, 1}


def main():
    checks = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for check in checks:
        check()
        print('ok  ', check.__name__)
    print(len(checks), 'checks passed')


if __name__ == '__main__':
    main()
