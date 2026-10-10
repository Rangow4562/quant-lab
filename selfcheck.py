"""Self-checks for the parts. Run:  python selfcheck.py"""
import os
import tempfile

import numpy as np

import gridquant as gq
import methods as mt
import tiny_lm as lm
import tiny_lang as tl
from bitpack import pack_codes, pack_unsigned, unpack_codes, unpack_unsigned


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


def test_unsigned_pack_roundtrip_is_exact():
    rng = np.random.default_rng(5)
    for bits in (8, 4, 3, 2):
        codes = rng.integers(0, 2 ** bits, (7, 13)).astype(np.uint8)
        packed = pack_unsigned(codes, bits)
        assert len(packed) == -(-codes.size * bits // 8)
        assert np.array_equal(unpack_unsigned(packed, bits, codes.shape), codes)


def test_asym_grid_hugs_a_skewed_table():
    w = np.random.default_rng(6).lognormal(0.0, 1.0, (10, 40)).astype(np.float32)
    codes, s, z = mt.encode_asym(w, 3)
    back = mt.decode_asym(codes, s, z)
    assert back.min() >= 0
    assert np.abs(back - w).max() <= float(s.reshape(-1)[0]) / 2 + 1e-5
    assert ((back - w) ** 2).sum() < ((gq.roundtrip(w, 3) - w) ** 2).sum()


def test_kmeans_centres_stay_symmetric_and_beat_the_grid():
    rng = np.random.default_rng(7)
    w = np.concatenate([rng.normal(-0.8, 0.08, 300), rng.normal(0.8, 0.08, 300)])
    w = w.astype(np.float32).reshape(1, 600)
    codes, centres = mt.kmeans_encode(w, 3)
    assert np.allclose(centres, -centres[::-1])
    back = mt.kmeans_decode(codes, centres)
    assert ((back - w) ** 2).sum() < ((gq.roundtrip(w, 3) - w) ** 2).sum()


def test_grouped_grid_beats_per_row_when_spikes_move():
    rng = np.random.default_rng(8)
    w = rng.normal(0, 0.05, (4, 64)).astype(np.float32)
    w[np.arange(4), 8 * np.arange(4) + 4] += 3.0
    back = mt.decode_grouped(*mt.encode_grouped(w, 3, 8), 64)
    per_row = gq.roundtrip(w, 3, per_row=True)
    assert ((back - w) ** 2).sum() < ((per_row - w) ** 2).sum()


def test_fold_leaves_the_model_unchanged():
    params = lm.init_params(3)
    rng = np.random.default_rng(9)
    s1 = rng.uniform(0.5, 2.0, lm.EMB).astype(np.float32)
    s2 = rng.uniform(0.5, 2.0, lm.HID).astype(np.float32)
    s3 = rng.uniform(0.5, 2.0, lm.HID).astype(np.float32)
    folded = mt.fold_all(params, s1, s2, s3)
    ctx, _ = lm.make_windows(tl.make_corpus(300, 0))
    a, _ = lm.predict(params, ctx)
    b, _ = lm.predict(folded, ctx)
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_gptq_beats_round_to_nearest():
    rng = np.random.default_rng(10)
    w = rng.normal(0, 0.5, (32, 64)).astype(np.float32)
    x = rng.normal(0, 1, (256, 64)).astype(np.float32)
    wg = gq.decode(*mt.gptq_layer(w, x, 3))
    wr = gq.roundtrip(w, 3, per_row=True)
    target = x @ w.T
    assert ((x @ wg.T - target) ** 2).sum() < ((x @ wr.T - target) ** 2).sum()


def test_int8_matmul_tracks_the_float_one():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, (8, 16)).astype(np.float32)
    w = rng.normal(0, 0.5, (12, 16)).astype(np.float32)
    xc, xs = mt.act_quant(x)
    wc, ws = gq.encode(w, 8, per_row=True)
    out8 = mt.int8_matmul(xc, xs, wc, ws)
    outq = gq.decode(xc, xs) @ gq.decode(wc, ws).T
    assert np.allclose(out8, outq, rtol=1e-4, atol=1e-5)
    bound = ((float(xs) / 2) * np.abs(w).sum(axis=1).max()
             + (float(ws.max()) / 2) * np.abs(x).sum(axis=1).max()
             + x.shape[1] * float(xs) * float(ws.max()) / 4)
    assert np.abs(out8 - x @ w.T).max() <= bound + 1e-6


def test_fp8_cast_is_idempotent_and_saturates():
    rng = np.random.default_rng(12)
    x = rng.normal(0, 1, (50, 50)).astype(np.float32)
    for fmt in (mt.FP8_E4M3, mt.FP8_E5M2):
        once = mt.cast_fp8(x, fmt)
        assert np.array_equal(once, mt.cast_fp8(once, fmt))
    assert float(mt.cast_fp8(np.float32(1e6), mt.FP8_E4M3)) == 448.0
    assert np.isinf(mt.cast_fp8(np.float32(1e6), mt.FP8_E5M2))
    assert np.isfinite(mt.cast_fp8(x, mt.FP8_E4M3)).all()


def main():
    checks = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for check in checks:
        check()
        print('ok  ', check.__name__)
    print(len(checks), 'checks passed')


if __name__ == '__main__':
    main()
