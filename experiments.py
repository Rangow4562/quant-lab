"""The story in fifteen runs: measure the model, shrink it, watch it break, repair it,
retrain it — then walk the Cast AI guide, method by method.

Usage:  python experiments.py <name>
        python experiments.py all

Runs: baseline, bits8, lowbits, why, clip, rows, mixed, qat,
      asym, kmeans, groups, awq, gptq, smooth, fp8
"""
import argparse
import os
import time

import numpy as np

import calib
import gridquant as gq
import methods as mt
import qat
import tiny_lang as tl
import tiny_lm as lm
from gridquant import Plan

BANNER = '=' * 68


def banner(name):
    print(f'{BANNER}\n{name}\n{BANNER}')


def exp_baseline():
    params = lm.get_trained()
    train_text, test_text = lm.datasets()
    print(f'trained on {len(train_text):,} words of the toy language')
    print('the tables inside the model:')
    for name, a in params.items():
        print(f'  {name:<6} {" x ".join(map(str, a.shape)):>12}  {a.size:>9,}')
    print(f'  total {lm.count(params):>16,}   stored at 32 bits = 4 bytes each')
    path = os.path.join(lm.ASSETS, 'model_fp32.npz')
    np.savez(path, **params)
    size = os.path.getsize(path)
    print(f'file: {size:,} bytes = {size / 1e6:.2f} MB')
    print(f'loss on unseen text: {lm.val_loss(params, test_text):.3f}   '
          f'(blind guessing among {len(tl.VOCAB)} words: {np.log(len(tl.VOCAB)):.3f})')
    for s in lm.say(params, 4):
        print('  ', s)
    good = lm.grammar_rate(params)
    print(f'1,000 written sentences: {good} grammatical')
    return f'{size / 1e6:.2f} MB at 32 bits, {good} of 1,000 grammatical'


def exp_bits8():
    params = lm.get_trained()
    w = params['l2']
    codes, s = gq.encode(w, 8)
    back = gq.decode(codes, s)
    print(f'l2: biggest entry {np.abs(w).max():.4f}, top code {gq.top_code(8)}, grid step {float(s):.6f}')
    print('  number        code   comes back as    error')
    for i in range(6):
        a, c, b = float(w[0, i]), int(codes[0, i]), float(back[0, i])
        print(f'  {a:+.6f}   {c:+5d}   {b:+.6f}    {abs(a - b):.6f}')
    worst = float(np.abs(w - back).max())
    print(f'largest error in the table: {worst:.6f}   (half a grid step is {float(s) / 2:.6f})')
    shrunk = gq.shrink(params, Plan(8))
    path = os.path.join(lm.ASSETS, 'model_8bit.npz')
    size = gq.save_shrunk(path, params, shrunk)
    model = gq.load_shrunk(path)
    good = lm.grammar_rate(model)
    print(f'whole model at 8 bits: {size:,} bytes = {size / 1e6:.2f} MB, {good} of 1,000 grammatical')
    return f'{size / 1e6:.2f} MB at 8 bits, {good} of 1,000 grammatical'


def exp_lowbits():
    params = lm.get_trained()
    test_text = lm.datasets()[1]
    print('bits codes    MB    loss  ok/1000   one written sentence')
    rows = []
    for bits in (8, 6, 5, 4, 3, 2):
        shrunk = gq.shrink(params, Plan(bits))
        path = os.path.join(lm.ASSETS, f'model_{bits}bit.npz')
        size = gq.save_shrunk(path, params, shrunk)
        model = gq.load_shrunk(path)
        good = lm.grammar_rate(model)
        loss = lm.val_loss(model, test_text)
        text = lm.say(model, 1, seed=7)[0]
        rows.append((bits, good))
        print(f'{bits:>4} {2 * gq.top_code(bits) + 1:>5} {size / 1e6:>6.2f}  {loss:.3f}  {good:>6}    {text[:56]}')
    return '  '.join(f'{b} bits {g}' for b, g in rows)


def exp_why():
    params = lm.get_trained()
    bits = 3
    print(f'at {bits} bits ({2 * gq.top_code(bits) + 1} codes), one grid for the whole table:')
    print('table  typical size  biggest  biggest/typ  grid step  codes on zero')
    alone = {}
    for t in gq.BIG:
        w = params[t]
        codes, s = gq.encode(w, bits)
        typical, big = float(np.abs(w).mean()), float(np.abs(w).max())
        print(f'{t:<6} {typical:>10.4f} {big:>9.4f} {big / typical:>9.1f} {float(s):>10.4f} '
              f'{(codes == 0).mean():>12.1%}')
    print('shrink one table at a time, leave the other three alone:')
    for t in gq.BIG:
        alone[t] = lm.grammar_rate(gq.compress(params, Plan({t: bits})))
        print(f'  only {t:<4} at {bits} bits: {alone[t]:>4} of 1,000 grammatical')
    fragile = min(alone, key=alone.get)
    return f'squeezed alone at {bits} bits, {fragile} is the fragile one: {alone[fragile]} of 1,000'


def exp_clip():
    params = lm.get_trained()
    bits = 3
    for t in gq.BIG:
        w = params[t]
        clip = gq.tune_clip(w, bits)
        codes, s = gq.encode(w, bits, clip)
        beyond = float((np.abs(w) > gq.top_code(bits) * s).mean())
        print(f'{t:<4}: best grid width {clip:.0%} of the biggest entry; {beyond:.2%} of entries lie '
              f'beyond it; {(codes == 0).mean():.1%} land on zero')
    good = lm.grammar_rate(gq.compress(params, Plan(bits, tune=True)))
    print(f'{bits} bits with narrowed grids: {good} of 1,000 grammatical')
    return f'{good} of 1,000 with narrowed grids'


def exp_rows():
    params = lm.get_trained()
    bits = 3
    extra = 0
    for t in gq.BIG:
        reach = np.abs(params[t]).max(axis=1)
        extra += len(reach)
        print(f'{t:<4}: biggest entry of a row across {len(reach)} rows: smallest {reach.min():.3f}, '
              f'middle {np.median(reach):.3f}, largest {reach.max():.3f}')
    print(f'price of a grid per row: {extra:,} extra scale numbers')
    for label, plan in (('one grid per row', Plan(bits, rows=True)),
                        ('one grid per row, each narrowed', Plan(bits, tune=True, rows=True))):
        good = lm.grammar_rate(gq.compress(params, plan))
        print(f'{bits} bits, {label}: {good} of 1,000 grammatical')
        last = good
    return f'{last} of 1,000 with a grid per row, each narrowed'


def exp_mixed():
    params = lm.get_trained()
    plan = Plan({'emb': 8, 'l1': 3, 'l2': 3, 'l3': 8}, tune=True, rows=True)
    shrunk = gq.shrink(params, plan)
    path = os.path.join(lm.ASSETS, 'model_mixed.npz')
    size = gq.save_shrunk(path, params, shrunk)
    model = gq.load_shrunk(path)
    same = all(np.array_equal(model[k], v) for k, v in gq.rebuild(params, shrunk).items())
    fp32 = os.path.getsize(os.path.join(lm.ASSETS, 'model_fp32.npz'))
    avg = gq.avg_bits(params, plan)
    print(f'l1 and l2 at 3 bits, emb and l3 at 8 bits: {avg:.2f} bits per number on average')
    print(f'file {size:,} bytes = {size / 1e6:.2f} MB, {fp32 / size:.1f} times smaller than the 32-bit file')
    good = lm.grammar_rate(model)
    print(f'read back from the file: {good} of 1,000 grammatical   (file reproduces the model: {same})')
    return f'{avg:.2f} bits on average, {size / 1e6:.2f} MB, {good} of 1,000 from the file'


def exp_qat(iters=2000, lr=3e-4):
    params = lm.get_trained()
    train_text, test_text = lm.datasets()
    plan = Plan(2, tune=True, rows=True)
    before = lm.grammar_rate(gq.compress(params, plan))
    print(f'every table at 2 bits, all repairs on, no retraining: {before} of 1,000 grammatical')
    t0 = time.time()
    shrunk = qat.qat_train(params, train_text, plan, iters, lr)
    path = os.path.join(lm.ASSETS, 'model_2bit_qat.npz')
    size = gq.save_shrunk(path, params, shrunk)
    model = gq.load_shrunk(path)
    good = lm.grammar_rate(model)
    print(f'after {iters:,} training steps with rounding in the loop ({time.time() - t0:.0f} s): '
          f'{good} of 1,000 grammatical, loss {lm.val_loss(model, test_text):.3f}')
    codes = shrunk['l2'][0]
    hist = {v: float((codes == v).mean()) for v in (-1, 0, 1)}
    print('  l2 now stores only three values: ' + ', '.join(f'{v}: {c:.0%}' for v, c in hist.items()))
    fp32 = os.path.getsize(os.path.join(lm.ASSETS, 'model_fp32.npz'))
    print(f'  file {size:,} bytes = {size / 1e6:.2f} MB, {fp32 / size:.1f} times smaller than the 32-bit file')
    return f'{before} -> {good} of 1,000 at 2 bits, {size / 1e6:.2f} MB, {fp32 / size:.1f}x smaller'


def _calibration(params, cap=4096):
    """Fresh, unseen text for the activation-aware methods to calibrate on."""
    return calib.layer_inputs(params, tl.make_corpus(6000, 5), cap)


def exp_asym():
    params = lm.get_trained()
    print('one grid per table, symmetric (codes both sides of zero) vs asymmetric (zero point):')
    last = None
    for bits in (3, 2):
        rows = {
            'symmetric': lm.grammar_rate(gq.compress(params, Plan(bits))),
            'symmetric, narrowed': lm.grammar_rate(gq.compress(params, Plan(bits, tune=True))),
        }
        for label, per_row in (('asymmetric', False), ('asymmetric, per row', True)):
            tables = {t: mt.decode_asym(*mt.encode_asym(params[t], bits, per_row)) for t in gq.BIG}
            rows[label] = lm.grammar_rate(mt.with_tables(params, tables))
        print(f'{bits} bits: ' + ', '.join(f'{k} {v}' for k, v in rows.items()))
        last = rows
    return (f'asymmetric grid: {last["asymmetric, per row"]} of 1,000 at 2 bits '
            f'({last["asymmetric"]} per table, symmetric {last["symmetric"]})')


def exp_kmeans():
    params = lm.get_trained()
    print('the grid places centres evenly; k-means places them where the numbers are:')
    last = None
    for bits in (3, 2):
        tables = {t: mt.kmeans_decode(*mt.kmeans_encode(params[t], bits)) for t in gq.BIG}
        rows = {
            'grid': lm.grammar_rate(gq.compress(params, Plan(bits))),
            'grid, narrowed': lm.grammar_rate(gq.compress(params, Plan(bits, tune=True))),
            'k-means': lm.grammar_rate(mt.with_tables(params, tables)),
        }
        print(f'{bits} bits: ' + ', '.join(f'{k} {v}' for k, v in rows.items()))
        last = rows
    return f'k-means {last["k-means"]} vs grid {last["grid"]} of 1,000 at 2 bits'


def exp_groups():
    params = lm.get_trained()
    bits = 3
    print(f'at {bits} bits, one grid per how many input columns:')
    print('  whole table, narrowed:  ', lm.grammar_rate(gq.compress(params, Plan(bits, tune=True))))
    print('  one grid per row:       ', lm.grammar_rate(gq.compress(params, Plan(bits, rows=True))))
    print('  per row, each narrowed: ',
          lm.grammar_rate(gq.compress(params, Plan(bits, tune=True, rows=True))))
    last = None
    for group in (128, 64, 32):
        tables = {t: mt.decode_grouped(*mt.encode_grouped(params[t], bits, group), params[t].shape[1])
                  for t in gq.BIG}
        good = lm.grammar_rate(mt.with_tables(params, tables))
        scales = sum(s.size for _, s in (mt.encode_grouped(params[t], bits, group) for t in gq.BIG))
        print(f'  one grid per {group:>3} columns: {good:>4}   ({scales:,} extra scale numbers)')
        last = good
    return f'{last} of 1,000 with one grid per 32 columns'


def exp_awq():
    params = lm.get_trained()
    cal = _calibration(params)
    print('activation-aware scales (AWQ): protect the channels the model actually uses')
    last = None
    for bits in (3, 2):
        base = lm.grammar_rate(gq.compress(params, Plan(bits, tune=True, rows=True)))
        model, alphas = mt.awq_model(params, bits, cal)
        good = lm.grammar_rate(model)
        print(f'{bits} bits: plain narrowed grids {base} of 1,000; with the AWQ fold {good}'
              f'   (alphas found: ' + ', '.join(f'{a:.2f}' for a in alphas) + ')')
        last = good
    return f'{last} of 1,000 with activation-aware scaling'


def exp_gptq():
    params = lm.get_trained()
    words = tl.make_corpus(6000, 5)
    print('round-to-nearest forgets each column\'s error; GPTQ pushes it into the columns '
          'still waiting')
    last = None
    for bits in (3, 2):
        rows = {
            'round-to-nearest': lm.grammar_rate(gq.compress(params, Plan(bits, rows=True))),
            'round-to-nearest, narrowed': lm.grammar_rate(
                gq.compress(params, Plan(bits, tune=True, rows=True))),
            'GPTQ': lm.grammar_rate(mt.gptq_model(params, words, bits)),
        }
        print(f'{bits} bits: ' + ', '.join(f'{k} {v}' for k, v in rows.items()))
        last = rows
    return f'GPTQ {last["GPTQ"]} vs round-to-nearest {last["round-to-nearest"]} of 1,000 at 2 bits'


def exp_smooth():
    params = lm.get_trained()
    train_text = tl.make_corpus(6000, 5)
    cal = _calibration(params)
    folded = mt.smooth_model(params, cal)
    print(f'the fold alone changes nothing: {lm.grammar_rate(folded)} of 1,000 grammatical')
    w8 = mt.quantize_w8(folded)
    static = {name: float(np.abs(x).max())
              for name, x in calib.layer_inputs(folded, train_text).items()}
    print('every table multiply now runs on int8 codes with int32 accumulation (W8A8):')
    rows = {}
    for label, st in (('dynamic activations', None), ('static activations', static)):
        good = mt.grammar_rate_w8a8(w8, folded, st)
        rows[label] = good
        print(f'  {label}: {good} of 1,000 grammatical, '
              f'loss {mt.val_loss_w8a8(w8, folded, train_text, st):.3f}')
    print('  why dynamic loses: a handful of outlier windows push the per-tensor range 250x')
    print('  wide, and every typical window collapses onto zero: the activation-outlier')
    print('  problem from the guide, fixed here by static ranges that clip what they clip.')
    return f'W8A8 int8 inference: {rows["dynamic activations"]} dynamic, {rows["static activations"]} static of 1,000'


def exp_fp8():
    params = lm.get_trained()
    test_text = lm.datasets()[1]
    print('FP8 keeps the floating-point idea at 8 bits (Cast AI: the Hopper-native format):')
    rates = {}
    for label, fmt in (('E4M3 (range, coarse steps)', mt.FP8_E4M3),
                       ('E5M2 (wider range, coarser)', mt.FP8_E5M2)):
        model = mt.with_tables(params, mt.cast_fp8_weights(params, fmt))
        good = lm.grammar_rate(model)
        rates[fmt] = good
        print(f'  {label}: {good} of 1,000 grammatical, loss {lm.val_loss(model, test_text):.3f}')
    print('for comparison: 8-bit grid 987, 4-bit grid 874 of 1,000')
    return (f'{rates[mt.FP8_E4M3]} of 1,000 with FP8 E4M3 weights, '
            f'{rates[mt.FP8_E5M2]} with E5M2')

RUNS = {'baseline': exp_baseline, 'bits8': exp_bits8, 'lowbits': exp_lowbits, 'why': exp_why,
        'clip': exp_clip, 'rows': exp_rows, 'mixed': exp_mixed, 'qat': exp_qat,
        'asym': exp_asym, 'kmeans': exp_kmeans, 'groups': exp_groups, 'awq': exp_awq,
        'gptq': exp_gptq, 'smooth': exp_smooth, 'fp8': exp_fp8}


def main():
    ap = argparse.ArgumentParser(description='Shrink a tiny language model, one experiment at a time.')
    ap.add_argument('run', choices=[*RUNS, 'all'])
    args = ap.parse_args()
    os.makedirs(lm.ASSETS, exist_ok=True)
    todo = RUNS if args.run == 'all' else {args.run: RUNS[args.run]}
    story = []
    for name, fn in todo.items():
        banner(name)
        line = fn()
        if line:
            story.append((name, line))
    if args.run == 'all' and story:
        banner('the story so far')
        width = max(len(name) for name, _ in story)
        for name, line in story:
            print(f'  {name:<{width}}  {line}')


if __name__ == '__main__':
    main()
