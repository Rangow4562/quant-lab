# quant-lab

**Shrink a tiny language model from 32 bits to 2 bits per number — and watch exactly where it
breaks, why it breaks, and which repairs bring it back.**

A language model is a long list of numbers stored at 32-bit floating point. Quantization replaces
every number with a small integer plus one shared scale, cutting the file by an order of magnitude.
This project makes that process fully observable: it trains a 443,162-parameter model on a toy
language, then squeezes it through a measured pipeline of experiments, each judged by the same
yardstick — *write 1,000 sentences, count how many exactly match the grammar.*

Pure Python 3 + numpy. No GPU, no frameworks, no black boxes. About 1,100 lines, every line
readable — including working versions of GPTQ, AWQ, SmoothQuant and FP8.

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [The model](#the-model)
- [The toy language and the yardstick](#the-toy-language-and-the-yardstick)
- [Grid quantization in one page](#grid-quantization-in-one-page)
- [Why 3 bits breaks](#why-3-bits-breaks)
- [The four repairs](#the-four-repairs)
- [The Cast AI guide, method by method](#the-cast-ai-guide-method-by-method)
- [The experiments](#the-experiments)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Using it as a library](#using-it-as-a-library)
- [Repository layout](#repository-layout)
- [Design notes and limits](#design-notes-and-limits)
- [Extending it](#extending-it)
- [Verification and reproducibility](#verification-and-reproducibility)
- [Credits and license](#credits-and-license)

## Results at a glance

| Precision | Size | Grammatical of 1,000 | Produced by |
| --- | --- | --- | --- |
| 32 bits (baseline) | 1.77 MB | 988 | `baseline` |
| 8 bits | 0.45 MB | 987 | `bits8` |
| 6 bits | 0.34 MB | 996 | `lowbits` |
| 5 bits | 0.28 MB | 984 | `lowbits` |
| 4 bits | 0.23 MB | 874 | `lowbits` |
| 3 bits, one grid per table | 0.17 MB | **4** | `lowbits` |
| 2 bits, one grid per table | 0.12 MB | **0** | `lowbits` |
| 3 bits, narrowed grids | — | 516 | `clip` |
| 3 bits, one grid per row | — | 735 | `rows` |
| 3 bits, per row + narrowed | — | 794 | `rows` |
| 3 bits, mixed precision | 0.20 MB | 973 | `mixed` |
| 2 bits, all repairs + retraining | 0.12 MB | **992** | `qat` |
| 3 bits, asymmetric grid, per row | — | 954 | `asym` |
| 3 bits, k-means centres | — | 779 | `kmeans` |
| 3 bits, one grid per 32 columns | — | 981 | `groups` |
| 3 bits, AWQ fold + narrowed grids | — | 941 | `awq` |
| 3 bits, GPTQ error compensation | — | **986** | `gptq` |
| 8-bit weights, int8 inference (SmoothQuant) | — | 991 | `smooth` |
| FP8 E4M3 weights (simulated) | — | 989 | `fp8` |

The endpoint: **0.12 MB — 14.4x smaller than the 32-bit file — while writing better sentences
than the baseline.** The interesting part is the middle of the table: where quality collapses,
why it collapses, and how each repair buys back grammar at almost no size cost. The rows below
the line come from the second half of the suite: the same yardstick applied to the quantization
methods the LLM era actually uses, each measured against its own symmetric baseline.

## The model

A three-layer MLP that reads the last 6 words and scores all 58 words of the vocabulary for what
comes next.

```
the wizard finds the golden map .        six words of context
        |
      [emb]      one row per word: 6 x 48 = 288 numbers
        |
      [ l1 ]     288 -> 512, then ReLU
        |
      [ l2 ]     512 -> 512, then ReLU
        |
      [ l3 ]     512 -> 58
        |
scores for every word that could come next
```

| Table | Shape | Numbers | Role |
| --- | --- | --- | --- |
| `emb` | 58 x 48 | 2,784 | one 48-number row per vocabulary word |
| `l1` | 512 x 288 | 147,456 | context to hidden |
| `l2` | 512 x 512 | 262,144 | hidden to hidden |
| `l3` | 58 x 512 | 29,696 | hidden to word scores |
| biases | — | 1,082 | one per neuron |
| **total** | | **443,162** | 1.77 MB at 32 bits |

Training: 120,000 grammar-generated sentences (1,133,412 words), Adam, learning rate 2e-3,
batch 256, 4,000 steps, seed 0. Validation loss 1.259 against 4.060 for blind guessing.

## The toy language and the yardstick

58 words across six sentence shapes:

```
a wizard and a dragon guards the ring .
near the cliff , the lost wizard builds a hidden compass .
the map is golden and the island is hidden .
```

`tiny_lang.py` both *generates* the corpus and *validates* written sentences by compiling each
shape into a regular expression. A sentence is grammatical only on an exact match — no human
judgment, no fuzzy scoring, which is what makes every number in this README reproducible.

## Grid quantization in one page

Pick a step size `s` and keep only multiples of it. A weight `w` becomes the integer code
`round(w / s)`; the code turns back into `code * s`. With `b` bits there are `2**(b-1) - 1`
codes on each side of zero:

| bits | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| codes per side | 127 | 63 | 31 | 15 | 7 | 3 | 1 |

The step size comes from the largest absolute value in the table: `s = max|w| * clip / top_code`.
Two facts drive everything that follows:

- the error of any single number is at most **half a step** (`s / 2`), asserted by a self-check;
- one grid per table means **one step size for thousands of very different numbers**.

At 8 bits the step is fine enough that rounding is nearly free (988 -> 987). At 3 bits it is not.

## Why 3 bits breaks

The tables are far from uniform. A few huge entries stretch the grid so far that the typical
entries — hundreds of times more numerous — collapse onto the zero code:

| Table | biggest / typical | Codes on zero at 3 bits | Squeezed alone, grammatical |
| --- | --- | --- | --- |
| `emb` | 5.1x | 49.7% | 983 of 1,000 |
| `l1` | 6.7x | 62.8% | 990 of 1,000 |
| `l2` | 13.3x | 92.1% | 982 of 1,000 |
| `l3` | 32.7x | **98.1%** | **12 of 1,000** |

The `why` run squeezes one table at a time and isolates the culprit: `l3`, the output table,
whose single large entry turns 98% of its weights into zeros. The `clip` and `rows` runs attack
exactly this, and the `mixed` run sidesteps it by leaving fragile tables at 8 bits.

## The four repairs

### 1. Narrow the grid

Cut the grid to a fraction of the biggest entry. The few outliers take the last code; the
typical entries get a much finer step. The best width is found by searching 64 candidates and
minimizing squared error. Effect at 3 bits: **4 -> 516** of 1,000.

### 2. One grid per row

Rows of the same table live on different scales (measured spread: 2.9x for `l2`). Give each row
its own step size — the price is one extra scale number per row, 1,140 numbers for this model.
Effect at 3 bits: **735**, and **794** with each row's grid also narrowed.

### 3. Mixed precision

Tables are not equally fragile, so they should not share a precision. Keep the small, fragile
tables (`emb`, `l3`) at 8 bits and squeeze the big hidden tables (`l1`, `l2`) to 3 bits with all
repairs: **3.37 bits per number on average, 0.20 MB, 973 of 1,000**, read back from the file and
verified to reproduce the in-memory model exactly.

### 4. Retrain with rounding in the loop

At 2 bits only three codes survive (-1, 0, +1) and no static trick helps: **1 of 1,000**.
The fix is quantization-aware training. Full-precision weights stay backstage; the grid scales
are frozen before training; every step, the backstage numbers are snapped onto their grids, the
snapped model runs, and its gradients nudge the backstage copy. After 2,000 steps the weights
arrange themselves so three values per table are enough: **992 of 1,000** — better than the
32-bit baseline. The `l2` distribution settles at -1: 27%, 0: 45%, +1: 28%.

## The Cast AI guide, method by method

The guide that inspired this second half distinguishes the classical tricks (K-means, linear
grids) from the LLM-era methods (GPTQ, AWQ, SmoothQuant, FP8). All of them now live in
`methods.py`, all measured by the same 1,000-sentence yardstick, most of them at the same
3-bit cliff where the original story bottoms out:

- **Asymmetric (zero-point) quantization** — the grid no longer has to sit symmetric around
  zero; a zero point says which code lands on zero, and `b` bits hold `2**b` codes instead of
  `2**b - 1`. At 3 bits with one grid per row: **954** of 1,000 (the symmetric pipeline's best
  static repair is 794). At 2 bits: 95 — the extra code alone is not enough.
- **K-means quantization** — Deep Compression's move: start from the grid, then let Lloyd's
  iterations pull each centre to the mean of the numbers it owns, pairs mirrored around zero.
  At 3 bits: **779** vs 516 for the narrowed grid. At 2 bits only three symmetric centres
  survive, and no static placement helps: 3 of 1,000 (that is what QAT is for).
- **Group-wise scales** — the GGML k-quant idea (Q4_K_M and friends): one grid per block of
  input columns, between per-table and per-row. Per 32 columns at 3 bits: **981** of 1,000,
  with 13,844 extra scale numbers.
- **GPTQ** — W4A16 with error compensation. The Hessian H = 2 X^T X, measured on calibration
  text, says how much each input channel moves the layer's output; columns are quantized in
  the order that loses least and each column's rounding error is pushed into the columns
  still waiting. At 3 bits: **986** of 1,000 — within two of the 32-bit baseline, where
  round-to-nearest per row manages 735. At 2 bits: 231 vs 0.
- **AWQ** — also W4A16, but the scales come from the activations: per channel,
  s = max|x|^a / max|w|^(1-a), with a searched per layer so the quantized layer's output
  drifts least on calibration text. The fold leaves the forward pass unchanged (asserted by
  a self-check), so only the quantization error changes. At 3 bits: 794 -> **941**.
- **SmoothQuant** — W8A8. The same fold, but with the difficulty migrated wholesale from
  activations into weights; then *every* table multiply runs on int8 codes with int32
  accumulation — real integer inference, not just smaller files. Dynamic per-tensor activation
  ranges give 986 grammatical sentences but a broken 4.06 loss: a handful of outlier windows
  steals the range and flattens everyone else — the guide's activation-outlier story, live.
  Static ranges (chosen once on calibration text, clipping what they clip) give **991** of
  1,000, loss 1.257.
- **FP8** — E4M3 and E5M2, simulated as nearest-value rounding in float32 (ties to even,
  subnormals, saturation or infinity at the top). The floating-point idea at 8 bits keeps
  dynamic range that int8 has to clip: E4M3 weights give **989** of 1,000 and E5M2 992, both
  at the 32-bit baseline's loss — better than the 8-bit integer grid's 987.

What is deliberately out of scope, following the guide's own distinctions: **GGUF is a file
format, not an algorithm** — this project's npz container plays that role, and the block-quant
idea inside GGML is the group-wise experiment above. **KV-cache quantization** needs attention,
and this model is a three-layer MLP with no attention to cache.

## The experiments

| Run | Question | Answer |
| --- | --- | --- |
| `baseline` | What are we squeezing? | 1.77 MB, loss 1.259, 988 of 1,000. |
| `bits8` | What does an 8-bit grid cost? | 0.45 MB, 987 — 4x smaller, almost free. |
| `lowbits` | Where does it break? | 6 bits 996, 5 bits 984, 4 bits 874, 3 bits 4, 2 bits 0. |
| `why` | Why does 3 bits break? | Outliers stretch the grid; `l3` alone drops to 12 of 1,000. |
| `clip` | Repair 1 | Narrowed grids: 516 of 1,000. |
| `rows` | Repair 2 | Grid per row: 735; also narrowed: 794. |
| `mixed` | Repair 3 | 3.37 bits on average, 0.20 MB, 973 from the file. |
| `qat` | Repair 4 | 2 bits: 1 -> 992 after retraining; 0.12 MB, 14.4x smaller. |
| `asym` | What does a zero point buy? | 3 bits per row: 954 (symmetric best: 794); 2 bits: 95. |
| `kmeans` | Where should centres live? | Where the numbers are: 779 at 3 bits vs 516 for the narrowed grid. |
| `groups` | How small can a scale group be? | Per 32 columns at 3 bits: 981; per 64: 967; per row: 794. |
| `awq` | Which channels matter? | The ones with big activations: 794 -> 941 at 3 bits, 1 -> 168 at 2. |
| `gptq` | Where should rounding errors go? | Into the columns still waiting: 986 at 3 bits, 231 at 2, vs 735/0. |
| `smooth` | Can every multiply run on int8? | Yes: 991 of 1,000 with static ranges, 986 with dynamic (outliers steal the range). |
| `fp8` | What if 8 bits stayed floating point? | E4M3: 989, E5M2: 992 — better than the 8-bit integer grid's 987. |

`python experiments.py all` runs them in narrative order and closes with a recap of the story.

## Installation

Requirements: Python 3.10+ and numpy >= 1.26. CPU only.

```bash
# as an installable package (recommended) - provides the quant-lab CLI
pip install -e .

# or just the dependency, and run the scripts directly
pip install numpy
```

## Quick start

```bash
# one command, with environment checks and colored output
./run_all.sh            # Linux/macOS
run_all.bat             # Windows (double-click friendly)

# or through the installed CLI
quant-lab-check         # thirteen property checks of the building blocks
quant-lab all           # the full story
quant-lab qat           # any single run: baseline bits8 lowbits why clip rows mixed qat asym kmeans groups awq gptq smooth fp8
```

The first run trains the model (~1 minute) and caches it in `assets/base.npz`; every later run
reuses it. All artifacts land in `assets/`, which is git-ignored.

## Using it as a library

```python
import gridquant as gq
import tiny_lm as lm

params = lm.get_trained()                    # trained model, cached after the first call

recipe = gq.Plan({'emb': 8, 'l1': 3, 'l2': 3, 'l3': 8}, tune=True, rows=True)
small = gq.compress(params, recipe)          # a runnable model at 3.37 bits on average
print(lm.grammar_rate(small))                # 973 of 1,000

shrunk = gq.shrink(params, recipe)           # codes + scales + bits per table
size = gq.save_shrunk('model.npz', params, shrunk)
print(lm.grammar_rate(gq.load_shrunk('model.npz')))   # identical to `small`

two_bit = qat.qat_train(params, lm.datasets()[0], gq.Plan(2, tune=True, rows=True),
                        iters=2000, lr=3e-4)

import methods as mt, calib

cal = calib.layer_inputs(params, unseen_words)      # what flows into each table
model, alphas = mt.awq_model(params, 3, cal)        # activation-aware fold, then quantize
print(lm.grammar_rate(model))                       # 941 of 1,000 at 3 bits
print(lm.grammar_rate(mt.gptq_model(params, unseen_words, 3)))   # 986 of 1,000

folded = mt.smooth_model(params, cal)               # SmoothQuant, then int8 all the way
w8 = mt.quantize_w8(folded)
print(mt.grammar_rate_w8a8(w8, folded, static={'l1': 1.045, 'l2': 4.193, 'l3': 21.367}))
```

A `Plan` is the entire recipe: how many bits per table, whether to search the grid width
(`tune`), and whether each row gets its own grid (`rows`). Tables left out of a per-table
`Plan` keep their ordinary numbers. The methods in `methods.py` take the trained model plus
calibration inputs and hand back a runnable model (or, for W8A8, int8 codes and a forward
pass that consumes them).

## Repository layout

```
quant-lab/
├── pyproject.toml        package metadata, dependencies, CLI entry points
├── README.md             this document
├── LICENSE               MIT
├── run_all.sh            one command for everything (Linux/macOS)
├── run_all.bat           the same on Windows
├── tiny_lang.py          the toy language: 58 words, six shapes, grammar checker
├── tiny_lm.py            the model: forward/backward, Adam, training, sampling, the yardstick
├── gridquant.py          grids, scales, encode/decode, width search, Plan, file save/load
├── bitpack.py            integer codes -> tightly packed bytes -> codes
├── calib.py              calibration: what flows into each table when the model runs
├── methods.py            the Cast AI guide, in code: asym, k-means, groups, AWQ, SmoothQuant,
│                         GPTQ, FP8 — plus the int8 forward pass
├── qat.py                training with the rounding inside the loop
├── experiments.py        the fifteen runs and the closing recap
├── selfcheck.py          thirteen property checks of the parts
└── assets/               base.npz and every saved artifact (created on first run)
```

## Design notes and limits

- **Disk savings first, compute second.** The original pipeline unpacks codes back to 32-bit
  floats before running. The one exception is the `smooth` run, whose W8A8 forward pass really
  accumulates in int32 — the small model does not get faster, but the machinery is honest.
- **The original pipeline stays symmetric.** `gridquant.py` is untouched: no zero point, no
  group scales. The asymmetric, k-means, group-wise, activation-aware and FP8 methods live in
  `methods.py` and are measured in memory rather than through the file pipeline.
- **Calibration text is fresh, not training text** (`experiments._calibration`), so the
  activation-aware methods are judged on data they have not seen.
- **The toy language keeps the signal clean.** With a 58-word grammar, "grammatical" is exact and
  cheap to measure; a real model would report perplexity, and practitioners usually stop at
  4 bits rather than 2.
- **npz is a zip container**, so file sizes include a small overhead over the raw bit counts.

## Extending it

Good next experiments, roughly in order of payoff:

1. **Entropy coding of codes** — the 2-bit `l2` histogram is far from uniform.
2. **Narrowed group grids**: the group-wise experiment uses plain absmax; searching each
   group's width the way `tune_clip` searches rows should buy more.
3. **GPTQ with activation ordering and clipping** — the guide's full recipe reorders columns by
   Hessian diagonal and re-fits scales per block; here only the error feedback is implemented.
4. **A GGUF-style container** — write the codes, scales and tokenizer into one portable file
   with the block layouts Q4_K_M uses.

(Implemented from the earlier version of this list: zero-point asymmetric quantization,
data-calibrated activation-aware scales, group-wise scales, GPTQ-style error compensation,
and integer inference.)

## Verification and reproducibility

`selfcheck.py` asserts the properties everything else relies on:

- `decode(encode(w))` never strays more than half a grid step from `w`, at 8/4/3/2 bits;
- bit packing is exact and achieves `ceil(n * bits / 8)` bytes for widths 8 down to 2, signed
  and unsigned;
- with one huge outlier, the width search always finds a narrowed grid that beats the full grid;
- a saved file reproduces the in-memory shrunk model bit for bit;
- 2 bits means exactly the codes {-1, 0, +1};
- the asymmetric grid hugs a skewed table tighter than the symmetric one, at half a step;
- k-means centres stay exactly symmetric and beat the grid on bimodal numbers;
- group scales beat per-row scales when the spike moves inside the row;
- folding scales into the weights leaves the forward pass unchanged (AWQ and SmoothQuant both
  ride on this);
- GPTQ's compensated reconstruction beats round-to-nearest on the same calibration inputs;
- the int8 matmul tracks the float one inside the quantization step bound;
- FP8 casting is idempotent, saturates at 448 on E4M3 and reaches infinity on E5M2.

Every random choice — training, sampling, evaluation — is seeded, and the evaluation seed is
fixed, so every number in this README regenerates identically: `python selfcheck.py`, then
`python experiments.py all`.

## Credits and license

An independent implementation of a classic quantization walkthrough, inspired by
Vuk Rosic's *Quantization from scratch* (MIT licensed). The second half of the suite —
asymmetric, k-means, group-wise, AWQ, SmoothQuant, GPTQ and FP8 — follows the method map in
Cast AI's *Demystifying Quantizations: Guide to Quantization Methods for LLMs*. The toy
language, the model, the code, and the explanations here were written fresh for this project.

Released under the [MIT License](LICENSE).
