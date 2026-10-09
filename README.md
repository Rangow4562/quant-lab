# quant-lab

**Shrink a tiny language model from 32 bits to 2 bits per number — and watch exactly where it
breaks, why it breaks, and which four repairs bring it back.**

A language model is a long list of numbers stored at 32-bit floating point. Quantization replaces
every number with a small integer plus one shared scale, cutting the file by an order of magnitude.
This project makes that process fully observable: it trains a 443,162-parameter model on a toy
language, then squeezes it through a measured pipeline of experiments, each judged by the same
yardstick — *write 1,000 sentences, count how many exactly match the grammar.*

Pure Python 3 + numpy. No GPU, no frameworks, no black boxes. About 700 lines, every line readable.

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [The model](#the-model)
- [The toy language and the yardstick](#the-toy-language-and-the-yardstick)
- [Grid quantization in one page](#grid-quantization-in-one-page)
- [Why 3 bits breaks](#why-3-bits-breaks)
- [The four repairs](#the-four-repairs)
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

The endpoint: **0.12 MB — 14.4x smaller than the 32-bit file — while writing better sentences
than the baseline.** The interesting part is the middle of the table: where quality collapses,
why it collapses, and how each repair buys back grammar at almost no size cost.

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
quant-lab-check         # five property checks of the building blocks
quant-lab all           # the full story
quant-lab qat           # any single run: baseline bits8 lowbits why clip rows mixed qat
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
```

A `Plan` is the entire recipe: how many bits per table, whether to search the grid width
(`tune`), and whether each row gets its own grid (`rows`). Tables left out of a per-table
`Plan` keep their ordinary numbers.

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
├── qat.py                training with the rounding inside the loop
├── experiments.py        the eight runs and the closing recap
├── selfcheck.py          five property checks of the parts
└── assets/               base.npz and every saved artifact (created on first run)
```

## Design notes and limits

- **Disk savings, not compute savings.** Codes are unpacked back to 32-bit floats before running.
  Real inference engines compute directly on integers; that machinery is out of scope here.
- **Symmetric grids only.** No zero point, no asymmetric ranges, no group-wise or per-channel
  block scales beyond whole rows.
- **Scales are chosen from the weights**, not calibrated on data; calibration is a natural next
  step (see below).
- **The toy language keeps the signal clean.** With a 58-word grammar, "grammatical" is exact and
  cheap to measure; a real model would report perplexity, and practitioners usually stop at
  4 bits rather than 2.
- **npz is a zip container**, so file sizes include a small overhead over the raw bit counts.

## Extending it

Good next experiments, roughly in order of payoff:

1. **Zero-point (asymmetric) quantization** for skewed distributions.
2. **Data-calibrated scales**: choose `clip` by minimizing activation error, not weight error.
3. **Group-wise scales** (e.g. one grid per 64 columns) between per-table and per-row.
4. **Entropy coding of codes** — the 2-bit `l2` histogram is far from uniform.
5. **GPTQ-style error compensation**: quantize columns sequentially, pushing each column's error
   into the not-yet-quantized ones.
6. **Integer inference**: a matmul over packed codes, so the small model also runs faster.

## Verification and reproducibility

`selfcheck.py` asserts the properties everything else relies on:

- `decode(encode(w))` never strays more than half a grid step from `w`, at 8/4/3/2 bits;
- bit packing is exact and achieves `ceil(n * bits / 8)` bytes for widths 8 down to 2;
- with one huge outlier, the width search always finds a narrowed grid that beats the full grid;
- a saved file reproduces the in-memory shrunk model bit for bit;
- 2 bits means exactly the codes {-1, 0, +1}.

Every random choice — training, sampling, evaluation — is seeded, and the evaluation seed is
fixed, so every number in this README regenerates identically: `python selfcheck.py`, then
`python experiments.py all`.

## Credits and license

An independent implementation of a classic quantization walkthrough, inspired by
Vuk Rosic's *Quantization from scratch* (MIT licensed). The toy language, the model, the code,
and the explanations here were written fresh for this project.

Released under the [MIT License](LICENSE).
