"""The tiny language model: read the last 6 words, score all 58 words for what comes next.

6 words -> 6 x 48 = 288 numbers -> l1 -> 512 -> l2 -> 512 -> l3 -> 58 scores.
Nearly all of its numbers live in four tables: emb (one row per word) and l1, l2, l3.
"""
import os

import numpy as np

from tiny_lang import VOCAB, word_index, make_corpus, is_grammatical

CTX, EMB, HID = 6, 48, 512
WEIGHTS = ('l1', 'l2', 'l3')
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def init_params(seed=0):
    """Small random numbers; each table has one row per output, one column per input."""
    rng = np.random.default_rng(seed)
    v, din = len(VOCAB), CTX * EMB
    p = {'emb': rng.normal(0, 0.3, (v, EMB)).astype(np.float32)}
    shapes = {'l1': (HID, din), 'l2': (HID, HID), 'l3': (v, HID)}
    for name, (rows, cols) in shapes.items():
        gain = np.sqrt(2 / cols) if name != 'l3' else np.sqrt(1 / cols)
        p[name] = (rng.normal(0, 1, (rows, cols)) * gain).astype(np.float32)
        p[name + '_b'] = np.zeros(rows, np.float32)
    return p


def count(params):
    """Total numbers stored in a parameter dictionary."""
    return int(sum(a.size for a in params.values()))


def predict(params, ctx):
    """Next-word scores for a batch of contexts, plus the input of every table."""
    x = params['emb'][ctx].reshape(len(ctx), -1)
    acts = [x]
    for name in WEIGHTS[:-1]:
        x = np.maximum(x @ params[name].T + params[name + '_b'], 0)
        acts.append(x)
    logits = x @ params['l3'].T + params['l3_b']
    return logits, acts


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grads(params, ctx, tgt, want_grads=True):
    """The mean loss, and for every number its gradient: how the loss reacts if the number grows."""
    n = len(tgt)
    logits, acts = predict(params, ctx)
    p = softmax(logits)
    loss = float(-np.log(p[np.arange(n), tgt] + 1e-12).mean())
    if not want_grads:
        return loss, None
    d = p
    d[np.arange(n), tgt] -= 1
    d /= n
    g = {}
    for i in reversed(range(len(WEIGHTS))):
        name = WEIGHTS[i]
        g[name] = d.T @ acts[i]
        g[name + '_b'] = d.sum(axis=0)
        d = d @ params[name]
        if i > 0:
            d *= acts[i] > 0
    g['emb'] = np.zeros_like(params['emb'])
    np.add.at(g['emb'], ctx, d.reshape(n, CTX, EMB))
    return loss, g


class Adam:
    """Turns gradients into small parameter updates."""

    def __init__(self, lr):
        self.lr, self.t, self.m, self.v = lr, 0, {}, {}

    def update(self, params, grads):
        self.t += 1
        for k, grad in grads.items():
            self.m[k] = 0.9 * self.m.get(k, 0) + 0.1 * grad
            self.v[k] = 0.999 * self.v.get(k, 0) + 0.001 * grad * grad
            m1 = self.m[k] / (1 - 0.9 ** self.t)
            v1 = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= (self.lr * m1 / (np.sqrt(v1) + 1e-8)).astype(np.float32)


def make_windows(words):
    """Every stretch of CTX words together with the word that follows it."""
    spans = np.arange(CTX) + np.arange(len(words) - CTX)[:, None]
    return words[spans], words[CTX:]


def train_net(params, words, iters, lr, batch=256, seed=0, history=None):
    """Look at a batch of windows, measure the loss, nudge every number, repeat."""
    ctx, tgt = make_windows(words)
    rng = np.random.default_rng(seed)
    opt = Adam(lr)
    for it in range(iters):
        pick = rng.integers(0, len(tgt), batch)
        loss, grads = loss_and_grads(params, ctx[pick], tgt[pick])
        opt.update(params, grads)
        if history is not None and (it % 25 == 0 or it == iters - 1):
            history.append((it, loss))


def val_loss(params, words, cap=20000):
    """Loss on text the model never trained on. Lower is better."""
    ctx, tgt = make_windows(words[:cap])
    return loss_and_grads(params, ctx, tgt, want_grads=False)[0]


SEED_LINE = ['the', 'wizard', 'finds', 'the', 'golden', 'map', '.']


def sample_sentences(params, n, seed=1, cap=24):
    """Write n sentences one word at a time, stopping at '.' or after cap words."""
    rng = np.random.default_rng(seed)
    window = [word_index[w] for w in SEED_LINE[-CTX:]]
    out = []
    for _ in range(n):
        words = []
        while len(words) < cap:
            logits, _ = predict(params, np.array([window]))
            probs = softmax(logits)[0].astype(np.float64)
            w = int(rng.choice(len(VOCAB), p=probs / probs.sum()))
            words.append(VOCAB[w])
            window = window[1:] + [w]
            if VOCAB[w] == '.':
                break
        out.append(words)
    return out


def grammar_rate(params, n=1000, seed=3):
    """The yardstick for every experiment: of n written sentences, how many are exactly right?"""
    return sum(is_grammatical(s) for s in sample_sentences(params, n, seed))


def say(params, n=3, seed=1):
    return [' '.join(s) for s in sample_sentences(params, n, seed)]


def datasets():
    """Text to train on and a smaller text reserved for measuring."""
    return make_corpus(120000, 0), make_corpus(1500, 9)


def get_trained(history=None, iters=4000, lr=2e-3):
    """The trained model, cached in assets/base.npz after the first run."""
    path = os.path.join(ASSETS, 'base.npz')
    if not os.path.exists(path):
        params, curve = init_params(0), []
        train_net(params, datasets()[0], iters, lr, history=curve)
        np.savez(path, _curve=np.array(curve), **params)
    saved = dict(np.load(path))
    if history is not None:
        history.extend((int(a), float(b)) for a, b in saved['_curve'])
    return {k: v for k, v in saved.items() if not k.startswith('_')}


def copy_params(params):
    return {k: v.copy() for k, v in params.items()}
