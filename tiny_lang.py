"""A toy language with 58 words and six sentence shapes.

The model's job is to imitate this language. A written sentence counts as
correct only when it matches one of the shapes exactly, so every later
experiment can be judged by the same yardstick: write 1,000 sentences,
count the correct ones. In a shape, "adj?" means an adjective appears
about a third of the time.
"""
import re

import numpy as np

PEOPLE = ['astronaut', 'captain', 'chef', 'dragon', 'engineer', 'fairy', 'ghost',
          'pilot', 'pirate', 'queen', 'robot', 'wizard']
THINGS = ['compass', 'crystal', 'lantern', 'map', 'pearl', 'ring', 'rocket',
          'scroll', 'shield', 'star', 'sword', 'torch']
PLACES = ['cave', 'cliff', 'desert', 'garden', 'island', 'mountain', 'ocean',
          'station', 'tower', 'valley']
VERBS = ['builds', 'carries', 'finds', 'guards', 'lifts', 'loses', 'seeks', 'sells']
ADJS = ['brave', 'calm', 'golden', 'hidden', 'lost', 'quiet', 'swift', 'tiny']
GLUE = ['the', 'a', 'and', 'in', 'near', 'is', ',', '.']

VOCAB = GLUE + ADJS + VERBS + PEOPLE + THINGS + PLACES
word_index = {w: i for i, w in enumerate(VOCAB)}

SLOTS = {'person': PEOPLE, 'thing': THINGS, 'place': PLACES, 'does': VERBS, 'adj': ADJS}

SHAPES = [
    'the adj? person does the adj? thing .',
    'a person and a person does the thing .',
    'in the place , the person does a thing .',
    'the thing is adj and the place is adj .',
    'the person does the thing near the place .',
    'near the place , the adj person does a adj thing .',
]


def make_sentence(rng):
    """One sentence of the language, as a list of words."""
    words = []
    for slot in SHAPES[rng.integers(len(SHAPES))].split():
        if slot == 'adj?':
            if rng.random() > 0.35:
                continue
            slot = 'adj'
        words.append(str(rng.choice(SLOTS[slot])) if slot in SLOTS else slot)
    return words


def make_corpus(n_sentences, seed=0):
    """n sentences glued into one long list of word numbers."""
    rng = np.random.default_rng(seed)
    words = [w for _ in range(n_sentences) for w in make_sentence(rng)]
    return np.array([word_index[w] for w in words], dtype=np.int64)


def _as_regex(shape):
    chunks = []
    for slot in shape.split():
        if slot == 'adj?':
            chunks.append('(?:(?:%s) )?' % '|'.join(ADJS))
        elif slot in SLOTS:
            chunks.append('(?:%s) ' % '|'.join(SLOTS[slot]))
        else:
            chunks.append(re.escape(slot) + ' ')
    return re.compile(''.join(chunks).rstrip() + r'\Z')


_CHECKERS = [_as_regex(s) for s in SHAPES]


def is_grammatical(words):
    """True when the words form one exact sentence of the language."""
    text = ' '.join(words)
    return any(c.match(text) for c in _CHECKERS)


if __name__ == '__main__':
    rng = np.random.default_rng(2)
    for _ in range(5):
        print(' '.join(make_sentence(rng)))
    print(len(VOCAB), 'words,', len(SHAPES), 'shapes')
