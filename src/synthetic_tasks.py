'''SYNTHETIC toy sequence task. Not a real dataset.

Reference implementation of a PROPOSAL. Tokens come from a seeded Markov chain. A made-up matrix
need[token][layer] says how much task loss is incurred if that token skips that layer.
'''
from __future__ import annotations

import json
import random
from pathlib import Path

from .cost_tables import CostTable

LABEL = 'SYNTHETIC toy data generated from a seeded Markov chain; not a real dataset'


def make_markov_tokens(n: int = 4000, vocab: int = 8, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    trans = []
    for _ in range(vocab):
        w = [rng.random() ** 3 + 1e-3 for _ in range(vocab)]
        s = sum(w)
        trans.append([x / s for x in w])
    tok = rng.randrange(vocab)
    out = []
    for _ in range(n):
        out.append(tok)
        tok = rng.choices(range(vocab), weights=trans[tok])[0]
    return out


def make_need_matrix(vocab: int = 8, n_layers: int = 8, seed: int = 0) -> list[list[float]]:
    '''need[t][l]: made-up loss incurred when token type t skips layer l.'''
    rng = random.Random(seed + 1)
    return [[rng.uniform(0.3, 1.0) if rng.random() < 0.35 else rng.uniform(0.0, 0.05) for _ in range(n_layers)]
            for _ in range(vocab)]


def load_synthetic(directory: str | Path):
    '''Load files written by data/generate_synthetic.py. Returns (tokens, need, table) or None if missing.'''
    d = Path(directory)
    try:
        tokens = json.loads((d / 'tokens.json').read_text())['tokens']
        need = json.loads((d / 'need_matrix.json').read_text())['need']
        table = CostTable.from_dict(json.loads((d / 'cost_table.json').read_text()))
    except FileNotFoundError:
        return None
    return tokens, need, table
