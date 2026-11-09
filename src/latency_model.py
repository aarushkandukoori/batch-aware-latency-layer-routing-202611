'''Closed-form batch-aware latency model.

Reference implementation of a PROPOSAL. No experiments have been run and no hardware was measured;
the per-layer costs a_l, b_l are inputs (in this repo: SYNTHETIC, made-up units).

    T_B(p) = sum_l [ a_l * (1 - (1 - p_l)**B) + b_l * B * p_l ]

a_l: fixed cost if layer l runs at all in the batch (weight loading, launch).
b_l: marginal cost per token that executes layer l.
Assumes routing decisions of the B tokens are independent (a known, crude assumption).
'''
from __future__ import annotations

import random
from typing import Sequence


def _validate(p: Sequence[float], a: Sequence[float], b: Sequence[float], B: int) -> None:
    if not (len(p) == len(a) == len(b)):
        raise ValueError('p, a, b must have the same length')
    if B < 1 or int(B) != B:
        raise ValueError('B must be a positive integer')
    if any(x < 0.0 or x > 1.0 for x in p):
        raise ValueError('visit probabilities must lie in [0, 1]')
    if any(x < 0.0 for x in a) or any(x < 0.0 for x in b):
        raise ValueError('costs a_l, b_l must be non-negative')


def expected_batch_latency(p: Sequence[float], a: Sequence[float], b: Sequence[float], B: int) -> float:
    '''Expected batch latency T_B under independent routing (model units, not seconds).'''
    _validate(p, a, b, B)
    return sum(al * (1.0 - (1.0 - pl) ** B) + bl * B * pl for pl, al, bl in zip(p, a, b))


def latency_grad(p: Sequence[float], a: Sequence[float], b: Sequence[float], B: int) -> list[float]:
    '''dT_B/dp_l = B * (a_l * (1 - p_l)**(B-1) + b_l)  (always >= 0).'''
    _validate(p, a, b, B)
    return [B * (al * (1.0 - pl) ** (B - 1) + bl) for pl, al, bl in zip(p, a, b)]


def monte_carlo_latency(
    p: Sequence[float], a: Sequence[float], b: Sequence[float], B: int, n_trials: int = 2000, seed: int = 0
) -> float:
    '''Simulate independent routing of B tokens and average the realised latency.

    Used only to sanity-check the closed form; it simulates the model's own assumptions.
    '''
    _validate(p, a, b, B)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_trials):
        for pl, al, bl in zip(p, a, b):
            count = sum(1 for _ in range(B) if rng.random() < pl)
            if count > 0:
                total += al + bl * count
    return total / n_trials
