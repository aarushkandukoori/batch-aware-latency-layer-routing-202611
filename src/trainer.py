'''Lagrangian dual-ascent training loop for the toy router (SYNTHETIC data only).

Reference implementation of a PROPOSAL. The task loss is a made-up proxy:
    mean_i sum_l need[token_i][l] * (1 - gate_il)
so it only tests the internal consistency of the idea, not real-world value.

For each penalty the constraint is penalty(p) <= budget_frac * penalty(dense), with the dual variable
updated as lam <- max(0, lam + eta * (penalty - tau) / penalty(dense)).
'''
from __future__ import annotations

import random

from .cost_tables import CostTable
from .latency_model import expected_batch_latency
from .penalties import PENALTIES
from .router import TabularRouter


def static_task_loss(mask: list[float], tokens: list[int], need: list[list[float]]) -> float:
    '''Toy task loss of a fixed (token-independent) layer mask.'''
    return sum(sum(need[t][l] * (1.0 - mask[l]) for l in range(len(mask))) for t in tokens) / len(tokens)


def evaluate(router: TabularRouter, tokens, need, table: CostTable, B: int, n_batches: int = 20, seed: int = 123):
    '''Average soft visit profile and toy task loss over random batches; modeled T_B at the mean profile.'''
    rng = random.Random(seed)
    L = router.n_layers
    p_sum, loss_sum = [0.0] * L, 0.0
    for _ in range(n_batches):
        batch = [tokens[rng.randrange(len(tokens))] for _ in range(B)]
        g = router.gates(batch)
        for l in range(L):
            p_sum[l] += sum(row[l] for row in g) / B
        loss_sum += sum(need[t][l] * (1.0 - g[i][l]) for i, t in enumerate(batch) for l in range(L)) / B
    p = [x / n_batches for x in p_sum]
    return {'p': p, 'task_loss': loss_sum / n_batches, 'modeled_T_B': expected_batch_latency(p, table.a, table.b, B)}


def train(router: TabularRouter, tokens, need, table: CostTable, penalty: str = 'batch_aware',
          budget_frac: float = 0.5, B: int = 16, steps: int = 400, lr: float = 0.5, eta: float = 0.5, seed: int = 0):
    pen_fn = PENALTIES[penalty]
    L = router.n_layers
    dense_val, _ = pen_fn([1.0] * L, table, B)
    tau = budget_frac * dense_val
    lam = 0.0
    rng = random.Random(seed)
    for _ in range(steps):
        batch = [tokens[rng.randrange(len(tokens))] for _ in range(B)]
        g = router.gates(batch)
        p = [sum(row[l] for row in g) / B for l in range(L)]   # minibatch mean gate as p-hat (biased for the nonlinear term)
        val, dp = pen_fn(p, table, B)
        dg = [[(-need[t][l] + lam * dp[l]) / B for l in range(L)] for t in batch]
        router.backward(batch, dg, lr)
        if penalty != 'none' and dense_val > 0:
            lam = max(0.0, lam + eta * (val - tau) / dense_val)
    out = evaluate(router, tokens, need, table, B)
    out['lambda'] = lam
    return out
