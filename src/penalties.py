'''Latency penalties compared in the proposal. Each returns (value, d value / d p).

Reference implementation of a PROPOSAL. Costs come from a SYNTHETIC cost table.
'''
from __future__ import annotations

from typing import Callable, Sequence

from .cost_tables import CostTable
from .latency_model import expected_batch_latency, latency_grad

PenaltyFn = Callable[[Sequence[float], CostTable, int], tuple[float, list[float]]]


def pen_none(p, table, B):
    return 0.0, [0.0] * len(p)


def pen_flop(p, table, B):
    '''Expected-layer-count / FLOP style: sum_l c_l p_l, with c_l the table flops proxy.'''
    c = table.flops
    return sum(ci * pi for ci, pi in zip(c, p)), list(c)


def pen_per_token(p, table, B):
    '''Per-token latency T_1 (batch size ignored).'''
    return expected_batch_latency(p, table.a, table.b, 1), latency_grad(p, table.a, table.b, 1)


def pen_batch_aware(p, table, B):
    '''Proposed batch-aware latency T_B.'''
    return expected_batch_latency(p, table.a, table.b, B), latency_grad(p, table.a, table.b, B)


PENALTIES: dict[str, PenaltyFn] = {
    'none': pen_none,
    'flop': pen_flop,
    'per_token': pen_per_token,
    'batch_aware': pen_batch_aware,
}
