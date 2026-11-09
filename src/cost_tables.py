'''SYNTHETIC per-layer cost tables (a_l, b_l).

Reference implementation of a PROPOSAL. Layer sizes, bandwidth and compute constants are MADE UP.
The values are NOT measurements of any real hardware. Units are arbitrary.
'''
from __future__ import annotations

import random
from dataclasses import dataclass

LABEL = 'SYNTHETIC: made-up roofline-style costs, not a hardware measurement'


@dataclass(frozen=True)
class CostTable:
    a: list[float]       # fixed cost per layer (weight-load-like), arbitrary units
    b: list[float]       # per-token cost per layer, arbitrary units
    flops: list[float]   # FLOP-style per-layer proxy (noise-free), arbitrary units
    synthetic: bool = True
    label: str = LABEL

    @property
    def n_layers(self) -> int:
        return len(self.a)

    def to_dict(self) -> dict:
        return {'synthetic': self.synthetic, 'label': self.label, 'a': self.a, 'b': self.b, 'flops': self.flops}

    @classmethod
    def from_dict(cls, d: dict) -> 'CostTable':
        return cls(a=list(d['a']), b=list(d['b']), flops=list(d['flops']), synthetic=bool(d.get('synthetic', True)),
                   label=d.get('label', LABEL))


def make_synthetic_cost_table(
    n_layers: int = 8, seed: int = 0, noise: float = 0.1, bandwidth: float = 1.0, compute: float = 1.0
) -> CostTable:
    '''Roofline-flavoured toy: a_l ~ size/bandwidth, b_l ~ flops_per_token/compute, with multiplicative noise.'''
    rng = random.Random(seed)
    a, b, flops = [], [], []
    for _ in range(n_layers):
        size = rng.uniform(0.5, 2.0)              # made-up layer size
        fpt = size * rng.uniform(0.8, 1.2)        # made-up FLOPs per token
        a.append(max(0.0, size / bandwidth * (1.0 + noise * rng.gauss(0, 1))))
        b.append(max(0.0, 0.02 * fpt / compute * (1.0 + noise * rng.gauss(0, 1))))
        flops.append(fpt)
    return CostTable(a=a, b=b, flops=flops)
