'''Stub for a learned latency predictor over layer-set encodings (optional extension in the proposal).

Reference implementation of a PROPOSAL. No predictor is trained here; predict() falls back to the
analytic SYNTHETIC cost table. fit() is intentionally unimplemented.
'''
from __future__ import annotations

from .cost_tables import CostTable


def encode_layer_set(mask: list[int]) -> tuple[int, ...]:
    '''Binary one-hot encoding of the executed layer set.'''
    return tuple(1 if m else 0 for m in mask)


class LatencyPredictorStub:
    def __init__(self, table: CostTable):
        self.table = table

    def fit(self, measurements) -> None:
        raise NotImplementedError('Planned: fit a predictor on real timings (Stage C). Not implemented.')

    def predict(self, mask: list[int], B: int) -> float:
        '''Latency of a fixed layer set run with B tokens, from the analytic table.'''
        enc = encode_layer_set(mask)
        return sum(m * (a + b * B) for m, a, b in zip(enc, self.table.a, self.table.b))
