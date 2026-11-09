'''Smoke tests for penalties, routers and the trainer (SYNTHETIC data; reference implementation of a PROPOSAL).'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.cost_tables import make_synthetic_cost_table  # noqa: E402
from src.latency_model import expected_batch_latency  # noqa: E402
from src.penalties import PENALTIES  # noqa: E402
from src.router import TabularRouter  # noqa: E402
from src.synthetic_tasks import make_markov_tokens, make_need_matrix  # noqa: E402
from src.trainer import train  # noqa: E402

TABLE = make_synthetic_cost_table(n_layers=4, seed=0)


def test_per_token_penalty_is_T1():
    p = [0.2, 0.4, 0.6, 0.8]
    val, _ = PENALTIES['per_token'](p, TABLE, 32)
    assert val == pytest.approx(expected_batch_latency(p, TABLE.a, TABLE.b, 1))


def test_flop_penalty_is_linear_and_none_is_zero():
    p = [0.5] * 4
    val, grad = PENALTIES['flop'](p, TABLE, 8)
    assert val == pytest.approx(0.5 * sum(TABLE.flops))
    assert grad == TABLE.flops
    assert PENALTIES['none'](p, TABLE, 8)[0] == 0.0


def test_coherent_router_shares_one_layer_set():
    r = TabularRouter(vocab=8, n_layers=4, coherent=True, seed=0)
    g = r.gates([0, 1, 2, 3, 4])
    assert all(row == g[0] for row in g)


def test_training_smoke_runs_and_lambda_nonnegative():
    tokens = make_markov_tokens(n=500, vocab=8, seed=0)
    need = make_need_matrix(vocab=8, n_layers=4, seed=0)
    router = TabularRouter(8, 4, seed=0)
    out = train(router, tokens, need, TABLE, penalty='batch_aware', B=8, steps=50)
    assert out['lambda'] >= 0.0
    assert len(out['p']) == 4 and all(0.0 <= x <= 1.0 for x in out['p'])
