'''Algebra checks for the closed-form model (SYNTHETIC inputs; reference implementation of a PROPOSAL).'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.latency_model import expected_batch_latency, latency_grad, monte_carlo_latency  # noqa: E402

A = [1.0, 2.0, 0.5]
B_ = [0.1, 0.2, 0.05]


def test_b1_reduces_to_expected_per_token_cost():
    p = [0.3, 0.6, 0.9]
    expected = sum((a + b) * x for a, b, x in zip(A, B_, p))
    assert expected_batch_latency(p, A, B_, 1) == pytest.approx(expected)


def test_dense_case():
    for B in (1, 4, 32):
        expected = sum(a + b * B for a, b in zip(A, B_))
        assert expected_batch_latency([1.0] * 3, A, B_, B) == pytest.approx(expected)


def test_gradient_matches_finite_differences():
    p, B, eps = [0.3, 0.7, 0.05], 5, 1e-6
    g = latency_grad(p, A, B_, B)
    for l in range(3):
        hi, lo = list(p), list(p)
        hi[l] += eps
        lo[l] -= eps
        fd = (expected_batch_latency(hi, A, B_, B) - expected_batch_latency(lo, A, B_, B)) / (2 * eps)
        assert g[l] == pytest.approx(fd, abs=1e-5)


def test_monte_carlo_close_to_closed_form():
    p, a, b, B = [0.2, 0.5], [1.0, 2.0], [0.1, 0.2], 8
    mc = monte_carlo_latency(p, a, b, B, n_trials=4000, seed=1)
    assert mc == pytest.approx(expected_batch_latency(p, a, b, B), abs=0.1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        expected_batch_latency([1.5], [1.0], [1.0], 2)
    with pytest.raises(ValueError):
        expected_batch_latency([0.5], [1.0], [1.0], 0)
