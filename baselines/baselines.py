'''Baseline stubs for the proposal (SYNTHETIC toy setting only).

Reference implementation of a PROPOSAL. Rows produced here are toy outputs on made-up data and are
NOT benchmark results.
'''
from __future__ import annotations

from src.latency_model import expected_batch_latency
from src.router import TabularRouter
from src.trainer import static_task_loss, train

# name -> (penalty, coherent router?)
TRAINED = {
    'task_only': ('none', False),
    'flop': ('flop', False),
    'per_token_latency': ('per_token', False),
    'batch_aware (proposed)': ('batch_aware', False),
    'batch_coherent': ('batch_aware', True),
}


def dense_mask(n_layers: int) -> list[float]:
    return [1.0] * n_layers


def fixed_skip_mask(n_layers: int, k: int) -> list[float]:
    '''Evenly spaced pattern keeping k layers.'''
    k = max(0, min(n_layers, k))
    keep = {int((i + 0.5) * n_layers / k) for i in range(k)} if k else set()
    return [1.0 if l in keep else 0.0 for l in range(n_layers)]


def evaluate_static(name, mask, tokens, need, table, B):
    return {'name': name, 'task_loss': static_task_loss(mask, tokens, need), 'layers': sum(mask),
            'modeled_T_B': expected_batch_latency(mask, table.a, table.b, B)}


def run_all(tokens, need, table, B=16, steps=400, budget_frac=0.5, seed=0):
    L, V = table.n_layers, len(need)
    rows = [evaluate_static('dense', dense_mask(L), tokens, need, table, B)]
    matched_layers = L
    for name, (pen, coherent) in TRAINED.items():
        router = TabularRouter(V, L, coherent=coherent, seed=seed)
        r = train(router, tokens, need, table, penalty=pen, budget_frac=budget_frac, B=B, steps=steps, seed=seed)
        rows.append({'name': name, 'task_loss': r['task_loss'], 'layers': sum(r['p']),
                     'modeled_T_B': r['modeled_T_B'], 'lambda': r['lambda']})
        if pen == 'batch_aware' and not coherent:
            matched_layers = round(sum(r['p']))
    rows.append(evaluate_static('fixed_skip (layer-count matched)', fixed_skip_mask(L, matched_layers),
                                tokens, need, table, B))
    return rows
