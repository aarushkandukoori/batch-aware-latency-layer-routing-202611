'''Toy routers: relaxed per-token gate and batch-coherent gate.

Reference implementation of a PROPOSAL. The router is a lookup table of logits over a SYNTHETIC
vocabulary (no neural network, stdlib only) so gradients can be written by hand and inspected.
'''
from __future__ import annotations

import math
import random


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class TabularRouter:
    '''gate[i][l] = sigmoid(w[token_i][l] / T).

    coherent=True: logits are pooled (mean) over the batch, so all tokens share one gate row, i.e. one
    layer set. This is the cost model's best case and a comparison point in the proposal.
    '''

    def __init__(self, vocab: int, n_layers: int, temperature: float = 1.0, coherent: bool = False,
                 init_logit: float = 2.0, seed: int = 0):
        rng = random.Random(seed)
        self.vocab, self.n_layers = vocab, n_layers
        self.temperature, self.coherent = temperature, coherent
        self.w = [[init_logit + rng.gauss(0, 0.1) for _ in range(n_layers)] for _ in range(vocab)]

    def gates(self, tokens: list[int]) -> list[list[float]]:
        T = self.temperature
        if self.coherent:
            n = len(tokens)
            row = [_sigmoid(sum(self.w[t][l] for t in tokens) / n / T) for l in range(self.n_layers)]
            return [list(row) for _ in tokens]
        return [[_sigmoid(self.w[t][l] / T) for l in range(self.n_layers)] for t in tokens]

    def backward(self, tokens: list[int], dgates: list[list[float]], lr: float) -> None:
        '''Gradient step given dLoss/dgate for every (token, layer).'''
        T, n = self.temperature, len(tokens)
        if self.coherent:
            row = self.gates(tokens)[0]
            for l in range(self.n_layers):
                dG = sum(dg[l] for dg in dgates)
                dlogit = dG * row[l] * (1.0 - row[l]) / T
                for t in tokens:
                    self.w[t][l] -= lr * dlogit / n
            return
        for i, t in enumerate(tokens):
            for l in range(self.n_layers):
                g = _sigmoid(self.w[t][l] / T)
                self.w[t][l] -= lr * dgates[i][l] * g * (1.0 - g) / T
