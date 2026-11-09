# Batch-Aware Latency Modeling for Dynamic Layer Routing: An Early-Stage Proposal

**Aarush Kandukoori** (Carnegie Mellon University) — Lead Student Author  
**Aditya Kandukoori** (Poolesville High School) — Co-Author

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Dynamic layer routing lets a language model choose, per token, which layers to execute, with the aim of lowering inference cost. Per-token savings do not always become wall-clock savings when requests are batched. A layer must still be run, and its weights loaded, if any sequence in the batch needs it. We propose an early-stage framework that adds a simple batch-aware latency model to the training objective of a routed network. The model separates a per-layer fixed cost (weight loading) from a per-token compute cost, and gives a closed-form, differentiable expected batch latency as a function of per-layer visit probabilities. We describe a Lagrangian-constrained training objective, a planned comparison against FLOP-based and per-token latency penalties, and a plan for using learned latency predictors from the neural architecture search literature as a drop-in replacement for the analytic cost table. This is a proposal only: no experiments have been run, and the proposed repository uses clearly labeled synthetic data. We are seeking expert feedback on whether the problem formulation matches real hardware behavior. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, and the authors reviewed and take responsibility for all content.

## Introduction & Motivation

**Authors**

- Aarush Kandukoori, Carnegie Mellon University (Lead Student Author)
- Aditya Kandukoori, Poolesville High School (Co-Author)

**Status.** This is an early-stage proposal, written to get expert feedback. It contains no experimental results. Everything below is a plan, and we expect parts of it to be wrong or to need changing.

**Motivation.** Dynamic-depth methods let each token skip or reorder transformer layers. Examples are Mixture-of-Depths [raposo2024] and Radial Networks [dotzel2024], which route tokens through layers chosen at run time. The usual justification is reduced compute per token. In serving, however, many sequences share a batch. During autoregressive decoding, much of the time for a layer goes to reading its weights from memory, and that cost is paid once per batch, not once per token. If tokens in the same batch choose different layers, the union of layers that must run can be much larger than any single token's path. Per-token savings then shrink at the batch level.

**Question.** Can the training objective of a routed model account for this batch effect with a simple, checkable latency model, and does doing so change which routing policies are learned compared with FLOP-based or per-token penalties?

This is not a claim that the effect is large on real hardware. Measuring that is part of what we would like to learn, and the model below is a hypothesis to be tested against real measurements.

## Related Work

**Dynamic depth and routing.** Mixture-of-Depths [raposo2024] uses a learned top-k router to decide which tokens take part in a layer's computation under a fixed compute budget. Radial Networks [dotzel2024] route between layers with a learned router, so a token's layer sequence is chosen dynamically. Both reduce per-token compute. To our knowledge, neither trains against a batch-level latency model. This is a statement about our reading of the literature, not a verified survey.

**Hardware-aware architecture search.** Early NAS used reinforcement learning [zoph2017] and later moved to differentiable relaxations [liu2019darts], including hardware-aware variants that train directly on a target platform [cai2019proxyless]. Once-for-All trains one network that supports many sub-networks for different deployment targets [cai2020ofa]. Latency is often estimated with a lookup table or a learned predictor. BRP-NAS uses a graph-based predictor [dudziak2020], HW-NAS-Bench provides hardware measurements for NAS search spaces [li2021hwnas], and "On Latency Predictors for Neural Architecture Search" [akhauri2024latency] studies how such predictors should be built and evaluated. "Encodings for Prediction-based Neural Architecture Search" [akhauri2024enc] studies how architectures should be encoded for prediction.

**Multi-token and attention variants.** Attamba [akhauri2024attamba] attends over multi-token states, which changes the cost structure of attention. We mention it only as a reminder that latency models must be re-fit when the block structure changes.

**Gap we hope to address.** Latency predictors in NAS typically model one architecture on one input shape. A routed model has a distribution over paths, and its cost depends on the batch. We propose a small bridge between the two.

## Proposed Framework

**Setup.** A network has $L$ layers. For each token, a router selects a subset of layers to execute, and we assume each layer is visited at most once per token. Let $p_l(\theta)\in[0,1]$ be the marginal probability, under router parameters $\theta$, that a random token executes layer $l$. In a soft or relaxed router, we take $p_l$ to be the mean gate value. This is a modeling heuristic, not an exact identity: a gate value is not literally a visit probability. For hard-routed models, $p_l$ would instead be the empirical visit frequency of layer $l$.

**Cost model.** Each layer has two costs on a given hardware target and batch regime:

- $a_l\ge 0$: fixed cost incurred if the layer runs at all in a batch (weight loading, kernel launch).
- $b_l\ge 0$: marginal cost per token that executes the layer.

**Expected batch latency.** Take a batch of $B$ tokens whose routing decisions are independent (an assumption we revisit below). Layer $l$ runs in the batch with probability $1-(1-p_l)^B$, and the expected number of tokens executing it is $Bp_l$. So

$$T_B(\theta)=\sum_{l=1}^{L}\Big[a_l\big(1-(1-p_l)^B\big)+b_l\,B\,p_l\Big].$$

*Checks.* For $B=1$ this reduces to $\sum_l (a_l+b_l)p_l$, the usual expected per-token cost. If every $p_l=1$, then $T_B=\sum_l (a_l+b_lB)$, the cost of the dense model. The derivative is

$$\frac{\partial T_B}{\partial p_l}=B\Big[a_l(1-p_l)^{B-1}+b_l\Big]\ge 0.$$

**Implication (a hypothesis to test).** The fixed-cost term is concave in $p_l$ and saturates quickly for large $B$. For $B$ large, a small $p_l$ already gives $1-(1-p_l)^B\approx 1$, so the layer is nearly always run, and reducing $p_l$ from, say, $0.3$ to $0.1$ saves little of the fixed cost $a_l$. The gradient of the fixed-cost term shrinks like $(1-p_l)^{B-1}$. A FLOP-style penalty, which is linear in $p_l$, would overvalue such reductions. Under this model, the batch-level saving comes mainly from driving some $p_l$ close to $0$, which means routing policies that are coherent across tokens.

**Training objective.** With task loss $\mathcal{L}_{\text{task}}(\theta)$ (e.g., next-token cross-entropy) and a latency budget $\tau$ for batch size $B$:

$$\min_\theta\ \mathcal{L}_{\text{task}}(\theta)\quad\text{s.t.}\quad T_B(\theta)\le\tau.$$

We would use the Lagrangian $\mathcal{L}_{\text{task}}+\lambda\,(T_B(\theta)-\tau)$ with dual ascent $\lambda\leftarrow\max(0,\lambda+\eta(T_B-\tau))$, with dual step size $\eta>0$ and $\lambda$ initialized at $0$. Since $T_B$ is differentiable in $p_l$, standard backpropagation applies when $p_l$ is estimated from a minibatch of gate values.

**Estimating $p_l$.** Within a minibatch we estimate $\hat p_l$ as the mean gate value for layer $l$. This estimate is biased for the nonlinear term, because $\mathbb{E}[1-(1-\hat p)^B]\ne 1-(1-p)^B$. Using the actual per-batch union (a hard indicator) with a straight-through estimator is the alternative. We plan to compare the two estimators.

**Replacing the analytic table with a predictor.** The independence assumption and the two-parameter cost per layer are crude. A learned latency predictor in the style of [akhauri2024latency] could take an encoding of the executed layer set [akhauri2024enc] and the batch size, and return a latency. It would replace $T_B$ in the Lagrangian. The cost is that differentiability must come from a relaxation or a surrogate gradient. We would first test the analytic form and consider the predictor as an extension.

**Where the independence assumption fails.** Real tokens within a sequence, and sequences with similar prompts, have correlated routing. Correlation increases the chance that the same layers are skipped across the batch, which would make the model above pessimistic. We list this as an open question.

## Methodology

**Overview.** The plan has three stages. (1) Implement the analytic latency model and check its algebra with unit tests. (2) Train small routed models on synthetic tasks under different latency penalties. (3) If the analytic model looks useful, collect real measurements on one GPU and compare analytic and learned latency models.

**Router.** A small linear gate over the hidden state produces a per-layer gate value in $[0,1]$. In training it is relaxed (sigmoid with temperature), and at inference it is thresholded. We would also include a simple batch-coherence variant in which the gate is computed from a pooled batch representation, so that all tokens in a batch share one layer set. This variant is a natural point of comparison because it matches the cost model's best case.

**Penalties compared.**

- No latency penalty (task loss only).
- Expected-layer-count / FLOP penalty: $\sum_l c_l p_l$, where $c_l\ge 0$ is a fixed per-layer cost proxy (FLOPs, or 1 for a pure layer count).
- Per-token latency penalty: $T_1(\theta)$.
- Batch-aware penalty: $T_B(\theta)$ from above, for several $B$.

**Cost parameters.** For synthetic experiments, $a_l$ and $b_l$ come from a roofline-style analytic formula with added noise (see the repository plan). For real hardware, they would be fit by least squares from timed runs of layers at several batch sizes, since for a fixed layer set the model is linear in $(a_l,b_l)$ per layer.

**Synthetic data statement.** The repository accompanying this proposal uses only clearly labeled synthetic and mock data: synthetic token sequences from a toy generator, and a synthetic latency table. Nothing in it is a measurement of real hardware or a real dataset, and it will say so in its README.

**Reproducibility.** Fixed seeds, configuration files per experiment, and unit tests for the closed-form expressions (checking the $B=1$ and dense cases above and the gradient formula against finite differences).

### AI Assistance Disclosure

AI tools (Claude) were used for literature synthesis, ideation, and draft preparation. The authors reviewed and take responsibility for all content in this proposal, including its citations and mathematical statements.

## Planned Evaluation & Baselines

This section is a protocol only. No results have been obtained.

**Stage A: synthetic sanity checks.**

- Verify numerically that $T_B$ matches Monte Carlo simulation of independent routing decisions for several $(p_l, B)$ settings, and that the analytic gradient matches finite differences.
- Train small routed models on a toy sequence task. Report the task metric, the learned $p_l$ profile, and the modeled $T_B$ at the budget, with multiple seeds.

**Stage B: baselines.** Dense model; the three alternative penalties above; the batch-coherent router; and a fixed-skip-pattern baseline matched in layer count.

**Stage C (conditional): real hardware.** If stage A looks informative, time layer subsets of a small open transformer at several batch sizes on one GPU, fit $(a_l,b_l)$, and test the model's prediction of batch latency for held-out layer sets. Compare against a learned predictor in the style of [akhauri2024latency], using rank correlation and mean absolute percentage error as in that literature's protocol.

**What would count as a negative outcome.** If the analytic model predicts held-out batch latency no better than a FLOP count, or the batch-aware penalty learns policies indistinguishable from the per-token penalty, the idea is not useful in this form. We would report that.

## Limitations & Open Questions

- **Independence assumption.** Token routing is correlated in practice, so $T_B$ may be biased. A correlation-aware version is an open problem.
- **Two-parameter cost.** Real kernels have shape-dependent efficiency, caching effects, and fusion across layers. Attention cost also depends on sequence length and the KV cache, which this model ignores.
- **Prefill versus decode.** The fixed-cost term is most plausible in memory-bound decoding. In compute-bound prefill, the $b_l$ term may dominate.
- **Serving systems.** Continuous batching and speculative decoding change what a batch is. We do not model them.
- **At-most-once visits.** The model assumes each layer is visited at most once per token. Routers that revisit layers would need a different $p_l$ definition.
- **Estimator bias.** The minibatch estimate of the nonlinear term is biased, as noted above.
- **Synthetic-only plan.** Initial work uses synthetic data, so it can only test the internal consistency of the idea, not its real-world value.
- **Novelty.** Related work we are not aware of may already cover this. We would value pointers.

**Feedback we would most like.** Does the fixed-versus-marginal cost split match what you see on real accelerators? Are there known batch-level effects in dynamic-depth serving that we are missing?

## References

- [raposo2024] Raposo et al., *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*, arXiv preprint 2024. https://arxiv.org/abs/2404.02258
- [dotzel2024] Dotzel et al., *Radial Networks: Dynamic Layer Routing for High-Performance Large Language Models*, arXiv preprint 2024. https://arxiv.org/abs/2404.04900
- [akhauri2024latency] Akhauri et al., *On Latency Predictors for Neural Architecture Search*, arXiv preprint 2024. https://arxiv.org/abs/2403.02446
- [akhauri2024enc] Akhauri et al., *Encodings for Prediction-based Neural Architecture Search*, arXiv preprint 2024. https://arxiv.org/abs/2403.02484
- [akhauri2024attamba] Akhauri et al., *Attamba: Attending To Multi-Token States*, arXiv preprint 2024. https://arxiv.org/abs/2411.17685
- [zoph2017] Zoph et al., *Neural Architecture Search with Reinforcement Learning*, ICLR 2017. https://arxiv.org/abs/1611.01578
- [liu2019darts] Liu et al., *DARTS: Differentiable Architecture Search*, ICLR 2019. https://arxiv.org/abs/1806.09055
- [cai2019proxyless] Cai et al., *ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware*, ICLR 2019. https://arxiv.org/abs/1812.00332
- [cai2020ofa] Cai et al., *Once-for-All: Train One Network and Specialize it for Efficient Deployment*, ICLR 2020. https://arxiv.org/abs/1908.09791
- [dudziak2020] Dudziak et al., *BRP-NAS: Prediction-based NAS using GCNs*, NeurIPS 2020. https://arxiv.org/abs/2007.08668
- [li2021hwnas] Li et al., *HW-NAS-Bench:Hardware-Aware Neural Architecture Search Benchmark*, ICLR 2021. https://arxiv.org/abs/2103.10584
