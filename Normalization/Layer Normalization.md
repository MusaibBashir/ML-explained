# Layer Normalization — Comprehensive Notes

> [!abstract] About These Notes Full treatment of Layer Normalization (Ba, Kiros & Hinton, 2016 — arXiv:1607.06450). Covers motivation as a direct response to BN's failures, the LN transform derived step by step, full backpropagation, learnable parameters, LN in RNNs, Pre-LN vs Post-LN in Transformers, the no-inference-gap advantage, and the normalization family (RMSNorm, Group Norm, Instance Norm).
> 
> **Paper:** _Layer Normalization_ · Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton · University of Toronto · 2016
> 
> **Companion files:** [[Batch Normalization]] (parent technique — same algorithm, different axis) · [[Momentum]] (optimization theory LN connects to) · [[Training Diagnostics and Optimizers]] (practical training heuristics) · [[Adam]] (LN + Adam is the standard Transformer training recipe)

---

## Overview of Topics (in order)

1. [[#1. Why Batch Normalization Is Not Always Enough|Why BN Falls Short]] — the three specific failure modes
2. [[#2. The Core Idea — Normalize Over Features, Not Batch|The Core Idea]] — the axis-switch and what it solves
3. [[#3. The Layer Normalization Transform — Full Derivation|LN Transform]] — every step derived and motivated
4. [[#4. Backpropagation Through Layer Norm — Complete Derivation|Backprop through LN]] — full gradient derivations from scratch
5. [[#5. Learnable Parameters — Re-scaling and Re-centering|Learnable Parameters]] — gain and bias, why they're necessary
6. [[#6. Layer Norm in Recurrent Neural Networks|LN in RNNs]] — the original motivation
7. [[#7. Layer Norm in the Transformer Architecture|LN in Transformers]] — Pre-LN vs Post-LN, the modern standard
8. [[#8. Geometric and Statistical Intuition|Geometric Intuition]] — what LN does to activation space
9. [[#9. Training vs. Inference — A Key Advantage Over BN|Training vs. Inference]] — why LN has no inference mode
10. [[#10. Layer Norm vs. Batch Norm — Deep Comparison|LN vs. BN]] — full comparison, when to use which
11. [[#11. Variants — RMSNorm, Group Norm, Instance Norm|Variants]] — the normalization family
12. [[#12. Key Formulas — Quick Reference|Quick Reference]]
13. [[#13. Things LN Does NOT Do|Misconceptions]]
14. [[#14. Connections to Broader Landscape|Broader context]] — LLMs, optimization landscape

---

## 1. Why Batch Normalization Is Not Always Enough

> [!tip] See Also
> 
> - The BN technique that LN is responding to: [[Batch Normalization#3. The Batch Normalization Transform|BN §3]]
> - BN's train/inference gap that LN eliminates: [[Batch Normalization#6. Training vs. Inference — A Critical Distinction|BN §6]]
> - The full comparison table: [[#10. Layer Norm vs. Batch Norm — Deep Comparison|§10 here]]

> [!danger] The Fundamental Limitation of BN [[Batch Normalization]] solves internal covariate shift elegantly — but only when a sufficiently large, fixed-structure mini-batch is available. Several critically important architectures violate this assumption.

### Recap: What BN Normalizes Over

[[Batch Normalization#3. The Batch Normalization Transform|BN (§3)]] normalizes a single scalar feature $x^{(k)}$ **across the batch dimension**: $$\mu_B^{(k)} = \frac{1}{m}\sum_{i=1}^m x_i^{(k)}, \qquad \sigma_B^{2(k)} = \frac{1}{m}\sum_{i=1}^m \left(x_i^{(k)} - \mu_B^{(k)}\right)^2$$

For each feature k, you look across all m examples in the batch. This requires:

1. Batch size m ≥ 1 (preferably m ≫ 1 for stable statistics)
2. All examples in the batch to have the same feature dimensionality and same sequence position
3. A fixed mapping from "feature index k" to "what concept feature k represents" across all examples

### Failure Mode 1 — Recurrent Neural Networks

In an RNN, the hidden state $\mathbf{h}_t$ is computed at each time step t: $$\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b})$$

**Problem 1:** Different sequences have different lengths. At time step t, some sequences may have already ended. The "batch" at step t is ragged — you can't compute a meaningful batch mean/variance over incomplete sequences.

**Problem 2:** The distribution of hidden states changes dramatically across time steps. Statistics appropriate for t=1 are completely wrong for t=50. BN would need separate γ^(t), β^(t) for every time step — and you can't maintain running averages for time steps longer than your training sequences.

> [!note] Connection to BN's inference problem [[Batch Normalization#6. Training vs. Inference — A Critical Distinction|BN §6]] shows that inference requires frozen population statistics. For RNNs, you'd need frozen statistics for every time step t — meaning you can't handle sequences longer than seen during training. LN's per-example statistics need no pre-computation.

### Failure Mode 2 — Small Batch Sizes and Online Learning

BN statistics become noisy as m → 1. At m = 1: $$\sigma^2_B = \frac{1}{1}\sum_{i=1}^1 (x_i - x_i)^2 = 0$$

Division by zero. Normalization is undefined. This rules out:

- Online learning (one example at a time)
- Reinforcement learning (effective batch size often 1 or very small)
- Large model training where GPU memory forces batch size 1 per device

### Failure Mode 3 — Variable-Length Inputs

In NLP, different sentences have different lengths. Padding and applying BN treats padding tokens as real data, corrupting the statistics. The batch statistics at position t include padding from shorter sequences — meaningless signal.

### The Solution: Rotate the Normalization Axis

The insight of Layer Normalization:

> **Instead of normalizing across the batch (over examples), normalize within each example (over its features).**

This axis-switch eliminates all three failures at once:

- No batch needed — statistics computed per example, independently
- Works at batch size 1
- Variable-length sequences: normalize each position independently, no padding corruption

---

## 2. The Core Idea — Normalize Over Features, Not Batch

> [!tip] See Also
> 
> - BN normalizes the other axis (columns of the data matrix): [[Batch Normalization#3. The Batch Normalization Transform|BN §3]]
> - The axis diagram below is the single most important thing to memorize about LN vs BN

### The Normalization Axis — The Key Diagram

Consider a mini-batch of m examples, each with H features, as a matrix **X** ∈ ℝ^(m×H):

```
              Feature dimension H
              ┌───────────────────────────────┐
Example 1  →  │  x₁₁   x₁₂   x₁₃   x₁₄  ...  │
Example 2  →  │  x₂₁   x₂₂   x₂₃   x₂₄  ...  │  ← LN: normalize each row
Example 3  →  │  x₃₁   x₃₂   x₃₃   x₃₄  ...  │
Example 4  →  │  x₄₁   x₄₂   x₄₃   x₄₄  ...  │
              └───────────────────────────────┘
                  ↑      ↑      ↑      ↑
               BN: normalize each column
```

- **Batch Normalization:** For each column k, compute μ and σ² across all m rows. Same feature, different examples.
- **Layer Normalization:** For each row i, compute μ and σ² across all H columns. Same example, different features.

**This is the entire conceptual difference.** Every practical consequence follows from this single choice.

### Why This Is Well-Defined for a Single Example

BN needs multiple rows (the batch). LN computes statistics over columns (H features of one example). Even with a single row (batch size = 1), you have H values — as long as H is large (512, 768, 1024, 4096 in Transformers), the statistics are stable. LN has no minimum batch size requirement.

### What "Layer" Means

The "layer" refers to one layer's activations for one input example. For a layer with H hidden units processing example i, the H pre-activations {a₁, a₂, ..., aₕ} are the set we normalize over. This set is complete and self-contained for every single forward pass, making LN fully independent of other examples.

---

## 3. The Layer Normalization Transform — Full Derivation

> [!tip] See Also
> 
> - BN's analogous four-step transform (same steps, different axis): [[Batch Normalization#3. The Batch Normalization Transform|BN §3]]
> - The learnable parameters g and b: [[#5. Learnable Parameters — Re-scaling and Re-centering|§5]]

LN is a differentiable transformation applied to the full activation vector **a** ∈ ℝ^H of a single example. Four steps.

### Setup and Notation

- $\mathbf{a} = (a_1, a_2, \ldots, a_H) \in \mathbb{R}^H$: vector of pre-activations at a layer, for one example
- $H$: number of hidden units in the layer
- $\mathbf{g} = (g_1, \ldots, g_H) \in \mathbb{R}^H$: learned gain (scale), initialized to **1**
- $\mathbf{b} = (b_1, \ldots, b_H) \in \mathbb{R}^H$: learned bias (shift), initialized to **0**

---

### Step 1 — Compute the Layer Mean

$$\mu = \frac{1}{H}\sum_{k=1}^H a_k$$

**Why:** We want to center the activations. The arithmetic mean is the natural center estimator. μ captures the overall "excitation level" of this layer for this example — subtracting it removes the global additive offset.

**Contrast with BN:** [[Batch Normalization#3. The Batch Normalization Transform|BN's μ_B^(k)]] averages _feature k across all m examples_. LN's μ averages _all H features of one example_. Same formula, different summation index.

**Verification:** After subtracting μ, the centered values have exactly zero mean: $\frac{1}{H}\sum_k (a_k - \mu) = \mu - \mu = 0\checkmark$

---

### Step 2 — Compute the Layer Variance

$$\sigma^2 = \frac{1}{H}\sum_{k=1}^H (a_k - \mu)^2$$

**Why biased (÷H, not ÷H−1):** Three reasons:

1. We're normalizing for optimization, not doing statistical inference — unbiasedness doesn't matter
2. The biased estimator is simpler and numerically better behaved
3. For large H (e.g., H = 1024), the difference is < 0.1% — negligible

**Alternative form** (mean of squares minus square of means): $$\sigma^2 = \frac{1}{H}\sum_{k=1}^H a_k^2 - \mu^2$$

This equals zero only if all activations are identical. Large σ² = heterogeneous activations; small σ² = similar activations.

> [!note] BN uses the biased estimator during training but corrects to unbiased at inference (×m/(m−1) — see [[Batch Normalization#6. Training vs. Inference — A Critical Distinction|BN §6]]). LN uses the biased estimator at both train and inference — there's no inference correction needed since statistics are per-example and always recomputed.

---

### Step 3 — Normalize

$$\hat{a}_k = \frac{a_k - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

Breaking into sub-steps:

**Step 3a — Center:** $(a_k - \mu)$. Mean of the result is exactly zero (shown above).

**Step 3b — Scale:** Divide by $\sqrt{\sigma^2 + \varepsilon}$. The result has unit variance: $$\frac{1}{H}\sum_{k=1}^H \hat{a}_k^2 = \frac{1}{H}\sum_{k=1}^H \frac{(a_k-\mu)^2}{\sigma^2} = \frac{\sigma^2}{\sigma^2} = 1\checkmark$$

**Step 3c — Why ε:** Prevents division by zero when all H activations are identical (σ² = 0). Typically ε = 10⁻⁵ or 10⁻⁸. Without ε, a dead layer would cause NaN.

**Geometric intuition:** After this step, **â** lies on a scaled $(H-1)$-dimensional hypersphere in the hyperplane orthogonal to the all-ones vector. Every activation vector, regardless of original scale or offset, is mapped to the same "surface." This is what removes internal covariate shift _within_ a layer, per example. See [[#8. Geometric and Statistical Intuition|§8]] for the full geometric picture.

---

### Step 4 — Scale and Shift

$$h_k = g_k \cdot \hat{a}_k + b_k \equiv \text{LN}_{\mathbf{g},\mathbf{b}}(a_k)$$

Vector form: $\mathbf{h} = \mathbf{g} \odot \hat{\mathbf{a}} + \mathbf{b}$ (element-wise multiplication).

**Why g and b are essential:** Pure normalization (step 3 only) is destructive. Consider:

- A layer feeding into a sigmoid: forcing unit-variance inputs constrains the sigmoid to its linear regime |x| ≲ 1. The nonlinearity is wasted.
- A ResNet residual branch: normalizing to zero mean forces the residual and skip connection onto equal footing — the network can't learn to initially suppress the residual.

Setting $g_k = \sigma_k, b_k = \mu_k$ exactly inverts the normalization. **The network can learn to turn LN off for any feature** if normalization is suboptimal there.

**Initialization:** g = **1**, b = **0** — pure normalization at start, adapts from there.

**Naming note:** The paper uses **g** (gain) and **b** (bias). [[Batch Normalization]] uses γ and β for the same parameters. Transformer literature uses both conventions. They are identical in role.

### Complete Forward Pass — Summary

$$\boxed{\mu = \frac{1}{H}\sum_{k=1}^H a_k, \qquad \sigma^2 = \frac{1}{H}\sum_{k=1}^H (a_k-\mu)^2, \qquad \hat{a}_k = \frac{a_k-\mu}{\sqrt{\sigma^2+\varepsilon}}, \qquad h_k = g_k\hat{a}_k + b_k}$$

Applied **independently** to each example. The **same** g and b are shared across all examples (and across all sequence positions in NLP — shared parameters, independent statistics).

---

## 4. Backpropagation Through Layer Norm — Complete Derivation

> [!tip] See Also
> 
> - Structurally identical derivation for BN (m replaces H): [[Batch Normalization#4. Backpropagation Through Batch Norm|BN §4]]
> - The three-path chain rule structure is the same in both BN and LN backprop

LN must be differentiable — gradients must flow through it. The chain rule through LN is non-trivial because μ and σ² both depend on all H inputs {aₖ}, so changing a single aᵢ affects all normalized outputs {â_j} through shared statistics.

### Computation Graph

```
a₁, a₂, ..., aₕ ──→ μ ──→ â₁, â₂, ..., âₕ ──→ h₁, h₂, ..., hₕ ──→ ℓ
                     ↑
a₁, a₂, ..., aₕ ──→ σ² ─┘
```

For each scalar aᵢ, it touches the loss through three paths:

1. Directly: aᵢ → âᵢ → hᵢ → ℓ
2. Via mean: aᵢ → μ → all â_j → all h_j → ℓ
3. Via variance: aᵢ → σ² → all â_j → all h_j → ℓ

Paths 2 and 3 are why we must sum over all H features — one input affects all outputs through shared statistics.

> [!note] This is the same three-path structure as [[Batch Normalization#4. Backpropagation Through Batch Norm|BN §4]], with H replacing m and features replacing examples. The derivation is structurally identical. If you understand one, you understand both.

### Given: Upstream Gradients

Layer after LN has backpropagated, giving us ∂ℓ/∂hₖ for all k = 1,...,H.

**Goal:** compute ∂ℓ/∂aₖ (to pass backward) and ∂ℓ/∂gₖ, ∂ℓ/∂bₖ (to update LN parameters).

---

### Step 1 — Gradient w.r.t. Normalized Activations âₖ

$h_k = g_k\hat{a}_k + b_k$ is linear in âₖ: $$\frac{\partial \ell}{\partial \hat{a}_k} = \frac{\partial \ell}{\partial h_k} \cdot g_k$$

_Intuition:_ gₖ is the gain on âₖ. Large gₖ → changes in âₖ matter more downstream → gradient amplified by gₖ.

---

### Step 2 — Gradient w.r.t. Variance σ²

σ² appears in every â_j through $(\sigma^2+\varepsilon)^{-1/2}$. Changing σ² changes all H outputs. Sum all paths: $$\frac{\partial \ell}{\partial \sigma^2} = \sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j} \cdot \frac{\partial \hat{a}_j}{\partial \sigma^2}$$

Local derivative of $\hat{a}_j = (a_j-\mu)(\sigma^2+\varepsilon)^{-1/2}$ w.r.t. σ²: $$\frac{\partial \hat{a}_j}{\partial \sigma^2} = (a_j - \mu)\cdot\left(-\frac{1}{2}\right)(\sigma^2+\varepsilon)^{-3/2}$$

Therefore: $$\frac{\partial \ell}{\partial \sigma^2} = \sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j} \cdot (a_j-\mu)\cdot\left(-\frac{1}{2}\right)(\sigma^2+\varepsilon)^{-3/2}$$

_Intuition:_ If σ² increases, the denominator grows → all âⱼ shrink. The gradient accumulates these effects across all H features, weighted by (aⱼ − μ) — features far from the mean drive the variance gradient most.

---

### Step 3 — Gradient w.r.t. Mean μ

μ appears in two places: in each numerator (aⱼ − μ), and inside σ² through (aⱼ − μ)². Two paths: $$\frac{\partial \ell}{\partial \mu} = \underbrace{\sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j}\cdot\frac{-1}{\sqrt{\sigma^2+\varepsilon}}}_{\text{Path 1: direct}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial \mu}}_{\text{Path 2: via variance}}$$

**Path 2 = 0** by the same beautiful cancellation as in [[Batch Normalization#4. Backpropagation Through Batch Norm|BN §4, Step 3]]: $$\frac{\partial \sigma^2}{\partial \mu} = \frac{-2}{H}\sum_{j=1}^H (a_j - \mu) = 0$$

Because the sum of deviations from the mean is always zero: $\sum_{j=1}^H (a_j - \mu) = H\mu - H\mu = 0$.

Therefore: $$\frac{\partial \ell}{\partial \mu} = \frac{-1}{\sqrt{\sigma^2+\varepsilon}} \sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j}$$

---

### Step 4 — Gradient w.r.t. Input aᵢ

A single aᵢ touches the loss through three paths: $$\frac{\partial \ell}{\partial a_i} = \underbrace{\frac{\partial \ell}{\partial \hat{a}_i}\cdot\frac{1}{\sqrt{\sigma^2+\varepsilon}}}_{\text{Self}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2}\cdot\frac{2(a_i-\mu)}{H}}_{\text{Via }\sigma^2} + \underbrace{\frac{\partial \ell}{\partial \mu}\cdot\frac{1}{H}}_{\text{Via }\mu}$$

Local derivatives:

- $\partial \hat{a}_i/\partial a_i = 1/\sqrt{\sigma^2+\varepsilon}$
- $\partial \sigma^2/\partial a_i = 2(a_i-\mu)/H$
- $\partial \mu/\partial a_i = 1/H$

Substituting everything (letting $\hat{\sigma} = \sqrt{\sigma^2+\varepsilon}$): $$\boxed{\frac{\partial \ell}{\partial a_i} = \frac{1}{H\hat{\sigma}}\left[H\frac{\partial \ell}{\partial \hat{a}_i} - \sum_{k=1}^H \frac{\partial \ell}{\partial \hat{a}_k} - \hat{a}_i\sum_{k=1}^H \frac{\partial \ell}{\partial \hat{a}_k}\hat{a}_k\right]}$$

**What the three terms mean:**

1. $H\cdot\partial\ell/\partial\hat{a}_i$: the raw upstream gradient, scaled up by H
2. $-\sum_k \partial\ell/\partial\hat{a}_k$: subtract the _sum_ of all upstream gradients (mean-centering the gradient)
3. $-\hat{a}_i\sum_k (\partial\ell/\partial\hat{a}_k)\hat{a}_k$: remove the component of gradient aligned with âᵢ (variance-normalizing the gradient)

> [!important] Forward-backward symmetry The forward pass removes the mean and variance from **a** to produce **â**. The backward pass removes the mean and a projection from the upstream gradient ∂ℓ/∂**â** before passing it back as ∂ℓ/∂**a**. **The normalization in the forward pass induces a dual de-normalization in the backward pass.** This is why LN gradients are always well-behaved: extreme upstream gradients are automatically projected away from the "trivial" directions (mean and variance scaling).

---

### Step 5 — Gradients w.r.t. Learnable Parameters gₖ and bₖ

g and b are **shared across all m examples** in the batch, so their gradients sum across the batch dimension:

**For bₖ (bias/shift):** $$\frac{\partial \ell}{\partial b_k} = \sum_{i=1}^m \frac{\partial \ell}{\partial h_k^{(i)}}$$

_Intuition:_ bₖ is a universal additive offset for feature k across all examples. Total upstream gradient.

**For gₖ (gain/scale):** $$\frac{\partial \ell}{\partial g_k} = \sum_{i=1}^m \frac{\partial \ell}{\partial h_k^{(i)}} \cdot \hat{a}_k^{(i)}$$

_Intuition:_ gₖ multiplies the normalized activation. Its gradient is the dot product of upstream error and the normalized data it scaled — across all examples in the batch.

---

## 5. Learnable Parameters — Re-scaling and Re-centering

> [!tip] See Also
> 
> - Same argument for γ and β in BN: [[Batch Normalization#3. The Batch Normalization Transform|BN §3, Step 4]]
> - Special cases table below shows what values of g and b "turn LN off"

### The Invariance Argument

Without g and b, the LN output **â** is completely determined by the _relative_ values of {aₖ} — the absolute scale and offset are removed. But this means:

- If the optimal representation needs non-unit variance (e.g., to drive a sigmoid into its saturated regime), LN can't achieve it
- If the optimal representation needs non-zero mean (e.g., a ReLU layer that works best when most inputs are slightly positive), LN can't achieve it

g and b restore these degrees of freedom. They're learned by gradient descent jointly with all other parameters.

### Re-centering and Re-scaling as a Learned Map

Think of LN as: (1) map **a** to a canonical form **â** (zero mean, unit variance), then (2) apply a learned linear map out of that canonical form: $$h_k = g_k \cdot \underbrace{\frac{a_k - \mu}{\hat{\sigma}}}_{\text{canonical form}} + b_k$$

The canonical form is fixed by the normalization. The linear map (g, b) is fully flexible.

### Special Cases Table

|gₖ value|bₖ value|Effect|
|---|---|---|
|1|0|Pure normalization — initial state|
|σₖ|μₖ|Exactly inverts normalization — LN is transparent|
|> 1|0|Amplified normalized features|
|0|c|Feature k becomes constant c — effectively gated off|

---

## 6. Layer Norm in Recurrent Neural Networks

> [!tip] See Also
> 
> - Why BN completely fails for RNNs: [[#1. Why Batch Normalization Is Not Always Enough|§1, Failure Mode 1]]
> - The general vanishing gradient problem LN helps with: [[Momentum#1. Gradient Descent — Basics and Failure Modes|Momentum §1]]

### The Original Motivation

The Ba et al. (2016) paper's primary application was RNNs, where [[Batch Normalization]] fails. LN is applied to the summed inputs at each time step, independently for each example.

### LSTM with Layer Normalization

Standard LSTM equations: $$\mathbf{a}_t = W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t$$ $$\begin{pmatrix}\mathbf{i}_t \ \mathbf{f}_t \ \mathbf{o}_t \ \mathbf{g}_t\end{pmatrix} = \begin{pmatrix}\sigma \ \sigma \ \sigma \ \tanh\end{pmatrix}\left(\mathbf{a}_t\right)$$

With Layer Normalization inserted before the nonlinearities: $$\begin{pmatrix}\mathbf{i}_t \ \mathbf{f}_t \ \mathbf{o}_t \ \mathbf{g}_t\end{pmatrix} = \begin{pmatrix}\sigma \ \sigma \ \sigma \ \tanh\end{pmatrix}\left(\text{LN}(\mathbf{a}_t)\right)$$

Each time step t has its own LN computation (its own μₜ, σ²ₜ computed fresh from **a**ₜ), but the **gain g and bias b are shared across all time steps**. This is analogous to weight tying across time — same transformation, different statistics.

### Why This Helps RNNs

**Gradient flow:** In deep RNNs or long sequences, gradients are multiplied by W_hh at each step — this causes exponential vanishing or explosion. LN stabilizes the magnitude of **h**ₜ at each step, reducing multiplicative compounding. (LN doesn't fully solve vanishing gradients — LSTM gating is needed for that — but substantially reduces its severity.)

**Stable hidden state distribution:** Without normalization, **h**ₜ can grow or shrink over long sequences, making the model sensitive to sequence length. LN keeps hidden states within a bounded regime at every step.

**No BN-style inference problem:** Because LN computes per-example statistics, there's no need to maintain running averages over time steps. LN at inference is identical to LN at training — the same formula applied to the same inputs produces the same outputs.

---

## 7. Layer Norm in the Transformer Architecture

> [!tip] See Also
> 
> - Why BN doesn't work for sequences: [[#1. Why Batch Normalization Is Not Always Enough|§1, Failure Mode 3]]
> - Gradient flow argument for Pre-LN: [[Momentum#8. Four Convergence Regimes|Convergence Regimes (Momentum §8)]] — Post-LN blocks the residual gradient highway
> - Adam + Pre-LN is the standard recipe: [[Adam#11. Practical Guide|Adam §11]]

### Why Transformers Use LN (Not BN)

Transformers process sequences in parallel — no temporal recurrence, just attention over all positions. LN is the natural choice because:

1. Each position (token) is processed independently — normalizing the d_model-dimensional vector of each token makes structural sense
2. Batch size can be small (especially for long sequences where memory is limited)
3. d_model is large (512, 768, 1024, 4096+), giving stable statistics

### Post-LN vs. Pre-LN

**Original Transformer (Vaswani et al., 2017) uses Post-LN:** LN is applied _after_ the residual addition: $$\mathbf{x}_{l+1} = \text{LN}\left(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l)\right)$$

**Modern architectures (GPT-2+, LLaMA, Mistral) use Pre-LN:** LN is applied _before_ the sub-layer: $$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}\left(\text{LN}(\mathbf{x}_l)\right)$$

|Property|Post-LN|Pre-LN|
|---|---|---|
|Training stability|Less stable; requires LR warmup|More stable; can use constant LR|
|Gradient flow at init|Can vanish in deep nets|Clean residual highway|
|Common in|Original Transformer|GPT-2, GPT-3, LLaMA, PaLM, Mistral|
|Final performance|Often slightly better (with tuning)|Slightly lower ceiling but much easier to train|

**Why Pre-LN is better for deep networks:**

In Post-LN, every residual path passes through LN:

```
x → x + SubLayer(x) → LN → next layer
```

At initialization, LN can distort gradients. In deep (100+ layer) Transformers, this distortion compounds — making it hard to train without careful warmup schedules. Compare: the gradient from output to layer l must pass through l LN operations.

In Pre-LN, the residual highway is clean:

```
x → x + SubLayer(LN(x))
```

The residual path `x_l → x_{l+1}` always has a direct gradient path (the identity shortcut) with **no LN in the way**. Gradients flow from the last layer to the first without passing through any normalization. This is analogous to why ResNets work — the skip connection preserves gradient flow — see [[Momentum#8. Four Convergence Regimes|Momentum §8]] on the importance of direct gradient paths.

### One LN Per Sub-Layer

A standard Transformer block has two sub-layers: Multi-Head Attention (MHA) and Feed-Forward Network (FFN). Each has its own LN. In Pre-LN:

```
Input x_l
  │
  ├──→ LN₁ → MHA → (+)──→ x_l + MHA(LN₁(x_l)) = x_l'
  │                  ↑
  └──────────────────┘

  │
  ├──→ LN₂ → FFN → (+)──→ x_l' + FFN(LN₂(x_l')) = x_{l+1}
  │                  ↑
  └──────────────────┘
```

---

## 8. Geometric and Statistical Intuition

### What LN Does to the Activation Vector

Consider **a** ∈ ℝ^H. The normalization $\hat{\mathbf{a}} = (\mathbf{a} - \mu\mathbf{1})/\hat{\sigma}$ does:

1. **Centering** ($\mathbf{a} - \mu\mathbf{1}$): Projects **a** onto the hyperplane $\mathbf{v} \cdot \mathbf{1} = 0$ (the subspace orthogonal to the all-ones vector). Removes the "DC component" — the global offset.
    
2. **Scaling** (÷$\hat{\sigma}$): Normalizes the length of the projected vector to √H. All activation vectors, regardless of original magnitude, are mapped to the same sphere of radius √H.
    

The result: **â** always satisfies $\sum_k \hat{a}_k = 0$ and $\frac{1}{H}\sum_k \hat{a}_k^2 = 1$. Every activation vector lives on the same surface — **only the direction (pattern of activations) is preserved**, not the magnitude or offset.

### The Statistical View: Z-Scoring Within a Layer

LN is exactly **z-scoring** the activations: $$\hat{a}_k = \frac{a_k - \mu}{\sigma}$$

This converts any distribution to one with zero mean and unit variance. The key assumption: the H activations of a layer are draws from some underlying distribution that LN forces to be approximately standard.

### Why This Controls Internal Covariate Shift

[[Batch Normalization#1. The Core Problem — Internal Covariate Shift|BN §1]] defines internal covariate shift as the change in a layer's input distribution during training. LN directly eliminates this within each layer by forcing every activation vector to the same canonical form — regardless of what the preceding layers do, the normalized inputs **â** always have zero mean and unit variance. The downstream layer always sees the same class of input distributions, just with different _directions_ (patterns).

---

## 9. Training vs. Inference — A Key Advantage Over BN

> [!tip] See Also
> 
> - BN's complicated train/inference distinction (the problem LN solves): [[Batch Normalization#6. Training vs. Inference — A Critical Distinction|BN §6]]
> - Practical implications for deployment: [[Training Diagnostics and Optimizers#3. Practical Gradcheck Tips|TDO §3]]

> [!important] LN's Greatest Practical Advantage Unlike [[Batch Normalization]], Layer Normalization has **identical behavior at training and inference**. There is no running mean, no running variance, no population statistics to maintain, no `model.eval()` mode difference for LN.

### Why BN Has a Train/Inference Distinction

In [[Batch Normalization#6. Training vs. Inference — A Critical Distinction|BN §6]], μ_B and σ²_B are computed from the mini-batch at training time. At inference, you may only have one example, and BN must substitute frozen population statistics (running averages maintained during training). This creates:

- Potential mismatch if running averages are stale or the test distribution differs from training
- The need for mode management (`model.train()` vs `model.eval()`)
- A common silent bug: forgetting eval() mode uses stochastic batch statistics at inference

### Why LN Has No Such Distinction

LN computes μ and σ² from **the single example's own activations** {aₖ}_{k=1}^H. These are:

- Fully determined by the current input
- Independent of any other examples
- Recomputed fresh for every forward pass

Whether processing a single example at inference or a batch of 512 at training, LN performs exactly the same computation per example. **Nothing to accumulate, nothing to switch, no population statistics to maintain.**

|Property|Batch Norm|Layer Norm|
|---|---|---|
|Train/inference difference|**Yes — critical**|No|
|Requires `model.eval()`|**Yes**|No|
|Running statistics needed|**Yes**|No|
|Works at batch size 1 (inference)|Yes (uses frozen stats)|**Yes (naturally)**|
|Sensitive to distribution shift at inference|**Yes**|No|

---

## 10. Layer Norm vs. Batch Norm — Deep Comparison

> [!tip] See Also
> 
> - Full BN treatment: [[Batch Normalization]]
> - Full comparison table with RMSNorm and Group Norm: [[#12. Key Formulas — Quick Reference|§12 quick reference]]

### The One Core Difference

Both use the same four-step algorithm: mean → variance → normalize → scale+shift. The **only** difference is the axis of normalization:

- **BN:** normalize each feature k across the batch → statistics depend on other examples
- **LN:** normalize all features of one example → statistics are self-contained

### When BN Wins

- **Large-batch vision tasks (CNNs):** Batch statistics over many examples and spatial locations are extremely stable. BN's per-channel normalization preserves spatial structure.
- **When batch statistics are informative:** i.i.d. examples from the same distribution → good batch mean/variance.
- **Regularization:** BN's stochastic batch statistics act as a regularizer ([[Batch Normalization#8. BN as a Regularizer|BN §8]]). LN does not regularize.

### When LN Wins

- **RNNs and sequential models:** Variable-length sequences, variable batch composition → BN is undefined or noisy. LN handles naturally.
- **Transformers:** Standard for all modern LLMs. Per-token normalization is coherent and batch-independent.
- **Small batch sizes:** RL, on-device inference, large model training.
- **Consistent train/inference behavior:** No mode switching, no running averages.
- **Deployment simplicity:** One less source of bugs ([[Batch Normalization#6. Training vs. Inference — A Critical Distinction|BN §6]]).

### The Axis Choice and Its Consequences

|Property|Batch Norm|Layer Norm|
|---|---|---|
|Normalizes over|Batch dimension (per feature)|Feature dimension (per example)|
|Statistics depend on|All examples in batch|Single example only|
|Works at batch size 1|**No**|Yes|
|Train ≠ Inference|**Yes (critical)**|No|
|Running statistics|Required|Not needed|
|Best for|CNNs, large-batch vision|Transformers, RNNs, NLP|
|Stochastic at train|Yes (batch varies)|No (deterministic per example)|
|Regularization|Yes|**Weak**|

---

## 11. Variants — RMSNorm, Group Norm, Instance Norm

> [!tip] See Also
> 
> - LN is the G=1 special case of Group Norm
> - RMSNorm drops the mean subtraction from LN
> - Group Norm sits between LN and Instance Norm

### RMSNorm — Root Mean Square Layer Normalization

**Paper:** Zhang & Sennrich (2019)

**Motivation:** Most of LN's benefit may come from re-scaling (controlling magnitude), not re-centering (removing mean). Drop the mean subtraction to get a simpler, faster operation.

**Formula:** $$\text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{H}\sum_{k=1}^H a_k^2}$$ $$\bar{a}_k = \frac{a_k}{\text{RMS}(\mathbf{a}) + \varepsilon}, \qquad h_k = g_k\bar{a}_k$$

Note: **no mean subtraction, no bias b**. Only re-scaling by RMS, then learned gain.

**Comparison to LN:** RMSNorm is LN without the centering step. LN normalizes both the first moment (mean) and second moment (variance). RMSNorm normalizes only the second moment (RMS energy), leaving the mean unconstrained.

**Why it works nearly as well:** The re-centering (subtracting μ) is primarily a convenience — it removes the DC offset, which the network could also learn to handle via the bias. The critical operation is re-scaling, which controls the signal magnitude and prevents explosion/vanishing. RMSNorm keeps the essential operation and drops the rest.

**Where it's used:** LLaMA, Mistral, Gemma — every major modern open-source LLM uses RMSNorm instead of LN for efficiency at scale. At H = 4096+ dimensions, the saved computation adds up across billions of tokens.

### Group Normalization

**Paper:** Wu & He (2018)

**Motivation:** A middle ground between LN (all features) and Instance Normalization (one feature at a time). Divide the H features into G groups of H/G features each, normalize within each group.

**Formula:** For group g containing features {k: k ∈ 𝒢_g}: $$\mu_g = \frac{1}{|\mathcal{G}_g|}\sum_{k\in\mathcal{G}_g} a_k, \qquad \sigma^2_g = \frac{1}{|\mathcal{G}_g|}\sum_{k\in\mathcal{G}_g} (a_k-\mu_g)^2$$ $$\hat{a}_k = \frac{a_k - \mu_g}{\sqrt{\sigma^2_g + \varepsilon}}, \qquad h_k = g_k\hat{a}_k + b_k$$

**Special cases:**

- G = 1: Group Norm = Layer Norm (one group = all features)
- G = H: Group Norm = Instance Normalization (each feature is its own group)

**When it helps:** Object detection and segmentation (Mask R-CNN) where batch size is small (2–4 images per GPU) — BN is noisy, but full LN over all channels may mix semantically different features.

### The Normalization Family — Unified View

All normalization methods share the same four-step template. They differ only in which subset of the data matrix they use to compute statistics:

```
             Feature dimension (H)
           ┌──────────────────────────┐
Example 1  │  ██████████████████████  │  ← LN: entire row
Example 2  │  ████   ████   ████      │  ← GN: groups within row
Example 3  │  █      █      █         │  ← IN: one cell at a time
           └──────────────────────────┘
              ↑      ↑      ↑
             BN: entire column
```

---

## 12. Key Formulas — Quick Reference

|Quantity|Formula|
|---|---|
|Layer mean|$\mu = \frac{1}{H}\sum_{k=1}^H a_k$|
|Layer variance|$\sigma^2 = \frac{1}{H}\sum_{k=1}^H (a_k-\mu)^2$|
|Normalized activation|$\hat{a}_k = (a_k - \mu)/\sqrt{\sigma^2+\varepsilon}$|
|LN output|$h_k = g_k\hat{a}_k + b_k$|
|Gradient w.r.t. âₖ|$\partial\ell/\partial\hat{a}_k = (\partial\ell/\partial h_k)\cdot g_k$|
|Gradient w.r.t. σ²|$\sum_{k=1}^H (\partial\ell/\partial\hat{a}_k)(a_k-\mu)(-\frac{1}{2})(\sigma^2+\varepsilon)^{-3/2}$|
|Gradient w.r.t. μ|$\frac{-1}{\sqrt{\sigma^2+\varepsilon}}\sum_{k=1}^H \partial\ell/\partial\hat{a}_k$|
|Gradient w.r.t. aᵢ (full)|$\frac{1}{H\hat{\sigma}}\left[H\frac{\partial\ell}{\partial\hat{a}_i} - \sum_k\frac{\partial\ell}{\partial\hat{a}_k} - \hat{a}_i\sum_k\frac{\partial\ell}{\partial\hat{a}_k}\hat{a}_k\right]$|
|Gradient w.r.t. bₖ|$\sum_{i=1}^m \partial\ell/\partial h_k^{(i)}$|
|Gradient w.r.t. gₖ|$\sum_{i=1}^m (\partial\ell/\partial h_k^{(i)})\cdot\hat{a}_k^{(i)}$|
|RMSNorm|$h_k = g_k \cdot a_k / (\text{RMS}(\mathbf{a})+\varepsilon)$|
|Distributional guarantee|$\sum_k \hat{a}_k = 0$, $\frac{1}{H}\sum_k \hat{a}_k^2 = 1$|
|Pre-LN Transformer|$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}(\text{LN}(\mathbf{x}_l))$|
|Post-LN Transformer|$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l))$|

---

## 13. Things LN Does NOT Do

> [!warning] Misconceptions — each of these is a commonly believed falsehood

**LN does NOT decorrelate features.** Like [[Batch Normalization#11. Things BN Does NOT Do|BN]], it normalizes the marginal distribution of the feature vector but does not remove correlations between features. Full whitening requires computing the covariance matrix — O(H²) — intractable for large H.

**LN does NOT guarantee zero mean and unit variance at the output.** After the scale-and-shift step, hₖ = gₖâₖ + bₖ has mean b_k and variance g_k². Only the _intermediate_ âₖ has zero mean and unit variance.

**LN does NOT eliminate vanishing gradients.** It mitigates gradient magnitude instability, but in very deep networks vanishing gradients can still occur. Residual connections (used in all modern Transformers) are the primary solution; LN is complementary. The gradient flow argument connects to [[Momentum#8. Four Convergence Regimes|Momentum §8]].

**LN does NOT regularize as strongly as BN.** [[Batch Normalization#8. BN as a Regularizer|BN §8]] explains that BN's stochastic batch statistics inject noise. LN is deterministic per example — always normalizes to the same values, regardless of batch composition. LN provides essentially no regularization.

**LN is NOT identical for all architectures.** In CNNs, "the activations of a layer" could mean per-channel, per-spatial-position, or over all spatial locations. In Transformers, it means the d_model-dimensional embedding of each token. Always verify which dimensions LN is operating over for a given architecture.

---

## 14. Connections to Broader Landscape

> [!tip] See Also
> 
> - Full BN treatment: [[Batch Normalization]]
> - Optimization landscape smoothing: [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]]
> - Adam + Pre-LN standard recipe: [[Adam#11. Practical Guide|Adam §11]]

**Relationship to [[Batch Normalization]]:** LN is the natural generalization of BN to the per-example setting. Same algorithm, rotated axis. Understanding either deeply means understanding both. The key table: [[#10. Layer Norm vs. Batch Norm — Deep Comparison|§10 here]].

**Relationship to weight normalization:** Weight Normalization (Salimans & Kingma, 2016) reparameterizes weight vectors directly ($\mathbf{w} = \frac{g}{|\mathbf{v}|}\mathbf{v}$) rather than normalizing activations. It also decouples magnitude from direction, but operates on parameters rather than activations.

**Relationship to the optimization landscape:** Santurkar et al. (2018) showed that [[Batch Normalization|BN]]'s primary benefit is smoothing the loss landscape — making the gradient more Lipschitz-continuous. The same argument applies to LN: by controlling activation scale, LN reduces the Lipschitz constant of the loss as a function of earlier-layer weights, enabling larger learning rates and more stable training. This connects to [[Momentum#5. Condition Number and Optimal Step Size|condition number κ]] — LN reduces the effective κ seen by each layer's optimizer, allowing the [[Momentum#10. Optimal Global Parameters and Quadratic Speedup|O(√κ) momentum benefit]] to apply more effectively.

**LN in modern LLMs:** Every major LLM uses a form of LN. GPT-2/3/4 use Pre-LN with standard LN. LLaMA, Mistral, Gemma use Pre-LN with RMSNorm. The trend is toward simpler, faster variants of LN. [[Adam]] + Pre-LN (or Pre-RMSNorm) is the universal Transformer training recipe as of 2024–2025.

**The "why does normalization work" open question:** Both BN and LN papers attribute their benefit to controlling internal covariate shift. Santurkar et al. (2018) argue the real effect is smoothing the optimization landscape. This remains an open research question. What's empirically clear: both BN and LN substantially accelerate training, and their practical benefits are well-established even if the theory is debated.

---

## 15. Concept Map — How Everything Connects

```
Problem: BN Fails for RNNs, Small Batches, Variable-Length Sequences (§1)
  │
  ├── BN normalizes over batch → needs m > 1 → [[Batch Normalization#1|BN §1]]
  ├── BN has train/inference gap → [[Batch Normalization#6|BN §6]]
  └── BN needs fixed sequence length → RNNs broken

LN Solution: Rotate the normalization axis (§2)
  ├── Same four steps as [[Batch Normalization#3|BN §3]]
  └── One μ, σ² per example (not per feature)

Forward Pass (§3):
  μ = (1/H)Σaₖ → σ² = (1/H)Σ(aₖ-μ)² → âₖ = (aₖ-μ)/σ̂ → hₖ = gₖâₖ + bₖ

Backprop (§4): Three-path chain rule
  ├── Self: aᵢ → âᵢ
  ├── Via σ²: aᵢ → σ² → all âⱼ        (sum over H features)
  ├── Via μ: aᵢ → μ → all âⱼ          (sum cancels — zero!)
  └── Same structure as [[Batch Normalization#4|BN §4]] with H replacing m

g and b (§5): restore expressive power
  └── Network can invert LN if needed (gₖ=σₖ, bₖ=μₖ)

Applications:
  ├── RNNs (§6): per time step, shared g/b → stable hidden states
  │     └── Solves BN's ragged-batch/variable-length failures
  └── Transformers (§7): Pre-LN vs Post-LN
        ├── Pre-LN: clean residual highway, stable training ← PREFERRED
        │     └── [[Momentum#8|Momentum §8]]: direct gradient path avoids distortion
        └── Post-LN: original design, needs warmup

Key advantage (§9): No train/inference distinction
  └── Contrast: [[Batch Normalization#6|BN §6]] requires frozen population statistics

Variants (§11):
  ├── RMSNorm: drop mean subtraction → faster, same performance → LLaMA/Mistral
  ├── Group Norm: G groups → between LN (G=1) and Instance Norm (G=H)
  └── Instance Norm: per-sample per-channel → style transfer

Optimization connection:
  └── LN reduces effective κ → [[Momentum#5|Momentum §5]] condition number
      → enables [[Momentum#10|O(√κ) speedup]] to kick in more effectively
```

---

## 16. Quick Comparison — LN vs. BN vs. RMSNorm vs. GroupNorm

|Property|Batch Norm|Layer Norm|RMSNorm|Group Norm|
|---|---|---|---|---|
|Normalizes over|Batch (per feature)|Features (per example)|Features, RMS only|Feature groups (per example)|
|Works at batch size 1|**No**|Yes|Yes|Yes|
|Train ≠ Inference|**Yes**|No|No|No|
|Running statistics|Required|Not needed|Not needed|Not needed|
|Mean subtraction|Yes|Yes|**No**|Yes|
|Learned parameters|γ, β per feature|g, b per feature|g per feature|g, b per feature|
|Regularizes|**Yes**|Weak|Weak|Weak|
|Typical use case|CNNs, vision|Transformers, RNNs|LLMs (LLaMA etc.)|Small-batch vision|
|Full notes|[[Batch Normalization]]|Here|[[#11. Variants — RMSNorm, Group Norm, Instance Norm\|§11]]|[[#11. Variants — RMSNorm, Group Norm, Instance Norm\|§11]]|

---

_Notes compiled from Ba, Kiros & Hinton (2016), arXiv:1607.06450 · Vaswani et al. (2017) · Zhang & Sennrich (2019) · Wu & He (2018) — Musaib, 19/05/2026_