# Batch Normalization — Comprehensive Notes

> [!abstract] About These Notes Full treatment of Batch Normalization (Ioffe & Szegedy, ICML 2015 — arXiv:1502.03167). Covers motivation, the bias-cancellation proof of why naive whitening fails, the BN transform step by step, backpropagation through BN derived from scratch, scale invariance, the train/inference distinction, convolutional BN, regularization, and experiments.
> 
> **Paper:** _Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_ · Sergey Ioffe, Christian Szegedy · Google Inc. · ICML 2015
> 
> **Companion files:** [[Layer Normalization]] (axis-switch fix for BN's failures) · [[Momentum]] (optimization theory — condition number, convergence) · [[Training Diagnostics and Optimizers]] (practical training heuristics) · [[Adam]] (adaptive optimizer that benefits from BN)

---

## Overview of Topics (in order)

1. [[#1. The Core Problem — Internal Covariate Shift|Internal Covariate Shift]] — what goes wrong in deep networks and why
2. [[#2. Why Whitening Is Ideal But Intractable|Whitening]] — the ideal fix and why it breaks gradient descent
3. [[#3. The Batch Normalization Transform|BN Transform (Algorithm 1)]] — the practical approximation, every step motivated
4. [[#4. Backpropagation Through Batch Norm|Backprop through BN]] — full gradient derivations from scratch
5. [[#5. The Scale-Invariance Property|Scale invariance]] — why BN enables high learning rates
6. [[#6. Training vs. Inference — A Critical Distinction|Training vs. Inference (Algorithm 2)]] — population statistics at test time
7. [[#7. Placing BN in the Network — Where and Why|Placement]] — before or after nonlinearity, and for convolutions
8. [[#8. BN as a Regularizer|Regularization effect]] — why BN partially replaces Dropout
9. [[#9. Experimental Results|Experiments]] — MNIST, ImageNet, ensemble SOTA
10. [[#10. Key Formulas — Quick Reference|Quick Reference]]
11. [[#11. Things BN Does NOT Do|Misconceptions]]
12. [[#12. Connections and Later Context|Broader context]] — LN, Group Norm, contested theory

---

## 1. The Core Problem — Internal Covariate Shift

> [!tip] See Also
> 
> - LN's direct response to this problem for RNNs/small batches: [[Layer Normalization#1. Why Batch Normalization Is Not Always Enough|LN §1]]
> - Why training slows down in general (condition number): [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]]
> - Practical loss curve symptoms: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|Loss Curve Diagnostics (TDO §5)]]

### What Is Covariate Shift?

**Covariate shift** (Shimodaira, 2000): when the input distribution of a learning system changes between training and test time. Classically handled by domain adaptation.

**Internal covariate shift**: the same phenomenon but applied _inside_ the network — the distribution of a layer's inputs changes _during training itself_, as the parameters of preceding layers change.

### Why This Slows Training

Consider a deep network:

```
ℓ = F₂( F₁(u, Θ₁), Θ₂ )
```

- F₁ maps raw input u → activations x
- F₂ takes x as its input

When Θ₁ updates, x shifts. Then Θ₂ must readjust to a new input distribution. This readjustment repeats every gradient step. The layers are not cooperating — they're chasing each other's moving targets.

> [!note] Connection to optimization theory From [[Momentum#5. Condition Number and Optimal Step Size|Momentum §5]], training is slow when the [[Momentum#5. Condition Number and Optimal Step Size|condition number κ = λₙ/λ₁]] is large. Internal covariate shift effectively increases the condition number seen by each layer — the input distribution changes, which changes the local curvature of the loss, which forces smaller learning rates. BN directly attacks this by stabilizing those input distributions.

### The Sigmoid Saturation Problem

A specific, severe instance. Consider:

```
z = g(Wu + b)
```

where `g(x) = 1/(1 + exp(−x))` is the sigmoid.

**The saturation issue:**

- When `|x| = |Wu + b|` is large, `g'(x) ≈ 0`
- Gradient flowing back through this layer **vanishes**
- Model trains extremely slowly

**Why internal covariate shift makes this worse:**

- W, b, and all lower-layer parameters update each step
- These updates shift the distribution of `x = Wu + b`
- Many dimensions of x drift into the saturated region (|x| ≫ 0)
- This compounds as the network deepens

> [!note] Historical fix: ReLU + careful initialization + small learning rates. BN provides a more principled solution that doesn't require switching activation functions.

> [!note] LN vs BN on sigmoid networks The experimental results (§9) show that BN-x5-Sigmoid achieves 69.8% on ImageNet — **enabling sigmoid networks that were previously completely untrainable**. This is the most striking demonstration of what internal covariate shift was costing. Compare: [[Layer Normalization#6. Layer Norm in Recurrent Neural Networks|LN §6]] also enables sigmoid RNNs by stabilizing hidden state distributions.

### Why Lower Learning Rates Are Forced

Without BN, a high learning rate causes a large update to Θ₁ → large shift in x → destabilizes F₂. To prevent this cascade, learning rates must stay small. This is exactly what BN eliminates — see [[#5. The Scale-Invariance Property|scale invariance (§5)]].

---

## 2. Why Whitening Is Ideal But Intractable

> [!tip] See Also
> 
> - LN's simplified whitening (per-example, no cross-feature correlation): [[Layer Normalization#2. The Core Idea — Normalize Over Features, Not Batch|LN §2]]
> - Why BN doesn't decorrelate either: [[#11. Things BN Does NOT Do|§11 Misconceptions]]

### The Ideal: Full Whitening

It has long been known (LeCun et al., 1998) that networks train faster if inputs are **whitened**:

- **Zero mean:** E[x] = 0
- **Unit variance:** Var[x] = 1
- **Decorrelated:** features are uncorrelated (off-diagonal covariance = 0)

If we could whiten the inputs _to every layer_ at every step, internal covariate shift would be eliminated by construction. The whitened activation would be:

```
Step 1: Compute covariance matrix
        Cov[x] = Eₓ∈X[xxᵀ] − E[x]E[x]ᵀ

Step 2: Compute inverse square root
        Cov[x]^(−1/2)

Step 3: Produce whitened activations
        x̂ = Cov[x]^(−1/2) · (x − E[x])
```

### Why This Breaks Gradient Descent — The Bias Cancellation Proof

This is the crucial argument the paper makes. It's easy to miss if you read it quickly.

**Setup:** Suppose we normalize just the mean, so `x̂ = x − E[x]` where `x = u + b`.

After a gradient step that updates b → b + Δb:

```
Step 1 (forward): new activation = u + (b + Δb)
Step 2 (normalize): x̂_new = u + (b + Δb) − E[u + (b + Δb)]
                           = u + b + Δb − E[u + b] − Δb
                           = u + b − E[u + b]
                           = x̂_old
```

**The Δb terms cancel exactly.** The normalization completely erases the effect of the bias update.

> [!important] Why this is catastrophic Gradient descent is computing ∂ℓ/∂b and updating b accordingly — but the normalization immediately undoes that update. The network _thinks_ it's moving b, but the normalized output doesn't change. In practice, b diverges to ±∞ while the loss stays fixed and the model stops learning entirely.
> 
> **The fix:** Make the normalization a _differentiable part of the computation graph_, so gradient descent can see and account for it. This is exactly what BN does — see [[#3. The Batch Normalization Transform|§3]].

### Computational Cost of Full Whitening

Computing `Cov[x]^(−1/2)` requires:

- Covariance matrix: O(d²) entries for d-dimensional activations
- Inverse square root: expensive eigendecomposition O(d³)
- Recomputed after every parameter update, over the whole training set
- Derivatives of these operations needed for backprop

This is prohibitively expensive. For a layer with 1000 hidden units, the covariance matrix has 10⁶ entries.

### The Two Necessary Simplifications BN Makes

1. **Normalize each feature independently** (no decorrelation) — just zero mean and unit variance per scalar dimension. Reduces O(d²) covariance to O(d) scalar statistics.
2. **Use mini-batch statistics** instead of full training-set statistics — allows normalization to participate in backpropagation cheaply.

> [!note] Simplification 1 vs. LN BN normalizes each scalar feature k across the batch: one μ and σ² per feature. [[Layer Normalization#3. The Layer Normalization Transform — Full Derivation|LN §3]] normalizes all features of one example together: one μ and σ² per example. Neither decorrelates; they just control mean and variance on different axes.

---

## 3. The Batch Normalization Transform

> [!tip] See Also
> 
> - LN's analogous transform (same four steps, different axis): [[Layer Normalization#3. The Layer Normalization Transform — Full Derivation|LN §3]]
> - Why γ and β are needed: [[Layer Normalization#5. Learnable Parameters — Re-scaling and Re-centering|LN §5]] (same argument, detailed)

BN is a differentiable transformation applied to a single scalar activation `x` over a mini-batch `B = {x₁, ..., xₘ}`. It has four steps.

### Algorithm 1 — BN Transform (Training)

**Input:** Mini-batch `B = {x₁, ..., xₘ}` (m scalar values of _one_ activation across m examples) **Parameters to learn:** γ (scale), β (shift) **Output:** `{yᵢ = BNᵧ,β(xᵢ)}`

---

**Step 1 — Mini-batch mean:** $$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

_Why:_ Estimate the mean from the mini-batch. This is an unbiased estimate of the true mean (E[μ_B] = E[x]). Subtracting it centers the distribution at zero, removing the global additive offset.

_What this achieves:_ After subtracting μ_B, the batch values {xᵢ − μ_B} have exactly zero sample mean: $\frac{1}{m}\sum_i (x_i - \mu_B) = 0$ by construction.

---

**Step 2 — Mini-batch variance:** $$\sigma^2_B = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

_Why:_ Estimate the spread. We normalize by this to make variance ≈ 1. Note: **biased estimator** (divides by m, not m−1). This is deliberate — we're normalizing for optimization, not doing unbiased statistical estimation. (The inference-time correction uses m/(m−1) — see [[#6. Training vs. Inference — A Critical Distinction|§6]].)

_What this achieves:_ σ²_B measures how spread out the batch values are around μ_B. Dividing by √σ²_B will make the spread approximately 1.

---

**Step 3 — Normalize:** $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \varepsilon}}$$

Breaking this into sub-steps:

_Step 3a — Center:_ Compute `x_i − μ_B`. The result has zero mean.

_Step 3b — Scale:_ Divide by `√(σ²_B + ε)`. The result has unit variance: $$\frac{1}{m}\sum_{i=1}^m \hat{x}_i^2 = \frac{1}{m}\sum_{i=1}^m \frac{(x_i - \mu_B)^2}{\sigma^2_B} = \frac{\sigma^2_B}{\sigma^2_B} = 1 \checkmark$$

_Step 3c — Why ε:_ Prevents division by zero when all m values in the batch are identical (σ²_B = 0). Typically ε = 10⁻⁵. Without ε, a dead activation (constant across the batch) would cause NaN.

> [!note] BN vs LN: what gets normalized Here, x̂ᵢ is the normalized version of _one example's activation_, computed using _all m examples in the batch_. In [[Layer Normalization#3. The Layer Normalization Transform — Full Derivation|LN]], the analogous normalized value uses _all H features of one example_. Same formula — completely different axis.

---

**Step 4 — Scale and shift:** $$y_i = \gamma \cdot \hat{x}_i + \beta$$

_Why γ and β are essential:_ Pure normalization (step 3 only) is **representationally destructive**. It forces every activation to have zero mean and unit variance — but the network might need a different operating point for a specific layer. Two critical examples:

1. **Sigmoid saturation fix breaks itself:** If we always force inputs to sigmoid into unit-variance range, the sigmoid is always in its linear regime. We've fixed the vanishing gradient problem but destroyed the nonlinearity we needed. γ lets the network _choose_ how much normalization to apply. Setting γ = √Var[x], β = E[x] exactly recovers the original distribution.
    
2. **Expressive power:** The BN layer with γ = 1, β = 0 is pure normalization. After learning, γ and β let the network specify any desired mean and variance. The network can _learn to undo BN_ if BN is unhelpful for that layer.
    

**Initialization:** γ = 1, β = 0 — pure normalization at the start, adapts during training.

> [!note] Naming convention The paper uses γ (scale) and β (shift). [[Layer Normalization]] uses **g** (gain) and **b** (bias) for the same parameters. In the Transformer literature, both are called γ and β. They are identical in role.

### What BN is NOT

BN is applied per-activation, per-mini-batch. It does **not** process each example independently — `BNᵧ,β(xᵢ)` depends on all m examples in the batch through μ_B and σ²_B. This batch-coupling is both BN's power (good statistics) and its weakness (fails at batch size 1) — see [[#8. BN as a Regularizer|§8]] and [[Layer Normalization#1. Why Batch Normalization Is Not Always Enough|LN §1]].

---

## 4. Backpropagation Through Batch Norm

> [!tip] See Also
> 
> - Analogous derivation for LN (same structure, features instead of batch): [[Layer Normalization#4. Backpropagation Through Layer Norm — Complete Derivation|LN §4]]
> - Why making normalization differentiable solves the bias-cancellation problem: [[#2. Why Whitening Is Ideal But Intractable|§2]]

BN must be fully differentiable — gradients must flow through it. This is non-trivial because μ_B and σ²_B both depend on _all m examples_, so changing one xᵢ affects _all m normalized outputs_ x̂_j through the shared statistics.

### The Computation Graph

```
x₁, x₂, ..., xₘ ──→ μ_B ──→ x̂₁, x̂₂, ..., x̂ₘ ──→ y₁, y₂, ..., yₘ ──→ ℓ
                      ↑
x₁, x₂, ..., xₘ ──→ σ²_B ─┘
```

A single input xᵢ influences the loss through **three paths**:

1. Directly: xᵢ → x̂ᵢ → yᵢ → ℓ
2. Via the mean: xᵢ → μ_B → _all_ x̂_j → _all_ yⱼ → ℓ
3. Via the variance: xᵢ → σ²_B → _all_ x̂_j → _all_ yⱼ → ℓ

Because of paths 2 and 3, backprop through BN requires summing over all m examples in the batch — this is the multivariate chain rule in action.

> [!note] Why this is the same structure as LN backprop In [[Layer Normalization#4. Backpropagation Through Layer Norm — Complete Derivation|LN §4]], a single input aᵢ affects the loss through three paths: directly, via μ (average over H features), and via σ² (variance over H features). The derivation is structurally identical — just replace "sum over m examples" with "sum over H features".

### Given: Upstream Gradients

The layer after BN has already backpropagated, giving us ∂ℓ/∂yᵢ for all i = 1,...,m.

---

### Step 1: Gradient w.r.t. Normalized Activations x̂ᵢ

The output yᵢ = γ·x̂ᵢ + β is linear in x̂ᵢ: $$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \frac{\partial y_i}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

_Intuition:_ γ is the gain on x̂ᵢ. If γ is large, changes in x̂ᵢ matter a lot downstream — gradient amplified by γ.

---

### Step 2: Gradient w.r.t. Variance σ²_B

σ²_B appears in every x̂_j through the denominator. Changing σ²_B changes _all m outputs_. Sum all paths: $$\frac{\partial \ell}{\partial \sigma^2_B} = \sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma^2_B}$$

The local derivative: $\hat{x}_j = (x_j - \mu_B)(\sigma^2_B + \varepsilon)^{-1/2}$, so: $$\frac{\partial \hat{x}_j}{\partial \sigma^2_B} = (x_j - \mu_B) \cdot \left(-\frac{1}{2}\right)(\sigma^2_B + \varepsilon)^{-3/2}$$

Therefore: $$\frac{\partial \ell}{\partial \sigma^2_B} = \sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j} \cdot (x_j - \mu_B) \cdot \left(-\frac{1}{2}\right)(\sigma^2_B + \varepsilon)^{-3/2}$$

_Intuition:_ The variance controls the "zoom level" of normalization. If σ²_B increases, the denominator grows, all x̂_j shrink toward zero. The gradient accumulates these shrinkage effects across all m examples.

---

### Step 3: Gradient w.r.t. Mean μ_B

μ_B appears in **two places** in the forward pass: directly in each numerator `(xⱼ − μ_B)`, and indirectly inside σ²_B through `(xⱼ − μ_B)²`. Apply the multivariate chain rule: $$\frac{\partial \ell}{\partial \mu_B} = \underbrace{\sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j} \cdot \frac{-1}{\sqrt{\sigma^2_B + \varepsilon}}}_{\text{Path 1: direct}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial \mu_B}}_{\text{Path 2: via variance}}$$

**Path 2 evaluates to zero** — this is a beautiful cancellation: $$\frac{\partial \sigma^2_B}{\partial \mu_B} = \frac{1}{m}\sum_{j=1}^m 2(x_j - \mu_B)(-1) = \frac{-2}{m}\sum_{j=1}^m (x_j - \mu_B)$$

But $\sum_{j=1}^m (x_j - \mu_B) = \sum x_j - m\mu_B = m\mu_B - m\mu_B = 0$.

> [!note] Why this cancellation happens The sum of deviations from the mean is always exactly zero — a fundamental property of the arithmetic mean. This means the variance is locally insensitive to small changes in the mean (while holding the xⱼ fixed). The same cancellation occurs in [[Layer Normalization#4. Backpropagation Through Layer Norm — Complete Derivation|LN §4, Step 3]].

Therefore, Path 2 = 0, and: $$\frac{\partial \ell}{\partial \mu_B} = \frac{-1}{\sqrt{\sigma^2_B + \varepsilon}} \sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j}$$

---

### Step 4: Gradient w.r.t. Input xᵢ

A single xᵢ reaches the loss through three paths: $$\frac{\partial \ell}{\partial x_i} = \underbrace{\frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2_B + \varepsilon}}}_{\text{Self}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{2(x_i - \mu_B)}{m}}_{\text{Via } \sigma^2_B} + \underbrace{\frac{\partial \ell}{\partial \mu_B} \cdot \frac{1}{m}}_{\text{Via } \mu_B}$$

The local derivatives used:

- $\partial \hat{x}_i / \partial x_i = 1/\sqrt{\sigma^2_B + \varepsilon}$ (xᵢ in numerator)
- $\partial \sigma^2_B / \partial x_i = 2(x_i - \mu_B)/m$ (xᵢ in squared term of variance)
- $\partial \mu_B / \partial x_i = 1/m$ (xᵢ is one of m terms in the average)

Substituting everything: $$\frac{\partial \ell}{\partial x_i} = \frac{1}{m\sqrt{\sigma^2_B + \varepsilon}} \left[ m\frac{\partial \ell}{\partial \hat{x}_i} - \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j}\hat{x}_j \right]$$

> [!note] What this compact form reveals The gradient has three terms: (1) the raw upstream gradient scaled up by m, (2) a correction that subtracts the _mean_ of all upstream gradients (mean-centering the gradient), (3) a projection correction that removes the component of the gradient aligned with x̂ᵢ (variance-normalizing the gradient). **The normalization in the forward pass induces a dual de-normalization structure in the backward pass.** Compare [[Layer Normalization#4. Backpropagation Through Layer Norm — Complete Derivation|LN §4, Step 4]] — identical structure, H replaces m.

---

### Step 5: Gradients w.r.t. Learnable Parameters γ and β

γ and β are shared across all m examples in the batch, so their gradients sum across the batch:

**For β (shift):** $$\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}$$ _Intuition:_ β is a universal additive offset. Total upstream gradient.

**For γ (scale):** $$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i$$ _Intuition:_ γ scales the normalized data. Its gradient is the dot product of upstream error and the data it was scaling.

> [!important] Why making this differentiable solves the bias cancellation problem In [[#2. Why Whitening Is Ideal But Intractable|§2]], naive normalization outside the gradient let the bias update cancel. Here, because gradient descent _sees_ ∂ℓ/∂μ_B and ∂ℓ/∂σ²_B explicitly — and these in turn account for how μ_B and σ²_B depend on all parameters — the optimizer knows normalization happened. There's no more cancellation: the gradients flowing through BN correctly account for the normalization. The bias b of the preceding linear layer becomes redundant (absorbed by β) and can be dropped.

---

## 5. The Scale-Invariance Property

> [!tip] See Also
> 
> - This is why BN allows higher LR in practice: [[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|LR Annealing (TDO §11)]]
> - LN has the same scale invariance property for the same reason: [[Layer Normalization#17. Connections to Broader Landscape|LN §17]]
> - Adam's gradient rescaling invariance is a related concept: [[Adam#3. Adam's Update Rule — Geometry and Intuition|Adam §3]]

### Scale Invariance of BN Output

For any scalar `a ≠ 0`: $$\text{BN}((aW)u) = \text{BN}(Wu)$$

_Why:_ Scaling W by a scales `x = Wu` by a. But BN normalizes by the standard deviation of x, which also scales by a. The a cancels: $$\hat{x} = \frac{aWu - \mu_{aB}}{\sigma_{aB}} = \frac{aWu - a\mu_B}{a\sigma_B} = \frac{Wu - \mu_B}{\sigma_B}$$

### Gradient Scale Invariance

The gradient w.r.t. the input u is unchanged: $$\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(Wu)}{\partial u}$$

The gradient w.r.t. W _decreases_ as a increases: $$\frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(Wu)}{\partial W}$$

> [!important] Why this enables high learning rates Without BN: if W grows large (due to a high learning rate step), gradients amplify through W and can cause explosion. With BN: larger W → gradients through W are _smaller_ (factor 1/a). The system is **self-regulating**. You can use a high learning rate and the network naturally resists blowup.
> 
> This is the mechanism behind BN's most famous practical benefit: allowing 5–30× higher learning rates than without BN (see [[#9. Experimental Results|§9]]).

### Conjecture: Singular Values Near 1

The authors conjecture (not fully proven) that BN encourages the Jacobian J of each layer to have singular values close to 1:

_Argument:_ After BN, both input x̂ and output ẑ of a layer have zero mean and unit variance. If we model F(x̂) ≈ Jx̂: $$I = \text{Cov}[\hat{z}] = J \cdot \text{Cov}[\hat{x}] \cdot J^\top = J \cdot I \cdot J^\top = JJ^\top$$

So JJ^T = I, meaning all singular values of J equal 1, meaning **gradient magnitudes are preserved** during backprop.

> [!note] Contested explanation Santurkar et al. (2018) "How Does Batch Normalization Help Optimization?" argue the real mechanism is **smoothing the loss landscape** — making the gradient more Lipschitz-continuous, reducing gradient variance across batches. This makes the optimization landscape easier to traverse regardless of the singular value argument. The debate connects to [[Momentum#5. Condition Number and Optimal Step Size|condition number theory in Momentum §5]]: BN effectively reduces the condition number seen by gradient descent.

---

## 6. Training vs. Inference — A Critical Distinction

> [!tip] See Also
> 
> - LN has **no** train/inference distinction — this is one of its key advantages: [[Layer Normalization#9. Training vs. Inference — A Key Advantage Over BN|LN §9]]
> - Practical debugging note: forgetting eval() mode is a common bug: [[Training Diagnostics and Optimizers#3. Practical Gradcheck Tips|Gradcheck Tips (TDO §3)]]

> [!warning] Most common BN implementation bug Forgetting to switch to inference mode (e.g., `model.eval()` in PyTorch) causes the network to use stochastic batch statistics at inference, producing different outputs each time — even for identical inputs. This bug is silent (no error, just wrong outputs).

### The Problem with Mini-Batch Statistics at Inference

During training, BN normalizes using batch statistics μ_B and σ²_B. But at inference:

- You may be processing a single example (batch size = 1) → σ²_B = 0, normalization undefined
- Even with batch size > 1, the output should not depend on which other examples happen to be in the batch

**We want inference to be deterministic:** same input → same output, every time.

### Solution: Population Statistics

Once training is complete, compute **fixed** statistics over the entire training set:

**Population mean** (via averaging mini-batch means): $$\mathbb{E}[x] = \mathbb{E}_B[\mu_B]$$

_In practice:_ maintained as a running exponential moving average during training: $$\mu_{\text{running}} \leftarrow (1-\alpha)\mu_{\text{running}} + \alpha \cdot \mu_B$$

**Unbiased population variance** (Bessel's correction): $$\text{Var}[x] = \frac{m}{m-1} \cdot \mathbb{E}_B[\sigma^2_B]$$

_Why m/(m−1)?_ The mini-batch uses a biased variance (1/m). Multiplying by m/(m−1) corrects this to an unbiased estimate of the population variance. For large m this correction is tiny, but at small batch sizes it matters.

### Inference Transform (Fixed, Linear)

At inference time: $$\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \varepsilon}}$$ $$y = \gamma \cdot \hat{x} + \beta$$

Since E[x], Var[x], γ, β, ε are all **constants** at inference, the entire BN transform is: $$y = \underbrace{\frac{\gamma}{\sqrt{\text{Var}[x] + \varepsilon}}}_{\text{slope}} \cdot x + \underbrace{\left(\beta - \frac{\gamma \cdot \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \varepsilon}}\right)}_{\text{intercept}}$$

This is a single affine transformation that can be **fused into the preceding linear layer** (absorb into W and b) — **zero extra computation at inference**.

> [!note] Contrast with LN [[Layer Normalization#9. Training vs. Inference — A Key Advantage Over BN|LN §9]] has no such distinction. Its statistics are computed from the current example's own activations — no accumulated population statistics, no mode switching. This makes LN simpler to deploy and immune to distribution shift between training and inference batches.

---

## 7. Placing BN in the Network — Where and Why

> [!tip] See Also
> 
> - LN placement in Transformers (Pre-LN vs Post-LN): [[Layer Normalization#7. Layer Norm in the Transformer Architecture|LN §7]]
> - Gradient flow argument for placement: [[Momentum#8. Four Convergence Regimes|Convergence Regimes (Momentum §8)]] — Pre-LN preserves gradient highway

### Before vs. After Nonlinearity

**The paper's recommendation:** Apply BN to `x = Wu + b` **before** the nonlinearity g:

```
z = g(BN(Wu))
```

(The bias b is dropped — it's absorbed by the learned β.)

**Why before?**

- `Wu + b` is more likely to be approximately Gaussian (by CLT-like arguments — sums of many weights × inputs)
- Normalizing a roughly Gaussian distribution to unit variance is well-defined
- After a nonlinearity (e.g., ReLU), the distribution is non-negative and highly non-Gaussian — normalizing it is less principled

> [!note] Later work (He et al. 2016 ResNets) sometimes places BN after the nonlinearity. This remains empirically debated. In Transformers, the analogous question is Pre-LN vs Post-LN — see [[Layer Normalization#7. Layer Norm in the Transformer Architecture|LN §7]], where Pre-LN wins for deep networks.

### Dropping the Bias b

Since BN includes a mean-subtraction step, the bias b is redundant:

```
BN(Wu + b) → subtracts mean → b cancels
```

The learned β plays the role of bias after normalization. So replace `z = g(Wu + b)` with `z = g(BN(Wu))`. Fewer parameters, no redundancy.

### For Convolutional Layers

**Challenge:** In a conv layer, the same filter slides over all spatial positions. We want to respect translational equivariance — the normalization should treat different spatial locations of the same feature map equivalently.

**Solution:** Treat all activations in a feature map across all spatial locations _and_ all mini-batch examples as one group for computing statistics.

For a mini-batch of size m and feature map of size p × q: $$\text{Effective mini-batch size: } m' = m \cdot p \cdot q$$

Compute one μ_B and one σ²_B **per feature map**. Learn one γ^(k) and one β^(k) **per feature map**.

_Why:_ A feature map represents a single learned concept (e.g., "edge detector"). The same concept should be normalized consistently regardless of where in the image it fires.

> [!note] Contrast with LN on sequences [[Layer Normalization#7. Layer Norm in the Transformer Architecture|LN in Transformers]] normalizes the d_model-dimensional embedding vector of each token independently. There's no spatial grouping — each token position is its own normalization unit. This is the key structural difference that makes LN natural for sequences and BN natural for images.

---

## 8. BN as a Regularizer

> [!tip] See Also
> 
> - Dropout (what BN partially replaces): [[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies|Early Stopping & Dropout (GDO §3)]]
> - Stochastic gradients as implicit regularizer: [[Momentum#15. Stochastic Gradients|Stochastic Gradients (Momentum §15)]]
> - LN does NOT regularize like BN: [[Layer Normalization#14. Things LN Does NOT Do|LN §14 Misconceptions]]

### The Mechanism

During training, `BN(xᵢ)` is not a deterministic function of xᵢ alone — it depends on which other xⱼ (j ≠ i) happen to be in the same mini-batch. Different mini-batch compositions → different normalization → different gradients.

This introduces **stochastic noise** into the training signal — similar in spirit to Dropout, which randomly zeros activations. The network cannot overfit to any single example's exact activation values, because those values are normalized relative to a random mini-batch.

**Mechanism connection to [[Momentum#15. Stochastic Gradients|Momentum §15]]:** The stochastic noise in BN is analogous to the noise in stochastic gradients — both are unbiased perturbations that act as implicit regularizers, helping find flatter minima that generalize better.

### The Batch Size Dependence

The regularization effect **depends on batch size**:

- **Larger batches** → mini-batch statistics ≈ population statistics → less randomness → less regularization
- **Smaller batches** → more noise per example → stronger regularization, but noisier training

This is why very large batch training can hurt generalization even though it should be computationally efficient — the BN regularization effect is diminished. And it's why [[Layer Normalization#8. Geometric and Statistical Intuition|LN]] (which is deterministic per example) provides essentially no regularization.

### Shuffling Training Examples

The paper recommends thorough shuffling. Reason: if the same examples always appear together in a mini-batch, the noise from random mini-batch composition is reduced, weakening the regularization. This is distinct from the gradient-level shuffling in [[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies|Shuffling (GDO §3)]].

---

## 9. Experimental Results

### 9.1 MNIST — Proof of Concept

**Setup:** Simple network, 3 fully-connected hidden layers of 100 sigmoid units each. Trained for 50,000 steps with batch size 60.

**Results:**

- BN network reaches higher test accuracy in far fewer steps
- Without BN: sigmoid inputs drift widely in mean and variance over training (measured at 15th, 50th, 85th percentiles of the distribution)
- With BN: sigmoid input distribution remains stable throughout training

**Key takeaway:** BN demonstrably reduces internal covariate shift, stabilizing the distribution of inputs to each sigmoid.

> [!note] BN-x5-Sigmoid (a separate experiment) The same sigmoid architecture with BN and 5× higher LR achieves 69.8% on ImageNet. Without BN, sigmoid networks are entirely untrainable on ImageNet (stay at chance ≈ 0.1%). This is the single most dramatic demonstration in the paper — it shows that sigmoid networks were failing not because of architecture weakness but because of internal covariate shift.

### 9.2 ImageNet — The Main Result

**Architecture:** Modified Inception (GoogLeNet variant). **Baseline:** LR = 0.0015, batch size 32.

**BN modifications applied:**

|Model|Steps to reach 72.2% top-1|Max accuracy|
|---|---|---|
|Inception (baseline)|31.0 × 10⁶|72.2%|
|BN-Baseline (just add BN)|13.3 × 10⁶|72.7%|
|BN-x5 (LR × 5)|2.1 × 10⁶|73.0%|
|BN-x30 (LR × 30)|2.7 × 10⁶|**74.8%**|
|BN-x5-Sigmoid|—|69.8%|

**The headline number:** BN-x5 achieves the same accuracy as Inception with **14× fewer training steps**.

### Accelerating BN Networks — The Full Recipe

Simply adding BN doesn't capture all the benefits. The paper applies these modifications together:

1. **Increase learning rate** — scale invariance makes high LR safe (see [[#5. The Scale-Invariance Property|§5]])
2. **Remove Dropout** — BN provides regularization (see [[#8. BN as a Regularizer|§8]])
3. **Reduce L2 weight decay by 5×** — BN regularizes, so less explicit regularization needed
4. **Decay LR 6× faster** — the network trains faster, so the schedule should be compressed (see [[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|LR Annealing (TDO §11)]])
5. **Remove Local Response Normalization (LRN)** — BN makes LRN redundant
6. **Shuffle training data more thoroughly** — enhances BN's regularization (see [[#8. BN as a Regularizer|§8]])
7. **Reduce photometric distortions** — network sees fewer training steps, so spend them on realistic images

### 9.3 Ensemble — ImageNet SOTA (at time of publication)

6 BN-x30 networks ensembled:

- Top-5 validation error: **4.9%**
- Top-5 test error: **4.82%** (ILSVRC server)
- Previous best (He et al., 2015): 4.94%

---

## 10. Key Formulas — Quick Reference

|Quantity|Formula|
|---|---|
|Mini-batch mean|$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$|
|Mini-batch variance|$\sigma^2_B = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$|
|Normalized activation|$\hat{x}_i = (x_i - \mu_B)/\sqrt{\sigma^2_B + \varepsilon}$|
|BN output (train)|$y_i = \gamma \hat{x}_i + \beta$|
|Population mean (inference)|$\mathbb{E}[x] = \mathbb{E}_B[\mu_B]$|
|Population variance (inference)|$\text{Var}[x] = \frac{m}{m-1}\mathbb{E}_B[\sigma^2_B]$|
|BN output (inference, fused)|$y = \frac{\gamma}{\sqrt{\text{Var}[x]+\varepsilon}} \cdot x + \left(\beta - \frac{\gamma\mathbb{E}[x]}{\sqrt{\text{Var}[x]+\varepsilon}}\right)$|
|Gradient w.r.t. x̂ᵢ|$\partial\ell/\partial\hat{x}_i = (\partial\ell/\partial y_i)\cdot\gamma$|
|Gradient w.r.t. σ²_B|$\sum_j (\partial\ell/\partial\hat{x}_j)(x_j-\mu_B)(-\frac{1}{2})(\sigma^2_B+\varepsilon)^{-3/2}$|
|Gradient w.r.t. μ_B|$\frac{-1}{\sqrt{\sigma^2_B+\varepsilon}}\sum_j \partial\ell/\partial\hat{x}_j$|
|Gradient w.r.t. xᵢ (full)|$\frac{1}{m\hat{\sigma}}\left[m\frac{\partial\ell}{\partial\hat{x}_i} - \sum_j\frac{\partial\ell}{\partial\hat{x}_j} - \hat{x}_i\sum_j\frac{\partial\ell}{\partial\hat{x}_j}\hat{x}_j\right]$|
|Gradient w.r.t. γ|$\sum_i (\partial\ell/\partial y_i)\cdot\hat{x}_i$|
|Gradient w.r.t. β|$\sum_i \partial\ell/\partial y_i$|
|Scale invariance|$\text{BN}((aW)u) = \text{BN}(Wu)$|
|Gradient scale invariance|$\partial\text{BN}((aW)u)/\partial(aW) = \frac{1}{a}\cdot\partial\text{BN}(Wu)/\partial W$|
|Conv BN effective batch size|$m' = m \cdot p \cdot q$ (p×q spatial, m images)|

---

## 11. Things BN Does NOT Do

> [!warning] Misconceptions — each of these is a commonly believed falsehood

**BN does NOT decorrelate features.** It normalizes each dimension independently. Full whitening requires the full covariance matrix — O(d²) — which BN deliberately avoids. Compare [[Layer Normalization#14. Things LN Does NOT Do|LN §14]] — LN also doesn't decorrelate.

**BN does NOT guarantee zero mean and unit variance at all times.** The _intermediate_ x̂ᵢ has zero mean and unit variance. The _output_ yᵢ = γx̂ᵢ + β has mean β and variance γ². After learning, γ and β can be anything.

**BN does NOT eliminate vanishing gradients entirely.** It substantially helps with sigmoid networks, but gradients can still vanish in very deep networks if the architecture is not carefully designed.

**BN at inference is NOT the same as at training.** Training uses stochastic batch statistics. Inference uses frozen population statistics. Forgetting `model.eval()` is a silent bug — wrong outputs with no error.

**BN does NOT work at batch size 1.** σ²_B = 0 for one example. This is the primary motivation for [[Layer Normalization]] (and Instance Norm, Group Norm).

**BN does NOT solve the problem of distribution shift between train and inference.** If the test distribution differs from training, the frozen population statistics will be mismatched. LN avoids this entirely since it computes per-example statistics.

---

## 12. Connections and Later Context

> [!tip] See Also
> 
> - Full comparison table BN vs LN vs RMSNorm vs GroupNorm: [[Layer Normalization#16. Quick Comparison — LN vs. BN vs. RMSNorm vs. GroupNorm|LN §16]]
> - How BN connects to optimization landscape smoothing: [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]]

**Layer Normalization (Ba et al., 2016):** Normalizes across features (not batch). Works for batch size 1, RNNs, variable-length sequences. Full treatment: [[Layer Normalization]]. The key relationship: **BN and LN are the same four-step algorithm with the normalization axis rotated 90°** — BN normalizes columns of the data matrix, LN normalizes rows.

**Instance Normalization:** Normalizes per sample per channel. Standard in style transfer. Each channel of each image normalized separately — extreme case of LN for spatial data.

**Group Normalization (Wu & He, 2018):** Compromise between LN (G=1, all channels) and Instance Norm (G=C, each channel). Normalizes within groups of channels per example. Works well for small batches in vision tasks where BN fails.

**Why BN's theoretical explanation is contested:** The original paper attributes BN's benefit to reducing internal covariate shift. Santurkar et al. (2018) argue that BN's real effect is **smoothing the loss landscape** — making the gradient of the loss more predictive and reducing its Lipschitz constant. This makes the optimization landscape easier to traverse. This connects directly to [[Momentum#5. Condition Number and Optimal Step Size|condition number theory]]: BN reduces the effective κ seen by each layer's optimizer, enabling larger and more stable gradient steps.

**BN in ResNets:** BN is placed after conv and before ReLU (sometimes after ReLU, debated). BN is partly responsible for why ResNets can be trained to 100+ layers — it keeps gradient magnitudes stable across layers, preventing the [[#1. The Core Problem — Internal Covariate Shift|internal covariate shift]] that would otherwise derail 50-layer+ networks.

**BN and the Adam connection:** [[Adam]] already addresses non-uniform gradient scales per-parameter. BN addresses non-uniform input scales per-layer. Together they tackle the ill-conditioning problem from two angles — see [[Adam#7. Relationship to Other Optimizers|Adam §7]] for the natural gradient / curvature connection.

---

## 13. Concept Map — How Everything Connects

```
Problem: Internal Covariate Shift (§1)
  │
  ├── Consequence: small LR forced ←→ [[Momentum#5|κ large → slow convergence]]
  ├── Consequence: careful init required
  └── Consequence: sigmoid saturation → vanishing gradients
  
Ideal Fix: Full Whitening (§2)
  │
  └── Problem: bias cancellation (normalization fights gradient descent)
      + O(d²) covariance matrix (computationally intractable)
      → Two simplifications: per-feature, per-mini-batch

BN Solution: Normalize within backprop, per dimension, per mini-batch (§3)
  │
  ├── Differentiable → gradients flow through (§4)
  │     └── Three-path chain rule: self, via σ², via μ
  │         └── Same structure in [[Layer Normalization#4|LN §4]] (H replaces m)
  │
  ├── γ, β parameters → restore representational power (§3)
  │     └── Network can invert BN if needed
  │
  ├── Scale invariance → enables high LR (§5)
  │     └── BN((aW)u) = BN(Wu) → self-stabilizing under large updates
  │         └── Same argument applies to [[Layer Normalization#17|LN §17]]
  │
  ├── Train ≠ Inference: population statistics required (§6)
  │     └── Key contrast: [[Layer Normalization#9|LN §9]] has NO inference distinction
  │
  ├── Conv layers: normalize per feature map (§7)
  │     └── Contrast: [[Layer Normalization#7|LN §7]] normalizes per token
  │
  └── Regularization via stochastic batch noise (§8)
        └── [[Momentum#15|Momentum §15]]: stochastic noise as implicit regularizer
        └── Contrast: [[Layer Normalization#14|LN §14]] — LN does NOT regularize

Limitations → spawn LN:
  ├── Batch size 1 → σ²_B = 0 → undefined
  ├── RNNs → variable length, ragged batch
  └── Train/inference gap → deployment complexity
  → Solution: [[Layer Normalization]] (rotate axis from batch to features)

Experiments (§9):
  ├── MNIST: stable sigmoid distribution, faster convergence
  ├── ImageNet: 14× fewer steps (BN-x5), SOTA ensemble
  └── Most dramatic: BN-x5-Sigmoid achieves 69.8% — sigmoid was not "broken",
      it was suffering from internal covariate shift
```

---

## 14. Quick Comparison — BN vs. Related Techniques

|Property|Batch Norm|Layer Norm|RMSNorm|Group Norm|Dropout|
|---|---|---|---|---|---|
|Normalizes over|Batch (per feature)|Features (per example)|Features, RMS only|Feature groups (per example)|—|
|Works at batch size 1|**No**|Yes|Yes|Yes|Yes|
|Train ≠ Inference|**Yes (critical)**|No|No|No|Yes|
|Running statistics|Required|Not needed|Not needed|Not needed|—|
|Mean subtraction|Yes|Yes|**No**|Yes|—|
|Learned parameters|γ, β per feature|g, b per feature|g per feature|g, b per feature|None|
|Regularizes|**Yes** (batch noise)|Weakly|Weakly|Weakly|Yes (primary)|
|Typical use case|CNNs, vision|Transformers, RNNs|LLMs|Small-batch vision|Any|
|Full notes|Here|[[Layer Normalization]]|[[Layer Normalization#11. Variants — RMS Norm, Group Norm, Instance Norm\|LN §11]]|[[Layer Normalization#11. Variants — RMS Norm, Group Norm, Instance Norm\|LN §11]]|[[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies\|GDO §3]]|

---

_Notes compiled from Ioffe & Szegedy (2015), arXiv:1502.03167v3 — Musaib, 19/05/2026_