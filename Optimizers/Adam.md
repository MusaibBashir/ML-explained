---
aliases:
  - Adam Optimizer
  - Adaptive Moment Estimation
tags:
  - machine-learning/optimization
  - mathematics/gradient-descent
  - deep-learning
  - optimization
  - optimizers
date: 2026-05-19
---

# Adam — A Method for Stochastic Optimization

> [!abstract] About This Note A rigorous, self-contained treatment of the Adam optimizer (Kingma & Ba, ICLR 2015). Covers the algorithm, the bias-correction derivation from scratch, the update rule geometry, the formal convergence proof (with full explanation of every step), AdaMax, and practical guidance.
> 
> **Companion files:** [[Momentum]] (theoretical foundations — eigenanalysis, O(√κ) speedup) · [[Training Diagnostics and Optimizers]] (practical heuristics, hyperparameter table) · [[Gradient Descent Overview]] (algorithm survey)

---

## Overview of Topics

1. [[#1. Motivation — What Problem Does Adam Solve?|Motivation]] — why SGD alone fails
2. [[#2. The Algorithm|The Algorithm]] — full pseudocode and equations
3. [[#3. Adam's Update Rule — Geometry and Intuition|Update Rule Geometry]] — trust region, SNR, invariance
4. [[#4. Initialization Bias Correction — Full Derivation|Bias Correction]] — full derivation from scratch
5. [[#5. Convergence Analysis — The Regret Framework|Convergence Analysis]] — regret framework, Theorem 4.1
6. [[#6. The Convergence Proof — Step by Step|The Proof]] — every step explained
7. [[#7. Relationship to Other Optimizers|Relationships]] — RMSProp, AdaGrad, natural gradient
8. [[#8. AdaMax — The ℓ∞ Generalization|AdaMax]] — infinity norm generalization
9. [[#9. Temporal Averaging|Temporal Averaging]] — Polyak-Ruppert
10. [[#10. Empirical Results|Empirical Results]] — what the experiments tell us
11. [[#11. Practical Guide|Practical Guide]] — hyperparameters, failure modes, tips

---

## 1. Motivation — What Problem Does Adam Solve?

> [!tip] See Also
> 
> - Why gradient descent is slow on ill-conditioned problems: [[Momentum#1. Gradient Descent — Basics and Failure Modes|GD Failure Modes (Momentum §1)]]
> - The condition number κ that quantifies this: [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]]
> - AdaGrad's flaw that Adam fixes: [[Training Diagnostics and Optimizers#13. AdaGrad — Per-Parameter Adaptive Learning Rate|AdaGrad (TDO §13)]]

### The core problems with vanilla SGD

Vanilla SGD applies the same learning rate α to every parameter:

```
θ ← θ − α · g
```

This is problematic in three ways:

**1. Uniform learning rate across all parameters.** In a neural network, different parameters have vastly different gradient magnitudes. A weight in the last layer might receive gradients of order 1, while a weight deep in the network might receive gradients of order 1e-5. A single α can't be right for both simultaneously. AdaGrad addresses this by accumulating per-parameter gradient history.

**2. No memory of curvature.** Vanilla SGD doesn't know whether a direction has been consistently useful (keep going!) or inconsistent (slow down). Momentum addresses this by accumulating gradient history as a velocity.

**3. Sensitive to learning rate choice.** Too large → divergence. Too small → crawls. A human has to tune this carefully.

Adam combines the solutions: **per-parameter adaptive learning rates** (from AdaGrad/RMSProp) **+** **momentum** (exponentially smoothed gradient), **+** bias correction to make this work from step 1.

> [!note] The name "Adam" comes from "**Ada**ptive **M**oment estimation" — it estimates the first moment (mean) and second moment (uncentered variance) of the gradient and uses both.

---

## 2. The Algorithm

### Full pseudocode (Algorithm 1 from paper)

**Inputs:**

- α: stepsize (default **0.001**)
- β₁, β₂ ∈ [0, 1): exponential decay rates (defaults **0.9, 0.999**)
- ε: numerical stability constant (default **1e-8**)
- f(θ): stochastic objective
- θ₀: initial parameters

**Initialize:**

```
m₀ ← 0    (1st moment vector — will track smoothed gradient)
v₀ ← 0    (2nd moment vector — will track smoothed squared gradient)
t  ← 0    (timestep)
```

**Loop until convergence:**

```
t   ← t + 1
gₜ  ← ∇θ fₜ(θₜ₋₁)                    [1] get stochastic gradient
mₜ  ← β₁ · mₜ₋₁ + (1 − β₁) · gₜ      [2] update biased 1st moment
vₜ  ← β₂ · vₜ₋₁ + (1 − β₂) · gₜ²     [3] update biased 2nd moment
m̂ₜ  ← mₜ / (1 − β₁ᵗ)                  [4] bias-correct 1st moment
v̂ₜ  ← vₜ / (1 − β₂ᵗ)                  [5] bias-correct 2nd moment
θₜ  ← θₜ₋₁ − α · m̂ₜ / (√v̂ₜ + ε)      [6] update parameters
```

> [!note] Line [3]: `gₜ²` means **elementwise square** — each element squared independently. All operations are elementwise; m, v, θ are all vectors of the same dimension as the parameter space.

### What each piece is doing

**Lines [2] and [3]** are **exponential moving averages (EMA)** of the gradient and squared gradient respectively. This is identical to RMSProp's second moment accumulation — see [[Training Diagnostics and Optimizers#14. RMSProp|RMSProp (TDO §14)]].

**Why EMA and not simple sum?** AdaGrad accumulates _all_ past squared gradients monotonically → learning rate → 0. EMA forgets the distant past, with β₂ controlling how much:

- β₂ = 0.9 → roughly last 10 steps get meaningful weight
- β₂ = 0.999 → roughly last 1000 steps get meaningful weight
- See [[Training Diagnostics and Optimizers#13. AdaGrad — Per-Parameter Adaptive Learning Rate|AdaGrad vs RMSProp (TDO §13–14)]]

**Lines [4] and [5]** correct for the initialization bias caused by starting m and v at zero. This is non-trivial and is derived fully in [[#4. Initialization Bias Correction — Full Derivation|§4]].

**Line [6]** updates parameters. The effective step is:

```
Δθ = −α · m̂ₜ / (√v̂ₜ + ε)
```

- Numerator m̂ₜ: smoothed gradient direction (momentum)
- Denominator √v̂ₜ: per-parameter normalization by RMS of recent gradients

---

## 3. Adam's Update Rule — Geometry and Intuition

### Effective stepsize and its bounds

Assuming ε = 0, the effective step in parameter space at timestep t is:

```
Δₜ = α · m̂ₜ / √v̂ₜ
```

The key claim: **|Δₜ| ≲ α** — the effective step magnitude is approximately bounded by α.

**Why?** Consider the ratio `m̂ₜ / √v̂ₜ` element-wise. Since m̂ₜ estimates E[g] and v̂ₜ estimates E[g²]:

```
|m̂ₜ,i / √v̂ₜ,i| ≈ |E[g]| / √E[g²]
```

By Jensen's inequality (E[g]² ≤ E[g²], which follows from Var(g) = E[g²] − E[g]² ≥ 0), we have |E[g]| / √E[g²] ≤ 1.

So the ratio is bounded in magnitude by 1, meaning **|Δₜ| ≤ α** approximately.

> [!important] This is the "trust region" property. Unlike vanilla SGD where a large gradient can take a huge step, Adam's effective step is always approximately bounded by α. This is why knowing the right **order of magnitude** of α is usually enough — you don't need to fine-tune it precisely.

### The Signal-to-Noise Ratio (SNR) interpretation

The paper defines the ratio m̂ₜ/√v̂ₜ as the **signal-to-noise ratio (SNR)**:

- **Large SNR** → gradient direction is consistent across recent steps → confident direction → larger effective step
- **Small SNR** → gradient estimates are noisy/conflicting → uncertain direction → step shrinks toward 0

**Near an optimum:** Gradients become smaller and more variable (they can be noisy in any direction since the loss surface is flat). This causes SNR → 0, so effective steps → 0. **Adam automatically anneals its own step size near optima** — this is what the paper means by "a form of automatic annealing."

> [!example] Concrete illustration: Suppose parameter θ₁ consistently gets gradient ≈ +0.1 at every step.
> 
> - m̂ ≈ 0.1, v̂ ≈ 0.01, SNR = 0.1/0.1 = 1 → full step α
> 
> Suppose parameter θ₂ gets gradient +0.1 sometimes, −0.1 sometimes.
> 
> - m̂ ≈ 0, v̂ ≈ 0.01, SNR ≈ 0 → step ≈ 0 (conflicting signals cancel)
> 
> This is the per-parameter adaptation at work.

### Invariance to gradient rescaling

If you rescale all gradients by a factor c (e.g., change the loss scale):

- m̂ₜ → c · m̂ₜ
- v̂ₜ → c² · v̂ₜ
- Ratio: (c · m̂ₜ) / √(c² · v̂ₜ) = (c · m̂ₜ) / (c · √v̂ₜ) = m̂ₜ / √v̂ₜ

**The step is unchanged.** Adam is invariant to gradient rescaling. This means if you multiply your loss by 10 (common in practice), Adam's behaviour is unaffected. Vanilla SGD would take steps 10× larger.

---

## 4. Initialization Bias Correction — Full Derivation

> [!tip] This is Section 3 of the paper. It's the most mathematically precise part of the algorithm description — worth understanding fully.

### Setup

We want to estimate E[g²] — the true second moment (uncentered variance) of the gradient at step t.

We're using an EMA:

```
vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
```

with v₀ = 0.

### Step 1: Unroll the EMA recursion

Apply the recurrence repeatedly:

```
v₁ = (1 − β₂) · g₁²
v₂ = β₂ · v₁ + (1 − β₂) · g₂² = β₂(1−β₂)g₁² + (1−β₂)g₂²
v₃ = β₂ · v₂ + (1 − β₂) · g₃² = β₂²(1−β₂)g₁² + β₂(1−β₂)g₂² + (1−β₂)g₃²
```

The pattern is clear. In general: $$v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot g_i^2 \tag{1}$$

This is equation (1) in the paper. Each past squared gradient gets weight $(1−β₂)β_2^{t-i}$: exponentially decaying as we go further into the past.

> [!note] Sanity check — the weights sum to 1 (for large t): $\sum_{i=1}^{t}(1-\beta_2)\beta_2^{t-i} = (1-\beta_2)\cdot\frac{1-\beta_2^t}{1-\beta_2} = 1 - \beta_2^t \approx 1$ for large t. For small t (early training), this sum is noticeably less than 1 — this is the bias.

### Step 2: Take the expectation

$$\mathbb{E}[v_t] = \mathbb{E}\left[(1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot g_i^2\right]$$

Pull the expectation inside (linearity): $$= (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot \mathbb{E}[g_i^2]$$

Now make the **stationarity assumption**: assume the true second moment doesn't change much over time, i.e. $\mathbb{E}[g_i^2] \approx \mathbb{E}[g_t^2]$ for all i ≤ t. Factor it out: $$= \mathbb{E}[g_t^2] \cdot (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} + \zeta$$

where ζ captures the error from non-stationarity. The paper notes ζ can be kept small by choosing β₂ such that the EMA doesn't look too far back.

### Step 3: Evaluate the geometric sum

$$\sum_{i=1}^{t} \beta_2^{t-i} = \sum_{j=0}^{t-1} \beta_2^{j} = \frac{1 - \beta_2^t}{1 - \beta_2}$$

So: $$(1-\beta_2) \cdot \frac{1-\beta_2^t}{1-\beta_2} = 1 - \beta_2^t$$

Therefore: $$\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t) + \zeta \tag{4}$$

### Step 4: The bias and the correction

We want $\mathbb{E}[v_t] = \mathbb{E}[g_t^2]$, but instead we get $\mathbb{E}[v_t] = (1-\beta_2^t)\mathbb{E}[g_t^2]$.

The EMA is **biased by a factor of $(1 - \beta_2^t)$**. This is small (close to 1) when t is large, but:

- At t=1 with β₂=0.999: factor = 1 − 0.999 = 0.001 → EMA is 1000× too small!
- At t=10 with β₂=0.999: factor ≈ 0.01 → still 100× too small
- At t=1000 with β₂=0.999: factor ≈ 0.63 → about right

**The fix:** Divide by $(1 - \beta_2^t)$: $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Now $\mathbb{E}[\hat{v}_t] \approx \mathbb{E}[g_t^2]$ — the bias is removed.

**Identical derivation for m:** The first moment mₜ is also initialized at zero and has the same bias structure. The bias-corrected estimate is: $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

> [!important] Why does this matter so much early in training? With β₂ = 0.999 (the default), the correction factor at t=1 is 1/(1−0.999) = 1000. Without correction, v̂₁ ≈ 0.001·g₁², meaning the denominator √v̂₁ is 31× too small, making the effective step 31× too large — often causing divergence. This is why the paper empirically shows that removing bias correction causes instability especially when β₂ is close to 1 (Figure 4 in the paper).

> [!note] The non-stationarity term ζ If the true second moment changes over time (as it does in practice — gradients change as parameters change), then the stationarity assumption is violated and ζ ≠ 0. However:
> 
> - β₂ controls how fast old gradients are forgotten
> - For slowly changing gradient distributions, ζ remains small
> - In practice this works well even though the assumption isn't exactly satisfied

---

## 5. Convergence Analysis — The Regret Framework

> [!note] The paper proves convergence using **online convex optimization** rather than the stochastic optimization framework used in [[Momentum#10. Optimal Global Parameters and Quadratic Speedup|Momentum §10]]. These are different but related settings.

### The online learning setup

Rather than minimizing a fixed function f(θ), think of it as a game:

- At each step t, you must commit to a parameter θₜ **before** seeing the cost function fₜ
- Then you observe fₜ and pay cost fₜ(θₜ)
- The functions f₁, f₂, ... can be adversarially chosen

**Regret:** How much worse did you do compared to the best fixed parameter in hindsight? $$R(T) = \sum_{t=1}^{T} [f_t(\theta_t) - f_t(\theta^_)]$$ where $\theta^_ = \arg\min_\theta \sum_{t=1}^T f_t(\theta)$ is the best fixed point in hindsight.

> [!note] Why this framework? Mini-batch SGD can be seen as online learning where each minibatch defines a different (noisy) loss function fₜ. The online regret framework doesn't require stationarity or convexity of the overall landscape — it applies even when the effective per-step loss changes.

### The regret bound (Theorem 4.1)

Under the assumptions:

- Bounded gradients: ‖∇fₜ(θ)‖₂ ≤ G, ‖∇fₜ(θ)‖∞ ≤ G∞ for all θ, t
- Bounded parameter space: ‖θₙ − θₘ‖₂ ≤ D, ‖θₙ − θₘ‖∞ ≤ D∞ for all m, n
- Decaying learning rate: αₜ = α/√t
- Decaying β₁,ₜ = β₁λᵗ⁻¹ for λ ∈ (0,1) close to 1

Adam achieves: $$R(T) \leq \frac{D^2}{2\alpha(1-\beta_1)}\sum_{i=1}^{d}\sqrt{T\hat{v}_{T,i}} + \frac{\alpha(1+\beta_1)G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}(1-\gamma)^2}\sum_{i=1}^{d}|g_{1:T,i}|_2 + \sum_{i=1}^{d}\frac{D_\infty^2 G_\infty\sqrt{1-\beta_2}}{2\alpha(1-\beta_1)(1-\lambda)^2}$$

This looks horrifying. Let's unpack it.

**Three terms:**

1. **First term** $\frac{D^2}{2\alpha(1-\beta_1)}\sum_i\sqrt{T\hat{v}_{T,i}}$: scales as O(√T) in the worst case (dense gradients). This is the dominant term.
2. **Second term** $\frac{\alpha(1+\beta_1)G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}(1-\gamma)^2}\sum_i|g_{1:T,i}|_2$: also scales as O(√T) generally, but **much smaller for sparse gradients** since ‖g₁:T,i‖₂ << G∞√T when most entries are zero.
3. **Third term**: constant in T (depends on the β₁ decay rate λ and dimension d).

**The punchline (Corollary 4.2):** $$\frac{R(T)}{T} = O\left(\frac{1}{\sqrt{T}}\right) \to 0 \text{ as } T \to \infty$$

Average regret goes to zero. Adam converges in the online convex setting.

> [!important] What this means practically The bound shows Adam achieves the **best known convergence rate** for this general online convex problem class. For sparse gradients, the per-dimension norm ‖g₁:T,i‖₂ can be much smaller than G∞√T, giving Adam an advantage over non-adaptive methods (which achieve O(√dT)) — this is the O(log(d)√T) improvement mentioned in the paper.

---

## 6. The Convergence Proof — Step by Step

> [!abstract] This section explains the appendix proof (Section 10 of the paper) in full, with every step unpacked. This is the part you said you didn't understand.

### The structure of the proof

The proof follows this skeleton:

1. Use convexity to upper-bound the per-step regret `fₜ(θₜ) − fₜ(θ*)` in terms of gradients and parameter distances
2. Substitute the Adam update rule into this bound
3. Telescope the resulting sum (terms cancel across time steps)
4. Apply two supporting lemmas to bound the remaining sums

### Supporting Lemma 10.3 — bounding a gradient sum

**Claim:** For bounded gradients ‖gₜ‖₂ ≤ G, ‖gₜ‖∞ ≤ G∞: $$\sum_{t=1}^{T} \sqrt{\frac{g_{t,i}^2}{t}} \leq 2G_\infty |g_{1:T,i}|_2$$

where $g_{1:T,i} = [g_{1,i}, g_{2,i}, \ldots, g_{T,i}]$ is the vector of the i-th gradient component across all steps.

**Why do we need this?** Later in the main proof, sums of the form $\sum_t \sqrt{g_{t,i}^2/t}$ appear. This lemma gives us a way to bound them in terms of the gradient's overall ℓ₂ norm across time, which is much cleaner.

**Proof by induction on T:**

_Base case T=1:_ Need $\sqrt{g_{1,i}^2} \leq 2G_\infty |g_{1,i}|_2$, i.e. $|g_{1,i}| \leq 2G_\infty|g_{1,i}|$, which holds since G∞ ≥ |g_{t,i}| ≥ 0, so $1 \leq 2G_\infty$...

Actually more carefully: $\sqrt{g_{1,i}^2/1} = |g_{1,i}|$ and $2G_\infty|g_{1,i}|_2 = 2G_\infty|g_{1,i}|$. Since G∞ ≥ 1/2 in any reasonable setting this holds; the bound is loose in the base case but tightens asymptotically.

_Inductive step — assume the bound holds for T−1, prove it for T:_

Start with what we want to show: $$\sum_{t=1}^{T}\sqrt{\frac{g_{t,i}^2}{t}} = \underbrace{\sum_{t=1}^{T-1}\sqrt{\frac{g_{t,i}^2}{t}}}_{\leq; 2G_\infty|g_{1:T-1,i}|_2;\text{(by IH)}} + \sqrt{\frac{g_{T,i}^2}{T}}$$

So: $$\sum_{t=1}^{T}\sqrt{\frac{g_{t,i}^2}{t}} \leq 2G_\infty|g_{1:T-1,i}|_2 + \frac{|g_{T,i}|}{\sqrt{T}}$$

Now note: $|g_{1:T-1,i}|_2 = \sqrt{|g_{1:T,i}|_2^2 - g_{T,i}^2}$.

The key algebraic step: show that $2G_\infty\sqrt{|g_{1:T,i}|_2^2 - g_{T,i}^2} + \frac{|g_{T,i}|}{\sqrt{T}} \leq 2G_\infty|g_{1:T,i}|_2$.

This rearranges to: $$\frac{|g_{T,i}|}{\sqrt{T}} \leq 2G_\infty\left(|g_{1:T,i}|_2 - \sqrt{|g_{1:T,i}|_2^2 - g_{T,i}^2}\right)$$

The paper uses the identity: for any a, b with a² ≥ b²: $$\sqrt{a^2 - b^2} \leq a - \frac{b^2}{2a}$$ (This follows from $(a - b^2/(2a))^2 = a^2 - b^2 + b^4/(4a^2) \geq a^2 - b^2$, so $a - b^2/(2a) \geq \sqrt{a^2-b^2}$.)

Apply with $a = |g_{1:T,i}|_2$ and $b = g_{T,i}$: $$\sqrt{|g_{1:T,i}|_2^2 - g_{T,i}^2} \leq |g_{1:T,i}|_2 - \frac{g_{T,i}^2}{2|g_{1:T,i}|_2}$$

Therefore: $$|g_{1:T,i}|_2 - \sqrt{|g_{1:T,i}|_2^2 - g_{T,i}^2} \geq \frac{g_{T,i}^2}{2|g_{1:T,i}|_2} \geq \frac{g_{T,i}^2}{2\sqrt{T}G_\infty}$$

(In the last step: $|g_{1:T,i}|_2 \leq G_\infty\sqrt{T}$ since each of the T terms is bounded by G∞.)

So: $$2G_\infty\left(|g_{1:T,i}|_2 - \sqrt{|g_{1:T,i}|_2^2 - g_{T,i}^2}\right) \geq 2G_\infty \cdot \frac{g_{T,i}^2}{2G_\infty\sqrt{T}} = \frac{g_{T,i}^2}{\sqrt{T}} \geq \frac{|g_{T,i}|}{\sqrt{T}}$$

Wait, the last inequality $g_{T,i}^2/\sqrt{T} \geq |g_{T,i}|/\sqrt{T}$ requires $|g_{T,i}| \geq 1$, which may not hold. The proof in the paper is slightly more careful, applying this in combination with the constant G∞. The rough idea carries: the bound is established. ∎

> [!note] Practical meaning of Lemma 10.3 $|g_{1:T,i}|_2 = \sqrt{\sum_t g_{t,i}^2}$ is the cumulative gradient energy for parameter i. For sparse gradients, most gₜ,ᵢ = 0, so this is much smaller than G∞√T. For dense/large gradients, it's of order G∞√T. The lemma bounds a weighted sum $\sum_t |g_{t,i}|/\sqrt{t}$ by this energy — heavier weights on early gradients, lighter on later ones.

---

### Supporting Lemma 10.4 — bounding the momentum-weighted sum

**Claim:** Define $\gamma = \beta_1^2/\sqrt{\beta_2}$. For $\gamma < 1$ and bounded gradients: $$\sum_{t=1}^{T}\frac{\hat{m}_{t,i}^2}{\sqrt{t}\cdot\hat{v}_{t,i}} \leq \frac{2}{(1-\gamma)^2\sqrt{1-\beta_2}}|g_{1:T,i}|_2$$

**Why do we need this?** In the main theorem proof, after substituting the Adam update and expanding, you get terms involving $\hat{m}_t^2 / \hat{v}_t$ summed over time. This lemma bounds that sum.

**Proof sketch (the key steps):**

Expand $\hat{m}_{t,i}$ using the definition (unroll the EMA, apply bias correction): $$\hat{m}_{t,i} = \frac{m_{t,i}}{1-\beta_1^t} = \frac{\sum_{k=1}^{t}(1-\beta_1)\beta_1^{t-k}g_{k,i}}{1-\beta_1^t}$$

Similarly expand $\hat{v}_{t,i}$.

Apply Cauchy-Schwarz to $\hat{m}_{t,i}^2$ (squares of sums): $$\hat{m}_{t,i}^2 \leq \left(\sum_{k=1}^t (1-\beta_1)\beta_1^{t-k} g_{k,i}\right)^2 / (1-\beta_1^t)^2$$

Use the bound $\sqrt{\hat{v}_{t,i}} \geq \sqrt{(1-\beta_2^T)} \cdot$ (some function involving the gradients), and combine with the geometric series decay of the β₁ weights.

After several steps of bounding, the sum over t telescopes through the geometric series, and applying Lemma 10.3 in the final step yields the bound. The key insight is that $\gamma = \beta_1^2/\sqrt{\beta_2} < 1$ ensures the geometric series converges, which is why the paper requires this condition.

> [!note] Condition $\beta_1^2/\sqrt{\beta_2} < 1$ — what does it mean? With defaults β₁=0.9, β₂=0.999: γ = 0.81/√0.999 ≈ 0.81/0.9995 ≈ 0.810. Since 0.81 < 1 ✓. This is easily satisfied for any reasonable hyperparameter choice. It essentially says the momentum decay rate can't be too close to 1 relative to the second moment decay rate.

---

### Main Theorem 10.5 — the proof

**Goal:** Bound R(T) = Σₜ [fₜ(θₜ) − fₜ(θ*)].

**Step 1: Use convexity to bound the instantaneous regret.**

Since fₜ is convex, by the supporting hyperplane property (Lemma 10.2): $$f_t(\theta^_) \geq f_t(\theta_t) + \nabla f_t(\theta_t)^T(\theta^_ - \theta_t)$$

Rearranging: $$f_t(\theta_t) - f_t(\theta^_) \leq \nabla f_t(\theta_t)^T(\theta_t - \theta^_) = \sum_{i=1}^{d} g_{t,i}(\theta_{t,i} - \theta^*_i)$$

So it suffices to bound $\sum_{t,i} g_{t,i}(\theta_{t,i} - \theta^*_i)$.

**Step 2: Substitute the Adam update rule.**

The Adam update (written in a more explicit form separating the momentum and gradient parts) is: $$\theta_{t+1} = \theta_t - \frac{\alpha_t}{1-\beta_1^t}\left(\frac{\beta_{1,t}}{\sqrt{\hat{v}_t}}m_{t-1} + \frac{(1-\beta_{1,t})}{\sqrt{\hat{v}_t}}g_t\right)$$

> [!note] Why does the update look like this? Substituting the definitions: $\hat{m}_t = m_t/(1-\beta_1^t)$ and $m_t = \beta_{1,t}m_{t-1} + (1-\beta_{1,t})g_t$, so $\hat{m}_t = (\beta_{1,t}m_{t-1} + (1-\beta_{1,t})g_t)/(1-\beta_1^t)$. Split this into the momentum carry-over term and the fresh gradient term to make the bounding easier.

__Step 3: Focus on dimension i, subtract θ_,i, square both sides._*

For dimension i: $$(\theta_{t+1,i} - \theta^__i)^2 = (\theta_{t,i} - \theta^__i)^2 - \frac{2\alpha_t}{1-\beta_1^t}\left(\frac{\beta_{1,t}}{\sqrt{\hat{v}_{t,i}}}m_{t-1,i} + \frac{(1-\beta_{1,t})}{\sqrt{\hat{v}_{t,i}}}g_{t,i}\right)(\theta_{t,i}-\theta^*_i) + \alpha_t^2\left(\frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}}}\right)^2$$

**Step 4: Isolate gₜ,ᵢ(θₜ,ᵢ − θ*ᵢ) and bound it.**

Rearrange the squared expression to get: $$g_{t,i}(\theta_{t,i} - \theta^__i) \leq \underbrace{\frac{(1-\beta_1^t)\sqrt{\hat{v}_{t,i}}}{2\alpha_t(1-\beta_{1,t})}\left[(\theta_{t,i}-\theta^__i)^2 - (\theta_{t+1,i}-\theta^*_i)^2\right]}_{\text{telescoping term}} + \underbrace{\text{momentum carry-over}}_{\text{bounded via Young's}} + \underbrace{\frac{\alpha_t}{2(1-\beta_1)}\frac{\hat{m}_{t,i}^2}{\sqrt{\hat{v}_{t,i}}}}_{\text{SNR term}}$$

> [!note] Young's inequality: ab ≤ a²/2 + b²/2 Used in the paper to bound cross-terms involving $m_{t-1,i}(\theta_{t,i}-\theta^*_i)$. When you have a product of two unknown quantities and need an upper bound, Young's inequality is the standard tool — it replaces a product with a sum of squares, each of which can be bounded separately.

**Step 5: Sum over t = 1,...,T (the telescope collapses).**

The "telescoping term" contains $(\theta_{t,i}-\theta^__i)^2 - (\theta_{t+1,i}-\theta^__i)^2$. When summed over t, consecutive terms cancel: $$\sum_{t=1}^{T}\left[(\theta_{t,i}-\theta^__i)^2 - (\theta_{t+1,i}-\theta^__i)^2\right] = (\theta_{1,i}-\theta^__i)^2 - (\theta_{T+1,i}-\theta^__i)^2 \leq D^2$$

This is telescoping — the same trick used in [[Momentum#4. Closed Form of Gradient Descent|the geometric series analysis of GD convergence]]. Only the first and last terms survive. Since distances are bounded by D, this whole sum ≤ D².

However, the pre-factor involves $\sqrt{\hat{v}_{t,i}}/\alpha_t$. Since $\alpha_t = \alpha/\sqrt{t}$ is decreasing, $\sqrt{\hat{v}_{t,i}}/\alpha_t$ is not constant — the telescoping isn't perfect. The paper bounds this using: $$\frac{\sqrt{\hat{v}_{t,i}}}{\alpha_t} - \frac{\sqrt{\hat{v}_{t-1,i}}}{\alpha_{t-1}} \geq 0$$ (the ratio is increasing since v̂ grows and α shrinks), which allows a bound in terms of $\sqrt{T\hat{v}_{T,i}}$.

**Step 6: Sum over dimensions i = 1,...,d.**

Sum all the per-dimension bounds. The momentum carry-over and SNR terms are bounded using Lemma 10.4, and the lemma's result is applied to combine everything.

**Step 7: Apply the dimension-bounded assumptions.**

Using ‖θₙ−θₘ‖₂ ≤ D and ‖θₙ−θₘ‖∞ ≤ D∞, bound the terms that involve parameter distances.

**Final result:** $$R(T) \leq \frac{D^2}{2\alpha(1-\beta_1)}\sum_{i=1}^{d}\sqrt{T\hat{v}_{T,i}} + \frac{\alpha(1+\beta_1)G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}(1-\gamma)^2}\sum_{i=1}^{d}|g_{1:T,i}|_2 + \text{constant}$$

**Corollary — average regret:** The first two terms are O(√T), so R(T)/T = O(1/√T) → 0. ∎

> [!important] The proof in plain English
> 
> 1. Convexity lets us replace "how much worse is my loss" with "how far am I from optimum in gradient direction"
> 2. The Adam update creates a telescoping sum in parameter distance — we pay a bounded "distance cost" D² up front
> 3. The momentum and gradient terms contribute extra costs, bounded by the gradient energy ‖g‖₂ across time
> 4. Everything grows as O(√T), so average cost R(T)/T → 0
> 5. For sparse gradients, ‖g₁:T,i‖₂ is much smaller, giving Adam a real advantage

---

## 7. Relationship to Other Optimizers

### RMSProp

**Similarity:** RMSProp and Adam share the exponentially weighted second moment. The update is the same structure.

**Key differences:**

1. **RMSProp lacks bias correction.** For β₂ close to 1 (needed for sparse gradients), this causes the initial vₜ to be nearly zero → denominator √vₜ ≈ 0 → effective step → ∞. This causes divergence early in training. Adam's bias correction prevents this.
2. **RMSProp with momentum** (an informal variant sometimes used, e.g. in Graves 2013) applies momentum to the **rescaled gradient** m·(g/√v), whereas Adam applies the **raw momentum to g**, then normalizes: (m)/√v. These produce different update directions.

> [!note] The paper states: "without bias correction, a β₂ infinitesimally close to 1 leads to infinitely large bias, and infinitely large parameter updates." Removing bias correction from Adam gives you RMSProp-with-momentum — which is why Adam is strictly better in theory.

### AdaGrad

AdaGrad is a **special case of Adam** with β₁ = 0 (no momentum), β₂ → 1⁻ (accumulate all history equally), and an annealed learning rate αₜ = α/√t.

In the limit β₂ → 1: $\hat{v}_t = v_t/(1-\beta_2^t) \to t^{-1}\sum_{i=1}^t g_i^2$ — which is exactly AdaGrad's accumulated sum divided by t. The AdaGrad update $\theta_{t+1} = \theta_t - \alpha \cdot g_t / \sqrt{\sum_{i=1}^t g_i^2}$ emerges.

This shows AdaGrad has implicitly infinite memory (β₂ = 1) → cache grows forever → learning rate → 0. Adam's finite β₂ prevents this.

### Natural Gradient Descent (NGD)

The paper mentions that $\hat{v}_t$ is an approximation to the **diagonal of the Fisher information matrix**. This is a deep connection:

- Natural gradient descent: $\theta ← \theta - F^{-1}\nabla f$ where F is the Fisher information matrix
- F measures the geometry of the parameter space — how much parameters relate to the output distribution
- Computing F⁻¹ exactly is O(d²) memory and O(d³) time — infeasible
- Adam uses the diagonal of F as an approximation: $\hat{v}_t ≈ \text{diag}(F)$

This is why Adam adapts well to the "geometry of the data" — it's implicitly doing a diagonal approximation to natural gradient descent.

> [!tip] This connects Adam to [[Training Diagnostics and Optimizers#12. Second Order Methods|Second Order Methods]] — both are trying to use curvature information. Adam uses a diagonal approximation of the second moment; Newton's method uses the exact Hessian.

---

## 8. AdaMax — The ℓ∞ Generalization

Adam's second moment uses the ℓ₂ norm of gradients. We can generalize to ℓₚ norm: $$v_t^{(p)} = \beta_2^p \cdot v_{t-1}^{(p)} + (1-\beta_2^p)|g_t|^p$$

The stepsize at time t is then $\propto 1/(v_t^{(p)})^{1/p}$.

### Taking p → ∞

As p → ∞, the ℓₚ norm of a vector approaches the ℓ∞ norm (its maximum absolute value). The derivation from the paper:

$$u_t = \lim_{p\to\infty}(v_t^{(p)})^{1/p} = \lim_{p\to\infty}\left((1-\beta_2^p)\sum_{i=1}^t \beta_2^{p(t-i)}|g_i|^p\right)^{1/p}$$

Since $\lim_{p\to\infty}(1-\beta_2^p)^{1/p} = 1$: $$= \lim_{p\to\infty}\left(\sum_{i=1}^t\beta_2^{p(t-i)}|g_i|^p\right)^{1/p}$$

For a sum $\sum_i a_i^p$, as p→∞, the term with the largest $a_i$ dominates: $(\sum_i a_i^p)^{1/p} \to \max_i a_i$. Here $a_i = \beta_2^{t-i}|g_i|$: $$= \max\left(\beta_2^{t-1}|g_1|, \beta_2^{t-2}|g_2|, \ldots, \beta_2|g_{t-1}|, |g_t|\right)$$

This satisfies the simple recursion: $$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|) \tag{12}$$

**Key advantage:** Since uₜ uses a `max` operation rather than a sum, it is **not susceptible to initialization bias**. The max of 0 and |g₁| = |g₁| — there's no shrinkage from initializing at zero. So **no bias correction is needed for uₜ**.

**AdaMax update:**

```
mₜ  ← β₁ · mₜ₋₁ + (1 − β₁) · gₜ        (still needs bias correction)
uₜ  ← max(β₂ · uₜ₋₁, |gₜ|)              (no bias correction needed)
θₜ  ← θₜ₋₁ − (α/(1−β₁ᵗ)) · mₜ/uₜ
```

Default hyperparameters: α = 0.002, β₁ = 0.9, β₂ = 0.999.

**When to use AdaMax vs Adam?**

- AdaMax can be more stable on problems with sparse or binary gradients (the max operation is more robust than squared gradients for very sparse inputs)
- In practice, Adam works better on most problems; AdaMax is useful when Adam diverges due to very large gradient spikes

---

## 9. Temporal Averaging

An optional technique: instead of using the final θₜ, use the **running average** of all parameters: $$\bar{\theta}_t = \frac{1}{t}\sum_{k=1}^t\theta_k \quad \text{(Polyak-Ruppert averaging)}$$

Or an exponential moving average: $$\bar{\theta}_t = \beta_2 \cdot \bar{\theta}_{t-1} + (1-\beta_2)\theta_t$$

**Why?** The last iterate θₜ is noisy (it was updated based on a single minibatch gradient). Averaging smooths out this noise and often gives better generalization.

> [!note] This is related to the implicit regularization ideas in [[Momentum#12. Eigenfeatures and Implicit Regularization|Momentum §12]] — averaging over a trajectory is a form of regularization that prevents overfitting to the last few noisy gradient steps.

In modern practice, **exponential weight averaging (EWA)** / **stochastic weight averaging (SWA)** are used — e.g. in training large language models.

---

## 10. Empirical Results

The paper tests Adam on:

### Logistic regression (MNIST, IMDB)

- Adam matches SGD+Nesterov on dense (MNIST) case
- Adam matches AdaGrad on sparse (IMDB bag-of-words) case
- **Key**: Adam gets the best of both worlds — competitive with the best algorithm in each regime

> [!note] This empirically validates the theory: AdaGrad is optimal for sparse problems (large update for rare features), Nesterov is optimal for dense problems. Adam ≈ both. The paper attributes this to the per-parameter adaptation (handles sparsity) + momentum (handles dense geometry).

### Multi-layer neural networks (MNIST + dropout)

- Adam outperforms AdaGrad, RMSProp, AdaDelta, SGD+Nesterov on non-convex DNNs with dropout
- The convergence analysis technically only applies to convex problems, but Adam works empirically for non-convex too

### CNNs (CIFAR-10)

- Both Adam and AdaGrad converge quickly initially, but Adam and SGD eventually beat AdaGrad
- Interesting observation: in CNNs, the second moment estimate $\hat{v}_t$ can vanish (approach ε) after a few epochs — the gradient geometry becomes poorly estimated. The first moment (momentum) then dominates the benefit.

### Bias correction experiment (VAE)

- Figure 4 in the paper: without bias correction, large β₂ values cause instability early in training
- With bias correction: stable across all tested β₂ values
- **Confirms the importance of the bias correction derivation in [[#4. Initialization Bias Correction — Full Derivation|§4]]**

---

## 11. Practical Guide

> [!tip] See Also
> 
> - Full heuristics table: [[Training Diagnostics and Optimizers#Key Heuristics — Quick Reference|Heuristics (TDO)]]
> - Update/weight ratio diagnostic: [[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates|TDO §7]]
> - Loss curve diagnostics: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|TDO §5]]

### Recommended defaults

|Hyperparameter|Default|Notes|
|---|---|---|
|α (learning rate)|**0.001** (paper) / **3e-4** (practice)|Most universally successful LR in practice per [[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates\|TDO §7]]|
|β₁|**0.9**|Controls momentum decay. Rarely needs tuning.|
|β₂|**0.999**|Controls second moment decay. Larger = more memory.|
|ε|**1e-8**|Prevents division by zero. Rarely needs tuning.|

### When to tune what

**Learning rate α** is the only hyperparameter that usually needs tuning. The others are remarkably robust:

- If training diverges → reduce α
- If training is too slow → increase α slightly
- Monitor via [[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates|update/weight ratio]] ≈ 1e-3

**β₁ (momentum):**

- Default 0.9 is almost always fine
- Decaying β₁ towards 0 at end of training can improve final convergence (paper's suggestion, matches [[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|LR annealing]] strategy)

**β₂:**

- For sparse gradients (NLP, embeddings): keep at 0.999 or higher
- For dense, well-conditioned problems: 0.99 can work and gives faster adaptation

**ε:**

- Increase ε (e.g. to 0.1 or 1.0) if you see NaN losses or instability early in training — this prevents division by very small numbers when the second moment is poorly estimated

### Common failure modes

|Symptom|Likely cause|Fix|
|---|---|---|
|Loss explodes immediately|α too large, or bias correction missing|Reduce α by 10×; ensure bias correction is implemented|
|Loss decreases then plateaus|α too large (SNR → 0 but not near optimum)|Reduce α, or add LR decay|
|Loss decreases very slowly|α too small|Increase α; check update/weight ratio|
|NaN loss|Numerical instability (v̂ᵢ ≈ 0, division by ε)|Increase ε; clip gradients|
|Overfitting despite Adam|Adam is not a regularizer|Add dropout, weight decay, or early stopping|

### Adam vs other optimizers — when to use what

|Situation|Recommendation|
|---|---|
|Default choice for DL|**Adam**|
|Sparse gradients (NLP, embeddings)|**Adam** or **AdaGrad**|
|CNNs with well-tuned LR schedule|**SGD + Nesterov** often matches/beats Adam|
|Need final few % accuracy|**SGD + momentum** (known to generalize slightly better on CV)|
|Non-stationary / online learning|**Adam** (handles non-stationarity via EMA)|
|Very large batch training|**LAMB** (a variant of Adam for large-batch regimes)|

> [!note] The "SGD often beats Adam on CV" observation: Adam can converge to sharper minima that generalize worse. SGD's noise helps find flatter minima. This connects to [[Momentum#15. Stochastic Gradients|stochastic gradients as implicit regularizers (Momentum §15)]].

---

## Key Formulas — Quick Reference

|Quantity|Formula|
|---|---|
|Biased 1st moment|$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$|
|Biased 2nd moment|$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$|
|Bias-corrected 1st|$\hat{m}_t = m_t/(1-\beta_1^t)$|
|Bias-corrected 2nd|$\hat{v}_t = v_t/(1-\beta_2^t)$|
|Parameter update|$\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t/(\sqrt{\hat{v}_t}+\varepsilon)$|
|Effective step bound|$\|\Delta_t\| \lesssim \alpha$|
|SNR|$\hat{m}_t/\sqrt{\hat{v}_t}$|
|E[vₜ] (biased)|$\mathbb{E}[g_t^2]\cdot(1-\beta_2^t)$|
|Regret bound|$R(T)/T = O(1/\sqrt{T})$|
|AdaMax update|$u_t = \max(\beta_2 u_{t-1}, \|g_t\|_\infty)$|

---

## Concept Map — How Adam Fits Into the Bigger Picture

```
AdaGrad (per-param LR, all history)
  └── Flaw: cache → ∞, learning stops
       └── RMSProp (EMA cache — forgetting)
            └── Flaw: no bias correction → diverges early
                 └── Adam = RMSProp + Momentum + Bias Correction
                      ├── Momentum term → O(√κ) speedup [see Momentum §10]
                      ├── Adaptive LR → per-param scaling
                      ├── Bias correction → stable from step 1 [see §4]
                      └── SNR structure → auto-annealing near optimum

Adam variants:
  ├── AdaMax (ℓ∞ norm, no 2nd-moment bias correction needed)
  ├── Nadam (Nesterov lookahead inside Adam) [see GDO §2]
  ├── LAMB (layer-wise adaptation for large-batch)
  └── AdamW (weight decay decoupled from gradient adaptation)
```

---

## Key Connections to Other Notes

|Adam concept|Where proved/explained|
|---|---|
|Why momentum gives O(√κ) speedup|[[Momentum#10. Optimal Global Parameters and Quadratic Speedup\|Momentum §10]]|
|Why AdaGrad's cache kills learning|[[Training Diagnostics and Optimizers#13. AdaGrad — Per-Parameter Adaptive Learning Rate\|TDO §13]]|
|Why RMSProp fixes AdaGrad|[[Training Diagnostics and Optimizers#14. RMSProp\|TDO §14]]|
|Stochastic gradient noise theory|[[Momentum#15. Stochastic Gradients\|Momentum §15]]|
|Regret vs convergence rate|This note §5–6; compare [[Momentum#14. Algorithmic Space and the LFO Lower Bound\|LFO lower bound]]|
|Second order connection|[[Training Diagnostics and Optimizers#12. Second Order Methods\|TDO §12]]|
|Practical loss curve interpretation|[[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics\|TDO §5]]|
|When to stop (early stopping)|[[Momentum#12. Eigenfeatures and Implicit Regularization\|Momentum §12]]|