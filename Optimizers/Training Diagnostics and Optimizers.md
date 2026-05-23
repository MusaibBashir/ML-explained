# Training Diagnostics and Optimizers — Notes

> [!abstract] About These Notes
> Practical, practitioner-focused notes on how to debug neural network training, verify gradients, monitor training health, and choose/tune optimizers. The depth is applied/intermediate — formulas are stated and used but not formally derived.
> 
> **Companion files:** [[Momentum]] (theoretical foundations) · [[Gradient Descent Overview]] (algorithm survey)

---

## Overview of Topics (in order)

1. [[#1. Gradient Checking|Gradient Checking]] — numerical vs analytical gradients
2. [[#2. Relative Error — Interpreting Gradcheck|Relative Error thresholds]] — how to interpret gradcheck results
3. [[#3. Practical Gradcheck Tips|Practical gradcheck tips]] — kinks, datapoints, h size, precision
4. [[#4. Sanity Checks Before Training|Sanity Checks Before Training]]
5. [[#5. Loss Curve Diagnostics|Training diagnostics]] — loss curves, learning rate regimes
6. [[#6. Train/Val Accuracy Curves|Train/Val accuracy curves]] — diagnosing overfit/underfit
7. [[#7. Ratio of Weights to Updates|Ratio of weights to updates]] — update magnitude heuristic
8. [[#8. Vanilla SGD Update|Vanilla SGD update]]
9. [[#9. Momentum Update|Momentum update]] — physical intuition (particle on landscape)
10. [[#10. Nesterov Momentum|Nesterov Momentum]] — lookahead gradient
11. [[#11. Annealing the Learning Rate|Annealing the learning rate]] — step decay, exp decay, 1/t decay
12. [[#12. Second Order Methods|Second Order Methods]] — Newton's method, L-BFGS
13. [[#13. AdaGrad — Per-Parameter Adaptive Learning Rate|AdaGrad]] — per-parameter adaptive learning rate
14. [[#14. RMSProp|RMSProp]] — fixing AdaGrad's monotonic cache with EMA
15. [[#15. Adam — RMSProp + Momentum|Adam]] — RMSProp + Momentum, bias correction
16. [[#16. Hyperparameter Optimization|Hyperparameter Optimization]] — random vs grid search, Bayesian

---

## 1. Gradient Checking

> [!tip] See Also
> - Why gradients matter theoretically: [[Momentum#1. Gradient Descent — Basics and Failure Modes|GD Basics (Momentum)]]
> - The gradient at convergence in eigenbasis: [[Momentum#4. Closed Form of Gradient Descent|Closed-Form GD (Momentum)]]

Used to verify that your analytically computed gradient is correct by comparing it to a numerically estimated one.

**Two finite difference formulas:**

**Forward difference (do NOT use):**
```
df(x)/dx ≈ [f(x+h) − f(x)] / h
```
- Error of order O(h) — too imprecise
- Only use as a last resort

**Central difference (use this instead):**
```
df(x)/dx ≈ [f(x+h) − f(x−h)] / (2h)
```
- Error of order O(h²) — much more precise
- Twice as expensive (2 forward passes) but worth it

> [!note] The relative error threshold for interpreting results is in [[#2. Relative Error — Interpreting Gradcheck|§2]].

---

## 2. Relative Error — Interpreting Gradcheck

Use relative error for comparison, not absolute error.

```
Relative Error = |f'_a − f'_n| / max(|f'_a|, |f'_n|)
```
where f'_a = analytical gradient, f'_n = numerical gradient.

**Thresholds:**

| Relative Error | Verdict |
|----------------|---------|
| > 1e-2 | Gradient almost certainly wrong |
| 1e-2 > RE > 1e-4 | Uncomfortable — investigate |
| 1e-4 > RE | Usually okay, but watch for kinks |
| < 1e-7 | Very good |

> [!important] **Key insight:** Deeper networks accumulate more floating point errors → higher relative errors are expected in deeper architectures. Always use **double precision (float64)** for gradcheck.

> [!note] Kinks (ReLU non-differentiabilities) can artificially inflate relative error — see [[#3. Practical Gradcheck Tips|§3]].

---

## 3. Practical Gradcheck Tips

> [!tip] See Also
> - Dropout / non-determinism affects gradcheck — see note on fixing seeds below
> - ReLU kinks are a special case of the non-smooth functions discussed in [[Momentum#8. Four Convergence Regimes|Regime 3: 1-step convergence]]

**Kinks in the objective:**
- e.g. ReLU at x = −1e-6 (since x < 0 the function is non-differentiable there)
- At a kink: f'_a = exactly zero; f'_n might be non-zero if f(x+h) crosses the kink (h > 1e-6)
- Fix: use only a **few datapoints** — fewer datapoints → fewer kinks → less likely to cross one during finite diff approximation
- Checking ~2 or 3 datapoints would almost certainly gradcheck for an entire batch

**Choosing h:**
- Smaller is not always better — precision problems when h is too small
- If you see errors, use h = 1e-4 or 1e-6
- There is a sweet spot (U-shaped error curve vs h)

**Don't overwhelm data with regularization:**
- Reg loss may overwhelm the data loss
- In which case ∇ would be mainly coming from reg term
- This can mask an incorrect implementation of loss gradient
- So: **check by turning reg off**

**Turn off dropout/augmentations:**
- Turn off any non-deterministic effects in the network before gradchecking
- Otherwise you might get huge errors
- Better solution: fix a random seed so dropout is deterministic during check

**Check only few dimensions:**
- In practice gradients can have millions of parameters
- Only practical to check some dimensions and assume others are correct

**Gradcheck during a "characteristic" mode of operation:**
- Gradcheck is performed at a particular (usually random) single point in parameter space
- A successful gradcheck doesn't ensure global correctness
- Best: use a short burn-in time during which network is allowed to learn and perform gradcheck after loss starts to go down

---

## 4. Sanity Checks Before Training

> [!tip] Run these before any serious training — they catch implementation bugs fast.

Before training properly, run these checks:

- **Look for correct loss at chance performance** — e.g. for 10-class classification, loss should be ~log(10) ≈ 2.3 at init
- **Increasing regularization strength should increase loss**
- **Overfit a tiny subset of data** — take ~5 training examples and achieve zero cost. If you can't overfit 5 examples, something is wrong with the model/loss

> [!note] The "overfit a tiny subset" check is directly related to the [[Momentum#8. Four Convergence Regimes|1-step convergence regime]] — you're verifying the optimizer can actually reach a minimum at all.

---

## 5. Loss Curve Diagnostics

> [!tip] See Also
> - Theoretical explanation of why loss slows down: [[Momentum#4. Closed Form of Gradient Descent|Closed-Form GD]] — eigenspaces with small λᵢ decay slowest
> - The four regimes here correspond directly to [[Momentum#8. Four Convergence Regimes|the four convergence regimes in Momentum]]
> - Fixing a bad LR: [[#11. Annealing the Learning Rate|§11]]

**Loss vs epoch — four regimes:**

| Curve shape | Diagnosis | Theory link |
|-------------|-----------|-------------|
| ① Very high LR | Loss explodes or stays high | [[Momentum#8. Four Convergence Regimes\|Regime 5: Divergence]] |
| ② Low LR | Loss decreasing very slowly, flat for long | [[Momentum#8. Four Convergence Regimes\|Regime 2: Monotonic]] |
| ③ High LR | Loss decreases fast initially then stops/diverges | [[Momentum#8. Four Convergence Regimes\|Regime 4: Monotonic Oscillations]] |
| ④ Good LR | Smooth decrease, converges well | [[Momentum#8. Four Convergence Regimes\|Ripples or critical damping]] |

**Amount of wiggle in loss is related to batch size:**
- Batch size = 1 → wiggle relatively high (high variance SGD — see [[Gradient Descent Overview#1. Gradient Descent Variants & Challenges|GDO §1]])
- Batch size = full dataset → wiggle minimal (every gradient update should improve loss monotonically, unless LR is too high)
- Connection to stochastic noise: [[Momentum#15. Stochastic Gradients|Stochastic Gradients (Momentum §15)]]

---

## 6. Train/Val Accuracy Curves

> [!tip] See Also
> - Early stopping as implicit regularization: [[Momentum#12. Eigenfeatures and Implicit Regularization|Eigenfeatures & Implicit Regularization (Momentum §12)]]
> - GDO early stopping note: [[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies|Early Stopping (GDO)]]

**Interpreting the gap between train and val accuracy:**

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Gap between train and val | Mild overfitting | Increase model capacity |
| Val accuracy strongly lagging | Significant overfitting | Add regularization |
| Val accuracy follows train accuracy closely | Underfitting or working well | Increase capacity or keep training |

> [!note] From [[Momentum#12. Eigenfeatures and Implicit Regularization|Momentum §12]]: stopping early (before small eigenfeatures overfit) is a form of regularization without any extra hyperparameter. This gives you the **entire** underfit-to-overfit spectrum from one run.

---

## 7. Ratio of Weights to Updates

A useful diagnostic heuristic:
```
Ratio = update magnitudes / value magnitudes
```

| Ratio | Diagnosis |
|-------|-----------|
| ~1e-3 | Learning rate roughly right |
| < 1e-4 | LR might be too low |
| > 1e-2 | LR likely very high |

> [!note] This heuristic is the practitioner's proxy for the [[Momentum#5. Condition Number and Optimal Step Size|optimal step size α* = 2/(λ₁+λₙ)]] — you can't compute α* directly without knowing the Hessian eigenvalues, so you monitor the update ratio instead.

Most universally successful LR for Adam is around **3e-4** — see [[#15. Adam — RMSProp + Momentum|§15]].

---

## 8. Vanilla SGD Update

> [!tip] See Also
> - Full theoretical analysis of SGD on quadratics: [[Momentum#4. Closed Form of Gradient Descent|Closed-Form GD (Momentum)]]
> - Why SGD is slow (condition number): [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum)]]
> - Algorithm comparison table: [[Gradient Descent Overview#1. Gradient Descent Variants & Challenges|GDO §1]]

```
x += (−LR) × dx
```
Simple gradient descent — gradient directly integrates **position**.

> [!warning] The key limitation: gradient acts directly on position, not on velocity. This is exactly what [[#9. Momentum Update|momentum]] fixes — gradient acts as a *force* (acceleration), which influences velocity, which in turn changes position.

---

## 9. Momentum Update

> [!tip] See Also
> - Full mathematical theory (eigenanalysis, critical damping, quadratic speedup): [[Momentum]]
> - The 2×2 transition matrix governing convergence: [[Momentum#7. The 2×2 Transition Matrix|Transition Matrix (Momentum §7)]]
> - Four convergence regimes (ripples, monotone, 1-step, diverge): [[Momentum#8. Four Convergence Regimes|Convergence Regimes (Momentum §8)]]
> - Nesterov extension: [[#10. Nesterov Momentum|§10]]
> - Algorithm table: [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|Momentum (GDO)]]

**Physical intuition:**
- Think of Loss → Height (potential energy = mgh ∝ h)
- Initializing params with random noise ≡ particle with zero velocity at some location on the landscape
- Optimization ≡ simulating the param vector (i.e. a particle) rolling on the landscape

**Since F = −∇U, force is the (negative) gradient of loss.**

This suggests an update where gradient directly influences **velocity**, which in turn affects position. (SGD gradient directly integrates position — that is its limitation.)

**Momentum update equations:**
```
v = μ·v − α·dx
x += v
```
- v → initialized to zero
- μ → hyperparameter (coefficient of friction), typically **start 0.5, anneal to 0.99**
- μ damps the velocity and reduces kinetic energy → so particle will stop at bottom of hill

> [!note] The theoretical optimal μ is derived in [[Momentum#9. Critical Damping — The Optimal β|Critical Damping (Momentum §9)]]:
> `β* = (1−√(αλᵢ))²`
> The practical heuristic of annealing from 0.5→0.99 approximates the transition from overdamped to critically damped as the landscape flattens near the minimum.

**Key difference from [[#8. Vanilla SGD Update|SGD]]:** Gradient influences velocity (acceleration), not position directly.

---

## 10. Nesterov Momentum

> [!tip] See Also
> - Theoretical basis: [[Momentum#6. Momentum — The Tweak|Momentum §6]] (Nesterov's optimality result) and [[Momentum#14. Algorithmic Space and the LFO Lower Bound|LFO lower bound (Momentum §14)]]
> - Algorithm table with full equations: [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|NAG (GDO)]]
> - Nadam (Nesterov + Adam): [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|Nadam (GDO)]]

Stronger theoretical convergence guarantees for convex functions and better in practice.

**Key idea:** Calculate gradient at the **lookahead (x + μv) point**, not at current x.

```
x_ahead = x + μ·v
v = μ·v − α·dx_ahead
x += v
```

**In practice, written like SGD (equivalent rearrangement):**
```
v_prev = v
v = μ·v − α·dx
x += −μ·v_prev + (1+μ)·v
```

> [!note] Nesterov essentially "commits" to the momentum step first, then corrects. This is why it has better theoretical guarantees than standard momentum — it's computing the gradient in a more informed location. The LFO lower bound ([[Momentum#14. Algorithmic Space and the LFO Lower Bound|Momentum §14]]) shows that this family of methods is provably optimal.

---

## 11. Annealing the Learning Rate

> [!tip] See Also
> - Why LR must decay: [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]] — optimal α* depends on the landscape; as we approach the minimum the effective curvature changes
> - Batch norm as an alternative stabilizer: [[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies|Batch Normalization (GDO)]]

Learning rate needs to decay over time — otherwise the system keeps bouncing around the optimum.

**Three common schemes:**

**Step decay** *(most common in practice)*:
- Reduce α by some factor every few epochs
- Heuristic: reduce α by a constant whenever val error stops improving (cf. [[#6. Train/Val Accuracy Curves|§6]])

**Exponential decay:**
```
α = α₀ · e^(−kt)
```
where α₀, k are hyperparameters, t = iteration/epoch

**1/t decay:**
```
α = α₀ / (1 + kt)
```

> [!note] From stochastic gradient theory ([[Momentum#15. Stochastic Gradients|Momentum §15]]): decaying LR reduces stochastic error but slows convergence rate — the fundamental tension. LR annealing is the practical resolution: use a large LR early (fast convergence), small LR late (low noise).

---

## 12. Second Order Methods

> [!tip] See Also
> - The Hessian A is exactly the matrix analyzed in [[Momentum#2. Convex Quadratic — The Canonical Setup|Convex Quadratic Setup (Momentum §2)]]
> - The 1-step convergence this simulates: [[Momentum#8. Four Convergence Regimes|Regime 3: 1-Step Convergence (Momentum §8)]]
> - Why high condition number κ makes GD slow (and second-order methods shine): [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]]

**Based on Newton's method:**
```
x ← x − [H f(x)]⁻¹ ∇f(x)
```
where H f(x) is the Hessian matrix.

**Intuition:** Hessian describes local curvature of loss function, allowing us to perform a more efficient update. Multiplying by inverse Hessian leads the optimizer to:
- Take more **aggressive** steps in directions of **shallow curvature** (small λᵢ)
- Take **shorter** steps in directions of **steep curvature** (large λᵢ)

> [!important] This is exactly [[Momentum#8. Four Convergence Regimes|Regime 3 (1-step convergence)]] applied simultaneously to **every eigenspace** — each eigenspace reaches its minimum in one step regardless of λᵢ.

**Problem:** Calculating Hessian and inverting it in explicit form is very costly in both space and time.
- e.g. 1 million param NN → Hessian size = (1M × 1M) = 3725 GB RAM

**Quasi-Newton methods:** Approximate the inverse Hessian.
- Most popular: **L-BFGS** — uses information in ∇ over time to form the approximation implicitly; full matrix is never computed

**L-BFGS downside:** Needs to be computed over the entire training set (unlike minibatch [[#8. Vanilla SGD Update|SGD]]). In practice it's not common to see L-BFGS or similar 2nd order methods applied to DL and CNNs — [[#9. Momentum Update|SGD variants]] are more standard.

---

## 13. AdaGrad — Per-Parameter Adaptive Learning Rate

> [!tip] See Also
> - Algorithm table: [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|AdaGrad (GDO)]]
> - RMSProp fixes AdaGrad's core flaw: [[#14. RMSProp|§14]]
> - AdaGrad is an LFO method: [[Momentum#14. Algorithmic Space and the LFO Lower Bound|LFO Methods (Momentum §14)]]

```
cache += (dx)²
x += −α × dx / (√cache + ε)
```

- Cache size = size of gradient; keeps track of per-parameter sum of squared gradients
- √ operation is very important — without it the algorithm performs much worse
- ε → to prevent division by zero, usually between **1e-4 and 1e-8**

> [!note] AdaGrad is particularly well-suited for **sparse data** (e.g. word embeddings) because infrequent parameters get larger effective updates — exactly the per-parameter adaptive LR idea from [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|GDO]].

**Downside in DL:** The monotonically increasing cache leads to α/√cache → 0, which stops learning rate too early. [[#14. RMSProp|RMSProp]] fixes this.

---

## 14. RMSProp

> [!tip] See Also
> - Algorithm table: [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|RMSprop (GDO)]]
> - Adam combines RMSProp with momentum: [[#15. Adam — RMSProp + Momentum|§15]]
> - EMA is conceptually related to the momentum accumulation in [[Momentum#6. Momentum — The Tweak|Momentum §6]]

Fixes [[#13. AdaGrad — Per-Parameter Adaptive Learning Rate|AdaGrad]]'s monotonically increasing cache problem by using **Exponential Moving Average (EMA)**:

```
cache = decay_rate × cache + (1 − decay_rate) × (dx)²
x += −α × dx / (√cache + ε)
```

- β (decay_rate) typically: **0.9, 0.99, or 0.999**
- This allows the effective learning rate to recover when gradients become small
- Suggested default LR η = 0.001

> [!note] RMSProp was proposed by Geoff Hinton in a Coursera lecture — it was never formally published. Adadelta ([[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|GDO]]) was developed independently around the same time and is essentially identical in its first update vector.

---

## 15. Adam — RMSProp + Momentum

> [!tip] See Also
> - Full algorithm table: [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|Adam (GDO)]]
> - Momentum theory: [[Momentum#6. Momentum — The Tweak|Momentum §6]]
> - Why Adam is an LFO method: [[Momentum#14. Algorithmic Space and the LFO Lower Bound|LFO Methods (Momentum §14)]]
> - Adam update/weight ratio heuristic: [[#7. Ratio of Weights to Updates|§7]]
> - Nadam (Nesterov + Adam): [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|Nadam (GDO)]]

Adam = [[#14. RMSProp|RMSProp]] with [[#9. Momentum Update|momentum]]. Looks like the most universally successful optimizer.

**Update equations:**
```
m = β₁·m + (1−β₁)·dx          ← momentum term (smoothed gradient)
v = β₂·v + (1−β₂)·(dx)²       ← RMSProp cache
x += −α·m / (√v + ε)
```

Instead of raw dx we use a smoothed version m.

**Recommended hyperparameters:**
- ε = **1e-8**
- β₁ = **0.9**
- β₂ = **0.999**
- Most universally successful LR ≈ **3e-4**

**Bias correction mechanism:**
- Full Adam update includes bias correction which compensates for the fact that in the first few steps, vectors m and v are initialized and biased at zero
- This is the same initialization bias discussed in [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|GDO (Adam entry)]]
- Bias-corrected estimates:
```
m_c = m / (1 − β₁ᵗ)
v_c = v / (1 − β₂ᵗ)
```
Put m_c and v_c in the x update.

> [!note] The momentum term m here provides the O(√κ) speedup discussed in [[Momentum#10. Optimal Global Parameters and Quadratic Speedup|Momentum §10]], while the RMSProp term v provides per-parameter scaling that effectively pre-conditions the gradient (approximating what [[#12. Second Order Methods|second-order methods]] do with the Hessian).

---

## 16. Hyperparameter Optimization

> [!tip] See Also
> - Gradient noise as a related stochastic strategy: [[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies|Gradient Noise (GDO)]]
> - Stochastic gradients from a theoretical perspective: [[Momentum#15. Stochastic Gradients|Stochastic Gradients (Momentum §15)]]

**Prefer random search to grid search:**
- Random search gives you more probability of hitting the optimal zone at least once
- P(success) ≈ 1 − (1−p)ⁿ of hitting the optimal zone at least once

**Other guidelines:**
- Careful with best values on border — if optimal is at boundary of your search range, expand the range
- Stage search from coarse to fine — first broad search, then narrow around promising region
- Bayesian Hyperparameter Optimization — uses a probabilistic model to choose next hyperparameter to try

---

## Key Formulas — Quick Reference

| Method | Update Rule |
|--------|-------------|
| Vanilla SGD | x += −LR × dx |
| Momentum | v = μv − α·dx; x += v |
| Nesterov | x_ahead = x+μv; v = μv − α·dx_ahead; x += v |
| AdaGrad | cache += dx²; x += −α·dx/(√cache+ε) |
| RMSProp | cache = β·cache+(1−β)·dx²; x += −α·dx/(√cache+ε) |
| Adam | m=β₁m+(1−β₁)dx; v=β₂v+(1−β₂)dx²; x += −α·m/(√v+ε) |
| Second Order | x ← x − H⁻¹∇f(x) |

---

## Key Heuristics — Quick Reference

| Topic | Heuristic | Theory |
|-------|-----------|--------|
| Gradcheck h size | Use h = 1e-4 or 1e-6 | [[#1. Gradient Checking\|§1]] |
| Gradcheck datapoints | ~2–3 datapoints is enough for entire batch | [[#3. Practical Gradcheck Tips\|§3]] |
| Relative error threshold | < 1e-7 very good; > 1e-2 wrong | [[#2. Relative Error — Interpreting Gradcheck\|§2]] |
| Precision | Always use float64 (double precision) | [[#2. Relative Error — Interpreting Gradcheck\|§2]] |
| Momentum μ | Start 0.5, anneal to 0.99 | [[Momentum#9. Critical Damping — The Optimal β\|Critical Damping]] |
| Adam ε | 1e-8 | [[#15. Adam — RMSProp + Momentum\|§15]] |
| Adam β₁, β₂ | 0.9, 0.999 | [[#15. Adam — RMSProp + Momentum\|§15]] |
| Adam LR | ~3e-4 most universally successful | [[#7. Ratio of Weights to Updates\|§7]] |
| Update/weight ratio | ~1e-3 is healthy | [[#7. Ratio of Weights to Updates\|§7]] |
| RMSProp β | 0.9, 0.99, or 0.999 | [[#14. RMSProp\|§14]] |
| AdaGrad ε | 1e-4 to 1e-8 | [[#13. AdaGrad — Per-Parameter Adaptive Learning Rate\|§13]] |

---

## Key Analogies Used

| Concept | Analogy | Full Theory |
|---------|---------|-------------|
| Momentum optimization | Particle rolling on a loss landscape | [[Momentum#9. Critical Damping — The Optimal β\|Momentum §9]] |
| Loss | Height / potential energy (mgh) | [[Momentum#13. Colorization Problem — Momentum on a Graph\|Momentum §13]] |
| Gradient | Force on particle | [[Momentum#6. Momentum — The Tweak\|Momentum §6]] |
| Velocity (v) | Particle velocity | [[Momentum#7. The 2×2 Transition Matrix\|Momentum §7]] |
| μ (momentum coeff) | Coefficient of friction — damps velocity | [[Momentum#9. Critical Damping — The Optimal β\|Momentum §9]] |
| Nesterov | Look ahead to where you'll be, then compute gradient there | [[Momentum#6. Momentum — The Tweak\|Momentum §6]] |
| Second order methods | Scale steps by local curvature — big steps in flat directions, small in steep | [[Momentum#8. Four Convergence Regimes\|Regime 3 (Momentum §8)]] |