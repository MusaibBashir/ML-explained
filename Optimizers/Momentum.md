# Momentum Optimization — Comprehensive Notes

> [!abstract] About These Notes
> These notes cover the mathematical theory of why momentum works in optimization, developed from first principles. The depth is graduate-level ML theory. All results are derived, not just stated. The notes are structured as a lecture — each section builds on the last.
> 
> **Companion files:** [[Training Diagnostics and Optimizers]] (practical/applied) · [[Gradient Descent Overview]] (algorithm survey)

---

## Overview of Topics (in order)

1. [[#1. Gradient Descent — Basics and Failure Modes|Gradient Descent]] — basics and failure modes
2. [[#2. Convex Quadratic — The Canonical Setup|Convex Quadratic setup]] — canonical model of curvature
3. [[#3. Eigendecomposition and Change of Basis|Eigendecomposition and decoupling]] — reducing n-D to n×1D
4. [[#4. Closed Form of Gradient Descent|Closed form of Gradient Descent]] — geometric series analysis
5. [[#5. Condition Number and Optimal Step Size|Condition number]] — the key quantity governing convergence
6. [[#6. Momentum — The Tweak|Momentum]] — the tweak and its dynamics
7. [[#7. The 2×2 Transition Matrix|The 2×2 transition matrix]] — per-eigenspace analysis
8. [[#8. Four Convergence Regimes|Four convergence regimes]] — ripples, monotone, 1-step, divergence
9. [[#9. Critical Damping — The Optimal β|Critical damping]] — the optimal β
10. [[#10. Optimal Global Parameters and Quadratic Speedup|Optimal parameters]] — global (α\*, β\*) and the quadratic speedup
11. [[#11. Polynomial Regression — Concrete Application|Polynomial Regression]] — concrete application of the eigenstructure
12. [[#12. Eigenfeatures and Implicit Regularization|Eigenfeatures and implicit regularization]] — early stopping connection
13. [[#13. Colorization Problem — Momentum on a Graph|Colorization Problem]] — momentum on a graph, wave vs diffusion
14. [[#14. Algorithmic Space and the LFO Lower Bound|Algorithmic space and LFO lower bound]] — Convex Rosenbrock
15. [[#15. Stochastic Gradients|Stochastic Gradients]] — noise decomposition and implicit regularization

---

## 1. Gradient Descent — Basics and Failure Modes

> [!tip] See Also
> - Practical update rules: [[Training Diagnostics and Optimizers#8. Vanilla SGD Update|Vanilla SGD Update (TDO)]]
> - Algorithm variants (Batch, SGD, Mini-batch): [[Gradient Descent Overview#1. Gradient Descent Variants & Challenges|GD Variants (GDO)]]
> - Loss curve diagnostics when GD misbehaves: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|Loss Curve Diagnostics]]

**Update rule:**
```
ω^(k+1) = ω^k − α ∇f(ω^k)
```

- For small step size α, GD makes monotonic improvement at every iteration
- Always converges (to a local minimum at least)
- Initially loss decreases fast, but **slows down significantly** as iterations progress

**Pathological curvature** — called the "optimizer's old nemesis":
- Regions of f which are not scaled properly
- The loss landscape has valleys, trenches, canals, or ravines
- Iterates either jump back and forth across valleys OR approach optimum in tiny timid steps
- This is the core problem that motivates [[#6. Momentum — The Tweak|momentum]]

> [!note] The condition number κ (derived in [[#5. Condition Number and Optimal Step Size|§5]]) is the precise measure of how "pathological" the curvature is.

---

## 2. Convex Quadratic — The Canonical Setup

> [!tip] See Also
> - The Hessian connection: [[Training Diagnostics and Optimizers#12. Second Order Methods|Second Order Methods (TDO)]] — Newton's method inverts this exact Hessian A

**Why analyze quadratics?**
- Near any local minimum of a smooth function, the second-order Taylor expansion gives a quadratic
- The Hessian matrix A plays the role of curvature
- So quadratics capture the local behavior of essentially any function
- A can model pathological curvature (e.g. Hessian, Fisher Information Matrix)

**The function:**
```
f(ω) = ½ ωᵀAω − bᵀω,   ω ∈ ℝⁿ
```
where A is symmetric and invertible.

**Optimal solution:** ω\* = A⁻¹b

**Gradient:** ∇f(ω) = Aω − b

**GD update on this function:**
```
ω^(k+1) = ω^k − α(Aω^k − b)
```

> [!example] Concrete Application
> The matrix A = ZᵀZ in [[#11. Polynomial Regression — Concrete Application|Polynomial Regression (§11)]] is exactly this Hessian — symmetric, positive semidefinite, and its eigenstructure governs the entire optimization landscape.

---

## 3. Eigendecomposition and Change of Basis

> [!tip] The key technique that makes everything tractable. All subsequent convergence analysis flows from this decoupling.

**Eigenvalue decomposition of A:**
```
A = Q Λ Qᵀ
```
- Q = [q₁, q₂, ..., qₙ] — orthonormal eigenvectors (columns)
- Λ = diag(λ₁, λ₂, ..., λₙ) — eigenvalues, ordered λ₁ ≤ λ₂ ≤ ... ≤ λₙ
- Q is a pure rotation matrix (orthogonal: QᵀQ = I)

**Change of basis — two simultaneous operations:**
```
x^k = Qᵀ(ω^k − ω*)
```
- Rotating coordinate axes to align with eigenvectors of A
- Shifting origin to the optimal point ω\*

**What this achieves:** After substitution, the loss becomes:
```
f(x) = ½ xᵀΛx − C = Σᵢ (½ λᵢ xᵢ²) − C
```

**No cross terms** — the n-dimensional problem has broken into a sum of n independent 1D quadratics. Each component xᵢ evolves completely independently. This is the fundamental insight.

**For polynomial regression specifically:**
- ω = original polynomial coefficients
- After rotating: ω̄ = Qᵀω (weights in eigenvector basis)
- Features p(ξ) counter-rotated: p̄(ξ) = Qᵀp(ξ) → eigenfeatures
- Now instead of correlated monomials ξ, ξ², ... we have independent eigenfeatures p̄ᵢ
- Each can be optimized greedily → see [[#11. Polynomial Regression — Concrete Application|§11]]

> [!note] This same rotation applies again in [[#7. The 2×2 Transition Matrix|§7]] when deriving the momentum transition matrix — Qᵀ decouples the momentum system too.

---

## 4. Closed Form of Gradient Descent

> [!tip] See Also
> - How this decay rate connects to loss curve shape: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|Loss Curve Diagnostics]]
> - The "why training slows down" observation here is exactly what [[#6. Momentum — The Tweak|momentum]] fixes

**Per-component GD update:**
```
xᵢ^(k+1) = xᵢ^k − α · λᵢ · xᵢ^k = (1 − αλᵢ) xᵢ^k
```

**Closed form after k steps:**
```
xᵢ^k = (1 − αλᵢ)^k · xᵢ⁰
```
A pure geometric series. Each component decays (or grows) by factor (1−αλᵢ) per step.

**Full error in original space:**
```
ω^k − ω* = Q x^k = Σᵢ xᵢ⁰ (1−αλᵢ)^k qᵢ
```

**Decomposing the error:**
- Each xᵢ⁰ is the component of the initial error in the Q-basis (eigenvector basis)
- There are n such errors, each following its own path to minimum
- Each decays exponentially with compounding rate (1−αλᵢ)
- The closer (1−αλᵢ) is to 1, the slower that component converges

**Loss excess:**
```
f(ω^k) − f(ω*) = Σᵢ (1−αλᵢ)^(2k) · λᵢ · (xᵢ⁰)²/2
```

**Key observation:**
- Eigenvectors with **largest** eigenvalues converge fastest → initial rapid fall in loss
- Eigenvectors with **smallest** eigenvalues struggle → this is why training slows down later
- This directly motivates [[#5. Condition Number and Optimal Step Size|the condition number analysis in §5]]

> [!example] The scalar (1−αλᵢ)^k for GD is exactly analogous to the matrix Rᵏ for momentum — see [[#7. The 2×2 Transition Matrix|§7]].

---

## 5. Condition Number and Optimal Step Size

> [!tip] See Also
> - Practical LR heuristics: [[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates|Update/Weight Ratio heuristic]]
> - LR scheduling: [[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|Annealing the Learning Rate]]
> - How κ appears in the **momentum** speedup: [[#10. Optimal Global Parameters and Quadratic Speedup|§10]]
> - LFO lower bound references κ directly: [[#14. Algorithmic Space and the LFO Lower Bound|§14]]

**Convergence requirement:** Need |1−αλᵢ| < 1 for all i, i.e.:
```
0 < αλᵢ < 2   for all i
```
> [!note] Momentum relaxes this: stability holds up to `αλᵢ < 2 + 2β` — see [[#8. Four Convergence Regimes|§8 (Regime 5)]].

**Overall convergence rate** (determined by worst component):
```
rate(α) = max |1−αλᵢ| = max { |1−αλ₁|, |1−αλₙ| }
```
Only the two extreme eigenvalues matter; everything in between is fine.

**Optimal step size** — minimize rate(α) by equating the two extremes:
```
|1−αλ₁| = |1−αλₙ|
→  α* = 2/(λ₁ + λₙ)
```

**Optimal convergence rate:**
```
rate* = (λₙ − λ₁)/(λₙ + λ₁) = (κ−1)/(κ+1)
```

**Condition number:**
```
κ = λₙ/λ₁
```
- Named because it measures how close A is to singular
- Also measures how robust A⁻¹b is to variations in b
- Also measures how poorly gradient descent will perform
- κ = 1 (ideal) → convergence in one step
- Large κ → rate ≈ 1 → extremely slow convergence
- The ratio λₙ/λ₁ **decides the convergence rate entirely**

> [!important] κ is THE central quantity of these notes. It appears in:
> - GD rate: `(κ−1)/(κ+1)` (here)
> - Momentum rate: `(√κ−1)/(√κ+1)` → [[#10. Optimal Global Parameters and Quadratic Speedup|§10]]
> - LFO lower bound: `((√K−1)/(√K+1))^k` → [[#14. Algorithmic Space and the LFO Lower Bound|§14]]
> - Colorization: massive κ explains O(L²) slowness → [[#13. Colorization Problem — Momentum on a Graph|§13]]

---

## 6. Momentum — The Tweak

> [!tip] See Also
> - Practical momentum update + μ heuristics: [[Training Diagnostics and Optimizers#9. Momentum Update|Momentum Update (TDO)]]
> - Algorithm table with equations: [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|Momentum (GDO)]]
> - Nesterov extension: [[#7. The 2×2 Transition Matrix|§7]] and [[Training Diagnostics and Optimizers#10. Nesterov Momentum|Nesterov (TDO)]]

**Momentum update rule (two equations):**
```
z^(k+1) = β z^k + ∇f(ω^k)
ω^(k+1) = ω^k − α z^(k+1)
```
- z is a velocity/momentum variable
- β ∈ [0,1) is the momentum coefficient
- For β=0: reduces to normal gradient descent
- For larger β: regains speed in iterations we were losing

**Key property:** Momentum gives up to a **quadratic speedup** on many functions — O(√n) vs SGD's O(n). Full proof in [[#10. Optimal Global Parameters and Quadratic Speedup|§10]].

**Nesterov's result:** In a certain very narrow and technical sense, momentum is optimal among first-order methods. This is proven rigorously via the [[#14. Algorithmic Space and the LFO Lower Bound|LFO lower bound in §14]].

**On the quadratic**, since ∇f(ω^k) = Aω^k − b = A(ω^k − ω\*):
```
z^(k+1) = β z^k + A(ω^k − ω*)
ω^(k+1) − ω* = (ω^k − ω*) − α z^(k+1)
```

> [!note] Physical Intuition
> - Gradient → **force** on a particle (not velocity, unlike SGD)
> - z → **velocity/momentum** of the particle
> - β → **friction coefficient** that damps velocity
> - Full physical analogy in [[Training Diagnostics and Optimizers#9. Momentum Update|TDO §9]] and [[#9. Critical Damping — The Optimal β|§9 here]]

---

## 7. The 2×2 Transition Matrix

> [!tip] This section is the momentum analog of [[#4. Closed Form of Gradient Descent|§4]] — just as GD decays as (1−αλᵢ)^k, momentum evolves as R^k.

After applying the rotational transformation Qᵀ (from [[#3. Eigendecomposition and Change of Basis|§3]]) and substituting the spectral decomposition A = QΛQᵀ, the system decouples. Define:
- xᵢ^k = i-th component of Qᵀ(ω^k − ω\*)
- yᵢ^k = i-th component of Qᵀ z^k

**Per-eigenspace 2×2 system (1D decoupled):**
```
yᵢ^(k+1) = β yᵢ^k + λᵢ xᵢ^k
xᵢ^(k+1) = xᵢ^k − α yᵢ^(k+1)
```

**Stacked as matrix recurrence:**
```
[yᵢ^(k+1)]   =  R  [yᵢ^k]
[xᵢ^(k+1)]         [xᵢ^k]
```

**Transition matrix R:**
```
R = [ β        λᵢ   ]
    [ −αβ    1−αλᵢ  ]
```

**After k steps:**
```
[yᵢ^k]  =  R^k  [yᵢ⁰]
[xᵢ^k]          [xᵢ⁰]
```

**Convergence** depends entirely on the eigenvalues σ₁, σ₂ of R:
```
Convergence rate = max{|σ₁|, |σ₂|}
```

**Formula for eigenvalues of R** (from characteristic polynomial):
```
σ = ½ { (1−αλᵢ+β) ± √((1−αλᵢ+β)² − 4β) }
```

**The discriminant** Δ = (1−αλᵢ+β)² − 4β determines the regime → see [[#8. Four Convergence Regimes|§8]].

**R^k has an elegant closed form** (since R is 2×2):
- If σ₁ ≠ σ₂: R^k = (σ₁^k R₁ − σ₂^k R₂)/(σ₁−σ₂), where Rⱼ = (R−σⱼI)/(σᵢ−σⱼ)
- If σ₁ = σ₂: R^k = σ₁^k(kR/σ₁ − (k−1)I) ← *this is the [[#9. Critical Damping — The Optimal β|critically damped]] case*

> [!note] The role R^k plays for momentum is exactly the role (1−αλᵢ)^k plays for gradient descent ([[#4. Closed Form of Gradient Descent|§4]]).

In [[#15. Stochastic Gradients|§15]], stochastic noise enters as an additive perturbation to exactly this recurrence: `Rᵏ[yᵢ⁰; xᵢ⁰] + Σⱼ Rᵏ⁻ʲ εᵢʲ [1; −α]`.

---

## 8. Four Convergence Regimes

> [!tip] See Also
> - Loss curve shapes in practice: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|Loss Curve Diagnostics]]
> - Regime 3 (1-step) is what second-order methods simulate: [[Training Diagnostics and Optimizers#12. Second Order Methods|Second Order Methods]]
> - The physical spring analogy for regimes: [[#9. Critical Damping — The Optimal β|§9]]

These are determined by the eigenvalues of R ([[#7. The 2×2 Transition Matrix|§7]]) — their nature (real/complex) and magnitude.

### Regime 1: Ripples (Complex Conjugate Roots, Δ < 0)
- **Condition:** High β paired with small α; discriminant negative
- **Behavior:** Optimizer circles around optimum, overshooting and undershooting in a wave-like manner
- **Loss curve:** Looks like a damped sinewave
- **Key property:** Rate of decay bounded by 2√β, **independent of the specific eigenvalue** — this is crucial for understanding [[#10. Optimal Global Parameters and Quadratic Speedup|the quadratic speedup]]
- **Physical picture:** Underdamped oscillator → see [[#9. Critical Damping — The Optimal β|§9]]

### Regime 2: Monotonic Decrease (Real, Positive Roots, Norm < 1)
- **Condition:** Conservative step size, low or absent momentum
- **Behavior:** Algorithm behaves like normal gradient descent, slowing asymptotically
- **Loss curve:** Drops sharply first, then flattens out into a plateau
- **Physical picture:** Overdamped system → see [[#9. Critical Damping — The Optimal β|§9]]

### Regime 3: 1-Step Convergence (β=0, α=1/λᵢ)
- **Condition:** Exactly β=0, α=1/λᵢ — a mathematical singularity
- **Behavior:** Gradient lands at the bottom of that 1D parabola in exactly one iteration
- **Note:** [[Training Diagnostics and Optimizers#12. Second Order Methods|Second-order optimization algorithms]] (like Newton's method) try to simulate this by computing the inverse Hessian, forcing **every** eigenspace to undergo 1-step convergence simultaneously.

### Regime 4: Monotonic Oscillations (Real, Negative Roots)
- **Condition:** Step size α too aggressive for this eigenvalue (α > 1/λᵢ)
- **Behavior:** Optimizer steps all the way across the canyon floor, landing higher on the other side but still converging
- **Loss curve:** Goes down overall but looks incredibly jagged — step-ladder pattern

### Regime 5: Divergence (Roots > 1)
- **Condition:** Step size completely unstable (α > 2/λᵢ for GD, or α > (2+2β)/λᵢ for momentum)
- **Behavior:** Loss rockets upward exponentially
- **Weights:** Explode to infinity

> [!important] Stability condition for momentum (generalization of [[#5. Condition Number and Optimal Step Size|§5]]):
> ```
> 0 < αλᵢ < 2 + 2β,   for 0 ≤ β < 1
> ```
> Momentum allows step size up to factor of (1+β) larger before diverging.

---

## 9. Critical Damping — The Optimal β

> [!tip] See Also
> - Practical μ heuristics (start 0.5 → anneal to 0.99): [[Training Diagnostics and Optimizers#9. Momentum Update|Momentum Update (TDO)]]
> - Critical damping result feeds directly into [[#10. Optimal Global Parameters and Quadratic Speedup|§10]]

**Physical analogy — damped harmonic oscillator:**

The momentum system (for small α) is a discretization of a damped harmonic oscillator:
- yᵢ^(k+1) = β yᵢ^k + λᵢ xᵢ^k → **velocity**, damped at each step, perturbed by external force field λᵢxᵢ^k
- xᵢ^(k+1) = xᵢ^k − α yᵢ^(k+1) → **position of particle**, moved each step by small amount in direction of velocity

**Spring-mass analogy:** Think of a weight suspended on a spring. Pull it down one unit and watch the path it takes back to equilibrium. β controls damping:

| β range | Regime | Description |
|---------|--------|-------------|
| β too small (near 0) | Overdamping | Particle in viscous fluid. Snaps back slowly, no oscillation. Equivalent to [[#1. Gradient Descent — Basics and Failure Modes|gradient descent]]. |
| β = critical value | **Critical damping** | Returns to equilibrium as fast as possible **without** oscillating. The sweet spot. |
| β too large (near 1) | Underdamping | Spring oscillates forever, missing optimal value. |

**Critical damping condition** — discriminant of R (from [[#7. The 2×2 Transition Matrix|§7]]) equals exactly zero:
```
(1−αλᵢ+β)² − 4β = 0
→  β* = (1−√(αλᵢ))²
```
Both eigenvalues coalesce to a single repeated root:
```
σ₁ = σ₂ = 1 − √(αλᵢ)
```
This gives convergence rate **(1−√(αλᵢ))** in eigenspace i — a square root improvement over GD's rate of (1−αλᵢ).

**Critical damping convergence rate (per eigenspace i):**
```
1 − √(αλᵢ)   vs GD's   1 − αλᵢ
```

> [!note] This √ improvement per eigenspace is the micro-level reason for the global O(√κ) speedup proven in [[#10. Optimal Global Parameters and Quadratic Speedup|§10]].

---

## 10. Optimal Global Parameters and Quadratic Speedup

> [!tip] See Also
> - Practical Adam LR (~3e-4): [[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum|Adam (TDO)]]
> - The LFO lower bound proves this speedup is **optimal** (can't do better): [[#14. Algorithmic Space and the LFO Lower Bound|§14]]
> - How the speedup manifests in graphs: [[#13. Colorization Problem — Momentum on a Graph|§13]]

To get a global convergence rate, optimize over both α and β simultaneously, using the [[#9. Critical Damping — The Optimal β|critical damping result]].

**Optimal parameters:**
```
α* = ( 2/(√λ₁ + √λₙ) )²

β* = ( (√λₙ − √λ₁)/(√λₙ + √λ₁) )²
```

**Resulting convergence rates:**

| Method | Convergence rate | Iterations to ε-accuracy |
|--------|-----------------|--------------------------|
| [[#1. Gradient Descent — Basics and Failure Modes\|Gradient Descent]] | (κ−1)/(κ+1) | O(κ) |
| **Momentum** | **(√κ−1)/(√κ+1)** | **O(√κ)** |

**The quadratic speedup:** Momentum has **square-rooted the condition number** [[#5. Condition Number and Optimal Step Size|κ]]. For κ=10,000: GD needs ~10,000 steps, momentum needs ~100.

For large κ:
- GD rate ≈ 1 − 2/κ
- Momentum rate ≈ 1 − 2/√κ

**Practical guidelines (when λ₁, λₙ not known explicitly):**
- Set β ≈ 1 and find largest α for which convergence holds
- Optimal α for momentum ≈ 4/λₙ (twice that of GD's 2/λₙ)
- β ≈ (√(λₙ/λ₁) − 1)² / (√(λₙ/λ₁) + 1)² ≈ 1

> [!warning] Optimal parameters give the fastest **asymptotic** convergence rate, not necessarily the fastest early convergence. See [[#14. Algorithmic Space and the LFO Lower Bound|LFO dead zone (§14)]].

---

## 11. Polynomial Regression — Concrete Application

> [!tip] See Also
> - Eigenfeature implicit regularization (early stopping): [[#12. Eigenfeatures and Implicit Regularization|§12]]
> - The optimization matrix A here is exactly the quadratic from [[#2. Convex Quadratic — The Canonical Setup|§2]]

**Setup:** Given 1D data {ξᵢ, dᵢ}, fit a degree-n polynomial model:
```
model(ξ) = w₁p₁(ξ) + w₂p₂(ξ) + ... + wₙpₙ(ξ)
```
where pᵢ(ξ) = ξ^(i−1).

**Objective:**
```
min_ω  ½‖Zω − d‖²
```
where Z is the **Vandermonde matrix:**
```
Z = [ 1  ξ₁  ξ₁²  ...  ξ₁^(n-1) ]
    [ 1  ξ₂  ξ₂²  ...  ξ₂^(n-1) ]
    [ ...                         ]
    [ 1  ξₘ  ξₘ²  ...  ξₘ^(n-1) ]
```

**Optimization matrix:** A = ZᵀZ — symmetric positive semidefinite. This is the matrix A from [[#2. Convex Quadratic — The Canonical Setup|§2]].

**The optimization landscape** is dictated entirely by A = ZᵀZ and its eigenstructure from [[#3. Eigendecomposition and Change of Basis|§3]].

**Eigenfeatures interpretation:**
- Rotating ω into Qω: ω̄ = Qᵀω (eigenvector coordinate weights)
- Counter-rotating features: p̄(ξ) = Qᵀp(ξ) (eigenfeatures)
- Model becomes: model(ξ) = ω̄₁p̄₁(ξ) + ... + ω̄ₙp̄ₙ(ξ)
- Optimization breaks into n independent 1D problems, each optimizable greedily

**Statistical interpretation:**
- Eigenfeatures sort features by sensitivity to perturbations in the data dᵢ
- Most robust components appear first (largest eigenvalues, fast convergence per [[#4. Closed Form of Gradient Descent|§4]])
- Most sensitive (noisy) components appear last (smallest eigenvalues, slow convergence)
- "Pathological directions" = eigenspaces that converge slowest = most sensitive to noise

---

## 12. Eigenfeatures and Implicit Regularization

> [!tip] See Also
> - Early stopping in practice: [[Gradient Descent Overview#3. Parallel/Distributed SGD & Additional Strategies|Early Stopping (GDO)]]
> - Stochastic noise also acts as implicit regularizer: [[#15. Stochastic Gradients|§15]]
> - Train/Val accuracy diagnostics for overfitting: [[Training Diagnostics and Optimizers#6. Train/Val Accuracy Curves|Train/Val Curves (TDO)]]

**Convergence of eigenspace** = reducing the component of the error vector lying within that eigenspace. Fast GD minimizes error along the direction of the specific eigenvector.

**Implicit Regularization via Early Stopping:**
- If you stop training early, the noisy, jagged, high-frequency components (small eigenfeatures) don't have enough time to grow
- By stopping early you get better generalizing results
- This is a form of regularization **without any explicit regularization parameter**

**Comparison to Tikhonov Regression (L1/L2 penalty):**
- Early stopping is **better** than Tikhonov because no regularization parameter has to be set once step size is decided
- In early stopping you have a full family of models from underfit to overfit from a single run
- By saving model weights at every epoch, you can choose the exact best-performing moment after the run is complete
- So infinite different model complexities from just a single run

> [!note] The stochastic noise in [[#15. Stochastic Gradients|§15]] provides a second, independent implicit regularization mechanism — even without early stopping.

---

## 13. Colorization Problem — Momentum on a Graph

> [!tip] This section gives an intuitive, visual demonstration of why the O(κ) → O(√κ) speedup matters. The condition number here is enormous (proportional to grid area L²).

**Problem setup:**
- On a grid of pixels, let G be the graph with vertices as pixels, E = edges connecting each pixel to its 4 neighbouring pixels
- D = small set of a few distinguished vertices (seed pixels)
- Minimize:
```
min_ω  ½ Σ_{i∈D} (ωᵢ−1)² + ½ Σ_{i,j∈E} (ωᵢ−ωⱼ)²
```
- First term (colorizer): pulls distinguished pixels to value 1
- Second term (smoother): spreads the color out

**Optimal solution:** A vector of all 1's.

**GD update for pixel i ∉ D:**
```
ωᵢ^(k+1) = ωᵢ^k − α Σⱼ∈N (ωᵢ^k − ωⱼ^k)
```
This is local averaging — weighted average of current value and its neighbours.

**Why GD is slow here:**
- Resembles the **diffusion equation (heat equation)** — O(L²) propagation
- Information travels at exactly 1 pixel per iteration
- Local smoothing directions correspond to large eigenvalues (fast, from [[#4. Closed Form of Gradient Descent|§4]])
- Global shift corresponds to microscopic λ₁ (extremely slow)
- The optimization matrix has a massive [[#5. Condition Number and Optimal Step Size|condition number]] K

**Momentum update for the same pixel:**
```
z^(k+1) = β z^k + Σⱼ (ωᵢ^k − ωⱼ^k)
ω^(k+1) = ω^k − α z^(k+1)
```
Instead of treating gradient as velocity, momentum treats it as a **force (acceleration)** acting on a particle with mass (compare [[Training Diagnostics and Optimizers#9. Momentum Update|TDO §9]]).

**Why momentum is fast here:**
- Resembles the **wave equation (propagation)** — O(L) propagation
- Diffusion: information flows like a viscous fluid, O(L²)
- Wave: information behaves like a shockwave rippling across water, O(L) — quadratic payoff
- When consecutive pixels agree on value 1, it snowballs through the grid
- This is the O(√κ) speedup from [[#10. Optimal Global Parameters and Quadratic Speedup|§10]] made viscerally concrete

---

## 14. Algorithmic Space and the LFO Lower Bound

> [!important] This section proves that momentum is **optimal** — no linear first-order method can do better. This is the theoretical crown of these notes.

### Linear First Order (LFO) Methods

Both GD and momentum can be "unrolled":

**GD unrolled:**
```
ω^(k+1) = ω⁰ − α Σᵢ₌₀^k ∇f(ωⁱ)
```

**Momentum unrolled:**
```
ω^(k+1) = ω⁰ + α Σᵢ₌₀^k [(1−β^(k+1-i))/(1−β)] ∇f(ωⁱ)
```

**General form of all LFO methods:**
```
ω^(k+1) = ω⁰ + Σᵢ γᵢ^k ∇f(ωⁱ)         (scalar step sizes)
ω^(k+1) = ω⁰ + Σᵢ Γᵢ^k ∇f(ωⁱ)         (diagonal matrix step sizes)
```
This class covers most popular NN training algorithms: [[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum|Adam]], [[Training Diagnostics and Optimizers#13. AdaGrad — Per-Parameter Adaptive Learning Rate|AdaGrad]], Avg Gradient, Conjugate Gradient.

### Convex Rosenbrock — Where All LFO Methods Fail

**The function:**
```
fⁿ(ω) = ½(ω₁−1)² + ½ Σᵢ₌₁^(n-1) (ωᵢ−ωᵢ₊₁)² + (2/K−1)‖ω‖²
```
- Term 1: colorizer of one node (cf. [[#13. Colorization Problem — Momentum on a Graph|§13]])
- Term 2: strong coupling of adjacent nodes along a path
- Term 3: small regularization term

**Optimal solution:**
```
ωᵢ* = ((√K−1)/(√K+1))^i
```
[[#5. Condition Number and Optimal Step Size|Condition number]] approaches K as n→∞.

**The dead zone argument (lower bound proof):**

Each component of the gradient depends only on the values directly before and after it:
```
∇f(xₖ)ᵢ = 2ωᵢ − ωᵢ₋₁ − ωᵢ₊₁ + (A/(K−1))ωᵢ,   i ≠ 1
```
Starting from ω⁰ = 0, by induction:
```
ω^k = [w₁^k, w₂^k, ..., w_k^k, 0, 0, ..., 0]
```
After k iterations, components beyond the k-th are **exactly 0** regardless of which LFO algorithm is used. This is a fundamental information propagation limit — gradient signal travels at most 1 component per step. (Compare the colorization graph where GD propagates 1 pixel/step.)

**Lower bound:**
```
‖ω^k − ω*‖ ≥ ((√K−1)/(√K+1))^(k+1) · ‖ω⁰−ω*‖
```

**The optimality result:**
- This lower bound matches momentum's convergence rate [[#10. Optimal Global Parameters and Quadratic Speedup|from §10]] exactly
- As n→∞, condition number K → n, and the convergence rate that momentum promises matches the best any LFO method can do
- **We cannot do better than this**

> [!warning] Optimal parameters give the fastest **asymptotic** rate, but the dead-zone means early iterations are wasted regardless of algorithm choice.

---

## 15. Stochastic Gradients

> [!tip] See Also
> - Mini-batch SGD overview: [[Gradient Descent Overview#1. Gradient Descent Variants & Challenges|GD Variants (GDO)]]
> - Batch size and loss wiggle: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|Loss Curve Diagnostics]]
> - Implicit regularization also appears in early stopping: [[#12. Eigenfeatures and Implicit Regularization|§12]]

All prior proofs assume access to the true gradient ∇f(ω). In modern ML, full batch passes are very expensive. Instead, randomized approximations (minibatch sampling) are used as a plug-in replacement for ∇f(ω).

**Decomposition of stochastic gradient:**
```
∇f̃(ω) = ∇f(ω) + error(ω)
```
- ∇f(ω) = true gradient
- error(ω) = approximation error
- The estimator is unbiased: E[error(ω)] = 0

**Effect on system:** Like injecting a special kind of noise. On the quadratic, error terms separate cleanly. After the change of variables from [[#3. Eigendecomposition and Change of Basis|§3]]:

```
[yᵢ^(k+1)]  =  R [yᵢ^k]  +  εᵢ^k [1 ]
[xᵢ^(k+1)]       [xᵢ^k]           [−α]
```
where R is the [[#7. The 2×2 Transition Matrix|transition matrix from §7]].

**Unrolling the noisy recurrence:**
```
[yᵢ^k]  =  R^k [yᵢ⁰]  +  εᵢ^k Σⱼ₌₁^k R^(k-j) [1 ]
[xᵢ^k]         [xᵢ⁰]                             [−α]
```
- First term: noiseless deterministic iterates
- Second term: decaying sum of errors — **stochastic part**

**E[f(ω) − f(ω*)] breaks into a deterministic part and a stochastic part.**

**The fundamental tension:**
- Lowering step size α → reduces stochastic error BUT slows convergence rate (from [[#5. Condition Number and Optimal Step Size|§5]])
- Increasing β → causes errors to compound (momentum accumulates past noise)
- These cannot both be optimized simultaneously

**Despite this:** SGD with momentum has competitive performance because:
- Noise acts as an **implicit regularizer** (compare [[#12. Eigenfeatures and Implicit Regularization|§12]])
- Prevents overfitting to training set
- Often finds flatter, better generalizing minima

---

## Key Formulas — Quick Reference

| Quantity | Formula |
|---|---|
| GD update | ω^(k+1) = ω^k − α∇f(ω^k) |
| Momentum update | z^(k+1) = βz^k + ∇f(ω^k); ω^(k+1) = ω^k − αz^(k+1) |
| Closed form GD (eigenbasis) | xᵢ^k = (1−αλᵢ)^k · xᵢ⁰ |
| Optimal GD step size | α\* = 2/(λ₁+λₙ) |
| GD convergence rate | (κ−1)/(κ+1) |
| Condition number | κ = λₙ/λ₁ |
| Momentum transition matrix R | [[#7. The 2×2 Transition Matrix\|see §7]] |
| Critical damping β | (1−√(αλᵢ))² |
| Rate at critical damping | 1−√(αλᵢ) |
| Optimal momentum α\* | (2/(√λ₁+√λₙ))² |
| Optimal momentum β\* | ((√λₙ−√λ₁)/(√λₙ+√λ₁))² |
| Momentum convergence rate | (√κ−1)/(√κ+1) |
| LFO lower bound | ((√K−1)/(√K+1))^(k+1) · ‖ω⁰−ω\*‖ |

---

## Key Analogies and Examples Used

| Concept | Analogy / Example |
|---------|-------------------|
| Pathological curvature | Valleys, trenches, canals, ravines |
| Condition number | How close A is to singular; how robust A⁻¹b is to noise in b |
| Momentum critical damping | Weight suspended on a spring returning to equilibrium |
| Overdamping (small β) | Particle immersed in viscous fluid |
| Underdamping (large β) | Spring oscillating forever, missing optimal value |
| Convergence in eigenspace | Driving xᵢ^k to 0 |
| GD on colorization graph | Heat/diffusion equation — O(L²) information propagation |
| Momentum on colorization graph | Wave equation — O(L) shockwave propagation |
| Early stopping | Implicit regularization without any extra tuning parameter |
| LFO dead zone | Triangle on weights-vs-error plot; components beyond k-th are exactly 0 |
| Stochastic gradient | Injecting special noise; estimator is unbiased |

---

## Concept Map — How Sections Connect

```
§2 Quadratic Setup
  └─→ §3 Eigendecomposition (decouples everything)
        ├─→ §4 Closed-form GD: (1−αλᵢ)^k
        │     └─→ §5 Condition number κ = λₙ/λ₁
        │           └─→ GD rate (κ−1)/(κ+1)
        └─→ §7 Transition matrix R (momentum analog of §4)
              └─→ §8 Four regimes (from eigenvalues of R)
                    └─→ §9 Critical damping β* = (1−√(αλᵢ))²
                          └─→ §10 Global optimum → rate (√κ−1)/(√κ+1)
                                └─→ §14 LFO lower bound proves optimality

§11 Polynomial Regression ─ concrete instance of §2–§5
§12 Early Stopping ────────── implicit regularization from eigenstructure
§13 Colorization ──────────── O(L²) vs O(L); κ made visual
§15 Stochastic Gradients ──── noise enters §7 recurrence additively
```

---

*Notes transcribed by Musaib, 17/05/2025*