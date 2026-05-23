# Optimization Notes — Map of Content

> [!abstract] Vault Overview Four interlocked notes covering gradient-based optimization from theory to practice. Use this file as your navigation hub.

---

##  The Four Files

|File|Focus|Depth|
|---|---|---|
|[[Gradient Descent Overview]]|Algorithm survey — equations, variants, strategies|Introductory / reference|
|[[Training Diagnostics and Optimizers]]|Debugging, heuristics, practical tuning|Applied / intermediate|
|[[Momentum]]|Mathematical theory — derivations, proofs, convergence|Graduate-level theory|
|[[Adam]]|Full Adam paper treatment — derivations, bias correction, convergence proof explained|Paper-level + annotated|

---

## 🔗 Concept Map — How Ideas Flow Across Files

```
THEORY (Momentum)                   PRACTICE (TDO)                SURVEY (GDO)
─────────────────                   ──────────────                ────────────
§1  GD Failure Modes ────────────→  §8  Vanilla SGD Update    ←── Batch/SGD/Mini-batch
§2  Convex Quadratic                §5  Loss Curve Diagnostics ←── (loss regime table)
§3  Eigendecomposition
│
├─→ §4  Closed-Form GD ──────────→  §5  Loss Curve (why slows) 
│         (1−αλᵢ)^k
│
├─→ §5  Condition Number κ ───────→  §7  Update/Weight Ratio
│         λₙ/λ₁                     §11 LR Annealing
│                                    §12 Second Order Methods ←── L-BFGS (GDO)
│
├─→ §6  Momentum Tweak ───────────→  §9  Momentum Update      ←── Momentum (GDO)
│                                    §10 Nesterov Momentum    ←── NAG (GDO)
│
├─→ §7  2×2 Transition Matrix R      
│         (momentum analog of §4)    
│
├─→ §8  Four Convergence Regimes ─→  §5  Loss Curve Regimes
│         ripples/monotone/1-step    §12 Second Order (Regime 3)
│
├─→ §9  Critical Damping β* ──────→  §9  μ heuristic (0.5→0.99)
│
├─→ §10 Quadratic Speedup ────────→  §15 Adam (momentum part)  ←── Adam (GDO)
│         O(κ) → O(√κ)
│
├─→ §11 Polynomial Regression
│
├─→ §12 Eigenfeatures / Early Stop → §6  Train/Val Curves      ←── Early Stopping (GDO)
│
├─→ §13 Colorization (wave vs diff)
│
├─→ §14 LFO Lower Bound ─────────→  (proves Nesterov optimal) ←── NAG (GDO)
│         momentum is OPTIMAL
│
└─→ §15 Stochastic Gradients ────→  §5  Batch size & wiggle
                                     §16 Hyperparameter Opt   ←── (GDO strategies)

AdaGrad ─────────────────────────→  §13 AdaGrad (TDO)         ←── AdaGrad (GDO)
  └── cache → ∞ flaw                                           ←── §7 Adam (Adam.md)
RMSProp ─────────────────────────→  §14 RMSProp (TDO)         ←── RMSprop (GDO)
  └── no bias correction flaw                                  ←── §7 Adam (Adam.md)
Adam = RMSProp + Momentum ───────→  §15 Adam (TDO)            ←── Adam (GDO)
  ├── Bias correction derivation                               ←── §4 Adam (Adam.md) ★
  ├── Regret bound O(1/√T)                                     ←── §5–6 Adam (Adam.md) ★
  ├── AdaMax (ℓ∞ generalization)                               ←── §8 Adam (Adam.md) ★
  └── SNR / trust region geometry                              ←── §3 Adam (Adam.md) ★
```

---

## 🧭 Reading Paths

### Path 1: "I want the full theory first"

[[Momentum#1. Gradient Descent — Basics and Failure Modes|M§1]] → [[Momentum#2. Convex Quadratic — The Canonical Setup|M§2]] → [[Momentum#3. Eigendecomposition and Change of Basis|M§3]] → [[Momentum#4. Closed Form of Gradient Descent|M§4]] → [[Momentum#5. Condition Number and Optimal Step Size|M§5]] → [[Momentum#6. Momentum — The Tweak|M§6]] → [[Momentum#7. The 2×2 Transition Matrix|M§7]] → [[Momentum#8. Four Convergence Regimes|M§8]] → [[Momentum#9. Critical Damping — The Optimal β|M§9]] → [[Momentum#10. Optimal Global Parameters and Quadratic Speedup|M§10]] → [[Momentum#14. Algorithmic Space and the LFO Lower Bound|M§14]]

### Path 2: "I need to debug a training run right now"

[[Training Diagnostics and Optimizers#4. Sanity Checks Before Training|TDO§4]] → [[Training Diagnostics and Optimizers#1. Gradient Checking|TDO§1–3]] → [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|TDO§5]] → [[Training Diagnostics and Optimizers#6. Train/Val Accuracy Curves|TDO§6]] → [[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates|TDO§7]] → [[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|TDO§11]]

### Path 3: "I want to understand all the optimizers"

[[Gradient Descent Overview#1. Gradient Descent Variants & Challenges|GDO§1]] → [[Gradient Descent Overview#2. Gradient Descent Optimization Algorithms|GDO§2]] → [[Training Diagnostics and Optimizers#9. Momentum Update|TDO§9]] → [[Training Diagnostics and Optimizers#13. AdaGrad — Per-Parameter Adaptive Learning Rate|TDO§13]] → [[Training Diagnostics and Optimizers#14. RMSProp|TDO§14]] → [[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum|TDO§15]]

### Path 5: "I want to deeply understand Adam specifically"

[[Adam#1. Motivation — What Problem Does Adam Solve?|Adam§1]] → [[Adam#2. The Algorithm|Adam§2]] → [[Adam#3. Adam's Update Rule — Geometry and Intuition|Adam§3]] → [[Adam#4. Initialization Bias Correction — Full Derivation|Adam§4]] → [[Adam#5. Convergence Analysis — The Regret Framework|Adam§5]] → [[Adam#6. The Convergence Proof — Step by Step|Adam§6]] → [[Adam#8. AdaMax — The ℓ∞ Generalization|Adam§8]] [[Momentum#6. Momentum — The Tweak|M§6]] (momentum part) → [[Momentum#9. Critical Damping — The Optimal β|M§9]] → [[Momentum#10. Optimal Global Parameters and Quadratic Speedup|M§10]] → [[Training Diagnostics and Optimizers#14. RMSProp|TDO§14]] (RMSProp part) → [[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum|TDO§15]]

---

## 🔑 The Central Quantities

|Quantity|Definition|Where derived|Where used|
|---|---|---|---|
|**Condition number κ**|λₙ/λ₁|[[Momentum#5. Condition Number and Optimal Step Size\|M§5]]|GD rate, momentum rate, colorization, LFO bound|
|**Convergence rate (GD)**|(κ−1)/(κ+1)|[[Momentum#5. Condition Number and Optimal Step Size\|M§5]]|Compared to momentum in M§10|
|**Convergence rate (momentum)**|(√κ−1)/(√κ+1)|[[Momentum#10. Optimal Global Parameters and Quadratic Speedup\|M§10]]|Proven optimal in M§14|
|**Transition matrix R**|2×2 matrix|[[Momentum#7. The 2×2 Transition Matrix\|M§7]]|Regimes (M§8), critical damping (M§9), stochastic (M§15)|
|**Critical damping β***|(1−√(αλᵢ))²|[[Momentum#9. Critical Damping — The Optimal β\|M§9]]|Practical μ heuristic (TDO§9)|
|**Update/weight ratio**|~1e-3 healthy|[[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates\|TDO§7]]|Proxy for optimal α*|

---

## ⚡ Key Insights at a Glance

> [!important] The Big Idea Chain
> 
> 1. GD is slow on ill-conditioned problems ([[Momentum#1. Gradient Descent — Basics and Failure Modes|M§1]])
> 2. The Hessian eigenstructure explains exactly _why_ ([[Momentum#3. Eigendecomposition and Change of Basis|M§3]]–[[Momentum#4. Closed Form of Gradient Descent|M§4]])
> 3. Condition number κ = λₙ/λ₁ quantifies the severity ([[Momentum#5. Condition Number and Optimal Step Size|M§5]])
> 4. Momentum square-roots κ: O(κ) → O(√κ) ([[Momentum#10. Optimal Global Parameters and Quadratic Speedup|M§10]])
> 5. This speedup is **provably optimal** among all first-order methods ([[Momentum#14. Algorithmic Space and the LFO Lower Bound|M§14]])
> 6. Adam approximates this + per-parameter scaling ([[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum|TDO§15]])

> [!tip] Practical Bottom Line
> 
> - Use **Adam** with LR ≈ 3e-4 as default ([[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum|TDO§15]])
> - Monitor **update/weight ratio** ≈ 1e-3 ([[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates|TDO§7]])
> - Use **step decay** for LR annealing ([[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|TDO§11]])
> - Always **gradcheck** with central differences + float64 ([[Training Diagnostics and Optimizers#1. Gradient Checking|TDO§1]])
> - **Early stopping** is free regularization ([[Momentum#12. Eigenfeatures and Implicit Regularization|M§12]])

---

_Notes transcribed by Musaib, 17/05/2025_