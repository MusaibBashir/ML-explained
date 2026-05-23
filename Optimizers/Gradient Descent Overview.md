# Overview of Gradient Descent Optimization Algorithms

> [!abstract] About These Notes A survey of gradient descent variants and optimization algorithms, from vanilla SGD through modern adaptive methods. Covers practical challenges, algorithm equations, and training strategies.
> 
> **Companion files:** [[Momentum]] (deep theoretical analysis) · [[Training Diagnostics and Optimizers]] (practical debugging & heuristics)

---

## 1. Gradient Descent Variants & Challenges

> [!tip] See Also
> 
> - Full theoretical analysis of why GD slows down: [[Momentum#1. Gradient Descent — Basics and Failure Modes|GD Failure Modes (Momentum §1)]]
> - The condition number κ as the precise measure of GD difficulty: [[Momentum#5. Condition Number and Optimal Step Size|Condition Number (Momentum §5)]]
> - Practical LR diagnostics: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|Loss Curve Diagnostics (TDO)]]
> - Practical SGD update: [[Training Diagnostics and Optimizers#8. Vanilla SGD Update|Vanilla SGD Update (TDO)]]

|**Concept / Algorithm**|**Description & Equations**|
|:--|:--|
|**1) Batch / Vanilla Gradient Descent**|• Uses whole dataset for one update.<br>• Does not allow online model updates.<br>• Guaranteed to converge to global minima for convex surfaces & to a local minimum for non-convex.<br><br>$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta)$|
|**2) Stochastic Gradient Descent (SGD)**|• One update every time (per training example).<br>• Frequent updates with high variance.<br>• When LR is slowly decreased, SGD shows the same convergence behaviour as batch GD.<br><br>$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta; x^{(i)}; y^{(i)})$|
|**3) Mini-batch Stochastic Gradient Descent**|• One update for every batch.<br>• Reduces variance of param updates → more stable convergence.<br>• **Algorithm of choice when training a Neural Network.**<br>• However, doesn't guarantee good convergence on its own.<br><br>$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta; x^{(i:i+n)}; y^{(i:i+n)})$|
|**Challenges of Optimization**|• **Learning Rate:** Hard to choose properly — see [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics\|Loss Curve Diagnostics]] and [[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates\|Update/Weight Ratio]].<br>• **Uniform Updates:** The same LR applies to all parameters — suboptimal for sparse data. AdaGrad/Adam fix this ([[#2. Gradient Descent Optimization Algorithms\|§2]]).<br>• **Saddle Points:** Deep networks often get trapped in saddle points rather than local minima.|

> [!note] The stochastic noise in SGD is analyzed mathematically in [[Momentum#15. Stochastic Gradients|Stochastic Gradients (Momentum §15)]]. Key result: noise is an unbiased estimator of the true gradient and acts as an **implicit regularizer**.

> [!note] Batch size and loss wiggle: [[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics|TDO §5]] — batch size = 1 → high wiggle; full dataset → near-monotone loss.

---

## 2. Gradient Descent Optimization Algorithms

|**Algorithm**|**Description & Equations**|
|:--|:--|
|**4) [[Momentum\|Momentum]]**|• SGD has trouble navigating ravines (areas curving much more steeply in one dimension) — see [[Momentum#1. Gradient Descent — Basics and Failure Modes\|pathological curvature]].<br>• Momentum helps accelerate SGD in relevant direction & dampens oscillations.<br>• For full theory (eigenanalysis, critical damping, O(√κ) speedup): [[Momentum\|Momentum notes]].<br>• For practical heuristics (μ start/anneal, physical intuition): [[Training Diagnostics and Optimizers#9. Momentum Update\|TDO §9]].<br><br>$v_t = \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta)$<br><br>$\theta = \theta - v_t$|
|**5) Nesterov Accelerated Gradient (NAG)**|• Gives our momentum term **prescience** (look-ahead capability).<br>• Computes the gradient not w.r.t. current parameters, but w.r.t. the approx future position.<br>• Stronger theoretical convergence guarantees — [[Momentum#14. Algorithmic Space and the LFO Lower Bound\|LFO lower bound (Momentum §14)]] proves this family is optimal among first-order methods.<br>• Practical implementation: [[Training Diagnostics and Optimizers#10. Nesterov Momentum\|TDO §10]].<br><br>$v_t = \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta - \gamma v_{t-1})$<br><br>$\theta = \theta - v_t$|
|**6) Adagrad**|• Adapts learning rate per parameter: larger updates for infrequent params, smaller for frequent.<br>• Highly suited for **sparse data** (e.g., training word embeddings like GloVe).<br>• Uses a default learning rate of 0.01, removing the need for manual tuning.<br>• **Flaw:** $G_t$ accumulates the sum of squared past gradients — keeps growing → LR shrinks to zero and learning stops.<br>• Practical notes: [[Training Diagnostics and Optimizers#13. AdaGrad — Per-Parameter Adaptive Learning Rate\|TDO §13]].<br><br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$|
|**7) Adadelta**|• Resolves Adagrad's radically diminishing learning rates.<br>• Instead of accumulating _all_ past gradients, restricts the window by using an **exponentially decaying average** of past squared gradients — same fix as [[Training Diagnostics and Optimizers#14. RMSProp\|RMSProp]].<br>• Also defines an exponentially decaying average of squared parameter updates to ensure units match.<br>• Eliminates the need to set a default learning rate entirely.<br><br>$\Delta \theta_t = - \frac{RMS[\Delta \theta]_{t-1}}{RMS[g]_t} g_t$|
|**8) RMSprop**|• Unpublished — proposed by Geoff Hinton in a Coursera class.<br>• Developed independently around the same time as Adadelta to resolve Adagrad's flaw.<br>• Identical to the first update vector of Adadelta.<br>• Full practical notes: [[Training Diagnostics and Optimizers#14. RMSProp\|TDO §14]].<br>• Suggested default: $\eta = 0.001$, decay rate = 0.9.<br><br>$E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g_t^2$<br><br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$|
|**9) Adam** (Adaptive Moment Estimation)|• Combines [[Training Diagnostics and Optimizers#14. RMSProp\|RMSProp]] ($v_t$, second moment/variance) and [[Training Diagnostics and Optimizers#9. Momentum Update\|Momentum]] ($m_t$, first moment/mean).<br>• Because $m_t$ and $v_t$ are initialized as vectors of 0's, they are heavily biased towards zero initially → **bias correction** fixes this.<br>• Full practical notes + recommended hyperparams: [[Training Diagnostics and Optimizers#15. Adam — RMSProp + Momentum\|TDO §15]].<br>• Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, LR ≈ 3e-4.<br><br>$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$<br><br>$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad \text{and} \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$<br><br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$|
|**10) AdaMax**|• Generalizes Adam's update rule from the $l_2$ norm to the more numerically stable $l_\infty$ norm ($u_t$).<br>• Because $u_t$ relies on a `max` operation, it is not susceptible to the zero-bias that affects Adam's $v_t$ → no bias correction needed for $u_t$.<br>• Defaults: $\eta = 0.002$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.<br><br>$u_t = \max(\beta_2 \cdot u_{t-1}, \vert g_t \vert)$<br><br>$\theta_{t+1} = \theta_t - \frac{\eta}{u_t}\hat{m}_t$|
|**11) Nadam**|• **Nesterov-accelerated Adaptive Moment Estimation** — NAG inside Adam.<br>• Instead of the previous bias-corrected momentum vector ($\hat{m}_{t-1}$) to update params, uses the _current_ momentum vector ($\hat{m}_t$) to look ahead (cf. [[Training Diagnostics and Optimizers#10. Nesterov Momentum\|Nesterov (TDO §10)]]).<br><br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\beta_1 \hat{m}_t + \frac{(1-\beta_1)g_t}{1-\beta_1^t})$|

> [!important] Algorithm Lineage
> 
> ```
> SGD
>  ├── + momentum    → Momentum
>  │    └── + lookahead  → Nesterov (NAG)
>  ├── AdaGrad (per-param LR)
>  │    └── + EMA cache  → RMSProp / Adadelta
>  │         └── + momentum → Adam
>  │              └── + Nesterov → Nadam
>  │              └── + l∞ norm  → AdaMax
> ```
> 
> Full practical notes for each: [[Training Diagnostics and Optimizers]] Full theory for momentum family: [[Momentum]]

---

## 3. Parallel/Distributed SGD & Additional Strategies

> [!tip] See Also
> 
> - Early stopping as **implicit regularization** (theoretical basis): [[Momentum#12. Eigenfeatures and Implicit Regularization|Eigenfeatures & Implicit Regularization (Momentum §12)]]
> - Gradient noise and stochastic gradients: [[Momentum#15. Stochastic Gradients|Stochastic Gradients (Momentum §15)]]
> - Train/Val curves to decide when to stop: [[Training Diagnostics and Optimizers#6. Train/Val Accuracy Curves|Train/Val Curves (TDO)]]

|**Concept**|**Description & Key Points**|
|:--|:--|
|**Parallel & Distributed SGD**|• **Hogwild!:** Allows SGD updates in parallel on CPUs with shared memory; optimal for sparse data (rarely overwrites info).<br>• **Downpour SGD:** Asynchronous replicas process subsets of data and send updates to a parameter server.<br>• **Elastic Averaging SGD (EASGD):** Links local parameters with an "elastic force" to a central variable, allowing more exploration.|
|**Shuffling & Curriculum Learning**|• **Shuffling:** Avoid providing data in a meaningful order to prevent algorithmic bias; shuffle data after every epoch.<br>• **Curriculum Learning:** For progressively harder problems, purposely supplying examples in order of increasing difficulty can improve convergence.|
|**Batch Normalization**|• Reestablishes zero mean and unit variance for every mini-batch.<br>• Allows use of higher learning rates and acts as a strong regularizer (often replacing the need for Dropout).<br>• Reduces internal covariate shift — effectively normalizes the curvature seen by each layer, reducing the [[Momentum#5. Condition Number and Optimal Step Size\|condition number κ]] that each layer's optimizer faces.|
|**Early Stopping**|• "Beautiful free lunch." Always monitor error on a validation set and stop training if it stops improving to prevent overfitting.<br>• Theoretical basis: [[Momentum#12. Eigenfeatures and Implicit Regularization\|Eigenfeatures & Implicit Regularization (Momentum §12)]] — stopping early prevents noisy small eigenfeatures from growing, giving the **entire underfit-to-overfit family** of models from a single run.<br>• Practical diagnosis: [[Training Diagnostics and Optimizers#6. Train/Val Accuracy Curves\|Train/Val Curves (TDO §6)]].|
|**Gradient Noise**|• Adding Gaussian noise $N(0, \sigma_t^2)$ to gradient updates makes networks more robust to poor initialization.<br>• $g_{t,i} = g_{t,i} + N(0, \sigma_t^2)$<br>• Gives deep networks a better chance to escape saddle points.<br>• Theoretical connection: [[Momentum#15. Stochastic Gradients\|Stochastic Gradients (Momentum §15)]] — mini-batch noise is already an unbiased implicit regularizer; gradient noise is a deliberate amplification of this effect.|

---

## Quick Cross-Reference: Where to Find the Theory

|You want to understand...|Go here|
|---|---|
|**Why** GD slows down near optimal|[[Momentum#4. Closed Form of Gradient Descent\|Closed-Form GD (Momentum §4)]]|
|**How bad** the slowdown will be|[[Momentum#5. Condition Number and Optimal Step Size\|Condition Number κ (Momentum §5)]]|
|**Why momentum is faster** (math)|[[Momentum#9. Critical Damping — The Optimal β\|Critical Damping (Momentum §9)]] + [[Momentum#10. Optimal Global Parameters and Quadratic Speedup\|§10]]|
|**Proof momentum is optimal**|[[Momentum#14. Algorithmic Space and the LFO Lower Bound\|LFO Lower Bound (Momentum §14)]]|
|**Diagnosing a bad training run**|[[Training Diagnostics and Optimizers#5. Loss Curve Diagnostics\|Loss Curve Diagnostics (TDO §5)]]|
|**Choosing hyperparameters**|[[Training Diagnostics and Optimizers#16. Hyperparameter Optimization\|Hyperparameter Optimization (TDO §16)]]|
|**Verifying gradients**|[[Training Diagnostics and Optimizers#1. Gradient Checking\|Gradient Checking (TDO §1–3)]]|
|**When to stop training**|[[Training Diagnostics and Optimizers#6. Train/Val Accuracy Curves\|Train/Val Curves (TDO §6)]] + [[Momentum#12. Eigenfeatures and Implicit Regularization\|Momentum §12]]|