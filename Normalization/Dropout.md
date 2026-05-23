# Dropout — Comprehensive Notes

> [!abstract] About These Notes These notes cover the full theory and practice of Dropout (Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, 2014 — JMLR 15:1929–1958). The depth spans motivation, the formal model, the training/test-time asymmetry, the model-averaging interpretation, mathematical analysis via marginalization, and all experimental results. Every key idea is given both its formal statement and its intuition. Structured so each section builds on the last.
> 
> **Paper:** _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_ · Srivastava et al. · University of Toronto · JMLR 2014

---

## Overview of Topics (in order)

1. [[#1. The Core Problem — Overfitting in Large Neural Networks|Overfitting in large networks]] — why it happens and why standard fixes are insufficient
2. [[#2. The Key Idea — Thinned Networks and Exponential Ensembles|The key idea]] — dropping units, thinned networks, 2ⁿ implicit models
3. [[#3. Biological and Evolutionary Motivation|Biological motivation]] — the sexual reproduction analogy and co-adaptation
4. [[#4. Formal Model Description|Formal model]] — Bernoulli masks, forward pass equations, weight sharing
5. [[#5. Training — Backpropagation Through Dropped Networks|Training]] — backprop through thinned networks, max-norm regularization, momentum
6. [[#6. Test Time — The Weight Scaling Trick|Test time]] — weight scaling vs. Monte-Carlo averaging, why they agree
7. [[#7. Dropout as Approximate Model Averaging|Model averaging interpretation]] — geometric mean, Bayesian connection, exponential ensemble
8. [[#8. Effect on Features and Sparsity|Effect on features]] — co-adaptation, what features look like with and without dropout
9. [[#9. Hyperparameter Guide|Hyperparameter guide]] — p, network size, learning rate, momentum, max-norm
10. [[#10. Marginalizing Dropout — The Regularization View|Marginalizing dropout]] — linear regression closed form → ridge regression equivalence
11. [[#11. Multiplicative Gaussian Noise — A Generalization|Gaussian dropout]] — generalization, entropy, equivalence of Bernoulli and Gaussian forms
12. [[#12. Dropout RBMs|Dropout RBMs]] — extension to graphical models
13. [[#13. Experimental Results — Summary|Experimental results]] — vision, speech, text, genetics across seven datasets
14. [[#14. Key Formulas — Quick Reference|Quick Reference]] — all formulas in one place
15. [[#15. Concept Map — How Everything Connects|Concept Map]]

---

## 1. The Core Problem — Overfitting in Large Neural Networks

> [!danger] The Fundamental Tension Deep neural networks are expressive enough to memorize training data. More parameters → more expressive → more overfitting. But restricting parameters means less capacity to learn complex functions. Dropout resolves this tension.

### Why Large Networks Overfit

Deep networks with many non-linear layers can learn very complicated relationships between inputs and outputs. With limited training data, many of these relationships are the result of **sampling noise** — they exist in the training set but not in the real data distribution.

> [!intuition] What "sampling noise" means Imagine you train on 60,000 images. Some quirky feature (e.g., background color correlating with label) might hold by coincidence in that sample but not in general. A powerful network will latch onto it.

**Existing fixes and their problems:**

|Method|Limitation|
|---|---|
|Early stopping|Wastes potential capacity; hard to tune the stopping point|
|L1 / L2 weight decay|Shrinks all weights globally; doesn't break specific co-adaptations|
|Soft weight sharing|Complex to implement; limited effectiveness|
|Model averaging (train many nets)|Exponentially expensive; each large net costs a lot to train and evaluate|

The ideal solution — **average predictions of all possible parameter settings** weighted by posterior probability — is Bayesian model averaging. It is mathematically the right answer but computationally intractable for large networks.

### The Exponential Ensemble Problem

With n units in a network, there are 2ⁿ possible subsets of units. Training all 2ⁿ models separately:

- Requires training 2ⁿ separate networks (exponential compute)
- Each model would have very little data to train on (data spread thin)
- Using all 2ⁿ at test time is infeasible for real-time applications

Dropout solves all three problems simultaneously.

---

## 2. The Key Idea — Thinned Networks and Exponential Ensembles

> [!tip] The Core Insight Instead of training 2ⁿ networks separately, train all of them at once with **shared weights**. Each training step randomly samples and trains one thinned sub-network. At test time, approximate the ensemble with a single forward pass using scaled weights.

### What "Dropping a Unit" Means

Dropping a unit means **temporarily removing it from the network, along with all its incoming and outgoing connections** for one forward/backward pass. It is as if the unit does not exist for that training step.

```
Standard network:  all n units active
Thinned network:   random subset of units active (each retained with probability p)
```

### The 2ⁿ Implicit Models

A network with n units can be seen as a collection of **2ⁿ possible thinned networks**. All of these share the same weight parameters (weight sharing), so the total number of parameters remains O(n²) regardless of how many sub-models are implicitly represented.

> [!important] This is the key to dropout's efficiency We get the variance-reduction benefit of model averaging (exponentially many models vote) without exponentially many training runs. The cost is just a stochastic mask per training step.

**During training:** For each training example, a new thinned network is sampled and trained. Each sub-network gets trained very rarely (if at all), but gradients accumulate into shared weights — so all sub-networks benefit from every training step.

**At test time:** Use a single unthinned network with weights scaled down by p. This approximates the geometric mean of all 2ⁿ thinned networks' predictions.

---

## 3. Biological and Evolutionary Motivation

> [!note] This section motivates _why_ the specific mechanism of dropout (random independent masking) is a good inductive bias — not just any regularizer.

### The Sexual Reproduction Analogy

**Asexual reproduction:** offspring inherits a slightly mutated copy of the parent's genome. Good co-adapted gene sets are passed on intact. Intuitively, this should optimize individual fitness better.

**Sexual reproduction:** offspring gets half the genes from each parent + small random mutations. This _breaks up_ co-adapted gene sets. Naively, this seems worse — why break up things that work well together?

**The evolutionary answer:** over the long term, natural selection may optimize not individual fitness but **mix-ability of genes** — the ability of a gene to work well with a _random_ set of partners. Each gene must be useful _on its own_ rather than relying on a specific large conspiracy of co-adapted partners.

### The Neural Network Parallel

In a standard network, each hidden unit can learn to **fix the mistakes of other units**. Units co-adapt — they form a "conspiracy" that works for the training data but fails on novel test data (since the specific co-adaptation is fragile).

With dropout, any hidden unit might be absent at any step. **A unit cannot rely on specific other units being present.** It must learn to be useful in a wide variety of contexts — with many different random partners. This forces each unit to detect a genuinely useful feature on its own.

> [!intuition] The conspiracy analogy "Ten conspiracies each involving five people is probably a better way to create havoc than one big conspiracy requiring fifty people to all play their parts correctly."
> 
> Complex co-adaptations work on training data but fail on novel test data. Multiple simpler, independent features generalize better.

---

## 4. Formal Model Description

### Standard Network (No Dropout)

For a network with L hidden layers, indexing layers by l ∈ {1, ..., L}:

- z^(l) = vector of inputs into layer l
- y^(l) = vector of outputs from layer l (y^(0) = x, the input)
- W^(l), b^(l) = weights and biases at layer l

**Standard forward pass (for any hidden unit i):**

```
Step 1: Compute pre-activation
        z_i^(l+1) = W_i^(l+1) · y^(l) + b_i^(l+1)

Step 2: Apply nonlinearity
        y_i^(l+1) = f(z_i^(l+1))
```

where f is any activation function (sigmoid, ReLU, etc.).

### Dropout Network

**Forward pass with dropout:**

```
Step 1: Sample binary mask for layer l
        r_j^(l) ~ Bernoulli(p)     [independently for each unit j]

Step 2: Apply mask to layer output (element-wise product)
        ỹ^(l) = r^(l) * y^(l)

Step 3: Compute pre-activation using masked output
        z_i^(l+1) = W_i^(l+1) · ỹ^(l) + b_i^(l+1)

Step 4: Apply nonlinearity
        y_i^(l+1) = f(z_i^(l+1))
```

**The mask r^(l):** a vector of independent Bernoulli(p) random variables. If r_j^(l) = 1, unit j is retained; if r_j^(l) = 0, unit j is dropped (its output becomes 0 for this step).

> [!note] The mask is resampled **for each training example in each mini-batch**. Every forward/backward pass uses a different thinned network.

### Test Time

Weights are scaled down at test time:

```
W_test^(l) = p · W^(l)
```

**Why this scaling?** During training, a unit is present with probability p. Its expected contribution to the next layer is `p · w · y`. At test time, the unit is always present, so its contribution would be `w · y` — too large by a factor of 1/p. Multiplying weights by p restores the expected value.

> [!warning] Inverted Dropout (Modern Practice) In modern frameworks (PyTorch, TensorFlow), the **inverted dropout** convention is used instead: scale activations **up by 1/p during training**, and use unmodified weights at test time. This is mathematically equivalent but more convenient — you can turn dropout on/off without changing the rest of the network.
> 
> ```
> Training: ỹ^(l) = (r^(l) / p) * y^(l)    # scale up during training
> Test:      y^(l) used as-is                 # no scaling needed
> ```

---

## 5. Training — Backpropagation Through Dropped Networks

### Backpropagation

Dropout networks are trained with standard SGD. The only difference is that for each training step, a thinned sub-network is sampled, and forward + backward pass are done **only on that sub-network**.

- Units not in the thinned network contribute a gradient of zero for that step
- Gradients for each parameter are averaged over training cases in the mini-batch
- The shared weights accumulate gradient signal from many different thinned networks

### Max-Norm Regularization

> [!important] The paper identifies max-norm regularization as especially synergistic with dropout. This combination is stronger than either alone.

**Max-norm constraint:** for each hidden unit, constrain the L2 norm of its incoming weight vector:

```
‖w‖₂ ≤ c
```

where c is a tunable hyperparameter (typical values: c = 3 to 4).

**Enforcement:** if the weight vector moves outside the ball of radius c during a gradient step, project it back onto the surface:

```
if ‖w‖₂ > c:
    w ← w · (c / ‖w‖₂)
```

**Why max-norm works well with dropout:**

Dropout introduces large noise in gradients — different sub-networks pull weights in different directions. High learning rates are needed to compensate. But high learning rates risk weight explosion. Max-norm provides a hard ceiling that prevents this, allowing you to use both high learning rates and high momentum simultaneously without instability.

> [!intuition] Max-norm as a ball constraint Think of each hidden unit's weights as a vector in weight space. Max-norm keeps it inside a ball of radius c. High learning rates + dropout let you _explore_ the ball efficiently. Without max-norm, the weights would escape to infinity.

### Recommended Training Configuration

The paper found the best results using all four together:

1. **Dropout** — the core regularizer
2. **Max-norm regularization** (c = 3–4 typically)
3. **Large decaying learning rate** (10–100× that of standard nets)
4. **High momentum** (0.95–0.99 vs. the standard 0.9)

**Rationale for high LR + high momentum:** Dropout noise causes gradients to cancel each other out. High LR compensates for the effectively reduced signal. High momentum helps "smooth out" the noisy gradient trajectory over many steps.

### Pretraining with Dropout

If the network is pretrained (e.g., using RBMs or autoencoders):

- Pretraining proceeds without dropout (standard procedure)
- Pretrained weights must be **scaled up by 1/p** before dropout fine-tuning begins

_Why scale up?_ During pretraining, all units are always present. During dropout fine-tuning, each unit is present only with probability p. To maintain the same expected activation magnitudes, inflate the weights by 1/p. Use a smaller learning rate than for randomly initialized nets to preserve the pretrained information.

---

## 6. Test Time — The Weight Scaling Trick

> [!tip] Critical Distinction Training uses a stochastic thinned network. Test time uses a single deterministic network with scaled weights. This asymmetry is fundamental — getting it wrong means your test-time predictions are systematically wrong.

### Why Weight Scaling Approximates the Ensemble

**True ensemble prediction** (intractable):

```
ŷ_ensemble = (1 / 2ⁿ) · Σ_{all thinned networks m} f(x; W_m)
```

This requires 2ⁿ forward passes — exponential.

**Weight scaling approximation** (one forward pass):

```
ŷ_approx = f(x; p · W)
```

Scale all weights by p. The expected output at any unit matches the expected output under dropout — because:

```
E[output of unit] = p · (output when present) + (1-p) · 0 = p · w·y
```

Multiplying weights by p achieves exactly this expectation.

> [!warning] This is an _approximation_, not exact The weight scaling trick gives the correct expected activation at each unit individually. But since neural networks are non-linear, E[f(x)] ≠ f(E[x]) in general. The geometric mean of sub-network predictions ≠ the prediction of the mean-weight network. In practice, the approximation works extremely well.

### Monte-Carlo vs. Weight Scaling

**Monte-Carlo averaging:** sample k thinned networks at test time, average their predictions:

```
ŷ_MC = (1/k) · Σ_{j=1}^{k} f(x; W_j)    where W_j is a random thinned network
```

As k → ∞, this converges to the true ensemble average.

**Empirical finding (MNIST):** At k ≈ 50 samples, Monte-Carlo averaging matches the weight scaling method. Beyond k = 50, Monte-Carlo is slightly better but within one standard deviation. The weight scaling method is a very good approximation of the true model average.

**Practical takeaway:** Weight scaling (one forward pass) is the standard. Monte-Carlo is only needed if you want the best possible accuracy and can afford 50× the compute.

---

## 7. Dropout as Approximate Model Averaging

> [!tip] This section provides the theoretical grounding for why dropout works beyond the heuristic explanation.

### The Geometric Mean Interpretation

Dropout approximates an **equally-weighted geometric mean** of the predictions of 2ⁿ models with shared weights.

**Why geometric mean, not arithmetic?**

- The geometric mean of probabilities is the natural combination for probabilistic predictions
- It corresponds to the model that maximizes likelihood under a uniform prior over models
- It's more robust to outlier predictions than the arithmetic mean

**Why "equally-weighted"?** Each thinned network is sampled with equal probability (each unit independently retained with prob p). This is not the Bayesian gold standard (which would weight by posterior probability), but it's a reasonable approximation and far cheaper.

### Comparison with Bayesian Neural Networks

|Property|Dropout|Bayesian NN|
|---|---|---|
|Model weighting|Equal|Posterior-weighted (correct)|
|Computational cost|Low (one forward pass at test)|High (many samples or variational methods)|
|Scales to large networks|Yes|Difficult|
|Theoretical grounding|Approximate|Exact (in principle)|
|Best when data is...|Moderate to large|Scarce|

**Practical result (Alternative Splicing dataset):** Bayesian NNs outperform dropout (as expected — they're theoretically more correct), but dropout significantly outperforms all other methods and is far cheaper. Dropout closes most of the gap between standard NNs and Bayesian NNs.

### Connection to Denoising Autoencoders

Dropout is related to Denoising Autoencoders (Vincent et al., 2008), which add noise to inputs and train the network to reconstruct the clean version. The difference: dropout applies to hidden layers too (not just inputs), uses a much higher noise rate (50% vs. DAE's 5%), and is interpreted as model averaging rather than denoising.

---

## 8. Effect on Features and Sparsity

### Co-Adaptation and Feature Quality

**In a standard network:** each parameter update is made with the knowledge of what all other units are doing simultaneously. Units can specialize to fix each other's errors — they co-adapt. This produces complex joint features that work on training data but are brittle on novel data.

**With dropout:** a unit cannot rely on any specific other unit being present. It must learn features that are independently useful in many different contexts.

**Visual evidence:** Autoencoders trained on MNIST:

- **Without dropout:** learned features are entangled — no single filter detects a meaningful feature on its own. Features are globally messy.
- **With dropout (p = 0.5):** learned features are localized, interpretable edges, strokes, and spots. Each hidden unit detects something meaningful on its own.

> [!intuition] Why this matters for generalization If a feature is only useful in combination with very specific other features, it won't generalize. If each feature is independently meaningful, the network can compose them flexibly for novel inputs.

### Induced Sparsity

A notable side effect: dropout spontaneously produces **sparse hidden unit activations**, even without any explicit sparsity regularizer.

**Without dropout:** mean activation of hidden units ≈ 2.0. Many units have high activation simultaneously — dense representations.

**With dropout (p = 0.5):** mean activation drops to ≈ 0.7. Very few units have high activation for any given input — sparse representations.

**Why sparsity emerges:**

- Units must be useful in many different contexts (with different random co-activating units)
- A unit that fires for everything is no more useful than the background — it needs to be selective
- Dropout implicitly incentivizes each unit to fire only when it has something specific and useful to contribute

Sparse representations are generally considered better for generalization — they encode information more efficiently and have lower overlap between representations of different inputs.

---

## 9. Hyperparameter Guide

> [!tip] The paper's Appendix A provides practical guidance rarely found in ML papers. Reproduced here with added intuition.

### Network Size

**Rule of thumb:** if a layer of size n is optimal for a standard net, use at least **n/p** units in a dropout net.

**Why?** After dropout, only pn units are active in expectation. Moreover, these units are not allowed to build co-adaptations freely (they form a different random subset each time). So you need more raw units to get the same effective capacity.

> [!example] If your standard net uses 512 units per layer with p = 0.5 → use at least 1024 units. The dropout net will have the same effective capacity but much better regularization.

### Dropout Rate p

**Default:** p = 0.5 for hidden layers, p = 0.8–1.0 for input layers.

**Effect of varying p (with fixed network size n):**

- p too small (e.g., 0.1): very few units active → severe underfitting, even training error is high
- p in [0.4, 0.8]: flat region — performance is relatively stable in this range
- p close to 1: dropout disabled → overfitting returns

**Effect of varying p (with fixed expected active units pn):**

- Small p + large n vs. large p + small n: the small-p version performs better because the larger test network (all n units) has more capacity
- Performance is more stable and the minimum is around p = 0.6, with p = 0.5 being a safe default

**Input layer:** typical p = 0.8 for continuous features (images, speech). Keep more input features since each individual pixel/feature is less informative.

### Learning Rate

Use **10–100× the learning rate** that would be optimal without dropout.

**Why?** Dropout introduces large noise in the gradient — many gradient signals cancel. A higher learning rate compensates for the effectively reduced signal-to-noise ratio in each update.

### Momentum

Use momentum **0.95 to 0.99** (vs. the standard 0.9).

**Why?** High momentum smooths out the noisy gradient trajectory. It effectively averages gradients over more steps, reducing the variance introduced by dropout noise.

### Max-Norm Constant c

Typical values: **c = 3 to 4** for fully connected layers.

Prevents the combination of high LR + high momentum from causing weight explosion. It's a hard constraint — the ball projection is applied after every gradient step.

### Summary Table

|Hyperparameter|Standard Net|Dropout Net|
|---|---|---|
|Hidden units per layer|n|≥ n/p|
|Dropout p (hidden)|—|0.5 (default)|
|Dropout p (input)|—|0.8 (continuous); omit for discrete|
|Learning rate|α|10α to 100α|
|Momentum|0.9|0.95 to 0.99|
|Max-norm c|optional|3–4 (strongly recommended)|
|Weight decay|moderate|can reduce; max-norm replaces it|

---

## 10. Marginalizing Dropout — The Regularization View

> [!tip] This section shows that dropout, when marginalized over the noise distribution, is equivalent to a form of L2 regularization (in the linear case). This gives a precise mathematical characterization of _what_ dropout regularizes.

### Setup: Dropout on Linear Regression

**Problem:** find w ∈ ℝᴰ minimizing:

```
‖y − Xw‖²
```

where X ∈ ℝᴺˣᴰ is the data matrix and y ∈ ℝᴺ are targets.

**Apply input dropout:** each input dimension is retained with probability p. The perturbed input is R * X where R ∈ {0,1}ᴺˣᴰ has Rᵢⱼ ~ Bernoulli(p) independently.

### Deriving the Marginalized Objective

**Stochastic objective with dropout:**

```
minimize_w  E_{R ~ Bernoulli(p)} [ ‖y − (R * X)w‖² ]
```

**Expanding the expectation:**

Step 1 — Expand the squared norm:

```
E[‖y − (R*X)w‖²] = E[‖y‖² − 2yᵀ(R*X)w + wᵀ(R*X)ᵀ(R*X)w]
```

Step 2 — Use linearity of expectation and E[Rᵢⱼ] = p:

```
E[(R*X)w] = pXw
```

Step 3 — For the quadratic term, use E[Rᵢⱼ²] = p and E[Rᵢⱼ · Rᵢₖ] = p² for j ≠ k (independence):

```
E[(R*X)ᵀ(R*X)]_jk = p·(XᵀX)_jk    if j ≠ k
                   = p·(XᵀX)_jj     if j = k
```

This gives: `E[(R*X)ᵀ(R*X)] = p²·XᵀX + p(1-p)·diag(XᵀX)`

Step 4 — Combine to get the marginalized objective:

```
minimize_w  ‖y − pXw‖² + p(1-p) ‖Γw‖²
```

where **Γ = diag(XᵀX)^(1/2)** — the diagonal matrix of feature standard deviations.

### Interpretation: Data-Dependent Ridge Regression

**This is ridge regression** (L2 regularization) with a special structure:

- The regularization penalty on weight wᵢ is **proportional to the standard deviation of feature xᵢ** (via Γᵢᵢ)
- Features with high variance are penalized more strongly
- Features with low variance (less informative, potentially noise) are penalized less

> [!intuition] Why this makes sense If a feature varies a lot across the training set, a high weight on it creates high variance in predictions. The regularizer penalizes this. If a feature hardly varies, a large weight on it doesn't hurt much — the regularizer is light. This is adaptive, data-dependent regularization — smarter than uniform L2 decay.

### Dependence of Regularization Strength on p

Substituting w̃ = pw (rescaling):

```
minimize_{w̃}  ‖y − Xw̃‖² + ((1-p)/p) ‖Γw̃‖²
```

The regularization constant is **(1-p)/p**:

- p = 1 (no dropout): regularization = 0
- p = 0.5: regularization = 1
- p = 0.1 (heavy dropout): regularization = 9

More dropout → stronger regularization. This is the precise mathematical sense in which p controls regularization intensity.

### What About Deep Networks?

For logistic regression and deep networks, a closed form does not exist. Wang & Manning (2013) showed that the marginalized model can be approximated under Gaussian assumptions (the "Fast Dropout" method). The Gaussian approximation weakens with more layers, so it's not directly applicable to deep networks — but the linear regression result gives strong intuition.

---

## 11. Multiplicative Gaussian Noise — A Generalization

> [!note] This section shows dropout is a special case of a broader family: multiplying activations by random noise variables.

### Bernoulli Dropout Revisited

Standard dropout multiplies hidden activation hᵢ by a Bernoulli variable rᵦ:

```
rᵦ = 1/p  with probability p
rᵦ = 0    with probability (1-p)
```

(using the inverted dropout convention)

Key statistics: E[rᵦ] = 1 and Var[rᵦ] = (1-p)/p.

### Gaussian Dropout

Replace the Bernoulli mask with multiplicative Gaussian noise:

```
hᵢ → hᵢ · rg     where rg ~ N(1, σ²)
```

or equivalently:

```
hᵢ → hᵢ + hᵢ · ε     where ε ~ N(0, σ²)
```

("add noise proportional to the activation magnitude")

**Setting σ² = (1-p)/p** makes Gaussian dropout match Bernoulli dropout in the first two moments:

- E[rg] = 1 (same expected activation)
- Var[rg] = (1-p)/p (same variance as Bernoulli)

**Key advantage of Gaussian dropout:** No weight scaling needed at test time — since E[rg] = 1, expected activations are already correct without any correction.

### Entropy Comparison

Given fixed mean and variance, Gaussian noise has **higher entropy** than Bernoulli noise (which has the _lowest_ entropy among noise distributions with the same moments).

**Empirical result:** Gaussian dropout performs slightly better than Bernoulli dropout (MNIST: 0.95% vs. 1.08% error). The higher entropy (more randomness) may provide stronger regularization.

> [!note] This generalizes further. Any noise distribution with E[r] = 1 can serve as dropout noise. The Bernoulli and Gaussian are the two most studied extremes. This perspective connects dropout to the broader framework of noisy neural networks and multiplicative noise regularization.

---

## 12. Dropout RBMs

> [!note] A brief but important extension — dropout is not limited to feed-forward nets. It applies to any model where units can be randomly masked.

### Standard RBM

A Restricted Boltzmann Machine defines a joint distribution over visible units v ∈ {0,1}ᴰ and hidden units h ∈ {0,1}ᶠ:

```
P(h, v; θ) = (1/Z(θ)) · exp(vᵀWh + aᵀh + bᵀv)
```

where θ = {W, a, b} are parameters and Z is the partition function.

### Dropout RBM

Augment with a binary mask r ∈ {0,1}ᶠ, rⱼ ~ Bernoulli(p) independently:

```
P(r, h, v; p, θ) = P(r; p) · P(h, v | r; θ)
```

If rⱼ = 0, then hⱼ is forced to 0 (dropped). If rⱼ = 1, hⱼ behaves as in a standard RBM.

**Conditional distributions:**

```
P(hⱼ = 1 | rⱼ, v) = 1(rⱼ = 1) · σ(bⱼ + Σᵢ Wᵢⱼ vᵢ)
```

When rⱼ = 0, hⱼ is deterministically 0. When rⱼ = 1, standard RBM conditional applies.

Conditioned on r, the Dropout RBM is equivalent to an RBM using only the retained hidden units. So a Dropout RBM is a **mixture of 2ᶠ RBMs with shared weights** — the same ensemble interpretation as in the feed-forward case.

**Learning:** Use standard Contrastive Divergence (CD-1). Only difference: sample r first, then use only retained hidden units for the CD update.

**Effects observed:**

- Features are coarser and more distinct in Dropout RBMs vs. standard RBMs
- Much sparser hidden unit activations (same phenomenon as in feed-forward case)
- Fewer "dead" units (units that never activate)

---

## 13. Experimental Results — Summary

### Overview of Datasets

|Dataset|Domain|Input Dim|Train Size|Test Size|
|---|---|---|---|---|
|MNIST|Vision (digits)|784|60K|10K|
|SVHN|Vision (street numbers)|3072|600K|26K|
|CIFAR-10|Vision (natural images)|3072|60K|10K|
|CIFAR-100|Vision (natural images)|3072|60K|10K|
|ImageNet|Vision (large scale)|65536|1.2M|150K|
|TIMIT|Speech|2520|1.1M frames|58K frames|
|Reuters-RCV1|Text|2000|200K|200K|
|Alternative Splicing|Genetics|1014|2932|733|

### Key Results

**MNIST:**

|Method|Error %|
|---|---|
|Standard NN, 2 layers, 800 units (Simard et al.)|1.60|
|SVM (Gaussian kernel)|1.40|
|Dropout NN, 3 layers, 1024 units (sigmoid)|1.35|
|Dropout NN, 3 layers, 1024 units (ReLU)|1.25|
|Dropout + max-norm, ReLU, 3 layers 1024|1.06|
|Dropout + max-norm, ReLU, 2 layers 8192|**0.95**|
|DBM + dropout fine-tuning|**0.79**|

Notable: the 2-layer, 8192-unit network has >65 million parameters trained on 60,000 examples. Normally this would massively overfit. Dropout prevents it entirely, achieving 0.95% error without early stopping.

**SVHN:**

|Method|Error %|
|---|---|
|Multi-stage Conv Net (best no-dropout)|4.90|
|Conv Net + max-pooling (no dropout)|3.95|
|Conv Net + dropout (fully connected layers only)|3.02|
|Conv Net + dropout (all layers)|**2.55**|

Key finding: dropout in convolutional layers (which have fewer parameters) still helps — not because conv layers themselves overfit, but because adding noise to conv outputs regularizes the fully connected layers above.

**CIFAR-10 / CIFAR-100:**

|Method|CIFAR-10|CIFAR-100|
|---|---|---|
|Best without dropout|15.13%|42.51%|
|Dropout (fully connected layers)|14.32%|41.26%|
|Dropout (all layers)|**12.61%**|**37.20%**|

**TIMIT (Speech):**

|Method|Phone Error Rate %|
|---|---|
|Standard NN (6 layers)|23.4|
|Dropout NN (6 layers)|21.8|
|DBN-pretrained (8 layers)|20.7|
|DBN-pretrained (8 layers) + dropout|**19.7**|

**ImageNet (ILSVRC-2010):**

|Method|Top-1|Top-5|
|---|---|---|
|SIFT + Fisher Vectors|45.7%|25.7%|
|Conv Net + Dropout (AlexNet predecessor)|**37.5%**|**17.0%**|

ILSVRC-2012: Dropout Conv Net won the competition with a ~10 percentage point margin over the best non-neural-network methods.

**Alternative Splicing (vs. Bayesian NN):**

- Bayesian NN: 623 (code quality, higher = better)
- Dropout NN: 567
- Standard NN: 440
- Dropout loses to Bayes (as expected theoretically) but is much cheaper and far outperforms standard NN.

### Key Cross-Domain Findings

1. Dropout improves performance **consistently across all domains** — vision, speech, text, genetics
2. Improvements are largest where overfitting is most severe (small datasets, large networks)
3. Text (Reuters) shows smallest improvement — large dataset (200K training examples) means overfitting is less of an issue
4. Dropout enables training networks far larger than what would otherwise overfit

---

## 14. Key Formulas — Quick Reference

|Quantity|Formula|
|---|---|
|Bernoulli mask|r_j^(l) ~ Bernoulli(p)|
|Masked output|ỹ^(l) = r^(l) * y^(l)|
|Dropout forward pass|z_i^(l+1) = W_i^(l+1) · ỹ^(l) + b_i^(l+1)|
|Test-time weights|W_test^(l) = p · W^(l)|
|Inverted dropout (train)|ỹ^(l) = (r^(l) / p) * y^(l); test weights unchanged|
|Max-norm constraint|‖w‖₂ ≤ c; project if violated: w ← w · (c / ‖w‖₂)|
|Pretraining scale-up|W_finetune = W_pretrained / p|
|Network size heuristic|n_dropout ≥ n_standard / p|
|Recommended LR|α_dropout = 10α to 100α|
|Marginalized linear regression|min_w ‖y − pXw‖² + p(1-p) ‖Γw‖²|
|Regularization constant|(1-p)/p|
|Γ (feature std dev matrix)|Γ = diag(XᵀX)^(1/2)|
|Gaussian dropout noise|r_g ~ N(1, σ²); set σ² = (1-p)/p to match Bernoulli|
|2ⁿ implicit models|n = total number of units|
|MC samples to match weight scaling|k ≈ 50|

---

## 15. Concept Map — How Everything Connects

```
Problem: Large NNs overfit
  │
  └── Ideal fix: average all 2ⁿ models
        │
        └── Problem: exponentially expensive
  
Dropout: Sample and train thinned networks with shared weights
  │
  ├── Training: stochastic Bernoulli mask r^(l) ~ Bernoulli(p)
  │     ├── Each step trains a different sub-network
  │     └── Gradients shared → all 2ⁿ models improve simultaneously
  │
  ├── Test time: single forward pass with scaled weights (W_test = p·W)
  │     └── Approximates geometric mean of 2ⁿ model predictions
  │
  ├── Why it works:
  │     ├── Biological: forces each unit to be independently useful
  │     │     └── Cannot rely on specific co-activating partners
  │     ├── Feature quality: breaks co-adaptations → interpretable features
  │     ├── Sparsity: units become selective → sparse activations
  │     └── Regularization: equivalent to adaptive ridge regression (linear case)
  │
  ├── Best with:
  │     ├── Max-norm (c = 3-4): prevents weight explosion under high LR
  │     ├── High LR (10-100×): compensates for gradient noise
  │     ├── High momentum (0.95-0.99): smooths noisy gradient trajectory
  │     └── Larger network (n/p units): maintains capacity after dropout
  │
  ├── Generalizations:
  │     ├── Gaussian dropout: r_g ~ N(1, (1-p)/p) — higher entropy, same moments
  │     └── Dropout RBMs: mixture of 2ᶠ RBMs with shared weights
  │
  └── Mathematical grounding:
        ├── Linear case: marginalizing = adaptive ridge regression
        ├── Regularization strength = (1-p)/p
        └── Bayesian NNs: the theoretically correct but expensive version
```

---

## 16. What Dropout Does NOT Do — Common Misconceptions

> [!warning] Misconceptions

**Dropout does NOT always help.** For very small datasets (100–500 examples), the network can still memorize training data despite dropout noise. Dropout works best when the dataset is large enough that the noise doesn't prevent all learning, but small enough that overfitting is a real problem.

**Dropout does NOT make training faster.** It typically takes **2–3× longer** than standard training. Each training case uses a different sub-network, so gradients are noisy and parameter updates are less efficient. The paper is explicit about this.

**Dropout does NOT simply reduce effective model size.** The effective model has 2ⁿ sub-models with shared weights — this is very different from a smaller model. A smaller model would underfit; dropout allows you to train a large model that generalizes.

**Dropout is NOT the same as adding noise to weights.** Dropout masks entire units (inputs and outputs), not individual weights. The zero-out happens at the activation level, not the weight level.

**p = 0.5 is NOT always optimal.** It's a good default for hidden layers. Input layers typically need p = 0.8 or higher. For very small datasets or very large networks, different values may be better. Always tune on a validation set.

**Dropout does NOT remove the need for other regularization.** The paper consistently uses dropout _together_ with max-norm. L2 weight decay can sometimes be reduced but max-norm is still strongly recommended.

---

## Key Analogies and Intuitions Used in the Paper

|Concept|Analogy / Intuition|
|---|---|
|Co-adaptation problem|Units conspire to fix each other's mistakes; works on training data, fails on test|
|Why dropout breaks co-adaptations|Sexual reproduction: forces genes to be individually mix-able, not co-adapted|
|Each unit must be independently useful|Like a gene that must work with a random set of partners|
|Complex co-adaptations vs. simple ones|One 50-person conspiracy vs. ten 5-person conspiracies|
|Weight scaling at test time|Unit present with prob p during training → scale weight by p to match expected activation|
|Regularization strength|(1-p)/p: more dropout = higher regularization|
|Network size heuristic|Use n/p units so expected active units = n (same as standard net)|
|High LR with dropout|Gradient noise means many updates cancel — higher LR compensates|
|Max-norm + high LR|Ball constraint lets you explore aggressively without flying off to infinity|

---

_Notes compiled from Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014), JMLR 15:1929–1958 — Musaib, 19/05/2026_