# Batch Normalization — Written Notes

> **Paper:** Ioffe & Szegedy, ICML 2015 (arXiv:1502.03167) **Links:** [[Batch Normalization]] (full reference version) · [[Layer Normalization]] · [[Momentum]] · [[Adam]] · [[Training Diagnostics and Optimizers]]

---

## The Core Problem — Why Do We Need This?

When you train a deep network, every layer's weights update at every step. The problem is that when layer 1's weights change, the **distribution of the values it sends to layer 2 also changes**. Now layer 2 has to deal with a moving target — it spends its capacity re-adapting to the new distribution instead of learning the actual task. This gets worse and worse as you go deeper: by layer 10, the inputs have been distorted by the drift of nine layers above. This is called **internal covariate shift** — covariate shift (change in input distribution) happening _inside_ the network during training.

The two most painful consequences:

**1. Forces tiny learning rates.** Suppose you use a large LR. Layer 1's weights change a lot → layer 2's inputs shift a lot → the loss explodes or oscillates. To prevent this cascade, you're forced to use tiny LRs. Small LRs mean slow training.

**2. Sigmoid saturation + vanishing gradients.** The sigmoid $\sigma(x) = 1/(1+e^{-x})$ is nearly flat (gradient ≈ 0) when $|x|$ is large. As weights drift during training, $x = Wu + b$ drifts into large values → gradients vanish → earlier layers stop learning. This is the main reason deep networks with sigmoids were basically untrainable before BN.

> [!note] The LR connection to condition number From [[Momentum#5. Condition Number and Optimal Step Size|Momentum §5]], training slows when the condition number $\kappa = \lambda_n / \lambda_1$ is large. Internal covariate shift effectively increases $\kappa$ seen by each layer, because the curvature of the loss _as seen by that layer's parameters_ keeps changing. BN attacks this by stabilising the input distribution each layer sees.

---

## The Ideal Fix — Whitening (and Why It Doesn't Work Directly)

The ideal solution has been known since LeCun et al. (1998): **whiten the inputs to every layer**. Whitened inputs have:

- Zero mean
- Unit variance
- No correlations between features (off-diagonal covariance = 0)

If the inputs to every layer were always whitened, the layer's effective condition number would be 1 and training would be maximally fast. The whitened activation would be:

$$\hat{x} = \text{Cov}[x]^{-1/2} \cdot (x - \mathbb{E}[x])$$

where $\text{Cov}[x] = \mathbb{E}[xx^\top] - \mathbb{E}[x]\mathbb{E}[x]^\top$.

### Problem 1 — It's Computationally Intractable

For a layer with $d$ hidden units:

- Covariance matrix: $d \times d = O(d^2)$ entries to store and compute
- Inverse square root of that matrix: $O(d^3)$ eigendecomposition
- Has to be recomputed every parameter update, over the whole dataset
- You'd also need derivatives of the whitening operation for backprop

A layer with 1000 hidden units → $10^6$ entries in the covariance matrix. Not feasible.

### Problem 2 — Naive Whitening Outside the Gradient Breaks Everything

This is the subtle but crucial insight the paper makes. Suppose you do something simpler: just subtract the mean. So $\hat{x} = x - \mathbb{E}[x]$ where $x = u + b$ (the pre-activation with bias $b$). Now run a gradient step that updates $b \to b + \Delta b$:

$$\hat{x}_\text{new} = (u + b + \Delta b) - \mathbb{E}[u + b + \Delta b]$$

Expand the expectation — since $\Delta b$ is a constant added to every element:

$$= u + b + \Delta b - \mathbb{E}[u + b] - \Delta b = u + b - \mathbb{E}[u + b] = \hat{x}_\text{old}$$

**The $\Delta b$ terms cancel perfectly.** The normalised output is identical before and after the update to $b$.

> [!important] Why this is catastrophic Gradient descent computed $\partial \ell / \partial b$ and moved $b$ — but the output of the layer didn't change at all because the normalisation undid the update. In practice, what happens is that $b$ diverges to $\pm \infty$ while the loss stays flat. The optimiser is working, but it's wasting all its effort chasing a bias that the normalisation immediately removes. The model doesn't learn.
> 
> **The fix:** The normalisation must be _inside_ the computation graph — a differentiable operation that the gradient can flow through. Then the optimiser _knows_ the normalisation is there and accounts for it. That is exactly what BN does.

### The Two Simplifications BN Makes

Since full whitening is intractable, BN makes two practical compromises:

1. **Normalise each feature independently** — no decorrelation, just zero mean and unit variance per scalar activation. Turns $O(d^2)$ covariance into $O(d)$ scalars.
2. **Use mini-batch statistics** instead of full-dataset statistics — cheap to compute, and now the normalisation is differentiable with respect to the inputs in the batch.

Neither of these is perfect, but together they give most of the benefit at a tiny fraction of the cost.

---

## The BN Transform — How It Actually Works

BN operates on **one scalar activation** at a time, across a mini-batch of $m$ examples. If you have $d$ features in a layer, you run BN independently $d$ times — one for each feature. For one feature, the mini-batch values are $x_1, x_2, \ldots, x_m$ (one value per example).

BN has **four steps**.

---

### Step 1 — Compute the Mini-Batch Mean

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

This is just the arithmetic mean of the $m$ values in the batch for this feature.

**Why:** We want to centre the distribution. After subtracting $\mu_B$, the values have zero mean: $$\frac{1}{m} \sum_{i=1}^m (x_i - \mu_B) = \frac{1}{m}\left(\sum x_i\right) - \mu_B = \mu_B - \mu_B = 0 \checkmark$$

---

### Step 2 — Compute the Mini-Batch Variance

$$\sigma^2_B = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

This is the average squared deviation from the mean — the spread of the batch values.

**Why divide by $m$ and not $m-1$?** Normally in statistics you'd use $m-1$ (Bessel's correction) for an unbiased estimate of the population variance. Here we deliberately use $m$ because we're not trying to do inference — we just need a number to normalise by, and $m$ is simpler and numerically better-behaved. (At inference time, we'll use $m/(m-1)$ to get an unbiased population estimate — see [[#Training vs. Inference|§ Training vs. Inference]].)

---

### Step 3 — Normalise

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \varepsilon}}$$

**Breaking this down:**

- $(x_i - \mu_B)$: centres the value (subtracts the mean)
- $\sqrt{\sigma^2_B + \varepsilon}$: the standard deviation of the batch, with a small $\varepsilon$ added for numerical safety
- Dividing by the std dev scales the values so they have unit variance

**Proof that the result has unit variance:** $$\frac{1}{m}\sum_{i=1}^m \hat{x}_i^2 = \frac{1}{m}\sum_{i=1}^m \frac{(x_i - \mu_B)^2}{\sigma^2_B + \varepsilon} \approx \frac{\sigma^2_B}{\sigma^2_B} = 1 \checkmark \quad \text{(ignoring } \varepsilon \text{)}$$

**What is $\varepsilon$?** A tiny constant (typically $10^{-5}$). It prevents division by zero in the pathological case where all $m$ values in the batch are identical ($\sigma^2_B = 0$). Without it, a dead or constant activation would produce NaN.

**What $\hat{x}_i$ actually is:** The $z$-score of $x_i$ within the batch. You're measuring how many standard deviations $x_i$ is from the batch mean. Every value gets mapped into the same canonical distribution — zero mean, unit variance — regardless of what the raw values were.

---

### Step 4 — Scale and Shift (The Learned Re-Parameterisation)

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

Here $\gamma$ and $\beta$ are **learned parameters** — they are trained by gradient descent just like weights and biases. There is one $\gamma$ and one $\beta$ per feature.

**Why are these necessary?** Pure normalisation (step 3 only) is actually destructive to the network's capacity. Here's why:

Suppose this layer feeds into a sigmoid $\sigma(x)$. After normalisation, the input to the sigmoid always has unit variance — so its values are mostly in the range $[-2, 2]$. But the sigmoid is nearly linear in that range! We just fixed the vanishing gradient problem (good) but in doing so forced the sigmoid into its linear regime (bad — we've killed its nonlinearity). The network can no longer use sigmoid's saturation for gating or selection.

With $\gamma$ and $\beta$, the network can learn to _undo_ the normalisation if it wants. Setting $\gamma = \sigma_B$ and $\beta = \mu_B$ recovers the original distribution exactly. So the network has full freedom — it can keep the normalisation (small $\gamma$, zero $\beta$), amplify it (large $\gamma$), or ignore it entirely (set $\gamma$ and $\beta$ to the original scale).

**Initialisation:** $\gamma = 1$, $\beta = 0$ — start as pure normalisation, then adapt.

> [!note] Naming The paper uses $\gamma$ (scale) and $\beta$ (shift). [[Layer Normalization]] uses $g$ and $b$ for the same parameters. In Transformer papers you'll see both conventions. Functionally identical.

**The bias b is now redundant.** The preceding layer has $z = Wu + b$. But BN subtracts the mean of $Wu + b$ — and the mean of a constant $b$ is $b$ itself, so $b$ cancels out of the normalised output. The learned $\beta$ plays the role of the bias after normalisation. So you can (and should) drop the bias from the preceding linear layer when using BN: `z = g(BN(Wu))`.

---

### What BN is NOT

One thing to be careful about: BN does **not** process each example independently. The output $y_i = \text{BN}(x_i)$ depends on the entire batch ${x_1, \ldots, x_m}$ through $\mu_B$ and $\sigma^2_B$. Two identical inputs in different batches can get different normalised values if the rest of their batches differ.

This is both BN's strength (stable statistics from $m$ samples) and its weakness (fails when $m$ is small — see [[Layer Normalization#1. Why Batch Normalization Is Not Always Enough|LN §1]]).

---

## Backpropagation Through BN — From Scratch

This is the part most people skip but it's important to understand — it's what makes BN actually work (as opposed to the broken naive whitening from §2).

**The key difficulty:** $\mu_B$ and $\sigma^2_B$ are computed from _all $m$ inputs in the batch_. So when we change one input $x_i$, it affects not just its own output $\hat{x}_i$ but also $\mu_B$ and $\sigma^2_B$, which affect _every_ output $\hat{x}_j$ in the batch. The gradient is a multi-path problem.

### The Computation Graph

Lay out the dependencies explicitly:

```
x₁, x₂, …, xₘ  →  μ_B  →  x̂₁, x̂₂, …, x̂ₘ  →  y₁, y₂, …, yₘ  →  ℓ
                     ↑
x₁, x₂, …, xₘ  →  σ²_B  ┘
```

A single $x_i$ affects $\ell$ through **three paths**:

- **Path 1 (self):** $x_i \to \hat{x}_i \to y_i \to \ell$
- **Path 2 (via mean):** $x_i \to \mu_B \to$ all $\hat{x}_j \to$ all $y_j \to \ell$
- **Path 3 (via variance):** $x_i \to \sigma^2_B \to$ all $\hat{x}_j \to$ all $y_j \to \ell$

We need to sum contributions from all three paths when computing $\partial \ell / \partial x_i$.

We work backwards through the graph, computing each intermediate gradient before needing the next.

---

### Backward Step 1 — Gradient w.r.t. $\hat{x}_i$

We receive from the layer above: $\partial \ell / \partial y_i$ for all $i$.

The forward step was $y_i = \gamma \hat{x}_i + \beta$, which is linear in $\hat{x}_i$:

$$\frac{\partial y_i}{\partial \hat{x}_i} = \gamma$$

So by the chain rule:

$$\boxed{\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma}$$

**Intuition:** $\gamma$ is the multiplier on $\hat{x}_i$. If $\gamma$ is large, a small change in $\hat{x}_i$ has a large effect on $y_i$ and thus on $\ell$ — so the gradient flowing back is amplified by $\gamma$.

---

### Backward Step 2 — Gradient w.r.t. $\sigma^2_B$

$\sigma^2_B$ appears in the denominator of every $\hat{x}_j = (x_j - \mu_B)(\sigma^2_B + \varepsilon)^{-1/2}$. Changing $\sigma^2_B$ affects all $m$ outputs. We sum all $m$ contributions:

$$\frac{\partial \ell}{\partial \sigma^2_B} = \sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma^2_B}$$

Find the local piece $\partial \hat{x}_j / \partial \sigma^2_B$. Write $\hat{x}_j = (x_j - \mu_B) \cdot (\sigma^2_B + \varepsilon)^{-1/2}$. By the chain rule and the power rule $\frac{d}{dt}t^{-1/2} = -\frac{1}{2}t^{-3/2}$:

$$\frac{\partial \hat{x}_j}{\partial \sigma^2_B} = (x_j - \mu_B) \cdot \left(-\frac{1}{2}\right) \cdot (\sigma^2_B + \varepsilon)^{-3/2}$$

Plug in:

$$\boxed{\frac{\partial \ell}{\partial \sigma^2_B} = \left(-\frac{1}{2}\right)(\sigma^2_B + \varepsilon)^{-3/2} \sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j} \cdot (x_j - \mu_B)}$$

**Intuition:** $\sigma^2_B$ controls the zoom level of the normalisation. If $\sigma^2_B$ gets bigger, the denominator gets bigger, and all $\hat{x}_j$ shrink towards zero. The gradient with respect to $\sigma^2_B$ captures how much the loss cares about that zoom level — summed over all $m$ examples, weighted by how far each $x_j$ is from the mean.

---

### Backward Step 3 — Gradient w.r.t. $\mu_B$

$\mu_B$ appears in two places in the forward pass:

1. **Directly** in each numerator: $(x_j - \mu_B)$ in $\hat{x}_j$
2. **Indirectly** inside $\sigma^2_B = \frac{1}{m}\sum_j (x_j - \mu_B)^2$, which then feeds into all $\hat{x}_j$

By the multivariate chain rule, we must sum both contributions:

$$\frac{\partial \ell}{\partial \mu_B} = \underbrace{\sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \mu_B}}_{\text{Path 1: direct}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial \mu_B}}_{\text{Path 2: through variance}}$$

**Path 1:** $\hat{x}_j = (x_j - \mu_B)(\sigma^2_B + \varepsilon)^{-1/2}$, so treating $\sigma^2_B$ as fixed: $$\frac{\partial \hat{x}_j}{\partial \mu_B} = \frac{-1}{\sqrt{\sigma^2_B + \varepsilon}}$$

Path 1 $= \displaystyle\frac{-1}{\sqrt{\sigma^2_B + \varepsilon}} \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j}$

**Path 2:** $\sigma^2_B = \frac{1}{m}\sum_j (x_j - \mu_B)^2$, so by the chain rule: $$\frac{\partial \sigma^2_B}{\partial \mu_B} = \frac{1}{m} \sum_{j=1}^m 2(x_j - \mu_B) \cdot (-1) = \frac{-2}{m}\sum_{j=1}^m (x_j - \mu_B)$$

**But this equals zero!** Because $\sum_{j=1}^m (x_j - \mu_B) = \sum x_j - m\mu_B = m\mu_B - m\mu_B = 0$.

> [!note] Why Path 2 vanishes The sum of deviations from the mean is always zero — this is literally the definition of the mean. It means: if you shift the mean slightly while holding all $x_j$ fixed, the variance doesn't change to first order. The variance is locally flat with respect to the mean. So the gradient of $\ell$ w.r.t. $\mu_B$ _through the variance_ is zero — only the direct path matters.

Path 2 = 0, so:

$$\boxed{\frac{\partial \ell}{\partial \mu_B} = \frac{-1}{\sqrt{\sigma^2_B + \varepsilon}} \sum_{j=1}^{m} \frac{\partial \ell}{\partial \hat{x}_j}}$$

**Intuition:** The gradient of the loss w.r.t. the batch mean is just the sum of all the upstream gradients (from all $m$ examples), scaled by $-1/\hat{\sigma}$. The negative sign makes sense: if the mean increases, the normalised values all decrease (since we're subtracting the mean), so the gradient of any output w.r.t. the mean is negative.

---

### Backward Step 4 — Gradient w.r.t. Input $x_i$

Now the main event. $x_i$ affects the loss through three paths:

$$\frac{\partial \ell}{\partial x_i} = \underbrace{\frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i}}_{\text{Path 1: self}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial x_i}}_{\text{Path 2: via variance}} + \underbrace{\frac{\partial \ell}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i}}_{\text{Path 3: via mean}}$$

We need three local derivatives:

**Local derivative for Path 1** ($x_i$ appears in the numerator of $\hat{x}_i$): $$\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sqrt{\sigma^2_B + \varepsilon}}$$

**Local derivative for Path 2** ($x_i$ is inside $\sigma^2_B = \frac{1}{m}\sum_j (x_j - \mu_B)^2$ — it's one of the $m$ terms): $$\frac{\partial \sigma^2_B}{\partial x_i} = \frac{1}{m} \cdot 2(x_i - \mu_B) \cdot 1 = \frac{2(x_i - \mu_B)}{m}$$

**Local derivative for Path 3** ($x_i$ is one of $m$ terms in $\mu_B = \frac{1}{m}\sum_j x_j$): $$\frac{\partial \mu_B}{\partial x_i} = \frac{1}{m}$$

Plug everything in:

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2_B + \varepsilon}} + \frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial \ell}{\partial \mu_B} \cdot \frac{1}{m}$$

Now substitute the expressions for $\partial \ell / \partial \sigma^2_B$ and $\partial \ell / \partial \mu_B$ that we found in Steps 2 and 3. Let $\hat{\sigma} = \sqrt{\sigma^2_B + \varepsilon}$ for brevity. After substitution and simplification:

$$\boxed{\frac{\partial \ell}{\partial x_i} = \frac{1}{m\hat{\sigma}} \left[ m \cdot \frac{\partial \ell}{\partial \hat{x}_i} - \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j} - \hat{x}_i \cdot \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j} \hat{x}_j \right]}$$

**What the three terms mean — a beautiful symmetry with the forward pass:**

|Forward pass|Backward pass|
|---|---|
|Subtract mean $\mu_B$ from activations|Subtract mean of upstream gradients $\frac{1}{m}\sum \partial\ell/\partial\hat{x}_j$|
|Divide by std dev $\hat{\sigma}$|Remove component of gradient aligned with $\hat{x}_i$ (the $\hat{x}_i \cdot \sum(\cdot)\hat{x}_j$ term)|

The forward pass removes the mean and variance from the _data_. The backward pass removes the mean and a projection from the _gradient_. This is not a coincidence — it's a consequence of the chain rule through a normalisation operation.

> [!important] Why this solves the bias cancellation problem from §2 In the broken naive approach (§2), the normalisation happened _outside_ the gradient. So the optimiser computed $\partial \ell / \partial b$ and moved $b$, but the normalised output was unchanged. Here, the gradient _explicitly flows through_ $\mu_B$ and $\sigma^2_B$. The optimiser knows that when it changes $b$, it shifts $\mu_B$, which changes all the $\hat{x}_j$, which changes the loss. There's no cancellation — the gradient accounts for the whole chain. That's the entire point of making normalisation differentiable.

---

### Backward Step 5 — Gradients w.r.t. $\gamma$ and $\beta$

$\gamma$ and $\beta$ are shared parameters across all $m$ examples in the batch. Their gradients sum over the batch.

**For $\beta$ (shift):** $y_i = \gamma \hat{x}_i + \beta$, so $\partial y_i / \partial \beta = 1$ for all $i$:

$$\boxed{\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}}$$

This is just the total incoming gradient summed over the batch.

**For $\gamma$ (scale):** $\partial y_i / \partial \gamma = \hat{x}_i$:

$$\boxed{\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i}$$

This is a dot product between the upstream errors and the normalised data that $\gamma$ was scaling.

---

## Scale Invariance — Why BN Enables High Learning Rates

This is the most practically important property BN gives us.

### The Core Claim

For any scalar $a \neq 0$:

$$\text{BN}((aW)u) = \text{BN}(Wu)$$

**Proof:**

If we scale $W$ by $a$, then the pre-activation becomes $x' = (aW)u = a \cdot Wu = a \cdot x$.

Now compute the batch statistics for $x'$: $$\mu_{B'} = \frac{1}{m}\sum_i x'_i = a \cdot \frac{1}{m}\sum_i x_i = a\mu_B$$ $$\sigma^2_{B'} = \frac{1}{m}\sum_i (x'_i - \mu_{B'})^2 = \frac{1}{m}\sum_i (ax_i - a\mu_B)^2 = a^2 \cdot \frac{1}{m}\sum_i (x_i - \mu_B)^2 = a^2 \sigma^2_B$$

Now normalise $x'_i$: $$\hat{x}'_i = \frac{x'_i - \mu_{B'}}{\sqrt{\sigma^2_{B'} + \varepsilon}} \approx \frac{ax_i - a\mu_B}{\sqrt{a^2\sigma^2_B}} = \frac{a(x_i - \mu_B)}{|a|\sigma_B} = \pm\frac{x_i - \mu_B}{\sigma_B} = \pm\hat{x}_i$$

The scale $a$ cancels completely. $\text{BN}(aWu) = \text{BN}(Wu)$. $\square$

### What This Means for Gradients

Since the output is the same, the gradient w.r.t. the _input_ $u$ is also the same: $$\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(Wu)}{\partial u}$$

But the gradient w.r.t. $W$ itself _shrinks_ as $a$ grows: $$\frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(Wu)}{\partial W}$$

**This is the key.** Without BN: if the weights $W$ grow large (say, due to an aggressive learning rate update), the gradients through $W$ also grow large → instability → you need a smaller LR. With BN: if $W$ grows large by a factor $a$, the gradients through $W$ shrink by $1/a$. The system is **self-stabilising**. Large weights automatically produce small gradients, preventing the runaway that would otherwise force you to use tiny LRs.

This is why the paper can use 5×, 10×, even 30× larger learning rates with BN — the scale invariance acts as a natural safety net.

> [!note] Related: Adam's invariance [[Adam#3. Adam's Update Rule — Geometry and Intuition|Adam §3]] has a related property: Adam is invariant to rescaling the _gradient_ (e.g., multiplying the loss by a constant). Both properties reduce sensitivity to scale — Adam does it in gradient space, BN does it in weight space.

### Conjecture About Singular Values

The paper also makes a conjecture (not a proof) that BN encourages the Jacobian $J$ of each layer to have singular values near 1. The argument: after BN, both input $\hat{x}$ and output $\hat{z}$ of a layer have unit variance. If we approximate the layer as linear ($F(\hat{x}) \approx J\hat{x}$):

$$I = \text{Cov}[\hat{z}] \approx J \cdot \text{Cov}[\hat{x}] \cdot J^\top = J \cdot I \cdot J^\top = JJ^\top$$

So $JJ^\top \approx I$, meaning all singular values of $J$ are approximately 1. This would mean gradients neither explode nor vanish as they pass through the layer — preserved in magnitude.

> [!note] Contested later Santurkar et al. (2018) "How Does Batch Normalization Help Optimization?" challenged this explanation. Their experiments showed the ICS reduction BN provides doesn't directly correlate with the training speed benefit. Instead, they argue BN's real effect is **making the loss landscape smoother** — the gradient of the loss is more stable and predictable (more Lipschitz-continuous) with BN, making it easier for gradient descent to take larger steps confidently. This connects to [[Momentum#5. Condition Number and Optimal Step Size|the condition number κ]]: BN reduces the effective κ of the loss landscape, which is the actual reason gradient descent converges faster.

---

## Training vs. Inference — A Critical Distinction

> [!warning] Most common BN bug Forgetting to switch to inference mode (calling `model.eval()` in PyTorch, or the equivalent) means your network uses stochastic mini-batch statistics at inference time. Two identical inputs in different batches will get different normalised outputs. This bug is **silent** — no error, just wrong results.

### The Problem

At training time, BN uses $\mu_B$ and $\sigma^2_B$ computed from the current mini-batch. At inference time, this is problematic for two reasons:

1. You might only have one example ($m = 1$) → $\sigma^2_B = 0$ → division by zero
2. Even with $m > 1$, the output of your model shouldn't depend on which other unrelated examples happen to be in the same batch

We need the inference behaviour to be deterministic: same input → same output, every time.

### The Solution: Population Statistics

During training, in parallel with the forward pass, maintain running estimates of the true population mean and variance. The simplest approach: exponential moving averages.

After each mini-batch with statistics $(\mu_B, \sigma^2_B)$, update: $$\mu_\text{running} \leftarrow (1 - \alpha)\mu_\text{running} + \alpha \cdot \mu_B$$

where $\alpha$ is a small momentum (e.g., 0.1 in PyTorch's default). This is just a low-pass filter — it tracks the long-run average of $\mu_B$ across batches.

At the end of training, $\mu_\text{running} \approx \mathbb{E}[x]$ over the training set.

For the variance, there's a correction needed. The mini-batch used the biased estimator $\sigma^2_B = \frac{1}{m}\sum (x_i - \mu_B)^2$. To convert this to an unbiased estimate of the population variance, multiply by $m/(m-1)$ (Bessel's correction):

$$\text{Var}[x] = \frac{m}{m-1} \cdot \mathbb{E}_B[\sigma^2_B]$$

### The Inference Transform

At inference, fix $\mathbb{E}[x]$ and $\text{Var}[x]$ to the trained running averages and apply:

$$\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \varepsilon}}, \qquad y = \gamma \hat{x} + \beta$$

Since all four quantities $\mathbb{E}[x]$, $\text{Var}[x]$, $\gamma$, $\beta$ are constants at inference, this is an affine map:

$$y = \underbrace{\frac{\gamma}{\sqrt{\text{Var}[x]+\varepsilon}}}_\text{slope} \cdot x + \underbrace{\beta - \frac{\gamma \cdot \mathbb{E}[x]}{\sqrt{\text{Var}[x]+\varepsilon}}}_\text{intercept}$$

You can pre-compute this slope and intercept once after training and fuse them into the preceding linear layer's weight matrix and bias. **BN adds zero computational overhead at inference.**

> [!note] Contrast with LN [[Layer Normalization#9. Training vs. Inference — A Key Advantage Over BN|LN §9]] computes statistics from the example's own activations — so the inference behaviour is identical to training, no running averages needed, no mode switching. This is one of the main practical reasons LN won out in Transformers.

---

## Where to Put BN in the Network

### Before or After the Nonlinearity?

The paper's recommendation: **before** the nonlinearity.

```
z = g(BN(Wu))
```

instead of

```
z = BN(g(Wu))
```

**Why before?**

The pre-activation $Wu$ is approximately Gaussian by a CLT-like argument: it's a weighted sum of many input values, and sums of many random variables tend towards Gaussian by CLT. Normalising a Gaussian to zero mean, unit variance is well-defined and natural.

After a ReLU, the distribution is non-negative and non-Gaussian (half the values clipped to zero). Normalising that distribution is less principled.

**In practice:** Later work (He et al., ResNets 2016) sometimes places BN after ReLU. This is still debated empirically and there's no clean theoretical consensus. The paper's recommendation of pre-nonlinearity is the safer default.

### Dropping the Bias

When you use BN before the activation, the bias $b$ in the preceding linear layer `z = Wu + b` becomes redundant. Here's why: BN subtracts the batch mean of $Wu + b$ from each value. The mean of $Wu + b$ is $\mathbb{E}[Wu] + b$ — so $b$ is a constant shift that gets subtracted right away. It has no effect on the BN output. The learnable $\beta$ in BN takes over the role of bias. Drop $b$ from the linear layer to save parameters.

### For Convolutional Layers

In a CNN, a convolutional filter slides across the entire spatial extent of the image, producing a feature map. The same filter (same weights) is applied at every spatial position. It's therefore natural to normalise the same filter's outputs together.

**What BN does:** For each feature map (each filter), compute one $\mu_B$ and one $\sigma^2_B$ using **all spatial positions across all examples in the batch**.

For a batch of $m$ images with feature maps of size $p \times q$: $$m' = m \cdot p \cdot q \quad \text{(effective batch size for statistics)}$$

So for a batch of 32 images with $14 \times 14$ feature maps, you're computing mean and variance from $32 \times 196 = 6272$ values — one per pixel position per image in the batch. This gives very stable statistics.

There is one $\gamma$ and one $\beta$ **per feature map** (per filter), not per pixel position. The same learned scale and shift is applied regardless of where in the image the filter fires — consistent with the filter's translational equivariance.

---

## BN as a Regularizer

### The Mechanism

At training time, the output $\text{BN}(x_i)$ is not a deterministic function of $x_i$ alone — it depends on $\mu_B$ and $\sigma^2_B$, which are computed from the entire mini-batch ${x_1, \ldots, x_m}$. So the gradient that $x_i$ receives depends on the other $m-1$ examples in its batch.

Different mini-batches (different random subsets of the training data) → different $\mu_B$, different $\sigma^2_B$ → the normalised value of the same training example changes each epoch. This is stochastic noise injected into the training signal.

The network can't overfit to any single example's exact activation values, because those values change every time the example appears in a different batch context. This is the regularisation.

**It's the same mechanism as stochastic gradients** (see [[Momentum#15. Stochastic Gradients|Momentum §15]]): both are unbiased perturbations that help the optimiser find flatter minima, which generalise better. BN is essentially injecting structured noise that helps regularise.

### Batch Size Matters for Regularisation

- **Large batches** → $\mu_B \approx \mathbb{E}[x]$, $\sigma^2_B \approx \text{Var}[x]$ → statistics are stable → less noise → less regularisation
- **Small batches** → statistics are noisy → more regularisation, but also more training instability

This is one reason why very large batch training (using hundreds or thousands of GPUs) can hurt generalisation even when the loss converges fine — BN's regularisation effect disappears when the batch is large enough that statistics are no longer stochastic.

### This Is Why Dropout Can Be Removed

Dropout randomly zeros out activations during training, adding stochastic noise. BN is also adding stochastic noise (through the batch statistics). You don't need both — and indeed, BN networks in the paper are trained without Dropout.

**Important nuance:** They use less regularisation overall (lower weight decay too), not just less Dropout. BN changes the effective regularisation regime of the whole training setup.

---

## Experimental Results

### MNIST

Simple test: 3 fully-connected hidden layers, 100 sigmoid units each, batch size 60.

Key findings:

- BN network achieves higher accuracy in far fewer steps
- Most importantly: the distribution of sigmoid inputs was measured throughout training. **Without BN:** the distribution drifts and spreads throughout training, spending a lot of time in the saturated regime (|x| > 2). **With BN:** the distribution stays centered and stable throughout. This directly confirms the paper's motivation.

### ImageNet — The Main Result

Architecture: a modified Inception (GoogLeNet), trained on ILSVRC 2012.

|Model|Steps to 72.2%|Max accuracy|
|---|---|---|
|Inception baseline|31.0M|72.2%|
|BN-Baseline (just add BN)|13.3M|72.7%|
|BN-x5 (5× higher LR)|2.1M|73.0%|
|BN-x30 (30× higher LR)|2.7M|**74.8%**|
|BN-x5-Sigmoid|—|69.8%|

**The headline: BN-x5 matches the baseline accuracy with 14× fewer training steps.**

The most striking result is BN-x5-Sigmoid at 69.8%. Without BN, a sigmoid network on ImageNet stays at chance (≈0.1%) — completely untrainable. With BN, it achieves competitive performance. Sigmoid networks were not architecturally bad — they were suffering from internal covariate shift.

### The Full Recipe for Accelerating BN Networks

Simply swapping in BN and nothing else (BN-Baseline) helps but doesn't capture most of the benefit. To get the full speedup:

1. **Increase learning rate** — scale invariance makes this safe
2. **Remove Dropout** — BN provides regularisation
3. **Reduce L2 weight decay** (by 5×) — again, BN regularises
4. **Decay LR faster** — network trains faster, so compress the schedule (see [[Training Diagnostics and Optimizers#11. Annealing the Learning Rate|TDO §11]])
5. **Remove LRN (Local Response Normalisation)** — made redundant by BN
6. **Shuffle training data more** — strengthens BN's regularisation effect
7. **Reduce photometric distortions** — fewer training steps, so use them on realistic images

### Ensemble SOTA

6 BN-x30 networks with slight variations, ensembled:

- Top-5 test error: **4.82%** (on ILSVRC test server)
- Previous best: 4.94% (He et al. 2015)

---

## Common Misconceptions

**"BN removes correlations between features."** No. BN normalises each feature independently — it has no knowledge of other features. Removing correlations (whitening) would require computing the full $d \times d$ covariance matrix. BN explicitly avoids this.

**"After BN, activations always have zero mean and unit variance."** No. After the $\gamma, \beta$ step, the output $y_i = \gamma \hat{x}_i + \beta$ has mean $\beta$ and variance $\gamma^2$. Only the intermediate $\hat{x}_i$ has zero mean and unit variance. After the network learns $\gamma$ and $\beta$, outputs can be anything.

**"BN fixes vanishing gradients."** It substantially helps — especially with sigmoids — by keeping inputs away from the saturated regime. But in very deep networks, gradients can still vanish even with BN. Residual connections (ResNets) are the more complete solution.

**"Inference with BN is the same as training."** No — this is one of the most important operational distinctions. Training uses stochastic batch statistics; inference uses frozen population statistics. Forgetting `model.eval()` is a silent, hard-to-catch bug.

**"BN works at any batch size."** No. At batch size 1, $\sigma^2_B = 0$ and normalisation is undefined. This motivated [[Layer Normalization]], Group Norm, and Instance Norm — all designed for cases where BN breaks.

---

## Connections and Later Context

**[[Layer Normalization]] (Ba et al. 2016):** The direct successor, designed specifically for the cases where BN fails — RNNs, small batches, variable-length sequences. The key insight: rotate the normalisation axis from "across the batch, per feature" to "across features, per example." Same four-step algorithm, different axis. No train/inference distinction. Full notes: [[Layer Normalization]].

**Group Normalization (Wu & He 2018):** A middle ground between BN (normalise per feature across the full batch) and LN (normalise all features per example). Split features into $G$ groups, normalise within each group per example. Works well for small-batch detection/segmentation tasks (Mask R-CNN etc.) where BN's statistics are too noisy.

**Instance Normalization:** Extreme case of Group Norm with $G = d$ (one feature per group). Standard in style transfer. Normalises each channel of each image independently.

**The contested theory:** The paper claims BN's benefit comes from reducing internal covariate shift. Santurkar et al. (2018) showed experimentally that this may not be the primary mechanism — BN actually makes the loss landscape smoother (more Lipschitz gradients), which is what enables larger learning rates. The ICS reduction and the training acceleration may be somewhat independent effects. The practical benefits of BN are undisputed; the theoretical explanation is still debated.

**BN + Adam:** [[Adam]] handles non-uniform gradient scales _per parameter_. BN handles non-uniform input scales _per layer_. They attack the ill-conditioning problem from complementary angles — Adam in gradient space, BN in activation space. Using both together (standard in modern CNNs) is more effective than either alone.

**BN in ResNets:** BN is a key ingredient that makes very deep ResNets (50, 101, 152 layers) trainable. Without BN, the internal covariate shift compounds across 100+ layers and training collapses. BN keeps gradient magnitudes stable across depth, which is the main reason ResNets scaled so well.

---

## Quick Reference

### The Four Steps

|Step|Formula|What it does|
|---|---|---|
|Mean|$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$|Centre the batch|
|Variance|$\sigma^2_B = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$|Measure the spread|
|Normalise|$\hat{x}_i = (x_i - \mu_B)/\sqrt{\sigma^2_B + \varepsilon}$|Zero mean, unit variance|
|Re-parameterise|$y_i = \gamma \hat{x}_i + \beta$|Restore representational freedom|

### Backprop Gradients

|Gradient|Formula|
|---|---|
|$\partial\ell/\partial\hat{x}_i$|$(\partial\ell/\partial y_i) \cdot \gamma$|
|$\partial\ell/\partial\sigma^2_B$|$(-1/2)(\sigma^2_B+\varepsilon)^{-3/2} \sum_j (\partial\ell/\partial\hat{x}_j)(x_j - \mu_B)$|
|$\partial\ell/\partial\mu_B$|$(-1/\hat{\sigma})\sum_j \partial\ell/\partial\hat{x}_j$|
|$\partial\ell/\partial x_i$|$\frac{1}{m\hat{\sigma}}\left[m\frac{\partial\ell}{\partial\hat{x}_i} - \sum_j\frac{\partial\ell}{\partial\hat{x}_j} - \hat{x}_i\sum_j\frac{\partial\ell}{\partial\hat{x}_j}\hat{x}_j\right]$|
|$\partial\ell/\partial\gamma$|$\sum_i (\partial\ell/\partial y_i)\hat{x}_i$|
|$\partial\ell/\partial\beta$|$\sum_i \partial\ell/\partial y_i$|

### Key Properties

|Property|Statement|
|---|---|
|Scale invariance|$\text{BN}((aW)u) = \text{BN}(Wu)$ for all $a \neq 0$|
|Gradient scale invariance|$\partial\text{BN}((aW)u)/\partial(aW) = (1/a)\cdot\partial\text{BN}(Wu)/\partial W$|
|Conv effective batch size|$m' = m \cdot p \cdot q$ (batch × height × width per feature map)|
|Inference variance correction|$\text{Var}[x] = \frac{m}{m-1}\mathbb{E}_B[\sigma^2_B]$|
|Inference fused form|$y = \frac{\gamma}{\sqrt{\text{Var}[x]+\varepsilon}} \cdot x + \left(\beta - \frac{\gamma\mathbb{E}[x]}{\sqrt{\text{Var}[x]+\varepsilon}}\right)$|

---

_Notes from Ioffe & Szegedy (2015), arXiv:1502.03167 — Musaib_