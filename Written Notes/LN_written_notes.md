# Layer Normalization — Written Notes

> **Paper:** Ba, Kiros & Hinton, 2016 (arXiv:1607.06450) **Prerequisite:** [[BN_written_notes]] — these notes build directly on BN. Every section assumes you know the BN transform, its backprop, and why it was invented. **Links:** [[Layer Normalization]] (full reference version) · [[Batch Normalization]] · [[Momentum]] · [[Adam]] · [[Training Diagnostics and Optimizers]]

---

## Why We're Here — BN Has Three Fatal Failure Modes

We just spent all that time understanding Batch Normalization. BN works by computing a mean $\mu_B$ and variance $\sigma^2_B$ **across the batch dimension** — you pool $m$ examples together, compute statistics, and normalise each value relative to its batch. That batch-coupling is what gives BN its power: stable statistics from $m$ samples.

But that same batch-coupling is also BN's achilles heel. There are three architectures where it simply breaks.

---

### Failure Mode 1 — Recurrent Neural Networks

In an RNN, the network processes a sequence step by step. At each time step $t$, the hidden state is: $$\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b})$$

BN would need to normalise $\mathbf{h}_t$ across the batch. But the batch at time step $t$ has a critical problem: **different sequences in the batch have different lengths**. A batch of sentences — one 5 words long, another 20 words long — means that at $t = 10$, the short sentence has already ended. The "batch" at step $t$ is ragged and incomplete. You can't meaningfully average across it.

There's a second problem even if you ignore the ragged batch: the distribution of hidden states changes dramatically as $t$ increases. The statistics appropriate for $t=1$ (the hidden state just after seeing the first word) are completely wrong for $t=50$ (after seeing 50 words — the hidden state has integrated a lot of context). BN would need a separate $\gamma^{(t)}, \beta^{(t)}$ for every single time step. Worse, recall from [[BN_written_notes#Training vs. Inference|BN inference]] — at test time you'd need frozen population statistics for every time step, meaning your model literally cannot handle sequences longer than those seen during training.

---

### Failure Mode 2 — Batch Size 1

From [[BN_written_notes#The BN Transform|BN Step 2]], the batch variance is: $$\sigma^2_B = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

At $m=1$: there is only one value, $\mu_B = x_1$, so $(x_1 - \mu_B)^2 = 0$. Division by $\sigma_B = 0$. BN is undefined.

This kills BN for:

- **Reinforcement learning** — you typically train on one environment transition at a time
- **Large model training** — when a model is so big that you can only fit one example per GPU
- **Online learning** — updating on one example as it arrives

---

### Failure Mode 3 — Variable-Length Sequences (NLP)

When you have sentences of different lengths and you pad the shorter ones to the same length, BN computes batch statistics that include padding positions. The "feature" at position 10 in a padded sentence is meaningless noise — but BN mixes it into the statistics for all other examples. The statistics are corrupted.

---

### The Fix — One Sentence

> Instead of normalising across the batch (over different examples), normalise within a single example (over its features).

That's it. Same four steps as BN — mean, variance, normalise, scale+shift — but the set of values you compute statistics over changes completely. This is Layer Normalization.

---

## The Axis Switch — The Single Most Important Diagram

Write out a mini-batch as a matrix $\mathbf{X} \in \mathbb{R}^{m \times H}$ — $m$ examples (rows), $H$ features (columns):

```
              Feature dimension  →  H features
              ┌──────────────────────────────────┐
Example 1  →  │  x₁₁   x₁₂   x₁₃   x₁₄   ...   │
Example 2  →  │  x₂₁   x₂₂   x₂₃   x₂₄   ...   │
Example 3  →  │  x₃₁   x₃₂   x₃₃   x₃₄   ...   │
Example 4  →  │  x₄₁   x₄₂   x₄₃   x₄₄   ...   │
              └──────────────────────────────────┘
                  ↑      ↑      ↑      ↑
              BN computes statistics down each COLUMN
              (same feature, different examples)

              ← LN computes statistics across each ROW →
              (same example, all its features)
```

- **BN:** for each column $k$, pool the $m$ values and compute $(\mu_B^{(k)}, \sigma_B^{2(k)})$. One set of statistics per column.
- **LN:** for each row $i$, pool the $H$ values and compute $(\mu^{(i)}, \sigma^{2(i)})$. One set of statistics per row.

**This diagram is the entire conceptual content of the paper.** Everything else — the maths, the properties, the applications — follows from this one choice. Print it, memorise it.

Notice immediately why LN fixes all three failures:

- **RNNs:** statistics are per-example, so ragged batches don't matter — you just compute the statistics for each sequence independently
- **Batch size 1:** you're averaging over $H$ features, not $m$ examples — as long as $H$ is large (and it always is: 512, 768, 1024, 4096), statistics are stable even at $m=1$
- **Variable-length sequences:** each token position normalises itself independently — padding in other sequences is completely irrelevant

---

## The LN Transform — Four Steps (Same as BN, Different Axis)

LN operates on the full activation vector $\mathbf{a} = (a_1, a_2, \ldots, a_H) \in \mathbb{R}^H$ of **one example** at **one layer**. This is the $H$-dimensional vector of pre-activations at that layer for that single example.

Notation: $H$ = number of hidden units in the layer. $g_k$ = learned gain, $b_k$ = learned bias.

---

### Step 1 — Layer Mean

$$\mu = \frac{1}{H}\sum_{k=1}^{H} a_k$$

This is the arithmetic mean of all $H$ activations of this one example at this layer.

**Compare to BN:** In [[BN_written_notes#Step 1 — Compute the Mini-Batch Mean|BN Step 1]], $\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$ — same formula but summing over $m$ examples for one feature. Here we sum over $H$ features for one example.

**What it captures:** $\mu$ is the overall "excitation level" of this layer for this particular input. If the layer is generally very active (large values), $\mu$ is large. Subtracting $\mu$ removes this global offset — the network learns, through $b_k$, whatever offset it actually needs.

**Verification:** $\frac{1}{H}\sum_k (a_k - \mu) = \frac{1}{H}\sum_k a_k - \mu = \mu - \mu = 0$. After centering, the mean is zero. $\checkmark$

---

### Step 2 — Layer Variance

$$\sigma^2 = \frac{1}{H}\sum_{k=1}^{H} (a_k - \mu)^2$$

The average squared deviation from $\mu$, across all $H$ activations of this example.

**Why biased (divide by $H$, not $H-1$)?** Same reason as BN — we're normalising for optimisation, not doing statistics. We don't need an unbiased population estimate; we just need a scale. For large $H$ (e.g., $H = 1024$), the difference is less than 0.1% — completely negligible.

**Contrast with BN at inference:** Remember from [[BN_written_notes#Training vs. Inference|BN §Training vs. Inference]] that BN uses the biased estimator $\sigma^2_B = \frac{1}{m}\sum(x_i-\mu_B)^2$ during training, but then corrects it by $\frac{m}{m-1}$ (Bessel's correction) to get an unbiased population variance at inference. **LN never needs this correction** — since statistics are per-example and recomputed fresh every forward pass, there's no distinction between training-time and inference-time statistics. The same formula runs both times.

**Equivalent formula:** $\sigma^2 = \frac{1}{H}\sum_k a_k^2 - \mu^2$ (mean of squares minus square of mean). Useful as a sanity check — it equals zero iff all $a_k$ are identical.

---

### Step 3 — Normalise

$$\hat{a}_k = \frac{a_k - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

Identical in form to BN's normalisation. Subtract the mean (centres to zero), divide by the standard deviation (scales to unit variance), add $\varepsilon$ (prevents divide-by-zero when all activations are identical).

**Proof of unit variance:** $$\frac{1}{H}\sum_{k=1}^H \hat{a}_k^2 = \frac{1}{H}\sum_{k=1}^H \frac{(a_k - \mu)^2}{\sigma^2 + \varepsilon} \approx \frac{\sigma^2}{\sigma^2} = 1 \quad \checkmark \text{ (ignoring } \varepsilon\text{)}$$

**What $\hat{a}_k$ is:** The $z$-score of $a_k$ within this layer's activations for this example. It measures "how many standard deviations is this activation away from the layer's mean?" Every layer, for every example, gets mapped to the same canonical distribution.

**Geometric picture:** After normalisation, the vector $\hat{\mathbf{a}}$ lives on an $(H-1)$-dimensional hypersphere — it has been projected out of the direction of the all-ones vector (removing the mean) and then scaled to lie on the surface of a sphere of radius $\sqrt{H}$ (unit variance). No matter what the raw activations $\mathbf{a}$ were, $\hat{\mathbf{a}}$ always ends up on this same surface. Only the _direction_ (relative pattern of activations) is preserved; the _magnitude_ and _offset_ are discarded and replaced by the learned $g$ and $b$.

---

### Step 4 — Scale and Shift

$$h_k = g_k \cdot \hat{a}_k + b_k$$

Learned parameters $g_k$ and $b_k$ per feature, initialised to $g=1, b=0$.

**Why these are essential — same argument as BN, but worth restating in the LN context:**

Pure normalisation forces every layer's output to have zero mean and unit variance. But different layers need different operating regimes. Consider a Transformer's feed-forward sublayer feeding into a GELU activation — GELU's nonlinear behaviour is most expressive in a particular range of values. If LN always squashes inputs to unit variance, the network has no freedom to choose that range. With $g_k$, the network can amplify the normalised activations to whatever scale works best for the downstream operation.

More concretely: if the network sets $g_k = \hat{\sigma}_k$ and $b_k = \hat{\mu}_k$ (the original scale and mean), it exactly inverts the normalisation — LN becomes a no-op. So **the network can always choose to ignore LN** if it's unhelpful for a particular layer. The existence of $g$ and $b$ makes LN strictly more expressive than pure normalisation.

**Naming:** This paper uses $g$ (gain) and $b$ (bias). [[BN_written_notes]] uses $\gamma$ and $\beta$ for the same parameters. Modern Transformer code often uses `weight` and `bias`. They are all the same thing — learned scale and shift applied after normalisation.

**Complete forward pass in one line:** $$\mu = \frac{1}{H}\sum_k a_k, \quad \sigma^2 = \frac{1}{H}\sum_k(a_k-\mu)^2, \quad \hat{a}_k = \frac{a_k-\mu}{\sqrt{\sigma^2+\varepsilon}}, \quad h_k = g_k\hat{a}_k + b_k$$

Applied **independently** to each example. The parameters $g$ and $b$ are **shared** across all examples and (in Transformers) across all sequence positions.

---

## Backpropagation Through LN

If you followed the BN backprop in [[BN_written_notes#Backpropagation Through BN|BN written notes]], this will feel like déjà vu — and that's intentional. The derivation is structurally identical. The only change is: **$m$ (number of examples in the batch) becomes $H$ (number of features in the layer), and "summing over examples" becomes "summing over features".**

I'll do it properly so the structure is fully clear.

### The Computation Graph (same shape as BN)

```
a₁, a₂, …, aₕ  →  μ  →  â₁, â₂, …, âₕ  →  h₁, h₂, …, hₕ  →  ℓ
                    ↑
a₁, a₂, …, aₕ  →  σ² ┘
```

A single $a_i$ affects the loss through three paths:

- **Path 1 (self):** $a_i \to \hat{a}_i \to h_i \to \ell$
- **Path 2 (via mean):** $a_i \to \mu \to$ all $\hat{a}_j \to$ all $h_j \to \ell$
- **Path 3 (via variance):** $a_i \to \sigma^2 \to$ all $\hat{a}_j \to$ all $h_j \to \ell$

This is identical to BN's three-path structure. In BN, a single $x_i$ (one example's value) affected $\mu_B$ and $\sigma^2_B$, which then affected all $m$ normalised outputs. Here, a single $a_i$ (one feature of one example) affects $\mu$ and $\sigma^2$, which affect all $H$ normalised outputs.

We receive from above: $\partial\ell / \partial h_k$ for all $k = 1, \ldots, H$.

---

### Backward Step 1 — Gradient w.r.t. $\hat{a}_k$

$h_k = g_k \hat{a}_k + b_k$ is linear in $\hat{a}_k$:

$$\boxed{\frac{\partial \ell}{\partial \hat{a}_k} = \frac{\partial \ell}{\partial h_k} \cdot g_k}$$

The gain $g_k$ amplifies how much the upstream loss cares about changes in $\hat{a}_k$. Identical to the BN result except $\gamma \to g_k$ and $\hat{x}_i \to \hat{a}_k$.

---

### Backward Step 2 — Gradient w.r.t. $\sigma^2$

$\sigma^2$ is in the denominator of every $\hat{a}_j = (a_j - \mu)(\sigma^2 + \varepsilon)^{-1/2}$. It affects all $H$ outputs — sum over all $H$:

$$\frac{\partial \ell}{\partial \sigma^2} = \sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j} \cdot \frac{\partial \hat{a}_j}{\partial \sigma^2}$$

Local piece — differentiate $(a_j-\mu)(\sigma^2+\varepsilon)^{-1/2}$ with respect to $\sigma^2$:

$$\frac{\partial \hat{a}_j}{\partial \sigma^2} = (a_j - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2+\varepsilon)^{-3/2}$$

Plugging in:

$$\boxed{\frac{\partial \ell}{\partial \sigma^2} = \left(-\frac{1}{2}\right)(\sigma^2+\varepsilon)^{-3/2} \sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j} \cdot (a_j - \mu)}$$

Same formula as BN — just $H$ in the sum instead of $m$, and $\sigma^2$ is the layer variance instead of batch variance.

---

### Backward Step 3 — Gradient w.r.t. $\mu$

$\mu$ appears in two places — directly in each numerator $(a_j - \mu)$, and inside $\sigma^2 = \frac{1}{H}\sum_j(a_j - \mu)^2$. Two paths:

$$\frac{\partial \ell}{\partial \mu} = \underbrace{\sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j} \cdot \frac{-1}{\sqrt{\sigma^2+\varepsilon}}}_{\text{Path 1: direct}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu}}_{\text{Path 2: via variance}}$$

Path 2 is zero. Proof: $$\frac{\partial \sigma^2}{\partial \mu} = \frac{-2}{H}\sum_{j=1}^H (a_j - \mu)$$

and $\sum_{j=1}^H (a_j - \mu) = \sum_j a_j - H\mu = H\mu - H\mu = 0$.

> [!note] Same cancellation as in BN In [[BN_written_notes#Backward Step 3 — Gradient w.r.t. μ_B|BN backward step 3]], Path 2 was zero for the exact same reason — the sum of deviations from the mean is always zero. This is a universal property of the arithmetic mean, so it cancels in both BN and LN regardless of what values you're summing over (examples in BN, features in LN).

Therefore: $$\boxed{\frac{\partial \ell}{\partial \mu} = \frac{-1}{\sqrt{\sigma^2+\varepsilon}} \sum_{j=1}^H \frac{\partial \ell}{\partial \hat{a}_j}}$$

---

### Backward Step 4 — Gradient w.r.t. Input $a_i$

Three paths, three local derivatives:

$$\frac{\partial \ell}{\partial a_i} = \underbrace{\frac{\partial \ell}{\partial \hat{a}_i} \cdot \frac{1}{\sqrt{\sigma^2+\varepsilon}}}_{\text{Path 1: self}} + \underbrace{\frac{\partial \ell}{\partial \sigma^2} \cdot \frac{2(a_i-\mu)}{H}}_{\text{Path 2: via }\sigma^2} + \underbrace{\frac{\partial \ell}{\partial \mu} \cdot \frac{1}{H}}_{\text{Path 3: via }\mu}$$

Local derivatives (all immediate from the definitions):

- $\partial \hat{a}_i / \partial a_i = 1/\sqrt{\sigma^2+\varepsilon}$ ($a_i$ is in the numerator of $\hat{a}_i$)
- $\partial \sigma^2 / \partial a_i = 2(a_i - \mu)/H$ ($a_i$ is one of $H$ terms in the variance sum)
- $\partial \mu / \partial a_i = 1/H$ ($a_i$ is one of $H$ terms in the mean)

Substitute $\partial\ell/\partial\sigma^2$ (from Step 2) and $\partial\ell/\partial\mu$ (from Step 3), let $\hat{\sigma} = \sqrt{\sigma^2+\varepsilon}$, and simplify:

$$\boxed{\frac{\partial \ell}{\partial a_i} = \frac{1}{H\hat{\sigma}} \left[ H\frac{\partial \ell}{\partial \hat{a}_i} - \sum_{k=1}^H \frac{\partial \ell}{\partial \hat{a}_k} - \hat{a}_i \sum_{k=1}^H \frac{\partial \ell}{\partial \hat{a}_k}\hat{a}_k \right]}$$

**Comparing with the BN result** from [[BN_written_notes#Backward Step 4 — Gradient w.r.t. Input xᵢ|BN backward step 4]]: $$\frac{\partial \ell}{\partial x_i}\bigg|_\text{BN} = \frac{1}{m\hat{\sigma}_B}\left[ m\frac{\partial \ell}{\partial \hat{x}_i} - \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j} - \hat{x}_i\sum_{j=1}^m\frac{\partial \ell}{\partial \hat{x}_j}\hat{x}_j \right]$$

They are literally the same formula. Replace $m \to H$, $\hat{x} \to \hat{a}$, $\hat{\sigma}_B \to \hat{\sigma}$. **If you can do BN backprop, you can do LN backprop.** The only structural difference is what you sum over.

**The three-term interpretation** carries over unchanged:

|Term|What it does|
|---|---|
|$H \cdot \partial\ell/\partial\hat{a}_i$|Raw upstream gradient, scaled up by $H$|
|$-\sum_k \partial\ell/\partial\hat{a}_k$|Subtracts the sum of all upstream gradients (mean-centres the gradient)|
|$-\hat{a}_i \sum_k (\partial\ell/\partial\hat{a}_k)\hat{a}_k$|Removes the component of the gradient aligned with $\hat{a}_i$ (variance-normalises the gradient)|

The forward pass removes mean and variance from the _data_. The backward pass removes mean and a projection from the _gradient_. The normalisation in the forward direction induces a dual de-normalisation in the backward direction.

---

### Backward Step 5 — Gradients w.r.t. $g_k$ and $b_k$

$g_k$ and $b_k$ are shared across all $m$ examples in the batch, so their gradients sum over the batch:

$$\boxed{\frac{\partial \ell}{\partial b_k} = \sum_{i=1}^m \frac{\partial \ell}{\partial h_k^{(i)}}} \qquad \boxed{\frac{\partial \ell}{\partial g_k} = \sum_{i=1}^m \frac{\partial \ell}{\partial h_k^{(i)}} \cdot \hat{a}_k^{(i)}}$$

Note the superscript $(i)$ — these sums are over examples in the batch, not over features. This is because $g_k$ and $b_k$ are parameters (not per-example statistics), and their gradients accumulate signal from every example that used them.

---

## No Train/Inference Distinction — The Big Practical Win

This is arguably the single most important practical difference between LN and BN.

Recall from [[BN_written_notes#Training vs. Inference|BN Training vs. Inference]]: BN's normalisation at training time uses stochastic mini-batch statistics $\mu_B$ and $\sigma^2_B$. At inference, you can't use these — they depend on which batch you happen to have. So BN maintains running exponential moving averages of $\mu$ and $\sigma^2$ throughout training, and swaps to those frozen estimates at inference. This requires a mode switch (`model.eval()` in PyTorch), and forgetting it is one of the most common silent bugs in deep learning.

**LN has none of this.** LN's statistics $\mu$ and $\sigma^2$ are computed from ${a_1, \ldots, a_H}$ — the current example's own activations. This computation is identical regardless of whether you are training or serving. Whether you're processing one example or a million, LN runs exactly the same formula. There are no running averages, no population statistics, no mode flags.

**Practical consequence:** You can deploy an LN-based model as a pure function — same input always produces the same output, no hidden state required. This also means LN is immune to the train/test distribution shift problem that BN faces: if the test data has a different distribution from training, BN's frozen population statistics are mismatched. LN is unaffected because it never relied on training-time population statistics in the first place.

---

## LN in RNNs — The Original Application

The paper was primarily motivated by RNNs, specifically LSTMs. Recall that an LSTM at time step $t$ computes:

$$\mathbf{a}_t = W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t$$

and then applies a set of gates (input gate $\mathbf{i}$, forget gate $\mathbf{f}$, output gate $\mathbf{o}$, cell gate $\mathbf{g}$) each through a sigmoid or tanh:

$$[\mathbf{i}_t, \mathbf{f}_t, \mathbf{o}_t, \mathbf{g}_t] = [\sigma, \sigma, \sigma, \tanh](https://claude.ai/chat/%5Cmathbf%7Ba%7D_t)$$

With LN, you insert normalisation on the pre-activations before the gates:

$$[\mathbf{i}_t, \mathbf{f}_t, \mathbf{o}_t, \mathbf{g}_t] = [\sigma, \sigma, \sigma, \tanh]$$

The LN at time step $t$ computes fresh $\mu_t = \frac{1}{H}\sum_k a_{t,k}$ and $\sigma^2_t = \frac{1}{H}\sum_k (a_{t,k} - \mu_t)^2$ from the $H$-dimensional vector $\mathbf{a}_t$. Every time step gets its own statistics — they don't pool across the batch, they don't share across time steps.

**The shared parameters:** Even though the statistics are recomputed at each time step, the learned gain $\mathbf{g}$ and bias $\mathbf{b}$ are **shared across all time steps**. This is analogous to how the LSTM's weights $W_{hh}$ and $W_{xh}$ are shared across time — the same transformation is applied at every step, but with different statistics. This is weight tying in the normalisation.

**Why this helps:** Without any normalisation, the hidden state $\mathbf{h}_t$ can grow or shrink exponentially over long sequences as $W_{hh}$ compounds. LN keeps the magnitude of $\mathbf{a}_t$ controlled at every step — after LN, $\mathbf{a}_t$ always has zero mean and unit variance before hitting the gates. This doesn't fully solve the vanishing gradient problem (that's what LSTM gating is for), but it prevents the hidden state distribution from drifting to extreme values, stabilising training significantly.

---

## LN in Transformers — Where It Became Universal

The Transformer (Vaswani et al., 2017) uses attention and feed-forward sublayers. A standard Transformer block has the structure:

```
Input → [Sublayer 1: Multi-Head Attention] → [Sublayer 2: Feed-Forward Network] → Output
```

Each sublayer has a residual connection: the output is $x + \text{SubLayer}(x)$. LN is applied somewhere around each sublayer. There are two choices for where.

### Post-LN (Original Design)

$$\mathbf{x}_{l+1} = \text{LN}\left(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l)\right)$$

LN is applied _after_ the residual addition. The gradient from layer $l+1$ to layer $l$ must pass through the LN operation. In a 24-layer Transformer, the gradient from the output must pass through 24 LN operations on its way to the first layer.

At initialisation, LN's parameters are $g=1, b=0$ — it's approximately the identity map. But as training proceeds and $g$ moves away from 1, LN begins distorting the gradient. In very deep networks (> ~12 layers), this compounding distortion makes Post-LN hard to train — it typically requires a learning rate warmup schedule to stabilise the early phase of training.

### Pre-LN (Modern Standard)

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}\left(\text{LN}(\mathbf{x}_l)\right)$$

LN is applied _before_ the sublayer, _inside_ the residual branch. The residual path $\mathbf{x}_l \to \mathbf{x}_{l+1}$ is just an addition — there is **no LN in the residual highway**.

The gradient from layer $l+1$ to layer $l$ (via the residual) is just $1$ — no multiplicative distortion whatsoever. LN only sits in the sublayer branch, which is an additive correction. This means a gradient originating at the output can propagate all the way back to the first layer through the residual connections without ever passing through a single LN operation.

> [!note] Connection to ResNet intuition This is the same reason ResNets (He et al. 2016) train successfully at 50, 100, 150 layers — the skip connections provide a clean gradient highway. Pre-LN in Transformers is the same principle applied to normalisation: keep the direct gradient path clean, let the normalisation live in the branches. From [[Momentum#8. Four Convergence Regimes|Momentum §8]], a direct gradient path is essential for the convergence regime not to collapse — Post-LN effectively adds a filter on the gradient highway at every layer, which in deep networks accumulates to near-zero signal at the input layers.

**Why Pre-LN is preferred in practice:**

|Property|Post-LN|Pre-LN|
|---|---|---|
|Training stability|Needs LR warmup|More stable out of the box|
|Gradient flow at init|Distorted through 24 LN ops|Clean residual highway|
|Final performance|Slightly higher ceiling|Slightly lower ceiling|
|Used in|Original Transformer|GPT-2/3/4, LLaMA, Mistral, PaLM|

The tradeoff: Post-LN _can_ achieve slightly better final accuracy if you tune carefully. But Pre-LN is far easier to train and scales better to very deep models. Modern practice overwhelmingly uses Pre-LN.

**One LN per sublayer.** A full Transformer block in Pre-LN looks like this:

```
x_l
 │
 ├──→  LN₁  →  [Multi-Head Attention]  →  (+)  →  x_l'
 │                                          ↑
 └──────────────────────────────────────────┘  (residual)

x_l'
 │
 ├──→  LN₂  →  [Feed-Forward Network]  →  (+)  →  x_{l+1}
 │                                          ↑
 └──────────────────────────────────────────┘  (residual)
```

Two separate LN operations per block — each normalises the $d_\text{model}$-dimensional embedding vector for each token independently before passing it into the sublayer.

---

## What LN Actually Does to Activation Space

Think geometrically. Your activation vector $\mathbf{a} \in \mathbb{R}^H$ is a point in $H$-dimensional space. LN does two things to it:

**1. Centering ($\mathbf{a} - \mu \mathbf{1}$):** Projects $\mathbf{a}$ onto the hyperplane that is orthogonal to the all-ones vector $\mathbf{1} = (1,1,\ldots,1)$. This removes the "DC component" — the global offset. Any two vectors that differ only by a constant shift (i.e., $\mathbf{a}$ and $\mathbf{a} + c\mathbf{1}$) get mapped to the same centered vector.

**2. Scaling ($\div \hat{\sigma}$):** Scales the centered vector to have length $\sqrt{H}$. All vectors of different magnitudes get mapped to the same sphere.

The result $\hat{\mathbf{a}}$ satisfies:

- $\sum_k \hat{a}_k = 0$ — lies on the hyperplane orthogonal to $\mathbf{1}$
- $\frac{1}{H}\sum_k \hat{a}_k^2 = 1$ — lies on a sphere of radius $\sqrt{H}$

So $\hat{\mathbf{a}}$ lies on the intersection of these two: an $(H-1)$-dimensional sphere. Every activation vector gets mapped to this surface. **The only information preserved is the direction (pattern) of the activations** — which features are relatively large or small compared to each other. The absolute scale and global offset are completely discarded, and the learned $g$ and $b$ then add back whatever scale and offset the network decides it needs.

This is why LN eliminates internal covariate shift _within_ each layer: whatever the preceding layers do to the magnitude or offset of the activations, LN always maps the result to the same canonical sphere. The downstream layer always receives activations in a consistent regime.

---

## The Normalization Family — Everything in One Picture

All normalization methods are the same four-step algorithm. They differ only in _which cells of the data matrix_ they compute statistics over.

```
                Feature dimension  →  H features
              ┌──────────────────────────────────┐
Example 1  →  │  ██████████████████████████████  │  ← LN: whole row
Example 2  →  │  ██████   ██████   ██████         │  ← Group Norm: groups of columns within a row
Example 3  →  │  █         █         █            │  ← Instance Norm: one cell per group
              └──────────────────────────────────┘
                  ↑          ↑          ↑
                 BN: whole column
```

**BN:** statistics from column $k$ across all $m$ rows (one $\mu, \sigma^2$ per feature across all examples in batch).

**LN:** statistics from row $i$ across all $H$ columns (one $\mu, \sigma^2$ per example across all features).

**Group Norm (Wu & He, 2018):** statistics from a _subset_ of $H/G$ columns within row $i$ (one $\mu, \sigma^2$ per group per example). Sits between LN and Instance Norm.

- $G=1$: Group Norm = LN (one group = all features)
- $G=H$: Group Norm = Instance Norm (each feature is its own group)

**Instance Norm:** each feature of each example normalised independently — one value, one statistic. Used in style transfer.

### RMSNorm — The LN Variant Used in Modern LLMs

Zhang & Sennrich (2019) asked: do we actually need the mean subtraction step?

In LN:

1. Compute mean $\mu$
2. Compute variance $\sigma^2$
3. Normalise: $(a_k - \mu)/\sqrt{\sigma^2+\varepsilon}$
4. Scale and shift: $g_k \hat{a}_k + b_k$

RMSNorm drops steps 1 and 2 entirely and replaces them with just the RMS:

$$\text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{H}\sum_{k=1}^H a_k^2}$$

$$\bar{a}_k = \frac{a_k}{\text{RMS}(\mathbf{a}) + \varepsilon}, \qquad h_k = g_k \bar{a}_k$$

Note: no mean subtraction, no bias $b$. Just scale-normalisation, then learned gain.

**Why this works nearly as well as LN:** The critical operation in LN is re-scaling — controlling the signal magnitude to prevent explosion or vanishing, and to keep the activations in a consistent regime. The mean subtraction (re-centering) is less critical: it removes a DC offset that the network could alternatively handle through the learned bias $b_k$. RMSNorm keeps the essential operation (re-scaling by the RMS) and drops the rest.

**Where it's used:** LLaMA, Mistral, Gemma, and most state-of-the-art open-source LLMs as of 2024–2025. At $H = 4096$ (or larger), the saved computation of skipping the mean subtraction adds up meaningfully across billions of tokens. RMSNorm is essentially the de facto standard in modern LLMs.

---

## What LN Does NOT Do

These are common misconceptions, and it's worth being explicit:

**LN does not decorrelate features.** LN normalises the marginal statistics (mean and variance) of the activation vector, but it knows nothing about the covariance between different features. Two features that are highly correlated will remain highly correlated after LN. Full decorrelation (whitening) would require the $H \times H$ covariance matrix — $O(H^2)$ entries for $H = 4096$ means 67 million entries. Same reason BN doesn't decorrelate either, as noted in [[BN_written_notes#Common Misconceptions|BN misconceptions]].

**LN does not guarantee zero mean and unit variance at the output.** After the learned $g_k$ and $b_k$ are applied, the output $h_k = g_k\hat{a}_k + b_k$ has mean $b_k$ and variance $g_k^2$. Only the intermediate $\hat{a}_k$ has zero mean and unit variance.

**LN does not eliminate vanishing gradients.** It helps by keeping activation magnitudes stable, but in sufficiently deep networks or networks with poor architectures, gradients can still vanish. Residual connections are the primary fix for vanishing gradients; LN is a complement, not a substitute.

**LN does not regularise like BN.** This is an important practical difference. Recall from [[BN_written_notes#BN as a Regularizer|BN as Regularizer]]: BN's regularisation comes from the stochasticity of batch statistics — different batches give different normalisations for the same training example, which prevents overfitting to any single example's exact values. LN has no such stochasticity — for a fixed input $\mathbf{a}$, LN always produces the same output $\hat{\mathbf{a}}$ regardless of what batch it's in. LN is completely deterministic per example. This means when you use LN you should keep Dropout (or use other regularisers) — you can't remove it the way you can with BN.

---

## Quick Reference

### The Four Steps

|Step|Formula|What it does|
|---|---|---|
|Mean|$\mu = \frac{1}{H}\sum_{k=1}^H a_k$|Centre the example's activations|
|Variance|$\sigma^2 = \frac{1}{H}\sum_{k=1}^H (a_k - \mu)^2$|Measure the spread|
|Normalise|$\hat{a}_k = (a_k - \mu)/\sqrt{\sigma^2 + \varepsilon}$|Zero mean, unit variance|
|Re-parameterise|$h_k = g_k\hat{a}_k + b_k$|Restore representational freedom|

### Backprop Gradients

|Gradient|Formula|
|---|---|
|$\partial\ell/\partial\hat{a}_k$|$(\partial\ell/\partial h_k) \cdot g_k$|
|$\partial\ell/\partial\sigma^2$|$(-1/2)(\sigma^2+\varepsilon)^{-3/2}\sum_{j=1}^H (\partial\ell/\partial\hat{a}_j)(a_j - \mu)$|
|$\partial\ell/\partial\mu$|$(-1/\hat{\sigma})\sum_{j=1}^H \partial\ell/\partial\hat{a}_j$|
|$\partial\ell/\partial a_i$|$\frac{1}{H\hat{\sigma}}\left[H\frac{\partial\ell}{\partial\hat{a}_i} - \sum_k\frac{\partial\ell}{\partial\hat{a}_k} - \hat{a}_i\sum_k\frac{\partial\ell}{\partial\hat{a}_k}\hat{a}_k\right]$|
|$\partial\ell/\partial b_k$|$\sum_{i=1}^m \partial\ell/\partial h_k^{(i)}$ (sum over batch)|
|$\partial\ell/\partial g_k$|$\sum_{i=1}^m (\partial\ell/\partial h_k^{(i)}) \cdot \hat{a}_k^{(i)}$ (sum over batch)|

### BN vs LN — The Full Comparison

|Property|BN ([[BN_written_notes]])|LN (this note)|
|---|---|---|
|Normalises over|$m$ examples (one feature at a time)|$H$ features (one example at a time)|
|Statistics depend on|Entire batch|Single example only|
|Works at batch size 1|No ($\sigma^2_B = 0$)|Yes|
|Train ≠ Inference|Yes — running averages needed|No — same formula always|
|Requires `model.eval()`|Yes|No|
|Regularises|Yes (stochastic batch noise)|No (deterministic)|
|Keep Dropout?|Can reduce/remove|Yes, keep it|
|Typical use|CNNs, large-batch vision|Transformers, RNNs, NLP|
|RMSNorm variant|N/A|Drop mean subtraction: $h_k = g_k \cdot a_k / \text{RMS}(\mathbf{a})$|
|Pre/Post placement|Before nonlinearity|Pre-LN (inside residual branch)|

### Transformer Placement

|Variant|Formula|Used in|
|---|---|---|
|Post-LN|$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l))$|Original Transformer (2017)|
|Pre-LN|$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}(\text{LN}(\mathbf{x}_l))$|GPT-2/3/4, LLaMA, Mistral|

---

_Notes from Ba, Kiros & Hinton (2016), arXiv:1607.06450 — Musaib_