# Dropout — Written Notes

> **Papers:**
> 
> 1. Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov — _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_, JMLR 2014
> 2. Gal & Ghahramani — _Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning_, ICML 2016
> 
> **Prerequisite context:** [[BN_written_notes]] (BN as a regulariser — contrast with dropout) · [[LN_written_notes]] (LN also does not regularise, unlike dropout) **Links:** [[Momentum]] · [[Adam]] · [[Training Diagnostics and Optimizers]] · [[Gradient Descent Overview]]

---

## The Overfitting Problem and Why Ensembles Work

Before getting into dropout, you need to understand the two things it's simultaneously trying to solve.

### The Overfitting Problem

A deep neural network has millions of parameters and enormous expressive capacity. Given enough parameters, a network can memorise the training data perfectly — it learns the noise, the idiosyncrasies, the outliers — and then fails completely on new data. This is overfitting.

The standard culprit is **co-adaptation**: units in the network learn to work together as a tightly coordinated team. Unit $A$ detects one pattern, unit $B$ detects another, and unit $C$ only fires correctly if _both_ $A$ and $B$ are there to correct each other's mistakes. This works beautifully on training data. But these elaborate co-dependencies are fragile — they've been tuned to the specific noise structure of the training set and don't transfer to new examples.

Think of it like a conspiracy. A conspiracy involving 50 people requires all 50 to play their parts correctly. If conditions change — even slightly — the whole thing collapses. Ten smaller conspiracies of 5 people each are much more robust: if one fails, the others still work.

### Why Ensembles Are the Gold Standard Fix

The theoretically correct solution to overfitting is **Bayesian model averaging**: weight every possible model configuration by its posterior probability given the data, and average their predictions. In practice this is impossibly expensive — even averaging over just a few hundred large networks is prohibitive.

But ensembles consistently work. Different models trained on different data or with different architectures each see a different slice of the hypothesis space, overfit in different directions, and average out each other's errors. The problem is cost: training $k$ large networks takes $k$ times the compute, and using them all at test time requires $k$ forward passes per prediction.

**Dropout solves both problems at once.** It prevents co-adaptation _and_ approximates averaging over an exponential number of models — essentially getting ensemble performance at the cost of a single model.

---

## What Dropout Is — The Mechanism

### During Training

At each forward pass, for each training example, you independently zero out each hidden unit with probability $(1-p)$ — equivalently, each unit _survives_ with probability $p$.

Formally, for layer $l$ with output vector $\mathbf{y}^{(l)}$:

$$\mathbf{r}^{(l)} \sim \text{Bernoulli}(p)$$

$$\tilde{\mathbf{y}}^{(l)} = \mathbf{r}^{(l)} \ast \mathbf{y}^{(l)}$$

$$z_i^{(l+1)} = \mathbf{w}_i^{(l+1)} \tilde{\mathbf{y}}^{(l)} + b_i^{(l+1)}$$

$$y_i^{(l+1)} = f(z_i^{(l+1)})$$

where $\ast$ is elementwise multiplication and $f$ is any activation function. The vector $\mathbf{r}^{(l)}$ is a mask of zeros and ones: zeros kill units for this forward pass, ones keep them. A new mask is sampled independently for every training example and every layer.

**What the surviving units see:** A thinned network. With $n$ units in a layer, each has probability $p$ of surviving, so you get a random subnetwork of expected size $pn$.

**Why a new mask each time?** If you used the same mask every time, you'd just be training a smaller fixed network. Resampling the mask at every step means every gradient update trains a _different_ subnetwork, and no unit can build stable co-adaptations with any specific other unit, because those other units might not be there on the next pass.

### The Exponential Ensemble View

A network with $n$ units can produce $2^n$ different thinned networks (each unit is either present or absent). With $n = 4096$, that's $2^{4096}$ possible architectures — an astronomically large ensemble. All of these networks **share weights**: the weights trained by one subnetwork are the same weights used by every other subnetwork. This is what makes dropout feasible: you're implicitly training an exponentially large ensemble with the parameter count of a single network ($O(n^2)$ weights, same as without dropout).

Each individual thinned network is trained extremely rarely — most get trained for only one step in the entire training run, or even not at all. Yet the final network has learned weights that are useful across all of them, because those weights are shared.

### During Testing — Weight Scaling

At test time, you want to approximate the average prediction of all $2^n$ thinned networks. Explicitly computing $2^n$ forward passes is impossible.

**The approximation:** Use the full network (no dropout) but scale all weights by $p$. If a unit was retained with probability $p$ during training, its outgoing weights are multiplied by $p$ at test time.

$$W^{(l)}_\text{test} = p \cdot W^{(l)}$$

**Why this works:** For any hidden unit, its expected output at test time should match its expected output during training. During training, the unit is present with probability $p$ and absent with probability $(1-p)$, so its expected contribution to the next layer is $p \cdot w \cdot y$. At test time, the unit is always present but its weight is $pw$, so its contribution is $pw \cdot y$ — identical in expectation.

$$\mathbb{E}_\text{train}[\text{contribution}] = p \cdot w \cdot y + (1-p) \cdot 0 = pw \cdot y = \mathbb{E}_\text{test}[\text{contribution}]$$

This is the key identity that makes the weight scaling approximation valid.

> [!note] Inverted Dropout (Modern Implementation) In practice, most frameworks use **inverted dropout**: scale up by $1/p$ during training, and use full weights at test time unchanged. This is mathematically equivalent but more convenient — you don't need to remember to scale weights when switching to inference mode.
> 
> Standard dropout: train with weights $w$, test with $pw$. Inverted dropout: train with weights $w/p$ (after masking), test with weights $w$.
> 
> Both give the same expected output. Inverted dropout is preferred because the test-time model is simpler to deploy (no transformation needed), and it's compatible with frameworks where `model.eval()` just disables the mask.

> [!note] Contrast with BN and LN [[BN_written_notes#Training vs. Inference|BN also has a train/inference distinction]] — it switches from stochastic batch statistics to frozen population statistics. Dropout's train/inference distinction is different in nature: it's about the _mask_ (present during training, removed at test), not about statistics. LN has no such distinction at all ([[LN_written_notes#No Train Inference Distinction|LN §]]).

---

## Why Dropout Works — Three Perspectives

### Perspective 1: Preventing Co-Adaptation

This is the most direct intuition. In a standard network, the gradient update for unit $A$ says "adjust in this direction _given what all the other units are currently doing_." The network learns to rely on specific other units — if unit $B$ is always there to correct a specific pattern, unit $A$ never has to learn to handle that pattern alone.

Dropout makes the presence of any other unit unreliable. Unit $A$ can't count on unit $B$ being around. So unit $A$ has to become useful on its own, detecting a meaningful feature independently, without depending on specific co-adapting partners.

**The empirical evidence:** If you train an autoencoder on MNIST without dropout, the learned features are entangled — each filter doesn't detect anything interpretable on its own, but together they conspire to reconstruct the digit. With dropout ($p=0.5$), the features are clean: each filter detects an interpretable local structure (edges, strokes, corners). The units have been forced to learn independent, meaningful features.

The analogy to sexual reproduction from the paper is illuminating: asexual reproduction preserves successful gene combinations perfectly, but those combinations are fragile and tightly co-adapted. Sexual reproduction breaks up co-adaptations, producing genes that must be useful in many random combinations — making them more individually robust. Dropout does the same for hidden units.

### Perspective 2: Implicit Ensemble Averaging

From the model combination perspective, dropout is training an ensemble of $2^n$ networks (with shared weights) and approximating their geometric mean prediction at test time via weight scaling.

The geometric mean is a natural combination for classification models because it rewards consistent agreement: if one model gives probability 0.9 and another gives 0.1 for the same class, the geometric mean is $\sqrt{0.9 \times 0.1} = 0.3$ — the disagreement drags the combined prediction down. This is more conservative than the arithmetic mean ($0.5$) and tends to be better calibrated.

### Perspective 3: Noise Injection as Regularisation

Dropout is equivalent to adding multiplicative Bernoulli noise to the hidden units: each unit's activation is multiplied by $r \sim \text{Bernoulli}(p)$. This connects to a long line of regularisation methods that inject noise — denoising autoencoders, data augmentation, label smoothing.

The noise prevents the network from memorising by making the exact values of activations unreliable. The network must learn representations that are robust to random ablation, which are exactly the kind of general representations that transfer to new data.

**The stochastic gradient connection:** From [[Momentum#15. Stochastic Gradients|Momentum §15]], stochastic gradients (mini-batches) also inject noise that acts as an implicit regulariser — the gradient noise prevents settling into sharp minima and helps find flatter, better-generalising minima. Dropout adds a second, stronger layer of this same effect: not just noise in the gradient due to data subsampling, but noise in the network's activations themselves.

---

## Dropout as Approximate Bayesian Inference (Gal & Ghahramani, 2016)

This is the second paper and it's the deeper theoretical result. It gives a completely different — and mathematically rigorous — justification for why dropout works and opens up the powerful idea of **MC Dropout** for uncertainty estimation.

### The Problem With Standard Deep Learning Uncertainty

When a neural network outputs a softmax probability vector, practitioners often interpret these probabilities as the model's confidence. This is wrong.

Consider a binary classifier trained on data in the region $[-2, 2]$. For an input $x^* = 100$ — far outside the training distribution — the model has never seen anything like this. The network has no idea what the correct answer is. But the softmax will still output some probability, possibly even a very confident one (close to 0 or 1), because the network is just extrapolating its learned decision boundary into a region it knows nothing about.

**What we want:** A model that says "I am confident this is class 1" near training data, but "I have no idea" far from training data. This is _epistemic uncertainty_ — uncertainty due to lack of knowledge, as opposed to _aleatoric uncertainty_ (irreducible noise in the data itself).

Bayesian neural networks address this by placing distributions over the weights rather than treating them as fixed point estimates. The predictive distribution over $y^_$ for a new input $x^_$ is:

$$p(y^* \mid x^_, \mathbf{X}, \mathbf{Y}) = \int p(y^_ \mid x^*, \omega) , p(\omega \mid \mathbf{X}, \mathbf{Y}) , d\omega$$

where $\omega = {W_i}_{i=1}^L$ are the weight matrices. The posterior $p(\omega \mid \mathbf{X}, \mathbf{Y})$ encodes what we know about the weights given the training data. The integral marginalises over all possible weight configurations, weighted by their posterior probability.

**The problem:** This integral is intractable. Computing $p(\omega \mid \mathbf{X}, \mathbf{Y})$ exactly requires visiting all possible weight configurations — an infinite-dimensional space. Approximate methods (variational inference, MCMC) exist but are slow and require modifying the model.

### The Key Insight: Dropout _Already Is_ Bayesian Inference

Gal & Ghahramani's insight is that a neural network trained with dropout is already performing approximate Bayesian inference — you just need to interpret it correctly.

#### Setting Up the Variational Distribution

We want to approximate the true posterior $p(\omega \mid \mathbf{X}, \mathbf{Y})$ with a tractable distribution $q(\omega)$.

The choice of $q(\omega)$ is key. For each weight matrix $W_i$ of dimensions $K_i \times K_{i-1}$, define:

$$W_i = M_i \cdot \text{diag}([\mathbf{z}_{i,j}]_{j=1}^{K_{i-1}})$$

$$z_{i,j} \sim \text{Bernoulli}(p_i)$$

where $M_i$ is a matrix of variational parameters (think of it as the "mean" weight matrix) and $\mathbf{z}_{i,j}$ are the Bernoulli dropout masks. When $z_{i,j} = 0$, the entire $j$-th column of $W_i$ is zeroed — which is exactly what happens when unit $j$ in layer $i-1$ is dropped out.

So the variational distribution $q(\omega)$ is a distribution over weight matrices where each column is randomly zeroed with probability $(1-p_i)$.

**This is exactly the dropout distribution.** Training a dropout network is optimising the variational parameters $M_i$ — the actual weight values — under this masked distribution.

#### What Objective Are We Minimising?

The standard approach to variational inference minimises the KL divergence between the approximate posterior $q(\omega)$ and the true posterior $p(\omega \mid \mathbf{X}, \mathbf{Y})$:

$$\text{KL}(q(\omega) | p(\omega \mid \mathbf{X}, \mathbf{Y}))$$

This KL divergence can be rewritten (via the ELBO, the evidence lower bound) as:

$$-\int q(\omega) \log p(\mathbf{Y} \mid \mathbf{X}, \omega) , d\omega + \text{KL}(q(\omega) | p(\omega))$$

**First term:** Expected log-likelihood under the approximate posterior — how well do models drawn from $q(\omega)$ fit the data?

**Second term:** KL divergence from the prior — penalises the approximate posterior for being too far from the prior. With a standard Gaussian prior on the weights, this KL term becomes (proportional to) an L2 weight penalty.

Now approximate the first term by Monte Carlo integration with a single sample $\hat{\omega}_n \sim q(\omega)$ for each data point $n$:

$$\approx \frac{1}{N}\sum_{n=1}^N -\log p(y_n \mid x_n, \hat{\omega}_n)$$

For a regression model with Gaussian likelihood, $-\log p(y_n \mid x_n, \hat{\omega}_n) = \frac{\tau}{2}|y_n - \hat{y}(x_n, \hat{\omega}_n)|^2 + \text{const}$.

The full objective becomes:

$$\mathcal{L}_\text{GP-MC} \propto \frac{1}{N}\sum_{n=1}^N E(y_n, \hat{y}(x_n, \hat{\omega}_n)) + \lambda \sum_{i=1}^L \left(|M_i|_2^2 + |m_i|_2^2\right)$$

where $E(\cdot, \cdot)$ is the loss function (cross-entropy or squared error) and $\lambda$ is a weight decay constant.

**This is exactly the standard dropout training objective** (equation 1 in the Gal & Ghahramani paper):

$$\mathcal{L}_\text{dropout} = \frac{1}{N}\sum_{n=1}^N E(y_n, \hat{y}_n) + \lambda \sum_{i=1}^L \left(|W_i|_2^2 + |b_i|_2^2\right)$$

When you train a dropout network with L2 weight decay, you are minimising the KL divergence between a Bernoulli approximate posterior over weights and the posterior of a deep Gaussian process. **No modifications to the model are needed — this is what standard dropout training already does.**

> [!important] The connection between weight decay $\lambda$ and model precision $\tau$ The paper shows that the weight decay $\lambda$ and the model precision $\tau$ (inverse noise variance in a regression model) are related by: $$\tau = \frac{p \ell^2}{2N\lambda}$$ where $\ell$ is the prior length-scale (a hyperparameter encoding prior beliefs about the smoothness of the function). This relationship tells you: given a trained dropout network with known $\lambda$ and $p$, you can compute the implied model precision $\tau$, which you need to evaluate the predictive log-likelihood (see MC Dropout section below).

### What Gaussian Process Is Being Approximated?

The Gaussian process being approximated has covariance function:

$$K(x, y) = \int p(w) p(b) , \sigma(w^\top x + b) , \sigma(w^\top y + b) , dw , db$$

where $\sigma(\cdot)$ is the network's nonlinearity and $p(w), p(b)$ are the weight and bias priors.

Different nonlinearities correspond to different GP covariance functions:

- **ReLU** → an arc-cosine kernel. The resulting GP uncertainty increases _without bound_ as you move away from training data.
- **TanH** → a kernel related to the arc-tangent covariance. The uncertainty _saturates_ far from training data (remains bounded) because TanH saturates.

This explains the empirical observation from the experiments: the ReLU model's uncertainty grows as you extrapolate farther from training data (sensible — you should be more uncertain farther from data), while the TanH model's uncertainty flattens out (also reasonable, just with a different prior over function smoothness).

---

## MC Dropout — Extracting Uncertainty From a Trained Model

This is the practical payoff of the Bayesian interpretation. If a dropout network is performing approximate Bayesian inference, then the predictive distribution is:

$$q(y^* \mid x^_) = \int p(y^_ \mid x^*, \omega) , q(\omega) , d\omega$$

We can estimate this by Monte Carlo integration: sample $T$ different weight configurations from $q(\omega)$ (i.e., perform $T$ forward passes with different dropout masks) and average the predictions.

### Computing Predictive Mean

$$\mathbb{E}_{q(y^* \mid x^_)}[y^_] \approx \frac{1}{T}\sum_{t=1}^T \hat{y}^_(x^_, W_1^t, \ldots, W_L^t)$$

where $W_i^t = M_i \cdot \text{diag}(\mathbf{z}_i^t)$ is the weight matrix at forward pass $t$ with a freshly sampled dropout mask $\mathbf{z}_i^t$.

In code, this is just: **run the model $T$ times with dropout turned on at test time, and average the outputs**.

This is called **MC Dropout** (Monte Carlo Dropout). It gives a better estimate of the predictive mean than standard dropout's weight-scaling approximation (though both are close — empirically, $T \approx 50$ is enough to match the weight-scaling approximation, and larger $T$ gives even better results).

### Computing Predictive Variance (Uncertainty)

The predictive variance has two terms:

$$\text{Var}_{q(y^* \mid x^_)}[y^_] \approx \underbrace{\tau^{-1} I_D}_\text{aleatoric} + \underbrace{\frac{1}{T}\sum_{t=1}^T \hat{y}^{_T}_t \hat{y}^__t - \mathbb{E}[y^_]^T\mathbb{E}[y^_]}_\text{epistemic}$$

**Breaking this down:**

**Aleatoric uncertainty ($\tau^{-1} I_D$):** The irreducible noise in the data — even with infinite data, predictions would have this much uncertainty (it's the observation noise). $\tau$ is the model precision, derived from the weight decay $\lambda$ via the relation $\tau = p\ell^2 / (2N\lambda)$.

**Epistemic uncertainty (the sample variance term):** The uncertainty in the model's predictions across different dropout masks. This is the _variance_ of the $T$ forward passes. When all $T$ passes agree, this is small — the model is confident. When they disagree (high variance across passes), the model is uncertain.

Note that the epistemic uncertainty is just the sample variance of $T$ stochastic forward passes through the model: $$\text{Var}_\text{epistemic} \approx \frac{1}{T}\sum_{t=1}^T \hat{y}_t^2 - \left(\frac{1}{T}\sum_{t=1}^T \hat{y}_t\right)^2$$

**This is the key practical result:** To get uncertainty estimates from a dropout network, you simply:

1. Keep dropout turned on at test time
2. Run $T$ forward passes
3. The mean of the $T$ outputs is your prediction
4. The variance of the $T$ outputs is your uncertainty estimate

No retraining, no model modification, no extra parameters.

### Predictive Log-Likelihood

For regression tasks, the full predictive log-likelihood (a measure of how well the model fits the data including its uncertainty calibration) is:

$$\log p(y^* \mid x^_, \mathbf{X}, \mathbf{Y}) \approx \text{logsumexp}_t\left(-\frac{1}{2}\tau|y^_ - \hat{y}_t|^2\right) - \log T - \frac{1}{2}\log 2\pi - \frac{1}{2}\log \tau^{-1}$$

where the logsumexp is over the $T$ forward passes. This is the log of the average likelihood across the $T$ Monte Carlo samples — a better calibration metric than just RMSE because it penalises confident wrong predictions.

### When Does the Uncertainty Make Sense?

The key insight from the experiments: standard dropout (weight scaling at test time) gives you a good _mean prediction_ but **throws away uncertainty**. It collapses the ensemble of $2^n$ networks into a single deterministic output and discards the variance information. MC Dropout recovers that variance.

The experiments on the CO₂ dataset illustrate this beautifully:

- Standard dropout: extrapolates confidently (and wrongly) for points far from training data
- MC Dropout with ReLU: extrapolates with increasing uncertainty as you move farther away
- MC Dropout with TanH: extrapolates with bounded (but still visible) uncertainty

The model can now say "I don't know" instead of confidently outputting something wrong.

---

## The Forward and Backward Pass in Detail

### Forward Pass With Dropout

Layer-by-layer for a standard feedforward network. For each layer $l$ and each training example:

**Step 1 — Sample mask:** $$r_j^{(l)} \sim \text{Bernoulli}(p) \quad \text{for each unit } j \text{ in layer } l$$

**Step 2 — Apply mask to layer outputs:** $$\tilde{y}_j^{(l)} = r_j^{(l)} \cdot y_j^{(l)}$$

The surviving units keep their activations; the dropped units are set to zero.

**Step 3 — Feed into the next layer as usual:** $$z_i^{(l+1)} = \sum_j w_{ij}^{(l+1)} \tilde{y}_j^{(l)} + b_i^{(l+1)}$$ $$y_i^{(l+1)} = f(z_i^{(l+1)})$$

### Backward Pass With Dropout

Backpropagation flows only through the surviving units. If unit $j$ in layer $l$ was dropped ($r_j^{(l)} = 0$), then:

- Its activation was zero: no signal flowed forward through it
- Its gradient is also zero: no signal flows backward through it
- Its parameters receive a gradient of zero for this step: they are not updated

This is exactly how you'd backpropagate through a zero-multiplication — the chain rule gives zero. So the implementation is simple: use the same mask $\mathbf{r}^{(l)}$ in the backward pass that was used in the forward pass. Gradients are averaged over the training cases in the mini-batch; cases that dropped a unit contribute zero gradient for that unit's parameters.

> [!note] Connection to gradient noise From [[Momentum#15. Stochastic Gradients|Momentum §15]], mini-batch SGD injects unbiased noise into the gradient. Dropout injects a second, correlated layer of noise: the mask $\mathbf{r}$ is the same across the full forward and backward pass for one example, but different across examples in the mini-batch. The overall effect is that dropout introduces significantly more noise than mini-batch SGD alone — which is why dropout networks need 10–100× higher learning rates to make progress.

---

## Dropout in Linear Regression — An Exact Analysis

This is a beautiful theoretical result from the Srivastava et al. paper: for linear regression, you can analytically marginalise out the dropout noise and derive the equivalent regulariser exactly.

### Setup

Consider linear regression: $X \in \mathbb{R}^{N \times D}$, targets $y \in \mathbb{R}^N$, weights $w \in \mathbb{R}^D$.

Standard objective: $\min_w |y - Xw|^2$.

Now apply dropout to the inputs: each dimension is retained with probability $p$, so the perturbed input is $R \ast X$ where $R \in {0,1}^{N \times D}$ with $R_{ij} \sim \text{Bernoulli}(p)$ independently.

### Deriving the Equivalent Regulariser

The dropout objective is:

$$\min_w ; \mathbb{E}_{R}\left[|y - (R \ast X)w|^2\right]$$

Expand the squared norm:

$$\mathbb{E}_R\left[|y - (R \ast X)w|^2\right] = \mathbb{E}_R\left[y^\top y - 2y^\top(R \ast X)w + w^\top(R \ast X)^\top(R \ast X)w\right]$$

Take each term. The first term $y^\top y$ doesn't depend on $R$: stays as is.

**Second term:** $\mathbb{E}_R[2y^\top(R \ast X)w]$. Since $\mathbb{E}[R_{ij}] = p$, we have $\mathbb{E}[(R \ast X)] = pX$, so this becomes $2y^\top(pX)w$.

**Third term:** $\mathbb{E}_R[w^\top(R \ast X)^\top(R \ast X)w]$. This requires expanding $(R \ast X)^\top(R \ast X)$. The $(j,k)$-th element is $\sum_i R_{ij}X_{ij}R_{ik}X_{ik}$. Since $R_{ij}$ and $R_{ik}$ are independent for $j \neq k$ (but $R_{ij}^2 = R_{ij}$ since $R_{ij} \in {0,1}$):

$$\mathbb{E}[R_{ij}R_{ik}] = \begin{cases} p & \text{if } j = k \ p^2 & \text{if } j \neq k \end{cases}$$

Therefore:

$$\mathbb{E}[(R \ast X)^\top(R \ast X)]_{jk} = \begin{cases} p \sum_i X_{ij}^2 & j = k \ p^2 \sum_i X_{ij}X_{ik} & j \neq k \end{cases}$$

$$= p^2 X^\top X + p(1-p),\text{diag}(X^\top X)$$

Substituting into the objective and simplifying (absorbing the $p^2 X^\top X$ term back into the squared loss with weights $pw$):

$$\mathbb{E}_R[|y - (R \ast X)w|^2] = |y - pXw|^2 + p(1-p)|\Gamma w|^2$$

where $\Gamma = (\text{diag}(X^\top X))^{1/2}$.

**The element $\Gamma_{jj}$ is the RMS value of the $j$-th input dimension** — a measure of how much that feature varies across the dataset. So the dropout regulariser scales the penalty on weight $w_j$ by how much input dimension $j$ varies: features that vary more are penalised more.

Substituting $\tilde{w} = pw$ (absorbing the factor of $p$ into the weights):

$$\min_{\tilde{w}} ; |y - X\tilde{w}|^2 + \frac{1-p}{p}|\Gamma \tilde{w}|^2$$

**This is exactly ridge regression** ($\ell_2$ regularisation) with a data-dependent regularisation matrix $\Gamma$. The regularisation strength $\frac{1-p}{p}$ is directly controlled by the dropout rate: more dropout ($p \to 0$) means stronger regularisation. No dropout ($p = 1$) means no regularisation.

> [!important] Why this matters This result establishes dropout as a theoretically grounded regulariser, not just a heuristic. For linear models, it's exactly equivalent to a specific form of L2 regularisation. The fact that the regulariser depends on the data ($\Gamma$ involves the feature variances) means dropout is _adaptive_: it regularises rare/low-variance features less aggressively than common/high-variance ones. This is different from uniform L2 regularisation and is arguably more appropriate.
> 
> For deep nonlinear networks, an exact closed form doesn't exist — but the same qualitative intuition holds: dropout acts as an adaptive regulariser that accounts for the statistics of the data.

---

## Properties of Dropout — What the Experiments Show

### Effect on Features: Co-Adaptation Is Broken

The clearest evidence that dropout prevents co-adaptation is the feature visualisation experiment. Train an autoencoder on MNIST with and without dropout.

**Without dropout:** The learned filters are entangled and uninterpretable — each one does not detect a meaningful visual concept on its own. They conspire to produce reconstructions through complex interactions.

**With dropout ($p=0.5$):** Each filter detects a clean, localisable feature — an edge, a stroke endpoint, a corner, a specific curvature. The filters are **interpretable** because each unit has been forced to be useful independently.

This is not just an aesthetic observation — interpretable, non-co-adapted features generalise better because they correspond to real structure in the data rather than specific noise patterns in the training set.

### Effect on Sparsity: A Free Side Effect

Without any explicit sparsity constraint, dropout networks learn sparser representations. The histogram of hidden unit activations shifts dramatically: without dropout, most units have mean activation ~2.0, with a broad distribution. With dropout ($p=0.5$), most units have mean activation ~0.7, with a sharp peak at zero.

**Why this happens:** A unit that is often zero during training is a unit that has been dropped many times and had to compensate by being more concentrated when present. Units are driven to fire strongly and specifically rather than softly and diffusely. This sparsity is "free" — you get it without using any sparsity-inducing penalty.

Sparse representations are computationally efficient (fewer active units = faster inference) and often correspond to more semantically meaningful features.

### Effect of the Dropout Rate $p$

The dropout rate $p$ (probability of _retaining_ a unit) is the key hyperparameter:

**When architecture is fixed (varying $p$):**

- Very small $p$ (few units survive): severe underfitting — not enough capacity to learn the task
- $p \approx 0.4$–$0.8$: flat region of good performance — robust to the exact choice
- $p \to 1$ (no dropout): overfitting resumes

The takeaway: if you fix the architecture and vary $p$, there's a comfortable range around $p=0.5$ for hidden layers.

**When $pn$ is fixed (varying architecture):**

- Fix the expected number of surviving units ($pn$) and change both $p$ and $n$ jointly
- Networks with small $p$ (more dropout, more units) can recover their performance better than when architecture is fixed
- The best performance is around $p \approx 0.6$ for this setting
- The default $p=0.5$ remains close to optimal

**Rule of thumb:** $p = 0.5$ for hidden layers, $p = 0.8$ for input layers (you want to retain more of the raw signal).

### Effect of Dataset Size

Dropout's benefit depends heavily on dataset size:

- **Tiny datasets (< 500 examples):** No improvement. With so little data, the model can overfit _through_ the dropout noise — it memorises the training set despite dropout.
- **Medium datasets (thousands of examples):** Large gains from dropout. This is the sweet spot.
- **Large datasets (hundreds of thousands):** Smaller but still meaningful gains. As dataset size grows, overfitting becomes less severe anyway, so the dropout benefit diminishes.

This suggests dropout is most valuable when you have a medium-sized dataset and a powerful model — exactly the common practical setting.

### Monte Carlo Averaging vs. Weight Scaling

The paper verifies empirically that the weight-scaling approximation is nearly as good as true model averaging. Running $T$ stochastic forward passes at test time (Monte Carlo model averaging):

- $T = 50$: essentially as good as the weight-scaling approximation
- $T > 50$: slightly better, but within one standard deviation

This validates that the weight-scaling procedure is a good approximation to true ensemble averaging.

---

## Practical Recipe for Training Dropout Networks

The paper includes a detailed practical guide. The key parameters interact with each other, so you need to tune them together.

### Network Size: Make It Bigger

If a layer of $n$ units is optimal without dropout, you need at least $n/p$ units with dropout. With $p=0.5$, **double every layer's width**. Intuition: at any given forward pass, only $pn$ units are active. The effective capacity of the network at each step is $pn$ units, so to achieve the same effective capacity as a width-$n$ non-dropout network, you need $n/p$ units total.

### Learning Rate: Use 10–100× More

Dropout introduces a lot of noise into the gradient — each step is based on a different random subnetwork, so consecutive gradient steps may point in very different directions. To make progress despite this noise, you need a significantly higher learning rate than you'd use for the same architecture without dropout.

**Why this doesn't cause instability:** The high learning rate is stabilised by max-norm regularisation (see below) and high momentum.

**Connection to [[Adam]]:** From [[Adam#3. Adam's Update Rule — Geometry and Intuition|Adam §3]], Adam's adaptive learning rate effectively scales the step by the signal-to-noise ratio. With dropout noise, the SNR for parameters of dropped units is zero — their gradient is zero for that step. Adam handles this gracefully by maintaining per-parameter momentum. This is one reason Adam is particularly well-suited for dropout training.

### Momentum: Use High Values (0.95–0.99)

Standard non-dropout networks use momentum around 0.9. Dropout networks benefit from 0.95–0.99.

**Why:** High momentum averages out the noisy gradient updates. Each individual gradient step with dropout is very noisy (different random subnetwork), but the momentum accumulates signal across many steps. From [[Momentum#9. Momentum Update|Momentum §9]], momentum accumulates velocity — and that velocity, built up over many noisy steps, points in the direction that's consistently useful.

### Max-Norm Regularisation: Essential, Not Optional

This is the most important practical addition and the one most often overlooked. Constrain the $\ell_2$ norm of the weight vector incoming to each hidden unit:

$$|\mathbf{w}|_2 \leq c$$

This is implemented by projecting $\mathbf{w}$ back onto the ball of radius $c$ whenever it violates the constraint:

$$\mathbf{w} \leftarrow \mathbf{w} \cdot \min\left(1, \frac{c}{|\mathbf{w}|_2}\right)$$

Typical values: $c \in {3, 4}$.

**Why this is crucial with high learning rates:** A high learning rate can cause weight vectors to grow very large very quickly (especially with high momentum). Max-norm prevents this by enforcing a hard upper bound on the weight magnitude. It acts as a safety net that lets you use huge learning rates without weights exploding.

**Without max-norm, the recipe doesn't work:** The combination of high LR + high momentum + dropout without max-norm tends to diverge. With max-norm, the combination works and is much more powerful than any individual component.

### Putting It All Together: The Full Recipe

|Component|Non-dropout setting|Dropout setting|
|---|---|---|
|Hidden layer width|$n$|$n/p$ (typically $2n$ for $p=0.5$)|
|Hidden layer dropout $p$|N/A|0.5|
|Input layer dropout $p$|N/A|0.8|
|Learning rate|$\alpha$|$10\alpha$ to $100\alpha$|
|Momentum|0.9|0.95–0.99|
|Max-norm $c$|Not used|3–4|
|Weight decay|$\lambda$|$\lambda$ (same, but max-norm does the heavy lifting)|

---

## Comparison With Other Regularisers

On MNIST (architecture 784-1024-1024-2048-10, ReLU):

|Method|Test Error %|
|---|---|
|L2 weight decay only|1.62|
|L2 + L1|1.60|
|L2 + KL-sparsity|1.55|
|Max-norm only|1.35|
|Dropout + L2|1.25|
|**Dropout + Max-norm**|**1.05**|

**Key observations:**

- Dropout alone (with L2) outperforms all other regularisers
- Max-norm alone is surprisingly effective — it constrains weight magnitudes, preventing the "runaway" learning that leads to overfitting
- The combination of dropout + max-norm is the strongest configuration — they are complementary

**Why dropout beats L1/L2 alone:** L2 regularisation penalises large weights uniformly, pushing all weights toward zero. This doesn't prevent co-adaptation — units can still learn to depend on each other with small weights. Dropout fundamentally changes the _structure_ of what's learned, not just the magnitude.

---

## Dropout Restricted Boltzmann Machines

Dropout isn't limited to feedforward networks. For an RBM with visible units $v$ and hidden units $h$, a Dropout RBM augments the model with binary random variables $r_j \sim \text{Bernoulli}(p)$. If $r_j = 0$, hidden unit $h_j$ is forced to zero:

$$P(h_j = 1 \mid r_j, v) = \mathbf{1}(r_j = 1) \cdot \sigma\left(b_j + \sum_i W_{ij} v_i\right)$$

If $r_j = 0$, $h_j = 0$ regardless of the input. Learning uses contrastive divergence with dropout masks sampled for each training case.

The effects mirror those in feedforward networks: features become sparser and more interpretable, and test performance improves. The Dropout RBM can be seen as a mixture of exponentially many sub-RBMs with shared weights, each using a different subset of hidden units.

---

## Multiplicative Gaussian Noise — A Smoother Alternative

Standard dropout uses Bernoulli noise ($r \in {0, 1}$). An alternative is Gaussian noise:

$$h_i \to h_i \cdot r', \quad r' \sim \mathcal{N}(1, \sigma^2)$$

This multiplies each activation by a Gaussian random variable with mean 1 (so the expected activation is unchanged) and variance $\sigma^2$.

**Setting $\sigma^2 = (1-p)/p$** makes the Gaussian and Bernoulli dropout have the same first two moments:

- $\mathbb{E}[r_\text{Bernoulli}] = p$, $\text{Var}[r_\text{Bernoulli}] = p(1-p)$ — scale to mean 1 by dividing by $p$: $\mathbb{E} = 1$, $\text{Var} = (1-p)/p$
- $\mathbb{E}[r'] = 1$, $\text{Var}[r'] = (1-p)/p$

So both forms of dropout have the same mean and variance. The difference is in higher-order moments and the shape of the distribution: Bernoulli is bimodal (either 0 or $1/p$), Gaussian is unimodal.

**What the experiments show:** Gaussian dropout performs similarly to Bernoulli dropout, with a slight edge on MNIST (0.95% vs 1.08% for the conditions tested). Gaussian dropout requires no weight scaling at test time (since the expected activation is always the unit activation), making deployment simpler.

---

## MC Dropout in Practice — Implementation Notes

### The Three Steps

```python
# Step 1: Keep dropout ON at test time
model.train()  # NOT model.eval() — we want stochastic dropout

# Step 2: Run T forward passes
T = 50  # 20-100 is usually enough
predictions = []
with torch.no_grad():
    for _ in range(T):
        y_hat = model(x_test)
        predictions.append(y_hat)

predictions = torch.stack(predictions)  # shape: [T, batch, output_dim]

# Step 3: Compute mean and variance
mean = predictions.mean(dim=0)       # predictive mean
variance = predictions.var(dim=0)    # epistemic uncertainty
```

### How Many Forward Passes?

The paper shows that $T \approx 10$ gives a reasonable approximation, and $T \approx 50$ essentially matches the weight-scaling approximation. For production systems, $T = 20$–$50$ is a good default. These forward passes are independent and can be run in parallel.

### What Uncertainty Tells You

High epistemic variance from MC Dropout means:

- The model has not seen enough similar training examples to be confident
- The input might be out-of-distribution
- The model's weight configurations strongly disagree on this input

**Where this matters:**

- **Medical diagnosis:** High uncertainty → flag for human review
- **Autonomous systems:** High uncertainty → default to conservative action
- **Active learning:** High uncertainty → this example is most informative to label
- **Reinforcement learning:** High uncertainty → explore (Thompson sampling); low uncertainty → exploit

---

## What Dropout Does NOT Do

**Dropout does not eliminate overfitting on tiny datasets.** With fewer than ~500 examples, the network can memorise the training set even through dropout noise. Dropout needs enough data that the noise actually prevents memorisation.

**Dropout does not speed up training — it slows it down.** A dropout network takes 2–3× longer to converge than an equivalent non-dropout network, because the gradient updates are very noisy. This is the cost of the regularisation benefit.

**Standard dropout does not give calibrated uncertainty estimates.** The weight-scaled test-time prediction throws away all variance information. Only MC Dropout gives meaningful uncertainty — and even then, only the epistemic component (from the model's uncertainty about weights). Aleatoric uncertainty (data noise) requires knowing the model precision $\tau$.

**Dropout is not the same as BN's regularisation.** [[BN_written_notes#BN as a Regularizer|BN regularises]] through stochastic batch statistics — the normalisation depends on which examples are in the batch. Dropout regularises through stochastic unit removal. Both are implicit regularisers through noise, but the noise has completely different structure and the effects are complementary. Using both together is common and effective.

**Dropout does not fix architectural problems.** If your network is simply too small for the task, dropout won't help — it reduces effective capacity. Dropout only helps when the network is large enough that overfitting is the bottleneck.

---

## Quick Reference

### Dropout Mechanics

|Quantity|Formula|Note|
|---|---|---|
|Mask|$r_j^{(l)} \sim \text{Bernoulli}(p)$|Independent per unit, per layer, per example|
|Thinned activations|$\tilde{y}_j^{(l)} = r_j^{(l)} \cdot y_j^{(l)}$|Element-wise multiplication|
|Test weights (standard)|$W_\text{test} = p \cdot W_\text{train}$|Scale by keep probability|
|Inverted dropout (training)|$\tilde{y} = (r/p) \cdot y$|Scale up at train; no change at test|

### MC Dropout Uncertainty

|Quantity|Formula|
|---|---|
|Predictive mean|$\mathbb{E}[y^_] \approx \frac{1}{T}\sum_{t=1}^T \hat{y}^__t$|
|Epistemic variance|$\frac{1}{T}\sum_t \hat{y}_t^2 - \left(\frac{1}{T}\sum_t \hat{y}_t\right)^2$|
|Aleatoric variance|$\tau^{-1} I$ where $\tau = p\ell^2 / (2N\lambda)$|
|Predictive log-likelihood|$\text{logsumexp}_t(-\frac{\tau}{2}\|y^* - \hat{y}_t\|^2) - \log T - \frac{1}{2}\log 2\pi\tau^{-1}$|

### Linear Regression Equivalence

|Dropout objective|$\mathbb{E}_R[\|y - (R \ast X)w\|^2]$|
|---|---|
|Equivalent to|$\|y - pXw\|^2 + p(1-p)\|\Gamma w\|^2$|
|Which is|Ridge regression with data-dependent $\Gamma = (\text{diag}(X^\top X))^{1/2}$|
|Regularisation strength|$\frac{1-p}{p}$ — increases as $p$ decreases (more dropout = stronger regularisation)|

### Hyperparameter Defaults

|Hyperparameter|Default|Range|
|---|---|---|
|Hidden layer keep prob $p$|0.5|0.4–0.8|
|Input layer keep prob $p$|0.8|0.7–0.9|
|Layer width|$2n$ (vs non-dropout $n$)|$n/p$|
|Learning rate|10–100× non-dropout|Tune on validation|
|Momentum|0.95–0.99|Higher than non-dropout|
|Max-norm $c$|3–4|2–5|
|MC Dropout samples $T$|50|10–100|

---

## Connections to Other Notes

|Concept|Connection|
|---|---|
|Dropout as implicit regulariser via noise|[[Momentum#15. Stochastic Gradients\|Stochastic Gradients (Momentum §15)]] — same mechanism: noise prevents sharp minima|
|High momentum for dropout nets|[[Momentum#9. Momentum Update\|Momentum §9]] and [[Training Diagnostics and Optimizers#9. Momentum Update\|TDO §9]] — accumulates signal across noisy steps|
|High LR + max-norm combination|[[Training Diagnostics and Optimizers#11. Annealing the Learning Rate\|LR Annealing (TDO §11)]] — LR schedule important for dropout|
|Adam + dropout|[[Adam#11. Practical Guide\|Adam §11]] — Adam handles dropout noise well via adaptive LR|
|BN as complementary regulariser|[[BN_written_notes#BN as a Regularizer\|BN as Regulariser]] — different noise structure, can use both|
|LN does NOT regularise|[[LN_written_notes#What LN Does NOT Do\|LN §What LN Does Not Do]] — must keep dropout with LN|
|Early stopping as regulariser|[[Momentum#12. Eigenfeatures and Implicit Regularization\|Eigenfeatures (Momentum §12)]] — another form of implicit regularisation|
|Update/weight ratio monitoring|[[Training Diagnostics and Optimizers#7. Ratio of Weights to Updates\|TDO §7]] — with high LR for dropout, monitor this ratio carefully|
|Train/inference mode distinction|[[BN_written_notes#Training vs. Inference\|BN §Train vs. Inference]] — BN and dropout both require mode switching|

---

_Notes from Srivastava et al. (2014), JMLR 15:1929–1958 and Gal & Ghahramani (2016), ICML — Musaib_