# Dropout as a Bayesian Approximation — Gal & Ghahramani (2016)

> [!abstract] About These Notes This note is the **companion to** [[Dropout]] and should be read **after** it. Together the two notes cover everything you need from the literature on dropout.
> 
> **This note covers:** Gal & Ghahramani, _Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning_, ICML 2016.
> 
> **What [[Dropout]] already covered:** the mechanics of dropout (Bernoulli masks, weight scaling, inverted dropout), model averaging interpretation, regularization view, linear-case equivalence to ridge regression, experimental results on vision/speech/text.
> 
> **What this note adds:** the Bayesian re-interpretation — showing that dropout training is approximate variational inference in a deep Gaussian process, deriving how to extract _calibrated uncertainty estimates_ from an already-trained dropout network at zero additional cost, and applications to regression benchmarks and reinforcement learning.

---

## Overview of Topics

1. [[#1. The Problem — Standard Deep Learning Has No Uncertainty|The problem]] — why softmax confidence ≠ model uncertainty
2. [[#2. Background — What We Need From Bayesian Models|Background]] — Gaussian processes, variational inference, the intractable posterior
3. [[#3. The Core Claim — Dropout is Variational Inference|The core claim]] — KL minimization ↔ dropout objective
4. [[#4. The Derivation — Step by Step|The derivation]] — from deep GP to the dropout loss, in full detail
5. [[#5. Obtaining Uncertainty — MC Dropout|MC Dropout]] — predictive mean, variance, log-likelihood formulas
6. [[#6. Connecting Hyperparameters to the Bayesian Model|Hyperparameter translation]] — weight decay ↔ precision ↔ length-scale
7. [[#7. Experiments|Experiments]] — regression benchmarks, classification uncertainty, RL with Thompson sampling
8. [[#8. Key Takeaways and Caveats|Takeaways and caveats]]
9. [[#9. Master Formula Sheet|Master formula sheet]]
10. [[#10. Concept Map — How This Connects to Dropout|Concept map]]

---

## 1. The Problem — Standard Deep Learning Has No Uncertainty

### The Softmax Confidence Trap

[[Dropout]] showed us that at test time we use a single deterministic forward pass with scaled weights. The softmax output gives us a probability distribution over classes — and it is tempting to read this as _model confidence_. This is wrong.

> [!danger] Softmax output ≠ model uncertainty A model can assign probability ≈ 1 to class A for an input that lies far outside the training distribution. The softmax doesn't know it has never seen anything like this input before. It just sees an activation pattern and exponentiates it.

**Concretely:** suppose you train a binary classifier. The decision boundary lives in some region of input space where you have training data. Far from that region, the model extrapolates using whatever function it learned. Passing a _point estimate_ of that function through softmax gives an S-shaped curve that asymptotes to 1. So for any input far enough from the training data, the model confidently predicts one class — even though it has no evidence either way.

What we _want_ is to pass a _distribution_ over functions through softmax — and that distribution should widen as we move away from training data, reflecting the model's genuine uncertainty about what the function looks like out there.

### Why This Matters in Practice

|Setting|Why uncertainty matters|
|---|---|
|Medical diagnosis|Uncertain predictions should be flagged for a human expert|
|Safety-critical systems (nuclear, aviation)|Must know when the model is guessing|
|Reinforcement learning|Agent needs to know when to exploit vs. explore|
|Active learning|Query the oracle on examples where the model is most uncertain|

### What Bayesian Methods Offer

A **Bayesian neural network** places a prior $p(\omega)$ over weights $\omega$ and maintains a posterior $p(\omega \mid X, Y)$ after seeing data. Predictions integrate over all possible weight settings:

$$p(y^* \mid x^*, X, Y) = \int p(y^* \mid x^*, \omega) p(\omega \mid X, Y) \, d\omega$$

This is the right answer — but the posterior $p(\omega \mid X, Y)$ is intractable for any non-trivial network. Gal & Ghahramani's contribution: **dropout already implicitly does something very close to this, and we can extract the uncertainty without changing anything in the model.**

---

## 2. Background — What We Need From Bayesian Models

> [!note] This section establishes the tools. Skim if you know variational inference and Gaussian processes; read carefully if not.

### 2.1 Gaussian Processes

A **Gaussian process** (GP) is a distribution over functions. Every finite collection of function values is jointly Gaussian. A GP is fully specified by:

- A mean function $m(x)$ (often taken as 0)
- A covariance function (kernel) $K(x, x')$

**Why GPs matter here:** infinitely wide single-hidden-layer neural networks with random weights converge to GPs (Neal, 1995). This means GPs are the natural "infinite limit" of neural networks. The paper exploits this connection in the opposite direction — showing that _finite_ dropout NNs approximate _deep_ GPs.

**The key covariance function:** for a non-linearity $\sigma(\cdot)$ and weight/bias distributions $p(w)$, $p(b)$:

$$K(x, y) = \int p(w), p(b), \sigma(w^T x + b), \sigma(w^T y + b), dw, db$$

Different non-linearities ($\text{ReLU}$, $\tanh$) give different kernels, which give different uncertainty properties. This is why the choice of activation function affects the shape of predictive uncertainty — not just model accuracy.

### 2.2 Deep Gaussian Processes

A **deep GP** stacks GP layers: each layer is a GP whose inputs are the outputs of the previous layer. This gives a hierarchical non-parametric model with very flexible priors. Exact inference in deep GPs is intractable — we need approximations.

The paper uses the **sparse spectral approximation**: represent each GP layer with finitely many random Fourier features. This maps each GP layer to a layer of explicitly represented hidden units — exactly the structure of a neural network.

### 2.3 Variational Inference (VI)

Since the true posterior $p(\omega \mid X, Y)$ is intractable, VI introduces an _approximate_ distribution $q(\omega)$ from a tractable family, and minimizes the KL divergence from $q$ to the true posterior:

$$q^*(\omega) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\omega) | p(\omega \mid X, Y))$$

Minimizing this KL is equivalent (up to a constant) to maximizing the **Evidence Lower BOund (ELBO)**:

$$\text{ELBO} = \mathbb{E}_{q(\omega)}[\log p(Y \mid X, \omega)] - \text{KL}(q(\omega) | p(\omega))$$

The first term rewards fitting the data; the second term penalizes $q$ for deviating from the prior. This is the standard VI objective. The paper's insight: the dropout objective _is_ this ELBO, for a specific choice of $q$.

---

## 3. The Core Claim — Dropout is Variational Inference

> [!tip] The Big Insight Training a dropout neural network is mathematically equivalent to performing approximate variational inference in a deep Gaussian process. The dropout objective minimizes the KL divergence between an approximate variational distribution and the posterior of a deep GP. This means dropout NNs are implicitly Bayesian — and we can read off calibrated uncertainty estimates from them.

### The Standard Dropout Objective (from [[Dropout]])

From [[Dropout#4. Formal Model Description]], the training objective with L2 regularization is:

$$\mathcal{L}_{\text{dropout}} = \frac{1}{N} \sum_{i=1}^{N} E(y_i, \hat{y}_i) + \lambda \sum_{i=1}^{L} \left( |W_i|_2^2 + |b_i|_2^2 \right) \tag{1}$$

where $E(\cdot, \cdot)$ is softmax loss (classification) or squared loss (regression), $W_i$ are weight matrices, $b_i$ are biases, and $\lambda$ is the weight decay coefficient.

> [!question] What Gal & Ghahramani show Equation (1) is, up to a constant scaling, identical to the ELBO of a deep Gaussian process — when the variational distribution $q(\omega)$ is chosen to be a product of Bernoulli distributions (i.e., _exactly_ the dropout distribution). **No approximation is needed — the match is exact.**

---

## 4. The Derivation — Step by Step

### 4.1 Setting Up the Deep GP

Consider a deep GP with $L$ layers, covariance function $K(x,y)$ as above, and weight matrices $W_i$ of dimensions $K_i \times K_{i-1}$.

The **predictive probability** of the deep GP, integrated over the (finite-rank approximated) covariance parameters $\omega = {W_1, \ldots, W_L}$:

$$p(y \mid x, X, Y) = \int p(y \mid x, \omega), p(\omega \mid X, Y), d\omega \tag{2}$$

where

$$p(y \mid x, \omega) = \mathcal{N}!\left(y;; \hat{y}(x, \omega),; \tau^{-1} I_D\right)$$

and the network output is:

$$\hat{y}(x, \omega) = \sqrt{\frac{1}{K_L}} W_L, \sigma!\left(\sqrt{\frac{1}{K_{L-1}}} W_{L-1}, \sigma!\left(\cdots \sqrt{\frac{1}{K_1}} W_2, \sigma(W_1 x + m_1) \cdots\right)\right)$$

Here $\tau > 0$ is the **model precision** (inverse noise variance), and $m_i$ are mean vectors (variational parameters). The $\sqrt{1/K_i}$ factors come from the spectral decomposition of the GP covariance.

> [!note] Intuition for the network structure Each layer of the deep GP, after spectral approximation, looks exactly like a standard NN layer with a scaled weight matrix. The non-linearity $\sigma$ is the same one used in practice. The structure is not imposed — it _emerges_ from the GP approximation.

### 4.2 The Variational Distribution — Dropout in Disguise

The true posterior $p(\omega \mid X, Y)$ is intractable. We introduce an approximate distribution $q(\omega)$ defined as:

$$W_i = M_i \cdot \text{diag}([z_{i,j}]_{j=1}^{K_{i-1}})$$

$$z_{i,j} \sim \text{Bernoulli}(p_i) \quad \text{for } i = 1, \ldots, L,; j = 1, \ldots, K_{i-1} \tag{3}$$

where $M_i$ are deterministic **variational parameter matrices** and $p_i$ are the per-layer dropout retention probabilities.

**What this means:** the random matrix $W_i$ is formed by taking the variational parameter matrix $M_i$ and zeroing out random columns — with each column zeroed independently with probability $1 - p_i$. The variable $z_{i,j} = 0$ means unit $j$ in layer $i-1$ is **dropped out** as an input to layer $i$.

> [!important] This IS dropout The distribution $q(\omega)$ defined in (3) is exactly the distribution over weight matrices induced by the Bernoulli dropout mask from [[Dropout#4. Formal Model Description]]. We are not approximating — the variational family $q$ is precisely the dropout randomness.

The distribution $q(\omega)$ is **highly multimodal**: since each column of $W_i$ is either present or zero, the joint distribution over a layer's weights has $2^{K_{i-1}}$ modes. This is a very expressive variational family despite its simple form.

### 4.3 The VI Objective

We minimize the KL divergence between $q(\omega)$ and the true posterior:

$$\min_{q} \text{KL}(q(\omega) | p(\omega \mid X, Y))$$

which is equivalent (by Bayes' theorem and dropping constants) to minimizing:

$$-\int q(\omega), \log p(Y \mid X, \omega), d\omega + \text{KL}(q(\omega) | p(\omega)) \tag{4}$$

**Term 1 — Expected log-likelihood:**

Rewrite as a sum over data points:

$$-\sum_{n=1}^{N} \int q(\omega), \log p(y_n \mid x_n, \omega), d\omega$$

Each integral is approximated by **Monte Carlo with a single sample** $\hat{\omega}_n \sim q(\omega)$:

$$\approx -\sum_{n=1}^{N} \log p(y_n \mid x_n, \hat{\omega}_n)$$

This single-sample MC estimate is _unbiased_. Drawing $\hat{\omega}_n \sim q(\omega)$ is exactly what happens during a dropout forward pass — we sample a random Bernoulli mask, which gives a random realization of the weight matrices.

**Term 2 — KL from prior:**

With a Gaussian prior $p(\omega)$ with length-scale $\ell$ and the variational distribution $q(\omega)$, one can show (see Appendix, section 4.2 of the original paper) that:

$$\text{KL}(q(\omega) | p(\omega)) \approx \sum_{i=1}^{L} \left( \frac{p_i \ell^2}{2} |M_i|_2^2 + \frac{\ell^2}{2} |m_i|_2^2 \right)$$

This is an **L2 penalty on the variational parameters** $M_i$ and $m_i$ — exactly like weight decay.

**Putting it together:**

Scaling the full objective by $\frac{1}{\tau N}$:

$$\mathcal{L}_{\text{GP-MC}} \propto \frac{1}{N} \sum_{n=1}^{N} \frac{-\log p(y_n \mid x_n, \hat{\omega}_n)}{\tau} + \sum_{i=1}^{L} \left( \frac{p_i \ell^2}{2\tau N} |M_i|_2^2 + \frac{\ell^2}{2\tau N} |m_i|_2^2 \right) \tag{5}$$

### 4.4 Matching to the Dropout Objective

Now set:

$$E(y_n, \hat{y}(x_n, \hat{\omega}_n)) = \frac{-\log p(y_n \mid x_n, \hat{\omega}_n)}{\tau}$$

For regression with Gaussian likelihood, $-\log p(y \mid x, \omega)/\tau = \frac{1}{2}|y - \hat{y}|^2$ (up to constants). For classification, $\tau = 1$ and $E$ is the softmax cross-entropy loss.

The regularization term in (5) becomes:

$$\lambda \sum_{i=1}^{L} |M_i|_2^2 \quad \text{with} \quad \lambda = \frac{p_i \ell^2}{2\tau N}$$

This matches equation (1) exactly — the L2 weight decay in the dropout objective is precisely the KL term from variational inference, with weight decay $\lambda$ encoding the prior length-scale $\ell$, model precision $\tau$, and dropout probability $p$.

> [!success] The Equivalence $$\boxed{\mathcal{L}_{\text{dropout}} \equiv \mathcal{L}_{\text{GP-MC}}}$$ The standard dropout training objective (data loss + L2 weight decay) _is_, up to a multiplicative constant, the variational inference ELBO for a deep Gaussian process with Bernoulli dropout as the variational family. Training a dropout network with weight decay is performing approximate Bayesian inference. No modification to the model or training procedure is needed.

---

## 5. Obtaining Uncertainty — MC Dropout

Since dropout training is approximate VI, the trained model implicitly defines a variational posterior $q(\omega)$ over weights. We can now do Bayesian prediction: integrate out the weights using $q(\omega)$ as the approximate posterior.

### 5.1 The Approximate Predictive Distribution

$$q(y^* \mid x^*) = \int p(y^* \mid x^*, \omega) q(\omega) \, d\omega \tag{6}$$

We estimate this by **moment matching**: compute the first two moments empirically using $T$ stochastic forward passes through the network (each with a fresh dropout mask).

Sample $T$ realizations ${\hat{W}_1^t, \ldots, \hat{W}_L^t}_{t=1}^{T}$ from $q(\omega)$ by running $T$ dropout forward passes. Let $\hat{y}_t = \hat{y}(x^*, \hat{W}_1^t, \ldots, \hat{W}_L^t)$ be the output on the $t$-th pass.

### 5.2 Predictive Mean

$$\mathbb{E}_{q(y^* \mid x^*)}[y^*] \approx \frac{1}{T} \sum_{t=1}^{T} \hat{y}^{*t} \tag{7}$$

> [!note] This was already known This result — averaging multiple stochastic forward passes — was known empirically as "MC dropout" before this paper. Srivastava et al. showed it approximates weight scaling (see [[Dropout#6. Test Time — The Weight Scaling Trick]]). Gal & Ghahramani provide the _theoretical grounding_ that turns this from a heuristic into a principled Bayesian estimate.

**What this is:** the Monte Carlo estimate of the posterior predictive mean. As $T \to \infty$, it converges to $\mathbb{E}_{q}[\hat{y}(x^*, \omega)]$.

**Practical note:** even $T = 10$ gives reasonable estimates. $T = 1000$ was used for the paper's figures but is not necessary in practice.

### 5.3 Predictive Variance

From the law of total variance, the predictive variance decomposes into **epistemic** (model uncertainty) and **aleatoric** (irreducible noise) components. The full expression:

$$\text{Var}_{q(y^* \mid x^*)}[y^*] \approx \tau^{-1} I_D + \frac{1}{T}\sum_{t=1}^{T} (\hat{y}_t)^T \hat{y}_t - \mathbb{E}[y^*]^T \mathbb{E}[y^*] \tag{8}$$

Breaking this down:

| Term                                                                                    | Meaning                                                                          |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| $\tau^{-1} I_D$                                                                         | **Aleatoric uncertainty** — irreducible noise, set by the model precision $\tau$ |
| $\frac{1}{T}\sum_{t=1}^{T} (\hat{y}_t)^T \hat{y}_t - \mathbb{E}[y^*]^T \mathbb{E}[y^*]$ | **Epistemic uncertainty** — sample variance across $T$ forward passes            |

> [!tip] Intuition The aleatoric part ($\tau^{-1}$) is the noise floor — even if the model knew the true function perfectly, predictions would still vary by this amount. The epistemic part measures how much the $T$ forward passes _disagree_ with each other. If all $T$ passes give the same output, the model is confident. If they scatter widely, the model is uncertain.
> 
> **Near training data:** all $T$ passes agree → small epistemic uncertainty. **Far from training data:** passes diverge (different dropout masks choose different extrapolations) → large epistemic uncertainty.

**Scalar form** (for 1D output, simpler notation):

$$\text{Var}[y^*] \approx \underbrace{\tau^{-1}}_{\text{noise}} + \underbrace{\frac{1}{T}\sum_{t=1}^{T}(\hat{y}^{*t})^2 - \left(\frac{1}{T}\sum_{t=1}^{T} \hat{y}^{*t}\right)^2}_{\text{sample variance of forward passes}}$$

This is the **variance of a standard sample** from the $T$ forward passes, plus the noise term $\tau^{-1}$.

### 5.4 Predictive Log-Likelihood

To evaluate how well the model captures _both_ the mean and the uncertainty, use the predictive log-likelihood (analogous to a log-score):

$$\log p(y^* \mid x^*, X, Y) \approx \log \left( \frac{1}{T} \sum_{t=1}^{T} \exp\left(-\frac{\tau}{2}\|y^* - \hat{y}^{*t}\|^2\right) \right) - \frac{1}{2}\log 2\pi - \frac{1}{2}\log \tau^{-1} \tag{9}$$

This is a **log-sum-exp** over $T$ terms — the log of the average likelihood under the $T$ sampled models.

> [!important] What this measures A model with high accuracy but poorly calibrated uncertainty will have low predictive log-likelihood, because the variance term will be wrong (too small → overconfident; too large → underconfident). This is a joint measure of accuracy AND calibration.

---

## 6. Connecting Hyperparameters to the Bayesian Model

> [!note] This section is critical for actually using MC dropout in practice.

### The Key Identity

From the matching in §4.4, the relationship between the **weight decay** $\lambda$ you use in training, the **model precision** $\tau$, and the **GP prior length-scale** $\ell$ is:

$$\tau = \frac{p_i \ell^2}{2N\lambda} \tag{10}$$

where $p_i$ is the dropout retention probability for layer $i$, $N$ is the dataset size, and $\lambda$ is the L2 weight decay coefficient.

> [!tip] How to use this You don't need to set $\tau$ directly. You already set $\lambda$ (weight decay) during training. Pick a prior length-scale $\ell$ (the paper uses $\ell = 10^{-2}$ as a default for normalized data), and compute $\tau$ from equation (10). Then plug $\tau$ into equations (8) and (9) to get calibrated uncertainty estimates and log-likelihoods.

### Practical Procedure

1. Train the dropout network as usual, with L2 weight decay $\lambda$
2. Choose prior length-scale $\ell$ (try $\ell = 10^{-2}$, or tune via Bayesian optimization on validation log-likelihood)
3. Compute $\tau = p\ell^2 / (2N\lambda)$
4. At test time, run $T$ stochastic forward passes with dropout active
5. Compute mean via (7), variance via (8), log-likelihood via (9)

**No retraining required.** The uncertainty estimates come from the model you already have.

### What the Length-Scale $\ell$ Represents

The prior length-scale $\ell$ is a GP hyperparameter that controls how quickly the covariance function decays with distance. Small $\ell$ → function varies rapidly → model expects complex, wiggly functions → uncertainty grows quickly away from data. Large $\ell$ → smooth functions → uncertainty grows slowly. The paper sets it to $10^{-2}$ based on the normalized data range, which is a reasonable default.

---

## 7. Experiments

### 7.1 Regression — Qualitative (CO₂ Dataset)

The paper trains several models on atmospheric CO₂ concentration data (≈200 data points) and evaluates extrapolation beyond the training region.

**Setup:** 4–5 hidden layers, 1024 units, ReLU or TanH non-linearities, dropout probabilities 0.1 or 0.2.

**Key finding — standard dropout vs. MC dropout:**

- **Standard dropout** (weight averaging at test time, no uncertainty): confidently predicts a nonsensical value for points far from training data. No uncertainty signal at all. (See [[Dropout#6. Test Time — The Weight Scaling Trick]] — weight averaging is the approximation that discards uncertainty.)
- **MC dropout (ReLU)**: gives the same nonsensical predictive mean but uncertainty grows as we extrapolate further. The model "knows it doesn't know."
- **MC dropout (TanH)**: uncertainty grows and then _saturates_ (TanH saturates, so its GP covariance function has different tail behavior than ReLU's).
- **Gaussian process (SE kernel)**: similar uncertainty growth, but can also capture the periodicity with the right kernel.

> [!note] ReLU vs. TanH uncertainty behavior ReLU approximates a GP with a covariance function that doesn't saturate, so uncertainty grows unboundedly away from data. TanH saturates, so its corresponding GP kernel produces bounded uncertainty. Neither is universally better — the choice should reflect prior beliefs about the function. This connects back to the hyperparameter discussion in [[Dropout#9. Hyperparameter Guide]].

**Number of forward passes $T$:** 1000 was used for the figures, but $T = 10$ is visually indistinguishable for most purposes. $T$ is a computational knob, not a model parameter.

### 7.2 Classification — Uncertainty on Rotated MNIST Digits

The paper tests a **dropout LeNet** (convolutional, with dropout before the final fully connected layer, $p = 0.5$) on a continuously rotated image of the digit "1".

**What they measure:** 100 stochastic forward passes, looking at both the softmax input (the pre-softmax activations) and the softmax output.

**Key finding:** when the rotated digit is ambiguous (between how a 1 looks and how a 5 or 7 looks), the 100 forward passes _spread widely_ across classes in the softmax input space — the uncertainty envelope of class 1 intersects that of classes 5 and 7. Even though the softmax output for the predicted class can be arbitrarily high (since softmax is invariant to shared offsets), the **variance across passes** correctly signals that the model is deeply uncertain.

> [!important] This is exactly what softmax alone cannot tell you A single forward pass would give class 5 with probability 0.8 and you'd call it confident. But 100 passes reveal that on 40 of them, the model assigns highest probability to class 1 — the high confidence is an artifact of the argmax, not a reflection of genuine certainty. MC dropout captures this.

**Quantifying classification uncertainty:**

For classification, the predictive variance formula (8) applies with $\tau = 1$. Alternatively, use:

- **Entropy** of the averaged softmax: $H = -\sum_c \bar{p}_c \log \bar{p}_c$ where $\bar{p}_c = \frac{1}{T}\sum_t p_c^t$
- **Variation ratios:** fraction of passes that don't agree with the modal prediction

### 7.3 Regression — Quantitative Benchmarks

The paper compares MC dropout against two baselines on 10 UCI regression datasets:

|Method|What it is|
|---|---|
|**VI** (Graves, 2011)|Variational Bayes for BNNs — doubles number of parameters|
|**PBP** (Hernández-Lobato & Adams, 2015)|Probabilistic backpropagation — expectation propagation, state of the art at the time|
|**Dropout**|Standard dropout NN evaluated with MC dropout (equation 7–9)|

**Results summary (from Table 1 of the paper):**

Dropout outperforms both VI and PBP in **RMSE and test log-likelihood** on 9/10 datasets. The one exception is Yacht Hydrodynamics (PBP slightly better RMSE). Notably:

- On **Boston Housing**: Dropout RMSE = 2.97 vs. PBP = 3.01 vs. VI = 4.32
- On **Protein Structure** (45K points): Dropout RMSE = 4.36 vs. PBP = 4.73
- On **Year Prediction MSD** (515K points): Dropout RMSE = 8.849 vs. PBP = 8.879

The improvement in **log-likelihood** is the more important result — it demonstrates that dropout uncertainty is _well-calibrated_, not just that the mean prediction is accurate.

> [!tip] Extended training helps further (Table 2) With 10× more training epochs, results improve substantially (e.g., Boston RMSE: 2.97 → 2.80). With 2 hidden layers instead of 1, further improvement (Boston: 2.80 → 2.80 with better log-likelihood). The uncertainty framework is not limited by the single-hidden-layer setup used to match the baselines.

**Implementation:** Keras + Theano, Adam optimizer, mini-batches of 32. Dropout probabilities of 0.05 and 0.005 (small, because networks are small — 50 units — and datasets are small). Bayesian optimization over $\tau$ using validation log-likelihood (40 iterations of BO, then train 10× longer with optimal $\tau$). Running time comparable to PBP.

### 7.4 Reinforcement Learning — Thompson Sampling

In RL, an agent must balance **exploitation** (take the action it currently believes is best) and **exploration** (try actions it's uncertain about to learn more). The standard approach in DQN-style agents is $\epsilon$-greedy: with probability $\epsilon$, take a random action; otherwise take the greedy action.

**The problem with $\epsilon$-greedy:** exploration is _random_, not _informed_. The agent doesn't know _which_ actions it's most uncertain about — it just explores uniformly.

**Thompson sampling with MC dropout:**

1. At each step, sample a single stochastic forward pass through the dropout Q-network (this samples one $\omega \sim q(\omega)$)
2. Take the action that maximizes the Q-value under this sampled model
3. In replay, use a single stochastic forward pass and backpropagate through the sampled mask

This is **principled exploration**: the agent is effectively sampling from its posterior over Q-functions and acting optimally under that sample. Over time, actions that turn out to be good become more certain; uncertain actions get explored proportionally to their uncertainty.

**Results (Figure 6 of the paper):**

On a 2D maze navigation task with rewards and penalties:

- **$\epsilon$-greedy** (standard DQN): reaches reward > 1.0 after ≈175 batches
- **Thompson sampling + MC dropout**: reaches reward > 1.0 within 25 batches from burn-in

A **7× speedup in learning** just from replacing $\epsilon$-greedy with Thompson sampling using dropout uncertainty — with no change to the network architecture.

> [!note] The plateau at 1000 batches Thompson sampling stops improving earlier than $\epsilon$-greedy at long timescales. This is because Thompson sampling keeps sampling random forward passes (always exploring a little), while $\epsilon$-greedy decays $\epsilon$ and eventually becomes pure exploitation. For practical convergence speed (the regime that matters), Thompson sampling wins decisively.

---

## 8. Key Takeaways and Caveats

### What This Paper Gives You

> [!success] The Practical Gift Any dropout neural network you have already trained can immediately give calibrated uncertainty estimates. Run $T$ stochastic forward passes at test time (keep dropout active), average the outputs for the mean, compute their sample variance for the epistemic uncertainty, compute $\tau$ from equation (10). Done. No retraining. No architecture changes. No extra parameters.

### Caveats and Limitations

**The variational distribution is bimodal and multimodal.** Each weight column is either present or zero — a $2^{K_i}$ mode distribution. The moment matching in §5 only captures the first two moments and misses higher-order structure. The uncertainty estimates are a _glimpse_ into a much richer distribution.

**The approximation is for the posterior over weights, not the true posterior.** The Bernoulli variational family may be too restricted to capture the true weight posterior in all cases.

**TanH uncertainty is bounded, ReLU is not.** This is not a bug — it reflects different GP covariance functions (see §7.1). Choose the activation function with an eye toward what uncertainty behavior you expect.

**Standard dropout at test time (weight averaging) throws away all the uncertainty.** If you use the weight-scaling trick from [[Dropout#6. Test Time — The Weight Scaling Trick]], you are discarding the variance information. You must keep dropout active and run multiple passes to get uncertainty. These are two different use cases of the same trained model.

**The prior length-scale $\ell$ matters for calibration.** Poorly chosen $\ell$ gives correctly ranked uncertainties but miscalibrated magnitudes. Tune it on validation log-likelihood when calibration is important.

---

## 9. Master Formula Sheet

| Quantity                     | Formula                                                                                                                          | Equation |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Variational distribution     | $W_i = M_i \cdot \text{diag}([z_{i,j}])$, $;z_{i,j} \sim \text{Bern}(p_i)$                                                       | (3)      |
| Dropout = GP-VI objective    | $\mathcal{L}_{\text{dropout}} \equiv \mathcal{L}_{\text{GP-MC}}$ (up to scaling)                                                 | —        |
| Precision from weight decay  | $\tau = p\ell^2 / (2N\lambda)$                                                                                                   | (10)     |
| Predictive mean (MC dropout) | $\mathbb{E}[y^*] \approx \frac{1}{T}\sum_{t=1}^T \hat{y}^{*t}$                                                                   | (7)      |
| Predictive variance          | $\text{Var}[y^*] \approx \tau^{-1}I + \frac{1}{T}\sum_t (\hat{y}_t)^T\hat{y}_t - \mathbb{E}[y^*]^T\mathbb{E}[y^*]$               | (8)      |
| Predictive log-likelihood    | $\log p \approx \log\frac{1}{T}\sum_t \exp!\left(-\frac{\tau}{2}\|y^* - \hat{y}^{*t}\|^2\right) - \frac{1}{2}\log 2\pi\tau^{-1}$ | (9)      |
| GP covariance function       | $K(x,y) = \int p(w)p(b),\sigma(w^Tx+b),\sigma(w^Ty+b),dw,db$                                                                     | —        |
| KL term (approximated)       | $\text{KL}(q\|p) \approx \sum_i \left(\frac{p_i\ell^2}{2}\|M_i\|^2 + \frac{\ell^2}{2}\|m_i\|^2\right)$                           | —        |
| Classification uncertainty   | $H = -\sum_c \bar{p}_c \log \bar{p}_c$, $;\bar{p}_c = \frac{1}{T}\sum_t p_c^t$                                                   | —        |
| Thompson sampling action     | $a_t = \arg\max_a Q(s, a;, \hat{\omega})$ where $\hat{\omega} \sim q(\omega)$                                                    | —        |

---

## 10. Concept Map — How This Connects to [[Dropout]]

```
[[Dropout]] (Srivastava et al., 2014)
  │
  ├── Key mechanism: Bernoulli mask, weight scaling, 2ⁿ implicit models
  ├── Test time: weight averaging ≈ geometric mean of ensemble [THROWS AWAY UNCERTAINTY]
  ├── Regularization view: ≡ adaptive ridge regression (linear case)
  └── Open question: can we get uncertainty from dropout? [ANSWERED HERE]
  
Gal & Ghahramani (2016) — THIS NOTE
  │
  ├── SETUP: deep Gaussian process with spectral approximation
  │     └── Each GP layer → one NN layer with random weights
  │
  ├── VARIATIONAL FAMILY: q(ω) = Bernoulli column-masking of M_i
  │     └── This IS the dropout distribution — no new randomness introduced
  │
  ├── VI OBJECTIVE: minimize KL(q ‖ p(ω|X,Y))
  │     ├── Term 1: E_q[log likelihood] ≈ MC sample = dropout forward pass
  │     └── Term 2: KL(q ‖ prior) ≈ L2 weight decay on M_i
  │     └── Result: L_dropout ≡ L_GP-MC (up to constant) ✓
  │
  ├── CONSEQUENCE: trained dropout network implicitly has posterior q(ω)
  │     ├── Predictive mean: average T stochastic forward passes
  │     ├── Predictive variance: sample variance of T passes + τ⁻¹
  │     └── Calibration: τ set by weight decay, length-scale, dropout prob
  │
  └── APPLICATIONS
        ├── Regression: outperforms Bayesian NNs (VI, PBP) on 9/10 UCI datasets
        ├── Classification: reveals uncertainty that single softmax pass hides
        └── RL: Thompson sampling via MC dropout → 7× faster learning than ε-greedy
```

---

> [!warning] Key Distinction to Keep in Mind **Weight scaling** (from [[Dropout#6. Test Time — The Weight Scaling Trick]]): single forward pass, dropout off. Gives point estimate. Fast. Discards uncertainty.
> 
> **MC dropout** (this note, §5): $T$ forward passes, dropout **on**. Gives mean + variance. $T$× slower. Extracts uncertainty.
> 
> Same trained model. Two completely different uses. Don't confuse them.

---

_Notes compiled from Gal, Y. & Ghahramani, Z. (2016), ICML 2016 — companion to [[Dropout]] — Musaib, 22/05/2026_