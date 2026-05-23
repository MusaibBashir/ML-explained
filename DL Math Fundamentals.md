---
tags:
  - machine-learning
  - math
  - deep-learning
  - optimization
---

# Deep Learning Math Fundamentals

This document covers essential mathematical concepts frequently encountered in deep learning, focusing on vector norms (L1, L2, $L_\infty$) and the core mathematics behind landmark papers like Dropout, Batch Normalization, Layer Normalization, and Adam.

---

## 1. Vector Norms

In machine learning, norms are strictly non-negative functions used to measure the "size" or "length" of vectors. They are heavily utilized in regularization to constrain model weights and prevent overfitting.

The general $L_p$ norm for a vector $x \in \mathbb{R}^n$ is defined as:
$$||x||_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$$

### 1.1 The L1 Norm (Manhattan Norm)

The L1 norm calculates the sum of the absolute values of the vector components.

**Definition:**
$$||x||_1 = \sum_{i=1}^n |x_i|$$

**Mathematical Derivative (Subgradient):**
Because the absolute value function is not differentiable at $x_i = 0$, we use subgradients. The subgradient of $|x_i|$ is the sign function:
$$\frac{\partial ||x||_1}{\partial x_i} = \text{sign}(x_i) = \begin{cases} 1, & \text{if } x_i > 0 \\ -1, & \text{if } x_i < 0 \\ [-1, 1], & \text{if } x_i = 0 \end{cases}$$

> [!info] Usage in ML (Lasso Regularization)
> L1 regularization ($\lambda ||w||_1$) pushes weights exactly to zero, acting as an implicit feature selector. This happens because the gradient is constant ($\pm \lambda$) regardless of how small the weight gets, relentlessly pushing small weights to zero until they cross the origin (where the subgradient captures them).

### 1.2 The L2 Norm (Euclidean Norm)

The L2 norm represents the standard straight-line distance from the origin.

**Definition:**
$$||x||_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

**Mathematical Derivative:**
In machine learning, we almost always use the **squared L2 norm** ($||x||_2^2$) for regularization (Ridge) because it removes the square root, making the derivative beautifully simple and computationally cheap:
$$\frac{\partial}{\partial x} ||x||_2^2 = \frac{\partial}{\partial x} \left( \sum_{i=1}^n x_i^2 \right) = 2x$$

> [!info] Usage in ML (Ridge / Weight Decay)
> L2 regularization penalizes large weights heavily but becomes gentle as weights approach zero (since the gradient $2\lambda w$ scales with the weight itself). This leads to dense networks with small, smoothly distributed weights, preventing any single feature from dominating.

### 1.3 The $L_\infty$ Norm (Max Norm)

As $p \to \infty$, the largest absolute component of the vector completely dominates the sum.

**Definition:**
$$||x||_\infty = \max_{i} |x_i|$$

**Mathematical Derivation Concept:**
Consider $||x||_p = \left( \sum |x_i|^p \right)^{1/p}$. Let $x_{max} = \max_i |x_i|$. 
Factor it out: 
$$||x||_p = x_{max} \left( \sum \left( \frac{|x_i|}{x_{max}} \right)^p \right)^{1/p}$$
As $p \to \infty$, any term where $|x_i| < x_{max}$ becomes $0$. The sum approaches $1$ (assuming a unique maximum), and $1^{1/\infty} = 1$. Thus, the limit is simply $x_{max}$.

> [!info] Usage in ML
> Max-norm regularization constrains the maximum possible value of any weight vector entering a neuron (e.g., $||w||_\infty < c$). This is particularly useful in preventing exploding gradients in recurrent networks and is often paired with Dropout.

---

## 2. Math in Landmark ML Papers

Beyond standard calculus and linear algebra, foundational ML papers introduce specific probabilistic, statistical, and optimization constructs.

### 2.1 Dropout (Srivastava et al., 2014)

Dropout prevents complex co-adaptations by randomly zeroing out units. The core math relies on the **Bernoulli distribution** and **Expectation scaling**.

**Forward Pass (Training):**
For a given layer activation vector $y$, we sample a binary mask $r$ where each element follows a Bernoulli distribution with probability $p$ (chance of keeping the neuron).
$$r_i \sim \text{Bernoulli}(p)$$
$$\tilde{y}_i = r_i \cdot y_i$$

**Inverted Dropout (The Modern Math Approach):**
During inference, we want the network to behave deterministically. The expected value of a neuron during training is $\mathbb{E}[\tilde{y}_i] = p \cdot y_i$. 
To ensure the expected input to the next layer remains the same during testing (where dropout is off, meaning $p=1$), original dropout scaled the weights at test time by $p$.
Modern frameworks use *Inverted Dropout*, which scales the activations *during training* by $1/p$ so no changes are needed at test time:
$$\tilde{y}_i = \frac{r_i \cdot y_i}{p}$$
Now, $\mathbb{E}[\tilde{y}_i] = \frac{p \cdot y_i}{p} = y_i$.

### 2.2 Batch Normalization (Ioffe & Szegedy, 2015)

Batch Norm addresses Internal Covariate Shift by standardizing inputs across a mini-batch. 

Let $\mathcal{B} = \{x_{1...m}\}$ be a mini-batch of size $m$.
1. **Mini-batch Mean:** $$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i$$
2. **Mini-batch Variance:** $$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$
3. **Normalize:** $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$ (where $\epsilon$ prevents division by zero).
4. **Scale and Shift (Learnable Parameters):** $$y_i = \gamma \hat{x}_i + \beta$$

> [!note] The Smoothing Effect
> Batch norm mathematically smooths the optimization landscape. It scales the gradients inversely proportional to the variance, making the training resilient to the scale of weights.

### 2.3 Layer Normalization (Ba et al., 2016)

Batch Norm fails in Recurrent Neural Networks (RNNs) and small batch sizes because mini-batch statistics become unstable. Layer Norm calculates the statistics across the **feature dimension** for each *individual* sample, breaking the dependency on the batch entirely.

For a single sample $x$ with $H$ hidden units:
1. **Layer Mean:** $$\mu = \frac{1}{H} \sum_{j=1}^H x_j$$
2. **Layer Variance:** $$\sigma^2 = \frac{1}{H} \sum_{j=1}^H (x_j - \mu)^2$$
The normalization step remains identical to Batch Norm. Layer norm is the standard for Transformers (like the one writing this).

### 2.4 Adam Optimizer (Kingma & Ba, 2014)

Adam (Adaptive Moment Estimation) combines the ideas of Momentum and RMSProp. Its mathematical elegance lies in its calculation of moving averages and its **bias correction**.

Let $g_t$ be the gradient at time step $t$.

**1. Exponential Moving Averages (Moments):**
Adam keeps a running average of the gradient (1st moment, $m_t$) and the squared gradient (2nd moment uncentered variance, $v_t$).
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
*(Typically, $\beta_1 = 0.9$ and $\beta_2 = 0.999$)*

**2. Bias Correction (The Key Math Insight):**
Because $m_0$ and $v_0$ are initialized to $0$, the estimates are biased towards zero, especially in early time steps. The paper proves that taking the expected value $\mathbb{E}[m_t]$ results in the true first moment scaled by $(1 - \beta_1^t)$. To unbias the estimators, Adam divides by this term:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
*(Note: As $t \to \infty$, $\beta^t \to 0$, so the bias correction term naturally phases out).*

**3. Parameter Update:**
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
Here, the step size is bounded by $\alpha$ (the learning rate). The effective step taken for each parameter is invariant to the scale of the gradients, making Adam highly robust.


## 3. Bayesian Deep Learning (Prerequisites for "Bayes by Backprop")

To fully grasp Blundell et al.'s *Weight Uncertainty in Neural Networks* (arXiv:1506.02142), the transition from deterministic weights to probabilistic weights requires a strong foundation in Variational Inference (VI) and continuous probability distributions.

### 3.1 Bayes' Theorem for Neural Networks

Instead of finding a single optimal set of weights $w$, Bayesian learning seeks the **posterior distribution** of the weights given the training data $\mathcal{D}$:
$$P(w|\mathcal{D}) = \frac{P(\mathcal{D}|w)P(w)}{P(\mathcal{D})}$$

* **Prior $P(w)$:** Our initial belief about the weights (often a standard Gaussian). This acts mathematically similar to L2 regularization.
* **Likelihood $P(\mathcal{D}|w)$:** The probability of the data given the weights. Just as raw likelihood functions are formulated to derive test statistics or evaluate power functions in statistical inference, here it represents how well the network fits the data (e.g., categorical cross-entropy for classification).
* **Evidence $P(\mathcal{D})$:** The marginal likelihood. 
$$P(\mathcal{D}) = \int P(\mathcal{D}|w)P(w) dw$$
This high-dimensional integral over all possible continuous network weights is intractable, making direct Bayesian updates impossible for neural networks.

### 3.2 Variational Inference (VI)

Since we cannot compute the true posterior $P(w|\mathcal{D})$, Variational Inference approximates it with a tractable, parameterized distribution $q_\theta(w)$ (e.g., a Gaussian where $\theta = (\mu, \sigma)$). The goal transforms from an intractable integration problem into a continuous optimization problem: find the parameters $\theta$ that make $q_\theta(w)$ as close to $P(w|\mathcal{D})$ as possible.

### 3.3 Kullback-Leibler (KL) Divergence

To measure the distance between the true posterior and our approximation, we use KL Divergence. For continuous distributions, it is defined as the expectation of the log difference:
$$D_{KL}(q_\theta(w) || P(w|\mathcal{D})) = \int q_\theta(w) \log \frac{q_\theta(w)}{P(w|\mathcal{D})} dw = \mathbb{E}_{q_\theta} \left[ \log \frac{q_\theta(w)}{P(w|\mathcal{D})} \right]$$
Since KL Divergence is strictly non-negative, $D_{KL} \geq 0$.

### 3.4 The Evidence Lower Bound (ELBO)

We still have the intractable $P(w|\mathcal{D})$ in our KL equation. We can manipulate the formula using Bayes' theorem:
$$D_{KL}(q_\theta(w) || P(w|\mathcal{D})) = \mathbb{E}_{q_\theta} [\log q_\theta(w)] - \mathbb{E}_{q_\theta} [\log P(w|\mathcal{D})]$$
$$= \mathbb{E}_{q_\theta} [\log q_\theta(w)] - \mathbb{E}_{q_\theta} [\log P(\mathcal{D}|w) + \log P(w) - \log P(\mathcal{D})]$$
Since $\log P(\mathcal{D})$ does not depend on $w$, it acts as a constant and can be pulled out of the expectation:
$$= \mathbb{E}_{q_\theta} [\log q_\theta(w)] - \mathbb{E}_{q_\theta} [\log P(w)] - \mathbb{E}_{q_\theta} [\log P(\mathcal{D}|w)] + \log P(\mathcal{D})$$
Notice that the first two terms form another KL divergence—the divergence between our approximation and the prior ($D_{KL}(q_\theta(w) || P(w))$). Rearranging to solve for the log evidence gives us:
$$\log P(\mathcal{D}) - D_{KL}(q_\theta(w) || P(w|\mathcal{D})) = \mathbb{E}_{q_\theta} [\log P(\mathcal{D}|w)] - D_{KL}(q_\theta(w) || P(w))$$

The right side of this equation is the **Evidence Lower Bound (ELBO)**. Because $D_{KL} \geq 0$, minimizing the divergence to the true posterior is mathematically equivalent to **maximizing the ELBO**. 

In *Bayes by Backprop*, the cost function to be *minimized* is the negative ELBO:
$$\mathcal{F}(\mathcal{D}, \theta) = D_{KL}(q_\theta(w) || P(w)) - \mathbb{E}_{q_\theta} [\log P(\mathcal{D}|w)]$$
* **Term 1 (Complexity Cost):** How much our approximation deviates from the prior.
* **Term 2 (Likelihood Cost):** How well the model fits the data (expected log-likelihood).

### 3.5 The Reparameterization Trick

To optimize $\mathcal{F}(\mathcal{D}, \theta)$ using gradient descent, we must differentiate through the expectation $\mathbb{E}_{q_\theta}$. We cannot backpropagate through a random sampling node $w \sim \mathcal{N}(\mu, \sigma^2)$ because sampling is a stochastic, non-differentiable operation.

The **Reparameterization Trick** isolates the randomness into a parameter-free variable $\epsilon$.
Instead of sampling $w$ directly, we sample $\epsilon$ from a standard normal distribution $\epsilon \sim \mathcal{N}(0, I)$, and then deterministically transform it:
$$w = \mu + \sigma \odot \epsilon$$
*(Note: To ensure the variance $\sigma$ remains strictly positive during unconstrained optimization, the paper parameterizes it using the softplus function: $\sigma = \log(1 + \exp(\rho))$, and optimizes for $\theta = (\mu, \rho)$).*

By pushing the random variable $\epsilon$ to the edge of the computational graph, the path from the loss function back to the parameters $\mu$ and $\sigma$ becomes fully deterministic. Gradients can now flow freely through the node, allowing standard backpropagation to compute the derivatives of complex probability integrals.
