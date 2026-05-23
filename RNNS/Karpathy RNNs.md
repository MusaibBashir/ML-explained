# The Unreasonable Effectiveness of Recurrent Neural Networks — Written Notes

> **Source:** Andrej Karpathy, _The Unreasonable Effectiveness of Recurrent Neural Networks_, May 21 2015 **URL:** https://karpathy.github.io/2015/05/21/rnn-effectiveness/ **Code:** https://github.com/karpathy/char-rnn (multi-layer LSTM character language models)
> 
> **Links:** [[LN_written_notes]] (LN used in modern RNNs) · [[Dropout_written_notes]] (dropout used in char-rnn: 0.5 after each layer) · [[Momentum]] · [[Training Diagnostics and Optimizers]] · [[Adam]]

---

## The Central Claim

Vanilla neural networks and CNNs have a fixed API: one fixed-size input vector → one fixed-size output vector, computed in a fixed number of steps. They are **doomed from the get-go by a fixed computational budget**.

RNNs remove all three of these constraints. They can accept variable-length input sequences, produce variable-length output sequences, and the number of computational steps is not fixed — the same recurrent transformation can be applied as many times as needed. This makes them enormously more flexible, and — the central argument of this post — that flexibility translates into surprising, almost unsettling capability even in very simple instantiations.

The demonstration Karpathy chooses is deliberately humble: **character-level language modelling**. Feed a huge text file in character by character, ask the model to predict the next character at each step. That's it. No word-level tokenisation, no embeddings pre-trained on giant corpora, no explicit grammar rules. Just a sequence model over raw bytes. And yet the outputs — after training on Shakespeare, Linux source code, algebraic geometry LaTeX, Wikipedia — are stunning, not because they're perfect but because of how much structured knowledge emerges from such a bare-bones objective.

---

## What RNNs Are — The API and the Math

### The Fixed-to-Fixed Limitation of Vanilla Nets

A standard feedforward network computes: $$\mathbf{y} = f(\mathbf{x}; \theta)$$

where $\mathbf{x}$ and $\mathbf{y}$ are fixed-dimensional vectors and $\theta$ are the weights. The function $f$ has a fixed depth — a fixed number of composed operations. It doesn't matter what you put in, it always goes through the same number of layers and always outputs the same number of numbers.

This is fine for image classification (fixed input, fixed output). But it is fundamentally unable to handle variable-length inputs or outputs, or tasks where the amount of "thinking" needed varies with the input.

### The Sequence Regime

RNNs generalise this to **five input/output modes** (Karpathy's diagram makes this concrete):

|Mode|Example|
|---|---|
|One-to-one|Standard classification (no RNN needed)|
|One-to-many|Image captioning: image → sentence of variable length|
|Many-to-one|Sentiment analysis: sentence → positive/negative label|
|Many-to-many (async)|Machine translation: English sentence → French sentence|
|Many-to-many (synced)|Video labelling: every frame gets a label|

The critical insight: **in every case the recurrent transformation is fixed and can be applied as many times as needed**. The architecture doesn't change — only how many times you unroll it changes.

### The Core Computation

At each time step $t$, the RNN takes:

- The current input vector $\mathbf{x}_t$
- Its previous hidden state vector $\mathbf{h}_{t-1}$

And produces:

- A new hidden state $\mathbf{h}_t$ (the internal memory, updated based on what just happened)
- An output $\mathbf{y}_t$ (whatever the task requires)

For a **Vanilla RNN**, the update equations are:

$$\mathbf{h}_t = \tanh\left(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t\right)$$

$$\mathbf{y}_t = W_{hy}\mathbf{h}_t$$

Written as a Python class:

```python
class RNN:
    def step(self, x):
        # Update hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # Compute output
        y = np.dot(self.W_hy, self.h)
        return y
```

**Breaking down the hidden state update:**

The term $W_{hh}\mathbf{h}_{t-1}$ says: "what do I remember from before?" — it linearly combines all the components of the previous hidden state.

The term $W_{xh}\mathbf{x}_t$ says: "what's happening right now?" — it processes the current input.

These two streams are **added** (not concatenated — they must have the same dimension) and then squashed through $\tanh$ into the range $[-1, 1]$.

The weights $W_{hh}$, $W_{xh}$, $W_{hy}$ are the learnable parameters. Training finds the values of these matrices that make the outputs $\mathbf{y}_t$ useful for whatever task you care about.

**Initialisation:** $\mathbf{h}_0 = \mathbf{0}$ (zero vector). The hidden state starts with no memory of anything.

> [!note] Why tanh and not ReLU? The tanh squash is important for vanilla RNNs. Without it, the hidden state $\mathbf{h}_t$ could grow unboundedly as $W_{hh}$ is applied repeatedly. With tanh, values stay in $[-1,1]$. In practice, vanilla RNNs still suffer from vanishing/exploding gradients over long sequences — which is exactly why LSTMs (see below) became the standard. [[LN_written_notes#6. Layer Norm in Recurrent Neural Networks|LN §6]] discusses how Layer Normalisation further stabilises RNN training.

### RNNs as Programs

There is a deep way to think about the hidden state: **it is the RNN's memory of everything it has seen so far**. The RNN is not just a function of the current input — it is a function of the entire input history, compressed into the fixed-size hidden vector $\mathbf{h}_t$.

This gives RNNs a fundamentally different character from feedforward networks:

> "If training vanilla neural nets is optimisation over functions, training recurrent nets is optimisation over programs."

RNNs are Turing-complete (Siegelmann & Sontag, 1995): with the right weights, an RNN can simulate any program. In practice you shouldn't read too much into this (same caveat applies to universal approximation theorems for feedforward nets), but it captures the intuition that RNNs are fundamentally more general computation than feedforward nets.

### Going Deep — Stacked RNNs

Just like feedforward networks improve by adding depth, RNNs can be stacked. A 2-layer RNN:

```python
y1 = rnn1.step(x)   # first RNN takes raw input
y  = rnn2.step(y1)  # second RNN takes first RNN's output
```

The first RNN produces a sequence of hidden representations from the raw input. The second RNN takes those representations as its inputs and builds higher-level abstractions. In Karpathy's experiments, he uses 2–3 layer LSTMs with 512 hidden units each. Stacking works: deeper models generate more coherent text.

### The LSTM — Why It's Used in Practice

The vanilla RNN has a severe problem: **vanishing gradients**. When you backpropagate through time (BPTT), the gradient of the loss w.r.t. early hidden states involves products of the Jacobian $\partial \mathbf{h}_t / \partial \mathbf{h}_{t-1}$ across many time steps. If the largest singular value of this Jacobian is $< 1$, these products shrink exponentially → gradients vanish → the model cannot learn dependencies spanning more than a handful of time steps.

The **Long Short-Term Memory (LSTM)** (Hochreiter & Schmidhuber, 1997) solves this with a fundamentally different architecture. It maintains a **cell state** $\mathbf{c}_t$ — a separate memory that flows through time with only additive (not multiplicative) updates. The gates (input, forget, output) are learned sigmoid mechanisms that control:

- How much of the new input to write to cell memory (input gate)
- How much of the old cell memory to erase (forget gate)
- How much of the cell memory to expose as the hidden state output (output gate)

The crucial property: **gradients can flow back through the cell state without vanishing**, because the cell state update is additive. This is the RNN equivalent of ResNet's skip connections — a highway for gradients.

From [[LN_written_notes#6. Layer Norm in Recurrent Neural Networks|LN §6]], Layer Normalisation is often applied inside the LSTM's gates to further stabilise training by keeping the pre-gate activations in a reasonable range.

Karpathy notes: "everything I've said about RNNs stays exactly the same, except the mathematical form for computing the update gets a little more complicated." All the experiments in the post use LSTMs.

---

## Character-Level Language Modelling — The Task

### What We're Doing

Feed the model a sequence of characters. At each time step, it must predict the probability distribution over all possible next characters. That's the entire task.

More formally: given a vocabulary of $K$ possible characters, the model maintains a hidden state and at each step outputs a vector of $K$ logits. Apply softmax to get a probability distribution. The training objective is cross-entropy loss between the predicted distribution and the actual next character (one-hot encoded).

### The "helo" Example — Making It Concrete

Vocabulary: `{h, e, l, o}` (just 4 characters). Training sequence: `"hello"`. This one string contains 4 training examples:

|Context|Target|
|---|---|
|`h`|`e`|
|`he`|`l`|
|`hel`|`l`|
|`hell`|`o`|

**Encoding:** Each character is one-hot encoded. With vocabulary size 4: $$\text{h} \to [1,0,0,0], \quad \text{e} \to [0,1,0,0], \quad \text{l} \to [0,0,1,0], \quad \text{o} \to [0,0,0,1]$$

Feed these one-hot vectors into the RNN one at a time. At each step, the output is a 4-dimensional logit vector — one logit per possible next character.

**Example (from the diagram):** After seeing `h`, the RNN outputs:

- h: 1.0 (wrong — should be low)
- **e: 2.2** (correct — should be high)
- l: -3.0 (wrong — should be low)
- o: 4.1 (wrong — should be low)

We apply softmax to get probabilities, compute cross-entropy loss against the one-hot target `e`, and backpropagate.

> [!note] Crucial observation The character `l` appears twice in `hello`, but with different targets (`l` the first time, `o` the second time). The model **cannot solve this from the input alone** — it must use the hidden state to remember where it is in the sequence. This is exactly what recurrent connections are for.

### Training Objective

Standard softmax cross-entropy, applied at every time step simultaneously. For a sequence of length $T$:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p(x_{t+1} \mid x_1, x_2, \ldots, x_t)$$

where $p(x_{t+1} \mid \ldots)$ is the model's predicted probability for the correct next character. We minimise this over the entire training corpus.

**Optimiser:** Karpathy uses RMSProp or Adam (see [[Adam]] and [[Training Diagnostics and Optimizers#14. RMSProp|TDO §14]]) for per-parameter adaptive learning rates. These are important here because different weights in the RNN receive very different gradient magnitudes — some units fire frequently, some rarely — so adaptive methods stabilise training.

### Sampling at Test Time

Training gives us a model that predicts next-character distributions. To generate text:

1. Feed a seed character (or a warm-up sequence) to get the initial hidden state
2. The model outputs a probability distribution over the vocabulary
3. **Sample** from this distribution to get the next character
4. Feed that character back in as the next input
5. Repeat forever

This is an **autoregressive** process: outputs become inputs. Each character is conditioned on all previously generated characters (through the hidden state).

### Temperature — Controlling Creativity vs. Coherence

The softmax distribution has a temperature parameter $T$ (not the same $T$ as time steps — different overloaded symbol):

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

where $z_i$ are the logits and $\tau$ is the temperature.

|Temperature $\tau$|Effect|
|---|---|
|$\tau \to 0$|Argmax — always picks the most likely character. Very repetitive.|
|$\tau = 1$|Standard softmax. The model's actual distribution.|
|$\tau > 1$|Flatter distribution — more random, more diverse, more mistakes|

**The Paul Graham example at near-zero temperature:**

> _"is that they were all the same thing that was a startup is that they were all the same thing that was a startup is that they were all the same thing..."_

This is the model's mode — what it thinks is most likely given what it's been saying. It loops because "startup is that they were all the same thing that was a" is the most common continuation it learned. High temperature would break this loop but introduce more incoherence.

**Practical guidance:** $\tau \approx 0.5$–$0.8$ gives the best results in practice — confident enough to be coherent, random enough to be varied.

---

## The Experiments — What the Model Learns

### Experiment 1: Paul Graham Essays (~1MB)

**Architecture:** 2-layer LSTM, 512 hidden units, dropout 0.5 after each layer, trained on 1M characters.

**What it learns:**

- English vocabulary and spelling (from scratch — no pre-training)
- Comma placement, apostrophes, spaces
- Essay-like argumentative structure
- Even footnote citation style (e.g., `[2]`)

**Sample:**

> _"The surprised in investors weren't going to raise money. I'm not the company with the time there are all interesting quickly, don't have to get off the same programmers."_

This is not coherent English, but it's remarkably English-like for a model that learned purely from character sequences with no linguistic knowledge built in.

### Experiment 2: Shakespeare (4.4MB)

**Architecture:** 3-layer LSTM, 512 hidden units per layer.

**What it learns:**

- Play format: `CHARACTER_NAME:\n dialogue text`
- Iambic rhythm and meter (somewhat)
- Dramatic vocabulary and style
- Speaker transitions

**Sample:**

```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.
```

The model generates speaker names and dialogue simultaneously. It maintains consistent character names and the correct structural format across many lines.

### Experiment 3: Wikipedia with Markdown (96MB)

**What it learns:**

- Wikilink syntax: `[[Article Name]]` — opens and closes correctly
- Citation format: `{{cite journal | ...}}`
- External link format with URLs (it halluccinates URLs that don't exist but are syntactically valid)
- Section headers: `== Section Name ==`
- Lists: `* item`, `** subitem`
- XML-like nested structure

**Remarkable observation:** When the model generates a URL like `http://www.humah.yahoo.com/guardian.cfm/7754800786d17551963s89.htm`, the URL is syntactically valid but doesn't exist. The model has learned the _structure_ of URLs without memorising specific ones.

**Even more remarkable:** The model sometimes generates valid XML with correctly nested and closed tags, including realistic metadata (timestamps, usernames, IDs — all fabricated but plausible).

### Experiment 4: Algebraic Geometry LaTeX (16MB)

**What it learns:**

- LaTeX syntax: `\begin{proof}`, `\end{proof}`, `\begin{enumerate}`, etc.
- Mathematical notation: `$\mathcal{F}$`, `$\mathfrak{q}$`
- Theorem/lemma/proof structure
- Cross-referencing: `\hyperref[label]{text}`, `\label{...}`
- Mathematical vocabulary that sounds plausible

**The failure mode is telling:** It opens `\begin{proof}` but closes with `\end{lemma}`. The model forgets which environment it opened because the matching closing bracket is far away in the sequence — a classic **long-range dependency** failure. The model can handle short-range structure (brackets within a few dozen characters) but fails at long-range structure (matching environment tags that can be hundreds of characters apart).

With larger models, these errors become less common — bigger hidden state = better long-term memory.

### Experiment 5: Linux Source Code (474MB)

**Architecture:** 3-layer LSTM, ~10 million parameters, trained for several days on a GPU.

**What it learns:**

- C syntax: pointer notation (`*`, `->`), address-of (`&`), casting, bitwise operations
- Comment style: `/* ... */` blocks, inline comments
- Function signatures with correct return types
- `#include` statements with plausible (though invented) header names
- GNU license text (memorised exactly, reproduced character-for-character)
- Curly brace nesting, indentation
- Macro definitions: `#define NAME value`

**The Linux sample:**

```c
static int indicate_policy(void)
{
  int error;
  if (fd == MARN_EPT) {
    if (ss->segment < mem_total)
      unblock_graph_and_set_blocked();
    else
      ret = 1;
    goto bail;
  }
  ...
  return segtable;
}
```

When you scroll through megabytes of generated code, it genuinely feels like reading a C codebase. The model has internalized:

- C idioms (`goto bail`, `spin_unlock`, `mutex_unlock`)
- Naming conventions (`func`, `dev`, `buf`, `ptr`, `fd`)
- Common patterns (`for (i = 0; i < N; i++)`)

**The characteristic failure modes:**

- Uses undefined variables (it uses `rw` which was never declared)
- Declares variables it never uses (`int error;`)
- Functions declared `void` that return values, or vice versa
- `if (tty == tty)` — a vacuously true comparison that still uses a valid variable

All of these failures are **long-range consistency failures**: the model knows the local syntax but loses track of what was declared earlier in the function scope.

### Experiment 6: Baby Names (8000 names)

A tiny dataset — just 8000 names, one per line. The model generates new names that mostly don't appear in the training set (90% novel):

_Rudi, Levette, Berice, Lussa, Mareanne, Chrestina, Carissy..._

These look like real names — they have the phonological structure of English/Romance names — but most are novel. The model has learned the distribution over character n-grams that constitute "name-like" sequences.

---

## How the Model Learns — The Evolution of Training

Karpathy trains on Tolstoy's _War and Peace_ and samples every 100 iterations. The progression reveals exactly what the model learns, and in what order:

**Iteration 100:** Random character sequences. But — crucially — already learning word-space structure:

```
tyntd-iafhatawiaoihrdemot  lytdws  e ,tfti, astai f ogoh eoase rrranbyne
```

You can see spaces starting to appear between character clusters. The model has learned the most basic thing: words are separated by spaces.

**Iteration 300:** Learning quote and period structure:

```
"Tmont thithey" fomesscerliund
Keushey. Thom here
```

Words are now space-separated, periods appear at sentence ends, quotes appear (though the content is nonsense).

**Iteration 500:** Short common words are learned:

```
we counter. He stutn co des. His stanted out one ofler that concossions
```

`we`, `He`, `His`, `one`, `and` — the most frequent words are being memorised.

**Iteration 700:** More English-like structure:

```
Aftair fall unsuch that the hall for Prince Velzonski's
```

Longer words, proper nouns with apostrophe-s possessives, prepositional phrases.

**Iteration 2000:** Names, quotations, coherent syntax:

```
"Why do what that day," replied Natasha, and wishing to himself the fact
```

**The pattern of learning:**

1. **First:** whitespace structure (words are separated)
2. **Then:** punctuation rules (periods, commas, quotes)
3. **Then:** short common words (the, and, he, is)
4. **Then:** longer vocabulary words
5. **Last:** multi-word thematic consistency and long-range dependencies

This is a beautiful illustration of the implicit curriculum in language modelling. The model doesn't need to be told to learn in this order — it simply learns the most statistically regular things first.

---

## Understanding What's Happening Inside — Neuron Visualisations

Karpathy examines individual neurons (dimensions of the hidden state vector) by visualising their activation values as the RNN reads text. Some neurons develop interpretable functions:

### Neuron 1: URL Detector

A neuron that is **near zero outside URLs** but **strongly activated inside URLs** (like `http://www.something.com/path`). The RNN is using this neuron to track "am I currently inside a URL?" — a piece of boolean state that's useful for predicting the next character (inside a URL, alphanumeric characters and `/`, `.`, `-` are more likely; outside, they're less common).

### Neuron 2: Wikilink Bracket Tracker

A neuron that fires inside `[[ ]]` markdown links and turns off outside them. Interestingly, it doesn't activate after seeing the first `[` — it must wait for the second `[` before activating. This means another neuron (not shown) must be tracking "have I seen one `[` already?" and only after that confirmation does this neuron activate. A distributed counting circuit has emerged.

### Neuron 3: Position-within-Link Encoder

A neuron that increases **linearly** across the `[[ ]]` span — giving the model a time-aligned coordinate within the link. The model can use this to make different character choices early vs. late in a link (e.g., `|` for display text, `]]` to close).

### Neuron 4: WWW Counter

A neuron that fires sharply right after the first `w` in `www` and then turns off. The model uses this to count how many `w`s it has emitted so far in the URL prefix, allowing it to know when to stop emitting `w`s and start the actual URL.

### What This Tells Us

The model was never told:

- "URLs have structure"
- "Wikilinks use double brackets"
- "Track quote depth"

It was told only: "predict the next character." Yet to do that well, it discovered that these are useful things to track. Quote-detection, URL-detection, bracket-counting — these emerged because they're informationally useful for the task.

> [!note] About 5% of neurons become interpretable Karpathy notes that the vast majority of neurons are doing something distributed and not easily interpretable as a single concept. But ~5% appear to track specific, clean features. This is consistent with the general picture of distributed representations in deep learning: most representation is in superposition across many neurons, but some clean features also crystallise in individual neurons.

---

## Why This Matters — The Deeper Lessons

### Lesson 1: Sequence Modelling Is Surprisingly Powerful

The character-level language model is arguably the simplest possible sequence task. No word-level tokenisation, no pre-trained representations, no linguistic features. Just: what character comes next? And yet models trained this way learn spelling, grammar, style, structure, and even domain-specific conventions (LaTeX environments, C syntax, Wikipedia markup) essentially from scratch.

This is evidence that **the sequence prediction objective is a remarkably powerful training signal**. If you want a model to understand language or code, having it predict the next character (or token) is a surprisingly effective way to teach it.

### Lesson 2: Structure Emerges Without Being Programmed

The bracket-closing behaviour in XML, the correct nesting of LaTeX environments, the syntactically valid C function signatures — none of these were hardcoded. The model discovered that following these structural rules is what the training data does, so following them minimises loss. **Syntax is learned because it's statistically regular, not because we programmed it in.**

This is the "end-to-end training" argument: if you have enough data and the right objective, structural knowledge will emerge. You don't need to hand-engineer it.

### Lesson 3: Failures Reveal the Architecture's Limits

The characteristic failure mode across all experiments is **long-range dependency**. The model opens a LaTeX environment and forgets to close it correctly. It declares a variable and never uses it. It uses a variable it never declared. In every case, the context needed to avoid the error is far earlier in the sequence than the error itself.

This tells us something important about the vanilla LSTM: its ability to maintain information over very long ranges is limited. The hidden state has finite capacity, and information from hundreds of time steps ago can be diluted or overwritten. This is exactly what motivated:

- Better architectures: attention mechanisms, Transformers
- Better regularisation: Layer Normalisation ([[LN_written_notes]])
- External memory: Neural Turing Machines

### Lesson 4: Scale Helps, But There Are Diminishing Returns

Karpathy observes that larger models and more training reduce the frequency of long-range errors (the LaTeX environment mismatches become less common with larger/better models). But they don't disappear. There's a fundamental limitation to what a fixed-size hidden state can memorise over arbitrarily long sequences, regardless of how it's trained.

---

## Connections to Broader Research — Where RNNs Were Going in 2015

Karpathy situates this work within a broader landscape, which in retrospect is a remarkable snapshot of the field just before the Transformer revolution:

### NLP and Speech

- Speech-to-text transcription (Graves et al., 2014)
- Machine translation (Sutskever et al., 2014 — the seq2seq paper)
- Handwritten text generation (Graves, 2013)
- Language modelling (Sutskever et al.; Graves; Mikolov)

### Computer Vision

- Video classification (Karpathy et al., 2014)
- Image captioning (Vinyals et al., 2014; Karpathy's own work)
- Video captioning
- Visual question answering

### The Attention Mechanism — Karpathy's Bold Prediction

> "The concept of **attention** is the most interesting recent architectural innovation in neural networks."

This is written in 2015. The Transformer paper "Attention is All You Need" would appear in 2017. Karpathy was right.

**Soft attention** (Bahdanau et al., 2015): instead of compressing the entire input sequence into a fixed-size hidden vector, the model at each decoding step can attend to a distribution over all input positions. The attention weights are differentiable, so the model learns _where to look_ rather than having to memorise everything into a fixed vector.

**Hard attention** (non-differentiable): rather than a soft weighted sum over all positions, sample a specific position to attend to. Requires reinforcement learning (REINFORCE) because sampling is non-differentiable. More efficient in principle but harder to train.

**Neural Turing Machines** (Graves et al., 2014): external memory arrays addressed by soft attention. The model can read from and write to arbitrary memory locations, with learned addressing mechanisms.

### The RNN Limitation That Attention Solved

Karpathy articulates the key problem clearly:

> "RNNs unnecessarily couple their representation size to the amount of computation per step. If you double the size of the hidden state vector you'd quadruple the amount of FLOPS at each step due to the matrix multiplication."

The Transformer directly solves this: attention allows the model to route information between arbitrary positions without the information passing through a bottleneck hidden state. A position at the end of a sequence can directly attend to a position at the beginning — no need to store that early information in the hidden state for hundreds of steps.

---

## Practical Setup — Hyperparameters and Training

For reference, Karpathy's char-rnn configurations:

### Paul Graham (~1MB text)

- Architecture: 2-layer LSTM, 512 hidden units
- Parameters: ~3.5 million
- Dropout: 0.5 after each layer
- Batch size: 100
- BPTT length: 100 characters (truncated backpropagation through time)
- Hardware: TITAN Z GPU
- Speed: ~0.46 seconds per batch

### Linux Source (474MB)

- Architecture: 3-layer LSTM
- Parameters: ~10 million
- Training time: several days on GPU

### General Guidance

**Dropout** ([[Dropout_written_notes]]) is used after each LSTM layer with $p_\text{keep} = 0.5$. This is standard — dropout prevents the model from memorising specific sequences and forces it to learn more robust representations. With only 1MB of text (Paul Graham essays), dropout is especially important to prevent overfitting.

**Optimiser:** RMSProp ([[Training Diagnostics and Optimizers#14. RMSProp|TDO §14]]) or Adam ([[Adam]]) for adaptive per-parameter learning rates. This matters because different parts of the RNN receive very different gradient magnitudes — the embedding weights for rare characters receive sparse gradients, while frequent characters receive dense gradients. Adaptive methods handle this naturally.

**Truncated BPTT:** Rather than backpropagating through the entire sequence (potentially millions of characters), truncate at 100 characters. This means the model can only explicitly learn dependencies spanning up to 100 characters in a single gradient update. But in practice, the hidden state carries information forward across truncation boundaries — so the effective receptive field is longer.

---

## The Minimal 100-Line Implementation

Karpathy provides a [100-line numpy implementation](https://gist.github.com/karpathy/d4dee566867f8291f086) that captures the essential ideas. The key operations:

```python
# Forward pass
hprev = np.zeros((hidden_size, 1))
for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1))
    xs[t][inputs[t]] = 1                    # one-hot encode input character
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) +
                    np.dot(Whh, hs[t-1]) + bh)  # hidden state update
    ys[t] = np.dot(Why, hs[t]) + by        # output logits
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # softmax probabilities
    loss += -np.log(ps[t][targets[t], 0])  # cross-entropy loss

# Backward pass (BPTT)
dhnext = np.zeros_like(hs[0])
for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1                    # softmax + cross-entropy gradient
    dWhy += np.dot(dy, hs[t].T)
    dh = np.dot(Why.T, dy) + dhnext       # gradient flows through hidden state
    dhraw = (1 - hs[t] * hs[t]) * dh     # tanh backward: (1 - tanh²)
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)         # gradient for previous hidden state
```

The $\tanh$ backward pass is worth noting: if $h = \tanh(z)$, then $dz = (1 - h^2) \cdot dh$. The gradient is scaled by $(1 - h^2)$, which equals 1 when $h=0$ (maximum gradient flow) and equals 0 when $h = \pm 1$ (saturated — no gradient). This is the vanishing gradient problem in concrete form: if the hidden state saturates (all values near $\pm 1$), gradients can't flow back through the tanh.

---

## Summary — What to Remember

### The Architecture

The vanilla RNN computes at each step: $$\mathbf{h}_t = \tanh\left(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t\right), \qquad \mathbf{y}_t = W_{hy}\mathbf{h}_t$$

Three weight matrices. Hidden state carries memory. Tanh keeps values bounded.

### The Task

Character-level language modelling: predict $p(x_{t+1} \mid x_1, \ldots, x_t)$ for every $t$. Train with cross-entropy loss, sample autoregressively at test time.

### What Emerges

|Dataset|What the model learns|
|---|---|
|English essays|Spelling, grammar, essay structure, citation style|
|Shakespeare|Play format, speaker turns, dramatic vocabulary|
|Wikipedia|Markdown, wikilinks, XML, structured formatting|
|LaTeX|Environment tags, math notation, theorem structure|
|Linux C code|Function signatures, pointer notation, C idioms, GNU license|
|Baby names|Phonological structure of names, name-like character distributions|

### The Key Failure Mode

**Long-range dependencies.** The model opens a structure and forgets what it opened. Opens `\begin{proof}`, closes with `\end{lemma}`. Uses variable `rw` that was never declared. Declares `int error` and never uses it. These failures happen because the relevant context is too far back in the sequence for the hidden state to retain it.

### The Key Discovery

Individual neurons in the hidden state develop interpretable functions: URL detection, bracket tracking, position encoding within spans, character counting in repeated sequences. These emerge from training on the prediction objective alone — they're learned because they're useful for predicting the next character.

### The Broader Implication

The prediction objective (what comes next?) is a remarkably powerful training signal. Models trained purely to predict the next character develop syntactic, structural, and stylistic knowledge emergently. This principle — sequence prediction as self-supervised pre-training — is the foundation of everything that came after: word-level language models, BERT, GPT, and modern large language models.

> [!note] Connection to LN and modern Transformers The failure mode identified here — long-range dependency — is exactly what the Transformer (using [[LN_written_notes#7. Layer Norm in the Transformer Architecture|Pre-LN]] and attention) was designed to fix. Instead of routing all information through a sequential hidden state bottleneck, attention allows arbitrary position-to-position information flow. The seq2seq RNN was the precursor; the Transformer was the fix.

---

## Quick Reference

|Quantity|Formula|
|---|---|
|Hidden state update (vanilla RNN)|$\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t)$|
|Output|$\mathbf{y}_t = W_{hy}\mathbf{h}_t$|
|Training loss|$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p(x_{t+1} \mid x_{\leq t})$|
|Tanh gradient|$\partial \tanh(z)/\partial z = 1 - \tanh^2(z)$|
|Softmax with temperature|$p_i = \exp(z_i/\tau) / \sum_j \exp(z_j/\tau)$|

|Hyperparameter|Karpathy's setting|
|---|---|
|Layers|2–3 LSTM layers|
|Hidden units|512 per layer|
|Dropout|0.5 after each layer|
|Batch size|100 sequences|
|BPTT length|100 characters|
|Optimiser|RMSProp or Adam|
|Temperature (sampling)|0.5–1.0 typical|

---

_Notes from Andrej Karpathy's blog post, May 2015 — Musaib_