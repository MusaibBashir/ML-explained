![Cover Image of https://gemini.google.com/app/b6cc7cd44d9dd048](https://www.gstatic.com/lamda/images/gemini_aurora_thumbnail_4g_e74822ff0ca4259beb718.png?auto=format&w=600&fm=auto)
  
# Adam Optimizer Advantages Explained - Google Gemini
Meet Gemini, Google's AI assistant. Get help with writing, planning, brainstorming and more. Experience the power of generative AI.

🌐 https://gemini.google.com/app/b6cc7cd44d9dd048
🔗 https://app.web-highlights.com/page/6a0c3473a74d376c6f06d539

**Tags**: -



**Highlights & Notes**

> ϵ is a tiny constant, like 10 
−5
 , added to the denominator to prevent a mathematically illegal division by zero

> y 
i
​
 =γ 
x
^
  
i
​
 +β

> one of the most important architectural innovations in the history of deep learning.

> deep neural networks by making them faster, more stable, and significantly less sensitive to how you initialize your weights.

> Deep neural networks suffer from a similar problem called Internal Covariate Shift.

> During training, the weights of the first layer are updated, which changes the distribution of the outputs of that layer. The second layer now receives inputs with a completely new mean and variance

> because the deeper layers are constantly trying to adapt to a shifting foundation

> forces the network to learn very slowly

> Batch Normalization solves this by forcibly anchoring the distribution of the activations at every layer.

> applied to a specific layer's activations just before (or immediately after) the non-linear activation function

> four distinct mathematical steps

> Calculate the Mini-batch Mean

> Mini-batch Variance

> Normalize

> If we just stopped at Step 3, we would be forcing every layer to have a mean of 0 and a variance of 1. This is actually a bad idea. For example, if you feed a mean-zero distribution into a Sigmoid function, you only hit the linear, middle part of the curve, completely destroying the network's non-linear power.

> two learnable parameters for every single neuron:

γ (Gamma): The scale parameter.

β (Beta): The shift parameter.

> The network learns γ and β through backpropagation just like it learns standard weights.

> It allows the network to restore the exact representation power it needs, but in a controlled, stable way.

> Batch Norm behaves differently during training vs. inference:

> During Training: It normalizes using the μ 
B
​
  and σ 
B
2
​
  of the current mini-batch.

> During Inference: It normalizes using a Population Moving Average. While the network is training, it quietly keeps a running average of all the means and variances it sees across all batches. When you deploy the model, it locks these moving averages in place and uses them as fixed constants to process single inputs.

> original paper blamed "Internal Covariate Shift,"

> true superpower of Batch Norm is that it smooths the loss landscape

> t also makes the network highly robust to bad initial weight setups, and acts as a slight regularizer (because the random selection of mini-batches adds a tiny bit of noise to the mean/variance calculations, slightly reducing overfitting).
