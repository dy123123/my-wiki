# Introduction to the Transformer Architecture

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Institution:** Google Brain / Google Research  
**Published:** 2017-06-12 (arXiv preprint)

---

## Abstract

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

## 1. Introduction

Recurrent neural networks (RNNs), long short-term memory networks (LSTMs), and gated recurrent networks have long been established as state-of-the-art approaches in sequence modeling and transduction problems such as language modeling and machine translation.

The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

## 2. Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure. The encoder maps an input sequence to a sequence of continuous representations. Given those representations, the decoder then generates an output sequence one element at a time.

### 2.1 Encoder and Decoder Stacks

**Encoder:** Composed of N=6 identical layers, each with two sub-layers:
1. Multi-head self-attention mechanism
2. Position-wise fully connected feed-forward network

**Decoder:** Also composed of N=6 identical layers, with three sub-layers:
1. Masked multi-head self-attention
2. Multi-head attention over encoder output
3. Position-wise feed-forward network

### 2.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**Multi-Head Attention** allows the model to jointly attend to information from different representation subspaces at different positions.

### 2.3 Positional Encoding

Since the model contains no recurrence or convolution, positional encodings are added to give the model information about the relative or absolute position of tokens in the sequence.

## 3. Why Self-Attention?

Key advantages of self-attention over recurrent layers:
- **Total computational complexity per layer:** O(n²·d) vs O(n·d²)
- **Sequential operations:** O(1) vs O(n) — highly parallelizable
- **Maximum path length:** O(1) vs O(n) — better for long-range dependencies

## 4. Training

The model was trained on the WMT 2014 English-German dataset (4.5M sentence pairs) and WMT 2014 English-French dataset (36M sentence pairs).

### 4.1 Optimizer
Adam optimizer with β₁=0.9, β₂=0.98, ε=10⁻⁹.

### 4.2 Results

| Model | EN-DE BLEU | EN-FR BLEU |
|-------|-----------|-----------|
| Transformer (big) | 28.4 | 41.0 |
| Previous SOTA | 26.4 | 40.5 |

## 5. Conclusion

The Transformer, the first sequence transduction model based entirely on attention, replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.

---

*This is a mock/simplified summary for demonstration purposes. Based on "Attention Is All You Need" (Vaswani et al., 2017).*
