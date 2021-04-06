# Basic Transformer

The original Paper: [Vasvani, A. et al. Attention is all you need.](https://arxiv.org/pdf/1706.03762.pdf)

## Basic Information
- This requires Prior Knowledge of text generation and attention.
- Transformers supports variable sized inputs using stacks of `self-attention` layers instead on RNNs and CNNs.

## Pros & Cons

- Pros:
    - No assumptions about the spacial/temporal relationships across the data.
    - Layer Outputs can be calculated in Parallel.
    - Distant items can be affect each other's  output without passing through  many RNN-Steps or convolutional layers.
    - It can learn long range dependencies.
- Cons
    - For time-series, the Output is calculated by entire history, instead of just the input or current hidden state.
    - If the output doesn't have temporal/spatial relationships, some positional encoding is needed or else the model will treat it as like a Bag of Words.

## What does the code use?
- Tensorflow Datasets a.k.a TFDS.
- Tensorflow Text.
- Tensorflow Keras.

## Steps performed in the Notebook.
- Get tokenizers.
    - Go through all the inherited Methods.
    
- Create an Input Pipeline,
    - Uses `tf.data.Dataset`
- Create Positional Encoding.

<img src="https://latex.codecogs.com/gif.latex?O_t=pos(even) = \sin(pos/10000^{2i/d_{model}})" /> 
<img src="https://latex.codecogs.com/gif.latex?O_t=pos(odd) = \cos(pos/10000^{2i/d_{model}}" /> 

- Create Masks and Look Ahead Masks.
- Scaled Dot Product Attention.
<img src="https://latex.codecogs.com/gif.latex?O_t=Attention(Q, K, V) = softmax_{k}\dfrac{QK}{\sqrt{d_k}} V" /> 

- Using Multi-Head Attention.
- Create Encoder Layer.
    - Multi-Head Attention (with padding masks).
    - Point-wise feed forward network.
    - Output of each sublayer is `LayerNorm(x + sublayer(x))`.
- Create Decoder Layer.
    - Multi-Head attention (with Look Ahead Mask and Padding Mask)
    - Multi-Head Attention (with padding mask)
        - Value (V) and Key (K) receive encoder outputs as inputs. Query (Q) receives the output from the masked multi-head attention sublayer.
    - Point-wise Feed Forward network.
- Create an Encoder.
    - Input Embedding.
    - Positional Encoding.
    - N - encoder layers.
- Create a Decoder.
    - Output Embedding.
    - Positional Encoding.
    - N - decoded layers.
- Create a Transformer.
    - Input Embedding and Output Embedding.
    - Positional Encoding.
    - Encoder and Decoder with n layers.
    - Feed Forward.
- Optimizer.
    - Custom LR Scheduler with Adam Optimizer.
<img src="https://latex.codecogs.com/gif.latex?O_t=l_{rate} = d_{model}^{-0.5} \times min(\text{step_num}^{-0.5}, \text{step_num} \times \text{warmup_steps}^{-1.5})" /> 

Ref. [Paper](https://arxiv.org/pdf/1706.03762.pdf)

- Loss & Metrics
    - Loss used `SparseCategoricalCrossEntropy`.
    - Handle padded values as a value `0`, using a mask.
    - Metrics for Accuracy and Loss is `Mean`.

- Training and Checkpoint.
- Evaluate
- Plot Attention Heads and Weights.
