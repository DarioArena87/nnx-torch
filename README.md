# NNX
## Modern Neural Network Layers for PyTorch
===============================================

High level pytorch layers to supplement the ones already available in ``torch.nn`` with:

* Modern attention backends (SDPA, FlexAttention, LinearAttention, RWKV)
* HuggingFace-style attention masks (1/True = real, 0/False = padding)
* FFN variants (SwiGLU, GeGLU, MoE)
* Normalization layers (RMSNorm, ScaleNorm, AdaptiveRMSNorm)
* Positional encodings (RoPE, ALiBi, Sinusoidal, Learned)
* Composable Transformer blocks and stacks

