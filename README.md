# NNX-Torch

[![PyPI version](https://badge.fury.io/py/nnx-torch.svg)](https://badge.fury.io/py/nnx-torch)
[![Python](https://img.shields.io/pypi/pyversions/nnx-torch.svg)](https://pypi.org/project/nnx-torch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/Status-Experimental-orange.svg)](https://pypi.org/project/nnx-torch/)

**Modern, composable PyTorch layers with pluggable attention backends and HuggingFace-style masking.**

---

## ⚠️ Experimental Status Warning

> **Warning**: This project is in an **experimental/alpha stage** (v0.1.0).
>
> - **APIs may change without notice** between versions
> - **Breaking changes** are expected as the library evolves
> - **Not production-ready** - use at your own risk
> - Documentation is still being developed
> - Some features may be incomplete or have bugs
>
> If you find this project useful, consider pinning to a specific version in your dependencies. Feedback, bug reports,
> and contributions are welcome!

---

## 📖 Overview

NNX-Torch provides high-level PyTorch layers to supplement those available in `torch.nn`, focusing on modern transformer
architectures and attention mechanisms. The library is designed to be:

- **Composable**: Mix and match attention backends, normalization layers, and positional encodings
- **Flexible**: Use only what you need with minimal mandatory dependencies
- **Compatible**: HuggingFace-style attention masks (`1/True` = real tokens, `0/False` = padding)
- **Modern**: Support for cutting-edge attention implementations

### Features

| Category                 | Components                                                    |
|--------------------------|---------------------------------------------------------------|
| **Attention Backends**   | SDPA (PyTorch native), FlexAttention, LinearAttention, RWKV   |
| **Attention Masks**      | HuggingFace-style masking conventions                         |
| **FFN Variants**         | SwiGLU, GeGLU, Mixture of Experts (MoE)                       |
| **Normalization**        | RMSNorm, ScaleNorm, AdaptiveRMSNorm                           |
| **Positional Encodings** | RoPE (Rotary Position Embeddings), ALiBi, Sinusoidal, Learned |
| **Transformer Blocks**   | Composable blocks and stacks                                  |

---

## 📋 Requirements

- **Python**: >= 3.10
- **PyTorch**: >= 2.1.0 (minimum for core functionality)

### Optional Requirements by Feature

| Feature                                         | Extra    | Requirements                                                       |
|-------------------------------------------------|----------|--------------------------------------------------------------------|
| FlexAttention                                   | `flex`   | PyTorch >= 2.3.0                                                   |
| Linear Attention (GLA, DeltaNet, Based, RetNet) | `linear` | flash-linear-attention >= 0.3.0                                    |
| Triton kernels                                  | `triton` | triton >= 2.2.0                                                    |
| Full CUDA acceleration                          | `cuda`   | PyTorch >= 2.3.0, triton >= 2.2.0, flash-linear-attention >= 0.3.0 |
| Development tools                               | `dev`    | pytest >= 8.0, pytest-cov >= 5.0, ruff >= 0.4, mypy >= 1.10        |
| Everything                                      | `all`    | All optional dependencies                                          |

---

## 📦 Installation

This project is managed with [Astral uv](https://docs.astral.sh/uv/), a fast Python package manager.

### Basic Installation

The minimal installation includes only PyTorch and the core NNX layers:

```bash
# Using uv (recommended)
uv add nnx-torch

# Using pip
pip install nnx-torch
```

### Installation with Optional Dependencies

Choose the extras based on your use case (replace `uv add <package>` with `pip install <package>` if needed) :

```bash
# For FlexAttention support (requires PyTorch 2.3+)
uv add nnx-torch[flex]

# For Linear Attention backends (GLA, DeltaNet, etc.)
uv add nnx-torch[linear]

# For Triton kernel support
uv add nnx-torch[triton]

# Full CUDA-accelerated stack
uv add nnx-torch[cuda]

# Development tools (testing, linting, type checking)
uv add nnx-torch[dev]

# Install everything (all optional dependencies)
uv add nnx-torch[all]
```

### CUDA PyTorch Installation

If you need CUDA-enabled PyTorch, use the PyTorch index:

```bash
# Install with CUDA 12.1 support
uv add nnx-torch[cuda] --extra-index-url https://download.pytorch.org/whl/cu121

# Install with CUDA 11.8 support
uv add nnx-torch[cuda] --extra-index-url https://download.pytorch.org/whl/cu118
```

### Development Installation

For contributing to the project:

```bash
# Clone the repository
git clone https://github.com/yourname/nnx-torch.git
cd nnx-torch

# Install with development dependencies using uv
uv sync --extra dev --extra all
```

---

## 🚀 Quick Start

```python
import torch
from nnx.layers import TransformerBlock, RMSNorm
from nnx.attention import SDPA

# Create a transformer block with SDPA attention
block = TransformerBlock(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    attention=SDPA(),
    norm=RMSNorm(512),
)

# Forward pass with HuggingFace-style attention mask
x = torch.randn(2, 128, 512)  # (batch, seq_len, d_model)
mask = torch.ones(2, 128)  # 1 = real token, 0 = padding

output = block(x, attention_mask=mask)
```

---

## ⚡ Performance Optimizations

NNX-Torch includes a comprehensive set of performance optimizations for training and inference of transformer models.

### Grouped Query Attention (GQA)

Reduce KV cache memory by sharing KV heads across groups of query heads. Supports MHA, GQA, and MQA configurations.

```python
from nnx.attention import SDPAttention

# Standard Multi-Head Attention (MHA)
attn = SDPAttention(512, num_heads=8)

# Grouped Query Attention (GQA) — 4 query heads share 2 KV heads
attn = SDPAttention(512, num_heads=4, num_key_value_heads=2)

# Multi-Query Attention (MQA) — all query heads share 1 KV head
attn = SDPAttention(512, num_heads=4, num_key_value_heads=1)
```

### Packed Projections

Combine multiple linear projections into a single `nn.Linear` to reduce kernel launch overhead and improve memory
bandwidth.

```python
from nnx.attention import SDPAttention
from nnx.layers import GatedFFN

# Packed QKV projections in attention (single linear for Q, K, V)
attn = SDPAttention(512, num_heads=8)

# Packed gate+up projections in SwiGLU/GeGLU FFN
ffn = GatedFFN(512)
```

### KV Cache for Autoregressive Generation

Cache key/value tensors across forward passes for efficient token-by-token generation.

```python
from nnx.attention import SDPAttention

attn = SDPAttention(512, num_heads=8, use_cache=True)

# First pass
out1 = attn(x, use_cache=True)
past_kv = out1.past_key_value

# Subsequent passes with cached KV
out2 = attn(new_token, past_key_value=past_kv, use_cache=True)
```

### Gradient Checkpointing

Trade ~20-30% extra compute for ~50-70% reduction in activation memory during training.

```python
from nnx.layers import TransformerLayer, TransformerStack, GatedFFN

# Per-layer checkpointing
layer = TransformerLayer(512, num_heads=8, gradient_checkpointing=True)

# Stack-level checkpointing
stack = TransformerStack(
    n_layers=12, embed_dim=512, num_heads=8,
    gradient_checkpointing=True,
)

# FFN-level checkpointing
ffn = GatedFFN(512, gradient_checkpointing=True)
```

### NestedTensor for Variable-Length Sequences

Process batches of variable-length sequences without padding overhead using PyTorch's `torch.nested.nested_tensor`.

```python
from nnx.layers import TransformerStack

stack = TransformerStack(
    n_layers=6, embed_dim=512, num_heads=8,
    use_nested_tensor=True,  # Enable NestedTensor processing
)

# Variable-length sequences (mask: 1=real, 0=padding)
x = torch.randn(4, 128, 512)
mask = torch.tensor(
    [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0],
    ], dtype=torch.long
)

output = stack(x, attention_mask=mask)
```

### Causal Attention Dispatch

When `causal=True` and no attention bias is provided, NNX-Torch passes `is_causal=True` directly to PyTorch's SDPA,
enabling optimized causal attention kernels (~15% speedup).

```python
from nnx.layers import TransformerLayer

layer = TransformerLayer(512, num_heads=8, causal=True)
# Automatically uses native causal attention in SDPA
output = layer(x)
```

### Multiple Attention Backends

Choose the optimal attention backend for your workload:

| Backend                          | Complexity     | Best For                                           |
|----------------------------------|----------------|----------------------------------------------------|
| **SDPA**                         | O(N²)          | General purpose, auto-dispatches to FlashAttention |
| **FlexAttention**                | O(N²)          | Custom attention patterns (sliding window, sparse) |
| **Linear (GLA, DeltaNet, etc.)** | O(N)           | Long context, memory-constrained environments      |
| **RWKV**                         | O(1) inference | Streaming, real-time applications                  |

### Normalization Optimizations

| Layer               | Use Case                                                         |
|---------------------|------------------------------------------------------------------|
| **RMSNorm**         | Default for modern LLMs (LLaMA, Mistral) — faster than LayerNorm |
| **ScaleNorm**       | Parameter-efficient alternative                                  |
| **CosineNorm**      | Normalized output for stable training                            |
| **AdaptiveRMSNorm** | Conditional normalization for control networks                   |

### Performance Comparison

Expected improvements relative to a naive transformer implementation:

| Optimization           | Memory Savings         | Speed Improvement      |
|------------------------|------------------------|------------------------|
| GQA (4:2)              | ~50% KV cache          | ~10-20%                |
| Packed QKV             | —                      | ~5-10%                 |
| KV Cache (inference)   | ~50-70% per token      | ~3-5x throughput       |
| Gradient Checkpointing | ~50-70% activations    | -20-30% (trade-off)    |
| NestedTensor           | ~30-50% (variable len) | ~2-4x (padded batches) |
| Native Causal SDPA     | —                      | ~15%                   |
| RMSNorm vs LayerNorm   | ~25% params            | ~10-15%                |

---

## 🏗️ Project Structure

```
nnx-torch/
├── nnx/
│   ├── attention/       # Attention backends (SDPA, Flex, Linear, RWKV)
│   ├── layers/          # Building blocks (embeddings, FFN, norms, transformers)
│   └── utils/           # Utilities (masking, etc.)
├── tests/               # Test suite
├── pyproject.toml       # Project configuration
└── README.md
```

---

## 🧪 Testing

Run the test suite using pytest:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=nnx --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Since the project is in early development, please consider:

1. Opening an issue to discuss major changes before implementing
2. Following the existing code style (enforced by ruff)
3. Adding tests for new functionality
4. Keeping the minimal dependency principle

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
