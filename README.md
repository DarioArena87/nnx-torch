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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References & Acknowledgments

- [PyTorch](https://pytorch.org/) - The foundational deep learning framework
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) - Efficient linear attention
  implementations
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - For attention mask conventions

---

**Note**: This project is independently developed and is not affiliated with PyTorch or HuggingFace.
