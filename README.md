# mini-llm

A small, educational, and high-performance LLM inference engine implemented from scratch in Rust.

`mini-llm` is built without machine learning frameworks (no PyTorch, no Candle, no ONNX, no external APIs). It implements the complete low-level systems infrastructure required to load modern Transformer model weights and generate text token-by-token on CPU.

## Features

- **From-Scratch Tensor Engine**: Multi-dimensional contiguous and strided tensor operations (`matmul`, `softmax`, `rms_norm`, `rope`, slicing, broadcasting).
- **Zero Framework Bloat**: Pure Rust core linear algebra and numerical primitives.
- **Cache-Aware Performance**: Memory-layout-conscious kernel design (row-major memory access, loop reordering, pre-allocated scratch buffers).
- **GGUF Format Support**: Custom parser for binary headers, metadata key-values, and tensor descriptors with memory mapping (`mmap`).
- **Modern Transformer Architecture**: Decoder-only Transformer supporting RoPE (Rotary Positional Embeddings), RMSNorm, Multi-Head / Grouped-Query Attention (GQA), and SwiGLU / MLP blocks.
- **Autoregressive KV Cache**: Constant-time key-value cache preventing quadratic recomputation during decoding.
- **Flexible Sampling**: Greedy decoding, Temperature scaling, Top-K, and Top-P (nucleus) sampling.
- **Quantization Support**: Block-quantized weight representations (INT8 / INT4) with specialized dot products.

## Architecture

The engine follows the standard decoder-only inference pipeline:

1. **Model Loader (`src/model`)**: Maps GGUF files into memory using `mmap`, parses metadata, and loads tensor weights (FP32, FP16, Q8_0, Q4_0).
2. **Tokenizer (`src/tokenizer`)**: Converts prompt text into token IDs via Byte-Pair Encoding (BPE), handling byte-level merges and spaces.
3. **Tensor Runtime (`src/tensor`)**: Manages multidimensional arrays, contiguous/strided views, broadcasting, and core operations (`matmul`, `rms_norm`, `softmax`).
4. **Transformer Decoder (`src/transformer`)**:
   - **Embedding Lookup**: Maps token IDs to hidden states.
   - **Transformer Blocks**: Layers with Pre-RMSNorm, Causal Multi-Head / Grouped-Query Attention (GQA), RoPE positional embeddings, and SwiGLU feed-forward networks with residual connections.
   - **KV Cache (`src/cache`)**: Stores key and value states across steps to avoid recomputing previous tokens.
   - **LM Head**: Projects normalized hidden states to vocabulary logits.
5. **Sampler (`src/sampling`)**: Selects the next token applying temperature scaling, Top-K, and Top-P (nucleus) filtering.
6. **Inference Engine (`src/inference`)**: Orchestrates prompt prefill and the autoregressive token-by-token generation loop.

## Supported Model Format

- Format: **GGUF** (v2 and v3)
- Models: LLaMA family (LLaMA 2/3, TinyLlama, SmolLM), Mistral, and compatible decoder-only architectures.
- Precision: FP32, FP16, and block-quantized Q8_0 / Q4_0 formats.

## Build

Requires a modern Rust toolchain (Rust 1.80+):

```bash
# Debug build
cargo build

# Optimized release build
cargo build --release

# Run comprehensive test suite
cargo test
```

## Usage

Generate text using the standalone CLI:

```bash
cargo run --release -- \
    --model models/tinyllama-1.1b.gguf \
    --prompt "The future of artificial intelligence is" \
    --max-tokens 100 \
    --temperature 0.7 \
    --top-p 0.9
```

## Design Decisions

1. **Zero-Allocation Hot Path**: The autoregressive decoding loop reuses pre-allocated scratch buffers. Token generation does not invoke heap allocations (`malloc`), eliminating heap fragmentation and lock contention.
2. **Explicit Strided Views**: Tensor slicing and transposition operate via strides and offsets without copying memory unless `.contiguous()` is explicitly requested.
3. **Numerically Stable Kernels**: Softmax and RMSNorm implement normalization and max-subtraction to avoid IEEE 754 floating-point underflow/overflow.
4. **Safety without Garbage Collection**: Leveraging Rust's ownership model and RAII to guarantee memory safety and data-race freedom across threads without runtime overhead.

## Limitations

- Optimized primarily for CPU execution (SIMD / multi-threading); GPU compute backends (Vulkan/Metal/CUDA) are intentionally deferred.
- Training and backpropagation are not supported; the runtime is purely dedicated to forward-pass inference.
