//! # mini-llm
//!
//! A minimalist, educational, from-scratch LLM inference engine in Rust.
//!
//! Designed for systems engineering clarity, low-level efficiency, and zero ML-framework bloat.

pub mod cache;
pub mod inference;
pub mod memory;
pub mod model;
pub mod quantization;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;

pub use cache::{KvCache, KvCacheError, LayerKvCache};
pub use inference::{InferenceEngine, InferenceError};
pub use memory::{DoubleBuffer, MemoryError, ScratchArena};
pub use model::{GgufFile, ModelConfig, TransformerWeights};
pub use quantization::{
    dequantize_q4_0, dequantize_q8_0, evaluate_quantization_error, quantize_q4_0, quantize_q8_0,
    vec_dot_q8_0,
};
pub use sampling::{ConfigurableSampler, GreedySampler, Sampler};
pub use tensor::{DType, Shape, Tensor, TensorError};
pub use tokenizer::{BpeTokenizer, TokenizerError};
pub use transformer::TransformerDecoder;
