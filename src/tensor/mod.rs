pub mod core;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod shape;

pub use core::Tensor;
pub use dtype::DType;
pub use error::TensorError;
pub use ops::{
    add, add_scalar, argmax, div, gelu, matmul, matmul_naive, matmul_parallel, matmul_tiled, max,
    mean, mul, rms_norm, scale, silu, softmax, sub, sum,
};
pub use shape::Shape;
