use std::fmt;

use crate::tensor::dtype::DType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    IncompatibleDimensions(String),
    InvalidBroadcast {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },
    InvalidSlice {
        dim: usize,
        start: usize,
        end: usize,
        size: usize,
    },
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    InvalidReshape {
        from: Vec<usize>,
        to: Vec<usize>,
    },
    NonContiguousError,
    EmptyTensor,
    DTypeMismatch {
        expected: DType,
        actual: DType,
    },
    DimensionOutOfBounds {
        dim: usize,
        rank: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {:?}, found {:?}", expected, actual)
            }
            Self::IncompatibleDimensions(msg) => {
                write!(f, "Incompatible dimensions: {}", msg)
            }
            Self::InvalidBroadcast { shape_a, shape_b } => {
                write!(
                    f,
                    "Cannot broadcast shapes {:?} and {:?}",
                    shape_a, shape_b
                )
            }
            Self::InvalidSlice { dim, start, end, size } => {
                write!(
                    f,
                    "Invalid slice on dim {}: [{}..{}] for dimension of size {}",
                    dim, start, end, size
                )
            }
            Self::IndexOutOfBounds { index, shape } => {
                write!(
                    f,
                    "Index {:?} out of bounds for tensor of shape {:?}",
                    index, shape
                )
            }
            Self::InvalidReshape { from, to } => {
                write!(
                    f,
                    "Cannot reshape tensor with total elements ({:?}) to target shape ({:?})",
                    from, to
                )
            }
            Self::NonContiguousError => {
                write!(
                    f,
                    "Operation requires contiguous tensor in memory; call .contiguous() first"
                )
            }
            Self::EmptyTensor => write!(f, "Tensor cannot be empty for this operation"),
            Self::DTypeMismatch { expected, actual } => {
                write!(
                    f,
                    "DType mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            Self::DimensionOutOfBounds { dim, rank } => {
                write!(
                    f,
                    "Dimension index {} is out of bounds for rank {}",
                    dim, rank
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}
