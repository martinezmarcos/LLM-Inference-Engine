use std::fmt;

use crate::tensor::error::TensorError;

/// Multi-dimensional shape descriptor for tensors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new Shape from a slice of dimensions.
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    /// Creates a scalar (rank-0) shape.
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Returns the dimensions slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1 // Scalar has 1 element
        } else {
            self.dims.iter().product()
        }
    }

    /// Computes default row-major (C-contiguous) strides.
    ///
    /// For shape `[d_0, d_1, ..., d_{n-1}]`:
    /// * `stride_{n-1} = 1`
    /// * `stride_k = stride_{k+1} * d_{k+1}`
    pub fn default_strides(&self) -> Vec<usize> {
        let rank = self.rank();
        if rank == 0 {
            return Vec::new();
        }

        let mut strides = vec![1; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Computes the linear 1D memory offset for a multi-dimensional coordinate index.
    pub fn compute_offset(
        &self,
        indices: &[usize],
        strides: &[usize],
    ) -> Result<usize, TensorError> {
        if indices.len() != self.rank() {
            return Err(TensorError::IncompatibleDimensions(format!(
                "Index rank {} does not match tensor rank {}",
                indices.len(),
                self.rank()
            )));
        }

        let mut offset = 0;
        for (i, (&idx, &dim)) in indices.iter().zip(self.dims.iter()).enumerate() {
            if idx >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    index: indices.to_vec(),
                    shape: self.dims.clone(),
                });
            }
            offset += idx * strides[i];
        }

        Ok(offset)
    }

    /// Broadcasts two shapes together according to standard multi-dimensional broadcasting rules:
    /// 1. Shapes are aligned from the right (innermost dimensions).
    /// 2. Two dimensions are compatible if they are equal, or if one of them is 1.
    /// 3. Missing leading dimensions are treated as 1.
    pub fn broadcast_shapes(s1: &Shape, s2: &Shape) -> Result<Shape, TensorError> {
        let rank1 = s1.rank();
        let rank2 = s2.rank();
        let max_rank = rank1.max(rank2);

        let mut out_dims = vec![0; max_rank];

        for i in 0..max_rank {
            let dim1 = if i < rank1 {
                s1.dims[rank1 - 1 - i]
            } else {
                1
            };
            let dim2 = if i < rank2 {
                s2.dims[rank2 - 1 - i]
            } else {
                1
            };

            if dim1 == dim2 || dim2 == 1 {
                out_dims[max_rank - 1 - i] = dim1;
            } else if dim1 == 1 {
                out_dims[max_rank - 1 - i] = dim2;
            } else {
                return Err(TensorError::InvalidBroadcast {
                    shape_a: s1.dims.clone(),
                    shape_b: s2.dims.clone(),
                });
            }
        }

        Ok(Shape::new(&out_dims))
    }

    /// Computes strides for a shape when virtually broadcasted to `target_shape`.
    /// Dimensions that are broadcast from size 1 to $D > 1$ receive a stride of 0,
    /// enabling zero-copy broadcast indexing.
    pub fn broadcast_strides(
        &self,
        current_strides: &[usize],
        target_shape: &Shape,
    ) -> Result<Vec<usize>, TensorError> {
        let current_rank = self.rank();
        let target_rank = target_shape.rank();

        if current_rank > target_rank {
            return Err(TensorError::InvalidBroadcast {
                shape_a: self.dims.clone(),
                shape_b: target_shape.dims.clone(),
            });
        }

        let mut new_strides = vec![0; target_rank];
        let rank_diff = target_rank - current_rank;

        for (i, &current_stride) in current_strides.iter().enumerate().take(current_rank) {
            let target_idx = i + rank_diff;
            let current_dim = self.dims[i];
            let target_dim = target_shape.dims[target_idx];

            if current_dim == target_dim {
                new_strides[target_idx] = current_stride;
            } else if current_dim == 1 {
                new_strides[target_idx] = 0; // Stride 0 allows repeating the single element
            } else {
                return Err(TensorError::InvalidBroadcast {
                    shape_a: self.dims.clone(),
                    shape_b: target_shape.dims.clone(),
                });
            }
        }

        Ok(new_strides)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape::new(&dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape { dims }
    }
}
