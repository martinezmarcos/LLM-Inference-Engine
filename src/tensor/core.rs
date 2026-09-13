use std::fmt;
use std::sync::Arc;

use crate::tensor::dtype::DType;
use crate::tensor::error::TensorError;
use crate::tensor::shape::Shape;

/// Core multi-dimensional Tensor structure.
///
/// Data is backed by an `Arc<Vec<f32>>` to allow zero-copy views
/// (such as slicing, transposing, and reshaping) without unnecessary allocations.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
    dtype: DType,
    data: Arc<Vec<f32>>,
}

impl Tensor {
    /// Constructs a Tensor directly from its raw constituent parts (used by memory arenas and views).
    pub fn from_raw_parts(
        shape: Shape,
        strides: Vec<usize>,
        offset: usize,
        dtype: DType,
        data: Arc<Vec<f32>>,
    ) -> Self {
        Self {
            shape,
            strides,
            offset,
            dtype,
            data,
        }
    }

    /// Creates a tensor from an owned vector of f32 data and a given shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self, TensorError> {
        let shape = Shape::new(shape);
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                actual: vec![data.len()],
            });
        }
        let strides = shape.default_strides();
        Ok(Self {
            shape,
            strides,
            offset: 0,
            dtype: DType::F32,
            data: Arc::new(data),
        })
    }

    /// Creates a tensor by copying data from a slice.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self, TensorError> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let strides = shape_obj.default_strides();
        Self {
            shape: shape_obj,
            strides,
            offset: 0,
            dtype: DType::F32,
            data: Arc::new(vec![0.0; numel]),
        }
    }

    /// Creates a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let strides = shape_obj.default_strides();
        Self {
            shape: shape_obj,
            strides,
            offset: 0,
            dtype: DType::F32,
            data: Arc::new(vec![1.0; numel]),
        }
    }

    /// Creates a scalar tensor (rank 0).
    pub fn scalar(val: f32) -> Self {
        Self {
            shape: Shape::scalar(),
            strides: Vec::new(),
            offset: 0,
            dtype: DType::F32,
            data: Arc::new(vec![val]),
        }
    }

    /// Creates a tensor filled with pseudo-random numbers in range `[-1.0, 1.0]`
    /// using a deterministic SplitMix64 / XorShift generator (zero external dependencies).
    pub fn randn(shape: &[usize], seed: u64) -> Self {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let strides = shape_obj.default_strides();

        let mut state = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        let mut data = Vec::with_capacity(numel);

        for _ in 0..numel {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let val = ((z as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            data.push(val);
        }

        Self {
            shape: shape_obj,
            strides,
            offset: 0,
            dtype: DType::F32,
            data: Arc::new(data),
        }
    }

    /// Returns a reference to the tensor's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the dimensions slice.
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Returns the tensor rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Returns the strides vector.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the memory offset of this tensor view.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Checks if the tensor is stored in contiguous C-order (row-major) memory.
    pub fn is_contiguous(&self) -> bool {
        if self.rank() == 0 {
            return true;
        }
        let default_strides = self.shape.default_strides();
        self.strides == default_strides
    }

    /// Ensures the tensor data is contiguous in memory.
    /// If it is already contiguous, this is a zero-copy clone (clones Arc pointer).
    /// If strided/discontiguous, it allocates a new contiguous buffer and packs elements.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 && self.data.len() == self.numel() {
            return self.clone();
        }

        let numel = self.numel();
        let mut new_data = Vec::with_capacity(numel);

        let rank = self.rank();
        if rank == 0 {
            new_data.push(self.data[self.offset]);
        } else {
            let mut coord = vec![0; rank];
            for _ in 0..numel {
                let mut off = self.offset;
                for (i, &c) in coord.iter().enumerate().take(rank) {
                    off += c * self.strides[i];
                }
                new_data.push(self.data[off]);

                // Increment coordinate
                for dim in (0..rank).rev() {
                    coord[dim] += 1;
                    if coord[dim] < self.shape.dims()[dim] {
                        break;
                    }
                    coord[dim] = 0;
                }
            }
        }

        let shape = self.shape.clone();
        let strides = shape.default_strides();
        Self {
            shape,
            strides,
            offset: 0,
            dtype: self.dtype,
            data: Arc::new(new_data),
        }
    }

    /// Returns a direct slice to the data if contiguous and aligned.
    pub fn as_slice(&self) -> Result<&[f32], TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NonContiguousError);
        }
        let end = self.offset + self.numel();
        Ok(&self.data[self.offset..end])
    }

    /// Returns the data copied into a flat `Vec<f32>` in contiguous order.
    pub fn to_vec(&self) -> Vec<f32> {
        if self.is_contiguous() {
            let end = self.offset + self.numel();
            self.data[self.offset..end].to_vec()
        } else {
            self.contiguous().to_vec()
        }
    }

    /// Reshapes the tensor to a new shape with the same total number of elements.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let new_shape_obj = Shape::new(new_shape);
        if new_shape_obj.numel() != self.numel() {
            return Err(TensorError::InvalidReshape {
                from: self.shape.dims().to_vec(),
                to: new_shape.to_vec(),
            });
        }

        let tensor = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };

        let new_strides = new_shape_obj.default_strides();
        Ok(Self {
            shape: new_shape_obj,
            strides: new_strides,
            offset: tensor.offset,
            dtype: tensor.dtype,
            data: tensor.data,
        })
    }

    /// Transposes two dimensions. Zero-copy O(1).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        let rank = self.rank();
        if dim0 >= rank || dim1 >= rank {
            return Err(TensorError::DimensionOutOfBounds {
                dim: dim0.max(dim1),
                rank,
            });
        }

        if dim0 == dim1 {
            return Ok(self.clone());
        }

        let mut new_dims = self.shape.dims().to_vec();
        let mut new_strides = self.strides.clone();

        new_dims.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Self {
            shape: Shape::new(&new_dims),
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            data: Arc::clone(&self.data),
        })
    }

    /// 2D Matrix transpose convenience shortcut.
    pub fn t(&self) -> Result<Self, TensorError> {
        if self.rank() < 2 {
            return Err(TensorError::IncompatibleDimensions(format!(
                "Transpose requires at least 2 dimensions, found rank {}",
                self.rank()
            )));
        }
        let rank = self.rank();
        self.transpose(rank - 2, rank - 1)
    }

    /// Creates a slice view along a specified dimension `[start..end]`. Zero-copy O(1).
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Self, TensorError> {
        let rank = self.rank();
        if dim >= rank {
            return Err(TensorError::DimensionOutOfBounds { dim, rank });
        }

        let dim_size = self.shape.dims()[dim];
        if start > end || end > dim_size {
            return Err(TensorError::InvalidSlice {
                dim,
                start,
                end,
                size: dim_size,
            });
        }

        let new_offset = self.offset + start * self.strides[dim];
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[dim] = end - start;

        Ok(Self {
            shape: Shape::new(&new_dims),
            strides: self.strides.clone(),
            offset: new_offset,
            dtype: self.dtype,
            data: Arc::clone(&self.data),
        })
    }

    /// Reads a single scalar element at given coordinates.
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let offset = self.shape.compute_offset(indices, &self.strides)?;
        Ok(self.data[self.offset + offset])
    }

    /// Sets a single scalar element at given coordinates with COW semantics.
    pub fn set(&mut self, indices: &[usize], val: f32) -> Result<(), TensorError> {
        let offset = self.shape.compute_offset(indices, &self.strides)?;
        let data_mut = Arc::make_mut(&mut self.data);
        data_mut[self.offset + offset] = val;
        Ok(())
    }

    /// Internal raw data pointer for low-level kernels.
    pub(crate) fn raw_data(&self) -> &Arc<Vec<f32>> {
        &self.data
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={}, strides={:?}, offset={}, dtype={})",
            self.shape, self.strides, self.offset, self.dtype
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor(shape={}, dtype={}):", self.shape, self.dtype)?;
        let rank = self.rank();
        if rank == 0 {
            writeln!(f, "  {}", self.data[self.offset])
        } else if rank == 1 {
            write!(f, "  [")?;
            for i in 0..self.shape.dims()[0] {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", self.get(&[i]).unwrap())?;
            }
            writeln!(f, "]")
        } else if rank == 2 {
            let (rows, cols) = (self.shape.dims()[0], self.shape.dims()[1]);
            for r in 0..rows {
                write!(f, "  [")?;
                for c in 0..cols {
                    if c > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", self.get(&[r, c]).unwrap())?;
                }
                writeln!(f, "]")?;
            }
            Ok(())
        } else {
            let numel_to_print = self.numel().min(16);
            let flat = self.to_vec();
            write!(f, "  [")?;
            for (i, &val) in flat.iter().enumerate().take(numel_to_print) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val)?;
            }
            if self.numel() > 16 {
                write!(f, ", ... ({} total elements)", self.numel())?;
            }
            writeln!(f, "]")
        }
    }
}
