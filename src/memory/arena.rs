use std::fmt;
use std::sync::Arc;
use crate::tensor::{DType, Shape, Tensor, TensorError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    OutOfMemory {
        requested_elements: usize,
        available_elements: usize,
    },
    TensorErr(TensorError),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory {
                requested_elements,
                available_elements,
            } => write!(
                f,
                "ScratchArena OOM: requested {} elements, but only {} remain",
                requested_elements, available_elements
            ),
            Self::TensorErr(e) => write!(f, "Tensor error in memory pool: {}", e),
        }
    }
}

impl std::error::Error for MemoryError {}

impl From<TensorError> for MemoryError {
    fn from(e: TensorError) -> Self {
        Self::TensorErr(e)
    }
}

#[derive(Debug)]
pub struct ScratchArena {
    storage: Arc<Vec<f32>>,
    capacity: usize,
    offset: usize,
    peak_offset: usize,
}

impl ScratchArena {
    pub fn new(capacity_elements: usize) -> Self {
        Self {
            storage: Arc::new(vec![0.0f32; capacity_elements]),
            capacity: capacity_elements,
            offset: 0,
            peak_offset: 0,
        }
    }

    pub fn alloc(&mut self, shape: &[usize]) -> Result<Tensor, MemoryError> {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();

        if self.offset + numel > self.capacity {
            return Err(MemoryError::OutOfMemory {
                requested_elements: numel,
                available_elements: self.capacity.saturating_sub(self.offset),
            });
        }

        let base_offset = self.offset;
        self.offset += numel;
        if self.offset > self.peak_offset {
            self.peak_offset = self.offset;
        }

        let strides = shape_obj.default_strides();
        let tensor = Tensor::from_raw_parts(
            shape_obj,
            strides,
            base_offset,
            DType::F32,
            Arc::clone(&self.storage),
        );

        Ok(tensor)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn current_usage(&self) -> usize {
        self.offset
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn peak_usage(&self) -> usize {
        self.peak_offset
    }
}

#[derive(Debug)]
pub struct DoubleBuffer {
    buffer_a: Tensor,
    buffer_b: Tensor,
    current_is_a: bool,
}

impl DoubleBuffer {
    pub fn new(shape: &[usize]) -> Self {
        Self {
            buffer_a: Tensor::zeros(shape),
            buffer_b: Tensor::zeros(shape),
            current_is_a: true,
        }
    }

    pub fn step(&mut self) -> (&Tensor, &mut Tensor) {
        if self.current_is_a {
            self.current_is_a = false;
            (&self.buffer_a, &mut self.buffer_b)
        } else {
            self.current_is_a = true;
            (&self.buffer_b, &mut self.buffer_a)
        }
    }

    pub fn current(&self) -> &Tensor {
        if self.current_is_a {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }
}
