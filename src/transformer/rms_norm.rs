use crate::tensor::{ops, Tensor, TensorError};

/// RMSNorm layer container.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        ops::rms_norm(x, &self.weight, self.eps)
    }
}
