use crate::model::weights::LayerWeights;
use crate::tensor::{matmul, ops, Tensor, TensorError};

pub struct FeedForward;

impl FeedForward {
    pub fn forward(x: &Tensor, weights: &LayerWeights) -> Result<Tensor, TensorError> {
        let w_gate_t = weights.w_gate.t()?;
        let w_up_t = weights.w_up.t()?;
        let w_down_t = weights.w_down.t()?;

        let gate = matmul(x, &w_gate_t)?;
        let gate_act = ops::silu(&gate);
        let up = matmul(x, &w_up_t)?;
        let inter = ops::mul(&gate_act, &up)?;

        matmul(&inter, &w_down_t)
    }
}
