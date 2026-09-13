use crate::cache::LayerKvCache;
use crate::model::weights::LayerWeights;
use crate::tensor::{ops, Tensor, TensorError};
use crate::transformer::attention::CausalSelfAttention;
use crate::transformer::mlp::FeedForward;

pub struct TransformerBlock {
    pub attention: CausalSelfAttention,
    pub eps: f32,
}

impl TransformerBlock {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, rope_theta: f32, eps: f32) -> Self {
        Self {
            attention: CausalSelfAttention::new(num_heads, num_kv_heads, head_dim, rope_theta),
            eps,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        weights: &LayerWeights,
        kv_cache: &mut LayerKvCache,
        start_pos: usize,
    ) -> Result<Tensor, TensorError> {
        let normed_attn = ops::rms_norm(x, &weights.attn_norm, self.eps)?;
        let attn_out = self.attention.forward(&normed_attn, weights, kv_cache, start_pos)?;
        let x1 = ops::add(x, &attn_out)?;

        let normed_ffn = ops::rms_norm(&x1, &weights.ffn_norm, self.eps)?;
        let ffn_out = FeedForward::forward(&normed_ffn, weights)?;

        ops::add(&x1, &ffn_out)
    }
}
