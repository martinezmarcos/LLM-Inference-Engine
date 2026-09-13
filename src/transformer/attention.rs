use crate::cache::LayerKvCache;
use crate::model::weights::LayerWeights;
use crate::tensor::{matmul, Tensor, TensorError};
use crate::transformer::rope::RoPE;

pub struct CausalSelfAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope: RoPE,
}

impl CausalSelfAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, rope_theta: f32) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            rope: RoPE::new(head_dim, rope_theta),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        weights: &LayerWeights,
        kv_cache: &mut LayerKvCache,
        start_pos: usize,
    ) -> Result<Tensor, TensorError> {
        let seq_len = x.dims()[0];
        let hidden_dim = x.dims()[1];
        let q_dim = self.num_heads * self.head_dim;

        let wq_t = weights.wq.t()?;
        let wk_t = weights.wk.t()?;
        let wv_t = weights.wv.t()?;

        let q = matmul(x, &wq_t)?;
        let k = matmul(x, &wk_t)?;
        let v = matmul(x, &wv_t)?;

        let q = q.reshape(&[seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[seq_len, self.num_kv_heads, self.head_dim])?;

        let q = self.rope.forward(&q, start_pos)?;
        let k = self.rope.forward(&k, start_pos)?;

        kv_cache
            .append(&k, &v)
            .map_err(|e| TensorError::IncompatibleDimensions(e.to_string()))?;

        let k_all = kv_cache
            .read_keys()
            .map_err(|e| TensorError::IncompatibleDimensions(e.to_string()))?;
        let v_all = kv_cache
            .read_values()
            .map_err(|e| TensorError::IncompatibleDimensions(e.to_string()))?;

        let total_seq_len = k_all.dims()[0];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let heads_per_kv = self.num_heads / self.num_kv_heads;

        let q_vec = q.to_vec();
        let k_vec = k_all.to_vec();
        let v_vec = v_all.to_vec();

        let mut context_vec = vec![0.0f32; seq_len * self.num_heads * self.head_dim];

        for h in 0..self.num_heads {
            let kv_h = h / heads_per_kv;

            for s in 0..seq_len {
                let current_token_pos = start_pos + s;
                let q_offset = s * (self.num_heads * self.head_dim) + h * self.head_dim;
                let q_head = &q_vec[q_offset..q_offset + self.head_dim];

                let mut scores = vec![f32::NEG_INFINITY; total_seq_len];
                let mut max_score = f32::NEG_INFINITY;

                for (t, score_item) in scores.iter_mut().enumerate().take(total_seq_len) {
                    if t > current_token_pos {
                        continue;
                    }

                    let k_offset = t * (self.num_kv_heads * self.head_dim) + kv_h * self.head_dim;
                    let k_head = &k_vec[k_offset..k_offset + self.head_dim];

                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_head[d] * k_head[d];
                    }
                    let score = dot * scale;
                    *score_item = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut sum_exp = 0.0f32;
                for score_val in scores.iter_mut().take(total_seq_len) {
                    if *score_val != f32::NEG_INFINITY {
                        let exp_val = (*score_val - max_score).exp();
                        *score_val = exp_val;
                        sum_exp += exp_val;
                    } else {
                        *score_val = 0.0;
                    }
                }

                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                for score_val in scores.iter_mut().take(total_seq_len) {
                    *score_val *= inv_sum;
                }

                let ctx_offset = s * (self.num_heads * self.head_dim) + h * self.head_dim;
                let ctx_slice = &mut context_vec[ctx_offset..ctx_offset + self.head_dim];

                for (t, &weight) in scores.iter().enumerate().take(total_seq_len) {
                    if weight == 0.0 {
                        continue;
                    }
                    let v_offset = t * (self.num_kv_heads * self.head_dim) + kv_h * self.head_dim;
                    let v_head = &v_vec[v_offset..v_offset + self.head_dim];

                    for d in 0..self.head_dim {
                        ctx_slice[d] += weight * v_head[d];
                    }
                }
            }
        }

        let context = Tensor::from_vec(context_vec, &[seq_len, q_dim])?;
        let wo_t = weights.wo.t()?;
        let output = matmul(&context, &wo_t)?;

        if output.dims()[1] != hidden_dim {
            return Err(TensorError::IncompatibleDimensions(format!(
                "Attention output dimension {} does not match hidden dimension {}",
                output.dims()[1],
                hidden_dim
            )));
        }

        Ok(output)
    }
}
