use crate::tensor::{Tensor, TensorError};

#[derive(Debug, Clone)]
pub struct RoPE {
    pub dim: usize,
    pub theta: f32,
}

impl RoPE {
    pub fn new(dim: usize, theta: f32) -> Self {
        Self { dim, theta }
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<Tensor, TensorError> {
        let rank = x.rank();
        if rank != 3 {
            return Err(TensorError::IncompatibleDimensions(format!(
                "RoPE requires [seq_len, num_heads, head_dim], found rank {}",
                rank
            )));
        }

        let seq_len = x.dims()[0];
        let num_heads = x.dims()[1];
        let head_dim = x.dims()[2];

        if !head_dim.is_multiple_of(2) {
            return Err(TensorError::IncompatibleDimensions(format!(
                "RoPE requires even head dimension, found {}",
                head_dim
            )));
        }

        let mut out_data = x.to_vec();

        for s in 0..seq_len {
            let pos = (start_pos + s) as f32;

            for h in 0..num_heads {
                let head_base = s * (num_heads * head_dim) + h * head_dim;

                for i in 0..(head_dim / 2) {
                    let theta_i = 1.0 / self.theta.powf((2 * i) as f32 / head_dim as f32);
                    let angle = pos * theta_i;
                    let cos = angle.cos();
                    let sin = angle.sin();

                    let idx0 = head_base + 2 * i;
                    let idx1 = head_base + 2 * i + 1;

                    let v0 = out_data[idx0];
                    let v1 = out_data[idx1];

                    out_data[idx0] = v0 * cos - v1 * sin;
                    out_data[idx1] = v0 * sin + v1 * cos;
                }
            }
        }

        Tensor::from_vec(out_data, x.dims())
    }
}
