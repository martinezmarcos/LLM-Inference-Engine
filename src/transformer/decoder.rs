use crate::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::model::weights::TransformerWeights;
use crate::tensor::{matmul, ops, Tensor, TensorError};
use crate::transformer::block::TransformerBlock;
use crate::transformer::embedding::embedding_lookup;
use rayon::prelude::*;

pub struct TransformerDecoder {
    pub config: ModelConfig,
    pub weights: TransformerWeights,
    pub blocks: Vec<TransformerBlock>,
    pub kv_cache: KvCache,
}

impl TransformerDecoder {
    pub fn new(config: ModelConfig, weights: TransformerWeights) -> Self {
        let head_dim = config.head_dim();
        let mut blocks = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                config.num_heads,
                config.num_kv_heads,
                head_dim,
                config.rope_theta,
                config.rms_norm_eps,
            ));
        }

        let kv_cache = KvCache::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            head_dim,
        );

        Self {
            config,
            weights,
            blocks,
            kv_cache,
        }
    }

    pub fn forward(
        &mut self,
        tokens: &[usize],
        start_pos: usize,
    ) -> Result<Tensor, TensorError> {
        if tokens.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        let seq_len = tokens.len();
        let mut x = embedding_lookup(tokens, &self.weights.token_embeddings)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let layer_w = &self.weights.layers[i];
            let layer_cache = &mut self.kv_cache.layers[i];
            x = block.forward(&x, layer_w, layer_cache, start_pos)?;
        }

        let x_norm = ops::rms_norm(&x, &self.weights.output_norm, self.config.rms_norm_eps)?;

        let last_token_x = if seq_len > 1 {
            x_norm.slice(0, seq_len - 1, seq_len)?
        } else {
            x_norm
        };

        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.weights.output.dims()[0];

        if self.weights.output.is_contiguous() && self.weights.output.dims()[1] == hidden_dim {
            let x_vec = last_token_x.to_vec();
            let w_slice = self.weights.output.as_slice()?;
            let mut logits = vec![0.0f32; vocab_size];

            logits
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_j = chunk_idx * 256;
                    for (offset, logit) in chunk.iter_mut().enumerate() {
                        let j = base_j + offset;
                        let w_row = &w_slice[j * hidden_dim..(j + 1) * hidden_dim];
                        let mut sum = 0.0f32;
                        for k in 0..hidden_dim {
                            sum += x_vec[k] * w_row[k];
                        }
                        *logit = sum;
                    }
                });

            Tensor::from_vec(logits, &[1, vocab_size])
        } else {
            let output_w_t = self.weights.output.t()?;
            matmul(&last_token_x, &output_w_t)
        }
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }
}
