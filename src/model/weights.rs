use crate::model::config::ModelConfig;
use crate::model::gguf::{GgufError, GgufFile};
use crate::tensor::Tensor;

/// Weights for a single Transformer Decoder block.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub ffn_norm: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

/// Full set of weights for a decoder-only Transformer.
#[derive(Debug, Clone)]
pub struct TransformerWeights {
    pub token_embeddings: Tensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Tensor,
    pub output: Tensor,
}

impl TransformerWeights {
    /// Loads weights from a parsed GGUF file according to standard LLaMA naming conventions.
    pub fn load_from_gguf(gguf: &GgufFile, config: &ModelConfig) -> Result<Self, GgufError> {
        let token_embeddings = gguf.get_tensor("token_embd.weight")?;
        let output_norm = gguf.get_tensor("output_norm.weight")?;

        // In LLaMA, if output.weight is not present, embedding weights are tied
        let output = if gguf.tensors.contains_key("output.weight") {
            gguf.get_tensor("output.weight")?
        } else {
            token_embeddings.clone()
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let prefix = format!("blk.{}", l);
            let attn_norm = gguf.get_tensor(&format!("{}.attn_norm.weight", prefix))?;
            let wq = gguf.get_tensor(&format!("{}.attn_q.weight", prefix))?;
            let wk = gguf.get_tensor(&format!("{}.attn_k.weight", prefix))?;
            let wv = gguf.get_tensor(&format!("{}.attn_v.weight", prefix))?;
            let wo = gguf.get_tensor(&format!("{}.attn_output.weight", prefix))?;
            let ffn_norm = gguf.get_tensor(&format!("{}.ffn_norm.weight", prefix))?;
            let w_gate = gguf.get_tensor(&format!("{}.ffn_gate.weight", prefix))?;
            let w_up = gguf.get_tensor(&format!("{}.ffn_up.weight", prefix))?;
            let w_down = gguf.get_tensor(&format!("{}.ffn_down.weight", prefix))?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        Ok(Self {
            token_embeddings,
            layers,
            output_norm,
            output,
        })
    }

    /// Generates compact synthetic model weights for headless testing and verification.
    pub fn random_compact(config: &ModelConfig, seed: u64) -> Self {
        let dim = config.hidden_dim;
        let head_dim = config.head_dim();
        let q_dim = config.num_heads * head_dim;
        let kv_dim = config.num_kv_heads * head_dim;
        let inter_dim = config.intermediate_dim;

        let token_embeddings = Tensor::randn(&[config.vocab_size, dim], seed);
        let output_norm = Tensor::ones(&[dim]);
        let output = Tensor::randn(&[config.vocab_size, dim], seed + 1);

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let s = seed + (l as u64) * 10;
            layers.push(LayerWeights {
                attn_norm: Tensor::ones(&[dim]),
                wq: Tensor::randn(&[q_dim, dim], s + 1),
                wk: Tensor::randn(&[kv_dim, dim], s + 2),
                wv: Tensor::randn(&[kv_dim, dim], s + 3),
                wo: Tensor::randn(&[dim, q_dim], s + 4),
                ffn_norm: Tensor::ones(&[dim]),
                w_gate: Tensor::randn(&[inter_dim, dim], s + 5),
                w_up: Tensor::randn(&[inter_dim, dim], s + 6),
                w_down: Tensor::randn(&[dim, inter_dim], s + 7),
            });
        }

        Self {
            token_embeddings,
            layers,
            output_norm,
            output,
        }
    }
}
