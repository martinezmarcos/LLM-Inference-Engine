use crate::model::gguf::{GgufFile, GgufValue};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn default_compact() -> Self {
        Self {
            architecture: "llama".to_string(),
            vocab_size: 32000,
            hidden_dim: 288,
            intermediate_dim: 768,
            num_layers: 4,
            num_heads: 6,
            num_kv_heads: 6,
            max_seq_len: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .architecture()
            .unwrap_or("llama")
            .to_string();

        let get_u32 = |key_suffix: &str, default: u32| -> usize {
            let full_key = format!("{}.{}", arch, key_suffix);
            match gguf.metadata.get(&full_key) {
                Some(GgufValue::Uint32(v)) => *v as usize,
                Some(GgufValue::Uint64(v)) => *v as usize,
                _ => default as usize,
            }
        };

        let get_f32 = |key_suffix: &str, default: f32| -> f32 {
            let full_key = format!("{}.{}", arch, key_suffix);
            match gguf.metadata.get(&full_key) {
                Some(GgufValue::Float32(v)) => *v,
                _ => default,
            }
        };

        let hidden_dim = get_u32("embedding_length", 288);
        let num_layers = get_u32("block_count", 4);
        let num_heads = get_u32("attention.head_count", 6);
        let num_kv_heads = get_u32("attention.head_count_kv", num_heads as u32);
        let intermediate_dim = get_u32("feed_forward_length", (hidden_dim * 8 / 3) as u32);
        let max_seq_len = get_u32("context_length", 512);
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon", 1e-5);
        let rope_theta = get_f32("rope.freq_base", 10000.0);

        let vocab_size = if let Some(t) = gguf.tensors.get("token_embd.weight") {
            t.shape[0]
        } else {
            32000
        };

        Ok(Self {
            architecture: arch,
            vocab_size,
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
}
