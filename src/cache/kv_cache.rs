use crate::tensor::{Tensor, TensorError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    CapacityExceeded {
        requested: usize,
        capacity: usize,
    },
    TensorErr(TensorError),
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapacityExceeded { requested, capacity } => write!(
                f,
                "KV Cache capacity exceeded: requested sequence length {}, max capacity is {}",
                requested, capacity
            ),
            Self::TensorErr(e) => write!(f, "KV Cache tensor error: {}", e),
        }
    }
}

impl std::error::Error for KvCacheError {}

impl From<TensorError> for KvCacheError {
    fn from(e: TensorError) -> Self {
        Self::TensorErr(e)
    }
}

#[derive(Debug, Clone)]
pub struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    current_len: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LayerKvCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let total_elements = max_seq_len * num_kv_heads * head_dim;
        Self {
            keys: vec![0.0; total_elements],
            values: vec![0.0; total_elements],
            current_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(), KvCacheError> {
        let new_len = new_k.dims()[0];
        if self.current_len + new_len > self.max_seq_len {
            return Err(KvCacheError::CapacityExceeded {
                requested: self.current_len + new_len,
                capacity: self.max_seq_len,
            });
        }

        let elem_per_token = self.num_kv_heads * self.head_dim;
        let start_idx = self.current_len * elem_per_token;
        let count = new_len * elem_per_token;

        if new_k.is_contiguous() && new_v.is_contiguous() {
            let k_slice = new_k.as_slice().map_err(KvCacheError::TensorErr)?;
            let v_slice = new_v.as_slice().map_err(KvCacheError::TensorErr)?;
            self.keys[start_idx..start_idx + count].copy_from_slice(k_slice);
            self.values[start_idx..start_idx + count].copy_from_slice(v_slice);
        } else {
            let k_vec = new_k.to_vec();
            let v_vec = new_v.to_vec();
            self.keys[start_idx..start_idx + count].copy_from_slice(&k_vec);
            self.values[start_idx..start_idx + count].copy_from_slice(&v_vec);
        }

        self.current_len += new_len;
        Ok(())
    }

    pub fn read_keys(&self) -> Result<Tensor, KvCacheError> {
        if self.current_len == 0 {
            return Ok(Tensor::zeros(&[0, self.num_kv_heads, self.head_dim]));
        }
        let count = self.current_len * self.num_kv_heads * self.head_dim;
        Tensor::from_slice(
            &self.keys[..count],
            &[self.current_len, self.num_kv_heads, self.head_dim],
        )
        .map_err(KvCacheError::TensorErr)
    }

    pub fn read_values(&self) -> Result<Tensor, KvCacheError> {
        if self.current_len == 0 {
            return Ok(Tensor::zeros(&[0, self.num_kv_heads, self.head_dim]));
        }
        let count = self.current_len * self.num_kv_heads * self.head_dim;
        Tensor::from_slice(
            &self.values[..count],
            &[self.current_len, self.num_kv_heads, self.head_dim],
        )
        .map_err(KvCacheError::TensorErr)
    }

    pub fn current_position(&self) -> usize {
        self.current_len
    }

    pub fn capacity(&self) -> usize {
        self.max_seq_len
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerKvCache::new(max_seq_len, num_kv_heads, head_dim));
        }
        Self {
            layers,
            max_seq_len,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub fn current_position(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers[0].current_position()
        }
    }

    pub fn estimate_memory_bytes(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> usize {
        2 * num_layers * max_seq_len * num_kv_heads * head_dim * dtype_bytes
    }
}
