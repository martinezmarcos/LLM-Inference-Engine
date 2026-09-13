use crate::tensor::{Tensor, TensorError};

/// Token embedding table lookup.
pub fn embedding_lookup(
    tokens: &[usize],
    weight: &Tensor,
) -> Result<Tensor, TensorError> {
    let rank = weight.rank();
    if rank != 2 {
        return Err(TensorError::IncompatibleDimensions(format!(
            "Embedding weight must be 2D [vocab_size, hidden_dim], found rank {}",
            rank
        )));
    }

    let vocab_size = weight.dims()[0];
    let hidden_dim = weight.dims()[1];
    let seq_len = tokens.len();

    let mut out_data = Vec::with_capacity(seq_len * hidden_dim);
    let w_contig = weight.contiguous();
    let w_slice = w_contig.as_slice()?;

    for &token in tokens {
        if token >= vocab_size {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![token],
                shape: vec![vocab_size],
            });
        }
        let start = token * hidden_dim;
        let end = start + hidden_dim;
        out_data.extend_from_slice(&w_slice[start..end]);
    }

    Tensor::from_vec(out_data, &[seq_len, hidden_dim])
}
