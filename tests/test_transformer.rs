use mini_llm::model::{ModelConfig, TransformerWeights};
use mini_llm::transformer::{embedding_lookup, RoPE, TransformerDecoder};
use mini_llm::tensor::Tensor;

#[test]
fn test_embedding_lookup() {
    let weight = Tensor::from_vec(
        vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ],
        &[3, 2],
    )
    .unwrap();

    let tokens = [2, 0];
    let emb = embedding_lookup(&tokens, &weight).unwrap();
    assert_eq!(emb.dims(), &[2, 2]);
    assert_eq!(emb.to_vec(), vec![5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn test_rope_rotation_preserves_magnitude() {
    let rope = RoPE::new(4, 10000.0);
    // [seq_len=1, num_heads=1, head_dim=4]
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();

    let rotated = rope.forward(&x, 5).unwrap();
    let r_vec = rotated.to_vec();

    // Norm of 2D pairs should be preserved under orthogonal rotation
    let orig_norm_0 = (1.0f32 * 1.0 + 2.0 * 2.0).sqrt();
    let rot_norm_0 = (r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1]).sqrt();
    assert!((orig_norm_0 - rot_norm_0).abs() < 1e-5);

    let orig_norm_1 = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt();
    let rot_norm_1 = (r_vec[2] * r_vec[2] + r_vec[3] * r_vec[3]).sqrt();
    assert!((orig_norm_1 - rot_norm_1).abs() < 1e-5);
}

#[test]
fn test_transformer_decoder_forward() {
    let mut config = ModelConfig::default_compact();
    config.vocab_size = 100;
    config.hidden_dim = 64;
    config.intermediate_dim = 128;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 2; // Test Grouped-Query Attention (GQA)
    config.max_seq_len = 64;

    let weights = TransformerWeights::random_compact(&config, 42);
    let mut model = TransformerDecoder::new(config.clone(), weights);

    // Prompt tokens
    let tokens = vec![5, 12, 48];
    let logits = model.forward(&tokens, 0).unwrap();

    assert_eq!(logits.dims(), &[1, config.vocab_size]);
    assert_eq!(model.kv_cache.current_position(), 3);

    // Next token single step
    let next_logits = model.forward(&[17], 3).unwrap();
    assert_eq!(next_logits.dims(), &[1, config.vocab_size]);
    assert_eq!(model.kv_cache.current_position(), 4);
}
