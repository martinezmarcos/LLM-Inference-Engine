use mini_llm::model::{f16_to_f32, GgufBuilder, GgufFile, GgufValue, ModelConfig};

#[test]
fn test_f16_conversion() {
    // 0.0 in f16 is 0x0000
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // 1.0 in f16 is 0x3C00 (sign=0, exp=15, mantissa=0)
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    // -2.0 in f16 is 0xC000
    assert_eq!(f16_to_f32(0xC000), -2.0);
}

#[test]
fn test_gguf_parsing_and_tensor_loading() {
    let mut builder = GgufBuilder::new();
    builder.add_string_meta("general.architecture", "llama");
    builder.add_u32_meta("llama.embedding_length", 128);
    builder.add_u32_meta("llama.block_count", 2);

    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    builder.add_tensor_f32("token_embd.weight", &[2, 3], &weight_data);

    let binary_gguf = builder.build();

    // Parse from bytes (mirrors mmap loading)
    let gguf = GgufFile::from_bytes(&binary_gguf).unwrap();
    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.architecture(), Some("llama"));

    // Verify metadata
    match gguf.metadata.get("llama.embedding_length") {
        Some(GgufValue::Uint32(val)) => assert_eq!(*val, 128),
        _ => panic!("Expected embedding_length to be Uint32(128)"),
    }

    // Load tensor
    let tensor = gguf.get_tensor("token_embd.weight").unwrap();
    assert_eq!(tensor.dims(), &[2, 3]);
    assert_eq!(tensor.to_vec(), weight_data);

    // Verify config extraction
    let config = ModelConfig::from_gguf(&gguf).unwrap();
    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 128);
    assert_eq!(config.num_layers, 2);
}
