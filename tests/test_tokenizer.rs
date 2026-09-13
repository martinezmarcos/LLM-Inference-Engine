use mini_llm::tokenizer::BpeTokenizer;

#[test]
fn test_bpe_roundtrip() {
    let tokenizer = BpeTokenizer::new_default();
    let text = "the meaning of life";

    let tokens = tokenizer.encode(text, false);
    assert!(!tokens.is_empty());

    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_bpe_bos_handling() {
    let tokenizer = BpeTokenizer::new_default();
    let text = "the life";

    let tokens_with_bos = tokenizer.encode(text, true);
    assert_eq!(tokens_with_bos[0], tokenizer.bos_id().unwrap());

    let tokens_no_bos = tokenizer.encode(text, false);
    assert_eq!(tokens_with_bos.len(), tokens_no_bos.len() + 1);

    // Decoding should skip BOS
    let decoded = tokenizer.decode(&tokens_with_bos).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_smollm_tokenizer() {
    let path = std::path::Path::new("models/SmolLM-135M.Q4_0.gguf");
    if !path.exists() {
        return;
    }
    let gguf = mini_llm::model::GgufFile::open(path).unwrap();
    let tokenizer = BpeTokenizer::from_gguf(&gguf.metadata).unwrap();
    let text = "The capital of France is";
    let encoded = tokenizer.encode(text, false);
    assert_eq!(encoded, vec![504, 3575, 282, 4649, 314]);
    let decoded = tokenizer.decode(&encoded).unwrap();
    assert_eq!(decoded, text);
}

