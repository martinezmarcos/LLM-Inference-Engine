use mini_llm::inference::InferenceEngine;
use mini_llm::model::{ModelConfig, TransformerWeights};
use mini_llm::sampling::GreedySampler;
use mini_llm::tokenizer::BpeTokenizer;
use mini_llm::transformer::TransformerDecoder;

#[test]
fn test_end_to_end_inference_loop() {
    let tokenizer = BpeTokenizer::new_default();
    let mut config = ModelConfig::default_compact();
    config.vocab_size = tokenizer.vocab_size();
    config.hidden_dim = 64;
    config.intermediate_dim = 128;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.max_seq_len = 128;

    let weights = TransformerWeights::random_compact(&config, 101);
    let model = TransformerDecoder::new(config, weights);

    let mut engine = InferenceEngine::new(model, tokenizer);
    let mut sampler = GreedySampler;

    let mut streamed_tokens = Vec::new();
    let generated = engine
        .generate("the meaning of", 10, &mut sampler, |token| {
            streamed_tokens.push(token.to_string());
            true
        })
        .unwrap();

    assert!(!generated.is_empty());
    let prompt_len = engine.tokenizer.encode("the meaning of", true).len();
    assert_eq!(engine.model.kv_cache.current_position(), prompt_len + 10);
}
