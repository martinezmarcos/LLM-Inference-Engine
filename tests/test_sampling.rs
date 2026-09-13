use mini_llm::sampling::{ConfigurableSampler, GreedySampler, Sampler};

#[test]
fn test_greedy_sampler() {
    let mut sampler = GreedySampler;
    let logits = [1.0, 5.5, 3.2, 0.1];
    assert_eq!(sampler.sample(&logits), 1);
}

#[test]
fn test_top_k_sampling() {
    // With top_k=1, it must behave identically to greedy
    let mut sampler = ConfigurableSampler::new(0.7, 1, 1.0, 42);
    let logits = [10.0, 2.0, 5.0];
    assert_eq!(sampler.sample(&logits), 0);
}

#[test]
fn test_low_temperature_greedy_behavior() {
    let mut sampler = ConfigurableSampler::new(0.0001, 10, 1.0, 42);
    let logits = [0.1, 0.2, 9.9, 0.4];
    assert_eq!(sampler.sample(&logits), 2);
}
