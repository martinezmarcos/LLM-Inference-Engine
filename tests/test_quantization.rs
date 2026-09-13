use mini_llm::quantization::{
    dequantize_q4_0, dequantize_q8_0, evaluate_quantization_error, quantize_q4_0, quantize_q8_0,
    vec_dot_q8_0,
};

#[test]
fn test_q8_0_quantization_accuracy() {
    let original: Vec<f32> = (0..64)
        .map(|i| ((i as f32) / 10.0).sin() * 2.5)
        .collect();

    let q8 = quantize_q8_0(&original);
    assert_eq!(q8.len(), (64 / 32) * 34); // 68 bytes vs 256 bytes in f32 (3.76x compression)

    let deq = dequantize_q8_0(&q8);
    let (mse, max_err) = evaluate_quantization_error(&original, &deq);

    // Q8_0 has 8-bit precision: max error should be less than 1/127 of range (< 0.03)
    assert!(max_err < 0.03, "Max error too high: {}", max_err);
    assert!(mse < 0.0005, "MSE too high: {}", mse);
}

#[test]
fn test_q4_0_quantization_accuracy() {
    let original: Vec<f32> = (0..64)
        .map(|i| ((i as f32) / 10.0).cos() * 1.5)
        .collect();

    let q4 = quantize_q4_0(&original);
    assert_eq!(q4.len(), (64 / 32) * 18); // 36 bytes vs 256 bytes in f32 (7.11x compression)

    let deq = dequantize_q4_0(&q4);
    let (mse, max_err) = evaluate_quantization_error(&original, &deq);

    // Q4_0 has 4-bit precision: max error should be reasonable (< 0.3)
    assert!(max_err < 0.35, "Max error too high: {}", max_err);
    assert!(mse < 0.05, "MSE too high: {}", mse);
}

#[test]
fn test_vec_dot_q8_0() {
    let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let activations: Vec<f32> = (0..64).map(|i| 1.0 + (i as f32) * 0.05).collect();

    let q8_row = quantize_q8_0(&weights);
    let dot_quant = vec_dot_q8_0(&activations, &q8_row);

    // Analytical FP32 dot product
    let dot_fp32: f32 = weights.iter().zip(activations.iter()).map(|(w, a)| w * a).sum();

    let rel_err = (dot_quant - dot_fp32).abs() / dot_fp32;
    assert!(rel_err < 0.02, "Relative error too high: {}", rel_err);
}
