use mini_llm::tensor::*;

#[test]
fn test_shape_and_strides() {
    let s = Shape::new(&[2, 3, 4]);
    assert_eq!(s.rank(), 3);
    assert_eq!(s.numel(), 24);
    assert_eq!(s.default_strides(), vec![12, 4, 1]);

    let scalar_s = Shape::scalar();
    assert_eq!(scalar_s.rank(), 0);
    assert_eq!(scalar_s.numel(), 1);
    assert_eq!(scalar_s.default_strides(), Vec::<usize>::new());
}

#[test]
fn test_shape_broadcasting() {
    let s1 = Shape::new(&[2, 1, 4]);
    let s2 = Shape::new(&[3, 4]);
    let b = Shape::broadcast_shapes(&s1, &s2).unwrap();
    assert_eq!(b.dims(), &[2, 3, 4]);

    let s3 = Shape::new(&[2, 3]);
    let s4 = Shape::new(&[2, 4]);
    assert!(Shape::broadcast_shapes(&s3, &s4).is_err());
}

#[test]
fn test_tensor_creation_and_indexing() {
    let mut t = Tensor::zeros(&[2, 3]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.get(&[0, 0]).unwrap(), 0.0);

    t.set(&[1, 2], 42.0).unwrap();
    assert_eq!(t.get(&[1, 2]).unwrap(), 42.0);
    assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);

    // Out of bounds check
    assert!(t.get(&[2, 0]).is_err());
    assert!(t.get(&[0, 3]).is_err());
}

#[test]
fn test_strided_transpose_and_contiguous() {
    // 2x3 matrix:
    // [1, 2, 3]
    // [4, 5, 6]
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(t.is_contiguous());

    let trans = t.t().unwrap();
    assert_eq!(trans.dims(), &[3, 2]);
    assert_eq!(trans.strides(), &[1, 3]); // Strides swapped!
    assert!(!trans.is_contiguous()); // Now non-contiguous

    // Logical index verification
    assert_eq!(trans.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(trans.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(trans.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(trans.get(&[1, 1]).unwrap(), 5.0);
    assert_eq!(trans.get(&[2, 0]).unwrap(), 3.0);
    assert_eq!(trans.get(&[2, 1]).unwrap(), 6.0);

    // Materializing contiguous layout
    let contig = trans.contiguous();
    assert!(contig.is_contiguous());
    assert_eq!(contig.strides(), &[2, 1]);
    assert_eq!(contig.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_zero_copy_slice() {
    let t = Tensor::from_vec((0..12).map(|x| x as f32).collect(), &[3, 4]).unwrap();
    // Slice rows 1..3
    let sl = t.slice(0, 1, 3).unwrap();
    assert_eq!(sl.dims(), &[2, 4]);
    assert_eq!(sl.offset(), 4); // Row 1 starts at index 4

    assert_eq!(sl.get(&[0, 0]).unwrap(), 4.0);
    assert_eq!(sl.get(&[1, 3]).unwrap(), 11.0);
}

#[test]
fn test_elementwise_and_broadcasting() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0], &[1, 2]).unwrap(); // Broadcast across rows

    let c = add(&a, &b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec(), vec![11.0, 22.0, 13.0, 24.0]);

    let s = scale(&a, 2.0);
    assert_eq!(s.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_reductions() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let s_all = sum(&a, None, false).unwrap();
    assert_eq!(s_all.get(&[]).unwrap(), 21.0);

    let s_rows = sum(&a, Some(0), false).unwrap();
    assert_eq!(s_rows.dims(), &[3]);
    assert_eq!(s_rows.to_vec(), vec![5.0, 7.0, 9.0]);

    let s_cols = sum(&a, Some(1), false).unwrap();
    assert_eq!(s_cols.dims(), &[2]);
    assert_eq!(s_cols.to_vec(), vec![6.0, 15.0]);

    let m = mean(&a, None, false).unwrap();
    assert_eq!(m.get(&[]).unwrap(), 3.5);

    let am = argmax(&a, 1).unwrap();
    assert_eq!(am, vec![2, 2]); // Index 2 has max value in each row
}

#[test]
fn test_numerically_stable_softmax() {
    // Large values that would overflow naive exp()
    let logits = Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]).unwrap();
    let probs = softmax(&logits, 1).unwrap();

    let p_vec = probs.to_vec();
    // Sum of probabilities must equal 1.0
    let total: f32 = p_vec.iter().sum();
    assert!((total - 1.0).abs() < 1e-6);

    // Relative ratios: p[2] / p[1] == e, p[1] / p[0] == e
    let e = std::f32::consts::E;
    assert!((p_vec[1] / p_vec[0] - e).abs() < 1e-4);
    assert!((p_vec[2] / p_vec[1] - e).abs() < 1e-4);
}

#[test]
fn test_rms_norm() {
    let x = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[1, 4]).unwrap();
    let w = Tensor::ones(&[4]);
    // Mean of squares is 4.0, sqrt(4.0) = 2.0. x / 2.0 = [1.0, 1.0, 1.0, 1.0]
    let y = rms_norm(&x, &w, 0.0).unwrap();
    for &val in &y.to_vec() {
        assert!((val - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_matmul_2d_analytical() {
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = [[19, 22], [43, 50]]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_variants_equivalence() {
    let a = Tensor::randn(&[32, 64], 123);
    let b = Tensor::randn(&[64, 48], 456);

    let c_opt = matmul(&a, &b).unwrap();
    let c_naive = matmul_naive(&a, &b).unwrap();
    let c_tiled = matmul_tiled(&a, &b, 16).unwrap();
    let c_par = matmul_parallel(&a, &b).unwrap();

    let v_opt = c_opt.to_vec();
    let v_naive = c_naive.to_vec();
    let v_tiled = c_tiled.to_vec();
    let v_par = c_par.to_vec();

    for i in 0..v_opt.len() {
        assert!((v_opt[i] - v_naive[i]).abs() < 1e-4);
        assert!((v_opt[i] - v_tiled[i]).abs() < 1e-4);
        assert!((v_opt[i] - v_par[i]).abs() < 1e-4);
    }
}

#[test]
fn test_matmul_non_contiguous_and_batched() {
    // Multiplies with transposed matrix (non-contiguous view)
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 7.0, 6.0, 8.0], &[2, 2]).unwrap();
    let b_t = b.t().unwrap(); // [[5, 6], [7, 8]]
    let c = matmul(&a, &b_t).unwrap();
    assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);

    // Batched 3D matmul: [2, 2, 2] x [2, 2, 2] -> [2, 2, 2]
    let batch_a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, // Batch 0
            2.0, 0.0, 0.0, 2.0, // Batch 1
        ],
        &[2, 2, 2],
    )
    .unwrap();

    let batch_b = Tensor::from_vec(
        vec![
            5.0, 6.0, 7.0, 8.0, // Batch 0
            1.0, 2.0, 3.0, 4.0, // Batch 1
        ],
        &[2, 2, 2],
    )
    .unwrap();

    let batch_c = matmul(&batch_a, &batch_b).unwrap();
    assert_eq!(batch_c.dims(), &[2, 2, 2]);
    assert_eq!(
        batch_c.to_vec(),
        vec![
            19.0, 22.0, 43.0, 50.0, // Batch 0 result
            2.0, 4.0, 6.0, 8.0,    // Batch 1 result
        ]
    );
}

#[test]
fn test_error_handling_edge_cases() {
    // Incompatible matmul dimensions
    let a = Tensor::zeros(&[2, 3]);
    let b = Tensor::zeros(&[2, 3]);
    assert!(matmul(&a, &b).is_err());

    // Invalid reshape
    let c = Tensor::zeros(&[2, 3]);
    assert!(c.reshape(&[2, 4]).is_err());

    // Invalid slice
    assert!(c.slice(0, 0, 5).is_err());
}
