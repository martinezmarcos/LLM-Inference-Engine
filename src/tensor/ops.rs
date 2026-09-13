use std::ops::{Add, Div, Mul, Sub};

use crate::tensor::core::Tensor;
use crate::tensor::error::TensorError;
use crate::tensor::shape::Shape;

// ============================================================================
// Elementwise Binary Operations (with Broadcasting)
// ============================================================================

/// Helper to execute a binary elementwise operation with full broadcasting support.
fn binary_op<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor, TensorError>
where
    F: Fn(f32, f32) -> f32,
{
    // Fast path: identical shapes and both contiguous
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        let mut out_data = Vec::with_capacity(a_slice.len());
        for (&x, &y) in a_slice.iter().zip(b_slice.iter()) {
            out_data.push(op(x, y));
        }
        return Tensor::from_vec(out_data, a.dims());
    }

    // General path with broadcasting
    let out_shape = Shape::broadcast_shapes(a.shape(), b.shape())?;
    let a_strides = a.shape().broadcast_strides(a.strides(), &out_shape)?;
    let b_strides = b.shape().broadcast_strides(b.strides(), &out_shape)?;

    let numel = out_shape.numel();
    let rank = out_shape.rank();
    let mut out_data = Vec::with_capacity(numel);

    let a_raw = a.raw_data();
    let b_raw = b.raw_data();
    let a_off = a.offset();
    let b_off = b.offset();

    if rank == 0 {
        let val = op(a_raw[a_off], b_raw[b_off]);
        out_data.push(val);
    } else {
        let mut coord = vec![0; rank];
        for _ in 0..numel {
            let mut off_a = a_off;
            let mut off_b = b_off;
            for r in 0..rank {
                off_a += coord[r] * a_strides[r];
                off_b += coord[r] * b_strides[r];
            }

            out_data.push(op(a_raw[off_a], b_raw[off_b]));

            // Advance coordinate
            for r in (0..rank).rev() {
                coord[r] += 1;
                if coord[r] < out_shape.dims()[r] {
                    break;
                }
                coord[r] = 0;
            }
        }
    }

    Tensor::from_vec(out_data, out_shape.dims())
}

/// Elementwise addition: `C = A + B` (with broadcasting).
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    binary_op(a, b, |x, y| x + y)
}

/// Elementwise subtraction: `C = A - B` (with broadcasting).
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    binary_op(a, b, |x, y| x - y)
}

/// Elementwise multiplication (Hadamard product): `C = A * B` (with broadcasting).
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    binary_op(a, b, |x, y| x * y)
}

/// Elementwise division: `C = A / B` (with broadcasting).
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    binary_op(a, b, |x, y| x / y)
}

/// Multiplies all elements by a scalar constant.
pub fn scale(a: &Tensor, factor: f32) -> Tensor {
    if a.is_contiguous() {
        let a_slice = a.as_slice().unwrap();
        let mut out = Vec::with_capacity(a_slice.len());
        for &val in a_slice {
            out.push(val * factor);
        }
        Tensor::from_vec(out, a.dims()).unwrap()
    } else {
        let contig = a.contiguous();
        scale(&contig, factor)
    }
}

/// Adds a scalar constant to all elements.
pub fn add_scalar(a: &Tensor, scalar: f32) -> Tensor {
    if a.is_contiguous() {
        let a_slice = a.as_slice().unwrap();
        let mut out = Vec::with_capacity(a_slice.len());
        for &val in a_slice {
            out.push(val + scalar);
        }
        Tensor::from_vec(out, a.dims()).unwrap()
    } else {
        let contig = a.contiguous();
        add_scalar(&contig, scalar)
    }
}

// ============================================================================
// Operator Overloads for Idiomatic Rust Syntax
// ============================================================================

impl Add for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn add(self, rhs: Self) -> Self::Output {
        add(self, rhs)
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn sub(self, rhs: Self) -> Self::Output {
        sub(self, rhs)
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs)
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn div(self, rhs: Self) -> Self::Output {
        div(self, rhs)
    }
}

// ============================================================================
// Reductions
// ============================================================================

/// Sums elements across an optional dimension. If `dim == None`, sums the entire tensor.
pub fn sum(a: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor, TensorError> {
    if a.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }

    match dim {
        None => {
            // Full reduction to scalar
            let flat = a.to_vec();
            let total: f32 = flat.iter().sum();
            if keepdim {
                let ones_shape = vec![1; a.rank()];
                Tensor::from_vec(vec![total], &ones_shape)
            } else {
                Ok(Tensor::scalar(total))
            }
        }
        Some(d) => {
            let rank = a.rank();
            if d >= rank {
                return Err(TensorError::DimensionOutOfBounds { dim: d, rank });
            }

            let mut out_dims = a.dims().to_vec();
            let reduce_size = out_dims[d];

            if keepdim {
                out_dims[d] = 1;
            } else {
                out_dims.remove(d);
            }

            let out_shape = Shape::new(&out_dims);
            let out_numel = out_shape.numel();
            let mut out_data = vec![0.0f32; out_numel];

            // Outer stride product before dim, inner stride product after dim
            let outer_size: usize = a.dims()[..d].iter().product();
            let inner_size: usize = a.dims()[d + 1..].iter().product();

            let a_contig = a.contiguous();
            let a_slice = a_contig.as_slice()?;

            for outer in 0..outer_size {
                for r in 0..reduce_size {
                    for inner in 0..inner_size {
                        let a_idx = outer * (reduce_size * inner_size) + r * inner_size + inner;
                        let out_idx = outer * inner_size + inner;
                        out_data[out_idx] += a_slice[a_idx];
                    }
                }
            }

            Tensor::from_vec(out_data, out_shape.dims())
        }
    }
}

/// Computes the arithmetic mean across an optional dimension.
pub fn mean(a: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor, TensorError> {
    let sum_tensor = sum(a, dim, keepdim)?;
    let count = match dim {
        None => a.numel() as f32,
        Some(d) => a.dims()[d] as f32,
    };
    Ok(scale(&sum_tensor, 1.0 / count))
}

/// Computes the maximum value across an optional dimension.
pub fn max(a: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor, TensorError> {
    if a.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }

    match dim {
        None => {
            let flat = a.to_vec();
            let mut max_val = f32::NEG_INFINITY;
            for &x in &flat {
                if x > max_val {
                    max_val = x;
                }
            }
            if keepdim {
                let ones_shape = vec![1; a.rank()];
                Tensor::from_vec(vec![max_val], &ones_shape)
            } else {
                Ok(Tensor::scalar(max_val))
            }
        }
        Some(d) => {
            let rank = a.rank();
            if d >= rank {
                return Err(TensorError::DimensionOutOfBounds { dim: d, rank });
            }

            let mut out_dims = a.dims().to_vec();
            let reduce_size = out_dims[d];

            if keepdim {
                out_dims[d] = 1;
            } else {
                out_dims.remove(d);
            }

            let out_shape = Shape::new(&out_dims);
            let out_numel = out_shape.numel();
            let mut out_data = vec![f32::NEG_INFINITY; out_numel];

            let outer_size: usize = a.dims()[..d].iter().product();
            let inner_size: usize = a.dims()[d + 1..].iter().product();

            let a_contig = a.contiguous();
            let a_slice = a_contig.as_slice()?;

            for outer in 0..outer_size {
                for r in 0..reduce_size {
                    for inner in 0..inner_size {
                        let a_idx = outer * (reduce_size * inner_size) + r * inner_size + inner;
                        let out_idx = outer * inner_size + inner;
                        let val = a_slice[a_idx];
                        if val > out_data[out_idx] {
                            out_data[out_idx] = val;
                        }
                    }
                }
            }

            Tensor::from_vec(out_data, out_shape.dims())
        }
    }
}

/// Returns indices of maximum values along a given dimension.
pub fn argmax(a: &Tensor, dim: usize) -> Result<Vec<usize>, TensorError> {
    let rank = a.rank();
    if dim >= rank {
        return Err(TensorError::DimensionOutOfBounds { dim, rank });
    }

    let reduce_size = a.dims()[dim];
    let outer_size: usize = a.dims()[..dim].iter().product();
    let inner_size: usize = a.dims()[dim + 1..].iter().product();

    let out_numel = outer_size * inner_size;
    let mut indices = vec![0usize; out_numel];
    let mut max_vals = vec![f32::NEG_INFINITY; out_numel];

    let a_contig = a.contiguous();
    let a_slice = a_contig.as_slice()?;

    for outer in 0..outer_size {
        for r in 0..reduce_size {
            for inner in 0..inner_size {
                let a_idx = outer * (reduce_size * inner_size) + r * inner_size + inner;
                let out_idx = outer * inner_size + inner;
                let val = a_slice[a_idx];
                if val > max_vals[out_idx] {
                    max_vals[out_idx] = val;
                    indices[out_idx] = r;
                }
            }
        }
    }

    Ok(indices)
}

// ============================================================================
// Activations & Normalizations
// ============================================================================

/// Numerically stable Softmax along a specified dimension:
///
/// $$\text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$
pub fn softmax(a: &Tensor, dim: usize) -> Result<Tensor, TensorError> {
    let rank = a.rank();
    if dim >= rank {
        return Err(TensorError::DimensionOutOfBounds { dim, rank });
    }

    let reduce_size = a.dims()[dim];
    let outer_size: usize = a.dims()[..dim].iter().product();
    let inner_size: usize = a.dims()[dim + 1..].iter().product();

    let a_contig = a.contiguous();
    let a_slice = a_contig.as_slice()?;
    let mut out_data = vec![0.0f32; a.numel()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base_offset = outer * (reduce_size * inner_size) + inner;

            let mut max_val = f32::NEG_INFINITY;
            for r in 0..reduce_size {
                let idx = base_offset + r * inner_size;
                let val = a_slice[idx];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0f32;
            for r in 0..reduce_size {
                let idx = base_offset + r * inner_size;
                let exp_val = (a_slice[idx] - max_val).exp();
                out_data[idx] = exp_val;
                sum_exp += exp_val;
            }

            let inv_sum = 1.0 / sum_exp;
            for r in 0..reduce_size {
                let idx = base_offset + r * inner_size;
                out_data[idx] *= inv_sum;
            }
        }
    }

    Tensor::from_vec(out_data, a.dims())
}

/// Root Mean Square Normalization (RMSNorm):
///
/// $$y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \text{weight}$$
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    let rank = x.rank();
    if rank == 0 {
        return Err(TensorError::EmptyTensor);
    }

    let hidden_dim = x.dims()[rank - 1];
    if weight.numel() != hidden_dim {
        return Err(TensorError::ShapeMismatch {
            expected: vec![hidden_dim],
            actual: weight.dims().to_vec(),
        });
    }

    let x_contig = x.contiguous();
    let weight_contig = weight.contiguous();
    let x_slice = x_contig.as_slice()?;
    let w_slice = weight_contig.as_slice()?;

    let total_rows = x.numel() / hidden_dim;
    let mut out_data = vec![0.0f32; x.numel()];

    for r in 0..total_rows {
        let row_start = r * hidden_dim;
        let row_x = &x_slice[row_start..row_start + hidden_dim];
        let row_out = &mut out_data[row_start..row_start + hidden_dim];

        let mut sum_sq = 0.0f32;
        for &val in row_x {
            sum_sq += val * val;
        }
        let mean_sq = sum_sq / (hidden_dim as f32);
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();

        for i in 0..hidden_dim {
            row_out[i] = row_x[i] * rms_inv * w_slice[i];
        }
    }

    Tensor::from_vec(out_data, x.dims())
}

/// Sigmoid Linear Unit (SiLU / Swish activation):
///
/// $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
pub fn silu(a: &Tensor) -> Tensor {
    let contig = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()
    };
    let slice = contig.as_slice().unwrap();
    let mut out = Vec::with_capacity(slice.len());
    for &x in slice {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        out.push(x * sigmoid);
    }
    Tensor::from_vec(out, a.dims()).unwrap()
}

/// Gaussian Error Linear Unit (GELU, tanh approximation):
///
/// $$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$
pub fn gelu(a: &Tensor) -> Tensor {
    let contig = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()
    };
    let slice = contig.as_slice().unwrap();
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut out = Vec::with_capacity(slice.len());

    for &x in slice {
        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        let val = 0.5 * x * (1.0 + inner.tanh());
        out.push(val);
    }
    Tensor::from_vec(out, a.dims()).unwrap()
}

// ============================================================================
// Matrix Multiplication (matmul)
// ============================================================================

/// Optimized 2D matrix multiplication kernel: `C = A x B`.
///
/// Uses cache-aware **(i, k, j) loop ordering** where the innermost loop iterates
/// over `j` (the contiguous dimension of B and C in row-major layout).
/// This eliminates strided column cache-misses and allows compiler auto-vectorization (SIMD).
#[allow(clippy::too_many_arguments)]
fn matmul_2d_ikj(
    a_data: &[f32],
    a_offset: usize,
    a_stride_m: usize,
    a_stride_k: usize,
    b_data: &[f32],
    b_offset: usize,
    b_stride_k: usize,
    b_stride_n: usize,
    c_data: &mut [f32],
    c_offset: usize,
    m: usize,
    k_dim: usize,
    n: usize,
) {
    let c_stride_m = n;

    for i in 0..m {
        let a_row_off = a_offset + i * a_stride_m;
        let c_row_off = c_offset + i * c_stride_m;

        for k in 0..k_dim {
            let a_val = a_data[a_row_off + k * a_stride_k];
            let b_row_off = b_offset + k * b_stride_k;

            // When B is contiguous (b_stride_n == 1), this loop is purely sequential
            if b_stride_n == 1 {
                let c_row = &mut c_data[c_row_off..c_row_off + n];
                let b_row = &b_data[b_row_off..b_row_off + n];
                for j in 0..n {
                    c_row[j] += a_val * b_row[j];
                }
            } else {
                for j in 0..n {
                    c_data[c_row_off + j] += a_val * b_data[b_row_off + j * b_stride_n];
                }
            }
        }
    }
}

/// Naive textbook 2D matrix multiplication kernel: `C = A x B` with (i, j, k) loop ordering.
///
/// This naive order suffers from severe cache misses on B because accessing B[k, j]
/// jumps across memory rows by `b_stride_k` elements on every step of the innermost loop `k`.
#[allow(clippy::too_many_arguments)]
fn matmul_2d_naive_ijk(
    a_data: &[f32],
    a_offset: usize,
    a_stride_m: usize,
    a_stride_k: usize,
    b_data: &[f32],
    b_offset: usize,
    b_stride_k: usize,
    b_stride_n: usize,
    c_data: &mut [f32],
    c_offset: usize,
    m: usize,
    k_dim: usize,
    n: usize,
) {
    let c_stride_m = n;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..k_dim {
                let a_val = a_data[a_offset + i * a_stride_m + k * a_stride_k];
                let b_val = b_data[b_offset + k * b_stride_k + j * b_stride_n];
                sum += a_val * b_val;
            }
            c_data[c_offset + i * c_stride_m + j] = sum;
        }
    }
}

/// Tiled / Blocked 2D matrix multiplication kernel to maximize L1/L2 cache residency.
#[allow(clippy::too_many_arguments)]
fn matmul_2d_tiled(
    a_data: &[f32],
    a_offset: usize,
    a_stride_m: usize,
    a_stride_k: usize,
    b_data: &[f32],
    b_offset: usize,
    b_stride_k: usize,
    b_stride_n: usize,
    c_data: &mut [f32],
    c_offset: usize,
    m: usize,
    k_dim: usize,
    n: usize,
    tile_size: usize,
) {
    let c_stride_m = n;
    let ts = if tile_size == 0 { 32 } else { tile_size };

    for i0 in (0..m).step_by(ts) {
        let i_end = (i0 + ts).min(m);
        for k0 in (0..k_dim).step_by(ts) {
            let k_end = (k0 + ts).min(k_dim);
            for j0 in (0..n).step_by(ts) {
                let j_end = (j0 + ts).min(n);

                for i in i0..i_end {
                    let a_row_off = a_offset + i * a_stride_m;
                    let c_row_off = c_offset + i * c_stride_m;

                    for k in k0..k_end {
                        let a_val = a_data[a_row_off + k * a_stride_k];
                        let b_row_off = b_offset + k * b_stride_k;

                        for j in j0..j_end {
                            c_data[c_row_off + j] += a_val * b_data[b_row_off + j * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}

/// Parallel 2D matrix multiplication kernel using Rayon.
///
/// Distributes rows of M across available CPU worker threads.
#[allow(clippy::too_many_arguments)]
fn matmul_2d_parallel(
    a_data: &[f32],
    a_offset: usize,
    a_stride_m: usize,
    a_stride_k: usize,
    b_data: &[f32],
    b_offset: usize,
    b_stride_k: usize,
    b_stride_n: usize,
    c_data: &mut [f32],
    c_offset: usize,
    m: usize,
    k_dim: usize,
    n: usize,
) {
    use rayon::prelude::*;

    let target_slice = &mut c_data[c_offset..c_offset + m * n];
    target_slice
        .par_chunks_exact_mut(n)
        .enumerate()
        .for_each(|(i, c_row)| {
            let a_row_off = a_offset + i * a_stride_m;
            for k in 0..k_dim {
                let a_val = a_data[a_row_off + k * a_stride_k];
                let b_row_off = b_offset + k * b_stride_k;

                if b_stride_n == 1 {
                    let b_row = &b_data[b_row_off..b_row_off + n];
                    for j in 0..n {
                        c_row[j] += a_val * b_row[j];
                    }
                } else {
                    for j in 0..n {
                        c_row[j] += a_val * b_data[b_row_off + j * b_stride_n];
                    }
                }
            }
        });
}

/// Naive Matrix Multiplication (primarily for benchmarking and analytical comparison).
pub fn matmul_naive(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.rank() != 2 || b.rank() != 2 {
        return Err(TensorError::IncompatibleDimensions(
            "matmul_naive currently supports 2D matrices".into(),
        ));
    }
    let m = a.dims()[0];
    let k_a = a.dims()[1];
    let k_b = b.dims()[0];
    let n = b.dims()[1];

    if k_a != k_b {
        return Err(TensorError::IncompatibleDimensions(format!(
            "Inner matrix dimensions do not match: [{}, {}] x [{}, {}]",
            m, k_a, k_b, n
        )));
    }

    let mut c_data = vec![0.0f32; m * n];
    matmul_2d_naive_ijk(
        a.raw_data(),
        a.offset(),
        a.strides()[0],
        a.strides()[1],
        b.raw_data(),
        b.offset(),
        b.strides()[0],
        b.strides()[1],
        &mut c_data,
        0,
        m,
        k_a,
        n,
    );
    Tensor::from_vec(c_data, &[m, n])
}

/// Tiled Matrix Multiplication (cache-blocking for large matrices).
pub fn matmul_tiled(a: &Tensor, b: &Tensor, tile_size: usize) -> Result<Tensor, TensorError> {
    if a.rank() != 2 || b.rank() != 2 {
        return Err(TensorError::IncompatibleDimensions(
            "matmul_tiled currently supports 2D matrices".into(),
        ));
    }
    let m = a.dims()[0];
    let k_a = a.dims()[1];
    let k_b = b.dims()[0];
    let n = b.dims()[1];

    if k_a != k_b {
        return Err(TensorError::IncompatibleDimensions(format!(
            "Inner matrix dimensions do not match: [{}, {}] x [{}, {}]",
            m, k_a, k_b, n
        )));
    }

    let mut c_data = vec![0.0f32; m * n];
    matmul_2d_tiled(
        a.raw_data(),
        a.offset(),
        a.strides()[0],
        a.strides()[1],
        b.raw_data(),
        b.offset(),
        b.strides()[0],
        b.strides()[1],
        &mut c_data,
        0,
        m,
        k_a,
        n,
        tile_size,
    );
    Tensor::from_vec(c_data, &[m, n])
}

/// Multi-threaded parallel Matrix Multiplication.
pub fn matmul_parallel(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.rank() != 2 || b.rank() != 2 {
        return Err(TensorError::IncompatibleDimensions(
            "matmul_parallel currently supports 2D matrices".into(),
        ));
    }
    let m = a.dims()[0];
    let k_a = a.dims()[1];
    let k_b = b.dims()[0];
    let n = b.dims()[1];

    if k_a != k_b {
        return Err(TensorError::IncompatibleDimensions(format!(
            "Inner matrix dimensions do not match: [{}, {}] x [{}, {}]",
            m, k_a, k_b, n
        )));
    }

    let mut c_data = vec![0.0f32; m * n];
    matmul_2d_parallel(
        a.raw_data(),
        a.offset(),
        a.strides()[0],
        a.strides()[1],
        b.raw_data(),
        b.offset(),
        b.strides()[0],
        b.strides()[1],
        &mut c_data,
        0,
        m,
        k_a,
        n,
    );
    Tensor::from_vec(c_data, &[m, n])
}


/// General Matrix Multiplication (`matmul`).
///
/// Supports:
/// * 2D matrix multiplication: `[M, K] x [K, N] -> [M, N]`
/// * Batched multi-dimensional matrix multiplication: `[..., M, K] x [..., K, N] -> [..., M, N]`
///   with full broadcasting support on the batch dimensions.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let rank_a = a.rank();
    let rank_b = b.rank();

    if rank_a < 2 || rank_b < 2 {
        return Err(TensorError::IncompatibleDimensions(format!(
            "matmul requires at least 2 dimensions, found rank_a={}, rank_b={}",
            rank_a, rank_b
        )));
    }

    let m = a.dims()[rank_a - 2];
    let k_a = a.dims()[rank_a - 1];
    let k_b = b.dims()[rank_b - 2];
    let n = b.dims()[rank_b - 1];

    if k_a != k_b {
        return Err(TensorError::IncompatibleDimensions(format!(
            "Inner matrix dimensions do not match: A is [..., {}, {}], B is [..., {}, {}]",
            m, k_a, k_b, n
        )));
    }

    // 2D case
    if rank_a == 2 && rank_b == 2 {
        let mut c_data = vec![0.0f32; m * n];
        let a_raw = a.raw_data();
        let b_raw = b.raw_data();

        matmul_2d_ikj(
            a_raw,
            a.offset(),
            a.strides()[0],
            a.strides()[1],
            b_raw,
            b.offset(),
            b.strides()[0],
            b.strides()[1],
            &mut c_data,
            0,
            m,
            k_a,
            n,
        );

        return Tensor::from_vec(c_data, &[m, n]);
    }

    // Batched case: broadcast leading dimensions
    let batch_a = &a.dims()[..rank_a - 2];
    let batch_b = &b.dims()[..rank_b - 2];

    let shape_batch_a = Shape::new(batch_a);
    let shape_batch_b = Shape::new(batch_b);
    let out_batch_shape = Shape::broadcast_shapes(&shape_batch_a, &shape_batch_b)?;

    let mut out_dims = out_batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);

    let batch_strides_a = shape_batch_a.broadcast_strides(
        &a.strides()[..rank_a - 2],
        &out_batch_shape,
    )?;
    let batch_strides_b = shape_batch_b.broadcast_strides(
        &b.strides()[..rank_b - 2],
        &out_batch_shape,
    )?;

    let num_batches = out_batch_shape.numel();
    let batch_rank = out_batch_shape.rank();
    let matrix_size_c = m * n;
    let mut c_data = vec![0.0f32; num_batches * matrix_size_c];

    let a_raw = a.raw_data();
    let b_raw = b.raw_data();
    let a_stride_m = a.strides()[rank_a - 2];
    let a_stride_k = a.strides()[rank_a - 1];
    let b_stride_k = b.strides()[rank_b - 2];
    let b_stride_n = b.strides()[rank_b - 1];

    let mut coord = vec![0; batch_rank];
    for b_idx in 0..num_batches {
        let mut off_a = a.offset();
        let mut off_b = b.offset();
        for r in 0..batch_rank {
            off_a += coord[r] * batch_strides_a[r];
            off_b += coord[r] * batch_strides_b[r];
        }

        let off_c = b_idx * matrix_size_c;

        matmul_2d_ikj(
            a_raw,
            off_a,
            a_stride_m,
            a_stride_k,
            b_raw,
            off_b,
            b_stride_k,
            b_stride_n,
            &mut c_data,
            off_c,
            m,
            k_a,
            n,
        );

        // Advance batch coordinate
        for r in (0..batch_rank).rev() {
            coord[r] += 1;
            if coord[r] < out_batch_shape.dims()[r] {
                break;
            }
            coord[r] = 0;
        }
    }

    Tensor::from_vec(c_data, &out_dims)
}
