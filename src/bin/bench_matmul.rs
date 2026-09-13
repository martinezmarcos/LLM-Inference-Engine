use std::time::Instant;
use mini_llm::tensor::{matmul, matmul_naive, matmul_parallel, matmul_tiled, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("mini-llm: Matrix Multiplication (MatMul) Benchmark");
    println!("Comparing: Naive (ijk) vs Optimized (ikj) vs Tiled vs Parallel");
    println!("============================================================\n");

    let sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512)];

    for &(m, k, n) in &sizes {
        println!("------------------------------------------------------------");
        println!("Matrix dimensions: M={}, K={}, N={}", m, k, n);
        let total_flops = 2.0 * (m as f64) * (k as f64) * (n as f64);

        let a = Tensor::randn(&[m, k], 42);
        let b = Tensor::randn(&[k, n], 99);

        // 1. Naive (ijk)
        let iters_naive = if m >= 512 { 2 } else { 10 };
        let start = Instant::now();
        for _ in 0..iters_naive {
            let _ = matmul_naive(&a, &b)?;
        }
        let dur_naive = start.elapsed().as_secs_f64() / (iters_naive as f64);
        let gflops_naive = (total_flops / dur_naive) / 1e9;
        println!(
            "  Naive (i, j, k)    : {:8.2} ms | {:6.2} GFLOPs/s | 1.00x baseline",
            dur_naive * 1000.0,
            gflops_naive
        );

        // 2. Optimized (ikj cache-friendly)
        let iters = 20;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = matmul(&a, &b)?;
        }
        let dur_opt = start.elapsed().as_secs_f64() / (iters as f64);
        let gflops_opt = (total_flops / dur_opt) / 1e9;
        let speedup_opt = dur_naive / dur_opt;
        println!(
            "  Optimized (i, k, j): {:8.2} ms | {:6.2} GFLOPs/s | {:5.2}x speedup",
            dur_opt * 1000.0,
            gflops_opt,
            speedup_opt
        );

        // 3. Tiled (cache-blocking)
        let start = Instant::now();
        for _ in 0..iters {
            let _ = matmul_tiled(&a, &b, 32)?;
        }
        let dur_tiled = start.elapsed().as_secs_f64() / (iters as f64);
        let gflops_tiled = (total_flops / dur_tiled) / 1e9;
        let speedup_tiled = dur_naive / dur_tiled;
        println!(
            "  Tiled (32x32 block): {:8.2} ms | {:6.2} GFLOPs/s | {:5.2}x speedup",
            dur_tiled * 1000.0,
            gflops_tiled,
            speedup_tiled
        );

        // 4. Parallel (Rayon multithreaded)
        let start = Instant::now();
        for _ in 0..iters {
            let _ = matmul_parallel(&a, &b)?;
        }
        let dur_par = start.elapsed().as_secs_f64() / (iters as f64);
        let gflops_par = (total_flops / dur_par) / 1e9;
        let speedup_par = dur_naive / dur_par;
        println!(
            "  Parallel (Rayon)   : {:8.2} ms | {:6.2} GFLOPs/s | {:5.2}x speedup",
            dur_par * 1000.0,
            gflops_par,
            speedup_par
        );
    }

    println!("\n============================================================");
    println!("Explanation:");
    println!("* Naive (i, j, k) constantly jumps across rows in matrix B, triggering L1 cache misses on every access.");
    println!("* Optimized (i, k, j) iterates sequentially along contiguous rows of B and C, maximizing 64-byte cache-line utilization and enabling compiler SIMD auto-vectorization.");
    println!("* Parallel distributes independent output rows across all available physical CPU cores.");
    println!("============================================================\n");
    Ok(())
}
