use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use mini_llm::inference::InferenceEngine;
use mini_llm::model::{GgufFile, ModelConfig, TransformerWeights};
use mini_llm::sampling::ConfigurableSampler;
use mini_llm::tokenizer::BpeTokenizer;
use mini_llm::transformer::TransformerDecoder;

struct CliArgs {
    model_path: Option<String>,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        model_path: None,
        prompt: "the meaning of life is".to_string(),
        max_tokens: 30,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        seed: 42,
    };

    let raw_args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < raw_args.len() {
                    args.model_path = Some(raw_args[i + 1].clone());
                    i += 1;
                }
            }
            "--prompt" | "-p" => {
                if i + 1 < raw_args.len() {
                    args.prompt = raw_args[i + 1].clone();
                    i += 1;
                }
            }
            "--max-tokens" | "-n" => {
                if i + 1 < raw_args.len() {
                    args.max_tokens = raw_args[i + 1].parse().unwrap_or(args.max_tokens);
                    i += 1;
                }
            }
            "--temperature" | "-t" => {
                if i + 1 < raw_args.len() {
                    args.temperature = raw_args[i + 1].parse().unwrap_or(args.temperature);
                    i += 1;
                }
            }
            "--top-k" => {
                if i + 1 < raw_args.len() {
                    args.top_k = raw_args[i + 1].parse().unwrap_or(args.top_k);
                    i += 1;
                }
            }
            "--top-p" => {
                if i + 1 < raw_args.len() {
                    args.top_p = raw_args[i + 1].parse().unwrap_or(args.top_p);
                    i += 1;
                }
            }
            "--seed" if i + 1 < raw_args.len() => {
                args.seed = raw_args[i + 1].parse().unwrap_or(args.seed);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    args
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("mini-llm: Systems Engineering LLM Inference Engine in Rust");
    println!("============================================================\n");

    let args = parse_args();

    let (model, tokenizer) = if let Some(ref path_str) = args.model_path {
        let path = Path::new(path_str);
        if path.exists() {
            println!("Loading GGUF model from: {}", path.display());
            let gguf = GgufFile::open(path)?;
            let config = ModelConfig::from_gguf(&gguf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            println!(
                "Model Architecture: {} | Layers: {} | Hidden: {} | Heads: {} | Vocab: {}",
                config.architecture, config.num_layers, config.hidden_dim, config.num_heads, config.vocab_size
            );

            let tokenizer = BpeTokenizer::from_gguf(&gguf.metadata).unwrap_or_else(|| {
                println!("Notice: Model metadata did not contain complete BPE tables, using default BPE tokenizer.");
                BpeTokenizer::new_default()
            });

            let weights = TransformerWeights::load_from_gguf(&gguf, &config)?;
            let model = TransformerDecoder::new(config, weights);
            (model, tokenizer)
        } else {
            println!("Warning: Specified model file '{}' not found.", path_str);
            println!("Initializing compact demo model with synthetic weights...\n");
            let tokenizer = BpeTokenizer::new_default();
            let mut config = ModelConfig::default_compact();
            config.vocab_size = tokenizer.vocab_size();
            let weights = TransformerWeights::random_compact(&config, args.seed);
            (TransformerDecoder::new(config, weights), tokenizer)
        }
    } else {
        println!("No --model path provided. Running standalone demonstration with compact Transformer model.\n");
        let tokenizer = BpeTokenizer::new_default();
        let mut config = ModelConfig::default_compact();
        config.vocab_size = tokenizer.vocab_size();
        let weights = TransformerWeights::random_compact(&config, args.seed);
        (TransformerDecoder::new(config, weights), tokenizer)
    };

    println!("Configuration:");
    println!("  Prompt      : \"{}\"", args.prompt);
    println!("  Max tokens  : {}", args.max_tokens);
    println!("  Temperature : {:.2}", args.temperature);
    println!("  Top-K       : {}", args.top_k);
    println!("  Top-P       : {:.2}", args.top_p);
    println!("------------------------------------------------------------");
    print!("{}", args.prompt);
    io::stdout().flush()?;

    let mut engine = InferenceEngine::new(model, tokenizer);
    let mut sampler = ConfigurableSampler::new(args.temperature, args.top_k, args.top_p, args.seed);

    let mut token_count = 0usize;
    let start_time = Instant::now();

    let _generated = engine.generate(&args.prompt, args.max_tokens, &mut sampler, |token_str| {
        print!("{}", token_str);
        let _ = io::stdout().flush();
        token_count += 1;
        true
    })?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let tok_per_sec = if elapsed > 0.0 { (token_count as f64) / elapsed } else { 0.0 };

    println!("\n------------------------------------------------------------");
    println!(
        "Generation completed: {} tokens in {:.3} s ({:.2} tokens/sec)",
        token_count, elapsed, tok_per_sec
    );
    println!("KV Cache position : {} tokens", engine.model.kv_cache.current_position());
    println!("============================================================");

    Ok(())
}
