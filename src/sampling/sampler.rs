pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> usize;
}

#[derive(Debug, Default, Clone)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> usize {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }
}

#[derive(Debug, Clone)]
pub struct ConfigurableSampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng_state: u64,
}

impl ConfigurableSampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature: if temperature <= 0.0 { 1e-4 } else { temperature },
            top_k,
            top_p: top_p.clamp(0.0, 1.0),
            rng_state: if seed == 0 { 0x12345678abcdef } else { seed },
        }
    }

    fn rand_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64 / (u64::MAX as f64 + 1.0)) as f32
    }
}

impl Sampler for ConfigurableSampler {
    fn sample(&mut self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }

        if self.temperature < 1e-3 {
            return GreedySampler.sample(logits);
        }

        let inv_temp = 1.0 / self.temperature;
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val * inv_temp))
            .collect();

        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if self.top_k > 0 && self.top_k < indexed_logits.len() {
            indexed_logits.truncate(self.top_k);
        }

        let max_val = indexed_logits[0].1;
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = indexed_logits
            .iter()
            .map(|&(idx, logit)| {
                let p = (logit - max_val).exp();
                sum_exp += p;
                (idx, p)
            })
            .collect();

        for item in &mut probs {
            item.1 /= sum_exp;
        }

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut cutoff_idx = probs.len();

            for (i, &(_, p)) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff_idx);

            let nucleus_sum: f32 = probs.iter().map(|item| item.1).sum();
            for item in &mut probs {
                item.1 /= nucleus_sum;
            }
        }

        let r = self.rand_f32();
        let mut cum = 0.0f32;
        for &(token_id, prob) in &probs {
            cum += prob;
            if r <= cum {
                return token_id;
            }
        }

        probs[0].0
    }
}
