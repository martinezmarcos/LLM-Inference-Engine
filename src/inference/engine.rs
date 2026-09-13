use crate::sampling::Sampler;
use crate::tokenizer::BpeTokenizer;
use crate::transformer::TransformerDecoder;

#[derive(Debug)]
pub enum InferenceError {
    TensorError(crate::tensor::TensorError),
    TokenizerError(crate::tokenizer::TokenizerError),
    EmptyPrompt,
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorError(e) => write!(f, "Tensor runtime error: {}", e),
            Self::TokenizerError(e) => write!(f, "Tokenizer error: {}", e),
            Self::EmptyPrompt => write!(f, "Prompt cannot be empty"),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<crate::tensor::TensorError> for InferenceError {
    fn from(e: crate::tensor::TensorError) -> Self {
        Self::TensorError(e)
    }
}

impl From<crate::tokenizer::TokenizerError> for InferenceError {
    fn from(e: crate::tokenizer::TokenizerError) -> Self {
        Self::TokenizerError(e)
    }
}

pub struct InferenceEngine {
    pub model: TransformerDecoder,
    pub tokenizer: BpeTokenizer,
}

impl InferenceEngine {
    pub fn new(model: TransformerDecoder, tokenizer: BpeTokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn generate<S, F>(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampler: &mut S,
        mut on_token: F,
    ) -> Result<String, InferenceError>
    where
        S: Sampler,
        F: FnMut(&str) -> bool,
    {
        if prompt.is_empty() {
            return Err(InferenceError::EmptyPrompt);
        }

        self.model.reset_cache();

        let prompt_tokens = self.tokenizer.encode(prompt, true);
        let mut all_generated_text = String::new();

        let logits = self.model.forward(&prompt_tokens, 0)?;
        let l_slice = logits.as_slice()?;
        let mut current_token = sampler.sample(l_slice);

        let eos_id = self.tokenizer.eos_id();

        for pos in (prompt_tokens.len()..).take(max_new_tokens) {
            if Some(current_token) == eos_id {
                break;
            }

            let token_str = self.tokenizer.decode(&[current_token])?;
            all_generated_text.push_str(&token_str);

            let keep_going = on_token(&token_str);
            if !keep_going {
                break;
            }

            let next_logits = self.model.forward(&[current_token], pos)?;
            let next_slice = next_logits.as_slice()?;
            current_token = sampler.sample(next_slice);
        }

        Ok(all_generated_text)
    }
}
