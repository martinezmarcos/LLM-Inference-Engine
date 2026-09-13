use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    TokenNotFound(usize),
    InvalidUtf8(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenNotFound(id) => write!(f, "Token ID {} not found in vocabulary", id),
            Self::InvalidUtf8(msg) => write!(f, "Invalid UTF-8 sequence during decode: {}", msg),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: Vec<String>,
    merges: HashMap<(String, String), usize>,
    bos_id: Option<usize>,
    eos_id: Option<usize>,
    unk_id: Option<usize>,
    add_bos_token: bool,
    byte_level_bpe: bool,
}

impl BpeTokenizer {
    pub fn new(
        vocab: HashMap<String, usize>,
        merges: HashMap<(String, String), usize>,
        bos_id: Option<usize>,
        eos_id: Option<usize>,
        unk_id: Option<usize>,
    ) -> Self {
        let mut reverse_vocab = vec![String::new(); vocab.len()];
        for (token, &id) in &vocab {
            if id < reverse_vocab.len() {
                reverse_vocab[id] = token.clone();
            }
        }

        let byte_level_bpe = vocab.contains_key("\u{0120}");

        Self {
            vocab,
            reverse_vocab,
            merges,
            bos_id,
            eos_id,
            unk_id,
            add_bos_token: true,
            byte_level_bpe,
        }
    }

    pub fn from_gguf(metadata: &HashMap<String, crate::model::gguf::GgufValue>) -> Option<Self> {
        let tokens_val = metadata.get("tokenizer.ggml.tokens")?;
        let tokens_arr = match tokens_val {
            crate::model::gguf::GgufValue::Array(arr) => arr,
            _ => return None,
        };

        let mut vocab = HashMap::with_capacity(tokens_arr.len());
        let mut reverse_vocab = Vec::with_capacity(tokens_arr.len());

        for (idx, item) in tokens_arr.iter().enumerate() {
            if let crate::model::gguf::GgufValue::String(s) = item {
                vocab.insert(s.clone(), idx);
                reverse_vocab.push(s.clone());
            }
        }

        let mut merges = HashMap::new();
        if let Some(crate::model::gguf::GgufValue::Array(m_arr)) = metadata.get("tokenizer.ggml.merges") {
            for (rank, item) in m_arr.iter().enumerate() {
                if let crate::model::gguf::GgufValue::String(merge_str) = item {
                    if let Some((p1, p2)) = merge_str.split_once(' ') {
                        merges.insert((p1.to_string(), p2.to_string()), rank);
                    }
                }
            }
        }

        let bos_id = match metadata.get("tokenizer.ggml.bos_token_id") {
            Some(crate::model::gguf::GgufValue::Uint32(v)) => Some(*v as usize),
            _ => None,
        };
        let eos_id = match metadata.get("tokenizer.ggml.eos_token_id") {
            Some(crate::model::gguf::GgufValue::Uint32(v)) => Some(*v as usize),
            _ => None,
        };
        let unk_id = match metadata.get("tokenizer.ggml.unknown_token_id") {
            Some(crate::model::gguf::GgufValue::Uint32(v)) => Some(*v as usize),
            _ => None,
        };
        let add_bos_token = match metadata.get("tokenizer.ggml.add_bos_token") {
            Some(crate::model::gguf::GgufValue::Bool(b)) => *b,
            _ => bos_id.is_some(),
        };

        let byte_level_bpe = vocab.contains_key("\u{0120}");

        Some(Self {
            vocab,
            reverse_vocab,
            merges,
            bos_id,
            eos_id,
            unk_id,
            add_bos_token,
            byte_level_bpe,
        })
    }

    pub fn new_default() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = Vec::new();
        let mut merges = HashMap::new();

        let special = ["<unk>", "<s>", "</s>"];
        for (i, &s) in special.iter().enumerate() {
            vocab.insert(s.to_string(), i);
            reverse_vocab.push(s.to_string());
        }

        for b in 0..=255u8 {
            let s = (b as char).to_string();
            let id = vocab.len();
            vocab.insert(s.clone(), id);
            reverse_vocab.push(s);
        }

        let sample_merges = [
            ("t", "h"),
            ("th", "e"),
            ("a", "n"),
            ("i", "n"),
            ("o", "n"),
            ("e", "r"),
            ("r", "e"),
            ("e", "s"),
            ("o", "u"),
            ("i", "s"),
            ("a", "t"),
            ("o", "r"),
            ("l", "i"),
            ("f", "e"),
            ("li", "fe"),
            ("m", "e"),
            ("me", "an"),
            ("mean", "ing"),
            (" ", "t"),
            (" ", "th"),
            (" ", "the"),
            (" ", "m"),
            (" ", "is"),
            (" ", "of"),
            (" ", "li"),
            (" ", "life"),
        ];

        for (rank, &(p1, p2)) in sample_merges.iter().enumerate() {
            merges.insert((p1.to_string(), p2.to_string()), rank);
            let merged = format!("{}{}", p1, p2);
            if !vocab.contains_key(&merged) {
                let id = vocab.len();
                vocab.insert(merged.clone(), id);
                reverse_vocab.push(merged);
            }
        }

        Self {
            vocab,
            reverse_vocab,
            merges,
            bos_id: Some(1),
            eos_id: Some(2),
            unk_id: Some(0),
            add_bos_token: true,
            byte_level_bpe: false,
        }
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        let should_add_bos = add_bos && self.add_bos_token;
        if text.is_empty() {
            return match (should_add_bos, self.bos_id) {
                (true, Some(bos)) => vec![bos],
                _ => Vec::new(),
            };
        }

        let mut token_ids = Vec::new();
        if should_add_bos {
            if let Some(bos) = self.bos_id {
                token_ids.push(bos);
            }
        }

        let prepared = if self.byte_level_bpe {
            text.replace(' ', "\u{0120}").replace('\n', "\u{010A}")
        } else {
            text.to_string()
        };

        let mut pieces: Vec<String> = prepared.chars().map(|c| c.to_string()).collect();

        loop {
            if pieces.len() < 2 {
                break;
            }

            let mut best_pair: Option<(usize, (String, String))> = None;
            let mut best_rank = usize::MAX;

            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some((i, pair));
                    }
                }
            }

            match best_pair {
                Some((idx, (p1, p2))) => {
                    let merged = format!("{}{}", p1, p2);
                    pieces[idx] = merged;
                    pieces.remove(idx + 1);
                }
                None => break,
            }
        }

        for piece in pieces {
            if let Some(&id) = self.vocab.get(&piece) {
                token_ids.push(id);
            } else if let Some(unk) = self.unk_id {
                token_ids.push(unk);
            }
        }

        token_ids
    }

    pub fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError> {
        let mut text = String::new();

        for &id in tokens {
            if Some(id) == self.bos_id || Some(id) == self.eos_id {
                continue;
            }

            if id < self.reverse_vocab.len() {
                let piece = &self.reverse_vocab[id];
                let cleaned = piece.replace('\u{0120}', " ").replace('\u{010A}', "\n");
                text.push_str(&cleaned);
            } else {
                return Err(TokenizerError::TokenNotFound(id));
            }
        }

        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn bos_id(&self) -> Option<usize> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<usize> {
        self.eos_id
    }

    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }
}
