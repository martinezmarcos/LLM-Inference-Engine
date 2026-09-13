use std::fmt;

/// Data types supported by the tensor engine.
///
/// Initially, computations run primarily on `F32`.
/// Lower bit-width types (`F16`, `BF16`, `I8`, `Q8_0`, `Q4_0`) are structured
/// to prepare the engine for weight loading and quantization stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    Q8_0,
    Q4_0,
}

impl DType {
    /// Returns the size in bytes of a single element for unquantized types.
    /// For block-quantized types (Q8_0, Q4_0), size per element is fractional.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::Q8_0 => 1, // Effective ~1 byte per weight (34 bytes per 32 weights)
            Self::Q4_0 => 1, // Effective ~0.5 byte per weight
        }
    }

    /// Whether this type is a 32-bit float.
    pub fn is_f32(&self) -> bool {
        matches!(self, Self::F32)
    }

    /// Whether this type is block-quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4_0)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q4_0 => write!(f, "q4_0"),
        }
    }
}
