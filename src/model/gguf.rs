use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::tensor::{DType, Tensor};

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub offset: usize,
    pub size_in_bytes: usize,
}

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    UnexpectedEof,
    InvalidUtf8(String),
    UnknownValueType(u32),
    UnknownGgmlType(u32),
    TensorNotFound(String),
    AlignmentError,
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GGUF I/O error: {}", e),
            Self::InvalidMagic(m) => write!(f, "Invalid GGUF magic: 0x{:08X}", m),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {}", v),
            Self::UnexpectedEof => write!(f, "Unexpected end of file while parsing GGUF"),
            Self::InvalidUtf8(s) => write!(f, "Invalid UTF-8 in GGUF metadata: {}", s),
            Self::UnknownValueType(t) => write!(f, "Unknown GGUF value type: {}", t),
            Self::UnknownGgmlType(t) => write!(f, "Unknown GGML tensor type: {}", t),
            Self::TensorNotFound(name) => write!(f, "Tensor '{}' not found in GGUF file", name),
            Self::AlignmentError => write!(f, "Tensor data offset is misaligned"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub enum GgufStorage {
    Mmap(Mmap),
    Owned(Arc<Vec<u8>>),
}

impl GgufStorage {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m.as_ref(),
            Self::Owned(v) => v.as_slice(),
        }
    }
}

/// GGUF Model File loader with zero-copy memory mapping.
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    storage: GgufStorage,
    tensor_data_start: usize,
}

impl GgufFile {
    /// Opens and memory-maps a GGUF file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse_storage(GgufStorage::Mmap(mmap))
    }

    /// Parses a GGUF file from an in-memory byte slice (used for synthetic test models).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GgufError> {
        Self::parse_storage(GgufStorage::Owned(Arc::new(bytes.to_vec())))
    }

    fn parse_storage(storage: GgufStorage) -> Result<Self, GgufError> {
        let bytes = storage.as_slice();
        let mut cursor = 0;

        // 1. Header
        let magic = read_u32(bytes, &mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = read_u32(bytes, &mut cursor)?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = read_u64(bytes, &mut cursor)? as usize;
        let metadata_kv_count = read_u64(bytes, &mut cursor)? as usize;

        // 2. Metadata Key-Value pairs
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_string(bytes, &mut cursor)?;
            let val = read_value(bytes, &mut cursor)?;
            metadata.insert(key, val);
        }

        // Determine tensor alignment (default 32 bytes)
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::Uint32(val)) => *val as usize,
            Some(GgufValue::Uint64(val)) => *val as usize,
            _ => 32,
        };

        // 3. Tensor Information
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(bytes, &mut cursor)?;
            let rank = read_u32(bytes, &mut cursor)? as usize;

            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_u64(bytes, &mut cursor)? as usize);
            }
            // GGUF stores dimensions in column-major order [cols, rows]; reverse to row-major [rows, cols]
            shape.reverse();

            let ggml_type = read_u32(bytes, &mut cursor)?;
            let dtype = match ggml_type {
                0 => DType::F32,
                1 => DType::F16,
                2 => DType::Q4_0,
                8 => DType::Q8_0,
                other => return Err(GgufError::UnknownGgmlType(other)),
            };

            let offset = read_u64(bytes, &mut cursor)? as usize;
            let numel: usize = shape.iter().product();
            let size_in_bytes = match dtype {
                DType::F32 => numel * 4,
                DType::F16 => numel * 2,
                DType::Q8_0 => (numel / 32) * 34,
                DType::Q4_0 => (numel / 32) * 18,
                _ => numel,
            };

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                    size_in_bytes,
                },
            );
        }

        // 4. Align tensor data offset
        let tensor_data_start = (cursor + alignment - 1) & !(alignment - 1);
        if tensor_data_start > bytes.len() && !tensors.is_empty() {
            return Err(GgufError::UnexpectedEof);
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            storage,
            tensor_data_start,
        })
    }

    /// Loads and returns a Tensor by name.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor, GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

        let start = self.tensor_data_start + info.offset;
        let end = start + info.size_in_bytes;

        let bytes = self.storage.as_slice();
        if end > bytes.len() {
            return Err(GgufError::UnexpectedEof);
        }

        let slice = &bytes[start..end];

        match info.dtype {
            DType::F32 => {
                let num_floats = info.size_in_bytes / 4;
                let mut float_vec = Vec::with_capacity(num_floats);
                for chunk in slice.chunks_exact(4) {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    float_vec.push(val);
                }
                Tensor::from_vec(float_vec, &info.shape)
                    .map_err(|e| GgufError::InvalidUtf8(e.to_string()))
            }
            DType::F16 => {
                // Dequantize / convert F16 to F32
                let num_floats = info.size_in_bytes / 2;
                let mut float_vec = Vec::with_capacity(num_floats);
                for chunk in slice.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    float_vec.push(f16_to_f32(bits));
                }
                Tensor::from_vec(float_vec, &info.shape)
                    .map_err(|e| GgufError::InvalidUtf8(e.to_string()))
            }
            DType::Q8_0 | DType::Q4_0 => {
                // Dequantize to F32 for initial FP32 transformer pass
                let numel: usize = info.shape.iter().product();
                let mut float_vec = vec![0.0f32; numel];
                if info.dtype == DType::Q8_0 {
                    dequantize_q8_0(slice, &mut float_vec);
                } else {
                    dequantize_q4_0(slice, &mut float_vec);
                }
                Tensor::from_vec(float_vec, &info.shape)
                    .map_err(|e| GgufError::InvalidUtf8(e.to_string()))
            }
            _ => Err(GgufError::UnknownGgmlType(info.dtype as u32)),
        }
    }

    /// Retrieves an architecture string, e.g. "llama".
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ============================================================================
// Low-level GGUF Binary Reader Functions
// ============================================================================

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, GgufError> {
    if *cursor >= bytes.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let val = bytes[*cursor];
    *cursor += 1;
    Ok(val)
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, GgufError> {
    if *cursor + 2 > bytes.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let val = u16::from_le_bytes([bytes[*cursor], bytes[*cursor + 1]]);
    *cursor += 2;
    Ok(val)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, GgufError> {
    if *cursor + 4 > bytes.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let val = u32::from_le_bytes([
        bytes[*cursor],
        bytes[*cursor + 1],
        bytes[*cursor + 2],
        bytes[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(val)
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, GgufError> {
    if *cursor + 8 > bytes.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(arr))
}

fn read_string(bytes: &[u8], cursor: &mut usize) -> Result<String, GgufError> {
    let len = read_u64(bytes, cursor)? as usize;
    if *cursor + len > bytes.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let str_bytes = &bytes[*cursor..*cursor + len];
    *cursor += len;
    String::from_utf8(str_bytes.to_vec())
        .map_err(|e| GgufError::InvalidUtf8(e.to_string()))
}

fn read_value(bytes: &[u8], cursor: &mut usize) -> Result<GgufValue, GgufError> {
    let type_id = read_u32(bytes, cursor)?;
    read_value_by_type(bytes, cursor, type_id)
}

fn read_value_by_type(
    bytes: &[u8],
    cursor: &mut usize,
    type_id: u32,
) -> Result<GgufValue, GgufError> {
    match type_id {
        0 => Ok(GgufValue::Uint8(read_u8(bytes, cursor)?)),
        1 => Ok(GgufValue::Int8(read_u8(bytes, cursor)? as i8)),
        2 => Ok(GgufValue::Uint16(read_u16(bytes, cursor)?)),
        3 => Ok(GgufValue::Int16(read_u16(bytes, cursor)? as i16)),
        4 => Ok(GgufValue::Uint32(read_u32(bytes, cursor)?)),
        5 => Ok(GgufValue::Int32(read_u32(bytes, cursor)? as i32)),
        6 => {
            let u = read_u32(bytes, cursor)?;
            Ok(GgufValue::Float32(f32::from_bits(u)))
        }
        7 => Ok(GgufValue::Bool(read_u8(bytes, cursor)? != 0)),
        8 => Ok(GgufValue::String(read_string(bytes, cursor)?)),
        9 => {
            let elem_type = read_u32(bytes, cursor)?;
            let count = read_u64(bytes, cursor)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_value_by_type(bytes, cursor, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(bytes, cursor)?)),
        11 => Ok(GgufValue::Int64(read_u64(bytes, cursor)? as i64)),
        12 => {
            let u = read_u64(bytes, cursor)?;
            Ok(GgufValue::Float64(f64::from_bits(u)))
        }
        other => Err(GgufError::UnknownValueType(other)),
    }
}

// ============================================================================
// Dequantization Helpers (F16, Q8_0, Q4_0)
// ============================================================================

/// Converts IEEE 754 half-precision float (f16) to single-precision (f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 0x0001) as u32;
    let exp = ((h >> 10) & 0x001f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 0;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x03ff;
            let f_exp = (127 - 15 + 1 - e) as u32;
            let f_mant = m << 13;
            f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
        }
    } else if exp == 31 {
        // Infinity or NaN
        let f_exp = 255u32;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    } else {
        // Normalized
        let f_exp = (exp + (127 - 15)) << 23;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | f_exp | f_mant)
    }
}

/// Dequantizes Q8_0 blocks (each block = 1x f16 scale + 32x i8 values).
pub fn dequantize_q8_0(raw: &[u8], out: &mut [f32]) {
    let num_blocks = raw.len() / 34;
    for b in 0..num_blocks {
        let block_offset = b * 34;
        let scale_bits = u16::from_le_bytes([raw[block_offset], raw[block_offset + 1]]);
        let scale = f16_to_f32(scale_bits);

        let out_offset = b * 32;
        for i in 0..32 {
            let q = raw[block_offset + 2 + i] as i8;
            out[out_offset + i] = (q as f32) * scale;
        }
    }
}

/// Dequantizes Q4_0 blocks (each block = 1x f16 scale + 16 bytes containing 32x 4-bit nibbles).
pub fn dequantize_q4_0(raw: &[u8], out: &mut [f32]) {
    let num_blocks = raw.len() / 18;
    for b in 0..num_blocks {
        let block_offset = b * 18;
        let scale_bits = u16::from_le_bytes([raw[block_offset], raw[block_offset + 1]]);
        let scale = f16_to_f32(scale_bits);

        let out_offset = b * 32;
        for i in 0..16 {
            let byte = raw[block_offset + 2 + i];
            let q0 = ((byte & 0x0F) as i8) - 8;
            let q1 = (((byte >> 4) & 0x0F) as i8) - 8;

            out[out_offset + i] = (q0 as f32) * scale;
            out[out_offset + i + 16] = (q1 as f32) * scale;
        }
    }
}

// ============================================================================
// Synthetic GGUF File Builder for Unit Testing
// ============================================================================

/// Helper to serialize an in-memory valid GGUF binary buffer for testing.
pub struct GgufBuilder {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<(String, Vec<usize>, DType, Vec<u8>)>,
    alignment: usize,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    pub fn add_string_meta(&mut self, key: &str, val: &str) {
        self.metadata
            .push((key.to_string(), GgufValue::String(val.to_string())));
    }

    pub fn add_u32_meta(&mut self, key: &str, val: u32) {
        self.metadata.push((key.to_string(), GgufValue::Uint32(val)));
    }

    pub fn add_tensor_f32(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        let mut raw = Vec::with_capacity(data.len() * 4);
        for &f in data {
            raw.extend_from_slice(&f.to_le_bytes());
        }
        self.tensors.push((
            name.to_string(),
            shape.to_vec(),
            DType::F32,
            raw,
        ));
    }

    pub fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();

        // 1. Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // 2. Metadata
        for (k, v) in &self.metadata {
            write_string_buf(&mut buf, k);
            write_value_buf(&mut buf, v);
        }

        // 3. Tensor Info
        let mut current_offset = 0usize;
        for (name, shape, dtype, data) in &self.tensors {
            write_string_buf(&mut buf, name);
            let rank = shape.len() as u32;
            buf.extend_from_slice(&rank.to_le_bytes());
            // Write dims reversed (column-major in GGUF)
            for &d in shape.iter().rev() {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            let ggml_type = match dtype {
                DType::F32 => 0u32,
                DType::F16 => 1u32,
                DType::Q4_0 => 2u32,
                DType::Q8_0 => 8u32,
                _ => 0u32,
            };
            buf.extend_from_slice(&ggml_type.to_le_bytes());
            buf.extend_from_slice(&(current_offset as u64).to_le_bytes());
            current_offset += data.len();
        }

        // 4. Align tensor data
        let current_len = buf.len();
        let aligned_len = (current_len + self.alignment - 1) & !(self.alignment - 1);
        buf.resize(aligned_len, 0);

        // 5. Tensor Binary Data
        for (_, _, _, data) in &self.tensors {
            buf.extend_from_slice(data);
        }

        buf
    }
}

fn write_string_buf(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn write_value_buf(buf: &mut Vec<u8>, val: &GgufValue) {
    match val {
        GgufValue::Uint32(v) => {
            buf.extend_from_slice(&4u32.to_le_bytes()); // type 4
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::String(s) => {
            buf.extend_from_slice(&8u32.to_le_bytes()); // type 8
            write_string_buf(buf, s);
        }
        _ => {}
    }
}
