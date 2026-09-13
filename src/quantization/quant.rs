use crate::model::gguf::f16_to_f32;

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0x00ff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;

    if exp <= 0 {
        0
    } else if exp >= 31 {
        (sign | 0x7c00) as u16
    } else {
        (sign | ((exp as u32) << 10) | (mant >> 13)) as u16
    }
}

pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(32),
        "Input length must be a multiple of block size 32"
    );

    let num_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(num_blocks * 34);

    for b in 0..num_blocks {
        let block = &data[b * 32..(b + 1) * 32];

        let mut amax = 0.0f32;
        for &x in block {
            let abs_x = x.abs();
            if abs_x > amax {
                amax = abs_x;
            }
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f32_to_f16(d);

        out.extend_from_slice(&d_f16.to_le_bytes());

        for &x in block {
            let q = (x * id).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }

    out
}

pub fn dequantize_q8_0(raw: &[u8]) -> Vec<f32> {
    assert!(
        raw.len().is_multiple_of(34),
        "Q8_0 raw length must be a multiple of 34 bytes"
    );
    let num_blocks = raw.len() / 34;
    let mut out = vec![0.0f32; num_blocks * 32];
    crate::model::gguf::dequantize_q8_0(raw, &mut out);
    out
}

pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(32),
        "Input length must be a multiple of block size 32"
    );

    let num_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);

    for b in 0..num_blocks {
        let block = &data[b * 32..(b + 1) * 32];

        let mut amax = 0.0f32;
        for &x in block {
            let abs_x = x.abs();
            if abs_x > amax {
                amax = abs_x;
            }
        }

        let d = amax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f32_to_f16(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        for i in 0..16 {
            let x0 = block[i];
            let x1 = block[i + 16];

            let q0 = ((x0 * id).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((x1 * id).round().clamp(-8.0, 7.0) as i8 + 8) as u8;

            let byte = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            out.push(byte);
        }
    }

    out
}

pub fn dequantize_q4_0(raw: &[u8]) -> Vec<f32> {
    assert!(
        raw.len().is_multiple_of(18),
        "Q4_0 raw length must be a multiple of 18 bytes"
    );
    let num_blocks = raw.len() / 18;
    let mut out = vec![0.0f32; num_blocks * 32];
    crate::model::gguf::dequantize_q4_0(raw, &mut out);
    out
}

pub fn vec_dot_q8_0(x: &[f32], q8_row: &[u8]) -> f32 {
    let num_blocks = q8_row.len() / 34;
    let mut total_sum = 0.0f32;

    for b in 0..num_blocks {
        let block_off = b * 34;
        let scale_bits = u16::from_le_bytes([q8_row[block_off], q8_row[block_off + 1]]);
        let scale = f16_to_f32(scale_bits);

        let x_off = b * 32;
        let mut block_acc = 0.0f32;

        for i in 0..32 {
            let q = q8_row[block_off + 2 + i] as i8;
            block_acc += x[x_off + i] * (q as f32);
        }

        total_sum += block_acc * scale;
    }

    total_sum
}

pub fn evaluate_quantization_error(original: &[f32], dequantized: &[f32]) -> (f32, f32) {
    assert_eq!(original.len(), dequantized.len());
    let mut sum_sq_err = 0.0f32;
    let mut max_err = 0.0f32;

    for (&orig, &deq) in original.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        sum_sq_err += err * err;
        if err > max_err {
            max_err = err;
        }
    }

    let mse = sum_sq_err / (original.len() as f32);
    (mse, max_err)
}
