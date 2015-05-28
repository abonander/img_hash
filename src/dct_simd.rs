extern crate simdty;
extern crate llvmint;

use self::simdty::f64x2;

use std::f64::consts::{PI, SQRT_2};

pub fn dct_1dx2(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    assert!(data.len() == 2, "`dct_1dx2 needs a slice of 2!");

    let data = vec_to_simd(data);
    let mut out = vec![f64x2(0.0, 0.0); data.len()];

    for (u, out) in (0 .. data.len()).zip(out.iter_mut()) {
        let u_mult = u as f64 * PI / 2.0 * data.len() as f64;

        for x in 0 .. data.len() {
            let mult = valx2(u_mult * (2 * x + 1) as f64);

            *out += cosx2(data[x] * mult);
        }

        if u == 0 {
            *out *= valx2(1.0 / SQRT_2);
        }
    }

    simd_to_vec(out)
}

fn vec_to_simd(data: &[Vec<f64>]) -> Vec<f64x2> {
    let len = data[0].len();
    assert!(
        data.iter().all(|vec| vec.len() == len), 
        "All vectors passed to `dct_1dx2` must be the same length!"
    );

    let mut out = Vec::with_capacity(len);

    for (&data0, &data1) in data[0].iter().zip(data[1].iter()) {
        out.push(f64x2(data0, data1));    
    }

    out
}

fn simd_to_vec(simd: Vec<f64x2>) -> Vec<Vec<f64>> {
    let len = simd.len();
    let vec = || Vec::with_capacity(len);

    let (mut out0, mut out1) = (vec(), vec());

    for f64x2(data0, data1) in simd.into_iter() {
        out0.push(data0);
        out1.push(data1);
    }

    vec![out0, out1]
}

#[inline(always)]
fn cosx2(val: f64x2) -> f64x2 {
    unsafe { llvmint::cos_v2f64(val) }
}

#[inline(always)]
fn valx2(val: f64) -> f64x2 {
    f64x2(val, val)
}
