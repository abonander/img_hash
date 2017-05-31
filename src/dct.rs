// Copyright (c) 2015-2017 The `img_hash` Crate Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::f64::consts::{PI, SQRT_2};

use bit_vec::BitVec;

use columns::*;

use super::{DCT2DFunc, HashImage, prepare_image};

thread_local! {
    static PRECOMPUTED_MATRIX: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

include!("generated/dct_header.rs");
include!("generated/dct.rs");

const HASH_SIZE_MULTIPLIER: u32 = 2;

pub fn custom_dct_hash<I: HashImage>(img: &I, size: u32, func: DCT2DFunc) -> BitVec {
    let large_size = size * HASH_SIZE_MULTIPLIER;

    let hash_values: Vec<_> = prepare_image(img, large_size, large_size)
        .into_iter().map(|val| (val as f64) / 255.0).collect();

    let mut dct = func.call(&hash_values, large_size as usize);
    crop_2d(&mut dct, size);

    let mean = dct.iter().fold(0f64, |b, &a| a + b) / dct.len() as f64;

    dct.into_iter().map(|x| x >= mean).collect()
}

fn dct_hash_dyn<I: HashImage>(img: &I, size: u32) -> BitVec {
    let large_size = size * HASH_SIZE_MULTIPLIER;

    let hash_values: Vec<_> = prepare_image(img, large_size, large_size)
        .into_iter().map(|val| (val as f64) / 255.0).collect();

    let mut dct = dct_2d(&hash_values, large_size as usize);
    crop_2d(&mut dct, size);

    let mean = dct.iter().fold(0f64, |b, &a| a + b) / dct.len() as f64;

    dct.into_iter().map(|x| x >= mean).collect()
}

/// Precompute the DCT matrix for a given hash size and memoize it in thread-local
/// storage.
///
///
/// If a precomputed matrix was already stored, even of the same length, it will be overwritten.
///
/// This can produce a significant runtime savings (an order of magnitude on the author's machine)
/// when performing multiple hashing runs with the same hash size, as compared to not performing
/// this step.
///
/// ## Note
/// This only affects the built-in DCT hash (`HashType::DCT`). It also includes
/// the hash size multiplier applied by the DCT hash algorithm, so just pass the same
/// hash size that you would to `ImageHash::hash()`.
///
/// ## Note: Thread-Local
/// Because this uses thread-local storage, this will need to be called on every thread
/// that will be using the DCT hash for the runtime benefit. You can also
/// have precomputed matrices of different sizes for each thread.
pub fn precompute_dct_matrix(size: u32) {
    // The DCT hash uses a hash size larger than the user provided, so we have to
    // precompute a matrix of the right size
    let size = size * HASH_SIZE_MULTIPLIER;
    precomp_exact(size);
}

/// Precompute a DCT matrix of the exact given size.
pub fn precomp_exact(size: u32) {
    PRECOMPUTED_MATRIX.with(|matrix| precompute_matrix(size as usize, &mut matrix.borrow_mut()));
}

#[cfg(feature = "bench")]
pub fn clear_precomputed_matrix() {
    PRECOMPUTED_MATRIX.with(|matrix| matrix.borrow_mut().clear());
}

fn precompute_matrix(size: usize, matrix: &mut Vec<f64>) {
    matrix.resize(size * size, 0.0);

    for i in 0 .. size {
        for j in 0 .. size {
            matrix[i * size + j] = (PI * i as f64 * (2 * j + 1) as f64 / (2 * size) as f64).cos();
        }
    }
}

fn with_precomputed_matrix<F>(size: usize, with_fn: F) -> bool
where F: FnOnce(&[f64]) {
    PRECOMPUTED_MATRIX.with(|matrix| {
        let matrix = matrix.borrow();

        if matrix.len() == size * size {
            with_fn(&matrix);
            true
        } else {
            false
        }
    })
}

pub fn dct_1d<I: IndexLen<Output=f64> + ?Sized, O: IndexMutLen<Output=f64> + ?Sized>(input: &I, output: &mut O, len: usize) {
    if with_precomputed_matrix(len, |matrix| dct_1d_precomputed(input, output, len, matrix)) {
        return;
    }

    dct_1d_dyn(input, output, len);
}

fn dct_1d_precomputed<I: ?Sized, O: ?Sized>(input: &I, output: &mut O, len: usize, matrix: &[f64])
where I: IndexLen<Output=f64>, O: IndexMutLen<Output=f64> {
    for i in 0 .. len {
        let mut z = 0.0;

        for j in 0 .. len {
            z += input[j] * matrix[i * len + j];
        }

        if i == 0 {
            z *= 1.0 / SQRT_2;
        }

        output[i] = z / 2.0;
    }
}

#[inline(always)]
fn dct_1d_dyn<I: ?Sized, O: ?Sized>(input: &I, output: &mut O, len: usize)
where I: IndexLen<Output = f64>, O: IndexMutLen<Output = f64> {
    for i in 0 .. len {
        let mut z = 0.0;

        for j in 0 .. len {
            z += input[j] * (
                PI * i as f64 * (2 * j + 1) as f64
                    / (2 * len) as f64
            ).cos();
        }

        if i == 0 {
            z *= 1.0 / SQRT_2;
        }

        output[i] = z / 2.0;
    }
}

/// Perform a 2D DCT on a 1D-packed vector with a given rowstride.
///
/// E.g. a vector of length 9 with a rowstride of 3 will be processed as a 3x3 matrix.
///
/// Returns a vector of the same size packed in the same way.
pub fn dct_2d(packed_2d: &[f64], rowstride: usize) -> Vec<f64> {
    assert_eq!(packed_2d.len() % rowstride, 0);

    let mut scratch = Vec::with_capacity(packed_2d.len() * 2);
    unsafe { scratch.set_len(packed_2d.len() * 2); }

    {
        let (col_pass, row_pass) = scratch.split_at_mut(packed_2d.len());

        for (row_in, row_out) in packed_2d.chunks(rowstride)
                .zip(row_pass.chunks_mut(rowstride)) {
            dct_1d(row_in, row_out, rowstride);
        }

        for (col_in, mut col_out) in Columns::from_slice(row_pass, rowstride)
                .zip(ColumnsMut::from_slice(col_pass, rowstride)) {
            dct_1d(&col_in, &mut col_out, rowstride);
        }
    }

    scratch.truncate(packed_2d.len());
    scratch
}

fn crop_2d(packed: &mut Vec<f64>, size: u32) {
    let size = size as usize;
    let large_size = size * (HASH_SIZE_MULTIPLIER as usize);

    for i in 1 .. size {
        for j in 0 .. size {
            packed[i * size + j] = packed[i * large_size + j];
        }
    }

    packed.truncate(size * size);
}
