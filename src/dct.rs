// Copyright (c) 2015-2017 The `img_hash` Crate Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::f32::consts::{PI, SQRT_2};
use std::ops::{Deref, DerefMut, Index, IndexMut};

pub const SIZE_MULTIPLIER: u32 = 2;
pub const SIZE_MULTIPLIER_U: usize = SIZE_MULTIPLIER as usize;

struct IdxCol<D> {
    data: D,
    col: usize,
    rowstride: usize,
}

impl<D> Index<usize> for IdxCol<D> where D: Deref, D::Target: Index<usize> {
    type Output = <<D as Deref>::Target as Index<usize>>::Output;
    #[inline(always)]
    fn index(&self, idx: usize) -> &Self::Output {
       &self.data[idx * self.rowstride + self.col]
    }
}

impl<D> IndexMut<usize> for IdxCol<D> where D: DerefMut, D::Target: IndexMut<usize> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
       &mut self.data[idx * self.rowstride + self.col]
    }
}

pub enum Coefficients {
    Square(Box<[f32]>),
    // a concatenation of the width coefficients and height coefficients
    Rect(Box<[f32]>)
}

impl Coefficients {
    pub fn precompute(width: u32, height: u32) -> Self {
        if width == height {
            Coefficients::Square(precompute_coeff(width).collect())
        } else {
            Coefficients::Rect(precompute_coeff(width).chain(precompute_coeff(height)).collect())
        }
    }

    pub fn row(&self, rowstride: usize) -> &[f32] {
        match *self {
            Coefficients::Square(ref square) => square,
            Coefficients::Rect(ref rect) => &rect[..rowstride],
        }
    }

    pub fn column(&self, rowstride: usize) -> &[f32] {
        match *self {
            Coefficients::Square(ref square) => square,
            Coefficients::Rect(ref rect) => &rect[rowstride..],
        }
    }
}

/// Precompute the 1D DCT coefficients for a given hash size.
fn precompute_coeff(size: u32) -> impl Iterator<Item=f32> {
    // The DCT hash uses a hash size larger than the user provided, so we have to
    // precompute a matrix of the right size
    let size =  size * SIZE_MULTIPLIER;

    (0 .. size).flat_map(move |i|
        (0 .. size).map(move |j| (PI * i as f32 * (2 * j + 1) as f32 / (2 * size) as f32).cos())
    )
}

fn dct_1d<I: ?Sized, O: ?Sized>(input: &I, output: &mut O, len: usize, coeff: &[f32])
where I: Index<usize, Output=f32>, O: IndexMut<usize, Output=f32> {
    for i in 0 .. len {
        let mut z = 0.0;

        for j in 0 .. len {
            z += input[j] * coeff[i * len + j];
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
pub fn dct_2d(packed_2d: &[f32], rowstride: usize, coeff: &Coefficients) -> Vec<f32> {
    // 2D DCT implemented in O(n ^2) time by performing a 1D DCT on the rows
    // and then on the columns

    assert_eq!(packed_2d.len() % rowstride, 0);

    let mut scratch = vec![0f32; packed_2d.len() * 2];

    {
        let (col_pass, row_pass) = scratch.split_at_mut(packed_2d.len());
    
        for (row_in, row_out) in packed_2d.chunks(rowstride)
                .zip(row_pass.chunks_mut(rowstride)) {                
            dct_1d(row_in, row_out, rowstride, coeff.row(rowstride));
        }

        for col in 0 .. rowstride {
            let col_in = IdxCol { data: &mut *row_pass, col, rowstride };
            // we overwrite the row pass with the column pass
            let mut col_out = IdxCol { data: &mut *col_pass, col, rowstride };
            dct_1d(&col_in, &mut col_out, rowstride, coeff.column(rowstride));
        }
    }

    scratch.truncate(packed_2d.len());
    scratch
}

/// Crop the values off a 1D-packed 2D DCT
///
/// Generic for easier testing
pub fn crop_2d_dct<T: Copy>(mut packed: Vec<T>, rowstride: usize) -> Vec<T> {
    // assert that the rowstride was previously multiplied by SIZE_MULTIPLIER
    assert_eq!(rowstride % SIZE_MULTIPLIER_U, 0);
    assert!(rowstride / SIZE_MULTIPLIER_U > 0, "rowstride cannot be cropped: {}", rowstride);

    let new_rowstride = rowstride / SIZE_MULTIPLIER_U;

    for new_row in 0 .. packed.len() / (rowstride * SIZE_MULTIPLIER_U) {
        let (dest, src) = packed.split_at_mut(new_row * new_rowstride + rowstride);
        let dest_start = dest.len() - new_rowstride;
        let src_start = new_rowstride * new_row;
        let src_end = src_start + new_rowstride;
        dest[dest_start..].copy_from_slice(&src[src_start..src_end]);
    }

    let new_len = packed.len() / (SIZE_MULTIPLIER_U * SIZE_MULTIPLIER_U);
    packed.truncate(new_len);

    packed
}

#[test]
fn test_crop_2d_dct() {
    let packed: Vec<i32> = (0 .. 64).collect();
    assert_eq!(
        crop_2d_dct(packed.clone(), 8),
        [
            0, 1, 2, 3, // 4, 5, 6, 7
            8, 9, 10, 11, // 12, 13, 14, 15
            16, 17, 18, 19, // 20, 21, 22, 23,
            24, 25, 26, 27, // 28, 29, 30, 31,
            // 32 .. 64
        ]
    );
}