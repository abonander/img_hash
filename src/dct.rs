// Copyright (c) 2015-2017 The `img_hash` Crate Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use super::Columns;

use std::cell::RefCell;
use std::f32::consts::{PI, SQRT_2};
use std::ops::{Index, IndexMut};

pub const SIZE_MULTIPLIER: u32 = 2;

struct ColumnsMut<'a, T: 'a> {
    data: &'a mut [T],
    rowstride: usize,
    curr: usize,
}

impl<'a, T: 'a> ColumnsMut<'a, T> {
    #[inline(always)]
    fn from_slice(data: &'a mut [T], rowstride: usize) -> Self {
        ColumnsMut {
            data,
            rowstride,
            curr: 0,
        }
    }
}

impl<'a, T: 'a> Iterator for ColumnsMut<'a, T> {
    type Item = ColumnMut<'a, T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.rowstride {
           let data = unsafe { &mut *(&mut self.data[self.curr..] as *mut [T]) };
            self.curr += 1;
            Some(ColumnMut {
                data,
                rowstride: self.rowstride,
            })
        } else {
            None
        }
    }
}

struct ColumnMut<'a, T: 'a> {
    data: &'a mut [T],
    rowstride: usize,
}

impl<'a, T: 'a> Index<usize> for ColumnMut<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: usize) -> &T {
       &self.data[idx * self.rowstride]
    }
}

impl<'a, T: 'a> IndexMut<usize> for ColumnMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut T {
       &mut self.data[idx * self.rowstride]
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

    (0 .. size).flat_map(|i|
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
    assert_eq!(packed_2d.len() % rowstride, 0);

    let mut scratch = Vec::with_capacity(packed_2d.len() * 2);
    unsafe { scratch.set_len(packed_2d.len() * 2); }

    {
        let (col_pass, row_pass) = scratch.split_at_mut(packed_2d.len());
    
        for (row_in, row_out) in packed_2d.chunks(rowstride)
                .zip(row_pass.chunks_mut(rowstride)) {                
            dct_1d(row_in, row_out, rowstride, coeff.row(rowstride));
        }

        for (col_in, mut col_out) in Columns::from_slice(row_pass, rowstride)
                .zip(ColumnsMut::from_slice(col_pass, rowstride)) {
            dct_1d(&col_in, &mut col_out, rowstride, coeff.column(rowstride));
        }
    }

    scratch.truncate(packed_2d.len());
    scratch
}

/*
#[cfg(feature = "simd")]
mod dct_simd {
    use simdty::f32x2;

    use std::f32::consts::{PI, SQRT_2};
    
    macro_rules! valx2 ( ($val:expr) => ( ::simdty::f32x2($val, $val) ) );

    const PI: f32x2 = valx2!(PI);
    const ONE_DIV_SQRT_2: f32x2 = valx2!(1 / SQRT_2);
    const SQRT_2: f32x2 = valx2!(SQRT_2);

    pub dct_rows(vals: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(vals.len());

        for pair in vals.iter().chunks(2) {
            if pair.len() == 2 {
                let vals = pair[0].iter().cloned().zip(pair[1].iter().cloned())
                    .map(f32x2)
                    .collect();

                dct_1dx2(vals);


        
        }
    }

    fn dct_1dx2(vec: Vec<f32x2>) -> Vec<f32x2> {
        let mut out = Vec::with_capacity(vec.len());

        for u in 0 .. vec.len() {
            let mut z = valx2!(0.0);

            for x in 0 .. vec.len() {
                z += vec[x] * cos_approx(
                    PI * valx2!(
                        u as f32 * (2 * x + 1) as f32 
                            / (2 * vec.len()) as f32
                    )
                );
            }

            if u == 0 {
                z *= ONE_DIV_SQRT_2;
            }

            out.insert(u, z / valx2!(2.0));
        }

        out 
    }

    fn cos_approx(x2: f32x2) -> f32x2 {
        #[inline(always)]
        fn powi(val: f32x2, pow: i32) -> f32x2 {
            unsafe { llvmint::powi_v2f32(val, pow) }
        }

        let x2 = powi(val, 2);
        let x4 = powi(val, 4);
        let x6 = powi(val, 6);
        let x8 = powi(val, 8);

        valx2!(1.0) - (x2 / valx2!(2.0)) + (x4 / valx2!(24.0)) 
            - (x6 / valx2!(720.0)) + (x8 / valx2!(40320.0))
    }
}
*/

