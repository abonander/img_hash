// Copyright (c) 2015-2017 The `img_hash` Crate Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustdct::{DCTplanner, Type2And3};
use std::sync::Arc;

pub const SIZE_MULTIPLIER: u32 = 2;
pub const SIZE_MULTIPLIER_U: usize = SIZE_MULTIPLIER as usize;

const BLOCK_SIZE: usize = 16;

#[inline(always)]
unsafe fn transpose_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, block_x: usize, block_y: usize) {
    for inner_x in 0..BLOCK_SIZE {
        for inner_y in 0..BLOCK_SIZE {
            let x = block_x * BLOCK_SIZE + inner_x;
            let y = block_y * BLOCK_SIZE + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

#[inline(always)]
unsafe fn transpose_endcap_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, block_x: usize, block_y: usize, block_width: usize, block_height: usize) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = block_x * BLOCK_SIZE + inner_x;
            let y = block_y * BLOCK_SIZE + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
// Use "Loop tiling" to improve cache-friendliness
pub fn transpose<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    assert_eq!(width*height, input.len());
    assert_eq!(width*height, output.len());

    let x_block_count = width / BLOCK_SIZE;
    let y_block_count = height / BLOCK_SIZE;

    let remainder_x = width - x_block_count * BLOCK_SIZE;
    let remainder_y = height - y_block_count * BLOCK_SIZE;

    for y_block in 0..y_block_count {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input, output,
                    width, height,
                    x_block, y_block);
            }
        }

        //if the width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height,
                    x_block_count, y_block,
                    remainder_x, BLOCK_SIZE);
            }
        }
    }

    //if the height is not cleanly divisible by BLOCK_SIZE, there are still a few columns that haven't been transposed
    if remainder_y > 0 {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height,
                    x_block, y_block_count,
                    BLOCK_SIZE, remainder_y,
                );
            }
        }

        //if the width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height,
                    x_block_count, y_block_count,
                    remainder_x, remainder_y);
            }
        }
    }
}

pub struct DctCtxt {
    row_dct: Arc<Type2And3<f32>>,
    col_dct: Arc<Type2And3<f32>>,
    width: usize,
    height: usize,
}

impl DctCtxt {
    pub fn new(width: u32, height: u32) -> Self {
        let mut planner = DCTplanner::new();
        let width = width as usize * SIZE_MULTIPLIER_U;
        let height = height as usize * SIZE_MULTIPLIER_U;

        DctCtxt {
            row_dct: planner.plan_dct2(width),
            col_dct: planner.plan_dct2(height),
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width as u32
    }

    pub fn height(&self) -> u32 {
        self.height as u32
    }

    /// Perform a 2D DCT on a 1D-packed vector with a given `width x height`.
    ///
    /// ### Panics
    /// If `self.width * self.height != packed_2d.len()`
    pub fn dct_2d(&self, mut packed_2d: Vec<f32>) -> Vec<f32> {
        let Self { ref row_dct, ref col_dct, width, height } = *self;

        assert_eq!(width * height, packed_2d.len());
        let mut scratch = vec![0f32; packed_2d.len()];

        for (row_in, row_out) in packed_2d.chunks_mut(width)
            .zip(scratch.chunks_mut(width)) {
            row_dct.process_dct2(row_in, row_out);
        }

        transpose(width, height, &scratch, &mut packed_2d);

        for (row_in, row_out) in packed_2d.chunks_mut(height)
            .zip(scratch.chunks_mut(height)) {
            col_dct.process_dct2(row_in, row_out);
        }

        transpose(width, height, &scratch, &mut packed_2d);

        packed_2d
    }

    pub fn crop_2d(&self, mut packed: Vec<f32>) -> Vec<f32> {
        crop_2d_dct(packed, self.width)
    }
}

/// Crop the values off a 1D-packed 2D DCT
///
/// Generic for easier testing
fn crop_2d_dct<T: Copy>(mut packed: Vec<T>, rowstride: usize) -> Vec<T> {
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

#[test]
fn test_transpose() {

}