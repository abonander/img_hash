use super::Columns;

use std::cell::RefCell;
use std::f64::consts::{PI, SQRT_2};
use std::ops::{Index, IndexMut};

struct ColumnsMut<'a, T: 'a> {
    data: &'a mut [T],
    rowstride: usize,
    curr: usize,
}

impl<'a, T: 'a> ColumnsMut<'a, T> {
    #[inline(always)]
    fn from_slice(data: &'a mut [T], rowstride: usize) -> Self {
        ColumnsMut {
            data: data,
            rowstride: rowstride,
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
                data: data,
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

thread_local! {
    static PRECOMPUTED_MATRIX: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

/// Precompute the DCT matrix for a given hash size and memoize it in thread-local
/// storage.
///
/// If a precomputed matrix was already stored, even of the same length, it will be overwritten.
///
/// This can produce a significant runtime savings (an order of magnitude on the author's machine)
/// when performing multiple hashing runs with the same hash, as compared to not performing this
/// step.
///
/// ## Note: Thread-Local
/// Because this uses thread-local storage, this will need to be called on every thread
/// that will be using the DCT hash for the runtime benefit. You can also
/// have precomputed matrices of different sizes for each thread.
pub fn precompute_dct_matrix(size: u32) {
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

pub fn dct_1d<I: Index<usize, Output=f64> + ?Sized, O: IndexMut<usize, Output=f64> + ?Sized>(input: &I, output: &mut O, len: usize) {
    if with_precomputed_matrix(len, |matrix| dct_1d_precomputed(input, output, len, matrix)) {
        return;
    }

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

fn dct_1d_precomputed<I: ?Sized, O: ?Sized>(input: &I, output: &mut O, len: usize, matrix: &[f64])
where I: Index<usize, Output=f64>, O: IndexMut<usize, Output=f64> {
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

/*
#[cfg(feature = "simd")]
mod dct_simd {
    use simdty::f64x2;

    use std::f64::consts::{PI, SQRT_2};
    
    macro_rules! valx2 ( ($val:expr) => ( ::simdty::f64x2($val, $val) ) );

    const PI: f64x2 = valx2!(PI);
    const ONE_DIV_SQRT_2: f64x2 = valx2!(1 / SQRT_2);
    const SQRT_2: f64x2 = valx2!(SQRT_2);

    pub dct_rows(vals: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(vals.len());

        for pair in vals.iter().chunks(2) {
            if pair.len() == 2 {
                let vals = pair[0].iter().cloned().zip(pair[1].iter().cloned())
                    .map(f64x2)
                    .collect();

                dct_1dx2(vals);


        
        }
    }

    fn dct_1dx2(vec: Vec<f64x2>) -> Vec<f64x2> {
        let mut out = Vec::with_capacity(vec.len());

        for u in 0 .. vec.len() {
            let mut z = valx2!(0.0);

            for x in 0 .. vec.len() {
                z += vec[x] * cos_approx(
                    PI * valx2!(
                        u as f64 * (2 * x + 1) as f64 
                            / (2 * vec.len()) as f64
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

    fn cos_approx(x2: f64x2) -> f64x2 {
        #[inline(always)]
        fn powi(val: f64x2, pow: i32) -> f64x2 {
            unsafe { llvmint::powi_v2f64(val, pow) }
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

