// Copyright (c) 2015-2017 The `img_hash` Crate Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! A crate that provides several perceptual hashing algorithms for images.
//! Supports images opened with the [image][1] crate from Piston.
//!
//!
//! ### Example
//! Hash two images, then compute their percentage difference.
//!
//! ```ignore
//! extern crate image;
//! extern crate img_hash;
//! 
//! use img_hash::{ImageHash, HashType};
//! 
//! fn main() {
//!     let image1 = image::open(&Path::new("image1.png")).unwrap();
//!     let image2 = image::open(&Path::new("image2.png")).unwrap();
//!     
//!     // These two lines produce hashes with 64 bits (8 ** 2),
//!     // using the Gradient hash, a good middle ground between 
//!     // the performance of Mean and the accuracy of DCT.
//!     let hash1 = ImageHash::hash(&image1, 8, HashType::Gradient);
//!     let hash2 = ImageHash::hash(&image2, 8, HashType::Gradient);
//!     
//!     println!("Image1 hash: {}", hash1.to_base64());
//!     println!("Image2 hash: {}", hash2.to_base64());
//!     
//!     println!("% Difference: {}", hash1.dist_ratio(&hash2));
//! }
//! ```
//! [1]: https://github.com/PistonDevelopers/image
#![deny(missing_docs)]
// Silence feature warnings for test module.
#![cfg_attr(all(test, feature = "bench"), feature(test))]

extern crate bit_vec;

#[cfg(any(test, feature = "rust-image"))]
extern crate image;

extern crate rustc_serialize as serialize;

use serialize::base64::{ToBase64, FromBase64, STANDARD};
// Needs to be fully qualified
pub use serialize::base64::FromBase64Error;

use bit_vec::BitVec;

use dct::dct_2d;

use std::{fmt, hash, ops};

#[cfg(any(test, feature = "rust-image"))]
mod rust_image;

mod dct;

mod block;

pub use dct::precompute_dct_matrix;

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash {
    /// The bits of the hash
    pub bitv: BitVec,
    /// The type of the hash
    pub hash_type: HashType,
}

impl ImageHash {
    /// Create a hash of `img` with a length of `hash_size * hash_size`
    /// (`* 2` when using `HashType::DoubleGradient`)
    /// using the hash algorithm described by `hash_type`.
    pub fn hash<I: HashImage>(img: &I, hash_size: u32, hash_type: HashType) -> ImageHash {
        let hash = hash_type.hash(img, hash_size);

        ImageHash {
            bitv: hash,
            hash_type: hash_type,
        }
    }

    /// Calculate the Hamming distance between this and `other`.
    /// Equivalent to counting the 1-bits of the XOR of the two `BitVec`.
    /// 
    /// Essential to determining the perceived difference between `self` and `other`.
    ///
    /// ###Panics
    /// If `self` and `other` have differing `bitv` lengths or `hash_type` values.
    pub fn dist(&self, other: &ImageHash) -> usize {
        assert!(self.hash_type == other.hash_type,
               "Image hashes must use the same algorithm for proper comparison!");
        assert!(self.bitv.len() == other.bitv.len(), 
                "Image hashes must be the same length for proper comparison!");

        self.bitv.iter().zip(other.bitv.iter())
            .filter(|&(left, right)| left != right).count()
    }

    /// Calculate the Hamming distance between `self` and `other`,
    /// then normalize it to `[0, 1]`, as a fraction of the total bits.
    /// 
    /// Roughly equivalent to the % difference between the two images,
    /// represented as a decimal.
    ///
    /// See `ImageHash::dist()`.
    pub fn dist_ratio(&self, other: &ImageHash) -> f32 {
        self.dist(other) as f32 / self.size() as f32
    }
    
    /// Get the hash size of this image. Should be equal to the number of bits in the hash.
    pub fn size(&self) -> u32 { self.bitv.len() as u32 }

    /// Get the `HashType` that this `ImageHash` was created with.
    pub fn hash_type(&self) -> HashType { self.hash_type }

    /// Build a grayscale image using the bits of the hash, 
    /// setting pixels to white (`0xff`) for `0` and black (`0x00`) for `1`.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bitv.iter()
            .map(|bit| (bit as u8) * 0xff)
            .collect()
    }

    /// Create an `ImageHash` instance from the given Base64-encoded string.
    /// ## Note:
    /// **Not** compatible with Base64-encoded strings created before `HashType` was added.
    ///
    /// Does **not** preserve the internal value of `HashType::UserDCT`.
    /// ## Errors:
    /// Returns a FromBase64Error::InvalidBase64Length when trying to hash a zero-length string
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash, FromBase64Error>{
        let mut data = try!(encoded_hash.from_base64());
        // The hash type should be the first bit of the hash
        if data.len() == 0 {
            return Err(FromBase64Error::InvalidBase64Length);
        }
        let hash_type = HashType::from_byte(data.remove(0));

        Ok(ImageHash{
            bitv: BitVec::from_bytes(&*data),
            hash_type: hash_type,
        })
    }

    /// Get a Base64 string representing the bits of this hash.
    ///
    /// Mostly for printing convenience.
    pub fn to_base64(&self) -> String {
        let mut bytes = self.bitv.to_bytes();
        // Insert the hash type as the first byte.
        bytes.insert(0, self.hash_type.to_byte());

        bytes.to_base64(STANDARD)
    }
}

/// The length of a row in a 2D matrix when packed into a 1D array.
pub type Rowstride = usize;

/// A 2-dimensional Discrete Cosine Transform function that receives 
/// and returns 1-dimensional packed data.
///
/// The function will be provided the pre-hash data as a 1D-packed vector, 
/// which should be interpreted as a 2D matrix with a given rowstride:
///
/// ```notest
/// Pre-hash data:
/// [ 1.0 2.0 3.0 ]
/// [ 4.0 5.0 6.0 ]
/// [ 7.0 8.0 9.0 ]
///
/// Packed: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] (rowstride 3)
/// ```
///
/// The function should then return a new 1D vector of the DCT values packed in the same manner.
#[derive(Copy)]
pub struct DCT2DFunc(pub fn(&[f64], Rowstride) -> Vec<f64>);

impl DCT2DFunc {
    fn as_ptr(&self) -> *const () {
        self.0 as *const ()
    }

    fn call(&self, data: &[f64], rowstride: Rowstride) -> Vec<f64> {
        (self.0)(data, rowstride) 
    } 
}

impl Clone for DCT2DFunc {
    /// Function pointers implement `Copy` but not `Clone`, so this simply returns a copy of `self`.
    fn clone(&self) -> Self {
        *self
    }
}

impl PartialEq for DCT2DFunc {
    /// Naive equality comparison just looking at the numerical values of the function pointers.
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }
}

impl Eq for DCT2DFunc {}

impl fmt::Debug for DCT2DFunc {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.write_fmt(format_args!("DCT2DFunc((fn ptr) {:?})", self.as_ptr()))
    }
}

impl hash::Hash for DCT2DFunc {
    /// Adds the contained function pointer as `usize`.
    fn hash<H>(&self, state: &mut H) where H: hash::Hasher {
       (self.as_ptr() as usize).hash(state) 
    }
}

/// An enum describing the hash algorithms that `img_hash` offers.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum HashType { 
    /// This algorithm first averages the pixels of the reduced-size and color image,
    /// and then compares each pixel to the average.
    ///
    /// Fast, but inaccurate. Really only useful for finding duplicates.
    Mean,
    /// The [Blockhash.io](http://blockhash.io) algorithm.
    ///
    /// Faster than `Mean` but also prone to more collisions and suitable only for finding
    /// duplicates.
    Block,
    /// This algorithm compares each pixel in a row to its neighbor and registers changes in
    /// gradients (e.g. edges and color boundaries).
    ///
    /// More accurate than `Mean` but much faster than `DCT`.
    Gradient,
    /// A version of `Gradient` that adds an extra hash pass orthogonal to the first 
    /// (i.e. on columns in addition to rows).
    /// 
    /// Slower than `Gradient` and produces a double-sized hash, but much more accurate.
    DoubleGradient,
    /// This algorithm runs a Discrete Cosine Transform on the reduced-color and size image,
    /// then compares each datapoint in the transform to the average.
    ///
    /// Slowest by far, but can detect changes in color gamut and sometimes relatively significant edits.
    ///
    /// Call `precompute_dct_matrix()` with your chosen hash size to memoize the DCT matrix for the
    /// given size, which can produce significant speedups in repeated hash runs.
    DCT,
    /// Equivalent to `DCT`, but allows the user to provide their own 2-dimensional DCT function. 
    /// See the `DCT2DFunc` docs for more info.
    ///
    /// Use this variant if you want a specialized or optimized 2D DCT implementation, such as from
    /// [FFTW][1]. (This cannot be the default implementation because of licensing conflicts.)
    ///
    /// [1]: http://www.fftw.org/
    UserDCT(DCT2DFunc),
    /// Discourage complete matches for backwards-compatibility.
    #[doc(hidden)]
    __BackCompat,
}

impl HashType {
    fn hash<I: HashImage>(self, img: &I, hash_size: u32) -> BitVec {
        use HashType::*; 

        match self {
            Mean => mean_hash(img, hash_size),
            Block => block::blockhash(img, hash_size),
            DCT => dct_hash(img, hash_size, DCT2DFunc(dct_2d)),
            Gradient => gradient_hash(img, hash_size),
            DoubleGradient => double_gradient_hash(img, hash_size),
            UserDCT(dct_2d_func) => dct_hash(img, hash_size, dct_2d_func),
            __BackCompat => panic!("`HashType::__BackCompat` is not an actual hash algorithm"),
        }
    }

    fn to_byte(self) -> u8 {
        use HashType::*;

        match self {
            Mean => 1,
            Block => 6,
            DCT => 2,
            Gradient => 3,
            DoubleGradient => 4,
            UserDCT(_) => 5,
            __BackCompat => panic!("`HashType::__BackCompat` is not an actual hash algorithm"),
        }
    }

    fn from_byte(byte: u8) -> HashType {
        use HashType::*;

        match byte {
            1 => Mean,
            2 => DCT,
            3 => Gradient,
            4 => DoubleGradient,
            5 => UserDCT(DCT2DFunc(dct_2d)),
            6 => Block,
            _ => panic!("Byte {:?} cannot be coerced to a `HashType`!", byte),
        }
    }
}

fn mean_hash<I: HashImage>(img: &I, hash_size: u32) -> BitVec {
    let hash_values = prepare_image(img, hash_size, hash_size);

    let mean = hash_values.iter().fold(0u32, |b, &a| a as u32 + b) 
        / hash_values.len() as u32;

    hash_values.into_iter().map(|x| x as u32 >= mean).collect()
}

const DCT_HASH_SIZE_MULTIPLIER: u32 = 4;

fn dct_hash<I: HashImage>(img: &I, hash_size: u32, dct_2d_func: DCT2DFunc) -> BitVec {
    let large_size = (hash_size * DCT_HASH_SIZE_MULTIPLIER) as usize;

    // We take a bigger resize than fast_hash, 
    // then we only take the lowest corner of the DCT
    let hash_values: Vec<_> = prepare_image(img, large_size as u32, large_size as u32)
        .into_iter().map(|val| (val as f64) / 255.0).collect();

    let dct = dct_2d_func.call(&hash_values, large_size);

    let original = (large_size, large_size);
    let new = (hash_size as usize, hash_size as usize);

    let cropped_dct = crop_2d_dct(&dct, original, new);

    let mean = cropped_dct.iter().fold(0f64, |b, &a| a + b) 
        / cropped_dct.len() as f64;

    cropped_dct.into_iter().map(|x| x >= mean).collect()
}

struct Columns<'a, T: 'a> {
    data: &'a [T],
    rowstride: usize,
    curr: usize,
}

impl<'a, T: 'a> Columns<'a, T> {
    #[inline(always)]
    fn from_slice(data: &'a [T], rowstride: usize) -> Self {
        Columns {
            data: data,
            rowstride: rowstride,
            curr: 0,
        }
    }
}

impl<'a, T: 'a> Iterator for Columns<'a, T> {
    type Item = Column<'a, T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.rowstride {
            let data = &self.data[self.curr..];
            self.curr += 1;
            Some(Column {
                data: data,
                rowstride: self.rowstride,
            })
        } else {
            None
        }
    }
}

struct Column<'a, T: 'a> {
    data: &'a [T],
    rowstride: usize,
}

impl<'a, T: 'a> ops::Index<usize> for Column<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: usize) -> &T {
       &self.data[idx * self.rowstride]
    }
}

/// The guts of the gradient hash, 
/// separated so we can reuse them for both `Gradient` and `DoubleGradient`.
fn gradient_hash_impl<I: ops::Index<usize, Output=u8> + ?Sized>(bytes: &I, len: u32, bitv: &mut BitVec) {
    let len = len as usize;

    for i in 1 .. len {
        let this = &bytes[i];
        let last = &bytes[i - 1];

        bitv.push(last < this);
    }
}

fn gradient_hash<I: HashImage>(img: &I, hash_size: u32) -> BitVec {
    // We have one extra pixel in width so we have `hash_size` comparisons per row.
    let bytes = prepare_image(img, hash_size + 1, hash_size);
    let mut bitv = BitVec::with_capacity((hash_size * hash_size) as usize);

    for row in bytes.chunks((hash_size + 1) as usize) {
        gradient_hash_impl(row, hash_size, &mut bitv); 
    }

    bitv
}

fn double_gradient_hash<I: HashImage>(img: &I, hash_size: u32) -> BitVec {
    // We have one extra pixel in each dimension so we have `hash_size` comparisons.
    let rowstride = hash_size + 1;
    let bytes = prepare_image(img, rowstride, rowstride);
    let mut bitv = BitVec::with_capacity((hash_size * hash_size * 2) as usize);

    
    for row in bytes.chunks(rowstride as usize) { 
        gradient_hash_impl(row, rowstride, &mut bitv);
    }

    for column in Columns::from_slice(&bytes, rowstride as usize) {
        gradient_hash_impl(&column, hash_size, &mut bitv);
    }

    bitv
}

/// A trait for describing an image that can be successfully hashed.
///
/// Implement this for custom image types.
pub trait HashImage {
    /// The image type when converted to grayscale.
    type Grayscale: HashImage;

    /// The dimensions of the image as (width, height).
    fn dimensions(&self) -> (u32, u32);

    /// Returns a copy, leaving `self` unmodified.
    fn resize(&self, width: u32, height: u32) -> Self;

    /// Convert `self` to grayscale.
    fn grayscale(&self) -> Self::Grayscale;

    /// Convert `self` to a byte-vector.
    fn to_bytes(self) -> Vec<u8>;

    /// Get the number of pixel channels in the image:
    ///
    /// * 1 -> Grayscale
    /// * 2 -> Grayscale with Alpha
    /// * 3 -> RGB
    /// * 4 -> RGB with Alpha
    fn channel_count() -> u8;

    /// Call `iter_fn` for each pixel in the image, passing `(x, y, [pixel data])`.
    fn foreach_pixel<F>(&self, iter_fn: F) where F: FnMut(u32, u32, &[u8]);
}

fn prepare_image<I: HashImage>(img: &I, width: u32, height: u32) -> Vec<u8> {
    img.resize(width, height).grayscale().to_bytes()
}

/// Crop the values off a 1D-packed 2D DCT
fn crop_2d_dct(packed: &[f64], original: (usize, usize), new: (usize, usize)) -> Vec<f64> {
    let (orig_width, orig_height) = original;

    assert!(packed.len() == orig_width * orig_height);

    let (new_width, new_height) = new;

    assert!(new_width < orig_width && new_height < orig_height);

    (0 .. new_height).flat_map(|y| {
        let start = y * orig_width;
        let end = start + new_width;

        packed[start .. end].iter().cloned()
    }).collect()
}

#[cfg(test)]
mod test {
    extern crate rand;

    use serialize::base64::*;

    use image::{Rgba, ImageBuffer};

    use self::rand::{weak_rng, Rng};

    use super::{DCT2DFunc, HashType, ImageHash};

    type RgbaBuf = ImageBuffer<Rgba<u8>, Vec<u8>>;

    fn gen_test_img(width: u32, height: u32) -> RgbaBuf {
        let len = (width * height * 4) as usize;
        let mut buf = Vec::with_capacity(len);
        unsafe { buf.set_len(len); } // We immediately fill the buffer.
        weak_rng().fill_bytes(&mut *buf);

        ImageBuffer::from_raw(width, height, buf).unwrap()
    }

    #[test]
    fn hash_equality() {
        let test_img = gen_test_img(1024, 1024);
        let hash1 = ImageHash::hash(&test_img, 32, HashType::Mean);
        let hash2 = ImageHash::hash(&test_img, 32, HashType::Mean);

        assert_eq!(hash1, hash2);
    }   

    #[test]
    fn dct_2d_equality() {
        fn dummy_dct(_ : &[f64], _: usize) -> Vec<f64> {
            unimplemented!();
        }

        let dct1 = DCT2DFunc(dummy_dct);
        let dct2 = DCT2DFunc(dummy_dct);

        assert_eq!(dct1, dct2);
    }

    #[test]
    fn dct_2d_inequality() {
        fn dummy_dct(_ : &[f64], _: usize) -> Vec<f64> {
            unimplemented!();
        }

        fn dummy_dct_2(_ : &[f64], _: usize) -> Vec<f64> {
            unimplemented!();
        }

        let dct1 = DCT2DFunc(dummy_dct);
        let dct2 = DCT2DFunc(dummy_dct_2);

        assert!(dct1 != dct2);
    }

    #[test]
    fn size() {
        let test_img = gen_test_img(1024, 1024);
        let hash = ImageHash::hash(&test_img, 32, HashType::Mean);
        assert_eq!(32*32, hash.size());
    }

    #[test]
    fn base64_encoding_decoding() {
        let test_img = gen_test_img(1024, 1024);
        let hash1 = ImageHash::hash(&test_img, 32, HashType::Mean);

        let base64_string = hash1.to_base64();
        let decoded_result = ImageHash::from_base64(&*base64_string);

        assert!(decoded_result.is_ok());

        assert_eq!(decoded_result.unwrap(), hash1);
    }  

    #[test]
    fn base64_error_on_empty() {
        let decoded_result = ImageHash::from_base64("");
        match decoded_result {
            Err(InvalidBase64Length) => (),
            _ => panic!("Expected a invalid length error")
        };
    }

    #[cfg(feature = "bench")]
    mod bench {
        use super::gen_test_img;
        use super::rand::{thread_rng, Rng};

        extern crate test;

        use ::{HashType, ImageHash};
        
        use self::test::Bencher;

        const BENCH_HASH_SIZE: u32 = 8;
        const TEST_IMAGE_SIZE: u32 = 64;

        fn bench_hash(b: &mut Bencher, hash_type: HashType) {
            let test_img = gen_test_img(TEST_IMAGE_SIZE, TEST_IMAGE_SIZE);
        
            b.iter(|| ImageHash::hash(&test_img, BENCH_HASH_SIZE, hash_type));
        }

        macro_rules! bench_hash {
            ($bench_fn:ident : $hash_type:expr) => (
                #[bench]
                fn $bench_fn(b: &mut Bencher) {
                    bench_hash(b, $hash_type);
                }
            )
        }

        bench_hash! { bench_mean_hash : HashType::Mean }
        bench_hash! { bench_gradient_hash : HashType::Gradient }
        bench_hash! { bench_dbl_gradient_hash : HashType::DoubleGradient }
        bench_hash! { bench_block_hash: HashType::Block }


        #[bench]
        fn bench_dct_hash(b: &mut Bencher) {
            ::dct::clear_precomputed_matrix();
            bench_hash(b, HashType::DCT);
        }

        #[bench]
        fn bench_dct_hash_precomp(b: &mut Bencher) {
            ::precompute_dct_matrix(BENCH_HASH_SIZE);
            bench_hash(b, HashType::DCT);
        }

        #[bench]
        fn bench_dct_1d(b: &mut Bencher) {
            const ROW_LEN: usize = 8;
            let mut test_vals = [0f64; ROW_LEN];

            fill_rand(&mut test_vals);

            let mut output = [0f64;  ROW_LEN];

            ::dct::clear_precomputed_matrix();

            // Explicit slicing is necessary
            b.iter(|| ::dct::dct_1d(&test_vals[..], &mut output[..], ROW_LEN));
        
            test::black_box(&output);
        }

        #[bench]
        fn bench_dct_1d_precomp(b: &mut Bencher) {
            const ROW_LEN: usize = 8;
            let mut test_vals = [0f64; ROW_LEN];

            fill_rand(&mut test_vals);

            let mut output = [0f64;  ROW_LEN];

            ::dct::precomp_exact(ROW_LEN as u32);

            // Explicit slicing is necessary
            b.iter(|| ::dct::dct_1d(&test_vals[..], &mut output[..], ROW_LEN));

            test::black_box(&output);
        }

        #[bench]
        fn bench_dct_2d(b: &mut Bencher) {
            const ROWSTRIDE: usize = 8;
            const LEN: usize = ROWSTRIDE * ROWSTRIDE;

            let mut test_vals = [0f64; LEN];

            fill_rand(&mut test_vals);

            ::dct::clear_precomputed_matrix();

            b.iter(|| ::dct::dct_2d(&test_vals[..], ROWSTRIDE));
        }

        #[bench]
        fn bench_dct_2d_precomp(b: &mut Bencher) {
            const ROWSTRIDE: usize = 8;
            const LEN: usize = ROWSTRIDE * ROWSTRIDE;

            let mut test_vals = [0f64; LEN];

            fill_rand(&mut test_vals);

            ::dct::precomp_exact(ROWSTRIDE as u32);

            b.iter(|| ::dct::dct_2d(&test_vals[..], ROWSTRIDE));
        }

        #[inline(never)]
        fn fill_rand(out: &mut [f64]) {
            let mut rng = thread_rng();

            for (val, out) in rng.gen_iter().zip(out) {
                *out = val;
            }
        }
    }
}
