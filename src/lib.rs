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
//! ```rust,no_run
//! extern crate image;
//! extern crate img_hash;
//! 
//! use img_hash::{ImageHash, HashType};
//!
//! fn main() {
//!     let image1 = image::open("image1.png").unwrap();
//!     let image2 = image::open("image2.png").unwrap();
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

// anti-footgun should this ever be compiled on a 16-bit platform
#[target_pointer_width="16"]
compile_error!("img_hash does not support 16-bit targets");

extern crate image;

use dct::dct_2d;

use std::borrow::Cow;
use std::{cmp, fmt, hash, mem, ops, slice};

use image::{GenericImageView, GrayImage, ImageBuffer, Pixel, FilterType};
use image::imageops;

mod dct;

mod alg;

use alg::HashAlg;

use std::marker::PhantomData;

pub type Bytes8 = [u8; 8];

/// Interface for types used for storing hash data.
///
/// This is implemented for `Vec<u8>`, `Box<[u8]>` and arrays that are multiples/combinations of
/// common SIMD register widths (64, 128, 256, 512 bits).
pub trait HashBytes: AsMut<[u8]> {
    /// Construct this type from an iterator of bytes.
    ///
    /// If this type has a finite capacity (i.e. an array) then it can ignore extra data
    /// (the hash API will not create a hash larger than this type can contain).
    fn from_iter<I: Iterator<Item = u8>>(iter: I) -> Self;

    /// Return the maximum capacity of this type, in bytes.
    ///
    /// If this type has an arbitrary/theoretically infinite capacity, return `usize::MAX`.
    fn max_size() -> usize;
}

impl HashBytes for Box<[u8]> {
    fn from_iter<I: Iterator<Item = u8>>(iter: I) -> Self {
        iter.collect()
    }

    fn max_size() -> usize {
        usize::max_value()
    }
}

impl HashBytes for Vec<u8> {
    fn from_iter<I: Iterator<Item=u8>>(iter: I) -> Self {
        iter.collect()
    }

    fn max_size() -> usize {
        usize::max_value()
    }
}

macro_rules! hash_bytes_array {
    ($n:expr) => {
        impl HashBytes for [u8; $n] {
            fn from_iter<I: Iterator<Item=u8>>(mut iter: I) -> Self {
                // optimizer should eliminate this zeroing
                let mut out = Self::default();

                for (src, dest) in iter.by_ref().zip(out.as_mut()) {
                    *dest = src;
                }

                out
            }

            fn check_size(size: usize) -> bool {
                size <= n
            }
        }
    };
    ($($n:expr),+) => {
        hash_bytes_array!($n);
    }
}

hash_bytes_array!(8, 16, 24, 32, 40, 48, 56, 64);

trait BitSet: HashBytes {
    fn from_bools<I: Iterator<Item = bool>>(mut iter: I) -> Self {
        let mut out = Self::default();

        for mut dest in out.as_mut() {
            *dest = iter.by_ref().take(8).fold(0u8, |accum, val| (accum << 1) | (val as u8));
        }

        out
    }

    fn xor(&self, other: &Self) -> Self {
        Self::from_iter(self.iter().zip(other.iter()).map(|(l, r)| l ^ r))
    }

    fn popcnt(&self) -> u32 {
        self.iter().map(|byte| byte.count_ones()).sum()
    }

    fn iter(&self) -> slice::Iter<u8> {
        self.as_ref().iter()
    }
}

impl<T: HashBytes> BitSet for T {}

pub struct HasherConfig<A, B = Bytes8> {
    hash_size: u32,
    gauss_sigmas: Option<[f32; 2]>,
    resize_filter: FilterType,
    dct: bool,
    alg: A,
    vectype: PhantomData<B>,
}

impl<A: HashAlg, B: HashBytes> HasherConfig<A, B> {
    /// Construct a new hasher config with sane, reasonably fast defaults.
    ///
    /// A default hash container type is provided as a default type parameter which is guaranteed
    /// to exactly fit the default hash size.
    pub fn new() -> HasherConfig<B> {
        Self {
            hash_size: 8,
            gauss_sigmas: None,
            resize_filter: FilterType::Lanczos3,
            dct: false,
            alg: A::default(),
            vectype: PhantomData,
        }

    }

    /// Set a new hash size, rounded to the next even integer.
    ///
    /// The number of bits in the resulting hash will be the square of this rounded value.
    /// The value is rounded to support the double-gradient hash which processes a half-size
    /// sample twice, so it produces the same hash size.
    ///
    /// ### Panics
    /// If `(hash_size + 1) * (hash_size + 1)` overflows `usize` or
    /// exceeds `B::max_size() * 8` (saturating).
    ///
    /// ### Recommended Values
    /// As `hash_size` is squared for all algorithms, a very small value is usually sufficient;
    /// see the source of [`Self::new()`](Self::new) for the default.
    ///
    /// If you are accepting user input for this configuration, you may prefer to clamp this value
    /// to below 64 for performance reasons.
    pub fn hash_size(self, hash_size: u32) -> Self {
        let max_hash_bits = B::max_size().saturating_mul(8);

        let plus_one = (hash_size as usize).checked_add(1)
            .expect("oveflowed usize evaluating `hash_size + 1`");
        let req_hash_bits = (plus_one).checked_mul(plus_one)
            .expect("overflowed usize evaluating `(hash_size + 1) * (hash_size + 1)`");

        assert!(req_hash_bits <= max_hash_bits, "`hash_size` too large: {}", hash_size);

        Self { hash_size, ..self  }
    }

    /// Set the filter used to resize images during hashing.
    ///
    /// Note when picking a filter that images are always reduced in size, never enlarged.
    /// Has no effect with the Blockhash algorithm as it does not resize.
    pub fn resize_filter(self, resize_filter: FilterType) -> Self {
        Self { resize_filter, ..self }
    }

    /// Enable preprocessing with the Discrete Cosine Transform (DCT).
    ///
    /// Does nothing when used with the Blockhash.io algorithm which does not scale the image.
    /// (RFC: it would be possible to shoehorn a DCT into the Blockhash algorithm but it's
    /// not clear what benefits, if any, that would provide).
    ///
    /// After conversion to grayscale, the image is scaled down to `hash_size * 2 x hash_size * 2`
    /// and then the Discrete Cosine Transform is performed on the luminance values. The DCT
    /// essentially transforms the 2D image from the spatial domain with luminance values
    /// to a 2D frequency domain where the values are amplitudes of cosine waves. The resulting
    /// 2D matrix is then cropped to the low `hash_size x hash_size` corner and the
    /// configured hash algorithm is performed on that.
    ///
    /// In layman's terms, this essentially converts the image into a mathematical representation
    /// of the "broad strokes" of the data, which allows the subsequent hashing step to ignore
    /// changes that may otherwise produce different hashes, such as significant edits to portions
    /// of the image (recoloring, additions, deletions).
    ///
    /// However, on most machines this usually adds an additional 50-100% to the average hash time.
    ///
    /// This is a very similar process to JPEG compression, although the implementation is too
    /// different for this to be optimized specifically for JPEG encoded images.
    ///
    /// Further Reading:
    /// * http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
    /// * https://en.wikipedia.org/wiki/Discrete_cosine_transform
    pub fn preproc_dct(self) -> Self {
        Self { dct: true, ..self }
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with default sigma values.
    ///
    /// Recommended only for use with the Blockhash.io algorithm as it significantly reduces
    /// entropy in the scaled down image for other algorithms.
    ///
    /// See [`Self::preproc_diff_gauss_sigmas()](Self::preproc_diff_gauss_sigmas) for more info.
    pub fn preproc_diff_gauss(self) -> Self {
        self.preproc_diff_gauss_sigmas(5.0, 10.0)
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with the given sigma values.
    ///
    /// Recommended only for use with the Blockhash.io algorithm as it significantly reduces
    /// entropy in the scaled down image for other algorithms.
    ///
    /// After the image is converted to grayscale, it is blurred with a Gaussian blur using
    /// two different sigmas, and then the images are subtracted from each other. This reduces
    /// the image to just sharp transitions in luminance, i.e. edges. Varying the sigma values
    /// changes how sharp the edges are^[citation needed].
    ///
    /// Further reading:
    /// * https://en.wikipedia.org/wiki/Difference_of_Gaussians
    /// * http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    /// (Difference of Gaussians is an approximation of a Laplacian of Gaussian filter)
    pub fn preproc_diff_gauss_sigmas(self, sigma_a: f32, sigma_b: f32) -> Self {
        Self { gauss_sigmas: Some([simgaA, simgaB]), ..self }
    }


    pub fn hash_alg_mean(self) -> Self {
       Self { alg: HashAlg::Mean, ..self }
    }

    /// Use the Gradient hashing algorithm.
    ///
    /// The image is converted to grayscale, scaled down to `hash_size + 1 x hash_size`,
    /// and then in row-major order the pixels are compared with each other, setting bits
    /// in the hash for each comparison. The extra pixel is needed to have `hash_size` comparisons
    /// per row.
    ///
    /// This hash algorithm is as fast or faster than Mean (because it only traverses the
    /// hash data once) and is more resistant to changes than Mean.
    ///
    /// Further Reading:
    /// http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    pub fn hash_alg_gradient(self) -> Self {
        Self { alg: HashAlg::Gradient, ..self }
    }

    /// Use the Double-Gradiant hashing algorithm.
    ///
    /// An advanced version of [`Self::hash_alg_gradient()`](Self::hash_alg_gradient);
    /// resizes the grayscaled image to `hash_size / 2 + 1 * hash_size / 2 + 1` and compares columns
    /// in addition to rows.
    ///
    /// This algorithm is slightly slower than `hash_alg_gradient()` (resizing the image dwarfs
    /// the hash time in most cases) but the extra comparison direction may improve results (though
    /// you might want to consider increasing `hash_size` to accommodate the extra comparisons).
    pub fn hash_alg_dbl_gradient(self) -> Self {

    }

    /// Create a [`Hasher`](Hasher) from this config.
    ///
    /// If DCT preprocessing was selected, this will precalculate the DCT coefficients for the
    /// chosen hash size.
    pub fn into_hasher(self) -> Hasher<B> {
        let coeffs = if self.dct {
            Some(dct::precompute_coeff(self.hash_size))
        }


    }
}

pub struct Hasher<B = Bytes8> {
    preproc: Preproc,
    hash_size: u32,
    alg: A,
    bytes_type: PhantomData<B>,
}

impl<A, B> Hasher<B> where A: HashAlg, B: HashBytes {
    pub fn hash_image<I>(&self, img: &I) -> ImageHash<B> where I: GenericImageView {
        let hash = self.alg.hash_image(&self.preproc, img, self.hash_size);
        ImageHash { hash }
    }
}

enum CowImage<'a, I> {
    Borrowed(&'a I),
    Owned(ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>),
}

impl<'a, I> CowImage<'a, I> {
    fn grayscale(&self) -> GrayImage {

    }
}

pub(crate) struct Preproc {
    gauss_sigmas: Option<[f32; 2]>,
    dct_coeffs: Option<Box<[f32]>>,
}

impl Preproc {
    pub fn pre_resize<I: GenericImageView + 'static>(&self, image: &I) -> CowImage<I> {
        if let Some([sigma_a, sigma_b]) = self.gauss_sigmas {
            let mut blur_a = imageops::blur(image, sigma_a);
            let blur_b = imageops::blur(image, sigma_b);

            blur_a.pixels_mut().zip(blur_b)
                .for_each(|lpx, rpx| lpx.apply2(&rpx, |lch, rch| lch - rch));

            CowImage::Owned(blur_a)
        } else {
            CowImage::Borrowed(image)
        }
    }

    pub fn pre_hash<I: GenericImageView + 'static>
}

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash<B> {
    hash: B
}

impl ImageHash<B> {
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
        assert_eq!(self.hash_type, other.hash_type,
               "Image hashes must use the same algorithm for proper comparison!");
        assert_eq!(self.bitv.len(), other.bitv.len(),
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
            bitv: HashBytes::from_bytes(&*data),
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
    fn hash<I: HashImage>(self, img: &I, hash_size: u32) -> HashBytes {
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

fn mean_hash<I: HashImage>(img: &I, hash_size: u32) -> HashBytes {
    let hash_values = prepare_image(img, hash_size, hash_size);

    let mean = hash_values.iter().fold(0u32, |b, &a| a as u32 + b) 
        / hash_values.len() as u32;

    hash_values.into_iter().map(|x| x as u32 >= mean).collect()
}

const DCT_HASH_SIZE_MULTIPLIER: u32 = 4;

fn dct_hash<I: HashImage>(img: &I, hash_size: u32, dct_2d_func: DCT2DFunc) -> HashBytes {
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
fn gradient_hash_impl<I: ops::Index<usize, Output=u8> + ?Sized>(bytes: &I, len: u32, bitv: &mut HashBytes) {
    let len = len as usize;

    for i in 1 .. len {
        let this = &bytes[i];
        let last = &bytes[i - 1];

        bitv.push(last < this);
    }
}

fn gradient_hash<I: HashImage>(img: &I, hash_size: u32) -> HashBytes {
    // We have one extra pixel in width so we have `hash_size` comparisons per row.
    let bytes = prepare_image(img, hash_size + 1, hash_size);
    let mut bitv = HashBytes::with_capacity((hash_size * hash_size) as usize);

    for row in bytes.chunks((hash_size + 1) as usize) {
        gradient_hash_impl(row, hash_size, &mut bitv); 
    }

    bitv
}

fn double_gradient_hash<I: HashImage>(img: &I, hash_size: u32) -> HashBytes {
    // We have one extra pixel in each dimension so we have `hash_size` comparisons.
    let rowstride = hash_size + 1;
    let bytes = prepare_image(img, rowstride, rowstride);
    let mut bitv = HashBytes::with_capacity((hash_size * hash_size * 2) as usize);

    
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
    ///
    /// The iteration order is unspecified. Implementations should use whatever is optimum.
    fn foreach_pixel<F>(&self, iter_fn: F) where F: FnMut(u32, u32, &[u8]);
}

fn prepare_image<I: HashImage>(img: &I, width: u32, height: u32) -> Vec<u8> {
    img.grayscale().resize(width, height).to_bytes()
}

/// Crop the values off a 1D-packed 2D DCT
fn crop_2d_dct(packed: &[f64], original: (usize, usize), new: (usize, usize)) -> Vec<f64> {
    let (orig_width, orig_height) = original;

    assert_eq!(packed.len(), orig_width * orig_height);

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

    macro_rules! test_hash_equality {
        ($fnname:ident, $size:expr, $type:ident) => {
            #[test]
            fn $fnname() {
                // square, powers of two
                test_hash_equality!(1024, 1024, $size, $type);
                // rectangular, powers of two
                test_hash_equality!(512, 256, $size, $type);
                // odd size, square
                test_hash_equality!(967, 967, $size, $type);
                // odd size, rectangular
                test_hash_equality!(967, 1023, $size, $type);
            }
        };
        ($width:expr, $height:expr, $size:expr, $type:ident) => {{
            let test_img = gen_test_img($width, $height);
            let hash1 = ImageHash::hash(&test_img, $size, HashType::$type);
            let hash2 = ImageHash::hash(&test_img, $size, HashType::$type);
            assert_eq!(hash1, hash2);
        }};
    }

    macro_rules! test_hash_type {
        ($type:ident, $modname:ident) => {
            mod $modname {
                use {HashType, ImageHash};
                use super::*;

                test_hash_equality!(hash_eq_8, 8, $type);
                test_hash_equality!(hash_eq_16, 16, $type);
                test_hash_equality!(hash_eq_32, 32, $type);
            }
        }
    }

    test_hash_type!(Mean, mean);
    test_hash_type!(Block, blockhash);
    test_hash_type!(Gradient, gradient);
    test_hash_type!(DoubleGradient, dbl_gradient);
    test_hash_type!(DCT, dct);

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

        assert_ne!(dct1, dct2);
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
