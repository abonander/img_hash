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
#![cfg_attr(feature = "nightly", feature(specialization))]

extern crate base64;

pub extern crate image;
pub use image::FilterType;

extern crate num_traits;

extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::borrow::Cow;
use std::ops;

use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Pixel};
use image::imageops;

mod dct;

mod alg;

pub use alg::HashAlg;

use std::marker::PhantomData;

/// Interface for types used for storing hash data.
///
/// This is implemented for `Vec<u8>`, `Box<[u8]>` and arrays that are multiples/combinations of
/// useful x86 bytewise SIMD register widths (64, 128, 256, 512 bits).
///
/// Please feel free to open a pull request [on Github](https://github.com/abonander/img_hash)
/// if you need this implemented for a different array size.
pub trait HashBytes {
    /// Construct this type from an iterator of bytes.
    ///
    /// If this type has a finite capacity (i.e. an array) then it can ignore extra data
    /// (the hash API will not create a hash larger than this type can contain). Unused capacity
    /// **must** be zeroed.
    fn from_iter<I: Iterator<Item = u8>>(iter: I) -> Self where Self: Sized;

    /// Return the maximum capacity of this type, in bits.
    ///
    /// If this type has an arbitrary/theoretically infinite capacity, return `usize::max_value()`.
    fn max_bits() -> usize;

    /// Get the hash bytes as a slice.
    fn as_slice(&self) -> &[u8];
}

impl HashBytes for Box<[u8]> {
    fn from_iter<I: Iterator<Item = u8>>(iter: I) -> Self {
        iter.collect()
    }

    fn max_bits() -> usize {
        usize::max_value()
    }

    fn as_slice(&self) -> &[u8] { self }
}

impl HashBytes for Vec<u8> {
    fn from_iter<I: Iterator<Item=u8>>(iter: I) -> Self {
        iter.collect()
    }

    fn max_bits() -> usize {
        usize::max_value()
    }

    fn as_slice(&self) -> &[u8] { self }
}

macro_rules! hash_bytes_array {
    ($($n:expr),*) => {$(
        impl HashBytes for [u8; $n] {
            fn from_iter<I: Iterator<Item=u8>>(mut iter: I) -> Self {
                // optimizer should eliminate this zeroing
                let mut out = [0; $n];

                for (src, dest) in iter.by_ref().zip(out.as_mut()) {
                    *dest = src;
                }

                out
            }

            fn max_bits() -> usize {
                $n * 8
            }

            fn as_slice(&self) -> &[u8] { self }
        }
    )*}
}

hash_bytes_array!(8, 16, 24, 32, 40, 48, 56, 64);

trait BitSet: HashBytes {
    fn from_bools<I: Iterator<Item = bool>>(iter: I) -> Self where Self: Sized {
        struct BoolsToBytes<I> {
            iter: I,
        }

        impl<I> Iterator for BoolsToBytes<I> where I: Iterator<Item=bool> {
            type Item = u8;

            fn next(&mut self) -> Option<<Self as Iterator>::Item> {
                self.iter.by_ref().take(8).fold(None, |accum, val| {
                    accum.map(|accum| (accum << 1) | (val as u8))
                })
            }
        }

       Self::from_iter(BoolsToBytes { iter })
    }

    fn hamming(&self, other: &Self) -> u32 {
        self.as_slice().iter().zip(other.as_slice()).map(|(l, r)| (l ^ r).count_ones()).sum()
    }
}

impl<T: HashBytes> BitSet for T {}

// TODO: implement `Debug`, needs adaptor for `FilterType`
#[derive(Serialize, Deserialize)]
pub struct HasherConfig<B = Box<[u8]>> {
    width: u32,
    height: u32,
    gauss_sigmas: Option<[f32; 2]>,
    #[serde(with = "SerdeFilterType")]
    resize_filter: FilterType,
    dct: bool,
    hash_alg: HashAlg,
    _bytes_type: PhantomData<B>,
}

impl<B: HashBytes> HasherConfig<B> {
    /// Construct a new hasher config with sane, reasonably fast defaults.
    ///
    /// A default hash container type is provided as a default type parameter which is guaranteed
    /// to fit any hash size.
    pub fn new() -> HasherConfig<B> {
        Self {
            width: 8,
            height: 8,
            gauss_sigmas: None,
            resize_filter: FilterType::Lanczos3,
            dct: false,
            hash_alg: HashAlg::Gradient,
            _bytes_type: PhantomData,
        }

    }

    /// Set a new hash width and height; these can be the same.
    ///
    /// The number of bits in the resulting hash will be `width * height`. If you are using
    /// a fixed-size `HashBytes` type then you must ensure it can hold at least this many bits.
    /// You can check this with [`HashBytes::max_bits()`](HashBytes::max_bits).
    ///
    /// ### Rounding Behavior
    /// Certain hash algorithms need to round this value to function properly:
    ///
    /// * [`DoubleGradient`](HashAlg::DoubleGradient) rounds to the next multiple of 2;
    /// * [`Blockhash`](HashAlg::Blockhash) rounds to the next multiple of 4.
    ///
    /// If the chosen values already satisfy these requirements then nothing is changed.
    ///
    /// ### Recommended Values
    /// The hash granularity increases with `width * height`, although there are diminishing
    /// returns for higher values. Start small. A good starting value to try is `8, 8`.
    ///
    /// When using DCT preprocessing having `width` and `height` be the same value will improve
    /// hashing performance as only one set of coefficients needs to be used.
    pub fn hash_size(self, width: u32, height: u32) -> Self {
        Self { width, height, ..self  }
    }

    /// Set the filter used to resize images during hashing.
    ///
    /// Note when picking a filter that images are almost always reduced in size.
    /// Has no effect with the Blockhash algorithm as it does not resize.
    pub fn resize_filter(self, resize_filter: FilterType) -> Self {
        Self { resize_filter, ..self }
    }

    /// Set the algorithm used to generate hashes.
    ///
    /// Each algorithm has different performance characteristics.
    pub fn hash_alg(self, hash_alg: HashAlg) -> Self {
        Self { hash_alg, ..self }
    }

    /// Enable preprocessing with the Discrete Cosine Transform (DCT).
    ///
    /// Does nothing when used with [the Blockhash.io algorithm](HashAlg::Blockhash)
    /// which does not scale the image.
    /// (RFC: it would be possible to shoehorn a DCT into the Blockhash algorithm but it's
    /// not clear what benefits, if any, that would provide).
    ///
    /// After conversion to grayscale, the image is scaled down to `width * 2 x height * 2`
    /// and then the Discrete Cosine Transform is performed on the luminance values. The DCT
    /// essentially transforms the 2D image from the spatial domain with luminance values
    /// to a 2D frequency domain where the values are amplitudes of cosine waves. The resulting
    /// 2D matrix is then cropped to the low `width * height` corner and the
    /// configured hash algorithm is performed on that.
    ///
    /// In layman's terms, this essentially converts the image into a mathematical representation
    /// of the "broad strokes" of the data, which allows the subsequent hashing step to be more
    /// robust against changes that may otherwise produce different hashes, such as significant
    /// edits to portions of the image.
    ///
    /// However, on most machines this usually adds an additional 50-100% to the average hash time.
    ///
    /// This is a very similar process to JPEG compression, although the implementation is too
    /// different for this to be optimized specifically for JPEG encoded images.
    ///
    /// Further Reading:
    /// * http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
    /// Krawetz describes a "pHash" algorithm which is equivalent to Mean + DCT preprocessing here.
    /// However there is nothing to say that DCT preprocessing cannot compose with other hash
    /// algorithms; Gradient + DCT might well perform better in some aspects.
    /// * https://en.wikipedia.org/wiki/Discrete_cosine_transform
    pub fn preproc_dct(self) -> Self {
        Self { dct: true, ..self }
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with default sigma values.
    ///
    /// Recommended only for use with [the Blockhash.io algorithm](HashAlg::Blockhash)
    /// as it significantly reduces entropy in the scaled down image for other algorithms.
    ///
    /// See [`Self::preproc_diff_gauss_sigmas()](Self::preproc_diff_gauss_sigmas) for more info.
    pub fn preproc_diff_gauss(self) -> Self {
        self.preproc_diff_gauss_sigmas(5.0, 10.0)
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with the given sigma values.
    ///
    /// Recommended only for use with [the Blockhash.io algorithm](HashAlg::Blockhash)
    /// as it significantly reduces entropy in the scaled down image for other algorithms.
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
        Self { gauss_sigmas: Some([sigma_a, sigma_b]), ..self }
    }

    /// Create a [`Hasher`](Hasher) from this config which can be used to hash images.
    ///
    /// If DCT preprocessing was selected, this will precalculate the DCT coefficients for the
    /// chosen hash size.
    ///
    /// ### Panics
    /// If the chosen hash size (`width x height`, rounded for the algorithm if necessary)
    /// is too large for the chosen container type (`B::max_bits()`).
    pub fn into_hasher(self) -> Hasher<B> {
        let Self { hash_alg, width, height, gauss_sigmas, resize_filter, dct, .. } = self;

        let (width, height) = hash_alg.round_hash_size(width, height);

        assert!((width * height) as usize <= B::max_bits(),
                "hash size too large for container: {} x {}", width, height);

        // Blockhash doesn't resize the image so don't waste time calculating coefficients
        let dct_coeffs = if dct && hash_alg != HashAlg::Blockhash {
            // calculate the coefficients based on the resize dimensions
            let (dct_width, dct_height) = hash_alg.resize_dimensions(width, height);
            Some(dct::Coefficients::precompute(dct_width, dct_height))
        } else {
            None
        };

        Hasher {
            ctxt: HashCtxt {
                gauss_sigmas, dct_coeffs, width, height, resize_filter,
            },
            hash_alg,
            bytes_type: PhantomData
        }

    }
}

pub struct Hasher<B = Box<[u8]>> {
    ctxt: HashCtxt,
    hash_alg: HashAlg,
    bytes_type: PhantomData<B>,
}

impl<B> Hasher<B> where B: HashBytes {
    pub fn hash_image<I: Image>(&self, img: &I) -> ImageHash<B> {
        let hash = self.hash_alg.hash_image(&self.ctxt, img);
        ImageHash { hash, __backcompat: () }
    }
}

enum CowImage<'a, I: Image> {
    Borrowed(&'a I),
    Owned(I::Buf),
}

impl<'a, I: Image> CowImage<'a, I> {
    fn to_grayscale(&self) -> Cow<GrayImage> {
        match *self {
            CowImage::Borrowed(ref img) => img.to_grayscale(),
            CowImage::Owned(ref img) => img.to_grayscale(),
        }
    }
}

enum HashVals {
    Floats(Vec<f32>),
    Bytes(Vec<u8>),
}

// TODO: implement `Debug`, needs adaptor for `FilterType`
struct HashCtxt {
    gauss_sigmas: Option<[f32; 2]>,
    dct_coeffs: Option<dct::Coefficients>,
    resize_filter: FilterType,
    width: u32,
    height: u32,
}

impl HashCtxt {
    fn gauss_preproc<'a, I: Image>(&self, image: &'a I) -> CowImage<'a, I> {
        if let Some([sigma_a, sigma_b]) = self.gauss_sigmas {
            let mut blur_a = image.blur(sigma_a);
            let blur_b = image.blur(sigma_b);
            blur_a.diff_inplace(&blur_b);

            CowImage::Owned(blur_a)
        } else {
            CowImage::Borrowed(image)
        }
    }

    fn calc_hash_vals(&self, img: &GrayImage, width: u32, height: u32) -> HashVals {
        if let Some(ref coeffs) = self.dct_coeffs {
            let width = width * dct::SIZE_MULTIPLIER;

            let img = imageops::resize(img, width, height * dct::SIZE_MULTIPLIER,
                                       self.resize_filter);

            let img_vals: Vec<f32> = img.into_vec().into_iter()
                .map(|x| x as f32 / 255 as f32).collect();

            let hash_vals = dct::dct_2d(&img_vals, width as usize, coeffs);
            HashVals::Floats(dct::crop_2d_dct(hash_vals, width as usize))
        } else {
            let img = imageops::resize(img, width, height, self.resize_filter);
            HashVals::Bytes(img.into_vec())
        }
    }
}

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash<B> {
    pub hash: B,
    __backcompat: (),
}

impl<B: HashBytes> ImageHash<B> {
    /// Calculate the Hamming distance between this and `other`.
    ///
    /// Equivalent to counting the 1-bits of the XOR of the two hashes.
    /// 
    /// Essential to determining the perceived difference between `self` and `other`.
    ///
    /// ### Note
    /// This return value is meaningless if these two hashes are from different hash sizes or
    /// algorithms.
    pub fn dist(&self, other: &Self) -> u32 {
        BitSet::hamming(&self.hash, &other.hash)
    }

    /// Create an `ImageHash` instance from the given Base64-encoded string.
    ///
    /// ## Errors:
    /// Returns `DecodeError::InvalidLength` if the decoded bytes can't fit in `B`.
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash<B>, base64::DecodeError>{
        let bytes = base64::decode(encoded_hash)?;

        if bytes.len() * 8 > B::max_bits() {
            return Err(base64::DecodeError::InvalidLength)
        }

        Ok(ImageHash {
            hash: B::from_iter(bytes.into_iter()),
            __backcompat: (),
        })
    }

    /// Get a Base64 string representing the bits of this hash.
    ///
    /// Mostly for printing convenience.
    pub fn to_base64(&self) -> String {
        base64::encode(self.hash.as_slice())
    }
}

/// Shorthand trait bound for APIs in this crate.
///
/// Currently only implemented for the types provided by `image` with 8-bit channels.
pub trait Image: GenericImageView + 'static {
    /// The equivalent `ImageBuffer` type for this container.
    type Buf: Image + DiffImage;

    /// Grayscale the image, reducing to 8 bit depth and dropping the alpha channel.
    fn to_grayscale(&self) -> Cow<GrayImage>;

    /// Blur the image with the given `Gaussian` sigma.
    fn blur(&self, sigma: f32) -> Self::Buf;

    /// Iterate over the image, passing each pixel's coordinates and values in `u8` to the closure.
    ///
    /// The iteration order is unspecified but each pixel **must** be visited exactly _once_.
    ///
    /// If the pixel's channels are wider than 8 bits then the values should be scaled to
    /// `[0, 255]`, not truncated.
    ///
    /// ### Note
    /// If the pixel data length is 2 or 4, the last index is assumed to be the alpha channel.
    /// A pixel data length outside of `[1, 4]` will cause a panic.
    fn foreach_pixel8<F>(&self, foreach: F) where F: FnMut(u32, u32, &[u8]);
}

/// Image types that can be diffed.
pub trait DiffImage {
    /// Subtract the pixel values of `other` from `self` in-place.
    fn diff_inplace(&mut self, other: &Self);
}

impl<P: 'static, C: 'static> Image for ImageBuffer<P, C> where P: Pixel<Subpixel = u8>, C: ops::Deref<Target=[u8]> {
    type Buf = ImageBuffer<P, Vec<u8>>;

    fn to_grayscale(&self) -> Cow<GrayImage> {
        Cow::Owned(imageops::grayscale(self))
    }

    fn blur(&self, sigma: f32) -> Self::Buf { imageops::blur(self, sigma) }

    fn foreach_pixel8<F>(&self, mut foreach: F) where F: FnMut(u32, u32, &[u8]) {
        self.enumerate_pixels().for_each(|(x, y, px)| foreach(x, y, px.channels()))
    }
}

impl<P: 'static> DiffImage for ImageBuffer<P, Vec<u8>> where P: Pixel<Subpixel = u8> {
    fn diff_inplace(&mut self, other: &Self) {
        self.iter_mut().zip(other.iter()).for_each(|(l, r)| *l -= r);
    }
}

impl Image for DynamicImage {
    type Buf = image::RgbaImage;

    fn to_grayscale(&self) -> Cow<GrayImage> {
        self.as_luma8().map_or_else(|| Cow::Owned(self.to_luma()), Cow::Borrowed)
    }

    fn blur(&self, sigma: f32) -> Self::Buf { imageops::blur(self, sigma) }

    fn foreach_pixel8<F>(&self, mut foreach: F) where F: FnMut(u32, u32, &[u8]) {
        self.pixels().for_each(|(x, y, px)| foreach(x, y, px.channels()))
    }
}

#[cfg(feature = "nightly")]
impl Image for GrayImage {
    type Buf = Self;

    fn to_grayscale(&self) -> Cow<GrayImage> {
        Cow::Borrowed(self)
    }

    fn blur(&self, sigma: f32) -> Self::Buf { imageops::blur(self, sigma) }

    fn foreach_pixel8<F>(&self, mut foreach: F) where F: FnMut(u32, u32, &[u8]) {
        self.enumerate_pixels().for_each(|(x, y, px)| foreach(x, y, px.channels()))
    }
}

/// Provide Serde a typedef for `image::FilterType`: https://serde.rs/remote-derive.html
/// This is automatically checked, if Serde complains then double-check with the original definition
#[derive(Serialize, Deserialize)]
#[serde(remote = "FilterType")]
enum SerdeFilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

/*
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
*/
