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
// Silence feature warnings for test module.
#![cfg_attr(test, feature(test))]

extern crate bit_vec;

#[cfg(any(test, feature = "rust-image"))]
extern crate image;
extern crate rustc_serialize as serialize;

use self::dct::{dct_2d, crop_dct};

use serialize::base64::{ToBase64, STANDARD, FromBase64, FromBase64Error};

use bit_vec::BitVec;

use std::{fmt, hash};

mod dct;

#[cfg(any(test, feature = "rust-image"))]
mod rust_image;

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash {
    /// The bits of the hash
    pub bitv: BitVec,
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

    /// Build an image using the bits of the hash, 
    /// setting pixels to white (`0xff`) for `0` and black (`0x00`) for `1`.
    ///
    /// ## Panics
    /// If `width * height != self.bitv.len()`.
    /// If you want a differently sized image then you should resize it yourself 
    /// using `image::imageops::resize()`.
    pub fn visualize(&self, width: u32, height: u32) -> LumaBytes {
        assert!(
            (width * height) as usize == self.bitv.len(), 
            "`width * height` must equal `self.bitv.len()!`"
        );

        self.bitv.iter()
            .map(|bit| (bit as u8) * 0xff)
            .collect()
    }

    /// Create an `ImageHash` instance from the given Base64-encoded string.
    /// ## Note:
    /// **Not** compatible with Base64-encoded strings created before `HashType` was added.
    ///
    /// Does **not** preserve the internal value of `HashType::UserDCT`.
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash, FromBase64Error>{
        let mut data = try!(encoded_hash.from_base64());
        // The hash type should be the first bit of the hash
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

pub type Width = usize;
pub type Height = usize;

/// A 2-dimensional Discrete Cosine Transform function that receives 
/// and returns 1-dimensional packed data.
///
/// The function will be provided the pre-hash data as a 1D-packed vector, 
/// which should be interpreted as a 2D matrix with width and height 
/// provided by the two `usize` parameters:
///
/// ```notest
/// Pre-hash data:
/// [ 1.0 2.0 3.0 ]
/// [ 4.0 5.0 6.0 ]
/// [ 7.0 8.0 9.0 ]
///
/// Packed: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] (width 3, height 3)
/// ```
///
/// The function should then return a new 1D vector of the DCT values packed in the same manner.
#[derive(Copy)]
pub struct DCT2DFunc(pub fn(&[f64], Width, Height) -> Vec<f64>);

impl DCT2DFunc {
    fn as_ptr(&self) -> *const () {
        self.0 as *const ()
    }

    fn call(&self, data: &[f64], width: Width, height: Height) -> Vec<f64> {
        (self.0)(data, width, height) 
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
    /// Fastest, but inaccurate. Really only useful for finding duplicates.
    Mean,
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
    DCT,
    /// Equivalent to `DCT`, but allows the user to provide their own 2-dimensional DCT function. 
    /// See the `DCT2DFunc` docs for more info.
    ///
    /// Use this variant if you want a specialized or optimized 2D DCT implementation, such as from
    /// [FFTW][1]. (This cannot be the default implementation because of licensing conflicts.)
    ///
    /// [1]: http://www.fftw.org/
    UserDCT(DCT2DFunc),
}

impl HashType {
    fn hash<I: HashImage>(self, img: &I, hash_size: u32) -> BitVec {
        use HashType::*; 

        match self {
            Mean => mean_hash(img, hash_size),
            DCT => dct_hash(img, hash_size, DCT2DFunc(dct_2d)),
            Gradient => gradient_hash(img, hash_size),
            DoubleGradient => double_gradient_hash(img, hash_size),
            UserDCT(dct_2d_func) => dct_hash(img, hash_size, dct_2d_func),
        }
    }

    fn to_byte(self) -> u8 {
        use HashType::*;

        match self {
            Mean => 1,
            DCT => 2,
            Gradient => 3,
            DoubleGradient => 4,
            UserDCT(_) => 5,
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
            _ => panic!("Byte {:?} cannot be coerced to a `HashType`!", byte),
        }
    }
}

fn mean_hash<I: HashImage>(img: &I, hash_size: u32) -> BitVec {
    let hash_values = img.to_hashable(hash_size);

    let mean = hash_values.iter().fold(0u32, |b, &a| a as u32 + b) 
        / hash_values.len() as u32;

    hash_values.into_iter().map(|x| x as u32 >= mean).collect()
}

fn dct_hash<I: HashImage>(img: &I, hash_size: u32, dct_2d_func: DCT2DFunc) -> BitVec {
    let large_size = (hash_size * 4) as usize;

    // We take a bigger resize than fast_hash, 
    // then we only take the lowest corner of the DCT
    let hash_values: Vec<_> = img.to_hashable(large_size as u32)
        .into_iter().map(|val| (val as f64) / 255.0).collect();

    let dct = dct_2d_func.call(&hash_values, large_size, large_size);

    let original = (large_size, large_size);
    let new = (hash_size as usize, hash_size as usize);

    let cropped_dct = crop_dct(dct, original, new);

    let mean = cropped_dct.iter().fold(0f64, |b, &a| a + b) 
        / cropped_dct.len() as f64;

    cropped_dct.into_iter().map(|x| x >= mean).collect()
}

/// The guts of the gradient hash, 
/// separated so we can reuse them for both `Gradient` and `DoubleGradient`.
fn gradient_hash_impl(resized: &[u8], hash_size: u32, bitv: &mut BitVec) {
    for row in resized.chunks(hash_size as usize) {
        for idx in 1 .. row.len() {
            // These two should never be out of bounds, so we can skip bounds checking.
            let this = unsafe { row.get_unchecked(idx) };
            let last = unsafe { row.get_unchecked(idx - 1) };

            bitv.push(last < this);
        }

        // Wrap the last comparison so we get `hash_size` total comparisons
        let last = unsafe { row.get_unchecked(row.len() - 1) };
        let first = unsafe { row.get_unchecked(0) };

        bitv.push(last < first);
    }
}

fn gradient_hash<I: HashImage>(img: &I, hash_size: u32) -> BitVec {
    let resized = img.to_hashable(hash_size);
    let mut bitv = BitVec::with_capacity((hash_size * hash_size) as usize);

    gradient_hash_impl(&resized, hash_size, &mut bitv); 

    bitv
}

fn double_gradient_hash<I: HashImage>(img: &I, hash_size: u32) -> BitVec {
    let mut hash_values = img.to_hashable(hash_size);
    let mut bitv = BitVec::with_capacity((hash_size * hash_size * 2) as usize);

    gradient_hash_impl(&hash_values, hash_size, &mut bitv);

    // Swap rows and columns
    swap_rows_columns_square(&mut hash_values, hash_size);
    gradient_hash_impl(&hash_values, hash_size, &mut bitv);

    bitv
}

// swap rows with columns
fn swap_rows_columns_square(bytes: &mut [u8], side: u32) {
    let side = side as usize;

    assert!(bytes.len() == side * side);

    for x in 0..side {
        for y in 0..side {
            bytes.swap(x + side * y, y + side * x);
        }
    }
}
/// A byte vector representing an 8-bit grayscale image.
pub type LumaBytes = Vec<u8>;

/// A trait for describing an image that can be successfully hashed.
///
/// Implement this for custom image types.
pub trait HashImage {
    /// Apply a grayscale filter and drop the alpha channel (if present),
    /// then resize the image to `size` by `size` (making it square).
    ///
    /// Returns a copy, leaving `self` unmodified.
    fn to_hashable(&self, size: u32) -> LumaBytes;    
}

#[cfg(test)]
mod test {
    extern crate rand;
    extern crate test;

    use image::{Rgba, ImageBuffer};

    use self::rand::{weak_rng, Rng};
    use self::test::Bencher;

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
        fn dummy_dct(_ : &[f64], _: usize, _: usize) -> Vec<f64> {
            unimplemented!();
        }

        let dct1 = DCT2DFunc(dummy_dct);
        let dct2 = DCT2DFunc(dummy_dct);

        assert_eq!(dct1, dct2);
    }

    #[test]
    fn dct_2d_inequality() {
        fn dummy_dct(_ : &[f64], _: usize, _: usize) -> Vec<f64> {
            unimplemented!();
        }

        fn dummy_dct_2(_ : &[f64], _: usize, _: usize) -> Vec<f64> {
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

    fn bench_hash(b: &mut Bencher, hash_type: HashType) {
        let test_img = gen_test_img(512, 512);
        
        b.iter(|| ImageHash::hash(&test_img, 8, hash_type));    
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
    bench_hash! { bench_dct_hash : HashType::DCT }
}
