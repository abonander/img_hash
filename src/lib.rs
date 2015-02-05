#![feature(collections, core, hash)]
// Silence feature warnings for test module.
#![cfg_attr(test, feature(rand, test))]

extern crate image;
extern crate "rustc-serialize" as serialize;

use self::dct::{dct_2d, crop_dct};

use image::{
    imageops,
    DynamicImage,
    FilterType,
    GrayImage,
    GrayAlphaImage,
    Pixel,
    RgbImage,
    RgbaImage,
};

use serialize::base64::{ToBase64, STANDARD, FromBase64, FromBase64Error};

use std::collections::Bitv;

const FILTER_TYPE: FilterType = FilterType::Nearest;

mod dct;

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash {
    bitv: Bitv,
}

impl ImageHash {

    /// Calculate the Hamming distance between this and `other`.
    /// Equivalent to counting the 1-bits of the XOR of the two `Bitv`.
    /// 
    /// Essential to determining the perceived difference between `self` and `other`.
    pub fn dist(&self, other: &ImageHash) -> usize {
        assert!(self.bitv.len() == other.bitv.len(), 
                "ImageHashes must be the same length for proper comparison!");

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
    pub fn size(&self) -> u32 { self.bitv.len() as u32}

    /// Create a hash of `img` with a length of `hash_size * hash_size`.
    ///
    /// If `fast == true`, use a simple average of the pixels (faster, but less accurate). 
    /// Else, perform a Discrete Cosine Transform & Mean hash (slower, but more accurate).
    ///
    /// Setting `fast = true` produces hashes that are really only useful for
    /// comparing *equality* and very *closely* similar images with minor edits or corrections. 
    /// Strong color/gamma correction will throw off the hash. 
    ///
    /// In practice, on a fast computer, using `fast = false` won't drastically increase the hash
    /// time for a single image. In a program that processes many images at once, the bottleneck
    /// will likely be in loading and decoding the images, and not in the hash function.
    pub fn hash<I: HashImage>(img: &I, hash_size: u32, fast: bool) -> ImageHash {
        let hash = if fast {
            fast_hash(img, hash_size)
        } else {
            dct_hash(img, hash_size)
        };

        assert!((hash_size * hash_size) as usize == hash.len());

        ImageHash {
            bitv: hash,
        }
    }

    /// Create an `ImageHash` instance from the given Base64-encoded string.
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash, FromBase64Error>{
        let data = try!(encoded_hash.from_base64());

        Ok(ImageHash{
            bitv: Bitv::from_bytes(&*data)
        })
    }

    /// Get a Base64 string representing the bits of this hash.
    ///
    /// Mostly for printing convenience.
    pub fn to_base64(&self) -> String {
        self.bitv.to_bytes().to_base64(STANDARD)
    }
}


fn fast_hash<I: HashImage>(img: &I, hash_size: u32) -> Bitv {
    let hash_values = img.gray_resize_square(hash_size).into_raw();

    let hash_sq = hash_size * hash_size;

    let mean = hash_values.iter().fold(0u32, |b, &a| a as u32 + b) / hash_sq;

    hash_values.into_iter().map(|x| x as u32 >= mean).collect()
}

fn dct_hash<I: HashImage>(img: &I, hash_size: u32) -> Bitv {
    let large_size = hash_size * 4;

    // We take a bigger resize than fast_hash, 
    // then we only take the lowest corner of the DCT
    let hash_values: Vec<_> = img.gray_resize_square(large_size)
        .into_raw().into_iter().map(|val| val as f64).collect();

    let dct = dct_2d(hash_values.as_slice(),
        large_size as usize, large_size as usize);

    let original = (large_size as usize, large_size as usize);
    let new = (hash_size as usize, hash_size as usize);

    let cropped_dct = crop_dct(dct, original, new);

    let mean = cropped_dct.iter().fold(0f64, |b, &a| a + b) 
        / (hash_size * hash_size) as f64;

    cropped_dct.into_iter().map(|x| x >= mean).collect()
}

/// A trait for describing an image that can be successfully hashed.
pub trait HashImage {
    /// Apply a grayscale filter and drop the alpha channel (if present),
    /// then resize the image to `size` width by `size` height (making it square).
    ///
    /// Returns a copy, leaving `self` unmodified.
    fn gray_resize_square(&self, size: u32) -> GrayImage;    
}

macro_rules! hash_img_impl {
    ($ty:ty) => (
        impl HashImage for $ty {
            fn gray_resize_square(&self, size: u32) -> GrayImage {
                let ref gray = imageops::grayscale(self);
                imageops::resize(gray, size, size, FILTER_TYPE)
            }
        }
    );
    ($($ty:ty),+) => ( $(hash_img_impl! { $ty })+ );
}

hash_img_impl! { GrayImage, GrayAlphaImage, RgbImage, RgbaImage, DynamicImage }

#[cfg(test)]
mod test {
    extern crate test;

    use image::{Rgba, ImageBuffer};

    use self::test::Bencher;
      
    use super::ImageHash;

    use std::rand::{weak_rng, Rng};
    
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
        let hash1 = ImageHash::hash(&test_img, 32, false);
        let hash2 = ImageHash::hash(&test_img, 32, false);

        assert_eq!(hash1, hash2);
    }


    #[test]
    fn size() {
        let test_img = gen_test_img(1024, 1024);
        let hash = ImageHash::hash(&test_img, 32, false);
        assert_eq!(32*32, hash.size());
    }

    #[test]
    fn base64_encoding_decoding() {
        let test_img = gen_test_img(1024, 1024);
        let hash1 = ImageHash::hash(&test_img, 32, false);

        let base64_string = hash1.to_base64();
        let decoded_result = ImageHash::from_base64(&*base64_string);

        assert!(decoded_result.is_ok());

        assert_eq!(decoded_result.unwrap(), hash1);
    }

    #[bench]
    fn bench_hash(b: &mut Bencher) {
        let test_img = gen_test_img(1024, 1024);
        
        b.iter(|| ImageHash::hash(&test_img, 32, false));    
    }
}
