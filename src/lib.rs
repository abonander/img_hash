extern crate image;
extern crate serialize;

use self::dct::{dct_2d, crop_dct};

use self::image::{
    GenericImage, ImageBuffer, Luma, Pixel, FilterType, Nearest, Rgba
};

use self::image::imageops::{grayscale, resize};

use self::serialize::base64::{ToBase64, STANDARD, FromBase64, FromBase64Error};

use std::collections::Bitv;

const FILTER_TYPE: FilterType = Nearest;

mod dct;

type LumaBuf = ImageBuffer<Vec<u8>, u8, Luma<u8>>;

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Show, Clone)]
pub struct ImageHash {
    bitv: Bitv,
}

impl ImageHash {

    /// Calculate the Hamming distance between this and `other`.
    /// Equivalent to counting the 1-bits of the XOR of the two `Bitv`.
    /// 
    /// Essential to determining the perceived difference between `self` and `other`.
    pub fn dist(&self, other: &ImageHash) -> uint {
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
    pub fn hash<Img: GenericImage<Rgba<u8>>>(img: &Img, hash_size: u32, fast: bool) -> ImageHash {
        let hash = if fast {
            fast_hash(img, hash_size)
        } else {
            dct_hash(img, hash_size)
        };

        assert!((hash_size * hash_size) as uint == hash.len());

        ImageHash {
            bitv: hash,
        }
    }

    /// Creates a Hash from Base64 string representing the bits of a hash.
    ///
    /// This is the counterpart of to_base64().
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash, FromBase64Error>{
        let data = try!(encoded_hash.from_base64());

        Ok(ImageHash{
            bitv: Bitv::from_bytes(data.as_slice())
        })
    }

    /// Get a Base64 string representing the bits of this hash.
    ///
    /// Mostly for printing convenience.
    pub fn to_base64(&self) -> String {
        let self_bytes = self.bitv.to_bytes();

        self_bytes.as_slice().to_base64(STANDARD)
    }
}

fn square_resize_and_gray<Img: GenericImage<Rgba<u8>>>(img: &Img, size: u32) -> LumaBuf {
        let small = resize(img, size, size, FILTER_TYPE);
        grayscale(&small)
}

fn fast_hash<Img: GenericImage<Rgba<u8>>>(img: &Img, hash_size: u32) -> Bitv {
    let temp = square_resize_and_gray(img, hash_size);

    let hash_values: Vec<u8> = temp.pixels().map(|px| px.channels()[0]).collect();

    let hash_sq = (hash_size * hash_size) as uint;

    let mean = hash_values.iter().fold(0u, |b, &a| a as uint + b) 
        / hash_sq;

    hash_values.into_iter().map(|x| x as uint >= mean).collect()
}

fn dct_hash<Img: GenericImage<Rgba<u8>>>(img: &Img, hash_size: u32) -> Bitv {
    let large_size = hash_size * 4;

    // We take a bigger resize than fast_hash, 
    // then we only take the lowest corner of the DCT
    let temp = square_resize_and_gray(img, large_size);

    // Our hash values are converted to doubles for the DCT
    let hash_values: Vec<f64> = temp.pixels().map(|px| px.channels()[0] as f64).collect();

    let dct = dct_2d(hash_values.as_slice(),
        large_size as uint, large_size as uint);

    let original = (large_size as uint, large_size as uint);
    let new = (hash_size as uint, hash_size as uint);

    let cropped_dct = crop_dct(dct, original, new);

    let mean = cropped_dct.iter().fold(0f64, |b, &a| a + b) 
        / (hash_size * hash_size) as f64;

    cropped_dct.into_iter().map(|x| x >= mean).collect()
}    

#[cfg(test)]
mod test {
    extern crate test;

    use image::{Rgba, ImageBuffer};

    use self::test::Bencher;
      
    use super::ImageHash;

    use std::rand::{weak_rng, Rng};
    
    type RgbaBuf = ImageBuffer<Vec<u8>, u8, Rgba<u8>>;

    fn gen_test_img(width: u32, height: u32) -> RgbaBuf {
        let len = (width * height * 4) as uint;
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
        let decoded_result = ImageHash::from_base64(base64_string.as_slice());

        assert!(decoded_result.is_ok());

        assert_eq!(decoded_result.unwrap(), hash1);
    }

    #[bench]
    fn bench_hash(b: &mut Bencher) {
        let test_img = gen_test_img(1024, 1024);
        
        b.iter(|| ImageHash::hash(&test_img, 32, false));    
    }
}
