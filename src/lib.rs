extern crate image;
extern crate serialize;

use self::dct::{dct_2d, crop_dct};

use self::image::{
    GenericImage, DynamicImage, 
    ImageBuf, Luma, Pixel, FilterType, Nearest, Rgba
};

use self::image::imageops::{grayscale, resize};

use self::serialize::base64::{ToBase64, STANDARD};

use std::collections::Bitv;

const FILTER_TYPE: FilterType = Nearest;

mod dct;

#[deriving(PartialEq, Eq, Hash, Show, Clone)]
pub struct ImageHash {
    size: u32,
    bitv: Bitv,
}

impl ImageHash {

    pub fn dist(&self, other: &ImageHash) -> uint {
        assert!(self.bitv.len() == other.bitv.len(), 
                "ImageHashes must be the same length for proper comparison!");

        self.bitv.iter().zip(other.bitv.iter())
            .filter(|&(left, right)| left != right).count()
    }

    pub fn dist_ratio(&self, other: &ImageHash) -> f32 {
        self.dist(other) as f32 / self.size as f32
    }    

    
    fn fast_hash<Img: GenericImage<Rgba<u8>>>(img: &Img, hash_size: u32) -> Bitv {
        let temp = square_resize_and_gray(img, hash_size);

        let hash_values: Vec<u8> = temp.pixels().map(|(_, _, x)| x.channel())
            .collect();

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
        let hash_values: Vec<f64> = temp.pixels()
            .map(|(_, _, x)| x.channel() as f64).collect();

        let dct = dct_2d(hash_values.as_slice(),
            large_size as uint, large_size as uint);

        let original = (large_size as uint, large_size as uint);
        let new = (hash_size as uint, hash_size as uint);

        let cropped_dct = crop_dct(dct, original, new);

        let mean = cropped_dct.iter().fold(0f64, |b, &a| a + b) 
            / (hash_size * hash_size) as f64;

        cropped_dct.into_iter().map(|x| x >= mean).collect()
    }    

    pub fn hash<Img: GenericImage<Rgba<u8>>>(img: &Img, hash_size: u32, fast: bool) -> ImageHash {
        let hash = if fast { 
            ImageHash::fast_hash(img, hash_size)   
        } else { 
            ImageHash::dct_hash(img, hash_size)             
        };

        assert!((hash_size * hash_size) as uint == hash.len());

        ImageHash {
            size: hash_size * hash_size,
            bitv: hash,
        }
    }

    pub fn to_base64(&self) -> String {
        let self_bytes = self.bitv.to_bytes();

        self_bytes.as_slice().to_base64(STANDARD)
    }
}

fn square_resize_and_gray<Img: GenericImage<Rgba<u8>>>(img: &Img, size: u32) -> ImageBuf<Luma<u8>> {
        let small = resize(img, size, size, FILTER_TYPE);
        grayscale(&small)
}

mod test {
    extern crate test;

    use image::{Rgba, Pixel, ImageBuf};

    use self::test::Bencher;
      
    use super::ImageHash;

    use std::rand::{Rng, random};

    
    fn rand_pixel() -> Rgba<u8> {  
        let (a, b, c, d) = random();
        Pixel::from_channels(a, b, c, d)
    }

    fn gen_test_img(width: u32, height: u32) -> ImageBuf<Rgba<u8>> {
        let mut buf: ImageBuf<Rgba<u8>> = ImageBuf::new(width, height);
        
        for px in buf.pixelbuf_mut().iter_mut() {
            *px = rand_pixel();    
        }

        buf
    }

    #[test]
    fn hash_equality() {
        let test_img = gen_test_img(1024, 1024);
        let hash1 = ImageHash::hash(&test_img, 32, false);
        let hash2 = ImageHash::hash(&test_img, 32, false);

        assert_eq!(hash1, hash2);            
    }

    #[bench]
    fn bench_hash(b: &mut Bencher) {
        let test_img = gen_test_img(1024, 1024);
        
        b.iter(|| ImageHash::hash(&test_img, 32, false));    
    }


    
}
