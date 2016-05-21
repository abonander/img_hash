use image::{
    imageops,
    DynamicImage,
    FilterType,
    GrayImage,
    GrayAlphaImage,
    RgbImage,
    RgbaImage,
    GenericImage,
    Pixel
};

use super::{HashImage, LumaBytes};

const FILTER_TYPE: FilterType = FilterType::Nearest;

macro_rules! hash_img_impl {
    ($ty:ty ($lumaty:ty)) => (
        impl HashImage for $ty {
            type Grayscale = $lumaty;

            fn dimensions(&self) -> (u32, u32) {
                <Self as GenericImage>::dimensions(self) 
            }

            fn resize(&self, width: u32, height: u32) -> Self {
                imageops::resize(self, width, height, FILTER_TYPE);
            }

            fn grayscale(&self) -> $lumaty {
                imageops::grayscale(self)
            }

            fn to_bytes(self) -> Vec<u8> {
                self.into_raw()
            }

            fn channel_count() -> u8 {
                <<Self as GenericImage>::Pixel as Pixel>::channel_count()
            }

            fn foreach_pixel(&self, mut iter_fn: F) where F: (u32, u32, &[u8]) {
                for (x, y, px) in self.pixels() {
                    iter_fn(x, y, px.channels());
                }
            }
        }
    );
    ($($ty:ty ($lumaty:ty)),+) => ( $(hash_img_impl! { $ty($lumaty) })+ );
}

hash_img_impl! { 
    GrayImage(GrayImage), GrayAlphaImage(GrayImage), 
    RgbImage(GrayImage), RgbaImage(GrayImage), 
    DynamicImage(GrayImage)
}
