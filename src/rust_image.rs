use image::{
    imageops,
    DynamicImage,
    FilterType,
    GrayImage,
    GrayAlphaImage,
    RgbImage,
    RgbaImage,
};

use super::{HashImage, LumaBytes};

const FILTER_TYPE: FilterType = FilterType::Nearest;

macro_rules! hash_img_impl {
    ($ty:ty) => (
        impl HashImage for $ty {
            fn to_hashable(&self, width: u32, height: u32) -> LumaBytes {
                let ref gray = imageops::grayscale(self);
                let resized = imageops::resize(gray, width, height, FILTER_TYPE);
                resized.into_raw()
            }
        }
    );
    ($($ty:ty),+) => ( $(hash_img_impl! { $ty })+ );
}

hash_img_impl! { GrayImage, GrayAlphaImage, RgbImage, RgbaImage, DynamicImage }
