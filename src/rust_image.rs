// Copyright (c) 2015-2017 The `img_hash` Crate Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use image::{
    imageops,
    DynamicImage,
    FilterType,
    GrayImage,
    GrayAlphaImage,
    RgbImage,
    RgbaImage,
    GenericImageView,
    Pixel
};

use super::HashImage;

const FILTER_TYPE: FilterType = FilterType::Nearest;

macro_rules! hash_img_impl {
    ($ty:ident ($lumaty:ty)) => (
        impl HashImage for $ty {
            type Grayscale = $lumaty;

            fn dimensions(&self) -> (u32, u32) {
                self.dimensions()
            }

            fn resize(&self, width: u32, height: u32) -> Self {
                imageops::resize(self, width, height, FILTER_TYPE)
            }

            fn grayscale(&self) -> $lumaty {
                imageops::grayscale(self)
            }

            fn to_bytes(self) -> Vec<u8> {
                self.into_raw()
            }

            fn channel_count() -> u8 {
                <<Self as GenericImageView>::Pixel as Pixel>::CHANNEL_COUNT
            }

            fn foreach_pixel<F>(&self, mut iter_fn: F) where F: FnMut(u32, u32, &[u8]) {
                for (x, y, px) in self.enumerate_pixels() {
                    iter_fn(x, y, px.channels());
                }
            }
        }
    );
    ($($ty:ident ($lumaty:ty)),+) => ( $(hash_img_impl! { $ty($lumaty) })+ );
}

hash_img_impl! { 
    GrayImage(GrayImage), GrayAlphaImage(GrayImage), 
    RgbImage(GrayImage), RgbaImage(GrayImage) 
}

impl HashImage for DynamicImage {
    type Grayscale = GrayImage;

    fn dimensions(&self) -> (u32, u32) {
        <Self as GenericImageView>::dimensions(self) 
    }

    fn resize(&self, width: u32, height: u32) -> Self {
        self.resize(width, height, FILTER_TYPE)
    }

    fn grayscale(&self) -> GrayImage {
        imageops::grayscale(self)
    }

    fn to_bytes(self) -> Vec<u8> {
        self.raw_pixels()
    }

    fn channel_count() -> u8 {
        <<Self as GenericImageView>::Pixel as Pixel>::CHANNEL_COUNT
    }

    fn foreach_pixel<F>(&self, mut iter_fn: F) where F: FnMut(u32, u32, &[u8]) {
        for (x, y, px) in self.pixels() {
            iter_fn(x, y, px.channels());
        }
    }
}
