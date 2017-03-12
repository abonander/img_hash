img_hash [![Build Status](https://travis-ci.org/abonander/img_hash.svg?branch=master)](https://travis-ci.org/abonander/img_hash) [![Crates.io shield](https://img.shields.io/crates/v/img_hash.svg)](https://crates.io/crates/img_hash)
========

##### Now builds on stable Rust! (But needs nightly to bench.)

A library for getting perceptual hash values of images.

Thanks to Dr. Neal Krawetz for the outlines of the Mean (aHash), Gradient (dHash), and DCT (pHash) perceptual hash algorithms:  
http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html (Accessed August 2014)

With the `rust-image` feature, this crate can operate directly on buffers from the [PistonDevelopers/image][1] crate.

[1]: https://github.com/PistonDevelopers/image 

Usage
=====
[Documentation](https://docs.rs/img_hash)


Add `img_hash` to your `Cargo.toml`:

    [dependencies.img_hash]
    version = "1.1"
    # For interop with `image`:
    features = ["rust-image"]
    
Example program:

```rust
extern crate image;
extern crate img_hash;

use std::path::Path;
use img_hash::{ImageHash, HashType};

fn main() {
    let image1 = image::open(&Path::new("image1.png")).unwrap();
    let image2 = image::open(&Path::new("image2.png")).unwrap();
    
    // These two lines produce hashes with 64 bits (8 ** 2),
    // using the Gradient hash, a good middle ground between 
    // the performance of Mean and the accuracy of DCT.
    let hash1 = ImageHash::hash(&image1, 8, HashType::Gradient);
    let hash2 = ImageHash::hash(&image2, 8, HashType::Gradient);
    
    println!("Image1 hash: {}", hash1.to_base64());
    println!("Image2 hash: {}", hash2.to_base64());
    
    println!("% Difference: {}", hash1.dist_ratio(&hash2));
}
```
   
Benchmarking
============

In order to build and test on Rust stable, the benchmarks have to be placed behind a feature gate. If you have Rust nightly installed and want to run benchmarks, use the following command:

```
cargo bench --features bench
```
