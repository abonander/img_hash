img_hash [![Build Status](https://travis-ci.org/cybergeek94/img_hash.svg?branch=master)](https://travis-ci.org/cybergeek94/img_hash) [![Crates.io shield](https://img.shields.io/crates/v/img_hash.svg)](https://crates.io/crates/img_hash)
========

##### Now builds on stable Rust! (But needs nightly to test.)

A library for getting perceptual hash values of images.

Thanks to Dr. Neal Krawetz for the outlines of the Mean (aHash), Gradient (dHash), and DCT (pHash) perceptual hash algorithms:  
http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html (Accessed August 2014)

Thanks to Emil Mikulic for the 2D Discrete Cosine Transform implementation in C, ported to Rust in `src/dct.rs`:  
http://unix4lyfe.org/dct/ (Implementation: http://unix4lyfe.org/dct/listing2.c) (Accessed August 2014)

With the `rust-image` feature, this crate can operate directly on buffers from the [PistonDevelopers/image][1] crate.

[1]: https://github.com/PistonDevelopers/image 

Usage
=====
[Documentation on Rust-CI](http://rust-ci.org/cybergeek94/img_hash/doc/img_hash/index.html)


Add `img_hash` to your `Cargo.toml`:

    [dependencies.img_hash]
    git = "https://github.com/cybergeek94/img_hash"
    # For interop with `image`:
    features = ["rust-image"]
    
Example program:

```rust
extern crate image;
extern crate img_hash;

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
    
