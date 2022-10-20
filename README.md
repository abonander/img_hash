# Image Hasher

A library for getting perceptual hash values of images.

Thanks to Dr. Neal Krawetz for the outlines of the Mean (aHash), Gradient (dHash), and DCT (pHash) perceptual hash
algorithms:  
http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html (Accessed August 2014)

Also provides an implementation of [the Blockhash.io algorithm](http://blockhash.io).

This crate can operate directly on buffers from the [PistonDevelopers/image][1] crate.

[1]: https://github.com/PistonDevelopers/image

This is fork of [img_hash](https://github.com/abonander/img_hash) library, but with updated dependencies without any
license changes.

Usage
=====
[Documentation](https://docs.rs/img_hash)

Add `image_hasher` to your `Cargo.toml`:

```
image_hasher = "1.1.0"
```

Example program:

```rust
 use imgage_hasher::{HasherConfig, HashAlg};

 fn main() {
     let image1 = image::open("image1.png").unwrap();
     let image2 = image::open("image2.png").unwrap();
     
     let hasher = HasherConfig::new().to_hasher();

     let hash1 = hasher.hash_image(&image1);
     let hash2 = hasher.hash_image(&image2);
     
     println!("Image1 hash: {}", hash1.to_base64());
     println!("Image2 hash: {}", hash2.to_base64());
     
     println!("Hamming Distance: {}", hash1.dist(&hash2));
 }
```

Benchmarking
============

In order to build and test on Rust stable, the benchmarks have to be placed behind a feature gate. If you have Rust
nightly installed and want to run benchmarks, use the following command:

```
cargo +nightly bench
```

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
