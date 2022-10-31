ihash
=====

A fork of [img_hash](https://github.com/abonander/img_hash)

A library for getting perceptual hash values of images.

Thanks to Dr. Neal Krawetz for the outlines of the Mean (aHash), Gradient (dHash), and DCT (pHash) perceptual hash algorithms:  
http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html (Accessed August 2014)

Also provides an implementation of [the Blockhash.io algorithm](http://blockhash.io).

This crate can operate directly on buffers from the [PistonDevelopers/image][1] crate.

[1]: https://github.com/PistonDevelopers/image 

Usage
=====
[Documentation](https://docs.rs/ihash)


Add `img_hash` to your `Cargo.toml`:

    [dependencies.img_hash]
    version = "3.0"
    
Example program:

```rust
use clap::Parser;
use img_hash::HasherConfig;

#[derive(Clone, Debug, Parser)]
struct Args {
    left: String,
    right: String,
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> anyhow::Result<()> {
    let image1 = image::open(&args.left)?;
    let image2 = image::open(&args.right)?;

    let hasher = HasherConfig::new().to_hasher();

    let hash1 = hasher.hash_image(&image1);
    let hash2 = hasher.hash_image(&image2);

    println!("Image1 hash: {}", hash1.to_base64());
    println!("Image2 hash: {}", hash2.to_base64());

    println!("Hamming Distance: {}", hash1.dist(&hash2));

    Ok(())
}
```
   
Benchmarking
============

In order to build and test on Rust stable, the benchmarks have to be placed behind a feature gate. If you have Rust nightly installed and want to run benchmarks, use the following command:

```
cargo bench --features bench
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
