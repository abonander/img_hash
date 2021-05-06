//! Hash an image and print the Base64 value

extern crate img_hash;
extern crate image;

use img_hash::HasherConfig;

fn main() -> Result<(), String> {
    let matches = clap::App::new("hash_image")
    .version("3.2.0")
    .arg(
        clap::Arg::with_name("IMAGE")
            .required(true)
            .index(1)
            .help("Image file to calculate hash over.")
    )
    .get_matches();

    let image_file = matches.value_of("IMAGE").unwrap();

    let image = image::open(image_file).map_err(|e| {
        format!("failed to open {}: {}", image_file, e)
    })?;

    let hash = HasherConfig::new().hash_size(8, 8).to_hasher().hash_image(&image);

    let hash_str = hash.as_bytes().iter().map(|b| format!("{:02x}", b)).collect::<String>();

    println!("{}: {}", image_file, hash_str);

    Ok(())
}