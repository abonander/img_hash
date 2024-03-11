//! Hash an image and print the Base64 value

use std::env;

use image_hasher::HasherConfig;

fn main() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 2);

    let image = image::open(&args[1]).map_err(|e| format!("failed to open {}: {}", &args[1], e))?;

    let hash = HasherConfig::new()
        .hash_size(8, 8)
        .to_hasher()
        .hash_image(&image);

    #[allow(clippy::format_collect)]
    let hash_str = hash
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    println!("{}: {}", &args[1], hash_str);

    Ok(())
}
