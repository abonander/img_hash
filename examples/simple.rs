use clap::Parser;
use visual_hash::HasherConfig;

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
