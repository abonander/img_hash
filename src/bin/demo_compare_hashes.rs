extern crate img_hash;
extern crate rayon;

use img_hash::demo::*;
use img_hash::{HashConfig, ImageHash};

fn hash_to_string(hash: &ImageHash) -> String {
    hash.bytes.iter().map(|b| format!("{:08b}", b)).collect()
}

fn main() -> Result<(), String> {
    let hash_width = 8;
    let hash_height = 8;
    let purpose = "Demos comparing hashes generated for FILE 1 and FILE 2";
    let ctxt = DemoCtxt::init("demo_compare_hashes", purpose, 2);
    let gif_height = ctxt.width / 2;

    let hasher = HashConfig::new().hash_size(hash_width, hash_height).to_hasher();

    let img_1 = &ctxt.images[0];
    let img_2 = &ctxt.images[1];

    let hash1 = hasher.hash_image(&img_1);
    let hash2 = hasher.hash_image(&img_2);

    let thumb_width = fmul(ctxt.width, 0.01);
    let thumb_height = thumb_width;

    let (thumb_1_x, thumb_1_y) = (fmul(ctxt.width, 0.1), fmul(gif_height, 0.1));
    // align the second image below the first
    let (thumb_2_x, thumb_2_y) = (thumb_1_x, thumb_1_y + thumb_height * 2);

    let (img_1_width, img_1_height) = dimen_fill_area(img_1.dimensions(), (ctxt.width, gif_height));
    let img_1_large = imageops::resize(&img_1, (img_1_width, img_1_height), Lanczos3);

    let (img_1_start_x, img_1_start_y) = (
        (ctxt.height - img_1_width) / 2,
        (gif_height - img_1_height) / 2
    );

    let lerp_1_start = [img_1_start_x, img_1_start_y, img_1_width, img_1_height];
    let lerp_1_end = [thumb_1_x, thumb_1_y, thumb_width, thumb_height];

    println!("generating first resize animation");
    let resize_anim_1: Vec<_> = lerp_iter(lerp_1_start, lerp_1_end, 1000, 15).map(
        |([x, y, width, height], frame_delay)| {
            let mut frame = rgba_fill_white(ctxt.width, gif_height);
            let resized = imageops::resize(&img_1_large, width, height, Lanczos3);
            imageops::overlay(&mut frame, &resized, x, y);
            Frame::from_parts(frame, 0, 0, frame_delay.into())
        }
    ).collect();

    let mut hash_1_str = hash_to_string(&hash_1);
    hash_1_str.insert(0, ": ");

    let hash_1_layout = ctxt.font.layout(&hash_1_str, )

    println!("generating hash animation");

}
