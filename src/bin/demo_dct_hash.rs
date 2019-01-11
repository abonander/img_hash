extern crate image;
extern crate img_hash;
extern crate rayon;
extern crate rusttype;

extern crate rustdct;
extern crate transpose;

use img_hash::demo::*;
use image::*;
use rusttype::{Scale, Point};

const HASH_WIDTH: u32 = 8;
const HASH_HEIGHT: u32 = 8;

macro_rules! handle(
    ($try:expr) => {
        if let Err(e) = $try {
            println!("{}", e);
        }
    }
);

fn main() -> Result<(), String> {
    let ref ctxt = DemoCtxt::init("demo_gradient", "HashAlg::Gradient")?;

    println!("generating DCT-mean hash demo");

    println!("generating grayscale animation");
    // 4 FPS over 5 seconds
    let grayscale_anim = ctxt.animate_grayscale(&ctxt.image, 20, 25);

    let ref grayscale = grayscale_anim.last().unwrap().buffer().clone();

    rayon::scope(move |s| {
        s.spawn(move |s| {
            println!("saving grayscale animation");
            handle!(ctxt.save_gif("grayscale", grayscale_anim));
        });

        s.spawn(move |s| {
            println!("generating resize animation");
            let resize_anim = ctxt.animate_resize(grayscale, HASH_WIDTH, HASH_HEIGHT, 20, 25);

            s.spawn(move |s| {
                println!("saving resize animation");
                handle!(ctxt.save_gif("resize", resize_anim));
            });
        });

        s.spawn(move |s| {
            println!("generating DCT processing animation");
            let dct_anim = animate_dct(ctxt, grayscale);

            s.spawn(move |s| {
                println!("saving DCT processing animation");
                handle!(ctxt.save_gif("dct", gradient_anim));
            })
        })

    });

    Ok(())
}

/// Multiply a `u32` by an `f32` with a truncated result
fn fmul(x: u32, y: f32) -> u32 {
    (x as f32 * y) as u32
}

fn animate_dct(ctxt: &DemoCtxt, grayscale: &RgbaImage) -> Vec<Frame> {
    // the final resized image
    let resized_small = imageops::resize(grayscale, HASH_WIDTH, HASH_HEIGHT, Lanczos3);

    let half_width = ctxt.width / 2;
    let gif_height = gif_height;
    let mut background = ImageBuffer::from_pixel(ctxt.width, gif_height, WHITE_A);

    let resize_width = fmul(half_width, 0.9);
    let resize_height = fmul(gif_height, 0.9);

    let resized = imageops::resize(&resized_small, resize_width, resize_height, Nearest);

    let overlay_x = (half_width - resize_width) / 2;
    let overlay_y = (gif_height - resize_height) / 2;

    let pixel_width = resize_width / HASH_WIDTH;
    let pixel_height = resize_height / HASH_HEIGHT;

    imageops::overlay(&mut background, &resized, overlay_x, overlay_y);

    let outline_thickness = fmul(pixel_width, 0.1);
    let px_outline = Outline::new(pixel_width, pixel_height, outline_thickness);

    let mut output = ImageBuffer::from_pixel(resize_width, resize_height, WHITE_A);

    x_y_iter(HASH_WIDTH, HASH_HEIGHT).map(|(x, y)| {
        let mut frame = background.clone();
        let output_x = overlay_x + half_width;
        let output_y = overlay_y;

        // TODO: simulate DCT by iterating pixels in row/columns and blending result into output px
    }).collect()
}

fn animate_gradient(ctxt: &DemoCtxt, grayscale: &RgbaImage) -> Vec<Frame> {
    // the final resized image
    let resized_small = imageops::resize(grayscale, HASH_WIDTH, HASH_HEIGHT, Lanczos3);

    let gif_height = ctxt.width / 2;

    let mut background = ImageBuffer::from_pixel(ctxt.width, gif_height, WHITE_A);

    // half the width with 10% padding
    let resize_width = fmul(ctxt.width / 2, 0.9);
    // match the resize aspect ratio
    let resize_height = resize_width * HASH_HEIGHT / HASH_WIDTH;

    // nearest filter will retain the individual pixels
    let resized = imageops::resize(&resized_small, resize_width, resize_height, Nearest);

    // center the resized image in the left half of `background`
    let overlay_x = (ctxt.width / 2 - resize_width) / 2;
    let overlay_y = (gif_height - resize_height) / 2;

    imageops::overlay(&mut background, &resized, overlay_x, overlay_y);

    let pixel_width = resize_width / HASH_WIDTH;
    let pixel_height = resize_height / HASH_HEIGHT;

    // configure an outline with 20% thickness
    let outline = Outline::new(pixel_width * 2, pixel_height, fmul(pixel_width, 0.1));

    // subtract the thickness of the outline from its overall offset
    let outline_x = overlay_x - outline.thickness;
    let outline_y = overlay_y - outline.thickness;

    let mut bitstring = Bitstring::new();

    // we touch HASH_HEIGHT * (HASH_WIDTH - 1) pixels
    x_y_iter(HASH_WIDTH - 1, HASH_HEIGHT).map(|(x, y)| {
            let mut frame = background.clone();

            let left = *resized_small.get_pixel(x, y);
            let right = *resized_small.get_pixel(x + 1, y);

            let left_pixel = ImageBuffer::from_pixel(pixel_width, pixel_height, left);
            let right_pixel = ImageBuffer::from_pixel(pixel_width, pixel_height, right);

            let bit = left.to_luma()[0] > right.to_luma()[0];
            let bit_color = if bit { GREEN } else { RED };

            outline.draw(&mut frame, outline_x + pixel_width * x, outline_y + pixel_height * y,
                         bit_color);

            // position the left pixel in the second third of the image's width
            let left_pixel_x = ctxt.width / 3 * 2;
            let right_pixel_x = left_pixel_x + (pixel_width * 2);
            let pixel_y = gif_height / 4;

            // between the two pixels draw either `<` or `>`
            let comp = if bit { '>' } else { '⩽' };
            bitstring.push_bit(bit);

            let comp_glyph = center_in_area(
                ctxt.font.glyph(comp)
                .scaled(Scale { x: pixel_width as f32, y: pixel_height as f32 })
                .positioned(Point { x: (left_pixel_x + pixel_width) as f32, y: pixel_y as f32 }),
                pixel_width, pixel_height);

            imageops::overlay(&mut frame, &left_pixel, left_pixel_x, pixel_y);
            imageops::overlay(&mut frame, &right_pixel, right_pixel_x, pixel_y);

            draw_glyph(&mut frame, &comp_glyph, &bit_color);

            // place the string just after the horizontal halfway point with some padding
            let string_x = fmul(ctxt.width / 2, 1.1);
            let string_y = gif_height * 3 / 4;
            ctxt.layout_text(bitstring.as_str(), string_x, string_y).enumerate()
                .for_each(|(i, g)| {
                    let color = if i + 1 == bitstring.as_str().len() { bit_color } else { BLACK };
                    draw_glyph(&mut frame, &g, &color)
                });

            // run faster after the first couple rows
            let delay = if y < 2 { 50 } else { 8 };
            Frame::from_parts(frame, 0, 0, delay.into())
        })
        .collect()
}
