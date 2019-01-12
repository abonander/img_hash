extern crate img_hash;
extern crate rayon;

use img_hash::demo::*;

const HASH_WIDTH: u32 = 8;
const HASH_HEIGHT: u32 = 8;

// we perform the DCT on an enlarged image
const DCT_WIDTH: u32 = HASH_WIDTH * 2;
const DCT_HEIGHT: u32 = HASH_HEIGHT * 2;

macro_rules! handle(
    ($try:expr) => {
        if let Err(e) = $try {
            println!("{}", e);
        }
    }
);

fn main() -> Result<(), String> {
    let ref ctxt = DemoCtxt::init("demo_gradient", "HashAlg::Gradient")?;

    println!("generating DCT-mean hash demo, this will take some time");

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
            let resize_anim = ctxt.animate_resize(grayscale, DCT_WIDTH, DCT_HEIGHT, 20, 25);

            s.spawn(move |s| {
                println!("saving resize animation");
                handle!(ctxt.save_gif("resize", resize_anim));
            });
        });

        s.spawn(move |s| {
            println!("generating DCT processing animation");
            let (dct_anim, dct) = animate_dct(ctxt, grayscale);

            s.spawn(move |s| {
                println!("saving DCT processing animation");
                handle!(ctxt.save_gif("dct", dct_anim));
            })
        })

    });

    Ok(())
}

/// A simple animation showing the DCT values sliding out of the original input
fn animate_dct(ctxt: &DemoCtxt, grayscale: &RgbaImage) -> (Vec<Frame>, RgbaImage) {
    let dct_ctxt = DctCtxt::new(DCT_WIDTH, DCT_HEIGHT);

    // the final resized image
    let resized_small = imageops::resize(grayscale, DCT_WIDTH, DCT_HEIGHT, Lanczos3);

    let input_len = resized_small.len() * 2;
    let mut vals_with_scratch = Vec::with_capacity(input_len);

    // put the image values in [..width * height] and provide scratch space
    vals_with_scratch.extend(resized_small.pixels().map(|px| px.to_luma()[0] as f32));
    // TODO: compare with `.set_len()`
    vals_with_scratch.resize(input_len, 0.);

    let dct_vals = dct_ctxt.dct_2d(vals_with_scratch);

    let mut dct_pxs = Vec::with_capacity(dct_vals.len() * 4);
    for val in dct_vals {
        dct_pxs.extend_from_slice(luma_rgba(val as u8).channels());
    }
    let dct_img = ImageBuffer::<Rgba<u8>, _>::from_vec(DCT_WIDTH, DCT_HEIGHT, dct_pxs).unwrap();

    let half_width = ctxt.width / 2;
    let gif_height = half_width;

    // 10% padding around the resized image
    let resize_width = fmul(half_width, 0.9);
    let resize_height = fmul(gif_height, 0.9);

    // center the input image in the left half of the gif
    let input_x = (half_width - resize_width) / 2;
    let input_y = (gif_height - resize_height) / 2;

    // center the output in the right half
    let output_x = input_x + half_width;
    let output_y = input_y;

    // as with `demo_gradient`, using Nearest gives sharp individual pixels
    let resized_input = imageops::resize(&resized_small, resize_width, resize_height, Nearest);
    let resized_output = imageops::resize(&dct_img, resize_width, resize_height, Nearest);

    let mut background = ImageBuffer::from_pixel(ctxt.width, gif_height, WHITE_A);
    imageops::overlay(&mut background, &resized_input, input_x, input_y);

    // first frame, just input, hold for 5 seconds
    let first_frame = Frame::from_parts(background.clone(), 0, 0, 500.into());

    let mut frames = Some(first_frame).into_iter().chain(
        lerp_iter(input_x, output_x, 5000, 24).map(|(x, frame_delay)| {
            let mut frame = background.clone();
            let mut output = resized_output.clone();

            if x < input_x + resize_width {
                // the part of the output image that overlaps the input image is inverted
                // and alpha set to one-half
                for x in 0 .. (resize_width - (x - input_x)) {
                    for y in 0 .. resize_height {
                        let mut px = output.get_pixel_mut(x, y);
                        px.invert();
                        px[3] = 127;
                    }
                }
            }

            imageops::overlay(&mut frame, &output, x, output_y);
            Frame::from_parts(frame, 0, 0, frame_delay.into())
        })
    ).collect::<Vec<_>>();

    imageops::overlay(&mut background, &resized_output, output_x, output_y);

    frames.push(Frame::from_parts(background, 0, 0, 5000.into()));

    (frames, dct_img)
}
