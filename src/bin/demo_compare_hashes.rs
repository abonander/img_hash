extern crate img_hash;
extern crate rayon;

use img_hash::demo::*;
use img_hash::{HasherConfig, ImageHash};

use std::iter;

fn hash_to_string(hash: &ImageHash) -> String {
    hash.as_bytes().iter().map(|b| format!("{:08b}", b)).collect()
}

fn main() -> Result<(), String> {
    let hash_width = 4;
    let hash_height = 4;
    let purpose = "Demos comparing hashes generated for FILE 1 and FILE 2";
    let ctxt = DemoCtxt::init("demo_compare_hashes", purpose, 2)?;
    let gif_height = ctxt.width / 2;

    let hasher = HasherConfig::new().hash_size(hash_width, hash_height).to_hasher();

    let img_1 = &ctxt.images[0];
    let img_2 = &ctxt.images[1];

    let hash_1 = hasher.hash_image(img_1);
    let hash_2 = hasher.hash_image(img_2);

    let thumb_width = fmul(ctxt.width, 0.1);
    let thumb_height = thumb_width;

    let (thumb_1_x, thumb_1_y) = (fmul(ctxt.width, 0.1), fmul(gif_height, 0.1));
    // align the second image below the first
    let (thumb_2_x, thumb_2_y) = (thumb_1_x, thumb_1_y + fmul(thumb_height, 1.5));

    let (img_1_width, img_1_height) = dimen_fill_area(img_1.dimensions(), (ctxt.width, gif_height));
    let img_1_large = imageops::resize(img_1, img_1_width, img_1_height, Lanczos3);

    let (img_1_start_x, img_1_start_y) = (
        (ctxt.width - img_1_width) / 2,
        (gif_height - img_1_height) / 2
    );

    let mut resize_1_start = rgba_fill_white(ctxt.width, gif_height);
    imageops::overlay(&mut resize_1_start, &img_1_large, img_1_start_x, img_1_start_y);
    
    let resize_1_start = Frame::from_parts(resize_1_start, 0, 0, 500.into());

    let lerp_1_start = [img_1_start_x, img_1_start_y, img_1_width, img_1_height];
    let lerp_1_end = [thumb_1_x, thumb_1_y, thumb_width, thumb_height];

    println!("generating first resize animation");
    let resize_anim_1: Vec<_> = iter::once(resize_1_start).chain(
        lerp_iter(lerp_1_start, lerp_1_end, 1000, 15).map(
            |([x, y, width, height], frame_delay)| {
                let mut frame = rgba_fill_white(ctxt.width, gif_height);
                let resized = imageops::resize(&img_1_large, width, height, Lanczos3);
                imageops::overlay(&mut frame, &resized, x, y);
                Frame::from_parts(frame, 0, 0, frame_delay.into())
            }
        )
    ).collect();

    let mut hash_1_str = hash_to_string(&hash_1);
    hash_1_str.insert_str(0, ":");

    let (hash_1_x, hash_1_y) = (thumb_1_x + fmul(thumb_width, 1.05), thumb_1_y);
    let hash_text_size = Scale::uniform(thumb_width as f32 / 1.5);

    let hash_1_layout = ctxt.font.layout(&hash_1_str, hash_text_size,
                                         point_f32(hash_1_x, hash_1_y));

    let (hash_1_width, hash_1_height) = size_of_text(&hash_1_layout);
    let hash_1_y = thumb_1_y + (thumb_width - hash_1_height) / 2;

    println!("generating first hash animation");
    let hash_1_anim: Vec<_> = lerp_iter(hash_1_x, hash_1_x + hash_1_width, 500, 25).map(
        |(alpha_x, frame_delay)| {
            let mut frame = resize_anim_1.last().unwrap().buffer().clone();

            for g in hash_1_layout.clone() {
                let Point { x, .. } = g.position();
                let g = g.into_unpositioned().positioned(Point { x, y: hash_1_y as f32 });

                draw_glyph_sampled(&mut frame, &g, |x, _|
                    if x > alpha_x { WHITE_A } else { BLACK_A }
                )
            }

            Frame::from_parts(frame, 0, 0, frame_delay.into())
        }
    ).collect();

    let (img_2_width, img_2_height) = dimen_fill_area(img_2.dimensions(), (ctxt.width, gif_height));
    let img_2_large = imageops::resize(img_2, img_2_width, img_2_height, Lanczos3);

    let (img_2_start_x, img_2_start_y) = (
        (ctxt.width - img_2_width) / 2,
        (gif_height - img_2_height) / 2
    );

    let mut resize_2_start = hash_1_anim[0].buffer().clone();
    imageops::overlay(&mut resize_2_start, &img_2_large, img_2_start_x, img_2_start_y);

    let resize_2_start = Frame::from_parts(resize_2_start, 0, 0, 250.into());

    let lerp_2_start = [img_2_start_x, img_2_start_y, img_2_width, img_2_height];
    let lerp_2_end = [thumb_2_x, thumb_2_y, thumb_width, thumb_height];
    
    println!("generating second resize animation");
    let resize_anim_2: Vec<_> = iter::once(resize_2_start).chain(
            lerp_iter(lerp_2_start, lerp_2_end, 1000, 15).map(
            |([x, y, width, height], frame_delay)| {
                let mut frame = hash_1_anim[0].buffer().clone();
                let resized = imageops::resize(&img_2_large, width, height, Lanczos3);
                imageops::overlay(&mut frame, &resized, x, y);
                Frame::from_parts(frame, 0, 0, frame_delay.into())
            }
        )
    ).collect();

    let mut hash_2_str = hash_to_string(&hash_2);
    hash_2_str.insert_str(0, ":");

    let (hash_2_x, hash_2_y) = (hash_1_x, thumb_2_y);

    let hash_2_layout = ctxt.font.layout(&hash_2_str, hash_text_size,
                                         point_f32(hash_2_x, hash_2_y));

    let (hash_2_width, hash_2_height) = size_of_text(&hash_2_layout);
    let hash_2_y = thumb_2_y + (thumb_width - hash_2_height) / 2;

    println!("generating second hash animation");
    let hash_2_anim: Vec<_> = lerp_iter(hash_2_x, hash_2_x + hash_2_width, 250, 25).map(
        |(alpha_x, frame_delay)| {
            let mut frame = resize_anim_2.last().unwrap().buffer().clone();

            for g in hash_2_layout.clone() {
                let Point { x, .. } = g.position();
                let g = g.into_unpositioned().positioned(Point { x, y: hash_2_y as f32 });

                draw_glyph_sampled(&mut frame, &g, |x, _|
                    if x > alpha_x { WHITE_A } else { BLACK_A }
                )
            }

            Frame::from_parts(frame, 0, 0, frame_delay.into())
        }
    ).collect();

    println!("generating comparison animation");
    let char_bb = hash_1_layout.clone().last().and_then(|g| g.pixel_bounding_box())
        .expect("hash should have a character bounding box");

    let Vector { x: char_width, y: char_height } = (char_bb.max - char_bb.min);
    let (char_width, char_height) = (char_width as u32, char_height as u32);

    let outline = Outline::new(char_width as u32, thumb_2_y - thumb_1_y + char_height,
                               fmul(char_width as u32, 0.1));

    let compare_frame_delay = 60;
    let compare_text_y = fmul(thumb_2_y, 1.5);
    let mut count = 0;
    let mut compare_anim: Vec<_> = hash_1_layout.skip(1).enumerate().map(|(i, g)| {
        let mut frame = hash_2_anim[0].buffer().clone();

        let Point { x, .. } = g.position();

        let left = hash_1_str.chars().nth(i as usize + 1).unwrap();
        let right = hash_2_str.chars().nth(i as usize + 1).unwrap();

        let bit = left != right;
        if bit {
            count += 1;
        }

        let color = if bit { GREEN } else { BLACK };

        outline.draw(&mut frame, x as u32, hash_1_y, color);

        let text = format!("{}{}{} {}{}", left, if bit { "≠" } else { "=" }, right, count,
                            if bit {"+"} else {""});

        let layout = center_text_in_area(
            ctxt.font.layout(&text, hash_text_size, point_f32(0, compare_text_y)),
            ctxt.width, gif_height - compare_text_y
        );

        for g in layout {
            draw_glyph(&mut frame, &g, &color);
        }

        Frame::from_parts(frame, 0, 0, compare_frame_delay.into())
    }).collect();

    let mut final_frame = hash_2_anim[0].buffer().clone();

    // NB: if you change `hash_width * hash_height` this needs to be updated
    let (desc_text, color) = match count {
        0 => ("STRONG MATCH", GREEN),
        1 ... 4 => ("VERY SIMILAR", GREEN),
        _ => ("DIFFERENT", RED),
    };

    let text = format!("={} {}", count, desc_text);

    let layout = center_text_in_area(
        ctxt.font.layout(&text, hash_text_size, point_f32(0, compare_text_y)),
        ctxt.width, gif_height - compare_text_y
    );

    for g in layout {
        draw_glyph(&mut final_frame, &g, &color);
    }

    compare_anim.push(Frame::from_parts(final_frame, 0, 0, compare_frame_delay.into()));

    println!("combining stages");
    let frames = resize_anim_1.into_iter()
        .chain(hash_1_anim)
        .chain(resize_anim_2)
        .chain(hash_2_anim)
        .chain(compare_anim)
        .collect();

    println!("saving comparison gif");
    ctxt.save_gif("compare_hashes", frames)
}
