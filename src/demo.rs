//! Common utilities for demoing hash types
//!
//! NOTE: not considered part of the crate's stable API
#![allow(missing_docs)]

extern crate interpolation;

use image::{self, *};
use self::interpolation::*;

use std::env;
use std::fs::{self, File};
use std::path::PathBuf;
use std::process;

// should come out to ~26 FPS
const FRAME_DELAY: u16 = 4;

pub struct DemoCtxt {
    pub image: RgbaImage,
    pub output_dir: PathBuf,
    pub width: u32,
}

#[macro_export]
macro_rules! explain {
    ($($arg:tt)*) => { |e| format!("{}: {}", format_args!($($arg)*), e) }
}

impl DemoCtxt {
    pub fn init(name: &str, alg: &str) -> Result<DemoCtxt, String> {
        let args = env::args().collect::<Vec<_>>();

        if args.len() != 4 {
            println!("args: {:?}", args);
            println!("\
                usage: {name} [FILE] [OUTPUT-DIR] [WIDTH]\r\n\
                demos `{alg}` for FILE, exporting gifs of each step to OUTPUT-DIR\r\n\
                each gif will be WIDTH wide; aspect ratio is fixed\r\n\
             ", name = name, alg = alg);
            process::exit(0);
        }

        let file = &args[1];
        let output = &args[2];
        let width = args[3].parse().map_err(explain!("could not parse WIDTH: {}", args[3]))?;

        let image = image::open(file).map_err(explain!("failed to open {}", file))?;

        fs::create_dir_all(output).map_err(explain!("failed to create output dir {}", output))?;

        Ok(Self {
            image: image.to_rgba(),
            output_dir: output.into(),
            width,
        })
    }

    pub fn resize_dimensions(&self, i: &RgbaImage) -> (u32, u32) {
        let (width, height) = i.dimensions();
        // retain the aspect ratio
        let resize_ratio = self.width as f32 / width as f32;
        let nheight = (height as f32 * resize_ratio) as u32;
        (self.width, nheight)
    }

    /// Save the frame set as a gif with the given name, without extension
    pub fn save_gif(&self, name: &str, frames: Vec<Frame>) -> Result<(), String> {
        let path = self.output_dir.join(name).with_extension("gif");
        let file = fs::File::create(&path)
            .map_err(explain!("failed to create {}", path.display()))?;

        let mut encoder = gif::Encoder::new(file);
        encoder.encode_frames(Frames::new(frames))
            .map_err(explain!("failed to write gif frames to {}", path.display()))
    }

    pub fn animate_grayscale(&self, i: &RgbaImage, frame_cnt: u32, on_frame: fn(u32)) -> Vec<Frame> {
        let (width, height) = self.resize_dimensions(i);
        let resized = imageops::resize(i, width, height, Lanczos3);

        frame_iter(frame_cnt, on_frame).map(|f| {
            Frame::from_parts(RgbaImage::from_fn(width, height, |x, y| {
                let mut px = resized.get_pixel(x, y).clone();

                // to desaturate, blend `px` with its B&w version, scaling the alpha
                let max_alpha = px[3];
                let mut desat = px.to_luma_alpha();
                desat[1] = lerp(&0., &(max_alpha as f32), &f) as u8;

                px.blend(&desat.to_rgba());

                px
            }), 0, 0, FRAME_DELAY.into())
        }).collect()
    }

    pub fn animate_resize(&self, i: &RgbaImage, frame_cnt: u32, on_frame: fn(u32)) -> Vec<Frame> {
        let (width, height) = i.dimensions();

        let mut frames: Vec<_> = frame_iter(frame_cnt, on_frame).map(|f| {
            let mut frame = RgbaImage::from_pixel(width, height,
                                                  Rgba::from_channels(255, 255, 255, 255));

            let [nwidth, nheight] = lerp(&[width as f32, height as f32], &[8., 8.], &f);
            let (nwidth, nheight) = (nwidth as u32, nheight as u32);
            // offset so the image shrinks toward center
            let left = width / 2 - (nwidth / 2);
            let top = height / 2 - (nheight / 2);

            let resized = imageops::resize(i, nwidth, nheight, Lanczos3);
            imageops::overlay(&mut frame, &resized, left, top);

            Frame::from_parts(frame, 0, 0, FRAME_DELAY.into())
        }).collect();

        // blow up the final frame using Nearest filter so we can see the individual pixels
        let smallest = imageops::resize(i, 8, 8, Lanczos3);
        let (width, height) = self.resize_dimensions(&smallest);
        let last = imageops::resize(&smallest, width, height, Nearest);

        frames.push(Frame::new(last));

        frames
    }
}

fn frame_iter(frame_cnt: u32, on_frame: fn(u32)) -> impl Iterator<Item = f32> {
    (0 ..= frame_cnt).map(move |f| { on_frame(f); f as f32 / frame_cnt as f32 })
}
