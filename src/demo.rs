//! Common utilities for demoing hash types
//!
//! NOTE: not considered part of the crate's stable API
#![allow(missing_docs)]

use image::{DynamicImage, gif};

use std::env;
use std::fs::{self, File};
use std::path::PathBuf;

pub struct DemoCtxt {
    pub image: DynamicImage,
    pub output_dir: PathBuf,
    pub width: u32,
    pub height: u32,
}

#[macro_export]
macro_rules! explain {
    ($($arg:tt)*) => { |e| format!("{}: {}", format_args!($($arg)*), e) }
}

impl DemoCtxt {
    pub fn init(name: &str, alg: &str) -> Result<DemoCtxt, String> {
        let args = env::args().collect::<Vec<_>>();

        if args.len() != 5 {
            return Err(format!("\
                usage: {name} [FILE] [OUTPUT-DIR] [WIDTH] [HEIGHT]\n\
                demos `{alg}` for FILE, exporting gifs of each step to OUTPUT-DIR\n\
                each gif will be WIDTH x HEIGHT in size; the hash config is fixed\n\
          ", name = name, alg = alg));
        }

        let file = &args[1];
        let output = &args[2];
        let width = args[3].parse::<u32>().map_err(explain!("could not parse WIDTH: {}", args[3]))?;
        let height = args[4].parse::<u32>().map_err(explain!("could not parse HEIGHT: {}", args[4]))?;

        let image = image::open(file).map_err(explain!("failed to open {}", file))?;

        fs::create_dir_all(output).map_err(explain!("failed to create output dir {}", output));

        Ok(Self {
            image,
            output_dir: output.into(),
            width,
            height
        })
    }

    /// Save the frame set as a gif with the given name, without extension
    pub fn save_gif(&self, name: &str, frames: image::Frames) -> Result<(), String> {
        let path = self.output_dir.join(name).with_extension(".gif");
        let file = fs::File::create(&path).map_err(explain!("failed to create {}", path.display()));

        let mut encoder = gif::Encoder::new(file);
        encoder.encode_frames(frames)
            .map_err(explain!("failed to write gif frames to {}", path.display()))
    }
}
