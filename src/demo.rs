//! Common utilities for demoing hash types
//!
//! NOTE: not considered part of the crate's stable API
#![allow(missing_docs)]

use image::DynamicImage;

use std::{env, fs, process};

pub struct DemoConfig {
    pub image: DynamicImage,
    pub output_dir: String,
    pub width: u32,
    pub height: u32,
}

#[macro_export]
macro_rules! explain {
    ($($arg:tt)*) => { |e| format!("{}: {}", format_args!($($arg)*), e) }
}

impl DemoConfig {
    pub fn init(name: &str, alg: &str) -> Result<DemoConfig, String> {
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

        let output_dir = fs::create_dir_all(output)
            .map_err(explain!("failed to create output dir {}", output));
    }

    pub fn open_output_file(&self, name: &str) -> Result<fs::File, String> {

    }
}
