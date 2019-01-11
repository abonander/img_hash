//! Common utilities for demoing hash types
//!
//! NOTE: not considered part of the crate's stable API
#![allow(missing_docs)]

extern crate interpolation;
extern crate rusttype;

use image::{self, *};
use self::interpolation::*;
use self::rusttype::*;

use std::env;
use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process;

pub struct DemoCtxt {
    pub image: RgbaImage,
    pub output_dir: PathBuf,
    pub width: u32,
    pub font: Font<'static>,
}

const DEMO_FONT: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

pub const WHITE_A: Rgba<u8> = Rgba { data: [255; 4] };
pub const BLACK: Rgb<u8> = Rgb{ data: [0, 0, 0 ] };
pub const RED: Rgb<u8> = Rgb { data: [255, 0, 0] };
pub const GREEN: Rgb<u8> = Rgb { data: [0, 255, 0] };

#[macro_export]
macro_rules! explain {
    ($($arg:tt)*) => { |e| format!("{}: {}", format_args!($($arg)*), e) }
}

impl DemoCtxt {
    pub fn init(name: &str, alg: &str) -> Result<DemoCtxt, String> {
        let font = Font::from_bytes(DEMO_FONT).expect("failed to read font");

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
            font,
        })
    }

    pub fn resize_dimensions(&self, i: &RgbaImage) -> (u32, u32) {
        let (width, height) = i.dimensions();
        // retain the aspect ratio
        let resize_ratio = self.width as f32 / width as f32;
        let nheight = (height as f32 * resize_ratio) as u32;
        (self.width, nheight)
    }

    pub fn text_scale(&self) -> Scale {
        Scale::uniform(self.width as f32 / 20.)
    }

    /// Save the frame set as a gif with the given name, without extension
    pub fn save_gif(&self, name: &str, frames: Vec<Frame>) -> Result<(), String> {
        let path = self.output_dir.join(name).with_extension("gif");
        let file = fs::File::create(&path)
            .map_err(explain!("failed to create {}", path.display()))?;

        let mut encoder = gif::Encoder::new(BufWriter::new(file));
        encoder.encode_frames(Frames::new(frames))
            .map_err(explain!("failed to write gif frames to {}", path.display()))
    }

    pub fn animate_grayscale(&self, i: &RgbaImage, frame_cnt: u32, frame_delay: u16) -> Vec<Frame> {
        let (width, height) = self.resize_dimensions(i);
        let resized = imageops::resize(i, width, height, Lanczos3);

        frame_iter(frame_cnt).map(|f| {
            Frame::from_parts(RgbaImage::from_fn(width, height, |x, y| {
                let mut px = resized.get_pixel(x, y).clone();

                // to desaturate, blend `px` with its B&w version, scaling the alpha
                let max_alpha = px[3];
                let mut desat = px.to_luma_alpha();
                desat[1] = lerp(&0., &(max_alpha as f32), &f) as u8;

                px.blend(&desat.to_rgba());

                px
            }), 0, 0, frame_delay.into())
        }).collect()
    }

    pub fn animate_resize(&self, i: &RgbaImage, rwidth: u32, rheight: u32, frame_cnt: u32,
                          frame_delay: u16) -> Vec<Frame> {
        let (width, height) = i.dimensions();

        let mut frames: Vec<_> = frame_iter(frame_cnt).map(|f| {
            let mut frame = RgbaImage::from_pixel(width, height,
                                                  Rgba::from_channels(255, 255, 255, 255));

            let [nwidth, nheight] = lerp(&[width as f32, height as f32],
                                         &[rwidth as f32, rheight as f32], &f);
            let (nwidth, nheight) = (nwidth as u32, nheight as u32);
            // offset so the image shrinks toward center
            let left = width / 2 - (nwidth / 2);
            let top = height / 2 - (nheight / 2);

            let resized = imageops::resize(i, nwidth, nheight, Lanczos3);
            imageops::overlay(&mut frame, &resized, left, top);

            Frame::from_parts(frame, 0, 0, frame_delay.into())
        }).collect();

        // blow up the final frame using Nearest filter so we can see the individual pixels
        let smallest = imageops::resize(i, 8, 8, Lanczos3);
        let (width, height) = self.resize_dimensions(&smallest);
        let last = imageops::resize(&smallest, width, height, Nearest);

        frames.push(Frame::new(last));

        frames
    }

    pub fn layout_text<'a, 'b>(&'a self, text: &'b str, x: u32, y: u32) -> LayoutIter<'a, 'b> {
        self.font.layout(text, self.text_scale(), Point { x: x as f32, y: y as f32 })
    }
}

fn frame_iter(frame_cnt: u32) -> impl Iterator<Item = f32> {
    (0 ..= frame_cnt).map(move |f| f as f32 / frame_cnt as f32)
}

/// Create an iterator that generates (x, y) coordinate pairs in row-major order
pub fn x_y_iter(width: u32, height: u32) -> impl Iterator<Item = (u32, u32)> {
    (0 .. height).flat_map(move |y| (0 .. widht).map(move |x| (x, y)))
}

pub fn draw_glyph(buf: &mut RgbaImage, glyph: &PositionedGlyph, color: &Rgb<u8>) {
    let Point { x, y } = glyph.position();
    let (pos_x, pos_y) = (x as u32, y as u32);

    // this doesn't provide offsets from the glyph position
    glyph.draw(|x, y, a| {
        let mut rgba = color.to_rgba();
        rgba[3] = (a * 255.) as u8;
        buf.get_pixel_mut(pos_x + x, pos_y + y).blend(&rgba);
    })
}

pub fn center_in_area(glyph: PositionedGlyph, width: u32, height: u32) -> PositionedGlyph {
    let Point { x, y } = glyph.position();
    let (x, y) = (x as u32, y as u32);

    let bounding_box = glyph.pixel_bounding_box().expect("need pixel bounding box to center");
    let Vector { x: gwidth, y: gheight } = bounding_box.max - bounding_box.min;
    let (gwidth, gheight) = (gwidth as u32, gheight as u32);
    assert!(gwidth <= width);
    assert!(gheight <= height);

    let new_x = x + (width - gwidth) / 2;
    let new_y = y + (width - gheight) / 2;

    glyph.into_unpositioned().positioned(Point { x: new_x as f32, y: new_y as f32 })
}

// at bitstring lengths above this value, ellipsize the middle
pub const MAX_BITSTR_LEN: usize = 16;
const ELLIPSIS_START: usize = 6;
const ELLIPSIS_PAT: &str = "[...]";

pub struct Bitstring(String);

impl Bitstring {
    pub fn new() -> Self { Bitstring(String::new()) }
    pub fn push_bit(&mut self, bit: bool) {
        self.0.push(if bit { '1' } else { '0' });
        if self.0.len() > MAX_BITSTR_LEN {
            let end = ELLIPSIS_START + ELLIPSIS_PAT.len();
            // clip the `end`th bit out of the string so it doesn't get longer
            self.0.replace_range(ELLIPSIS_START ..= end, ELLIPSIS_PAT);
        }
    }

    pub fn as_str(&self) -> &str { &self.0 }
}

pub struct Outline {
    pub inner_width: u32,
    pub inner_height: u32,
    pub thickness: u32,
}

impl Outline {
    pub fn new(inner_width: u32, inner_height: u32, thickness: u32) -> Self {
        Outline { inner_width, inner_height, thickness }
    }

    /// NOTE: x and y are the **inside** top-left corner of the outline
    pub fn draw(&self, i: &mut RgbaImage, x: u32, y: u32, color: Rgb<u8>) {
        let Outline { inner_width, inner_height, thickness } = *self;

        let x = x - self.thickness;
        let y = y - self.thickness;

        let outer_width = inner_width + thickness * 2;
        let outer_height = inner_height + thickness * 2;

        // draw the outline for pixels where:
        // `x` is less than the former or greater than the latter, OR
        let lower_x = x + thickness;
        let upper_x = x + outer_width - thickness;
        let max_x = x + outer_width;

        // `y` is less than the former or greater than the latter
        let lower_y = y + thickness;
        let upper_y = y + outer_height - thickness;
        let max_y = y + outer_height;

        (y .. lower_y).chain(upper_y .. max_y)
            .flat_map(|y| (x .. max_x).map(move |x| (x, y))) // top and bottom bars
            .chain(
                // left and right bars
                (lower_y .. upper_y).flat_map(|y|
                    (x .. lower_x).chain(upper_x .. max_x).map(move |x| (x, y))
                )
            )
            .for_each(|(x, y)| i.put_pixel(x, y, color.to_rgba()))
    }
}
