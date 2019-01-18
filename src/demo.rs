//! Common utilities for demoing hash types
//!
//! NOTE: not considered part of the crate's stable API
#![allow(missing_docs)]

extern crate interpolation;
extern crate rusttype;

pub use image::{self, *};
pub use self::interpolation::*;
pub use self::rusttype::*;

pub use dct::DctCtxt;

use std::cmp;
use std::env;
use std::fs;
use std::io::BufWriter;
use std::iter;
use std::path::PathBuf;
use std::process;

#[derive(Clone)]
pub struct DemoCtxt {
    pub images: Vec<RgbaImage>,
    pub output_dir: PathBuf,
    pub width: u32,
    pub font: Font<'static>,
}

const DEMO_FONT: &[u8] = include_bytes!("../assets/AverageMono.ttf");

pub const WHITE_A: Rgba<u8> = Rgba { data: [255; 4] };
pub const BLACK_A: Rgba<u8> = Rgba { data: [0, 0, 0, 255] };
pub const BLACK: Rgb<u8> = Rgb{ data: [0; 3 ] };
pub const RED: Rgb<u8> = Rgb { data: [255, 0, 0] };
pub const GREEN: Rgb<u8> = Rgb { data: [0, 255, 0] };

#[macro_export]
#[doc(hidden)]
macro_rules! explain {
    ($($arg:tt)*) => { |e| format!("{}: {}", format_args!($($arg)*), e) }
}

impl DemoCtxt {
    pub fn init(name: &str, purpose: &str, img_cnt: usize) -> Result<DemoCtxt, String> {
        let font = Font::from_bytes(DEMO_FONT).expect("failed to read font");

        let args = env::args().collect::<Vec<_>>();

        if args.len() != 3 + img_cnt {
            let file_args = if img_cnt == 1 {
                "[FILE]".to_string()
            } else {
                (1 ..= img_cnt).map(|i| format!("[FILE {}]", i)).collect::<Vec<_>>().join(" ")
            };

            println!("args: {:?}", args);
            println!("\
                usage: {name} {file_args} [OUTPUT-DIR] [WIDTH]\r\n\
                {purpose}, exporting gifs of each step to OUTPUT-DIR\r\n\
                each gif will be WIDTH wide; aspect ratio is fixed\r\n\
             ", name = name, file_args = file_args, purpose = purpose);
            process::exit(0);
        }

        let output_dir = PathBuf::from(&args[img_cnt + 1]);
        let width = args[img_cnt + 2].parse()
            .map_err(explain!("could not parse WIDTH: {}", args[img_cnt + 2]))?;

        let images = args[1 .. img_cnt + 1].iter().map(|file|
            image::open(file).map(|i| i.to_rgba()).map_err(explain!("failed to open {}", file))
        ).collect::<Result<Vec<_>, _>>()?;

        fs::create_dir_all(&output_dir)
            .map_err(explain!("failed to create output dir {}", output_dir.display()))?;

        Ok(Self {
            images,
            output_dir,
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
        encoder.encode_frames(frames)
            .map_err(explain!("failed to write gif frames to {}", path.display()))
    }

    pub fn animate_grayscale(&self, i: &RgbaImage, ms: u16, fps: u16) -> Vec<Frame> {
        let (width, height) = self.resize_dimensions(i);
        let resized = imageops::resize(i, width, height, Lanczos3);

        let (frame_delay, frame_cnt) = frame_delay_cnt(ms, fps);
        frame_iter(frame_cnt as u32).map(|aweight| {
            Frame::from_parts(RgbaImage::from_fn(width, height, |x, y| {
                let mut px = resized.get_pixel(x, y).clone();

                // to desaturate, blend `px` with its B&w version, scaling the alpha
                let max_alpha = px[3];
                let mut desat = px.to_luma_alpha();
                desat[1] = lerp(&0, &max_alpha, &aweight);

                px.blend(&desat.to_rgba());

                px
            }), 0, 0, frame_delay.into())
        }).collect()
    }

    pub fn animate_resize(&self, i: &RgbaImage, rwidth: u32, rheight: u32, ms: u16, fps: u16)
        -> Vec<Frame> {
        let (iwidth, iheight) = i.dimensions();

        let start = [iwidth, iheight];
        let end = [rwidth, rheight];
        let mut frames: Vec<_> = lerp_iter(start, end, ms, fps).map(
            |([nwidth, nheight], frame_delay)| {
            let mut frame = rgba_fill_white(iwidth, iheight);
            // offset so the image shrinks toward center
            let (left, top) = center_at_point(iwidth / 2, iheight / 2, nwidth, nheight);

            let resized = imageops::resize(i, nwidth, nheight, Lanczos3);
            imageops::overlay(&mut frame, &resized, left, top);

            Frame::from_parts(frame, 0, 0, frame_delay.into())
        }).collect();

        // blow up the final frame using Nearest filter so we can see the individual pixels
        let smallest = imageops::resize(i, rwidth, rheight, Lanczos3);
        let (nwidth, nheight) = dimen_fill_area(smallest.dimensions(), (iwidth, iheight));
        let resized = imageops::resize(&smallest, nwidth, nheight, Nearest);

        let (left, top) = center_at_point(iwidth / 2, iheight / 2, nwidth, nheight);
        let mut last = rgba_fill_white(iwidth, iheight);
        imageops::overlay(&mut last, &resized, left, top);

        let frame_delay = frames.last().unwrap().delay();
        frames.push(Frame::from_parts(last, 0, 0, frame_delay));

        frames
    }

    pub fn layout_text<'a, 'b>(&'a self, text: &'b str, x: u32, y: u32) -> LayoutIter<'a, 'b> {
        self.font.layout(text, self.text_scale(), Point { x: x as f32, y: y as f32 })
    }
}

pub fn point_f32(x: u32, y: u32) -> Point<f32> {
    Point { x: x as f32, y: y as f32 }
}

/// Given dimensions and the size of the region, find new dimensions that fill the region
/// while retaining aspect ratio
pub fn dimen_fill_area(dimen: (u32, u32), region: (u32, u32)) -> (u32, u32) {
    let (width, height) = dimen;
    let (rwidth, rheight) = region;

    let dratio = (width as f32 / height as f32);
    let rratio = (rwidth as f32 / rheight as f32);

    if dratio > rratio {
        // the dimensions are wider than the region, fit the width and scale the size accordingly
        (rwidth, fmul(rwidth, 1. / dratio))
    } else if dratio == rratio {
        // aspect ratios are exactly the same, just resize to the region
        (rwidth, rheight)
    } else {
        // dimensions are narrower than the region, fit the height and scale accordingly
        (fmul(rheight, dratio), rheight)
    }
}

pub fn rgba_fill_white(width: u32, height: u32) -> RgbaImage {
    ImageBuffer::from_pixel(width, height, WHITE_A)
}

/// Fill a rectangle in `img` with the given color
pub fn fill_color(img: &mut RgbaImage, color: Rgba<u8>, x: u32, y: u32, width: u32, height: u32) {
    for (off_x, off_y) in x_y_iter(width, height) {
        *img.get_pixel_mut(x + off_x, y + off_y) = color;
    }
}

/// Multiply a `u32` by an `f32` with a truncated result
pub fn fmul(x: u32, y: f32) -> u32 {
    (x as f32 * y) as u32
}

/// Cast a luminance value to an `Rgba` with full alpha
pub fn luma_rgba(luma: u8) -> Rgba<u8> {
    Luma{ data: [luma] }.to_rgba()
}

/// Generate an iterator that yields `frame_cnt` times in `[0, 1]`
pub fn frame_iter(frame_cnt: u32) -> impl Iterator<Item = f32> {
    (0 ..= frame_cnt).map(move |f| f as f32 / frame_cnt as f32)
}

/// Given the number of millis and framerate, return the frame delay and count, respectively.
pub fn frame_delay_cnt(ms: u16, fps: u16) -> (u16, u16) {
    assert!(fps <= 100);
    assert_ne!(fps, 0);
    let frame_delay = 100 / fps;
    let frame_cnt = ms / (1000 / fps);
    (frame_delay, frame_cnt)
}

/// Lerp between two interpolatable values over the given number of milliseconds at the given
/// framerate, yielding for each frame the lerped value and the frame delay
pub fn lerp_iter<S: Lerp<Scalar = f32>>(start: S, end: S, ms: u16, fps: u16)
    -> impl Iterator<Item = (S, u16)> {
    let (frame_delay, frame_cnt) = frame_delay_cnt(ms, fps);

    (0 ..= frame_cnt).map(move |f| {
        let weight = f as f32 / frame_cnt as f32;
        (lerp(&start, &end, &weight), frame_delay)
    })
}

/// Iterate over a cubic bezier with the given parameters, over the given number of milliseconds
/// at the given framerate, yielding for each frame the interpolated value and the frame delay.
pub fn bez3_iter<S: Lerp<Scalar = f32>>(params: [S; 4], ms: u16, fps: u16)
    -> impl Iterator<Item = (S, u16)> {
    let (frame_delay, frame_cnt) = frame_delay_cnt(ms, fps);

    (0 ..= frame_cnt).map(move |f| {
        let [ref start, ref ctl1, ref ctl2, ref end] = params;

        let weight = f as f32 / frame_cnt as f32;
        let pt = cub_bez(start, ctl1, ctl2, end, &weight);
        (pt, frame_delay)
    })
}

/// Create an iterator that generates (x, y) coordinate pairs in row-major order
pub fn x_y_iter(width: u32, height: u32) -> impl Iterator<Item = (u32, u32)> {
    (0 .. height).flat_map(move |y| (0 .. width).map(move |x| (x, y)))
}

pub fn draw_glyph(buf: &mut RgbaImage, glyph: &PositionedGlyph, color: &Rgb<u8>) {
    draw_glyph_sampled(buf, glyph, |_, _| color.to_rgba())
}

pub fn draw_glyph_sampled(buf: &mut RgbaImage, glyph: &PositionedGlyph,
                          mut sample: impl FnMut(u32, u32) -> Rgba<u8>) {

    let Point { x, y } = glyph.position();
    let (pos_x, pos_y) = (x as u32, y as u32);

    // this doesn't provide offsets from the glyph position
    glyph.draw(|x, y, a| {
        let mut rgba = sample(x, y);
        rgba[3] = (a * rgba[3] as f32) as u8;
        buf.get_pixel_mut(pos_x + x, pos_y + y).blend(&rgba);
    })
}

/// Given a point and the width and height of an object, give coordinates that will center
/// it on that point
pub fn center_at_point(x: u32, y: u32, width: u32, height: u32) -> (u32, u32) {
    assert!(x >= (width / 2));
    assert!(y >= (height / 2));

    (x - width / 2, y - height / 2)
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

pub fn size_of_text(layout: &LayoutIter) -> (u32, u32) {
    let mut layout = layout.clone().flat_map(|g| g.pixel_bounding_box());

    // get the text dimensions by subtracting the low point on the far left
    // from the high point on the far right
    let min = layout.next().map_or(Point { x: 0, y: 0 }, |bb| bb.min);
    let max = layout.last().map_or(Point { x: 0, y: 0 }, |bb| bb.max);

    let Vector { x: text_width, y: text_height } = max - min;

    (text_width as u32, text_height as u32)
}

pub fn center_text_in_area<'a, 'b>(layout: LayoutIter<'a, 'b>, width: u32, height: u32)
    -> iter::Map<LayoutIter<'a, 'b>, impl FnMut(PositionedGlyph<'a>) -> PositionedGlyph<'a>> {
    let (text_width, text_height) = size_of_text(&layout);

    assert!(text_width <= width, "text too wide: {} <= {}", text_width, width);
    assert!(text_height <= height, "text too tall: {} <= {}", text_height, height);

    let x = (width - text_width) as f32 / 2.0;
    let y = (height - text_height) as f32 / 2.0;

    layout.map(move |g| {
        let pos = g.position() + Vector { x, y };
        g.into_unpositioned().positioned(pos)
    })
}

pub fn overlay_generic<B, F>(bg: &mut B, fg: &F, x: u32, y: u32)
where B: GenericImage, F: GenericImageView<Pixel = B::Pixel> {
    let bg_dims = bg.dimensions();
    let fg_dims = fg.dimensions();

    // Crop our foreground image if we're going out of bounds
    let (range_width, range_height) = imageops::overlay_bounds(bg_dims, fg_dims, x, y);

    for fg_y in 0..range_height {
        for fg_x in 0..range_width {
            let p = fg.get_pixel(fg_x, fg_y);
            let mut bg_pixel = bg.get_pixel(x + fg_x, y + fg_y);
            bg_pixel.blend(&p);

            bg.put_pixel(x + fg_x, y + fg_y, bg_pixel);
        }
    }
}

// at bitstring lengths above this value, ellipsize the middle
pub const MAX_BITSTR_LEN: usize = 13;
const ELLIPSIS_START: usize = 4;
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

        assert!(max_x < i.dimensions().0 && max_y < i.dimensions().1,
                "outline will fall out of bounds: {:?}, bounds: {:?}",
                (max_x, max_y), i.dimensions());


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

#[test]
fn test_dimen_fill_area() {
    assert_eq!(dimen_fill_area((1280, 955), (1280, 640)), (857, 640));
    assert_eq!(dimen_fill_area((8, 8), (1280, 640)), (640, 640));
}