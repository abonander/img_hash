// Implementation of block-hashing as described here:
// https://github.com/commonsmachinery/blockhash-rfc/blob/master/main.md#process-of-identifier-assignment
// Main site: blockhash.io

use super::HashImage;

use algo::select;

use bit_vec::BitVec;

use std::cmp::Ordering;

const FLOAT_EQ_MARGIN: f64 = 0.001;

pub fn blockhash<I: HashImage>(img: &I, size: u32) -> BitVec {
    let size = next_multiple_of_4(size);
    let (width, height) = img.dimensions(); 

    // Skip the floating point math if it's unnecessary
    if width % size == 0 && height % size == 0 {
        blockhash_fast(img, size)
    } else {
        blockhash_slow(img, size)
    }        
} 

macro_rules! gen_hash {
    ($imgty:ty, $valty:ty, $blocks: expr, $size:expr, $block_width:expr, $block_height:expr, $eq_fn:expr) => ({
        let channel_count = <$imgty as HashImage>::channel_count() as u32;

        let group_len = (($size * $size) / 4) as usize;

        let block_area = $block_width * $block_height;

        let cmp_factor = match channel_count {
            3 | 4 => 255u32 as $valty * 3u32 as $valty,
            2 | 1 => 255u32 as $valty,
            _ => panic!("Unrecognized channel count from HashImage: {}", channel_count),
        }  
            * block_area 
            / (2u32 as $valty);

        let medians: Vec<$valty> = $blocks.chunks(group_len).map(get_median).collect();

        $blocks.chunks(group_len).zip(medians)
            .flat_map(|(blocks, median)| 
                blocks.iter().map(move |&block| 
                    block > median ||
                        ($eq_fn(block, median) && median > cmp_factor)
                )
            )
            .collect()
    })
}

fn blockhash_slow<I: HashImage>(img: &I, size: u32) -> BitVec {
    let mut blocks = vec![0f64; (size * size) as usize];

    let (width, height) = img.dimensions();
    
    // Block dimensions, in pixels
    let (block_width, block_height) = (width as f64 / size as f64, height as f64 / size as f64);

    let idx = |x, y| (y * size + x) as usize;

    img.foreach_pixel(|x, y, px| {
        let px_sum = sum_px(px) as f64;

        let (x, y) = (x as f64, y as f64);

        let block_x = (x / block_width).floor();
        let block_y = (y / block_height).floor();

        let div_x = block_x * block_width;
        let div_y = block_y * block_height;

        let horz_overlap = div_x - x < 1.0;
        let vert_overlap = div_y - y < 1.0;

        // quadrant I
        let area_a = div_x - x * div_y - y;
        // quad II
        let area_b = (x + 1.0 - div_x) * div_y - y;
        // quad III
        let area_c = div_x - x * (y + 1.0 - div_y);
        // quad IV
        let area_d = (x + 1.0 - div_x) * (y + 1.0 - div_y);

        let block_x = block_x as u32;
        let block_y = block_y as u32;

        match (horz_overlap, vert_overlap) {
            (true, true) => blocks[idx(block_x + 1, block_y +1)] += px_sum * area_d,
            (true, _) => blocks[idx(block_x + 1, block_y)] += px_sum * area_b,
            (_, true) => blocks[idx(block_x, block_y + 1)] += px_sum * area_c,
            _ => (),
        }

        blocks[idx(block_x, block_y)] += px_sum * area_a;
    });

    
    gen_hash!(I, f64, blocks, size, block_width, block_height,
        |l: f64, r: f64| (l - r).abs() < FLOAT_EQ_MARGIN)
}

fn blockhash_fast<I: HashImage>(img: &I, size: u32) -> BitVec {
    let mut blocks = vec![0u32; (size * size) as usize];
    let (width, height) = img.dimensions();

    let (block_width, block_height) = (width / size, height / size);

    let idx = |x, y| (y * size + x) as usize;

    img.foreach_pixel(|x, y, px| { 
        let px_sum = sum_px(px);

        let block_x = x / block_width;
        let block_y = y / block_width;

        blocks[idx(block_x, block_y)] += px_sum;
    });

    gen_hash!(I, u32, blocks, size, block_width, block_height, |l, r| l == r)    
}

#[inline(always)]
fn sum_px(px: &[u8]) -> u32 {
    // Branch prediction should eliminate the match after a few iterations
    match px.len() {
        4 => if px[3] == 0 { 255 * 3 } else { sum_px(&px[..3]) },
        3 => px[0] as u32 + px[1] as u32 + px[2] as u32,
        2 => if px[1] == 0 { 255 } else { px[0] as u32 },
        1 => px[0] as u32,
        // We can only hit this assertion if there's a bug where the number of values
        // per pixel doesn't match HashImage::channel_count
        _ => panic!("Channel count was different than actual pixel size"),
    }
}

// Get the next multiple of 4 up from x, or x if x is a multiple of 4
fn next_multiple_of_4(x: u32) -> u32 {
    x + 3 & !3
}

fn get_median<T: PartialOrd + Copy>(data: &[T]) -> T {
    *select::median_by(data, |l, r| l.partial_cmp(r).unwrap_or(Ordering::Less))
}