const LARGE_HSIZE: u32 = (HSIZE * HASH_SIZE_MULTIPLIER);
const LARGE_SQUARE_HSIZE: usize = (LARGE_HSIZE * LARGE_HSIZE) as usize;
const SQUARE_HSIZE: usize = (HSIZE * HSIZE) as usize;

// by establishing that the DCT will be constant-length, we should get some pretty sweet loop
// unrolling, constant-propagation and reordering to make this stupid fast,
// not to mention stack-local scratch space
pub fn dct_hash_HSIZE<I: HashImage>(img: &I) -> BitVec {
    let hash_values: Vec<_> = prepare_image(img, LARGE_HSIZE, LARGE_HSIZE)
        .into_iter().map(|val| (val as f64) / 255.0).collect();

    let mut dct = dct_2d_HSIZE(&hash_values);

    let cropped_dct = crop_2d_HSIZE(&mut dct);

    let mean = cropped_dct.iter().fold(0f64, |b, &a| a + b) / cropped_dct.len() as f64;

    cropped_dct.iter().map(|&x| x >= mean).collect()
}

fn dct_2d_HSIZE(packed_2d: &[f64]) -> [f64; LARGE_SQUARE_HSIZE] {
    assert_eq!(packed_2d.len(), LARGE_SQUARE_HSIZE);

    let mut row_pass = [0.; LARGE_SQUARE_HSIZE];
    let mut col_pass = [0.; LARGE_SQUARE_HSIZE];

    {
        for (row_in, row_out) in packed_2d.chunks(LARGE_HSIZE as usize)
            .zip(row_pass.chunks_mut(LARGE_HSIZE as usize)) {
            dct_1d_dyn(row_in, row_out, LARGE_HSIZE as usize);
        }

        for (col_in, mut col_out) in Columns::from_slice(&row_pass, LARGE_HSIZE as usize)
            .zip(ColumnsMut::from_slice(&mut col_pass, LARGE_HSIZE as usize)) {
            dct_1d_dyn(&col_in, &mut col_out, LARGE_HSIZE as usize);
        }
    }

    col_pass
}

fn crop_2d_HSIZE(packed: &mut [f64; LARGE_SQUARE_HSIZE]) -> &[f64] {
    for i in 1 .. HSIZE as usize {
        for j in 0 .. HSIZE as usize {
            packed[i * HSIZE as usize + j] = packed[i * LARGE_HSIZE as usize + j];
        }
    }

    &packed[.. SQUARE_HSIZE as usize]
}
