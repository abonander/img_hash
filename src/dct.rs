use std::f64::consts as f64_consts;

pub fn dct_2d(packed_2d: &[f64], width: usize, height: usize) -> Vec<f64> {
    assert!(packed_2d.len() == width * height, 
            "Slice length must be width * height!");

    let packed_2d = packed_2d.to_vec();

    let rows = rows(packed_2d, width, height);
    let dct_rows = dct_2d_part(rows);

    let columns = columns(dct_rows, width, height);
    let dct_columns = dct_2d_part(columns);

    from_columns(dct_columns, width, height)
}

#[cfg(feature = "dct-simd")]
fn dct_2d_part(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(data.len());

    for chunk in data.chunks(2) {
        if chunk.len() == 2 {
            out.extend(::dct_simd::dct_1dx2(chunk));
        } else {
            out.extend(chunk.iter().map(|vec| dct_1d(vec)));
        }
    }

    out
}

#[cfg(not(feature = "dct-simd"))]
fn dct_2d_part(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    data.iter().map(|row| dct_1d(row)).collect()
}

fn rows(packed_2d: Vec<f64>, width: usize, height: usize) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(height);

    for y in 0 .. height {
        let start = y * width;
        let end = start + width;
        rows.insert(y, packed_2d[start .. end].to_vec());         
    }

    rows
}

fn columns(rows: Vec<Vec<f64>>, width: usize, height: usize) -> Vec<Vec<f64>> {
    let mut columns: Vec<Vec<f64>> = Vec::new();

    for x in 0 .. width {
        let mut column = Vec::new();

        for y in 0 .. height {
            column.insert(y, rows[y][x]);        
        }

        columns.insert(x, column);
    }

    columns
}

fn from_columns(columns: Vec<Vec<f64>>, width: usize, height: usize) -> Vec<f64> {
    let mut packed = Vec::new();

    for y in 0 .. height {
        for x in 0 ..width {
            packed.insert(y * width + x, columns[x][y]);
        }
    }

    packed
}


// Converted from the C implementation here:
// http://unix4lyfe.org/dct/listing2.c
// Source page:
// http://unix4lyfe.org/dct/ (Accessed 8/10/2014)
fn dct_1d(vec: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();

    for u in 0 .. vec.len() {
        let mut z = 0f64;

        for x in 0 .. vec.len() {
            z += vec[x] * (f64_consts::PI * u as f64 * (2 * x + 1) as f64 
                / (2 * vec.len()) as f64).cos(); 
        }

        if u == 0 {
            z *= 1.0 / f64_consts::SQRT_2;
        }

        out.insert(u, z / 2f64);
    }

    out
}

pub fn crop_dct(dct: Vec<f64>, original: (usize, usize), new: (usize, usize)) -> Vec<f64> {
    let mut out = Vec::new();

    let (orig_width, orig_height) = original;

    assert!(dct.len() == orig_width * orig_height);

    let (new_width, new_height) = new;

    assert!(new_width < orig_width && new_height < orig_height);

    for y in 0 .. new_height {
        let start = y * orig_width;
        let end = start + new_width;

        out.push_all(&dct[start .. end]);
    }

    out
}

