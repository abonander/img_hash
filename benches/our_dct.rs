#[macro_use]
extern crate criterion;
extern crate img_hash;

use img_hash::dct;

use criterion::Criterion;

fn dct_08x08(c: &mut Criterion) {
    let coeffs = dct::Coefficients::precompute(8, 8);
    let signal: Vec<_> = (0 .. 64).map(|x| x as f32).collect();

    c.bench_function("Our DCT 8x8", move |b| b.iter(|| dct::dct_2d(&signal, 8, &coeffs)));
}

fn dct_16x16(c: &mut Criterion) {
    let coeffs = dct::Coefficients::precompute(16, 16);
    let signal: Vec<_> = (0 .. 256).map(|x| x as f32).collect();

    c.bench_function("Our DCT 16x16", move |b| b.iter(|| dct::dct_2d(&signal, 16, &coeffs)));
}

criterion_group!(benches, dct_08x08, dct_16x16);
criterion_main!(benches);
