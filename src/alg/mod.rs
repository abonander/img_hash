mod block;

use {HashBytes, HashCtxt, Image};

use self::HashAlg::*;

/// Hash algorithms implemented by this crate.
///
/// Implemented primarily based on the high-level descriptions on the blog Hacker Factor
/// written by Dr. Neal Krawetz: http://www.hackerfactor.com/
///
/// ### Choosing an Algorithm
/// Each algorithm has different performance characteristics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum HashAlg {
    /// The Mean hashing algorithm.
    ///
    /// The image is converted to grayscale, scaled down to `hash_width x hash_height`,
    /// the mean pixel value is taken, and then the hash bits are generated by comparing
    /// the pixels of the descaled image to the mean.
    ///
    /// This is the most basic hash algorithm supported, resistant only to changes in
    /// resolution, aspect ratio, and overall brightness.
    ///
    /// Further Reading:
    /// http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
    Mean,

    /// The Gradient hashing algorithm.
    ///
    /// The image is converted to grayscale, scaled down to `width + 1 xheight`,
    /// and then in row-major order the pixels are compared with each other, setting bits
    /// in the hash for each comparison. The extra pixel is needed to have `width` comparisons
    /// per row.
    ///
    /// This hash algorithm is as fast or faster than Mean (because it only traverses the
    /// hash data once) and is more resistant to changes than Mean.
    ///
    /// Further Reading:
    /// http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    Gradient,

    /// The Vertical-Gradient hashing algorithm.
    ///
    /// Equivalent to [`Gradient`](Self::Gradient) but operating on the columns of the image
    /// instead of the rows.
    VertGradient,

    /// The Double-Gradient hashing algorithm.
    ///
    /// An advanced version of [`Gradient`](Self::Gradient);
    /// resizes the grayscaled image to `(width / 2 + 1) x (height / 2 + 1)` and compares columns
    /// in addition to rows.
    ///
    /// This algorithm is slightly slower than `hash_alg_gradient()` (resizing the image dwarfs
    /// the hash time in most cases) but the extra comparison direction may improve results (though
    /// you might want to consider increasing `hash_size` to accommodate the extra comparisons).
    DoubleGradient,

    /// The [Blockhash.io](https://blockhash.io) algorithm.
    ///
    /// Compared to the other algorithms, this does not require any preprocessing steps and so
    /// may be significantly faster at the cost of some resilience.
    ///
    /// The algorithm is described in a high level here:
    /// https://github.com/commonsmachinery/blockhash-rfc/blob/master/main.md
    Blockhash,
    /// EXHAUSTIVE MATCHING IS NOT RECOMMENDED FOR BACKWARDS COMPATIBILITY
    #[doc(hidden)]
    #[serde(skip)]
    __Nonexhaustive,
}

fn next_multiple_of_2(x: u32) -> u32 { x + 1 & !1 }
fn next_multiple_of_4(x: u32) -> u32 { x + 3 & !3 }

impl HashAlg {
    pub (crate) fn hash_image<I, B>(&self, ctxt: &HashCtxt, image: &I) -> B
    where I: Image, B: HashBytes {
        let cow_gaussed = ctx.gauss_preproc();

        if *self == Blockhash {

        }
    }

    pub (crate) fn round_hash_size(&self, width: u32, height: u32) -> (u32, u32) {
        match *self {
            DoubleGradient => (next_multiple_of_2(width), next_multiple_of_2(height)),
            Blockhash => (next_multiple_of_4(width), next_multiple_of_4(height)),
        }
    }

    fn resize_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        match *self {
            Mean => (width, height),
            Blockhash => panic!("Blockhash algorithm does not resize"),
            Gradient => (width + 1, height),
            VertGradient => (width, height + 1),
            DoubleGradient => (width / 2 + 1, height / 2 + 1),
        }
    }
}

fn mean_hash<I: Image, B: HashBytes>(img: &I, hash_size: u32) -> B {
    let hash_values = prepare_image(img, hash_size, hash_size);

    let mean = hash_values.iter().fold(0u32, |b, &a| a as u32 + b)
        / hash_values.len() as u32;

    hash_values.into_iter().map(|x| x as u32 >= mean)
}


/// The guts of the gradient hash,
/// separated so we can reuse them for both `Gradient` and `DoubleGradient`.
fn gradient_hash_impl<I: ops::Index<usize, Output=u8> + ?Sized>(bytes: &I, len: u32, bitv: &mut HashBytes) {
    let len = len as usize;

    for i in 1 .. len {
        let this = &bytes[i];
        let last = &bytes[i - 1];

        bitv.push(last < this);
    }
}

fn gradient_hash<I: Image>(img: &I, hash_size: u32) -> HashBytes {
    // We have one extra pixel in width so we have `hash_size` comparisons per row.
    let bytes = prepare_image(img, hash_size + 1, hash_size);
    let mut bitv = HashBytes::with_capacity((hash_size * hash_size) as usize);

    for row in bytes.chunks((hash_size + 1) as usize) {
        gradient_hash_impl(row, hash_size, &mut bitv);
    }

    bitv
}

fn double_gradient_hash<I: Image>(img: &I, hash_size: u32) -> HashBytes {
    // We have one extra pixel in each dimension so we have `hash_size` comparisons.
    let rowstride = hash_size + 1;
    let bytes = prepare_image(img, rowstride, rowstride);
    let mut bitv = HashBytes::with_capacity((hash_size * hash_size * 2) as usize);


    for row in bytes.chunks(rowstride as usize) {
        gradient_hash_impl(row, rowstride, &mut bitv);
    }

    for column in Columns::from_slice(&bytes, rowstride as usize) {
        gradient_hash_impl(&column, hash_size, &mut bitv);
    }

    bitv
}