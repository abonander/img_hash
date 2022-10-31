//! A crate that provides several perceptual hashing algorithms for images.
//! Supports images opened with the [image] crate from Piston.
//!
//! ```rust,no_run
//! # use img_hash::{HasherConfig, HashAlg};
//! let image1 = image::open("image1.png").unwrap();
//! let image2 = image::open("image2.png").unwrap();
//!     
//! let hasher = HasherConfig::new().to_hasher();
//!
//! let hash1 = hasher.hash_image(&image1);
//! let hash2 = hasher.hash_image(&image2);
//!     
//! println!("Image1 hash: {}", hash1.to_base64());
//! println!("Image2 hash: {}", hash2.to_base64());
//!     
//! println!("Hamming Distance: {}", hash1.dist(&hash2));
//! ```
//! [image]: https://github.com/PistonDevelopers/image
#![deny(missing_docs)]

use std::{borrow::Cow, fmt, marker::PhantomData};

mod alg;
mod dct;
mod traits;

use dct::DctCtxt;
use image::imageops;
use image::GrayImage;
use serde::{Deserialize, Serialize};

pub use image::imageops::FilterType;

pub use alg::HashAlg;

pub(crate) use traits::BitSet;

pub use traits::{DiffImage, HashBytes, Image};

/// **Start here**. Configuration builder for [`Hasher`](::Hasher).
///
/// Playing with the various options on this struct allows you to tune the performance of image
/// hashing to your needs.
///
/// Sane, reasonably fast defaults are provided by the [`::new()`](#method.new) constructor. If
/// you just want to start hashing images and don't care about the details, it's as simple as:
///
/// ```rust
/// use img_hash::HasherConfig;
///
/// let hasher = HasherConfig::new().to_hasher();
/// // hasher.hash_image(image);
/// ```
///
/// # Configuration Options
/// The hash API is highly configurable to tune both performance characteristics and hash
/// resilience.
///
/// ### Hash Size
/// Setter: [`.hash_size()`](#method.hash_size)
///
/// Dimensions of the final hash, as width x height, in bits. A hash size of `8, 8` produces an
/// 8 x 8 bit (8 byte) hash. Larger hash sizes take more time to compute as well as more memory,
/// but aren't necessarily better for comparing images. The best hash size depends on both
/// the [hash algorithm](#hash-algorithm) and the input dataset. If your images are mostly
/// wide aspect ratio (landscape) then a larger width and a smaller height hash size may be
/// preferable. Optimal values can really only be discovered empirically though.
///
/// (As the author experiments, suggested values will be added here for various algorithms.)
///
/// ### Hash Algorithm
/// Setter: [`.hash_alg()`](#method.hash_alg)
/// Definition: [`HashAlg`](enum.HashAlg.html)
///
/// Multiple methods of calculating image hashes are provided in this crate under the `HashAlg`
/// enum. Each algorithm is different but they all produce the same size hashes as governed by
/// `hash_size`.
///
/// ### Hash Bytes Container / `B` Type Param
/// Use [`with_bytes_type::<B>()`](#method.with_bytes_type) instead of `new()` to customize.
///
/// This hash API allows you to specify the bytes container type for generated hashes. The default
/// allows for any arbitrary hash size (see above) but requires heap-allocation. Instead, you
/// can select an array type which allows hashes to be allocated inline, but requires consideration
/// of the possible sizes of hash you want to generate so you don't waste memory.
///
/// Another advantage of using a constant-sized hash type is that the compiler may be able to
/// produce more optimal code for generating and comparing hashes.
///
/// ```rust
/// # use img_hash::*;
///
/// // Use default container type, good for any hash size
/// let config = HasherConfig::new();
///
/// /// Inline hash container that exactly fits the default hash size
/// let config = HasherConfig::with_bytes_type::<[u8; 8]>();
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct HasherConfig<B = Box<[u8]>> {
    width: u32,
    height: u32,
    gauss_sigmas: Option<[f32; 2]>,
    #[serde(with = "SerdeFilterType")]
    resize_filter: FilterType,
    dct: bool,
    hash_alg: HashAlg,
    _bytes_type: PhantomData<B>,
}

impl HasherConfig<Box<[u8]>> {
    /// Construct a new hasher config with sane, reasonably fast defaults.
    ///
    /// A default hash container type is provided as a default type parameter which is guaranteed
    /// to fit any hash size.
    pub fn new() -> Self {
        Self::with_bytes_type()
    }

    /// Construct a new config with the selected [`HashBytes`](trait.HashBytes.html) impl.
    ///
    /// You may opt for an array type which allows inline allocation of hash data.
    ///
    /// ### Note
    /// The default hash size requires 64 bits / 8 bytes of storage. You can change this
    /// with [`.hash_size()`](#method.hash_size).
    pub fn with_bytes_type<B_: HashBytes>() -> HasherConfig<B_> {
        HasherConfig {
            width: 8,
            height: 8,
            gauss_sigmas: None,
            resize_filter: FilterType::Lanczos3,
            dct: false,
            hash_alg: HashAlg::Gradient,
            _bytes_type: PhantomData,
        }
    }
}

impl<B: HashBytes> HasherConfig<B> {
    /// Set a new hash width and height; these can be the same.
    ///
    /// The number of bits in the resulting hash will be `width * height`. If you are using
    /// a fixed-size `HashBytes` type then you must ensure it can hold at least this many bits.
    /// You can check this with [`HashBytes::max_bits()`](#method.max_bits).
    ///
    /// ### Rounding Behavior
    /// Certain hash algorithms need to round this value to function properly:
    ///
    /// * [`DoubleGradient`](enum.HashAlg.html#variant.DoubleGradient) rounds to the next multiple of 2;
    /// * [`Blockhash`](enum.HashAlg.html#variant.Blockhash) rounds to the next multiple of 4.
    ///
    /// If the chosen values already satisfy these requirements then nothing is changed.
    ///
    /// ### Recommended Values
    /// The hash granularity increases with `width * height`, although there are diminishing
    /// returns for higher values. Start small. A good starting value to try is `8, 8`.
    ///
    /// When using DCT preprocessing having `width` and `height` be the same value will improve
    /// hashing performance as only one set of coefficients needs to be used.
    pub fn hash_size(self, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..self
        }
    }

    /// Set the filter used to resize images during hashing.
    ///
    /// Note when picking a filter that images are almost always reduced in size.
    /// Has no effect with the Blockhash algorithm as it does not resize.
    pub fn resize_filter(self, resize_filter: FilterType) -> Self {
        Self {
            resize_filter,
            ..self
        }
    }

    /// Set the algorithm used to generate hashes.
    ///
    /// Each algorithm has different performance characteristics.
    pub fn hash_alg(self, hash_alg: HashAlg) -> Self {
        Self { hash_alg, ..self }
    }

    /// Enable preprocessing with the Discrete Cosine Transform (DCT).
    ///
    /// Does nothing when used with [the Blockhash.io algorithm](HashAlg::Blockhash)
    /// which does not scale the image.
    /// (RFC: it would be possible to shoehorn a DCT into the Blockhash algorithm but it's
    /// not clear what benefits, if any, that would provide).
    ///
    /// After conversion to grayscale, the image is scaled down to `width * 2 x height * 2`
    /// and then the Discrete Cosine Transform is performed on the luminance values. The DCT
    /// essentially transforms the 2D image from the spatial domain with luminance values
    /// to a 2D frequency domain where the values are amplitudes of cosine waves. The resulting
    /// 2D matrix is then cropped to the low `width * height` corner and the
    /// configured hash algorithm is performed on that.
    ///
    /// In layman's terms, this essentially converts the image into a mathematical representation
    /// of the "broad strokes" of the data, which allows the subsequent hashing step to be more
    /// robust against changes that may otherwise produce different hashes, such as significant
    /// edits to portions of the image.
    ///
    /// However, on most machines this usually adds an additional 50-100% to the average hash time.
    ///
    /// This is a very similar process to JPEG compression, although the implementation is too
    /// different for this to be optimized specifically for JPEG encoded images.
    ///
    /// Further Reading:
    /// * http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
    /// Krawetz describes a "pHash" algorithm which is equivalent to Mean + DCT preprocessing here.
    /// However there is nothing to say that DCT preprocessing cannot compose with other hash
    /// algorithms; Gradient + DCT might well perform better in some aspects.
    /// * https://en.wikipedia.org/wiki/Discrete_cosine_transform
    pub fn preproc_dct(self) -> Self {
        Self { dct: true, ..self }
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with default sigma values.
    ///
    /// Recommended only for use with [the Blockhash.io algorithm](enum.HashAlg#variant.Blockhash)
    /// as it significantly reduces entropy in the scaled down image for other algorithms.
    ///
    /// See [`Self::preproc_diff_gauss_sigmas()](#method.preproc_diff_gauss_sigmas) for more info.
    pub fn preproc_diff_gauss(self) -> Self {
        self.preproc_diff_gauss_sigmas(5.0, 10.0)
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with the given sigma values.
    ///
    /// Recommended only for use with [the Blockhash.io algorithm](enum.HashAlg#variant.Blockhash)
    /// as it significantly reduces entropy in the scaled down image for other algorithms.
    ///
    /// After the image is converted to grayscale, it is blurred with a Gaussian blur using
    /// two different sigmas, and then the images are subtracted from each other. This reduces
    /// the image to just sharp transitions in luminance, i.e. edges. Varying the sigma values
    /// changes how sharp the edges are^[citation needed].
    ///
    /// Further reading:
    /// * https://en.wikipedia.org/wiki/Difference_of_Gaussians
    /// * http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    /// (Difference of Gaussians is an approximation of a Laplacian of Gaussian filter)
    pub fn preproc_diff_gauss_sigmas(self, sigma_a: f32, sigma_b: f32) -> Self {
        Self {
            gauss_sigmas: Some([sigma_a, sigma_b]),
            ..self
        }
    }

    /// Create a [`Hasher`](struct.Hasher.html) from this config which can be used to hash images.
    ///
    /// ### Panics
    /// If the chosen hash size (`width x height`, rounded for the algorithm if necessary)
    /// is too large for the chosen container type (`B::max_bits()`).
    pub fn to_hasher(&self) -> Hasher<B> {
        let Self {
            hash_alg,
            width,
            height,
            gauss_sigmas,
            resize_filter,
            dct,
            ..
        } = *self;

        let (width, height) = hash_alg.round_hash_size(width, height);

        assert!(
            (width * height) as usize <= B::max_bits(),
            "hash size too large for container: {} x {}",
            width,
            height
        );

        // Blockhash doesn't resize the image so don't waste time calculating coefficients
        let dct_coeffs = if dct && hash_alg != HashAlg::Blockhash {
            // calculate the coefficients based on the resize dimensions
            let (dct_width, dct_height) = hash_alg.resize_dimensions(width, height);
            Some(DctCtxt::new(dct_width, dct_height))
        } else {
            None
        };

        Hasher {
            ctxt: HashCtxt {
                gauss_sigmas,
                dct_ctxt: dct_coeffs,
                width,
                height,
                resize_filter,
            },
            hash_alg,
            bytes_type: PhantomData,
        }
    }
}

// cannot be derived because of `FilterType`
impl<B> fmt::Debug for HasherConfig<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HasherConfig")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("hash_alg", &self.hash_alg)
            .field("resize_filter", &debug_filter_type(&self.resize_filter))
            .field("gauss_sigmas", &self.gauss_sigmas)
            .field("use_dct", &self.dct)
            .finish()
    }
}

impl Default for HasherConfig {
    fn default() -> Self {
        HasherConfig::new()
    }
}

/// Generates hashes for images.
///
/// Constructed via [`HasherConfig::to_hasher()`](struct.HasherConfig#method.to_hasher).
pub struct Hasher<B = Box<[u8]>> {
    ctxt: HashCtxt,
    hash_alg: HashAlg,
    bytes_type: PhantomData<B>,
}

impl<B> Hasher<B>
where
    B: HashBytes,
{
    /// Calculate a hash for the given image with the configured options.
    pub fn hash_image<I: Image>(&self, img: &I) -> ImageHash<B> {
        let hash = self.hash_alg.hash_image(&self.ctxt, img);
        ImageHash { hash }
    }
}

enum CowImage<'a, I: Image> {
    Borrowed(&'a I),
    Owned(I::Buf),
}

impl<'a, I: Image> CowImage<'a, I> {
    fn to_grayscale(&self) -> Cow<GrayImage> {
        match self {
            CowImage::Borrowed(img) => img.to_grayscale(),
            CowImage::Owned(img) => img.to_grayscale(),
        }
    }
}

enum HashVals {
    Floats(Vec<f32>),
    Bytes(Vec<u8>),
}

// TODO: implement `Debug`, needs adaptor for `FilterType`
struct HashCtxt {
    gauss_sigmas: Option<[f32; 2]>,
    dct_ctxt: Option<DctCtxt>,
    resize_filter: FilterType,
    width: u32,
    height: u32,
}

impl HashCtxt {
    /// If Difference of Gaussians preprocessing is configured, produce a new image with it applied.
    fn gauss_preproc<'a, I: Image>(&self, image: &'a I) -> CowImage<'a, I> {
        if let Some([sigma_a, sigma_b]) = self.gauss_sigmas {
            let mut blur_a = image.blur(sigma_a);
            let blur_b = image.blur(sigma_b);
            blur_a.diff_inplace(&blur_b);

            CowImage::Owned(blur_a)
        } else {
            CowImage::Borrowed(image)
        }
    }

    /// If DCT preprocessing is configured, produce a vector of floats, otherwise a vector of bytes.
    fn calc_hash_vals(&self, img: &GrayImage, width: u32, height: u32) -> HashVals {
        if let Some(ref dct_ctxt) = self.dct_ctxt {
            let img =
                imageops::resize(img, dct_ctxt.width(), dct_ctxt.height(), self.resize_filter);

            let img_vals = img.into_vec();
            let mut packed_2d: Vec<_> = img_vals.iter().copied().map(f32::from).collect();
            dct_ctxt.dct_2d(&mut packed_2d);
            HashVals::Floats(dct_ctxt.crop_2d(packed_2d))
        } else {
            let img = imageops::resize(img, width, height, self.resize_filter);
            HashVals::Bytes(img.into_vec())
        }
    }
}

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash<B = Box<[u8]>> {
    hash: B,
}

impl<B: AsRef<[u8]>> ImageHash<B> {
    /// Format this image has as a hex string.
    pub fn to_hex(&self) -> String {
        static CHARS: &[u8] = b"0123456789abcdef";

        let bytes = self.hash.as_ref();
        let mut v = Vec::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            v.push(CHARS[(byte >> 4) as usize]);
            v.push(CHARS[(byte & 0xf) as usize]);
        }

        unsafe { String::from_utf8_unchecked(v) }
    }
}

/// Error that can happen constructing a `ImageHash` from bytes.
#[derive(Debug, PartialEq)]
pub enum InvalidBytesError {
    /// Byte slice passed to `from_bytes` was the wrong length.
    BytesWrongLength {
        /// Number of bytes the `ImageHash` type expected.
        expected: usize,
        /// Number of bytes found when parsing the hash bytes.
        found: usize,
    },
    /// String passed was not valid base64.
    Base64(base64::DecodeError),
}

impl<B: HashBytes> ImageHash<B> {
    /// Get the bytes of this hash.
    pub fn as_bytes(&self) -> &[u8] {
        self.hash.as_slice()
    }

    /// Create an `ImageHash` instance from the given bytes.
    ///
    /// ## Errors:
    /// Returns a `InvalidBytesError::BytesWrongLength` error if the slice passed can't fit in `B`.
    pub fn from_bytes(bytes: &[u8]) -> Result<ImageHash<B>, InvalidBytesError> {
        if bytes.len() * 8 > B::max_bits() {
            return Err(InvalidBytesError::BytesWrongLength {
                expected: B::max_bits() / 8,
                found: bytes.len(),
            });
        }

        Ok(ImageHash {
            hash: B::from_iter(bytes.iter().copied()),
        })
    }

    /// Calculate the Hamming distance between this and `other`.
    ///
    /// Equivalent to counting the 1-bits of the XOR of the two hashes.
    ///
    /// Essential to determining the perceived difference between `self` and `other`.
    ///
    /// ### Note
    /// This return value is meaningless if these two hashes are from different hash sizes or
    /// algorithms.
    pub fn dist(&self, other: &Self) -> u32 {
        BitSet::hamming(&self.hash, &other.hash)
    }

    /// Create an `ImageHash` instance from the given Base64-encoded string.
    ///
    /// ## Errors:
    /// Returns `InvaidBytesError::Base64(DecodeError::InvalidLength)` if the string wasn't valid base64`.
    /// Otherwise returns the same errors as `from_bytes`.
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash<B>, InvalidBytesError> {
        let bytes = base64::decode(encoded_hash).map_err(InvalidBytesError::Base64)?;

        Self::from_bytes(&bytes)
    }

    /// Get a Base64 string representing the bits of this hash.
    ///
    /// Mostly for printing convenience.
    pub fn to_base64(&self) -> String {
        base64::encode(self.hash.as_slice())
    }
}

/// Provide Serde a typedef for `image::FilterType`: https://serde.rs/remote-derive.html
/// This is automatically checked, if Serde complains then double-check with the original definition
#[derive(Serialize, Deserialize)]
#[serde(remote = "FilterType")]
enum SerdeFilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

fn debug_filter_type(ft: &FilterType) -> &'static str {
    match ft {
        FilterType::Triangle => "Triangle",
        FilterType::Nearest => "Nearest",
        FilterType::CatmullRom => "CatmullRom",
        FilterType::Lanczos3 => "Lanczos3",
        FilterType::Gaussian => "Gaussian",
    }
}
