pub fn dct_hash<I: HashImage>(image: &I, hash_size: u32) -> BitVec {
    match hash_size {
        HSIZE => dct_hash_HSIZE(image),
        _ => dct_hash_dyn(image, hash_size),
    }
}
