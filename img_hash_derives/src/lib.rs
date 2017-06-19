#![recursion_limit="128"]

extern crate syn;

#[macro_use]
extern crate quote;

extern crate proc_macro;

use proc_macro::TokenStream;

use std::env;

mod dct;

#[proc_macro_derive(DctHash)]
pub fn derive_dct_hash(ts: TokenStream) -> TokenStream {
    let input = syn::parse_derive_input(&ts.to_string()).unwrap();
    dct::derive_dct_hash(input.ident)
}

struct IntoIterCloner<I>(I);

impl<'a, I: IntoIterator + Clone + 'a> IntoIterator for &'a IntoIterCloner<I> {
    type Item = I::Item;
    type IntoIter = I::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.clone().into_iter()
    }
}

const KEY_SIZES: &'static str = "HASH_SIZES";

fn get_precomp_sizes() -> Vec<usize> {
    let hash_sizes_string = env::var(KEY_SIZES).unwrap_or(String::new());

    if hash_sizes_string.is_empty() { return vec![] }

    hash_sizes_string.split(',')
        .map(|s| s.trim().parse().expect("`HASH_SIZES` env var must be comma-separated list of integers"))
        .collect()
}
