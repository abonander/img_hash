use proc_macro::TokenStream;

use syn::Ident;

use quote::Tokens;

use std::f64::consts::*;

use super::IntoIterCloner;

pub fn derive_dct_hash(name: Ident) -> TokenStream {

    let dct_fn = gen_dct_fn();

    quote! {
        impl DctHash for #name {
            #dct_fn
        }
    }
}

fn gen_dct_fn() -> Tokens {
    let headers = ::get_precomp_sizes().into_iter().map(gen_branch).collect::<Vec<_>>();
    let impls = headers.iter().map(gen_dct_impl).collect::<Vec<_>>();

    let branches = headers.into_iter().map(|header| {
        let size = header.size;
        let branch_tokens = header.branch_tokens;

        quote! {
            #size => #branch_tokens
        }
    });

    quote! {
        fn dct_hash<I: HashImage>(img: &I, size: u32) -> Bitv {
            match size {
                #(#branches),*
                _ => dct_hash_dyn(img, size),
            }
        }
    }
}

struct DctFnHeader {
    size: usize,
    name: String,
    branch_tokens:
}

fn gen_branch(size: usize) -> DctFnHeader {
    let name = format!("dct_hash_{}", size);
    let tokens = {
        let ref name = name;

        quote! {
            #size => #name(image)
        }
    };
}

fn gen_dct_impl(header: &DctFnHeader)

fn gen_impl


fn gen_dct_1d_fn(size: usize) -> Tokens {
    let vals = (0 .. size).map(|idx| gen_dct_1d_val(idx, size));

    quote! {
        fn dct_1d(input: &[f64; #size]) -> [f64; #size] {
            [#(#vals),*]
        }
    }
}

fn gen_dct_1d_val(i: usize, size: usize) -> Tokens {
    let mut tokens = quote!{};

    for j in 0 ..size {
        let multiplier = (PI * i as f64 * (2 * j + 1) as f64
                            / (2 * size) as f64).cos();

        tokens = quote! {
            #tokens + input[j] * #multiplier
        };
    }

    if i == 0 {
        quote! {
            #tokens / 2.0
        }
    } else {
        tokens
    }
}
