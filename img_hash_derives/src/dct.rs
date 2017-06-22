use proc_macro::TokenStream;

use syn::Ident;

use quote::Tokens;

use std::f32::consts::*;

use super::IntoIterCloner;

const SIZE_MULT: usize = 2;

pub fn derive_dct_hash(name: Ident) -> TokenStream {
    let dct_fn = gen_dct_fn();

    quote! (
        impl DctHash for #name {
            #dct_fn
        }
    ).parse().unwrap()
}

fn gen_dct_fn() -> Tokens {
    let headers = ::get_precomp_sizes().into_iter().map(gen_branch).collect::<Vec<_>>();
    let impls = headers.iter().map(gen_dct_hash_fn);

    let branches = headers.iter().map(|header| {
        let size = header.size;
        let branch_tokens = &header.branch_tokens;

        quote! {
            #size => #branch_tokens
        }
    });

    quote! {
        fn dct_hash<I: HashImage>(img: &I, size: u32) -> BitVec {
            #(#impls)*

            match size as usize {
                #(#branches,)*
                _ => dct_hash_dyn(img, size),
            }

        }
    }
}

struct DctFnHeader {
    size: usize,
    name: Ident,
    branch_tokens: Tokens,
}

fn gen_branch(size: usize) -> DctFnHeader {
    let name = Ident::new(format!("dct_hash_{}", size));
    let tokens = {
        let ref name = name;

        quote! {
            #name(img)
        }
    };

    DctFnHeader {
        size: size,
        name: name,
        branch_tokens: tokens,
    }
}

fn gen_dct_hash_fn(header: &DctFnHeader) -> Tokens {
    let ref name = header.name;
    let size = header.size;
    let large = size * SIZE_MULT;

    let dct2d = gen_dct_2d_fn(large);
    let mean_fn = gen_mean_fn(size);

    quote! {
        fn #name<I: HashImage>(img: &I) -> BitVec {
            #dct2d
            #mean_fn

            let mut hash_values = [[0f32; #large]; #large];

            let prepared = prepare_image(img, #large as u32, #large as u32);

            for (val, out) in prepared.into_iter().zip(hash_values.iter_mut().flat_map(|row| row)) {
                *out = val as f32 * (1.0 / 255.0);
            }

            let dct = dct_2d(&hash_values);

            let mean = mean_lower(&dct);

            dct.iter().take(#size).flat_map(|row| row.iter().take(#size))
                .map(|&x| x >= mean).collect()
        }
    }
}


fn gen_dct_2d_fn(size: usize) -> Tokens {
    let rows = (0 .. size).map(|row|
        quote! {
            dct_row(&input[#row])
        }
    );

    let dct_row = gen_dct_rows_fn(size);
    let dct_cols = gen_dct_cols_fn(size);

    quote! {
        fn dct_2d(input: &[[f32; #size]; #size]) -> [[f32; #size]; #size] {
            #dct_row
            #dct_cols

            let rows = [#(#rows),*];
            dct_cols(&rows)
        }
    }
}

fn gen_dct_rows_fn(size: usize) -> Tokens {
    let vals = (0 .. size).map(|idx| gen_dct_1d_val(idx, size));

    quote! {
        #[inline(always)]
        fn dct_row(input: &[f32; #size]) -> [f32; #size] {
            [#(#vals),*]
        }
    }
}

fn gen_dct_1d_val(i: usize, size: usize) -> Tokens {
    let terms = (0 .. size).map(|j| {
        let multiplier = dct_multiplier(i, j, size);

        quote! {
            input[#j] * #multiplier
        }
    });

    let mut tokens = quote!{ #(#terms)+* };

    if i == 0 {
        let inv_sqrt2 = 1.0 / SQRT_2;

        tokens = quote! {
            #tokens * #inv_sqrt2
        };
    }

    quote! {
        #tokens / 2.0
    }
}

fn gen_dct_cols_fn(size: usize) -> Tokens {
    let rows = (0 .. size).map(move |i| {
        let terms = (0 .. size).map(move |col| gen_dct_1d_col_val(col, i, size));
        quote!{
            [#(#terms),*]
        }
    });

    quote! {
        #[inline(always)]
        fn dct_cols(input: &[[f32; #size]; #size]) -> [[f32; #size]; #size] {
            [#(#rows),*]
        }
    }
}

fn gen_dct_1d_col_val(col: usize, i: usize, size: usize) -> Tokens {
    let terms = (0 .. size).map(|j|  {
        let multiplier = dct_multiplier(i, j, size);

        quote! {
            input[#j][#col] * #multiplier
        }
    });

    let mut tokens = quote!{ #(#terms)+* };

    if i == 0 {
        let inv_sqrt2 = 1.0 / SQRT_2;

        tokens = quote! {
            #tokens * #inv_sqrt2
        };
    }

    quote! {
        #tokens / 2.0
    }
}

fn dct_multiplier(i: usize, j: usize, size: usize) -> f32 {
    (PI * i as f32 * (2 * j + 1) as f32 / (2 * size) as f32).cos()
}

fn gen_mean_fn(size: usize) -> Tokens {
    let len = size * size;
    let large = size * SIZE_MULT;

    let rows = (0 .. size).map(|row| {
        let row = (0 .. size).map(|i| quote!{ input[#row][#i] });
        quote!{ #(#row)+* }
    });

    quote! {
        fn mean_lower(input: &[[f32; #large]; #large]) -> f32 {
            (#(#rows)+*) / #len as f32
        }
    }
}
