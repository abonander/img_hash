extern crate interpolation;
extern crate image;
extern crate img_hash;

use img_hash::demo::DemoCtxt;

use std::error::Error;
use std::env;
use std::fmt;
use std::fs;

fn main() -> Result<(), String> {
    let ctxt = DemoCtxt::init("demo_gradient", "HashAlg::Gradient")?;

    Ok(())
}
