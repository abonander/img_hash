extern crate image;
extern crate img_hash;

use img_hash::demo::DemoCtxt;

use std::error::Error;
use std::env;
use std::fmt;
use std::fs;

const FRAMES: u32 = 60;

fn main() -> Result<(), String> {
    let ctxt = DemoCtxt::init("demo_gradient", "HashAlg::Gradient")?;

    println!("generating Gradient hash demo");
    let grayscale_anim = ctxt.animate_grayscale(&ctxt.image, FRAMES, |f| {
        print!("\rgenerating grayscale animation: {}/{}", f, FRAMES);
    });

    println!();

    let resize_anim = ctxt.animate_resize(grayscale_anim.last().unwrap().buffer(), 60, |f| {
        print!("\rgenerating resize animation: {}/{}", f, FRAMES);
    });

    println!("\nsaving files");

    ctxt.save_gif("grayscale", grayscale_anim)?;
    ctxt.save_gif("resize", resize_anim)?;

    Ok(())
}
