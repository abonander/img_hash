img_hash [![Build Status](https://travis-ci.org/cybergeek94/img_hash.svg?branch=master)](https://travis-ci.org/cybergeek94/img_hash)
========

A library for getting perceptual hash values of images.

Thanks to Dr. Neal Krawetz for the outlines of the Average-Mean and DCT-Mean perceptual hash algorithms:  
http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html (Accessed August 2014)

Thanks to Emil Mikulic for the 2D Discrete Cosine Transform implementation in C, ported to Rust in `src/dct.rs`:  
http://unix4lyfe.org/dct/ (Implementation: http://unix4lyfe.org/dct/listing2.c) (Accessed August 2014)

Unfortunately, the AAN algorithm that provides `O(n log n)` performance didn't seem to be viable for arbitrary-length input vectors without massive code duplicaton. This shouldn't be much of a concern as the program is largely I/O bound and the actual time spent hashing will be insignificant compared to the run time of the program as a whole.

Importing
=====

Add `img_hash` to your `Cargo.toml`:

    [dependencies.img_hash]
    git = "https://github.com/cybergeek94/img_hash"
    
