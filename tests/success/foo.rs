use endiannezz::{Io, ext::{EndianReader, EndianWriter}, BigEndian};
use std::io::Write;

#[derive(Io, Copy, Clone, Debug)]
#[endian(little)]
struct Foo<const N: usize> {
    basic: [u64; N],
}

impl<const N: usize> Foo<N> {
    pub fn new(value: u64) -> Self {
        Self {
            basic: [value; N],
        }
    }
}


#[derive(Io, Copy, Clone, Debug)]
#[endian(big)]
struct Bar<const N: usize>  {
    value: u16,
    foo: [Foo<N>; 2]
}

impl<const N: usize> Bar<N> {
    pub fn new(value: u16) -> Self {
        Self {
            value,
            foo: Foo::new(value as u64)
        }
    }
}


fn generic_const() {
    let foo = Foo::<2_usize>::new(0xdead);
    let mut vec = Vec::<u8>::new();
    foo.write(&mut vec).unwrap();
    eprintln!("{:?}", vec);
    let bar = Bar::<4_usize>::new(0xA);
    eprintln!("{:?}", bar);
}

fn main() {
    generic_const();
}
