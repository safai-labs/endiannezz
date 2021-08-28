use endiannezz::{Io, ext::{EndianReader, EndianWriter}, BigEndian};
use std::io::Write;

#[derive(Io, Copy, Clone, Debug)]
#[endian(little)]
struct Foo {
    foo: [u64; 2]
}

impl Foo {
    pub fn new(value: u64) -> Self { 
        Self {
            foo: [0xdead, 0xbeef]
        }
    }
}


#[test]
fn generic_const() {
    let foo: Foo = Foo::new(0x10);
    let mut vec = Vec::<u8>::new();
    foo.write(&mut vec).unwrap();
    eprintln!("{:?}", vec);
}
