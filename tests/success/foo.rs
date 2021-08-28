use endiannezz::Io;

#[derive(Io, Copy, Clone, Debug, PartialEq)]
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


#[derive(Io, Copy, Clone, Debug, PartialEq)]
#[endian(big)]
struct Bar<const N: usize>  {
    value: u16,
    foo: Foo<N>,
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
    assert_eq!(vec, [173, 222, 0, 0, 0, 0, 0, 0, 173, 222, 0, 0, 0, 0, 0, 0]);
    let bar = Bar::<4_usize>::new(0xA);
    assert_eq!(bar, Bar { value: 10, foo: Foo { basic: [10, 10, 10, 10] } });
}

fn main() {
    generic_const();
}
