use crate::{Endian, Io, Primitive};
use std::{io::{Read, Result, Write}};

pub trait HackedPrimitive: Primitive {
    #[cfg_attr(feature = "inline_primitives", inline)]
    fn write_hacked<E: Endian, W: Write>(self, w: W) -> Result<()> {
        E::write(self, w)
    }

    #[cfg_attr(feature = "inline_primitives", inline)]
    fn read_hacked<E: Endian, R: Read>(r: R) -> Result<Self> {
        E::read(r)
    }
}

impl<T: Primitive> HackedPrimitive for T {}
impl<T: HackedPrimitive, const N: usize> HackedPrimitive for [T; N]
where T: Default + Copy + Clone + Sized,
[T]: Sized, <T as Primitive>::Buf: Copy,
Vec<T>: std::iter::FromIterator<<T as Primitive>::Buf>,
[T; N]: HackedPrimitive,
{
}

pub trait HackedIo: Io {
    #[cfg_attr(feature = "inline_io", inline(always))]
    fn write_hacked<E: Endian, W: Write>(&self, w: W) -> Result<()> {
        Io::write(self, w)
    }

    #[cfg_attr(feature = "inline_io", inline(always))]
    fn read_hacked<E: Endian, R: Read>(r: R) -> Result<Self> {
        Io::read(r)
    }
}

// impl <T, const N: usize> HackedIo for [T; N]
// where   T:      Copy + Default + Primitive + Sized,
//         <T as Primitive>::Buf: InternalDefault + Clone + Copy,
//        Vec<T>: From<<T as Primitive>::Buf>,

// {
//     fn write_hacked<E: Endian, W: Write>(&self, w: W) -> Result<()> {
//         E::write(*self, w)
//     }

//     fn read_hacked<E: Endian, R: Read>(r: R) -> Result<Self> {
//         Ok([T::default(); N])
//     }
// }
