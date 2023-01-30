/*!
This crate provides the ability to encode and decode all primitive types into [different endianness]

# How it works?
Crate automatically implements [`Primitive`] trait for each primitive type.

This allows to write abstractions and call the appropriate method depending on
the byte order that you passed to the function template. [`Endian`] it's something like a proxy to do it.

Macros create implementations for I/O endianness:
[`NativeEndian`], [`LittleEndian`] and [`BigEndian`]

All these types are enums, which means that you cannot create them, only pass to the template.

Now it's possible to have traits that expand [`Read`] and [`Write`] with new methods.

# Simple example
```rust
use endiannezz::ext::{EndianReader, EndianWriter};
use endiannezz::{BigEndian, LittleEndian, NativeEndian};
use std::io::Result;

fn main() -> Result<()> {
    let mut vec = Vec::new();

    vec.try_write::<LittleEndian, i32>(1)?;
    vec.try_write::<BigEndian, _>(2)?;
    vec.try_write::<NativeEndian, _>(3_u16)?;

    let mut slice = vec.as_slice();

    slice.try_read::<LittleEndian, i32>()?;
    let _num32: i32 = slice.try_read::<BigEndian, _>()?;
    let _num16: u16 = slice.try_read::<NativeEndian, _>()?;

    Ok(())
}
```

You can also use this syntax:
```rust
use endiannezz::{BigEndian, Endian, LittleEndian};
use std::io::Result;

fn main() -> Result<()> {
    let mut vec = Vec::new();
    BigEndian::write(1, &mut vec)?;
    LittleEndian::write::<u16, _>(2, &mut vec)?;
    assert_eq!(vec.as_slice(), &[0, 0, 0, 1, 2, 0]);

    Ok(())
}
```

# Using `#[derive(Io)]` to describe complex binary formats
```rust
use endiannezz::ext::{EndianReader, EndianWriter};
use endiannezz::{Io, LittleEndian};
use std::io::{Read, Result, Write};

struct Bytes(Vec<u8>);

//Custom implementation of read and write
//Use it for complex types, which can be built from primitives
impl Io for Bytes {
    fn write<W: Write>(&self, mut w: W) -> Result<()> {
        w.try_write::<LittleEndian, u32>(self.0.len() as u32)?;
        w.write_all(self.0.as_slice())?;
        Ok(())
    }

    fn read<R: Read>(mut r: R) -> Result<Self> {
        let capacity = r.try_read::<LittleEndian, u32>()? as usize;
        let mut vec = vec![0; capacity];
        r.read_exact(&mut vec)?;
        Ok(Self(vec))
    }
}

#[derive(Io)]
//default endian for fields of struct (except custom impl, such as Bytes)
#[endian(little)]
//There are 3 types of endianness and they can be written in the `#[endian]` attribute as follows:
// - NativeEndian: `_`, `ne`, `native`
// - LittleEndian: `le`, `little`
// - BigEndian: `be`, `big`
// - Skip: `skip`, `ignore`
struct Message {
    //will read/write data as is (according to implementation)
    bytes: Bytes,

    //u16 in little-endian
    distance: u16,

    //f32 in big-endian, you can override default endian!
    #[endian(big)]
    delta: f32,

    //will be skipped while writing, and filled with Default::default()
    //upon reading.
    #[endian(skip)]
    skipped_field: u16,

    //machine byte order
    #[endian(native)]
    machine_data: u32,
}

fn main() -> Result<()> {
    let message = Message {
        bytes: Bytes(vec![0xde, 0xad, 0xbe, 0xef]),
        distance: 5,
        delta: 2.41,
        machine_data: 41,
        skipped_field: 15, // this value is skipped
    };

    //writing message into Vec
    let mut vec = Vec::new();
    message.write(&mut vec)?;

    //explain
    let mut excepted = vec![
        4, 0, 0, 0, //bytes len in little-endian
        0xde, 0xad, 0xbe, 0xef, //buffer
        5, 0, //distance in little-endian
        0x40, 0x1a, 0x3d, 0x71, //delta in big-endian
    ];

    if cfg!(target_endian = "little") {
        excepted.extend(&[41, 0, 0, 0]); //machine_data on little-endian CPUs
    } else {
        excepted.extend(&[0, 0, 0, 41]); //machine_data on big-endian CPUs
    }

    assert_eq!(vec, excepted);

    //reading message from slice
    let mut slice = vec.as_slice();
    let _message1 = Message::read(&mut slice)?;

    Ok(())
}
```

[different endianness]: https://en.wikipedia.org/wiki/Endianness
[`Primitive`]: trait.Primitive.html
[`Endian`]: trait.Endian.html
[`NativeEndian`]: enum.NativeEndian.html
[`LittleEndian`]: enum.LittleEndian.html
[`BigEndian`]: enum.BigEndian.html
[`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
[`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
*/

use crate::private::*;
use std::convert::TryInto;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::mem;

#[cfg(feature = "derive")]
#[doc(hidden)]
pub use endiannezz_derive::*;

use crate::ext::{EndianReader, EndianWriter};

/// Internal module to simplify `proc_macro` implementation
///
/// The main goal is to be able to call `write` method on `
// Io` xor `Primitive` and get clean error on compile-time
#[cfg(feature = "derive")]
pub mod internal;

/// Provides extensions for [`Read`] and [`Write`] traits
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
pub mod ext;

/// This trait is implemented for all primitive types that exist in rust,
/// and allows to read types from bytes or write them into bytes
//noinspection RsSelfConvention
pub trait Primitive: Sized + Copy {
    type Buf: private::InternalAsRef<[u8]> + private::InternalAsMut<[u8]> + private::InternalDefault;

    fn to_ne_bytes(self) -> Self::Buf;
    fn to_le_bytes(self) -> Self::Buf;
    fn to_be_bytes(self) -> Self::Buf;

    fn from_ne_bytes(bytes: Self::Buf) -> Self;
    fn from_le_bytes(bytes: Self::Buf) -> Self;
    fn from_be_bytes(bytes: Self::Buf) -> Self;
}

macro_rules! delegate {
    (@simple $ty:ty, [$($method:ident),* $(,)?], ($param:ident : $param_ty:ty) -> $ret:ty) => {
        delegate!(@inner $ty, [$($method),*], $param, $param_ty, $ret);
    };
    (@to_array [$arr_ty:ty; $count:ident], [$($method:ident),* $(,)?], ($param:ident : $param_ty: ty) -> $ret: ty) => {
        $(
            #[inline]
            fn $method(self) -> $ret {
                const T_SIZE: usize =  mem::size_of::<$arr_ty>();
                let rv: Vec<[u8; T_SIZE]> =
                    self.iter().map(|x| {
                        let x = <$arr_ty as Primitive>::$method(*x);
                        x
                    }).collect::<Vec<[u8; T_SIZE]>>();
                //
                // NOTE: While panics and asserts have no place in a library,
                // here this choice is appropriate,
                // place because it's not a handle-able "normal error".
                //
                // This assert should never trigger. If it does
                // something is fundamentally wrong with the library
                // implementation
                //
                let rv: [[u8; T_SIZE]; $count] = rv.try_into()
                    .unwrap_or_else(|_| panic!("BUG()! inside endiannezz library, something went terribly wrong with type casting or something else."));
                // assert_eq!(rv.len(), $count >> 2);
                rv
            }
        )*
    };
    (@from_array [$arr_ty:ty; $count:ident], [$($method:ident),* $(,)?], ($param:ident : $param_ty: ty) -> $ret: ty) => {
        $(
            #[inline]
            fn $method($param: $param_ty) -> $ret {
                let mut x: [$arr_ty; $count]
                    = [0 as $arr_ty; $count];
                assert_eq!($count, $param.len());
                for c in 0..$count {
                    x[c] = <$arr_ty>::$method(
                        $param[c..(c+1)][0])
                }
                x
            }
        )*
    };
    (@inner $ty:ty, [$($method:ident),*], $param:ident, $param_ty:ty, $ret:ty) => {
        $(
            #[inline]
            fn $method ($param: $param_ty) -> $ret { <$ty>::$method($param) }
        )*
    };
}

macro_rules! impl_primitives {
    ($($ty:ty),* $(,)?) => {
        $(

            impl<const SO: usize> private::InternalDefault for [$ty; SO] {
                fn default() -> Self {
                    [0 as $ty; SO]
                }
            }

            impl <const SO: usize> private::InternalAsMut<[u8]> for
                [$ty; SO]
            {
                fn as_mut<'a> (&'a mut self) -> &'a mut [u8] {
                    unsafe {
                        ::core::slice::from_raw_parts_mut(
                            self.as_mut_ptr() as *mut u8, SO)
                    }
                }
            }

            impl <const SO: usize> private::InternalAsRef<[u8]> for
                [$ty; SO]
            {
                fn as_ref(&self) -> &[u8] {
                    use ::core::slice::from_raw_parts;
                    // Safety: size is always known at compile time
                    // because these are only implemented for numerics
                    unsafe { from_raw_parts(self.as_ptr() as *const u8, SO) }
                }
            }

            impl<const SO: usize, const N: usize> private::InternalDefault for
                [[$ty; SO]; N]
            {
                fn default() -> Self {
                    [[0 as $ty; SO]; N]
                }
            }

            impl <const SO: usize, const N: usize>
                private::InternalAsMut<[u8]>
            for
                [[$ty; SO]; N]
            {
                fn as_mut<'a> (&'a mut self) -> &'a mut [u8] {
                    unsafe {
                        ::core::slice::from_raw_parts_mut(
                            self.as_mut_ptr() as *mut u8, SO*N)
                    }
                }
            }

            impl <const N: usize, const SO: usize>
                private::InternalAsRef<[u8]>
            for [[$ty; SO]; N]
            {
                fn as_ref(&self) -> &[u8] {
                    use ::core::slice::from_raw_parts;
                    unsafe { from_raw_parts(self.as_ptr() as *const u8, N*SO) }
                }
            }


            impl<const N: usize> Primitive for [$ty; N] {
                type Buf = [[u8; mem::size_of::<$ty>()]; N];

                delegate!(@to_array [$ty; N], [
                    to_be_bytes,
                    to_le_bytes,
                    to_ne_bytes,
                ], (self: Self) -> Self::Buf);
                delegate!(@from_array [$ty; N], [
                    from_be_bytes,
                    from_le_bytes,
                    from_ne_bytes,
                ], (bytes: Self::Buf) -> Self);
            }

            impl Primitive for $ty {
                type Buf = [u8; mem::size_of::<$ty>()];
                delegate!(@simple $ty, [
                    to_ne_bytes,
                    to_le_bytes,
                    to_be_bytes,
                ], (self: Self) -> Self::Buf);

                delegate!(@simple $ty, [
                    from_ne_bytes,
                    from_le_bytes,
                    from_be_bytes,
                ], (bytes: Self::Buf) -> Self);
            }

        )*
    };
}

#[rustfmt::skip]
impl_primitives![
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
    f32, f64,
];

/// Proxy for reading and writing primitive types
pub trait Endian {
    fn write<T: Primitive, W: Write>(primitive: T, w: W) -> Result<()>;
    fn read<T: Primitive, R: Read>(r: R) -> Result<T>;
}

macro_rules! impl_endianness {
    ($($endian:ident $write:ident $read:ident,)*) => {
        $(
            pub enum $endian {}

            impl Endian for $endian {
                #[inline]
                fn write<T: Primitive, W: Write>(primitive: T, mut w: W) -> Result<()> {
                    w.write_all(primitive.$write().as_ref())
                }

                #[inline]
                fn read<T: Primitive, R: Read>(mut r: R) -> Result<T> {
                    let mut buf = T::Buf::default();
                    r.read_exact(&mut buf.as_mut())?;
                    Ok(T::$read(buf))
                }
            }
        )*
    };
}

impl_endianness![
    NativeEndian to_ne_bytes from_ne_bytes,
    LittleEndian to_le_bytes from_le_bytes,
    BigEndian    to_be_bytes from_be_bytes,
];

/// Allows the type to be encoded/decoded using binary format
pub trait Io: Sized {
    fn write<W: Write>(&self, w: W) -> Result<()>;

    fn read<R: Read>(r: R) -> Result<Self>;
}

/// Binary representation of a bool
impl Io for bool {
    #[cfg_attr(feature = "inline_primitives", inline)]
    fn write<W: Write>(&self, mut w: W) -> Result<()> {
        w.try_write::<NativeEndian, u8>(if *self { 1 } else { 0 })
    }

    #[cfg_attr(feature = "inline_primitives", inline)]
    fn read<R: Read>(mut r: R) -> Result<Self> {
        let byte = r.try_read::<NativeEndian, u8>()?;
        #[cfg(feature = "unchecked_bool")]
        {
            Ok(byte != 0)
        }

        #[cfg(not(feature = "unchecked_bool"))]
        match byte {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::from(ErrorKind::InvalidData)),
        }
    }
}

pub trait HardcodedPayload: Default {
    type Buf: AsRef<[u8]> + AsMut<[u8]> + Default + PartialEq;
    const PAYLOAD: Self::Buf;
}

impl<T: HardcodedPayload> Io for T {
    #[cfg_attr(feature = "inline_primitives", inline)]
    fn write<W: Write>(&self, mut w: W) -> Result<()> {
        w.write_all(<T as HardcodedPayload>::PAYLOAD.as_ref())
    }

    #[cfg_attr(feature = "inline_primitives", inline)]
    fn read<R: Read>(mut r: R) -> Result<Self> {
        let mut payload = T::Buf::default();

        r.read_exact(payload.as_mut())?;
        if payload == <T as HardcodedPayload>::PAYLOAD {
            Ok(Self::default())
        } else {
            Err(Error::from(ErrorKind::InvalidData))
        }
    }
}

mod private {

    /// Really an internal
    // Should we seal this trait?
    pub trait InternalDefault {
        fn default() -> Self;
    }

    pub trait InternalAsMut<T>
    where
        T: ?Sized, {
        fn as_mut(&mut self) -> &mut T;
    }

    pub trait InternalAsRef<T>
    where
        T: ?Sized, {
        fn as_ref(&self) -> &T;
    }
}
