use std::fs::File;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use bytemuck_derive::{Pod, Zeroable};

#[derive(Debug)]
struct RawArray2<T> {
    offset: u32,
    shape: [u32; 2],
    strides: [u32; 2],
    data: Vec<T>,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WsglArray2Info {
    offset: u32,
    shape: [u32; 2],
    strides: [u32; 2],
}

pub fn cli_npy(idx: usize) -> (WsglArray2Info, Vec<f32>) {
    let path = std::env::args().skip(idx).next().expect("missing cli argument for npy file");
    WsglArray2Info::new(raw_npy(path))
}

fn raw_npy(path: String) -> RawArray2<f32> {
    let reader = File::open(path).expect("IO error");
    let arr = Array2::<f32>::read_npy(reader).expect("npy format error");
    let shape = <Vec<usize> as TryInto<[usize; 2]>>::try_into(arr.shape().to_owned()).unwrap();
    let strides = <Vec<isize> as TryInto<[isize; 2]>>::try_into(arr.strides().to_owned()).unwrap();
    let (v, offset) = arr.into_raw_vec_and_offset();
    RawArray2 {
        offset: match offset {
            None => 0,
            _ => offset.unwrap().try_into().unwrap(),
        },
        shape: shape.map(|x| x as u32),
        strides: strides.map(|x| x as u32),
        data: v
    }
}

impl WsglArray2Info {
    fn new<T>(raw: RawArray2<T>) -> (Self, Vec<T>) {
        (WsglArray2Info {
            offset: raw.offset,
            shape: raw.shape,
            strides: raw.strides,
        }, raw.data)
    }
}
