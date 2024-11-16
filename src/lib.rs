use std::{fs::File, str::FromStr};
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
    pub offset: u32,
    pub rows: u32,
    pub cols: u32,
    pub row_strides: u32,
    pub col_strides: u32,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WsglArgs {
    pub data_info: WsglArray2Info,
    pub knn_info: WsglArray2Info,
    pub candidates: u32,
}

pub fn cli_npy(idx: usize) -> (WsglArray2Info, Vec<f32>) {
    let path = std::env::args().skip(idx).next().expect(
        "missing cli argument for npy file");
    WsglArray2Info::new(raw_npy(path))
}

pub fn cli() -> (WsglArgs, Vec<f32>, Vec<u32>) {
    let (data_info, data_input) = cli_npy(1);
    let numbers_input: Vec<usize> = std::env::args().skip(2).take(2).map(
        |s| usize::from_str(&s).expect("missing args")).collect();
    let (k, candidates) = (numbers_input[0], numbers_input[1]);
    let knn_init = RawArray2::new(
        Array2::<u32>::zeros((data_info.rows as usize, k)));
    let (knn_info, knn_input) = WsglArray2Info::new(knn_init);
    (WsglArgs {
        data_info: data_info,
        knn_info: knn_info,
        candidates: candidates as u32,
    }, data_input, knn_input)
}

fn raw_npy(path: String) -> RawArray2<f32> {
    let reader = File::open(path).expect("IO error");
    let arr = Array2::<f32>::read_npy(reader).expect("npy format error");
    RawArray2::new(arr)
}

impl<T> RawArray2<T> {
    fn new(arr: Array2<T>) -> Self {
        let shape = <Vec<usize> as TryInto<[usize; 2]>>::try_into(
            arr.shape().to_owned()).unwrap();
        let strides = <Vec<isize> as TryInto<[isize; 2]>>::try_into(
            arr.strides().to_owned()).unwrap();
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
}

impl WsglArray2Info {
    fn new<T>(raw: RawArray2<T>) -> (Self, Vec<T>) {
        (WsglArray2Info {
            offset: raw.offset,
            rows: raw.shape[0],
            cols: raw.shape[1],
            row_strides: raw.strides[0],
            col_strides: raw.strides[1],
        }, raw.data)
    }
}
