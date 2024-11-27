use std::{fs::File, str::FromStr};
use ndarray::{Array2, Array3};
use ndarray_npy::ReadNpyExt;
use bytemuck_derive::{Pod, Zeroable};

#[derive(Debug)]
struct RawArray2<T> {
    offset: u32,
    shape: [u32; 2],
    strides: [u32; 2],
    data: Vec<T>,
}

#[derive(Debug)]
struct RawArray3<T> {
    offset: u32,
    shape: [u32; 3],
    strides: [u32; 3],
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
pub struct WsglArray3Info {
    pub offset: u32,
    pub rows: u32,
    pub cols: u32,
    pub voxs: u32,
    pub row_strides: u32,
    pub col_strides: u32,
    pub vox_strides: u32,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WsglArgs {
    pub data_info: WsglArray2Info,
    pub knn_info: WsglArray2Info,
    pub distances_info: WsglArray2Info,
    pub scratch_info: WsglArray2Info,
    pub avl_info: WsglArray3Info,
    pub meta_info: WsglArray2Info,
    pub candidates: u32,
}

pub struct WsglBuffers {
    pub data: Vec<f32>,
    pub knn: Vec<i32>,
    pub distances: Vec<f32>,
    pub scratch: Vec<i32>,
    pub avl: Vec<i32>,
    pub meta: Vec<i32>,
}

pub struct WsglSlices<'a> {
    pub scratch: &'a [i32],
    pub avl: &'a [i32],
    pub meta: &'a [i32],
}

pub fn cli_npy(idx: usize) -> (WsglArray2Info, Vec<f32>) {
    let path = std::env::args().skip(idx).next().expect(
        "missing cli argument for npy file");
    WsglArray2Info::new(raw_npy(path))
}

pub fn cli() -> (WsglArgs, WsglBuffers) {
    let (data_info, data_input) = cli_npy(1);
    let numbers_input: Vec<usize> = std::env::args().skip(2).take(2).map(
        |s| usize::from_str(&s).expect("missing args")).collect();
    let (k, candidates) = (numbers_input[0], numbers_input[1]);
    let knn_init = RawArray2::new(
        -Array2::<i32>::ones((data_info.rows as usize, k)));
    let distances_init = RawArray2::new(
        Array2::<f32>::from_elem((data_info.rows as usize, k), f32::INFINITY));
    let scratch_init = RawArray2::new(
        -Array2::<i32>::ones((data_info.rows as usize, candidates)));
    let avl_init = RawArray3::new(
        -Array3::<i32>::ones((data_info.rows as usize, k, 4)));
    let meta_init = RawArray2::new(
        -Array2::<i32>::ones((data_info.rows as usize, 3)));
    let (knn_info, knn_input) = WsglArray2Info::new(knn_init);
    let (distances_info, distances_input) = WsglArray2Info::new(distances_init);
    let (scratch_info, scratch_input) = WsglArray2Info::new(scratch_init);
    let (avl_info, avl_input) = WsglArray3Info::new(avl_init);
    let (meta_info, meta_input) = WsglArray2Info::new(meta_init);
    (WsglArgs {
        data_info: data_info,
        knn_info: knn_info,
        distances_info: distances_info,
        scratch_info: scratch_info,
        avl_info: avl_info,
        meta_info: meta_info,
        candidates: candidates as u32,
    }, WsglBuffers {
        data: data_input,
        knn: knn_input,
        distances: distances_input,
        scratch: scratch_input,
        avl: avl_input,
        meta: meta_input,
    })
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

impl<T> RawArray3<T> {
    fn new(arr: Array3<T>) -> Self {
        let shape = <Vec<usize> as TryInto<[usize; 3]>>::try_into(
            arr.shape().to_owned()).unwrap();
        let strides = <Vec<isize> as TryInto<[isize; 3]>>::try_into(
            arr.strides().to_owned()).unwrap();
        let (v, offset) = arr.into_raw_vec_and_offset();
        RawArray3 {
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

impl WsglArray3Info {
    fn new<T>(raw: RawArray3<T>) -> (Self, Vec<T>) {
        (WsglArray3Info {
            offset: raw.offset,
            rows: raw.shape[0],
            cols: raw.shape[1],
            voxs: raw.shape[2],
            row_strides: raw.strides[0],
            col_strides: raw.strides[1],
            vox_strides: raw.strides[2],
        }, raw.data)
    }
}
