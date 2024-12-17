use std::{fs::File, str::FromStr, collections::HashSet};
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
pub struct WgslArray2Info {
    pub offset: u32,
    pub rows: u32,
    pub cols: u32,
    pub row_strides: u32,
    pub col_strides: u32,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WgslArray3Info {
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
pub struct WgslArgs {
    pub data_info: WgslArray2Info,
    pub knn_info: WgslArray2Info,
    pub avl_info: WgslArray3Info,
    pub meta_info: WgslArray2Info,
    pub link_info: WgslArray3Info,
    pub candidates: u32,
}

pub struct WgslBuffers {
    pub data: Vec<f32>,
    pub knn: Vec<i32>,
    pub avl: Vec<i32>,
    pub meta: Vec<i32>,
    pub link: Vec<i32>,
}

pub struct WgslSlices<'a> {
    pub avl: &'a [i32],
    pub meta: &'a [i32],
    pub link: &'a [i32],
}

pub fn cli_npy(idx: usize) -> (WgslArray2Info, Vec<f32>) {
    let path = std::env::args().skip(idx).next().expect(
        "missing cli argument for npy file");
    WgslArray2Info::new(raw_npy(path))
}

pub fn cli() -> (WgslArgs, WgslBuffers) {
    let (data_info, data_input) = cli_npy(1);
    let numbers_input: Vec<usize> = std::env::args().skip(2).take(2).map(
        |s| usize::from_str(&s).expect("missing args")).collect();
    let (k, candidates) = (numbers_input[0], numbers_input[1]);
    let knn_init = RawArray2::new(
        -Array2::<i32>::ones((data_info.rows as usize, k)));
    let avl_init = RawArray3::new(
        -Array3::<i32>::ones((data_info.rows as usize, k, 4)));
    let meta_init = RawArray2::new(
        -Array2::<i32>::ones((data_info.rows as usize, 4)));
    let link_init = RawArray3::new(
        -Array3::<i32>::ones((data_info.rows as usize, candidates, 2)));
    let (knn_info, knn_input) = WgslArray2Info::new(knn_init);
    let (avl_info, avl_input) = WgslArray3Info::new(avl_init);
    let (meta_info, meta_input) = WgslArray2Info::new(meta_init);
    let (link_info, link_input) = WgslArray3Info::new(link_init);
    (WgslArgs {
        data_info: data_info,
        knn_info: knn_info,
        avl_info: avl_info,
        meta_info: meta_info,
        link_info: link_info,
        candidates: candidates as u32,
    }, WgslBuffers {
        data: data_input,
        knn: knn_input,
        avl: avl_input,
        meta: meta_input,
        link: link_input,
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

impl WgslArray2Info {
    fn new<T>(raw: RawArray2<T>) -> (Self, Vec<T>) {
        (WgslArray2Info {
            offset: raw.offset,
            rows: raw.shape[0],
            cols: raw.shape[1],
            row_strides: raw.strides[0],
            col_strides: raw.strides[1],
        }, raw.data)
    }
}

impl WgslArray3Info {
    fn new<T>(raw: RawArray3<T>) -> (Self, Vec<T>) {
        (WgslArray3Info {
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

pub fn visualize(info: &WgslArgs, knn: &[i32], debug: &[i32]) {
    let meta_offset = (info.avl_info.offset +
        info.avl_info.row_strides * info.avl_info.rows) as usize;
    let link_offset = meta_offset + (info.meta_info.offset +
        info.meta_info.row_strides * info.meta_info.rows) as usize;
    let avl = &debug[info.avl_info.offset as usize..meta_offset];
    let meta = &debug[
        meta_offset + (info.meta_info.offset as usize)..link_offset];
    let link = &debug[
        link_offset + (info.link_info.offset as usize)..];
    log::info!("Meta: {:?}", meta);
    for i in 0..info.knn_info.rows {
        let row_knn = &knn[
                (i * info.knn_info.row_strides) as usize..
                (i * info.knn_info.row_strides +
                    info.knn_info.col_strides * info.knn_info.cols) as usize
            ];
        let row_avl = &avl[
                (i * info.avl_info.row_strides) as usize..
                (i * info.avl_info.row_strides +
                    info.avl_info.col_strides * info.avl_info.cols) as usize
            ];
        let row_link = &link[
                (i * info.link_info.row_strides) as usize..
                (i * info.link_info.row_strides +
                    info.link_info.col_strides * info.link_info.cols) as usize];
        let tree = walk(
            row_knn, row_avl,
            info.data_info.rows as i32,
            info.avl_info.col_strides as usize,
            meta[(info.meta_info.row_strides * i) as usize],
            String::from(""),
            String::from("")
        );
        log::info!("AVL: {:?}", row_avl);
        log::info!("link: {:?}", row_link);
        /*
        println!("row {} flag 0: {}", i, trace(
                &info.link_info,
                meta[(info.meta_info.row_strides * i + 2) as usize],
                &link, &mut HashSet::new()));
        */
        println!("row {} flag 1: {}", i, trace(
                &info.link_info,
                meta[(info.meta_info.row_strides * i + 3) as usize],
                &link, &mut HashSet::new()));
        println!("{}", &tree[0..std::cmp::max(tree.len(), 1) - 1]);
    }
}

fn walk(
    knn: &[i32],
    avl: &[i32],
    points: i32,
    strides: usize,
    node: i32,
    prefix: String,
    postfix: String
) -> String {
    if node == -1 { return "".to_owned(); }
    let mut line = format!(
        "{}{}", &prefix[0..std::cmp::max(prefix.len(), 3) - 3], postfix);
    let h = avl[node as usize * strides + 0];
    let l = avl[node as usize * strides + 1];
    let r = avl[node as usize * strides + 2];
    let u = avl[node as usize * strides + 3];
    if l == -1 && r == -1 {
        line.push_str(&format!(
            "\u{2500}({}^{}h{}) {}\n",
            node, u, h, knn[node as usize] % points));
    } else {
        line.push_str(&format!(
            "\u{252C}({}^{}h{}) {}\n",
            node, u, h, knn[node as usize] % points));
        line.push_str(&walk(knn, avl, points, strides, l, format!(
            "{}\u{2502}", prefix), "\u{251C}".to_owned()));
        line.push_str(&walk(knn, avl, points, strides, r, format!(
            "{}\u{2007}", prefix), "\u{2514}".to_owned()));
    }
    line
}

fn trace(
    link_info: &WgslArray3Info,
    start: i32,
    links: &[i32],
    seen: &mut HashSet<i32>
) -> String {
    if start == -1 {
        return "end".to_owned();
    } else if start >= (links.len() as i32) {
        return "oob".to_owned();
    } else if seen.contains(&start) {
        return "loop".to_owned();
    } else {
        seen.insert(start);
    }
    format!(
        "r{}c{}v{} -> {}",
        (start as u32) / link_info.row_strides,
        ((start as u32) % link_info.row_strides) / link_info.col_strides,
        ((start as u32) % link_info.col_strides) / link_info.vox_strides,
        trace(link_info, links[start as usize], links, seen))
}
