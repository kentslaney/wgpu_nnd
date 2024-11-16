struct Array2Info {
    offset: u32,
    rows: u32,
    cols: u32,
    row_strides: u32,
    col_strides: u32,
}

struct HostInterface {
    data_info: Array2Info,
    knn_info: Array2Info,
    candidates: u32,
}

@group(0)
@binding(0)
var<storage, read> info: HostInterface;

@group(0)
@binding(1)
var<storage, read> data: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> knn: array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    if data[wid.x] > 0.5 {
        knn[wid.x] += info.candidates;
    }
}
