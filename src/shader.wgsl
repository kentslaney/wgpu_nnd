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

@group(0) @binding(0) var<storage, read>       info: HostInterface;
@group(0) @binding(1) var<storage, read>       data: array<f32>;
@group(0) @binding(2) var<storage, read_write> knn:  array<u32>;

override points: u32 = 0u;
override k: u32 = 0u;
override candidates: u32 = 0u;
override seed: u32 = 0u;

fn rotate_left(x: u32, d: u32) -> u32 {
    return x << d | x >> (32 - d);
}

fn apply_round(v: vec2u, d: u32) -> vec2u {
    let v0 = v.x + v.y;
    let v1 = v0 ^ rotate_left(v.y, d);
    return vec2u(v0, v1);
}

fn apply_rounds(v: vec2u, r: vec4u) -> vec2u {
    return apply_round(apply_round(apply_round(apply_round(
        v, r.x), r.y), r.z), r.w);
}

fn apply_group(x: vec2u, r: vec4u, i: u32, j: u32, n: u32) -> vec2u {
    let y = apply_rounds(x, r);
    return vec2u(y.x + i, y.y + j + n);
}

fn threefry2x32(k: vec2u, x: vec2u) -> vec2u {
    let r0 = vec4u(13, 15, 26,  6);
    let r1 = vec4u(17, 29, 16, 24);
    let j = k.x ^ k.y ^ 0x1BD11BDA;
    return apply_group(apply_group(apply_group(apply_group(apply_group(x + k,
        r0, k.y,   j, 1u),
        r1,   j, k.x, 2u),
        r0, k.x, k.y, 3u),
        r1, k.y,   j, 4u),
        r0,   j, k.x, 5u);
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3u) {
    if data[wid.x] > 0.5 {
        knn[wid.x] += threefry2x32(vec2u(0, 0), vec2u(0, wid.x)).x;
    }
}
