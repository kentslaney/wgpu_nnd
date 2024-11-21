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
    distances_info: Array2Info,
    candidates: u32,
}

// max_bind_groups with wgpu is 4
@group(0) @binding(0) var<storage, read>       info:      HostInterface;
@group(0) @binding(1) var<storage, read>       data:      array<f32>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;
@group(0) @binding(3) var<storage, read_write> knn:       array<i32>;

override k: u32 = 15u;
override candidates: u32 = 15u;
override seed: u32 = 0u;

fn data_get(row: u32, col: u32) -> f32 {
    return data[
        info.data_info.offset +
        row * info.data_info.row_strides +
        col * info.data_info.col_strides];
}

fn distances_get(row: u32, col: u32) -> f32 {
    return distances[
        info.distances_info.offset +
        row * info.distances_info.row_strides +
        col * info.distances_info.col_strides];
}

fn knn_get(row: u32, col: u32) -> i32 {
    return knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col * info.knn_info.col_strides] % i32(info.knn_info.rows);
}

fn flag_get(row: u32, col: u32) -> bool {
    return knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col * info.knn_info.col_strides] >= i32(info.knn_info.rows);
}

fn distances_set(row: u32, col: u32, value: f32) {
    distances[
        info.distances_info.offset +
        row * info.distances_info.row_strides +
        col * info.distances_info.col_strides] = value;
}

fn knn_flag_set(row: u32, col: u32, index: i32, flag: bool) {
    var value = index;
    if (flag) {
        value += i32(info.knn_info.rows);
    }
    knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col * info.knn_info.col_strides] = value;
}

fn flag_reset(row: u32, col: u32) {
    knn[
            info.knn_info.offset +
            row * info.knn_info.row_strides +
            col * info.knn_info.col_strides
        ] = knn[
            info.knn_info.offset +
            row * info.knn_info.row_strides +
            col * info.knn_info.col_strides
        ] % i32(info.knn_info.rows);
}

fn index_swap(row: u32, col0: u32, col1: u32) {
    let idx0 = knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col0 * info.knn_info.col_strides];
    let idx1 = knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col1 * info.knn_info.col_strides];
    knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col0 * info.knn_info.col_strides] = idx1;
    knn[
        info.knn_info.offset +
        row * info.knn_info.row_strides +
        col1 * info.knn_info.col_strides] = idx0;

    let dist0 = distances[
        info.distances_info.offset +
        row * info.distances_info.row_strides +
        col0 * info.distances_info.col_strides];
    let dist1 = distances[
        info.distances_info.offset +
        row * info.distances_info.row_strides +
        col1 * info.distances_info.col_strides];
    distances[
        info.distances_info.offset +
        row * info.distances_info.row_strides +
        col0 * info.distances_info.col_strides] = dist1;
    distances[
        info.distances_info.offset +
        row * info.distances_info.row_strides +
        col1 * info.distances_info.col_strides] = dist0;
}

fn sifted(row: u32) {
    var col = 0u;
    while (col * 2 + 1 < k) {
        let left = col * 2 + 1;
        let right = left + 1;
        var swap = col;
        if (distances_get(row, swap) < distances_get(row, left)) {
            swap = left;
        }
        if (right < k && distances_get(row, swap) < distances_get(row, right)) {
            swap = right;
        }
        if (swap == col) { break; }
        index_swap(row, col, swap);
        col = swap;
    }
}

fn check(row: u32, index: i32) -> bool {
    for (var i = 0u; i < k; i++) {
        if (index == knn_get(row, i)) {
            return true;
        }
    }
    return false;
}

fn distance(row0: u32, row1: u32) -> f32 {
    var total = 0.;
    for (var i = 0u; i < k; i++) {
        total += pow(data_get(row0, i) - data_get(row1, i), 2.);
    }
    return total;
}

fn push(row: u32, candidate: i32) {
    let dist = distance(row, u32(candidate));
    if (distances_get(row, 0u) <= dist) { return; }
    if (check(row, candidate)) { return; }
    distances_set(row, 0u, dist);
    knn_flag_set(row, 0u, candidate, true);
    sifted(row);
}

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

fn bitcast_mantissa(x: u32) -> f32 {
    return bitcast<f32>(0x007FFFFF & x) - 1.;
}

// big endian
fn big_mod(x: vec2u, span: u32) -> u32 {
    var carry = 0u;
    let mul_ = 0x00010000 % span;
    let mul = (mul_ * mul_) % span;
    carry = ((carry * mul) % span + (x.x % span)) % span;
    carry = ((carry * mul) % span + (x.y % span)) % span;
    return carry;
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3u) {
    if data[wid.x] > 0.5 {
        knn[wid.x] += i32(candidates);
    }
}
