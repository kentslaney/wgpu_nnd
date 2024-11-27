// max_bind_groups with wgpu is 4
@group(0) @binding(0) var<storage, read>       data:      array<f32>;
@group(0) @binding(1) var<storage, read_write> distances: array<f32>;
@group(0) @binding(2) var<storage, read_write> knn:       array<i32>;
@group(0) @binding(3) var<storage, read_write> scratch:   array<i32>;

override k: u32 = 15u;
override candidates: u32 = 15u;
override seed: u32 = 0u;

override ndim: u32 = 0u;
override points: u32 = 0u;

override data_offset = 0u;
override data_row_strides = 0u;
override data_col_strides = 0u;

override distances_offset = 0u;
override distances_row_strides = 0u;
override distances_col_strides = 0u;

override knn_offset = 0u;
override knn_row_strides = 0u;
override knn_col_strides = 0u;

override scratch_offset = 0u;
override scratch_row_strides = 0u;
override scratch_col_strides = 0u;

override avl_offset = 0u;
override avl_row_strides = 0u;
override avl_col_strides = 0u;
override avl_vox_strides = 0u;

override meta_offset = 0u;
override meta_row_strides = 0u;
override meta_col_strides = 0u;

fn data_get(row: u32, col: u32) -> f32 {
    return data[data_offset + row * data_row_strides + col * data_col_strides];
}

fn distances_get(row: u32, col: u32) -> f32 {
    return distances[
        distances_offset +
        row * distances_row_strides +
        col * distances_col_strides];
}

fn distances_set(row: u32, col: u32, value: f32) {
    distances[
        distances_offset +
        row * distances_row_strides +
        col * distances_col_strides] = value;
}

fn scratch_get(row: u32, col: u32) -> i32 {
    return scratch[
        scratch_offset +
        row * scratch_row_strides +
        col * scratch_col_strides];
}

fn scratch_set(row: u32, col: u32, value: i32) {
    scratch[
        scratch_offset +
        row * scratch_row_strides +
        col * scratch_col_strides] = value;
}

fn avl_get(row: u32, col: u32, vox: u32) -> i32 {
    return scratch[
        avl_offset +
        row * avl_row_strides +
        col * avl_col_strides +
        vox * avl_vox_strides];
}

fn avl_set(row: u32, col: u32, vox: u32, value: i32) {
    scratch[
        avl_offset +
        row * avl_row_strides +
        col * avl_col_strides +
        vox * avl_vox_strides] = value;
}

fn meta_get(row: u32, col: u32) -> i32 {
    return scratch[
        meta_offset +
        row * meta_row_strides +
        col * meta_col_strides];
}

fn meta_set(row: u32, col: u32, value: i32) {
    scratch[
        meta_offset +
        row * meta_row_strides +
        col * meta_col_strides] = value;
}

fn knn_get(row: u32, col: u32) -> i32 {
    let res = knn[knn_offset + row * knn_row_strides + col * knn_col_strides];
    if res >= i32(points) {
        return res - i32(points);
    } else {
        return res;
    }
}

fn flag_get(row: u32, col: u32) -> bool {
    return knn[
        knn_offset +
        row * knn_row_strides +
        col * knn_col_strides] >= i32(points);
}

fn knn_flag_set(row: u32, col: u32, index: i32, flag: bool) {
    var value = index;
    if (flag) {
        value += i32(points);
    }
    knn[knn_offset + row * knn_row_strides + col * knn_col_strides] = value;
}

fn flag_reset(row: u32, col: u32) {
    knn[
            knn_offset +
            row * knn_row_strides +
            col * knn_col_strides
        ] = knn[
            knn_offset +
            row * knn_row_strides +
            col * knn_col_strides
        ] % i32(points);
}

fn index_swap(row: u32, col0: u32, col1: u32) {
    let idx0 = knn[knn_offset + row * knn_row_strides + col0 * knn_col_strides];
    let idx1 = knn[knn_offset + row * knn_row_strides + col1 * knn_col_strides];
    knn[knn_offset + row * knn_row_strides + col0 * knn_col_strides] = idx1;
    knn[knn_offset + row * knn_row_strides + col1 * knn_col_strides] = idx0;

    let dist0 = distances[
        distances_offset +
        row * distances_row_strides +
        col0 * distances_col_strides];
    let dist1 = distances[
        distances_offset +
        row * distances_row_strides +
        col1 * distances_col_strides];
    distances[
        distances_offset +
        row * distances_row_strides +
        col0 * distances_col_strides] = dist1;
    distances[
        distances_offset +
        row * distances_row_strides +
        col1 * distances_col_strides] = dist0;
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
    if (row == u32(candidate)) { return; }
    let dist = distance(row, u32(candidate));
    if (distances_get(row, 0u) <= dist) { return; }
    if (check(row, candidate)) { return; }
    distances_set(row, 0u, dist);
    knn_flag_set(row, 0u, candidate, true);
    sifted(row);
}

const avl_height = 0u;
const avl_left = 1u;
const avl_right = 2u;
const avl_up = 3u;

const avl_root = 0u;
const avl_max = 1u;
const avl_link = 2u;

fn avl_depth(row: u32, col: i32) -> i32 {
    if (col < 0) { return 0; }
    let res = avl_get(row, u32(col), avl_height);
    if (res < 0) { return 0; }
    return res;
}

fn avl_balance(row: u32, col: i32) -> i32 {
    if (col < 0) { return 0; }
    let l = avl_depth(row, avl_get(row, u32(col), avl_left));
    let r = avl_depth(row, avl_get(row, u32(col), avl_right));
    return l - r;
}

fn avl_measured(row: u32, col: i32) -> i32 {
    if (col < 0) { return 0; }
    let l = avl_depth(row, avl_get(row, u32(col), avl_left));
    let r = avl_depth(row, avl_get(row, u32(col), avl_right));
    return 1 + max(l, r);
}

fn avl_cmp(row: u32, col0: i32, col1: i32) -> i32 {
    if (col0 < 0) {
        if (col1 < 0) {
            return 0;
        } else {
            return 1;
        }
    }
    return avl_sign(
        row, distances_get(row, u32(col0)), knn_get(row, u32(col0)), col1);
}

fn isclose(f0: f32, f1: f32) -> bool {
    return abs(f0 - f1) / max(1e-5, abs(f0 + f1)) < 1e-5;
}

fn avl_sign(row: u32, primary: f32, secondary: i32, col: i32) -> i32 {
    if (col < 0) { return -1; }
    let col_primary = distances_get(row, u32(col));
    if (isclose(primary, col_primary)) {
        let col_secondary = knn_get(row, u32(col));
        if (secondary == col_secondary) {
            return 0;
        } else if (secondary > col_secondary) {
            return 1;
        } else {
            return -1;
        }
    } else if (primary > col_primary) {
        return 1;
    } else {
        return -1;
    }
}

fn avl_rotate_right(row: u32, y: u32) -> u32 {
    let x = avl_get(row, y, avl_left);
    if (x < 0) { return y; }
    let z = avl_get(row, u32(x), avl_right);
    avl_set(row, u32(x), avl_right, i32(y));
    avl_set(row, u32(y), avl_left,  i32(z));
    avl_set(row, u32(y), avl_height, avl_measured(row, i32(y)));
    avl_set(row, u32(x), avl_height, avl_measured(row, x));
    avl_set(row, u32(x), avl_up, avl_get(row, u32(y), avl_up));
    avl_set(row, u32(y), avl_up, x);
    avl_set(row, u32(z), avl_up, i32(y));
    return u32(x);
}

fn avl_rotate_left(row: u32, x: u32) -> u32 {
    let y = avl_get(row, x, avl_right);
    if (y < 0) { return x; }
    let z = avl_get(row, u32(y), avl_left);
    avl_set(row, u32(y), avl_left, i32(x));
    avl_set(row, u32(x), avl_right, i32(z));
    avl_set(row, u32(x), avl_height, avl_measured(row, i32(x)));
    avl_set(row, u32(y), avl_height, avl_measured(row, y));
    avl_set(row, u32(y), avl_up, avl_get(row, u32(x), avl_up));
    avl_set(row, u32(x), avl_up, y);
    avl_set(row, u32(z), avl_up, i32(x));
    return u32(y);
}

fn avl_pre_balance(row: u32, root: u32, balance: i32) {
    if (balance == 1) {
        avl_set(row, root, avl_left, i32(avl_rotate_left(row, u32(avl_get(
            row, root, avl_left)))));
    } else {
        avl_set(row, root, avl_right, i32(avl_rotate_right(row, u32(avl_get(
            row, root, avl_right)))));
    }
}

fn avl_re_balance(row: u32, root: u32, balance: i32) -> u32 {
    if (balance == 0) {
        return root;
    } else if (balance == 1) {
        return avl_rotate_right(row, root);
    } else {
        return avl_rotate_left(row, root);
    }
}

fn avl_search(row: u32, primary: f32, secondary: i32) -> vec2i {
    var dir = 0;
    var i = meta_get(row, avl_root);
    var j = i;
    while (j != -1) {
        i = j;
        dir = avl_sign(row, primary, secondary, i);
        if (dir == 0) {
            break;
        } else if (dir == 1) {
            j = avl_get(row, u32(i), avl_right);
        } else {;
            j = avl_get(row, u32(i), avl_left);
        }
    }
    return vec2i(i, dir);
}

fn avl_path(row: u32, x: u32) -> vec2i {
    return avl_search(row, distances_get(row, x), knn_get(row, x));
}

fn l1ge2(x: i32) -> i32 {
    if (x >= 2) {
        return 1;
    } else if (x <= -2) {
        return -1;
    }
    return 0;
}

fn avl_insert(row: u32, x: u32) {
    let path = avl_path(row, x);
    if (path.y == 0 && path.x != -1) { return; }
    var node = path.x;
    var prev = i32(x);
    var sign = path.y;
    var side = 0;
    avl_set(row, x, avl_up, node);
    avl_set(row, x, avl_height, 1);
    if (node != -1) {
        loop {
            if (sign == 1) {
                avl_set(row, u32(node), avl_right, prev);
            } else {
                avl_set(row, u32(node), avl_left, prev);
            }
            avl_set(row, u32(node), avl_height, avl_measured(row, node));
            let balance = l1ge2(avl_balance(row, node));
            if (balance == 0) {
                side = 1;
            } else if (balance == 1) {
                side = avl_cmp(row, node, avl_get(row, u32(node), avl_left));
            } else {
                side = avl_cmp(row, node, avl_get(row, u32(node), avl_right));
            }
            if (balance == side) {
                avl_pre_balance(row, u32(node), balance);
            }
            prev = i32(avl_re_balance(row, u32(node), balance));
            node = avl_get(row, u32(prev), avl_up);
            if (node == -1) { break; }
            sign = avl_cmp(row, i32(x), node);
        }
    }
    meta_set(row, avl_root, prev);
}

fn avl_set_max(row: u32) {
    var node = meta_get(row, avl_root);
    var prev = node;
    while (node != -1) {
        prev = node;
        node = avl_get(row, u32(node), avl_right);
    }
    meta_set(row, avl_max, prev);
}

fn avl_push(row: u32, candidate: i32) {
    if (row == u32(candidate)) { return; }
    let dist = distance(row, u32(candidate));
    let idx = meta_get(row, avl_max);
    if (check(row, candidate)) { return; }
    if (idx < 0) {
        distances_set(row, u32(~idx), dist);
        knn_flag_set(row, u32(~idx), candidate, true);
        avl_insert(row, u32(~idx));
        if (-idx < i32(k)) {
            meta_set(row, avl_max, idx - 1);
            return;
        }
    } else {
        if (distances_get(row, u32(idx)) <= dist) { return; }
        distances_set(row, u32(idx), dist);
        knn_flag_set(row, u32(idx), candidate, true);
        avl_insert(row, u32(idx));
    }
    avl_set_max(row);
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

// big endian; necessary to guarantee span << rng result
fn big_mod(x: vec2u, span: u32) -> u32 {
    var carry = 0u;
    let mul_ = 0x00010000 % span;
    let mul = (mul_ * mul_) % span;
    carry = ((carry * mul) % span + (x.x % span)) % span;
    carry = ((carry * mul) % span + (x.y % span)) % span;
    return carry;
}

fn randomize(rng: vec2u, row: u32) {
    for (var i = 0u; meta_get(row, avl_max) < 0; i++) {
        let rand = i32(big_mod(threefry2x32(rng, vec2u(0u, i)), points));
        avl_push(row, rand);
    }
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3u) {
    let rng = threefry2x32(vec2u(0, seed), vec2u(0, wid.x));
    randomize(rng, wid.x);
    for (var i = 0u; i < k; i++) {
        flag_reset(wid.x, i);
    }
}
