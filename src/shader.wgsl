// max_bind_groups with wgpu is 4
@group(0) @binding(0) var<storage, read>       data:      array<f32>;
@group(0) @binding(1) var<storage, read_write> knn:       array<i32>;
@group(0) @binding(2) var<storage, read_write> scratch:   array<i32>;

/*************************/
/* tensor alignment data */
/*************************/

override k: u32 = 15u;
override candidates: u32 = 15u;
override seed: u32 = 0u;

override ndim: u32;
override points: u32;

override data_offset: u32;
override data_row_strides: u32;
override data_col_strides: u32;

override knn_offset: u32;
override knn_row_strides: u32;
override knn_col_strides: u32;

override avl_offset: u32;
override avl_row_strides: u32;
override avl_col_strides: u32;
override avl_vox_strides: u32;

override meta_offset: u32;
override meta_row_strides: u32;
override meta_col_strides: u32;

override link_offset: u32;
override link_row_strides: u32;
override link_col_strides: u32;
override link_vox_strides: u32;

var<workgroup> reverse_ticket:
    array<atomic<i32>, points * candidate_col_strides>;

var<workgroup> distances: array<f32, points * k>;

override distances_offset: u32 = 0u;
override distances_row_strides: u32 = k;
override distances_col_strides: u32 = 1u;

var<workgroup> candidate_buffer:
    array<atomic<i32>, points * candidate_row_strides>;

override candidate_offset: u32 = 0u;
override candidate_row_strides: u32 = candidates * candidate_col_strides;
override candidate_col_strides: u32 = 2u;
override candidate_vox_strides: u32 = 1u;

var<workgroup> reservations: array<vec2i, points * reservations_row_strides>;

override reservations_row_strides: u32 = (
    reservations_col_strides * knn_row_strides);
override reservations_col_strides: u32 = 2u;
override reservations_vox_strides: u32 = 1u;

var<workgroup> boundary: array<f32, points * candidate_row_strides>;

override boundary_offset: u32 = 0u;
override boundary_row_strides: u32 = candidates * boundary_col_strides;
override boundary_col_strides: u32 = 2u;
override boundary_vox_strides: u32 = 1u;

/***********************/
/* tensor access utils */
/***********************/

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

fn candidate_get(row: u32, col: u32, vox: u32) -> i32 {
    return atomicLoad(&candidate_buffer[
        candidate_offset +
        row * candidate_row_strides +
        col * candidate_col_strides +
        vox * candidate_vox_strides]);
}

fn candidate_set(row: u32, col: u32, vox: u32, value: i32) {
    atomicStore(&candidate_buffer[
        candidate_offset +
        row * candidate_row_strides +
        col * candidate_col_strides +
        vox * candidate_vox_strides], value);
}

fn candidate_max(row: u32, col: u32, vox: u32, value: i32) {
    atomicMax(&candidate_buffer[
        candidate_offset +
        row * candidate_row_strides +
        col * candidate_col_strides +
        vox * candidate_vox_strides], value);
}

fn ticket_get(row: u32, col: u32) -> i32 {
    return atomicLoad(&reverse_ticket[row * candidate_col_strides + col]);
}

fn ticket_set(row: u32, col: u32, value: i32) {
    atomicStore(&reverse_ticket[row * candidate_col_strides + col], value);
}

fn ticket_take(row: u32, col: u32) -> i32 {
    loop {
        let ticket = ticket_get(row, col);
        let res = atomicCompareExchangeWeak(
            &reverse_ticket[row * candidate_col_strides + col],
            ticket, ticket + 1);
        if (res.exchanged) {
            return ticket;
        }
    }
    return 0;
}

fn ticket_exchange(row: u32, col: u32, value: i32) -> i32 {
    return atomicExchange(
        &reverse_ticket[row * candidate_col_strides + col], value);
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

fn link_get(index: u32) -> i32 {
    return scratch[link_offset + index];
}

fn link_set(row: u32, col: u32, vox: u32, value: i32) {
    scratch[
        link_offset +
        row * link_row_strides +
        col * link_col_strides +
        vox * link_vox_strides] = value;
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

fn reservations_get(row: u32, col: u32, vox: u32) -> vec2i {
    return reservations[
        row * reservations_row_strides +
        col * reservations_col_strides +
        vox * reservations_vox_strides];
}

fn reservations_set(row: u32, col: u32, vox: u32, value: vec2i) {
    reservations[
        row * reservations_row_strides +
        col * reservations_col_strides +
        vox * reservations_vox_strides] = value;
}

fn boundary_get(row: u32, col: u32, vox: u32) -> f32 {
    return boundary[
        boundary_offset +
        row * boundary_row_strides +
        col * boundary_col_strides +
        vox * boundary_vox_strides];
}

fn boundary_set(row: u32, col: u32, vox: u32, value: f32) {
    boundary[
        boundary_offset +
        row * boundary_row_strides +
        col * boundary_col_strides +
        vox * boundary_vox_strides] = value;
}

fn distance(row0: u32, row1: u32) -> f32 {
    var total = 0.;
    for (var i = 0u; i < k; i++) {
        total += pow(data_get(row0, i) - data_get(row1, i), 2.);
    }
    return total;
}

/************/
/* avl impl */
/************/

const avl_height = 0u;
const avl_left = 1u;
const avl_right = 2u;
const avl_up = 3u;

const avl_root = 0u;
const avl_max = 1u;
const avl_link_0 = 2u;
const avl_link_1 = 3u;

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
    if (z >= 0) { avl_set(row, u32(z), avl_up, i32(y)); }
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
    if (z >= 0) { avl_set(row, u32(z), avl_up, i32(x)); }
    return u32(y);
}

fn avl_pre_balance(row: u32, root: u32, balance: i32) {
    var node = 0u;
    if (balance == 1) {
        node = avl_rotate_left(row, u32(avl_get(row, root, avl_left)));
        avl_set(row, root, avl_left, i32(node));
    } else {
        node = avl_rotate_right(row, u32(avl_get(row, root, avl_right)));
        avl_set(row, root, avl_right, i32(node));
    }
    avl_set(row, node, avl_up, i32(root));
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
    var root = path.x;
    var node = i32(x);
    var prev = i32(x);
    var sign = path.y;
    var side = 0;
    avl_set(row, x, avl_up, root);
    avl_set(row, x, avl_height, 1);
    if (root != -1) {
        loop {
            if (sign == 1) {
                avl_set(row, u32(root), avl_right, node);
            } else {
                avl_set(row, u32(root), avl_left, node);
            }
            avl_set(row, u32(root), avl_height, avl_measured(row, root));
            let balance = l1ge2(avl_balance(row, root));
            if (balance == 0) {
                side = 1;
            } else if (balance == 1) {
                side = avl_cmp(row, prev, avl_get(row, u32(root), avl_left));
            } else {
                side = avl_cmp(row, prev, avl_get(row, u32(root), avl_right));
            }
            if (balance == side) {
                avl_pre_balance(row, u32(root), balance);
            }
            prev = node;
            node = i32(avl_re_balance(row, u32(root), balance));
            root = avl_get(row, u32(node), avl_up);
            if (root == -1) { break; }
            sign = avl_cmp(row, i32(node), root);
        }
    }
    meta_set(row, avl_root, node);
}

fn avl_remove(row: u32, x: u32) {
    let parent = avl_get(row, x, avl_up);
    let l = avl_get(row, x, avl_left);
    let r = avl_get(row, x, avl_right);
    var child = -1;
    var node = -1;
    if (l == -1 || r == -1) {
        node = parent;
        if (l == -1) {
            child = r;
            if (r != -1) {
                avl_set(row, u32(child), avl_up, parent);
            }
        } else {
            child = l;
            avl_set(row, u32(child), avl_up, parent);
        }
    } else {
        for (var i = r; i != -1; i = avl_get(row, u32(i), avl_left)) {
            child = i;
        }
        let grandchild = avl_get(row, u32(child), avl_right);
        node = avl_get(row, u32(child), avl_up);
        avl_set(row, u32(node), avl_left, grandchild);
        if (grandchild != -1) { avl_set(row, u32(grandchild), avl_up, node); }
        if (r == child) {
            avl_set(row, u32(child), avl_right, -1);
            node = child;
        } else {
            avl_set(row, u32(child), avl_right, r);
            avl_set(row, u32(r), avl_up, child);
        }
        avl_set(row, u32(child), avl_left, l);
        avl_set(row, u32(l), avl_up, child);
        avl_set(row, u32(child), avl_up, parent);
    }
    var sign = 0;
    if (parent == -1) {
        meta_set(row, avl_root, child);
    } else if (avl_get(row, u32(parent), avl_left) == i32(x)) {
        avl_set(row, u32(parent), avl_left, child);
        sign = -1;
    } else {
        avl_set(row, u32(parent), avl_right, child);
        sign = 1;
    }
    if (child != -1) {
        avl_set(row, u32(child), avl_up, parent);
    }
    avl_set(row, x, avl_left, -1);
    avl_set(row, x, avl_right, -1);
    avl_set(row, x, avl_up, -1);
    avl_set(row, x, avl_height, -1);
    if (node != -1) {
        var prev = -1;
        var side = 0;
        loop {
            if (prev != -1) {
                if (sign == 1) {
                    avl_set(row, u32(node), avl_right, prev);
                } else {
                    avl_set(row, u32(node), avl_left, prev);
                }
            }
            avl_set(row, u32(node), avl_height, avl_measured(row, node));
            let balance = l1ge2(avl_balance(row, node));
            if (balance == 0) {
                side = 0;
            } else if (balance == 1) {
                side = avl_balance(row, avl_get(row, u32(node), avl_left));
            } else {
                side = avl_balance(row, avl_get(row, u32(node), avl_right));
            }
            if (l1ge2(balance - side) != 0) {
                avl_pre_balance(row, u32(node), balance);
            }
            prev = i32(avl_re_balance(row, u32(node), balance));
            node = avl_get(row, u32(prev), avl_up);
            if (node == -1) { break; }
            sign = avl_cmp(row, prev, node);
        }
        meta_set(row, avl_root, prev);
    }
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

fn avl_check(row: u32, primary: f32, secondary: i32) -> bool {
    let res = avl_search(row, primary, secondary);
    return res.x != -1 && res.y == 0;
}

fn avl_push(row: u32, candidate: i32) {
    if (row == u32(candidate)) { return; }
    let dist = distance(row, u32(candidate));
    let idx = meta_get(row, avl_max);
    if (avl_check(row, dist, candidate)) { return; }
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
        avl_remove(row, u32(idx));
        distances_set(row, u32(idx), dist);
        knn_flag_set(row, u32(idx), candidate, true);
        avl_insert(row, u32(idx));
    }
    avl_set_max(row);
}

/************/
/* rng impl */
/************/

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
    // TODO
    return f32(x) / (f32(0xFFFFFFFF) + 1.);
    //return bitcast<f32>(0x007FFFFF & x) - 1.;
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

fn split(rng: vec2u, iota: u32) -> vec2u {
    return threefry2x32(rng, vec2u(0, iota));
}

/**********************/
/* reservoir sampling */
/**********************/

fn reservoir(ticket: i32, rng: vec2u, iota: vec2u) -> i32 {
    if (u32(ticket) >= candidates) {
        let x = bitcast_mantissa(threefry2x32(rng, iota).x);
        return i32(floor(x * f32(ticket)));
    }
    return ticket;
}

fn reserve(
    rng: vec2u,
    row: u32,
    col: u32,
    other: u32,
    direction: u32,
    flag: u32,
) {
    let ticket = ticket_take(other, flag);
    let overlay = reservoir(ticket, rng, vec2u(direction, col));
    if (u32(overlay) < candidates) {
        candidate_max(other, u32(overlay), flag, ticket);
    }
    reservations_set(row, col, direction, vec2i(ticket, overlay));
}

fn build(rng: vec2u, row: u32) {
    for (var i = 0u; i < 2; i++) {
        ticket_set(row, i, 0);
    }
    for (var i = 0u; i < candidates; i++) {
        for (var j = 0u; j < 2; j++) {
            candidate_set(row, i, j, -1);
        }
    }
    storageBarrier();
    for (var i = 0u; i < k; i++) {
        let other = knn_get(row, i);
        if (other < 0) { continue; }
        reserve(rng, row, i, row, 0u, u32(flag_get(row, i)));
    }
    for (var i = 0u; i < k; i++) {
        let other = knn_get(row, i);
        if (other < 0) { continue; }
        reserve(rng, row, i, u32(other), 1u, u32(flag_get(row, i)));
    }
    storageBarrier();
    for (var i = 0u; i < 2; i++) {
        ticket_set(row, i, -1);
    }
    storageBarrier();
    for (var i = 0u; i < k; i++) {
        let other = knn_get(row, i);
        if (other < 0) { continue; }
        let ticket = reservations_get(row, i, 0u);
        if (ticket.y >= i32(candidates)) { continue; }
        let flag = flag_get(row, i);
        let cmp = candidate_get(row, u32(ticket.y), u32(flag));
        if (cmp != ticket.x) { continue; }
        let linking = ticket_exchange(u32(other), u32(flag), (
            i32(row * link_row_strides) +
            ticket.y * i32(link_col_strides) +
            i32(flag) * i32(link_vox_strides)));
        link_set(row, u32(ticket.y), u32(flag), linking);
        candidate_set(row, u32(ticket.y), u32(flag), -2 - other);
    }
    for (var i = 0u; i < k; i++) {
        let other = knn_get(row, i);
        if (other < 0) { continue; }
        let ticket = reservations_get(row, i, 1u);
        if (ticket.y >= i32(candidates)) { continue; }
        let flag = flag_get(row, i);
        let cmp = candidate_get(u32(other), u32(ticket.y), u32(flag));
        if (cmp != ticket.x) { continue; }
        let linking = ticket_exchange(row, u32(flag), (
            other * i32(link_row_strides) +
            ticket.y * i32(link_col_strides) +
            i32(flag) * i32(link_vox_strides)));
        link_set(u32(other), u32(ticket.y), u32(flag), linking);
        candidate_set(u32(other), u32(ticket.y), u32(flag), -2 - i32(row));
    }
    storageBarrier();
    meta_set(row, avl_link_0, ticket_get(row, 0));
    meta_set(row, avl_link_1, ticket_get(row, 1));
    for (var i = 0u; i < candidates; i++) {
        for (var j = 0u; j < 2; j++) {
            candidate_set(row, i, j, -2 - candidate_get(row, i, j));
        }
    }
}

const max_init = 0x1.fffffep+127f;
const flag_new = 1u;

// stores to flag0 in boundary
fn bound(row: u32, flag0: u32, flag1: u32) {
    for (var i = 0u; i < candidates; i++) {
        var lo = max_init;
        let point0i = candidate_get(row, i, flag0);
        if (point0i == -1) { continue; }
        let point0 = u32(point0i);
        let threshold = meta_get(point0, avl_max);
        for (var j = 0u; j < candidates; j++) {
            if (i == j && flag0 == flag1) { continue; }
            let point1i = candidate_get(row, j, flag1);
            if (point1i == -1 || point0i == point1i) { continue; }
            let point1 = u32(point1i);
            let dist = distance(point0, point1);
            if (avl_sign(point0, dist, point1i, threshold) >= 0) { continue; }
            if (avl_check(point0, dist, point1i)) { continue; }
            if (dist < lo) {
                lo = dist;
            }
        }
        boundary_set(row, i, flag0, lo);
    }
}

fn link(row: u32, start: i32) {
    for (var node = start; node != -1; node = link_get(u32(node))) {
        let target_row = u32(node) / link_row_strides;
        let target_col = (u32(node) % link_row_strides) / link_col_strides;
        let target_vox = (u32(node) % link_col_strides) / link_vox_strides;
        let target_bound = boundary_get(target_row, target_col, target_vox);
        if (target_bound == max_init) { continue; }
        let threshold = meta_get(row, avl_max);
        if (avl_sign(row, target_bound, -1, threshold) >= 0) { continue; }
        for (var i = 0u; i < candidates; i++) {
            if (target_vox == flag_new && i == target_col) { continue; }
            let other = candidate_get(target_row, i, flag_new);
            if (other == -1) { continue; }
            avl_push(row, other);
        }
    }
}

@compute
@workgroup_size(points)
fn main(@builtin(local_invocation_index) row: u32) {
    randomize(split(vec2u(0, seed), row), row);
    build(split(vec2u(1, seed), row), row);
    storageBarrier();
    bound(row, 0u, flag_new);
    bound(row, 1u, flag_new);
    storageBarrier();
    link(row, meta_get(row, avl_link_0));
    link(row, meta_get(row, avl_link_1));
}
