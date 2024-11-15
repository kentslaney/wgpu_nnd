@group(0)
@binding(0)
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> b: array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    if a[wid.x] > 0.5 {
        b[wid.x] += 1u;
    }
}
