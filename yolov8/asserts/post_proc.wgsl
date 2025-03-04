// Vector type definition. Used for both input and output
struct Vector {
    data: [[stride(4)]] array<u32>;
};

// A, B and C vectors
[[group(0), binding(0)]] var<storage, read>  a: Vector;
[[group(0), binding(1)]] var<storage, read>  b: Vector;
[[group(0), binding(2)]] var<storage, read_write> c: Vector;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    c.data[global_id.x] = a.data[global_id.x] * b.data[global_id.x];
}