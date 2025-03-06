struct YoloInSlice {
    relative: array<array<f32, 16>, 4>,
    scores: array<f32, 80>,
}

struct YoloInData {
    slices: array<YoloInSlice>,
}

struct YoloInCfg {
    step: u32,
    width: u32,
    stride: vec2<f32>,
}

struct YoloOutSlice {
    bbox: array<vec2<f32>, 2>,
    scores: array<f32, 80>,
}

struct YoloOutData {
    slices: array<YoloOutSlice>,
}


// A, B and C vectors
@group(0) @binding(0) var<storage, read>  yolo_in: YoloInData;
@group(0) @binding(1) var<storage, read_write> yolo_out: YoloOutData;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn softmax_conv(slice_id: u32, seg_id: i32) -> f32 {
    let slice = &yolo_in.slices[slice_id];
    let relative = &(*slice).relative[seg_id];
    var max_val = (*relative)[0];
    var sum = 0.0;
    var temp: array<f32, 16>;
    var result = 0.0;
    
    // Find max value
    for (var i = 1; i < 16; i=i+1) {
        max_val = max(max_val, (*relative)[i]);
    }

    // Compute exp(x - max) and sum
    for (var i = 0; i < 16; i=i+1) {
        temp[i] = exp((*relative)[i] - max_val);
        sum = sum + temp[i];
    }
    
    // Normalize & Conv
    for (var i = 0; i < 16; i=i+1) {
        result = result + (temp[i] / sum) * f32(i);
    }
    return result;
}

fn update_scores(slice_id: u32) {
    let scores = &yolo_in.slices[slice_id].scores;
    let yolo_score_output = &yolo_out.slices[slice_id].scores;
    for (var i = 0; i < 80; i=i+1) {
        let raw_score: f32 = (*scores)[i];
        let score: f32 = sigmoid(raw_score);
        (*yolo_score_output)[i] = score ;
    }
}

fn proc_yolo_output(global_id: vec3<u32>, slice_base: u32, width: u32) {
    let stride = vec2(640.0 / f32(width), 640.0 / f32(width));
    let yolo_cfg = YoloInCfg(width * width / 100, width, stride);
    for (var step_id: u32 = 0; step_id < yolo_cfg.step; step_id++) {
        let slice_off: u32 = global_id.x * yolo_cfg.step + step_id;
        let y : u32 = slice_off / yolo_cfg.width;
        let x : u32 = slice_off % yolo_cfg.width;
        let slice_id = slice_base + slice_off;
        let pos = vec2<f32>(f32(x), f32(y)) + 0.5;
        // cal distance
        let x0 = softmax_conv(slice_id, 0);
        let y0 = softmax_conv(slice_id, 1);
        let x1 = softmax_conv(slice_id, 2);
        let y1 = softmax_conv(slice_id, 3);
        let distance0 = vec2<f32>(x0, y0);
        let distance1 = vec2<f32>(x1, y1);
        let point0 = (pos - distance0) * yolo_cfg.stride;
        yolo_out.slices[slice_id].bbox[0] = point0;
        let point1 = (pos + distance1) * yolo_cfg.stride;
        yolo_out.slices[slice_id].bbox[1] = point1;
        update_scores(slice_id);
    }
}

@compute @workgroup_size(200, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    proc_yolo_output(global_id, u32(0), u32(80));
    proc_yolo_output(global_id, u32(80 * 80), u32(40));
    proc_yolo_output(global_id, u32(80 * 80 + 40 * 40), u32(20));
}