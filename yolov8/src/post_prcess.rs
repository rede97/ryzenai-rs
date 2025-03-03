use image::Rgb;
use ndarray::{Array1, ArrayBase, Axis, Ix1, s};
use ort::session::SessionOutputs;
use ort::tensor::ArrayExtensions;
use std::fmt::Display;
use std::rc::Rc;

use crate::image::ImageScale;

#[rustfmt::skip]
pub const YOLOV8_CLASS_LABELS: [&'static str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

#[rustfmt::skip]
pub const RGB_COLORS: [[u8; 3]; 80] = [
    [144, 246, 100], [89, 80, 9], [30, 103, 243], [3, 0, 216], [16, 134, 240], [69, 12, 65],
    [237, 197, 11], [54, 233, 32], [4, 7, 124], [89, 31, 164], [15, 118, 27], [141, 88, 14],
    [87, 161, 83], [65, 221, 143], [118, 236, 49], [150, 228, 27], [117, 71, 59], [119, 81, 131],
    [19, 26, 138], [90, 143, 184], [33, 188, 178], [53, 5, 176], [39, 154, 73], [11, 6, 16],
    [228, 152, 10], [93, 224, 226], [40, 39, 189], [164, 126, 30], [46, 196, 150], [228, 216, 31],
    [48, 177, 126], [163, 149, 95], [114, 161, 144], [182, 65, 13], [232, 192, 205], [103, 133, 128],
    [21, 5, 226], [49, 254, 39], [116, 151, 158], [54, 66, 12], [62, 158, 202], [160, 117, 155],
    [159, 193, 210], [163, 173, 255], [212, 56, 250], [117, 252, 106], [237, 5, 93], [169, 197, 25],
    [94, 37, 56], [16, 143, 36], [79, 131, 51], [124, 220, 20], [68, 229, 134], [154, 73, 108],
    [190, 176, 120], [70, 60, 41], [203, 124, 118], [178, 46, 163], [64, 36, 162], [55, 44, 55],
    [206, 200, 122], [79, 92, 160], [25, 126, 89], [230, 216, 173], [43, 248, 240], [52, 151, 218],
    [255, 189, 187], [230, 162, 150], [116, 135, 231], [3, 145, 78], [135, 244, 177], [175, 158, 184],
    [78, 135, 160], [158, 188, 246], [254, 34, 36], [50, 33, 11], [161, 99, 152], [223, 10, 221],
    [204, 146, 108], [108, 107, 249]
];

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[allow(unused)]
impl BoundingBox {
    pub fn new<S>(distance: ArrayBase<S, Ix1>, x: f32, h: f32, stride_w: f32, stride_h: f32) -> Self
    where
        S: ndarray::Data<Elem = f32>,
    {
        let x1 = (x - distance[0]) * stride_w;
        let y1 = (h - distance[1]) * stride_h;
        let x2 = (x + distance[2]) * stride_w;
        let y2 = (h + distance[3]) * stride_h;

        Self { x1, y1, x2, y2 }
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    fn intersection(&self, box2: &BoundingBox) -> f32 {
        (self.x2.min(box2.x2) - self.x1.max(box2.x1))
            * (self.y2.min(box2.y2) - self.y1.max(box2.y1))
    }

    fn union(&self, box2: &BoundingBox) -> f32 {
        ((self.x2 - self.x1) * (self.y2 - self.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
            - self.intersection(box2)
    }
}

#[allow(unused)]
pub struct YoloResult {
    pub class_idx: usize,
    pub score: f32,
    pub bbox: Rc<BoundingBox>,
}

impl Display for YoloResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "class: {}, score: {:.2}, bbox: [({}, {}), ({}, {})]",
            self.label(),
            self.score,
            self.bbox.x1,
            self.bbox.y1,
            self.bbox.x2,
            self.bbox.y2,
        )
    }
}

#[allow(unused)]
impl YoloResult {
    pub fn label(&self) -> &'static str {
        YOLOV8_CLASS_LABELS[self.class_idx]
    }

    pub fn color(&self) -> Rgb<u8> {
        Rgb(RGB_COLORS[self.class_idx])
    }

    pub fn iou(&self, box2: &YoloResult) -> f32 {
        let i = self.bbox.intersection(&box2.bbox);
        let u = self.bbox.union(&box2.bbox);
        i / u
    }

    pub fn scale(&self, scale: ImageScale) -> BoundingBox {
        let x1 = self.bbox.x1 / scale.w as f32;
        let y1 = self.bbox.y1 / scale.h as f32;
        let x2 = self.bbox.x2 / scale.w as f32;
        let y2 = self.bbox.y2 / scale.h as f32;
        BoundingBox { x1, y1, x2, y2 }
    }
}

fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn apply_nms(result: &mut Vec<YoloResult>, mut boxes: Vec<Option<YoloResult>>, nms_threshold: f32) {
    for i in 0..boxes.len() {
        if let Some(b) = boxes[i].take() {
            for j in (i + 1)..boxes.len() {
                if let Some(b2) = boxes[j].as_ref() {
                    if b.iou(b2) > nms_threshold {
                        boxes[j].take();
                    }
                }
            }
            result.push(b);
        }
    }
}

pub fn amd_yolov8m_post_prcess(
    outputs: SessionOutputs<'_, '_>,
    batch_size: usize,
    conf_desigmoid: f32,
    max_boxes: usize,
    nms_threshold: f32,
    max_nms_num: usize,
) -> ort::Result<Vec<Vec<YoloResult>>> {
    // outputs shape: [[batch_size, 80, 80, 144] [batch_size, 40, 40, 144] [batch_size, 20, 20, 144]]
    let distance_conv_kernel_val = Array1::linspace(0.0, 15.0, 16);
    let distance_conv_kernel = distance_conv_kernel_val.to_shape((16, 1)).unwrap();
    let mut all_batch_results: Vec<Vec<YoloResult>> = Vec::new();
    for batch_idx in 0..batch_size {
        let mut result: Vec<YoloResult> = Vec::new();
        for (_, output) in &outputs {
            let output = output.try_extract_tensor::<f32>()?.into_owned();
            let output = output.slice(s![batch_idx, .., .., ..]);
            let output_shape = output.shape();
            let height = output_shape[0];
            let width = output_shape[1];
            let channels = output_shape[2];
            if channels != 144 {
                continue;
            }
            let stride_w = 640.0 / width as f32;
            let stride_h = 640.0 / height as f32;

            let mut boxes: Vec<YoloResult> = Vec::new();

            for w in 0..width {
                for h in 0..height {
                    let pre_output_unit = output
                        .slice(s![h, w, 0..64])
                        .to_shape((4, 16))
                        .unwrap()
                        .softmax(Axis(1)); // shape: [4, 16]
                    let distance = pre_output_unit.dot(&distance_conv_kernel); // shape: [4]
                    let bbox = Rc::new(BoundingBox::new(
                        distance.to_shape((4,)).unwrap(),
                        w as f32 + 0.5,
                        h as f32 + 0.5,
                        stride_w,
                        stride_h,
                    ));
                    // println!("distance: {:?} bbox: {:?}", distance.t(), bbox);

                    let scores = output.slice(s![h, w, 64..]);
                    scores
                        .iter()
                        .map(sigmoid)
                        .enumerate()
                        .filter(|(_, s)| *s > conf_desigmoid)
                        .for_each(|(class_idx, score)| {
                            boxes.push(YoloResult {
                                class_idx,
                                score,
                                bbox: bbox.clone(),
                            });
                        });
                }
            }
            boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            if boxes.len() > max_boxes {
                boxes.truncate(max_boxes);
            }

            // println!("boxes num: {:?}", boxes.len());
            let mut boxes_for_nms: [Vec<Option<YoloResult>>; 80] = [const { Vec::new() }; 80];
            for box_info in boxes {
                boxes_for_nms[box_info.class_idx].push(Some(box_info));
            }

            for boxes in boxes_for_nms {
                apply_nms(&mut result, boxes, nms_threshold);
            }
        }
        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        if result.len() > max_nms_num {
            result.truncate(max_nms_num);
        }
        all_batch_results.push(result);
    }

    Ok(all_batch_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_parallel_softmax() {
        let data = Array1::linspace(0.0, 63.0, 64);

        let chunks = data.to_shape((4, 16)).unwrap();

        println!("chunks: {:?}", chunks);

        let results = chunks.softmax(Axis(1));

        for result in results.axis_iter(Axis(0)) {
            println!("result sum: {:?}", result.sum());
        }
    }
}
