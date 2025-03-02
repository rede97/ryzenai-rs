use std::fmt::Display;
use std::rc::Rc;

use ndarray::{Array1, ArrayBase, Axis, Ix1, s};
use ort::session::SessionOutputs;
use ort::tensor::ArrayExtensions;

use crate::image::ImageSize;

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

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
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
pub struct BoxInfo {
    pub class_idx: usize,
    pub score: f32,
    pub bbox: Rc<BoundingBox>,
}

impl Display for BoxInfo {
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
impl BoxInfo {
    pub fn label(&self) -> &'static str {
        YOLOV8_CLASS_LABELS[self.class_idx]
    }

    pub fn iou(&self, box2: &BoxInfo) -> f32 {
        let i = self.bbox.intersection(&box2.bbox);
        let u = self.bbox.union(&box2.bbox);
        i / u
    }
}

fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn apply_nms(result: &mut Vec<BoxInfo>, mut boxes: Vec<Option<BoxInfo>>, nms_threshold: f32) {
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
) -> ort::Result<Vec<Vec<BoxInfo>>> {
    // outputs shape: [[batch_size, 80, 80, 144] [batch_size, 40, 40, 144] [batch_size, 20, 20, 144]]
    let distance_conv_kernel_val = Array1::linspace(0.0, 15.0, 16);
    let distance_conv_kernel = distance_conv_kernel_val.to_shape((16, 1)).unwrap();
    let mut all_batch_results: Vec<Vec<BoxInfo>> = Vec::new();
    for batch_idx in 0..batch_size {
        let mut result: Vec<BoxInfo> = Vec::new();
        for (_, output) in &outputs {
            let output = output.try_extract_tensor::<f32>()?.into_owned();
            let output = output.slice(s![batch_idx, .., .., ..]);
            let output_shape = output.shape();
            let width = output_shape[0];
            let height = output_shape[1];
            let channels = output_shape[2];
            if channels != 144 {
                continue;
            }
            let stride_w = 640.0 / width as f32;
            let stride_h = 640.0 / height as f32;
            let mut boxes: Vec<BoxInfo> = Vec::new();

            for w in 0..width {
                for h in 0..height {
                    let pre_output_unit = output
                        .slice(s![w, h, 0..64])
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

                    let scores = output.slice(s![w, h, 64..]);
                    scores
                        .iter()
                        .map(sigmoid)
                        .enumerate()
                        .filter(|(_, s)| *s > conf_desigmoid)
                        .for_each(|(class_idx, score)| {
                            boxes.push(BoxInfo {
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
            let mut boxes_for_nms: [Vec<Option<BoxInfo>>; 80] = [const { Vec::new() }; 80];
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
