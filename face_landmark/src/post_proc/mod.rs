use ai_common::image_utils::ImageScale;
use anyhow::Result;
use bytemuck::{AnyBitPattern, Zeroable};
use image::Rgb;
use ndarray::{ArrayBase, Ix1};
use ort::session::SessionOutputs;
use sdl3::{pixels::Color, render::FRect};
use std::fmt::Display;

mod post_proc_cpu;

pub use post_proc_cpu::PostProcCPU;

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}
unsafe impl AnyBitPattern for BoundingBox {}

impl Default for BoundingBox {
    fn default() -> Self {
        return Self {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
        };
    }
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
        (self.x2 - self.x1).round()
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).round()
    }

    fn intersection(&self, box2: &BoundingBox) -> f32 {
        (self.x2.min(box2.x2) - self.x1.max(box2.x1))
            * (self.y2.min(box2.y2) - self.y1.max(box2.y1))
    }

    fn union(&self, box2: &BoundingBox) -> f32 {
        ((self.x2 - self.x1) * (self.y2 - self.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
            - self.intersection(box2)
    }

    pub fn frect(&self) -> FRect {
        FRect::new(
            self.x1.clamp(0.0, 640.0),
            self.y1.clamp(0.0, 640.0),
            self.width().clamp(0.0, 640.0),
            self.height().clamp(0.0, 640.0),
        )
    }
}

#[derive(Debug)]
pub struct YoloResult {
    pub class_idx: usize,
    pub score: f32,
    pub bbox: BoundingBox,
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

    pub fn img_color(&self) -> Rgb<u8> {
        Rgb(RGB_COLORS[self.class_idx])
    }

    pub fn sdl_color(&self) -> sdl3::pixels::Color {
        let color = RGB_COLORS[self.class_idx];
        Color::RGB(color[0], color[1], color[2])
    }

    pub fn iou(&self, box2: &YoloResult) -> f32 {
        let i = self.bbox.intersection(&box2.bbox);
        let u = self.bbox.union(&box2.bbox);
        i / u
    }

    pub fn scale(&self, scale: ImageScale) -> BoundingBox {
        let (width_ratio, height_ratio, w_offset, h_offset) = match scale {
            ImageScale::KeepAspectRatio {
                scale_ratio: s,
                aspect_ratio: a,
            } => {
                if a > 1.0 {
                    (s, s, 0.0, (640.0 - 640.0 / a) / 2.0)
                } else {
                    (s, s, (640.0 - 640.0 * a) / 2.0, 0.0)
                }
            }
            ImageScale::ScaleRatio {
                wdith_ratio,
                height_ratio,
            } => (wdith_ratio, height_ratio, 0.0, 0.0),
        };
        let x1 = (self.bbox.x1 - w_offset) / width_ratio;
        let y1 = (self.bbox.y1 - h_offset) / height_ratio;
        let x2 = (self.bbox.x2 - w_offset) / width_ratio;
        let y2 = (self.bbox.y2 - h_offset) / height_ratio;
        BoundingBox { x1, y1, x2, y2 }
    }

    fn apply_nms(
        result: &mut Vec<YoloResult>,
        mut boxes: Vec<Option<YoloResult>>,
        nms_threshold: f32,
    ) {
        for i in 0..boxes.len() {
            if let Some(b) = boxes[i].take() {
                for j in (i + 1)..boxes.len() {
                    if let Some(b2) = boxes[j].as_ref() {
                        let nms = b.iou(b2);
                        if nms > nms_threshold {
                            boxes[j].take();
                        }
                    }
                }
                result.push(b);
            }
        }
    }
}

fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub trait AMDYoloV8PostProc {
    fn post_proc(
        &self,
        outputs: SessionOutputs<'_, '_>,
        batch_size: usize,
        conf_desigmoid: f32,
        max_boxes: usize,
        nms_threshold: f32,
        max_nms_num: usize,
    ) -> Result<Vec<Vec<YoloResult>>>;
}
