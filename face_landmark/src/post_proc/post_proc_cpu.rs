use anyhow::Result;
use ndarray::{Array1, ArrayBase, Axis, s};
use ort::session::SessionOutputs;
use ort::tensor::ArrayExtensions;
use rayon::prelude::*;

use super::{AMDYoloV8PostProc, YoloResult};

#[derive(Default)]
pub struct PostProcCPU {}

#[allow(unused)]
impl PostProcCPU {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AMDYoloV8PostProc for PostProcCPU {
    fn post_proc(
        &self,
        outputs: SessionOutputs<'_, '_>,
        batch_size: usize,
        conf_desigmoid: f32,
        max_boxes: usize,
        nms_threshold: f32,
        max_nms_num: usize,
    ) -> Result<Vec<Vec<YoloResult>>> {
        // outputs shape: [[batch_size, 80, 80, 144] [batch_size, 40, 40, 144] [batch_size, 20, 20, 144]]
        let distance_conv_kernel_val = Array1::linspace(0.0, 15.0, 16);
        let distance_conv_kernel = distance_conv_kernel_val.to_shape((16, 1)).unwrap();
        let mut all_batch_results: Vec<Vec<YoloResult>> = Vec::new();
        for batch_idx in 0..batch_size {
            let mut result: Vec<YoloResult> = Vec::new();
            for (_, output) in &outputs {
                let output = output.try_extract_tensor::<f32>()?.into_owned();
                let output: ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 3]>> =
                    output.slice(s![batch_idx, .., .., ..]);
                let output_shape = output.shape();
                let height = output_shape[0];
                let width = output_shape[1];
                let channels = output_shape[2];
                if channels != 144 {
                    continue;
                }

                let stride_w = 640.0 / width as f32;
                let stride_h = 640.0 / height as f32;

                let mut boxes: Vec<YoloResult> = (0..width)
                    .into_par_iter()
                    .map(|w| {
                        let mut w_boxes = Vec::new();
                        for h in 0..height {
                            let pre_output_unit = output
                                .slice(s![h, w, 0..64])
                                .to_shape((4, 16))
                                .unwrap()
                                .softmax(Axis(1)); // shape: [4, 16]
                            let distance = pre_output_unit.dot(&distance_conv_kernel); // shape: [4]
                            let bbox = super::BoundingBox::new(
                                distance.to_shape((4,)).unwrap(),
                                w as f32 + 0.5,
                                h as f32 + 0.5,
                                stride_w,
                                stride_h,
                            );

                            let scores = output.slice(s![h, w, 64..]);
                            scores
                                .iter()
                                .map(super::sigmoid)
                                .enumerate()
                                .filter(|(_, s)| *s > conf_desigmoid)
                                .for_each(|(class_idx, score)| {
                                    w_boxes.push(YoloResult {
                                        class_idx,
                                        score,
                                        bbox,
                                    });
                                });
                        }
                        w_boxes
                    })
                    .flatten()
                    .collect();

                boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                if boxes.len() > max_boxes {
                    boxes.truncate(max_boxes);
                }

                let mut boxes_for_nms: [Vec<Option<YoloResult>>; 80] = [const { Vec::new() }; 80];
                for box_info in boxes {
                    boxes_for_nms[box_info.class_idx].push(Some(box_info));
                }
                
                for boxes in boxes_for_nms {
                    YoloResult::apply_nms(&mut result, boxes, nms_threshold);
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
}
