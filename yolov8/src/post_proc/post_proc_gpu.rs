use std::num::NonZeroU64;

use anyhow::{Result, anyhow};
use bytemuck::{AnyBitPattern, NoUninit, Zeroable};
use ndarray::{ArrayBase, s};

use pollster::FutureExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use wgpu::util::DeviceExt;

use crate::post_proc::{self, BoundingBox};

use super::{AMDYoloV8PostProc, YoloResult};

#[repr(C)]
#[derive(Clone, Copy)]
struct YoloInSlice {
    relative: [f32; 64],
    scores: [f32; 80],
}
unsafe impl NoUninit for YoloInSlice {}

impl Default for YoloInSlice {
    fn default() -> Self {
        return Self {
            relative: [0f32; 64],
            scores: [0f32; 80],
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable)]
struct YoloOutSlice {
    bbox: BoundingBox,
    scores: [f32; 80],
}
unsafe impl AnyBitPattern for YoloOutSlice {}

#[repr(C)]
#[derive(Clone, Copy)]
struct YoloInCfg {
    step: u32,
    width: u32,
    stride_w: f32,
    stride_h: f32,
}
unsafe impl NoUninit for YoloInCfg {}

impl Default for YoloInCfg {
    fn default() -> Self {
        return Self {
            step: 8,
            width: 80,
            stride_w: 8.0,
            stride_h: 8.0,
        };
    }
}

const MAX_YOLO_SIZE: usize = 80 * 80;
const SHADER_DRC: &str = include_str!("../../asserts/post_proc.wgsl");
const SHADER_INPUT_SIZE: u64 = (size_of::<YoloInSlice>() * MAX_YOLO_SIZE) as u64;
const SHADER_OUTPUT_SIZE: u64 = (size_of::<YoloOutSlice>() * MAX_YOLO_SIZE) as u64;

pub struct PostProcWGPU {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    yolo_cfg: wgpu::Buffer,
    yolo_in: wgpu::Buffer,
    yolo_out: wgpu::Buffer,
    yolo_result: wgpu::Buffer,
}

impl PostProcWGPU {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = if let Some(adapter) = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .filter(|adapter| {
                adapter.get_info().device_type != wgpu::DeviceType::Other
                    && adapter
                        .features()
                        .contains(wgpu::Features::VERTEX_WRITABLE_STORAGE)
                    && adapter.limits().max_buffer_size > SHADER_INPUT_SIZE
            })
            .next()
        {
            adapter
        } else {
            return Err(anyhow!("Failed to create device"));
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .block_on()?;

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_DRC.into()),
        });

        let mut empty = Vec::with_capacity(MAX_YOLO_SIZE);
        empty.resize(MAX_YOLO_SIZE, YoloInSlice::default());

        let yolo_in: wgpu::Buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU-Side Input Buffer"),
            contents: bytemuck::cast_slice(&empty),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let yolo_out: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU-Side Output Buffer"),
            size: SHADER_OUTPUT_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let yolo_cfg: wgpu::Buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Config"),
            contents: bytemuck::cast_slice(&[YoloInCfg::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let yolo_result: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU-Side Download Buffer"),
            size: SHADER_OUTPUT_SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<YoloInSlice>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<YoloOutSlice>() as u64).unwrap(),
                        ),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<YoloInCfg>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: yolo_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: yolo_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: yolo_cfg.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            yolo_cfg,
            yolo_in,
            yolo_out,
            yolo_result,
        })
    }

    pub fn compute(
        &self,
        ort_output: &ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 3]>>,
        stride_w: f32,
        stride_h: f32,
    ) -> Result<()> {
        let output_size = ort_output.shape();
        let yolo_gpu_cfg = [YoloInCfg {
            step: (output_size[0] * output_size[1] / 200) as u32,
            width: output_size[1] as u32,
            stride_w,
            stride_h,
        }];
        let ort_out_data = ort_output
            .as_slice()
            .ok_or(anyhow!("Write ndarray to GPU failed"))?;
        self.queue
            .write_buffer(&self.yolo_in, 0, bytemuck::cast_slice(ort_out_data));
        self.queue
            .write_buffer(&self.yolo_cfg, 0, bytemuck::cast_slice(&yolo_gpu_cfg));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(200, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.yolo_out,
            0,
            &self.yolo_result,
            0,
            self.yolo_out.size(),
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}

impl AMDYoloV8PostProc for PostProcWGPU {
    fn post_proc(
        &self,
        outputs: ort::session::SessionOutputs<'_, '_>,
        batch_size: usize,
        conf_desigmoid: f32,
        max_boxes: usize,
        nms_threshold: f32,
        max_nms_num: usize,
    ) -> Result<Vec<Vec<post_proc::YoloResult>>> {
        // outputs shape: [[batch_size, 80, 80, 144] [batch_size, 40, 40, 144] [batch_size, 20, 20, 144]]
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

                self.compute(&output, stride_w, stride_h)?;

                let buffer_slice = self.yolo_result.slice(..);
                buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);

                let data = buffer_slice.get_mapped_range();
                let gpu_results: &[YoloOutSlice] = bytemuck::cast_slice(&data);

                let mut boxes: Vec<YoloResult> = (0..width)
                    .into_par_iter()
                    .map(|w| {
                        let mut w_boxes = Vec::new();
                        for h in 0..height {
                            let slice = gpu_results[h * width + w];
                            slice
                                .scores
                                .iter()
                                .map(|x| *x)
                                .enumerate()
                                .filter(|(_, s)| *s > conf_desigmoid)
                                .for_each(|(class_idx, score)| {
                                    w_boxes.push(YoloResult {
                                        class_idx,
                                        score,
                                        bbox: slice.bbox,
                                    });
                                });
                        }
                        w_boxes
                    })
                    .flatten()
                    .collect();
            
                drop(data);
                self.yolo_result.unmap();

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
