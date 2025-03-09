use std::time::Duration;

use crate::post_proc::{AMDYoloV8PostProc, YoloResult};
use crate::{cli_args, init_model};
use ai_common::camera_sdl3::{self, Camera, CameraIdExt, CamerasIdIter};
use ai_common::ttf_sdl3::Font;
use ai_common::{image_utils, measure_time, ttf_sdl3};
use anyhow::{Result, anyhow};
use log::info;
use ndarray::s;
use ndarray::{ArrayView, Axis};
use ort::inputs;
use ort::session::SessionOutputs;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::rect::Rect;
use sdl3::render::{Texture, TextureCreator};
use sdl3_sys::camera::SDL_CameraID;

struct FontCache<'a> {
    font: Font,
    textures: [Option<Texture<'a>>; 80],
}

impl<'a> FontCache<'a> {
    pub fn new() -> Self {
        let font = ttf_sdl3::Font::open(20.0);
        return Self {
            font,
            textures: [const { None }; 80],
        };
    }

    pub fn render_text<T, R, F: FnOnce(&Texture<'a>) -> R>(
        &mut self,
        creator: &'a TextureCreator<T>,
        r: &YoloResult,
        func: F,
    ) -> Result<R> {
        if let Some(texture) = &self.textures[r.class_idx] {
            return Ok(func(texture));
        } else {
            let text = self
                .font
                .render_text_blend(r.label(), r.sdl_color())
                .unwrap();
            let texture = creator.create_texture_from_surface(text)?;
            let ret = func(&texture);
            self.textures[r.class_idx] = Some(texture);
            return Ok(ret);
        }
    }
}

pub fn print_list_all_cameras() {
    let cameras_iter = CamerasIdIter::new().unwrap();
    if cameras_iter.len() == 0 {
        println!("No camera found!");
    }
    for cam_id in cameras_iter {
        println!("id[{}]: {}", cam_id, cam_id.name());
        for (fid, fmt) in cam_id.supported_formats().unwrap().enumerate() {
            println!("  +[{}]: {}", fid, fmt);
        }
    }
}

pub fn select_camera(select_cam_id: SDL_CameraID, select_fid: usize) -> Result<(Camera, u32, u32)> {
    let cameras_iter = CamerasIdIter::new().unwrap();
    for cam_id in cameras_iter {
        if cam_id == select_cam_id {
            for (fid, fmt) in cam_id.supported_formats().unwrap().enumerate() {
                if fid == select_fid {
                    println!("select camera: {}, {}", cam_id.name(), fmt);
                    return Ok((fmt.open_camera()?, fmt.width(), fmt.height()));
                }
            }
        }
    }
    return Err(anyhow!("No camera found!"));
}

fn surface_to_ndarray(surface: &sdl3::surface::Surface) -> ndarray::Array4<f32> {
    let width = surface.width() as usize;
    let height = surface.height() as usize;
    surface.with_lock(|pixels| {
        use ndarray::ShapeBuilder;
        let a = ArrayView::from_shape(
            (1, height, width, 3).strides((pixels.len(), width * 4, 4, 1)),
            &pixels[0..],
        )
        .unwrap();
        // BGR to RGB, u8 to float
        return a.slice(s![.., .., .., ..;-1]).map(|v| *v as f32 / 255.0);
    })
}

pub fn camera_task(args: &cli_args::Args, proc: Box<dyn AMDYoloV8PostProc>) -> Result<()> {
    let model = init_model(args)?;

    let sdl_context = sdl3::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    camera_sdl3::subsystem_init();
    ttf_sdl3::ttf_init();
    {
        print_list_all_cameras();
        let (mut cam, cam_w, cam_h) = select_camera(3, 8).unwrap();
        let min_cam_border = std::cmp::min(cam_h, cam_w);
        let src_rect = Rect::new(
            (cam_w - min_cam_border) as i32 / 2,
            (cam_h - min_cam_border) as i32 / 2,
            min_cam_border,
            min_cam_border,
        );

        let mut window = video_subsystem
            .window("Yolov8m", 640, 640)
            .position_centered()
            // .opengl()
            .build()
            .unwrap();

        let mut canvas = window.clone().into_canvas();
        unsafe {
            sdl3_sys::render::SDL_SetRenderVSync(canvas.raw(), 1);
        }

        let mut infer_duration = Duration::default();
        let tetxure_creator = canvas.texture_creator();
        let mut font = FontCache::new();
        let mut cam_raw_texture: Option<Texture> = None;

        let mut frame_count = 0;
        let mut start_time = std::time::Instant::now();
        let mut event_pump = sdl_context.event_pump().unwrap();
        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => break 'running,
                    Event::KeyUp {
                        keycode: Some(Keycode::P),
                        ..
                    } => {
                        if let Ok(surface) = canvas.read_pixels(None) {
                            let (arr, duration) = measure_time!({ surface_to_ndarray(&surface) });
                            info!(
                                "surface_to_ndarray duration: {}ms",
                                duration.as_micros() as f32 / 1000.0
                            );
                            image_utils::save_ndarray_as_png(
                                arr.index_axis(Axis(0), 0).view(),
                                "print_screen.png",
                            )
                            .unwrap();
                            info!("save  print_screen.png");
                        }
                    }
                    _ => {}
                }
            }
            if let Some(frame) = cam.acquire_frame() {
                let surface = frame.surface();
                if cam_raw_texture.is_none() {
                    let new_texture: Texture<'_> = tetxure_creator
                        .create_texture(
                            surface.pixel_format(),
                            sdl3::render::TextureAccess::Streaming,
                            surface.width(),
                            surface.height(),
                        )
                        .unwrap();
                    cam_raw_texture = Some(new_texture);
                };
                if let Some(cam_raw_texture) = cam_raw_texture.as_mut() {
                    surface.with_lock(|pixels| {
                        cam_raw_texture
                            .update(None, pixels, surface.pitch() as usize)
                            .unwrap();
                    });
                }
            }
            if let Some(texture) = cam_raw_texture.as_ref() {
                canvas.copy(&texture, src_rect, None).unwrap();
                let (results, duration) = measure_time!({
                    canvas.read_pixels(None).map(|surface| {
                        let img = surface_to_ndarray(&surface);
                        let outputs: SessionOutputs<'_, '_> =
                            model.run(inputs![img.view()].unwrap()).unwrap();
                        proc.post_proc(outputs, 1, 0.5, 100, 0.7, 100).unwrap()
                    })
                });
                infer_duration = duration;
                if let Ok(results) = results {
                    for batch_results in results {
                        for result in batch_results {
                            let color = result.sdl_color();
                            canvas.set_draw_color(color);
                            let frect = result.bbox.frect();
                            canvas.draw_rect(frect).unwrap();
                            font.render_text(&tetxure_creator, &result, |text| {
                                let r = Rect::new(
                                    frect.x as i32,
                                    frect.y as i32,
                                    text.width(),
                                    text.height(),
                                );
                                canvas.copy(text, None, r).unwrap();
                            })
                            .unwrap();
                        }
                    }
                    canvas.present();
                }
            }

            #[cfg(feature = "fps")]
            {
                // FPS
                frame_count += 1;
                let elapsed_time = start_time.elapsed();
                if elapsed_time.as_secs() >= 1 {
                    let fps = frame_count as f64 / elapsed_time.as_secs_f64();
                    window
                        .set_title(&format!(
                            "Yolov8m FPS: {:.1} infra: {:.2}ms",
                            fps,
                            infer_duration.as_micros() as f32 / 1000.0
                        ))
                        .unwrap();
                    frame_count = 0;
                    start_time = std::time::Instant::now();
                }
            }
        }
    }
    camera_sdl3::subsystem_deinit();
    ttf_sdl3::ttf_deinit();
    Ok(())
}
