use crate::camera_sdl3;
use crate::camera_sdl3::{Camera, CameraIdExt, CamerasIdIter};

use anyhow::{Result, anyhow};
use sdl3::event::Event;
use sdl3::get_error;
use sdl3::keyboard::Keycode;
use sdl3::pixels::Color;
use sdl3::rect::Rect;
use sdl3::render::Texture;
use sdl3_sys::camera::SDL_CameraID;
use sdl3_sys::pixels::SDL_PIXELFORMAT_RGB24;
use sdl3_sys::timer;
use std::process::exit;
use std::sync::Arc;
use std::time::Duration;

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

pub fn camera_task() {
    let sdl_context = sdl3::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    camera_sdl3::subsystem_init();
    {
        let mut window = video_subsystem
            .window("Yolov8m", 640, 640)
            .position_centered()
            .build()
            .unwrap();

        let mut canvas = window.clone().into_canvas();
        unsafe {
            sdl3_sys::render::SDL_SetRenderVSync(canvas.raw(), 1);
        }

        let tetxure_creator = canvas.texture_creator();
        let mut cam_raw_texture: Option<Texture> = None;
        let mut cam_640_texture: Texture = unsafe {
            let pixel_format = sdl3::pixels::PixelFormat::from_ll(SDL_PIXELFORMAT_RGB24);
            tetxure_creator
                .create_texture_target(pixel_format, 640, 640)
                .unwrap()
        };

        print_list_all_cameras();
        let (mut cam, cam_w, cam_h) = select_camera(3, 8).unwrap();
        let min_cam_border = std::cmp::min(cam_h, cam_w);
        let src_rect = Rect::new(
            (cam_w - min_cam_border) as i32 / 2,
            (cam_h - min_cam_border) as i32 / 2,
            min_cam_border,
            min_cam_border,
        );

        let mut frame_count = 0;
        let mut start_time = std::time::Instant::now();
        let mut fps = 0.0;
        let mut event_pump = sdl_context.event_pump().unwrap();
        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => break 'running,
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
                    canvas.with_texture_canvas(&mut cam_640_texture, |c| {
                        c.copy(cam_raw_texture, src_rect, None).unwrap();
                    });
                }
            }

            canvas.copy(&cam_640_texture, None, None).unwrap();
            canvas.present();

            {
                // FPS
                frame_count += 1;
                let elapsed_time = start_time.elapsed();
                if elapsed_time.as_secs() >= 1 {
                    fps = frame_count as f64 / elapsed_time.as_secs_f64();
                    window
                        .set_title(&format!("Yolov8m FPS: {:.1}", fps))
                        .unwrap();
                    frame_count = 0;
                    start_time = std::time::Instant::now();
                }
            }
        }
    }
    camera_sdl3::subsystem_deinit();
}
