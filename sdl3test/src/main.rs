extern crate sdl3;

#[allow(unused)]
mod camera;

use anyhow::{Result, anyhow};
use camera::{Camera, CameraIdExt, CamerasIdIter};
use sdl3::event::Event;
use sdl3::get_error;
use sdl3::keyboard::Keycode;
use sdl3::pixels::Color;
use sdl3::render::Texture;
use sdl3_sys::camera::SDL_CameraID;
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

pub fn select_camera(select_cam_id: SDL_CameraID, select_fid: usize) -> Result<Camera> {
    let cameras_iter = CamerasIdIter::new().unwrap();
    for cam_id in cameras_iter {
        if cam_id == select_cam_id {
            for (fid, fmt) in cam_id.supported_formats().unwrap().enumerate() {
                if fid == select_fid {
                    println!("select camera: {}, {}", cam_id.name(), fmt);
                    return Ok(fmt.open_camera()?);
                }
            }
        }
    }
    return Err(anyhow!("No camera found!"));
}

pub fn main() {
    {
        let sdl_context = sdl3::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();

        camera::subsystem_init();

        print_list_all_cameras();
        let mut cam = select_camera(3, 9).unwrap();

        let mut window = video_subsystem
            .window("rust-sdl3 demo", 800, 600)
            .position_centered()
            .build()
            .unwrap();

        let mut canvas = window.clone().into_canvas();
        let tetxure_creator = canvas.texture_creator();
        let mut texture: Option<Texture> = None;

        canvas.set_draw_color(Color::RGB(0, 255, 255));
        canvas.clear();
        canvas.present();

        let mut event_pump = sdl_context.event_pump().unwrap();
        let mut i = 0;
        'running: loop {
            i = (i + 1) % 255;
            canvas.set_draw_color(Color::RGB(i, 64, 255 - i));
            canvas.clear();

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
                if texture.is_none() {
                    window.set_size(surface.width(), surface.height()).unwrap();
                    let new_texture: Texture<'_> = tetxure_creator
                        .create_texture(
                            surface.pixel_format(),
                            sdl3::render::TextureAccess::Streaming,
                            surface.width(),
                            surface.height(),
                        )
                        .unwrap();
                    texture = Some(new_texture);
                };
                if let Some(texture) = texture.as_mut() {
                    surface.with_lock(|pixels| {
                        texture
                            .update(None, pixels, surface.pitch() as usize)
                            .unwrap();
                    })
                }
            }

            if let Some(texture) = &texture {
                canvas.copy(texture, None, None).unwrap();
            }

            canvas.present();
            ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        }

        camera::subsystem_deinit();
    }

    println!("{}", get_error());
}
