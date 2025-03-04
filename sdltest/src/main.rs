use anyhow::{anyhow, Result};

fn main() -> Result<()> {
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let _window = video_subsystem
        .window("Yolov8 on RyzenAI", 800, 600)
        .opengl() // this line DOES NOT enable opengl, but allows you to create/get an OpenGL context from your window.
        .build()
        .unwrap();

    // let mut canvas = window
    //     .into_canvas()
    //     .accelerated()
    //     // .present_vsync()
    //     .build()
    //     .map_err(|e| e.to_string())
    //     .map_err(|e| anyhow!("Create canvase error: {}", e))?;
    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow!("Create EventPump: {}", e))?;

    'mainloop: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Option::Some(Keycode::Escape),
                    ..
                } => break 'mainloop,

                _ => {}
            }
        }
    }
    Ok(())
}
