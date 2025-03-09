use sdl3::surface::Surface;
use sdl3_ttf_sys;

pub const FONT_DATA: &[u8] = include_bytes!("../asserts/DejaVuSans.ttf");

pub fn ttf_init() {
    unsafe {
        sdl3_ttf_sys::ttf::TTF_Init();
    }
}

pub fn ttf_deinit() {
    unsafe {
        sdl3_ttf_sys::ttf::TTF_Quit();
    }
}

pub struct Font {
    ttf: *mut sdl3_ttf_sys::ttf::TTF_Font,
}

impl Font {
    pub fn open(ptsize: f32) -> Self {
        let ttf: *mut sdl3_ttf_sys::ttf::TTF_Font = unsafe {
            let const_mem_io = sdl3_sys::iostream::SDL_IOFromConstMem(
                FONT_DATA.as_ptr() as *const _,
                FONT_DATA.len(),
            );
            sdl3_ttf_sys::ttf::TTF_OpenFontIO(const_mem_io, true, ptsize)
        };
        return Self { ttf };
    }

    pub fn render_text_blend<'a>(
        &self,
        text: &str,
        color: sdl3::pixels::Color,
    ) -> Result<Surface, sdl3::Error> {
        unsafe {
            let color = sdl3_sys::pixels::SDL_Color {
                r: color.r,
                g: color.g,
                b: color.b,
                a: color.a,
            };
            let surface_raw = sdl3_ttf_sys::ttf::TTF_RenderText_Blended(
                self.ttf,
                text.as_ptr() as *const _,
                text.len(),
                color,
            );
            if surface_raw.is_null() {
                return Err(sdl3::get_error());
            }
            return Ok(sdl3::surface::Surface::from_ll(surface_raw));
        }
    }
}

impl Drop for Font {
    fn drop(&mut self) {
        unsafe {
            sdl3_ttf_sys::ttf::TTF_CloseFont(self.ttf);
        }
    }
}
