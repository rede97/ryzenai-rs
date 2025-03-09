use std::{
    ffi::c_void, fmt::{Display, Formatter}, mem::ManuallyDrop, sync::Arc
};

use anyhow::anyhow;
use sdl3::{get_error, pixels::PixelFormat, surface::Surface};
use sdl3_sys::{
    camera::{self, SDL_CameraSpec},
    init::{SDL_INIT_CAMERA, SDL_InitSubSystem, SDL_QuitSubSystem},
    stdinc::SDL_free,
};

pub use sdl3_sys::camera::SDL_CameraID;

pub fn subsystem_init() {
    unsafe {
        SDL_InitSubSystem(SDL_INIT_CAMERA);
    }
}

pub fn subsystem_deinit() {
    unsafe {
        SDL_QuitSubSystem(SDL_INIT_CAMERA);
    }
}

pub struct CamerasIdIter {
    idx: usize,
    cnt: usize,
    devices: *mut camera::SDL_CameraID,
}

impl<'a> CamerasIdIter {
    pub fn new() -> Result<CamerasIdIter, sdl3::Error> {
        unsafe {
            let mut cam_cnt = 0;
            let devices: *mut camera::SDL_CameraID = camera::SDL_GetCameras(&mut cam_cnt);
            if devices.is_null() {
                return Err(get_error());
            }
            return Ok(CamerasIdIter {
                idx: 0,
                cnt: cam_cnt as usize,
                devices,
            });
        }
    }

    pub fn len(&self) -> usize {
        return self.cnt;
    }
}

impl Drop for CamerasIdIter {
    fn drop(&mut self) {
        unsafe {
            SDL_free(self.devices as *mut c_void);
        }
    }
}

impl Iterator for CamerasIdIter {
    type Item = camera::SDL_CameraID;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.cnt {
            let cam_id: SDL_CameraID = unsafe { *self.devices.add(self.idx) };
            self.idx += 1;
            if cam_id != 0 {
                return Some(cam_id);
            }
        }
        return None;
    }
}

pub struct CameraFormat<'a> {
    cam_id: camera::SDL_CameraID,
    raw: *const SDL_CameraSpec,
    markder: std::marker::PhantomData<&'a ()>,
}

impl<'a> CameraFormat<'a> {
    pub fn pixel_format(&self) -> PixelFormat {
        return unsafe { sdl3::pixels::PixelFormat::from_ll((*self.raw).format) };
    }

    pub fn width(&self) -> u32 {
        return unsafe { (*self.raw).width as u32 };
    }

    pub fn height(&self) -> u32 {
        return unsafe { (*self.raw).height as u32 };
    }

    pub fn fps(&self) -> f32 {
        return unsafe {
            (*self.raw).framerate_numerator as f32 / (*self.raw).framerate_denominator as f32
        };
    }

    pub fn open_camera(&self) -> Result<Camera, sdl3::Error> {
        unsafe {
            let camera = camera::SDL_OpenCamera(self.cam_id, self.raw);
            if camera.is_null() {
                return Err(get_error());
            }
            return Ok(Camera { raw: camera });
        }
    }
}

impl<'a> Display for CameraFormat<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}, size: {}x{}, fps: {:.2}",
            unsafe { (*self.raw).format },
            self.width(),
            self.height(),
            self.fps()
        )
    }
}

pub struct CameraFormatsIter<'a> {
    idx: usize,
    cnt: usize,
    formats: *mut *mut camera::SDL_CameraSpec,
    cam_id: camera::SDL_CameraID,
    markder: std::marker::PhantomData<&'a ()>,
}

impl<'a> CameraFormatsIter<'a> {
    pub fn len(&self) -> usize {
        return self.cnt;
    }
}

impl<'a> Drop for CameraFormatsIter<'a> {
    fn drop(&mut self) {
        unsafe {
            SDL_free(self.formats as *mut c_void);
        }
    }
}

impl<'a> Iterator for CameraFormatsIter<'a> {
    type Item = CameraFormat<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.cnt {
            let ptr_format: *const SDL_CameraSpec = unsafe { *self.formats.add(self.idx) };
            self.idx += 1;
            if !ptr_format.is_null() {
                return Some(CameraFormat {
                    cam_id: self.cam_id,
                    raw: ptr_format,
                    markder: Default::default(),
                });
            }
        }
        return None;
    }
}

pub trait CameraIdExt<'a> {
    fn name(&self) -> String;
    fn open(&self) -> Result<Camera, sdl3::Error>;
    fn supported_formats(&self) -> Result<CameraFormatsIter<'a>, sdl3::Error>;
}

impl<'a> CameraIdExt<'a> for SDL_CameraID {
    fn name(&self) -> String {
        unsafe {
            let name = camera::SDL_GetCameraName(*self);
            if name.is_null() {
                return String::new();
            }
            let c_str = std::ffi::CStr::from_ptr(name);
            c_str.to_string_lossy().into_owned()
        }
    }

    fn open(&self) -> Result<Camera, sdl3::Error> {
        unsafe {
            let camera = camera::SDL_OpenCamera(*self, std::ptr::null());
            if camera.is_null() {
                return Err(get_error());
            }
            return Ok(Camera { raw: camera });
        }
    }

    fn supported_formats(&self) -> Result<CameraFormatsIter<'a>, sdl3::Error> {
        unsafe {
            let mut formats_cnt: i32 = 0;
            let formats = camera::SDL_GetCameraSupportedFormats(*self, &mut formats_cnt);
            if formats.is_null() {
                return Err(get_error());
            }
            return Ok(CameraFormatsIter {
                idx: 0,
                cnt: formats_cnt as usize,
                formats,
                cam_id: *self,
                markder: Default::default(),
            });
        }
    }
}

pub struct Camera {
    raw: *mut camera::SDL_Camera,
}

impl Camera {
    pub fn acquire_frame<'a>(&'a mut self) -> Option<CameraFrame<'a>> {
        let mut timestamp: u64 = 0;
        unsafe {
            let frame = camera::SDL_AcquireCameraFrame(self.raw, &mut timestamp);
            if frame.is_null() {
                return None;
            }
            return Some(CameraFrame {
                camera: self,
                surface: ManuallyDrop::new(Surface::from_ll(frame)),
                timestamp,
            });
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        unsafe {
            camera::SDL_CloseCamera(self.raw);
        }
    }
}

pub struct CameraFrame<'a> {
    camera: &'a Camera,
    surface: ManuallyDrop<Surface<'a>>,
    pub timestamp: u64,
}

impl<'a> CameraFrame<'a> {
    pub fn surface(&self) -> &Surface<'a> {
        return &self.surface;
    }
}

impl<'a> Drop for CameraFrame<'a> {
    fn drop(&mut self) {
        let cnt = Arc::strong_count(&self.surface.context());
        if cnt > 2 {
            panic!(
                "Invalid CameraFrame Surface state, the Surface of CameraFrame may be cloned, ptr cnt: {}",
                cnt
            );
        }
        unsafe {
            // DO NOT DORP Surface BY DEFAULT
            // ManuallyDrop::drop(&mut self.surface);
            camera::SDL_ReleaseCameraFrame(self.camera.raw, self.surface.raw());
        }
    }
}

pub fn print_list_all_cameras() {
    sdl3::init().unwrap();
    {
        subsystem_init();
        let cameras_iter = CamerasIdIter::new().unwrap();
        if cameras_iter.len() == 0 {
            println!("No camera found!");
        }
        for (idx, cam_id) in cameras_iter.enumerate() {
            println!("id[{}]: {}", idx, cam_id.name());
            for (fid, fmt) in cam_id.supported_formats().unwrap().enumerate() {
                println!("  +[{}]: {}", fid, fmt);
            }
        }
        subsystem_deinit();
    }
}

fn select_format(
    cam_id: SDL_CameraID,
    select_fid: Option<usize>,
) -> anyhow::Result<(Camera, u32, u32)> {
    for (fid, fmt) in cam_id.supported_formats().unwrap().enumerate() {
        if let Some(select_fid) = select_fid {
            if fid == select_fid {
                println!("select camera: {}, {}", cam_id.name(), fmt);
                return Ok((fmt.open_camera()?, fmt.width(), fmt.height()));
            }
        } else {
            return Ok((fmt.open_camera()?, fmt.width(), fmt.height()));
        }
    }

    return Err(anyhow!("No camera format found!"));
}

pub fn select_camera(
    select_dev_idx: Option<usize>,
    select_fid: Option<usize>,
) -> anyhow::Result<(Camera, u32, u32)> {
    let cameras_iter = CamerasIdIter::new().unwrap();
    for (idx, cam_id) in cameras_iter.enumerate() {
        if let Some(select_dev_idx) = select_dev_idx {
            if idx == select_dev_idx {
                return select_format(cam_id, select_fid);
            }
        } else {
            return select_format(cam_id, None);
        }
    }
    return Err(anyhow!("No camera found!"));
}
