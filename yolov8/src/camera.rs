extern crate ffmpeg_next as ffmpeg;
extern crate ffmpeg_sys_next as ffmpeg_sys;

use anyhow::{Result, anyhow};
use ffmpeg::device;
use ffmpeg::ffi::AVInputFormat;

use std::borrow::Cow;
use std::ffi::CStr;
use std::ptr;

pub struct VideoDevice<'a> {
    pub name: Cow<'a, str>,
    pub desc: Cow<'a, str>,
}

pub struct VideoDeviceIter<'a> {
    device_list: *mut ffmpeg_sys::AVDeviceInfoList,
    device_index: isize,
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> VideoDeviceIter<'a> {
    pub fn register_all() {
        // 注册所有输入设备
        device::register_all();
    }

    pub fn new() -> Result<Self> {
        // 获取输入设备格式（例如 v4l2 或 dshow）
        let input_format: *const AVInputFormat = unsafe {
            #[cfg(target_os = "linux")]
            let name = CStr::from_bytes_with_nul(b"v4l2\0").unwrap(); // Linux 使用 v4l2
            #[cfg(target_os = "windows")]
            let name = CStr::from_bytes_with_nul(b"dshow\0").unwrap(); // Windows 使用 dshow
            ffmpeg::ffi::av_find_input_format(name.as_ptr())
        };
        if input_format.is_null() {
            return Err(anyhow!("Failed to find input format"));
        }

        // 列出设备
        let mut device_list: *mut ffmpeg_sys::AVDeviceInfoList = ptr::null_mut();

        unsafe {
            ffmpeg::ffi::avdevice_list_input_sources(
                input_format,
                ptr::null(),
                ptr::null_mut(),
                &mut device_list,
            );
        }

        if device_list.is_null() {
            return Err(anyhow!("No devices found!"));
        }

        return Ok(Self {
            device_list,
            device_index: 0,
            marker: std::marker::PhantomData,
        });
    }
}

impl<'a> Drop for VideoDeviceIter<'a> {
    fn drop(&mut self) {
        unsafe {
            ffmpeg::ffi::avdevice_free_list_devices(&mut self.device_list);
        }
    }
}

impl<'a> Iterator for VideoDeviceIter<'a> {
    type Item = VideoDevice<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let device_info_list = *self.device_list;
            let device_cnt = device_info_list.nb_devices as isize;
            for i in self.device_index..device_cnt {
                let device_ptr = *device_info_list.devices.offset(i);
                if device_ptr.is_null() {
                    return None;
                }
                let device = *device_ptr;
                let device_name = CStr::from_ptr(device.device_name).to_string_lossy();
                let device_desc = CStr::from_ptr(device.device_description).to_string_lossy();
                for j in 0..device.nb_media_types {
                    let device_type: ffmpeg_sys::AVMediaType =
                        *device.media_types.offset(j as isize);
                    if device_type == ffmpeg_sys::AVMediaType::AVMEDIA_TYPE_VIDEO {
                        self.device_index = i + 1;
                        return Some(VideoDevice {
                            name: device_name,
                            desc: device_desc,
                        });
                    }
                }
            }
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_video_device() {
        extern crate ffmpeg_next as ffmpeg;
        extern crate ffmpeg_sys_next as ffmpeg_sys;

        use super::*;
        use ffmpeg::codec::context::Context;
        use ffmpeg::format::open;
        use ffmpeg::util::frame::video::Video;

        // 初始化 FFmpeg
        ffmpeg::init().unwrap();
        VideoDeviceIter::register_all();

        let dshow = ffmpeg::device::input::video()
            .filter(|v| v.name() == "dshow")
            .next()
            .expect("No input device [dshow]");

        let ictx = match VideoDeviceIter::new().unwrap().next() {
            Some(dev) => {
                println!("device: {}, path: {}", dev.desc, dev.name);
                open(&format!("video={}", dev.desc), &dshow).expect("Failed to open camera")
            }
            None => {
                println!("No camera devices");
                return;
            }
        };

        let mut video_input = ictx.input();
        let video_stream = video_input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("No video stream found");
        let video_stream_index = video_stream.index();

        let context_decoder = Context::from_parameters(video_stream.parameters()).unwrap();
        let mut decoder = context_decoder.decoder().video().unwrap();
        println!(
            "decoder output format: {:?} {}x{}",
            decoder.format(),
            decoder.width(),
            decoder.height()
        );

        let mut frame_index = 0;
        for (stream, packet) in video_input.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet).unwrap();
            }
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                println!("Decoded frame {}", frame_index);

                frame_index += 1;
            }
            if frame_index > 5 {
                break;
            }
        }
    }
}
