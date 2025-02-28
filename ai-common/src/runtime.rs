use std::path::PathBuf;
use walkdir::WalkDir;
use windows::Win32::System::LibraryLoader::SetDllDirectoryA;
use windows::core::PCSTR;

pub fn init_runtime(p: Option<String>) -> PathBuf {
    let (runtime_install_path, mut runtime_path) = match p {
        Some(p) => (p.clone(), p),
        None => {
            if let Ok(ryzen_ai_path) = std::env::var("RYZEN_AI_INSTALLATION_PATH") {
                let mut runtime_path = ryzen_ai_path.clone();
                runtime_path.push_str("/onnxruntime/bin/");
                (ryzen_ai_path, runtime_path)
            } else {
                let default_path = "../runtime/".to_string();
                (default_path.clone(), default_path)
            }
        }
    };
    runtime_path.push('\0');
    unsafe {
        let path = PCSTR::from_raw(runtime_path.as_ptr());
        if SetDllDirectoryA(path).is_ok() {
            println!("DLL search path set successfully.");
        } else {
            println!("Failed to set DLL search path.");
        }
    }
    return PathBuf::from(runtime_install_path);
}

pub fn find_config_file(runtime_path: PathBuf, config_json: &str) -> std::io::Result<PathBuf> {
    let config_path = runtime_path.join(config_json);
    if config_path.exists() {
        return Ok(config_path);
    }
    for entry in WalkDir::new(runtime_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == config_json {
            return Ok(entry.path().to_path_buf());
        }
    }
    return Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Config file not found",
    ));
}
