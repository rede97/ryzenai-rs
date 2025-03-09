use std::path::PathBuf;
use walkdir::WalkDir;

pub fn find_config_file(
    config_json: &str,
    runtime_path: Option<PathBuf>,
) -> std::io::Result<PathBuf> {
    if let Some(runtime_path) = runtime_path
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        })
        .or_else(|| {
            std::env::var("RYZEN_AI_INSTALLATION_PATH")
                .map(|p| PathBuf::from(p))
                .ok()
        })
    {
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
    }
    return Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Config file not found",
    ));
}
