extern crate vcpkg;

fn main() {
    #[cfg(target_os = "windows")]
    {
        let lib = vcpkg::Config::new()
            .target_triplet("x64-windows")
            .find_package("sdl3")
            .expect("Could not find SDL3 using vcpkg");

        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
        println!("cargo:rerun-if-env-changed=VCPKGRS_DYNAMIC");

        println!("Found SDL3 using vcpkg!");
        println!("Include paths:");
        for path in &lib.include_paths {
            println!("cargo:include={}", path.display());
        }

        println!("Library paths:");
        for path in lib.link_paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
}
