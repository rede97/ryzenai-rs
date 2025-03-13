extern crate vcpkg;

fn find_vcpkg_lib(name: &str) {
    let lib = vcpkg::Config::new()
        .target_triplet("x64-windows")
        .find_package(name)
        .expect(&format!("Could not find {} using vcpkg", name));

    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=VCPKGRS_DYNAMIC");

    println!("Found {} using vcpkg!", name);
    println!("Include paths:");
    for path in &lib.include_paths {
        println!("cargo:include={}", path.display());
    }

    println!("Library paths:");
    for path in lib.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn main() {
    #[cfg(target_env = "msvc")]
    {
        find_vcpkg_lib("sdl3");
        find_vcpkg_lib("sdl3-ttf");
    }
}
