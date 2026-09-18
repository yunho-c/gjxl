#[cfg(feature = "native")]
use std::env;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GJXL_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_COMPILER");

    #[cfg(feature = "native")]
    build_native();
}

#[cfg(feature = "native")]
fn build_native() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let enable_metal = target_os == "macos";
    let enable_cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    if enable_cuda && target_os == "macos" {
        panic!("gjxl-sys: the cuda feature is not supported on macOS");
    }

    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    if host != target {
        panic!(
            "gjxl-sys: cross-compiling GJXL is not supported; host is {host}, target is {target}"
        );
    }

    let source_dir = source_dir();
    validate_source(&source_dir);
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("include/gjxl/gjxl.h").display()
    );

    let mut native = cmake::Config::new(&source_dir);
    native
        .profile("Release")
        .define("GJXL_BUILD_TESTS", "OFF")
        .define("GJXL_BUILD_BENCHMARKS", "OFF")
        .define("GJXL_ENABLE_LIBJXL_REFERENCE", "OFF")
        .define("GJXL_ENABLE_METAL", if enable_metal { "ON" } else { "OFF" })
        .define("GJXL_ENABLE_CUDA", if enable_cuda { "ON" } else { "OFF" });
    let install = native.build();

    let header = install.join("include/gjxl/gjxl.h");
    if !header.is_file() {
        panic!(
            "gjxl-sys: GJXL install did not produce {}",
            header.display()
        );
    }

    generate_bindings(&header);
    emit_link_directives(&install, &target_os, enable_metal, enable_cuda);
}

#[cfg(feature = "native")]
fn source_dir() -> PathBuf {
    env::var_os("GJXL_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must provide CARGO_MANIFEST_DIR"),
            )
            .join("../..")
        })
}

#[cfg(feature = "native")]
fn validate_source(source_dir: &Path) {
    if !source_dir.join("CMakeLists.txt").is_file()
        || !source_dir.join("include/gjxl/gjxl.h").is_file()
    {
        panic!(
            "gjxl-sys: set GJXL_SOURCE_DIR to a GJXL source checkout; no checkout was found at {}",
            source_dir.display()
        );
    }
}

#[cfg(feature = "native")]
fn generate_bindings(header: &Path) {
    let target = env::var("TARGET").expect("Cargo must provide TARGET");
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("--target={target}"))
        .allowlist_function("gjxl_.*")
        .allowlist_type("GJXL.*")
        .allowlist_var("GJXL_.*")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("gjxl-sys: failed to generate GJXL bindings");

    let output = PathBuf::from(env::var("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    bindings
        .write_to_file(output.join("bindings.rs"))
        .expect("gjxl-sys: failed to write GJXL bindings");
}

#[cfg(feature = "native")]
fn emit_link_directives(install: &Path, target_os: &str, enable_metal: bool, enable_cuda: bool) {
    println!(
        "cargo:rustc-link-search=native={}",
        install.join("lib").display()
    );

    // Keep this synchronized with the installed gjxl::c target. Static linkers
    // resolve references from left to right.
    for library in ["gjxl", "gjxl_codestream"] {
        println!("cargo:rustc-link-lib=static={library}");
    }
    if enable_cuda {
        println!("cargo:rustc-link-lib=static=gjxl_cuda");
    }
    if enable_metal {
        println!("cargo:rustc-link-lib=static=gjxl_metal");
    }
    for library in [
        "gjxl_gpu_butteraugli",
        "gjxl_gpu_ops",
        "gjxl_codec",
        "gjxl_gpu",
    ] {
        println!("cargo:rustc-link-lib=static={library}");
    }

    if enable_cuda {
        emit_cuda_link_directives(target_os);
    }
    if enable_metal {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }
    match target_os {
        "macos" => println!("cargo:rustc-link-lib=c++"),
        "linux" => println!("cargo:rustc-link-lib=stdc++"),
        _ => {}
    }
}

#[cfg(feature = "native")]
fn emit_cuda_link_directives(target_os: &str) {
    let cuda_path = env::var_os("CUDA_PATH").map(PathBuf::from).or_else(|| {
        ["CUDACXX", "CMAKE_CUDA_COMPILER"]
            .into_iter()
            .find_map(|name| env::var_os(name).map(PathBuf::from))
            .and_then(|compiler| compiler.parent()?.parent().map(Path::to_path_buf))
    });
    if let Some(cuda_path) = cuda_path {
        let library_dir = if target_os == "windows" {
            cuda_path.join("lib/x64")
        } else {
            cuda_path.join("lib64")
        };
        println!("cargo:rustc-link-search=native={}", library_dir.display());
    } else if target_os == "linux" {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
}
