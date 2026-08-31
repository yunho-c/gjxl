#[cfg(feature = "native")]
use std::env;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GJXL_SOURCE_DIR");

    #[cfg(feature = "native")]
    build_native();
}

#[cfg(feature = "native")]
fn build_native() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        panic!(
            "gjxl-sys: the native feature currently supports only macOS targets, got {target_os}"
        );
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

    let install = cmake::Config::new(&source_dir)
        .profile("Release")
        .define("GJXL_BUILD_TESTS", "OFF")
        .define("GJXL_BUILD_BENCHMARKS", "OFF")
        .define("GJXL_ENABLE_LIBJXL_REFERENCE", "OFF")
        .build();

    let header = install.join("include/gjxl/gjxl.h");
    if !header.is_file() {
        panic!(
            "gjxl-sys: GJXL install did not produce {}",
            header.display()
        );
    }

    generate_bindings(&header);
    emit_link_directives(&install);
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
fn emit_link_directives(install: &Path) {
    println!(
        "cargo:rustc-link-search=native={}",
        install.join("lib").display()
    );

    // Keep this synchronized with the installed gjxl::c target. Static linkers
    // resolve references from left to right.
    for library in [
        "gjxl",
        "gjxl_codestream",
        "gjxl_metal",
        "gjxl_gpu_butteraugli",
        "gjxl_gpu_ops",
        "gjxl_codec",
        "gjxl_gpu",
    ] {
        println!("cargo:rustc-link-lib=static={library}");
    }

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=c++");
}
