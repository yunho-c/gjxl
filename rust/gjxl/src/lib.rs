//! Safe Rust interface to GJXL's C encoder API.

#![cfg(feature = "native")]

use std::ffi::CStr;
use std::fmt;
use std::ptr;
use std::slice;

use gjxl_sys as sys;

/// Result type returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable category corresponding to a `GJXLResult` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    Unsupported,
    Unavailable,
    OutOfMemory,
    Backend,
    Internal,
}

/// Error returned by the GJXL C API or the safe Rust boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Returns the stable error category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the native or boundary diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for Error {}

/// Execution policy stored by a reusable [`Context`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

/// Maximum supported explicit CPU thread count.
pub const MAX_CPU_THREADS: usize = sys::GJXL_MAX_CPU_THREADS as usize;

/// Immutable execution policy stored by a reusable [`Context`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContextOptions {
    /// CPU/Metal/CUDA backend preference.
    pub backend: Backend,
    /// Maximum participating CPU threads per encode. `None` selects automatic.
    pub cpu_threads: Option<usize>,
}

/// Entropy/codestream search policy, independent of effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMode {
    /// Resolve balanced or high-density entropy behavior from effort.
    #[default]
    Automatic,
    /// Preserve the exhaustive entropy/codestream search as an explicit opt-in.
    Maximum,
}

/// Canonical compression controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderOptions {
    pub distance: f32,
    pub effort: i32,
    pub compression_mode: CompressionMode,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            distance: 1.0,
            effort: 7,
            compression_mode: CompressionMode::Automatic,
        }
    }
}

/// Packed nonlinear-sRGB pixel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8Srgb,
    Rgba8Srgb,
}

impl PixelFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8Srgb => 3,
            Self::Rgba8Srgb => 4,
        }
    }

    fn native(self) -> sys::GJXLPixelFormat {
        match self {
            Self::Rgb8Srgb => sys::GJXL_PIXEL_FORMAT_RGB8_SRGB as sys::GJXLPixelFormat,
            Self::Rgba8Srgb => sys::GJXL_PIXEL_FORMAT_RGBA8_SRGB as sys::GJXLPixelFormat,
        }
    }
}

/// Validated, non-owning view of packed sRGB pixels.
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a> {
    width: u32,
    height: u32,
    row_stride_bytes: usize,
    format: PixelFormat,
    pixels: &'a [u8],
}

impl<'a> ImageView<'a> {
    pub fn rgb8(
        width: u32,
        height: u32,
        row_stride_bytes: usize,
        pixels: &'a [u8],
    ) -> Result<Self> {
        Self::new(
            width,
            height,
            row_stride_bytes,
            pixels,
            PixelFormat::Rgb8Srgb,
        )
    }

    pub fn rgba8(
        width: u32,
        height: u32,
        row_stride_bytes: usize,
        pixels: &'a [u8],
    ) -> Result<Self> {
        Self::new(
            width,
            height,
            row_stride_bytes,
            pixels,
            PixelFormat::Rgba8Srgb,
        )
    }

    pub fn new(
        width: u32,
        height: u32,
        row_stride_bytes: usize,
        pixels: &'a [u8],
        format: PixelFormat,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "image dimensions must be nonzero",
            ));
        }
        let row_bytes = (width as usize)
            .checked_mul(format.bytes_per_pixel())
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument, "image row size overflow"))?;
        if row_stride_bytes < row_bytes {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "image row stride is smaller than a packed row",
            ));
        }
        let required = (height as usize - 1)
            .checked_mul(row_stride_bytes)
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument, "image byte size overflow"))?;
        if pixels.len() < required {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                format!(
                    "image buffer is too short: requires {required} bytes, got {}",
                    pixels.len()
                ),
            ));
        }
        Ok(Self {
            width,
            height,
            row_stride_bytes,
            format,
            pixels,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.row_stride_bytes
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.format
    }
}

/// Reusable immutable GJXL execution context.
#[derive(Debug)]
pub struct Context {
    raw: *mut sys::GJXLContext,
}

// The C API guarantees concurrent encode calls on one immutable context.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn new(backend: Backend) -> Result<Self> {
        Self::with_options(ContextOptions {
            backend,
            ..ContextOptions::default()
        })
    }

    pub fn with_options(context_options: ContextOptions) -> Result<Self> {
        let mut options = unsafe { std::mem::zeroed::<sys::GJXLContextOptions>() };
        check(unsafe {
            sys::gjxl_context_options_init(
                &mut options,
                std::mem::size_of::<sys::GJXLContextOptions>(),
            )
        })?;
        options.backend = match context_options.backend {
            Backend::Auto => sys::GJXL_BACKEND_AUTO,
            Backend::Cpu => sys::GJXL_BACKEND_CPU,
            Backend::Metal => sys::GJXL_BACKEND_METAL,
            Backend::Cuda => sys::GJXL_BACKEND_CUDA,
        } as sys::GJXLBackend;
        options.num_cpu_threads = match context_options.cpu_threads {
            None => 0,
            Some(0) => {
                return Err(Error::new(
                    ErrorKind::InvalidArgument,
                    "CPU thread count must be positive when specified",
                ));
            }
            Some(count) if count > MAX_CPU_THREADS => {
                return Err(Error::new(
                    ErrorKind::InvalidArgument,
                    format!("CPU thread count must not exceed {MAX_CPU_THREADS}"),
                ));
            }
            Some(count) => count as u32,
        };

        let mut raw = ptr::null_mut();
        check(unsafe { sys::gjxl_context_create(&options, &mut raw) })?;
        if raw.is_null() {
            return Err(Error::new(
                ErrorKind::Internal,
                "GJXL returned success without a context",
            ));
        }
        Ok(Self { raw })
    }

    pub fn encode(&self, image: &ImageView<'_>, options: EncoderOptions) -> Result<Vec<u8>> {
        let image = sys::GJXLImageView {
            struct_size: struct_size::<sys::GJXLImageView>()?,
            width: image.width,
            height: image.height,
            pixel_format: image.format.native(),
            pixels: image.pixels.as_ptr().cast(),
            pixels_size: image.pixels.len(),
            row_stride_bytes: image.row_stride_bytes,
        };

        let mut native_options = unsafe { std::mem::zeroed::<sys::GJXLEncoderOptions>() };
        check(unsafe {
            sys::gjxl_encoder_options_init(
                &mut native_options,
                std::mem::size_of::<sys::GJXLEncoderOptions>(),
            )
        })?;
        native_options.distance = options.distance;
        native_options.effort = options.effort;
        native_options.compression_mode = match options.compression_mode {
            CompressionMode::Automatic => sys::GJXL_COMPRESSION_AUTOMATIC,
            CompressionMode::Maximum => sys::GJXL_COMPRESSION_MAXIMUM,
        } as sys::GJXLCompressionMode;

        let mut output = NativeBuffer::empty();
        check(unsafe { sys::gjxl_encode(self.raw, &image, &native_options, &mut output.raw) })?;
        if output.raw.data.is_null() || output.raw.size == 0 {
            return Err(Error::new(
                ErrorKind::Internal,
                "GJXL returned success without an output codestream",
            ));
        }
        Ok(unsafe { slice::from_raw_parts(output.raw.data, output.raw.size) }.to_vec())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::gjxl_context_destroy(self.raw) };
    }
}

/// Converts a familiar quality value in `[0, 100]` to canonical distance.
pub fn distance_from_quality(quality: f32) -> f32 {
    unsafe { sys::gjxl_distance_from_quality(quality) }
}

fn struct_size<T>() -> Result<u32> {
    u32::try_from(std::mem::size_of::<T>())
        .map_err(|_| Error::new(ErrorKind::Internal, "GJXL ABI struct size exceeds uint32_t"))
}

fn check(result: sys::GJXLResult) -> Result<()> {
    if result == sys::GJXL_OK as sys::GJXLResult {
        return Ok(());
    }
    let message = unsafe {
        let pointer = sys::gjxl_get_last_error();
        if pointer.is_null() {
            "GJXL returned no diagnostic".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    let kind = if result == sys::GJXL_ERROR_INVALID_ARGUMENT as sys::GJXLResult {
        ErrorKind::InvalidArgument
    } else if result == sys::GJXL_ERROR_UNSUPPORTED as sys::GJXLResult {
        ErrorKind::Unsupported
    } else if result == sys::GJXL_ERROR_UNAVAILABLE as sys::GJXLResult {
        ErrorKind::Unavailable
    } else if result == sys::GJXL_ERROR_OUT_OF_MEMORY as sys::GJXLResult {
        ErrorKind::OutOfMemory
    } else if result == sys::GJXL_ERROR_BACKEND as sys::GJXLResult {
        ErrorKind::Backend
    } else {
        ErrorKind::Internal
    };
    Err(Error::new(kind, message))
}

struct NativeBuffer {
    raw: sys::GJXLBuffer,
}

impl NativeBuffer {
    fn empty() -> Self {
        Self {
            raw: sys::GJXLBuffer {
                data: ptr::null_mut(),
                size: 0,
            },
        }
    }
}

impl Drop for NativeBuffer {
    fn drop(&mut self) {
        unsafe { sys::gjxl_buffer_free(&mut self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn rgba_fixture(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    ((x * 255) / width.max(1)) as u8,
                    ((y * 255) / height.max(1)) as u8,
                    128,
                    255,
                ]);
            }
        }
        pixels
    }

    #[test]
    fn defaults_and_quality_mapping_are_canonical() {
        assert_eq!(ContextOptions::default().backend, Backend::Auto);
        assert_eq!(ContextOptions::default().cpu_threads, None);
        assert_eq!(EncoderOptions::default().distance, 1.0);
        assert_eq!(EncoderOptions::default().effort, 7);
        assert_eq!(
            EncoderOptions::default().compression_mode,
            CompressionMode::Automatic
        );
        assert_eq!(distance_from_quality(100.0), 0.0);
        assert!((distance_from_quality(80.0) - 1.9).abs() < 1.0e-6);
    }

    #[test]
    fn context_thread_budget_is_validated() {
        for cpu_threads in [Some(0), Some(MAX_CPU_THREADS + 1)] {
            let error = Context::with_options(ContextOptions {
                backend: Backend::Cpu,
                cpu_threads,
            })
            .expect_err("invalid CPU thread count must fail");
            assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_context_encodes_when_a_device_is_available() {
        let context = match Context::with_options(ContextOptions {
            backend: Backend::Cuda,
            cpu_threads: None,
        }) {
            Ok(context) => context,
            Err(error) if error.kind() == ErrorKind::Unavailable => return,
            Err(error) => panic!("CUDA context creation failed: {error}"),
        };
        let pixels = rgba_fixture(64, 48);
        let image = ImageView::rgba8(64, 48, 64 * 4, &pixels).unwrap();
        let codestream = context
            .encode(&image, EncoderOptions::default())
            .expect("CUDA encode must succeed after context creation");
        assert!(!codestream.is_empty());
    }

    #[test]
    fn explicit_thread_budgets_preserve_codestream_bytes() {
        let pixels = rgba_fixture(64, 64);
        let image = ImageView::rgba8(64, 64, 256, &pixels).unwrap();
        let encode = |cpu_threads| {
            Context::with_options(ContextOptions {
                backend: Backend::Cpu,
                cpu_threads,
            })
            .unwrap()
            .encode(&image, EncoderOptions::default())
            .unwrap()
        };
        let automatic = encode(None);
        for cpu_threads in [1, 2, 4, 8] {
            assert_eq!(encode(Some(cpu_threads)), automatic);
        }
    }

    #[test]
    fn image_view_checks_layout() {
        let pixels = rgba_fixture(2, 2);
        assert!(ImageView::rgba8(2, 2, 8, &pixels).is_ok());
        assert_eq!(
            ImageView::rgba8(2, 2, 7, &pixels).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            ImageView::rgba8(2, 2, 8, &pixels[..15]).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn encodes_tight_and_padded_rgba() {
        let context = Context::new(Backend::Cpu).expect("CPU context should initialize");
        let pixels = rgba_fixture(8, 8);
        let tight = ImageView::rgba8(8, 8, 32, &pixels).unwrap();
        let encoded = context.encode(&tight, EncoderOptions::default()).unwrap();
        assert!(encoded.starts_with(&[0xff, 0x0a]));

        let mut padded = vec![0u8; 8 * 40];
        for (source, destination) in pixels.chunks_exact(32).zip(padded.chunks_exact_mut(40)) {
            destination[..32].copy_from_slice(source);
        }
        let padded = ImageView::rgba8(8, 8, 40, &padded).unwrap();
        let encoded = context.encode(&padded, EncoderOptions::default()).unwrap();
        assert!(encoded.starts_with(&[0xff, 0x0a]));
    }

    #[test]
    fn maximum_compression_is_explicitly_selectable() {
        let context = Context::new(Backend::Cpu).expect("CPU context should initialize");
        let pixels = rgba_fixture(64, 64);
        let image = ImageView::rgba8(64, 64, 256, &pixels).unwrap();
        let automatic = context.encode(&image, EncoderOptions::default()).unwrap();
        let maximum = context
            .encode(
                &image,
                EncoderOptions {
                    compression_mode: CompressionMode::Maximum,
                    ..EncoderOptions::default()
                },
            )
            .unwrap();
        assert!(maximum.starts_with(&[0xff, 0x0a]));
        assert_ne!(maximum, automatic);
    }

    #[test]
    fn transparent_rgba_is_reported_as_unsupported() {
        let context = Context::new(Backend::Cpu).expect("CPU context should initialize");
        let mut pixels = rgba_fixture(2, 2);
        pixels[3] = 128;
        let image = ImageView::rgba8(2, 2, 8, &pixels).unwrap();
        let error = context
            .encode(&image, EncoderOptions::default())
            .expect_err("transparent input must not be discarded");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn one_context_supports_concurrent_encodes() {
        let context = Arc::new(Context::new(Backend::Cpu).expect("CPU context should initialize"));
        let pixels = Arc::new(rgba_fixture(16, 16));
        let handles = (0..4)
            .map(|_| {
                let context = Arc::clone(&context);
                let pixels = Arc::clone(&pixels);
                std::thread::spawn(move || {
                    let image = ImageView::rgba8(16, 16, 64, pixels.as_slice()).unwrap();
                    context.encode(&image, EncoderOptions::default())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            let encoded = handle.join().unwrap().unwrap();
            assert!(encoded.starts_with(&[0xff, 0x0a]));
        }
    }
}
