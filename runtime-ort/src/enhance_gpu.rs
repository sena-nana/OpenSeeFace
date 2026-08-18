//! GPU enhance: Metal (macOS) / CUDA (elsewhere) on device memory.
//!
//! The working BGR buffer stays in shared or pinned GPU-addressable memory.
//! Only tiny reductions (channel sums) are read by the CPU.

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use anyhow::bail;
use anyhow::Result;

use crate::enhance::{tile_grid, EnhanceCfg, HeMode};
use crate::preprocess::BgrImage;

#[allow(dead_code)]
const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Info {
  uint width;
  uint height;
  uint tiles_x;
  uint tiles_y;
  uint tile_w;
  uint tile_h;
  uint n;
  uint ahe;
  float clip_limit;
  float blend;
  float gb;
  float gg;
  float gr;
  float pad0;
  float pad1;
  float pad2;
};

kernel void wb_sum(
    device const uchar *bgr [[buffer(0)]],
    device atomic_uint *sums [[buffer(1)]],
    constant Info &info [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= info.n) return;
  uint p = gid * 3;
  atomic_fetch_add_explicit(&sums[0], (uint)bgr[p], memory_order_relaxed);
  atomic_fetch_add_explicit(&sums[1], (uint)bgr[p + 1], memory_order_relaxed);
  atomic_fetch_add_explicit(&sums[2], (uint)bgr[p + 2], memory_order_relaxed);
}

kernel void wb_apply(
    device uchar *bgr [[buffer(0)]],
    constant Info &info [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= info.n) return;
  uint p = gid * 3;
  float b = (float)bgr[p] * info.gb;
  float g = (float)bgr[p + 1] * info.gg;
  float r = (float)bgr[p + 2] * info.gr;
  bgr[p] = (uchar)clamp(round(b), 0.0f, 255.0f);
  bgr[p + 1] = (uchar)clamp(round(g), 0.0f, 255.0f);
  bgr[p + 2] = (uchar)clamp(round(r), 0.0f, 255.0f);
}

kernel void clahe_hist(
    device const uchar *bgr [[buffer(0)]],
    device atomic_uint *hist [[buffer(1)]],
    constant Info &info [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= info.n) return;
  uint x = gid % info.width;
  uint y = gid / info.width;
  uint txi = min(x / info.tile_w, info.tiles_x - 1);
  uint tyi = min(y / info.tile_h, info.tiles_y - 1);
  uint p = gid * 3;
  float Y = 0.114f * (float)bgr[p] + 0.587f * (float)bgr[p + 1] + 0.299f * (float)bgr[p + 2];
  uint bin = (uint)clamp(round(Y), 0.0f, 255.0f);
  uint tile = tyi * info.tiles_x + txi;
  atomic_fetch_add_explicit(&hist[tile * 256 + bin], 1u, memory_order_relaxed);
}

kernel void clahe_lut(
    device uint *hist [[buffer(0)]],
    device uchar *lut [[buffer(1)]],
    constant Info &info [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint ntiles = info.tiles_x * info.tiles_y;
  if (gid >= ntiles) return;
  uint txi = gid % info.tiles_x;
  uint tyi = gid / info.tiles_x;
  uint x1 = txi * info.tile_w;
  uint y1 = tyi * info.tile_h;
  uint x2 = (txi + 1 == info.tiles_x) ? info.width : (txi + 1) * info.tile_w;
  uint y2 = (tyi + 1 == info.tiles_y) ? info.height : (tyi + 1) * info.tile_h;
  uint tile_n = (x2 - x1) * (y2 - y1);
  device uint *h = hist + gid * 256;
  if (info.ahe == 0) {
    uint clip = (uint)floor(info.clip_limit * (float)tile_n / 256.0f);
    if (clip < 1u) clip = 1u;
    uint clipped = 0;
    for (uint i = 0; i < 256; i++) {
      if (h[i] > clip) {
        clipped += h[i] - clip;
        h[i] = clip;
      }
    }
    uint batch = clipped / 256u;
    uint residual = clipped - batch * 256u;
    for (uint i = 0; i < 256; i++) h[i] += batch;
    for (uint i = 0; i < residual; i++) h[i] += 1u;
  }
  float scale = 255.0f / (float)max(tile_n, 1u);
  uint sum = 0;
  device uchar *l = lut + gid * 256;
  for (uint i = 0; i < 256; i++) {
    sum += h[i];
    l[i] = (uchar)clamp(round((float)sum * scale), 0.0f, 255.0f);
  }
}

static float map_y(device const uchar *lut, constant Info &info, uint x, uint y, uint bin) {
  float fx = (float)x / (float)info.tile_w - 0.5f;
  float fy = (float)y / (float)info.tile_h - 0.5f;
  int tx1 = (int)floor(fx);
  int ty1 = (int)floor(fy);
  float wx = fx - (float)tx1;
  float wy = fy - (float)ty1;
  int txm = (int)info.tiles_x - 1;
  int tym = (int)info.tiles_y - 1;
  uint xa = (uint)clamp(tx1, 0, txm);
  uint xb = (uint)clamp(tx1 + 1, 0, txm);
  uint ya = (uint)clamp(ty1, 0, tym);
  uint yb = (uint)clamp(ty1 + 1, 0, tym);
  float s00 = (float)lut[(ya * info.tiles_x + xa) * 256 + bin];
  float s10 = (float)lut[(ya * info.tiles_x + xb) * 256 + bin];
  float s01 = (float)lut[(yb * info.tiles_x + xa) * 256 + bin];
  float s11 = (float)lut[(yb * info.tiles_x + xb) * 256 + bin];
  return s00 * (1.0f - wy) * (1.0f - wx) + s10 * (1.0f - wy) * wx
       + s01 * wy * (1.0f - wx) + s11 * wy * wx;
}

kernel void clahe_apply(
    device uchar *bgr [[buffer(0)]],
    device const uchar *lut [[buffer(1)]],
    constant Info &info [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= info.n) return;
  uint x = gid % info.width;
  uint y = gid / info.width;
  uint p = gid * 3;
  float b = (float)bgr[p];
  float g = (float)bgr[p + 1];
  float r = (float)bgr[p + 2];
  float yv = 0.114f * b + 0.587f * g + 0.299f * r;
  float cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;
  float cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
  uint bin = (uint)clamp(round(yv), 0.0f, 255.0f);
  float y2 = map_y(lut, info, x, y, bin);
  float b2 = y2 + 1.772f * (cb - 128.0f);
  float g2 = y2 - 0.344136f * (cb - 128.0f) - 0.714136f * (cr - 128.0f);
  float r2 = y2 + 1.402f * (cr - 128.0f);
  bgr[p] = (uchar)clamp(round(b2), 0.0f, 255.0f);
  bgr[p + 1] = (uchar)clamp(round(g2), 0.0f, 255.0f);
  bgr[p + 2] = (uchar)clamp(round(r2), 0.0f, 255.0f);
}

kernel void blend_orig(
    device uchar *bgr [[buffer(0)]],
    device const uchar *orig [[buffer(1)]],
    constant Info &info [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= info.n) return;
  uint p = gid * 3;
  float a = info.blend;
  float b = 1.0f - a;
  bgr[p] = (uchar)clamp(round((float)bgr[p] * a + (float)orig[p] * b), 0.0f, 255.0f);
  bgr[p + 1] = (uchar)clamp(round((float)bgr[p + 1] * a + (float)orig[p + 1] * b), 0.0f, 255.0f);
  bgr[p + 2] = (uchar)clamp(round((float)bgr[p + 2] * a + (float)orig[p + 2] * b), 0.0f, 255.0f);
}
"#;

#[allow(dead_code)]
const CUDA_SRC: &str = r#"
struct Info {
  unsigned int width, height, tiles_x, tiles_y;
  unsigned int tile_w, tile_h, n, ahe;
  float clip_limit, blend, gb, gg, gr;
  float pad0, pad1, pad2;
};

extern "C" __global__ void wb_sum(const unsigned char* bgr, unsigned int* sums, Info info) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= info.n) return;
  unsigned int p = gid * 3;
  atomicAdd(&sums[0], (unsigned int)bgr[p]);
  atomicAdd(&sums[1], (unsigned int)bgr[p + 1]);
  atomicAdd(&sums[2], (unsigned int)bgr[p + 2]);
}

extern "C" __global__ void wb_apply(unsigned char* bgr, Info info) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= info.n) return;
  unsigned int p = gid * 3;
  float b = (float)bgr[p] * info.gb;
  float g = (float)bgr[p + 1] * info.gg;
  float r = (float)bgr[p + 2] * info.gr;
  bgr[p] = (unsigned char)fminf(fmaxf(roundf(b), 0.f), 255.f);
  bgr[p + 1] = (unsigned char)fminf(fmaxf(roundf(g), 0.f), 255.f);
  bgr[p + 2] = (unsigned char)fminf(fmaxf(roundf(r), 0.f), 255.f);
}

extern "C" __global__ void clahe_hist(const unsigned char* bgr, unsigned int* hist, Info info) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= info.n) return;
  unsigned int x = gid % info.width;
  unsigned int y = gid / info.width;
  unsigned int txi = min(x / info.tile_w, info.tiles_x - 1);
  unsigned int tyi = min(y / info.tile_h, info.tiles_y - 1);
  unsigned int p = gid * 3;
  float Y = 0.114f * (float)bgr[p] + 0.587f * (float)bgr[p + 1] + 0.299f * (float)bgr[p + 2];
  unsigned int bin = (unsigned int)fminf(fmaxf(roundf(Y), 0.f), 255.f);
  atomicAdd(&hist[(tyi * info.tiles_x + txi) * 256 + bin], 1u);
}

extern "C" __global__ void clahe_lut(unsigned int* hist, unsigned char* lut, Info info) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ntiles = info.tiles_x * info.tiles_y;
  if (gid >= ntiles) return;
  unsigned int txi = gid % info.tiles_x;
  unsigned int tyi = gid / info.tiles_x;
  unsigned int x1 = txi * info.tile_w;
  unsigned int y1 = tyi * info.tile_h;
  unsigned int x2 = (txi + 1 == info.tiles_x) ? info.width : (txi + 1) * info.tile_w;
  unsigned int y2 = (tyi + 1 == info.tiles_y) ? info.height : (tyi + 1) * info.tile_h;
  unsigned int tile_n = (x2 - x1) * (y2 - y1);
  unsigned int* h = hist + gid * 256;
  if (info.ahe == 0) {
    unsigned int clip = (unsigned int)floorf(info.clip_limit * (float)tile_n / 256.f);
    if (clip < 1u) clip = 1u;
    unsigned int clipped = 0;
    for (int i = 0; i < 256; i++) {
      if (h[i] > clip) { clipped += h[i] - clip; h[i] = clip; }
    }
    unsigned int batch = clipped / 256u;
    unsigned int residual = clipped - batch * 256u;
    for (int i = 0; i < 256; i++) h[i] += batch;
    for (unsigned int i = 0; i < residual; i++) h[i] += 1u;
  }
  float scale = 255.f / (float)max(tile_n, 1u);
  unsigned int sum = 0;
  unsigned char* l = lut + gid * 256;
  for (int i = 0; i < 256; i++) {
    sum += h[i];
    float v = (float)sum * scale;
    l[i] = (unsigned char)fminf(fmaxf(roundf(v), 0.f), 255.f);
  }
}

__device__ float map_y(const unsigned char* lut, Info info, unsigned int x, unsigned int y, unsigned int bin) {
  float fx = (float)x / (float)info.tile_w - 0.5f;
  float fy = (float)y / (float)info.tile_h - 0.5f;
  int tx1 = (int)floorf(fx);
  int ty1 = (int)floorf(fy);
  float wx = fx - (float)tx1;
  float wy = fy - (float)ty1;
  int txm = (int)info.tiles_x - 1;
  int tym = (int)info.tiles_y - 1;
  unsigned int xa = (unsigned int)max(0, min(tx1, txm));
  unsigned int xb = (unsigned int)max(0, min(tx1 + 1, txm));
  unsigned int ya = (unsigned int)max(0, min(ty1, tym));
  unsigned int yb = (unsigned int)max(0, min(ty1 + 1, tym));
  float s00 = (float)lut[(ya * info.tiles_x + xa) * 256 + bin];
  float s10 = (float)lut[(ya * info.tiles_x + xb) * 256 + bin];
  float s01 = (float)lut[(yb * info.tiles_x + xa) * 256 + bin];
  float s11 = (float)lut[(yb * info.tiles_x + xb) * 256 + bin];
  return s00 * (1.f - wy) * (1.f - wx) + s10 * (1.f - wy) * wx
       + s01 * wy * (1.f - wx) + s11 * wy * wx;
}

extern "C" __global__ void clahe_apply(unsigned char* bgr, const unsigned char* lut, Info info) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= info.n) return;
  unsigned int x = gid % info.width;
  unsigned int y = gid / info.width;
  unsigned int p = gid * 3;
  float b = (float)bgr[p];
  float g = (float)bgr[p + 1];
  float r = (float)bgr[p + 2];
  float yv = 0.114f * b + 0.587f * g + 0.299f * r;
  float cb = 128.f - 0.168736f * r - 0.331264f * g + 0.5f * b;
  float cr = 128.f + 0.5f * r - 0.418688f * g - 0.081312f * b;
  unsigned int bin = (unsigned int)fminf(fmaxf(roundf(yv), 0.f), 255.f);
  float y2 = map_y(lut, info, x, y, bin);
  float b2 = y2 + 1.772f * (cb - 128.f);
  float g2 = y2 - 0.344136f * (cb - 128.f) - 0.714136f * (cr - 128.f);
  float r2 = y2 + 1.402f * (cr - 128.f);
  bgr[p] = (unsigned char)fminf(fmaxf(roundf(b2), 0.f), 255.f);
  bgr[p + 1] = (unsigned char)fminf(fmaxf(roundf(g2), 0.f), 255.f);
  bgr[p + 2] = (unsigned char)fminf(fmaxf(roundf(r2), 0.f), 255.f);
}

extern "C" __global__ void blend_orig(unsigned char* bgr, const unsigned char* orig, Info info) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= info.n) return;
  unsigned int p = gid * 3;
  float a = info.blend;
  float bb = 1.f - a;
  bgr[p] = (unsigned char)fminf(fmaxf(roundf((float)bgr[p] * a + (float)orig[p] * bb), 0.f), 255.f);
  bgr[p + 1] = (unsigned char)fminf(fmaxf(roundf((float)bgr[p + 1] * a + (float)orig[p + 1] * bb), 0.f), 255.f);
  bgr[p + 2] = (unsigned char)fminf(fmaxf(roundf((float)bgr[p + 2] * a + (float)orig[p + 2] * bb), 0.f), 255.f);
}
"#;

#[allow(dead_code)]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct GpuInfo {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    tile_w: u32,
    tile_h: u32,
    n: u32,
    ahe: u32,
    clip_limit: f32,
    blend: f32,
    gb: f32,
    gg: f32,
    gr: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
unsafe impl cudarc::driver::DeviceRepr for GpuInfo {}

#[allow(dead_code)]
fn make_info(w: u32, h: u32, cfg: &EnhanceCfg, gb: f32, gg: f32, gr: f32) -> GpuInfo {
    let (tx, ty, tw, th) = tile_grid(w, h, cfg.tiles);
    let ahe = matches!(cfg.he, HeMode::Ahe) || (cfg.he == HeMode::Clahe && cfg.clip_limit <= 0.0);
    GpuInfo {
        width: w,
        height: h,
        tiles_x: tx,
        tiles_y: ty,
        tile_w: tw,
        tile_h: th,
        n: w * h,
        ahe: u32::from(ahe),
        clip_limit: cfg.clip_limit,
        blend: cfg.blend.clamp(0.0, 1.0),
        gb,
        gg,
        gr,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Device-side WB / AHE / CLAHE. `run` returns a CPU-visible view of shared/pinned memory.
pub struct GpuEnhance {
    cfg: EnhanceCfg,
    inner: Inner,
}

enum Inner {
    Off,
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    Metal(MetalEnhance),
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    Cuda(CudaEnhance),
}

impl GpuEnhance {
    pub fn new(width: u32, height: u32, cfg: EnhanceCfg) -> Result<Self> {
        if cfg.is_off() {
            return Ok(Self {
                cfg,
                inner: Inner::Off,
            });
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            return Ok(Self {
                cfg,
                inner: Inner::Metal(MetalEnhance::new(width, height)?),
            });
        }
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        {
            return Ok(Self {
                cfg,
                inner: Inner::Cuda(CudaEnhance::new(width, height)?),
            });
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (width, height);
            Ok(Self {
                cfg,
                inner: Inner::Off,
            })
        }
    }

    #[allow(dead_code)]
    pub fn enabled(&self) -> bool {
        !matches!(self.inner, Inner::Off)
    }

    /// Enhance `src` on GPU. Returned slice is shared/pinned memory (no extra memcpy of the frame).
    pub fn run<'a>(&'a mut self, src: &'a BgrImage) -> Result<&'a [u8]> {
        let cfg = self.cfg.resolve(src);
        if cfg.is_off() {
            return Ok(src.data.as_slice());
        }
        match &mut self.inner {
            Inner::Off => Ok(src.data.as_slice()),
            #[cfg(all(feature = "gpu", target_os = "macos"))]
            Inner::Metal(m) => {
                if self.cfg.auto {
                    m.invalidate();
                }
                m.run(src, &cfg)
            }
            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            Inner::Cuda(c) => c.run(src, &cfg),
        }
    }

    /// In-place enhance of a GPU-addressable host buffer (CUDA pinned / Metal shared).
    #[allow(dead_code)]
    pub fn run_in_place(&mut self, data: &mut [u8], width: u32, height: u32) -> Result<()> {
        let cfg = if self.cfg.auto {
            self.cfg.resolve(&BgrImage {
                width,
                height,
                data: data.to_vec(),
            })
        } else {
            self.cfg
        };
        if cfg.is_off() {
            return Ok(());
        }
        match &mut self.inner {
            Inner::Off => Ok(()),
            #[cfg(all(feature = "gpu", target_os = "macos"))]
            Inner::Metal(m) => m.run_in_place(data, width, height, &cfg),
            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            Inner::Cuda(c) => c.run_in_place(data, width, height, &cfg),
        }
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
struct MetalEnhance {
    device: metal::Device,
    queue: metal::CommandQueue,
    wb_sum: metal::ComputePipelineState,
    wb_apply: metal::ComputePipelineState,
    hist: metal::ComputePipelineState,
    lut: metal::ComputePipelineState,
    apply: metal::ComputePipelineState,
    blend: metal::ComputePipelineState,
    bgr: metal::Buffer,
    orig: metal::Buffer,
    sums: metal::Buffer,
    hist_buf: metal::Buffer,
    lut_buf: metal::Buffer,
    info_buf: metal::Buffer,
    width: u32,
    height: u32,
    bytes: usize,
    last_ptr: usize,
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
impl MetalEnhance {
    fn new(width: u32, height: u32) -> Result<Self> {
        use metal::{Device, MTLResourceOptions};

        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
        let opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| anyhow::anyhow!("Metal shader: {e}"))?;
        let pso = |name: &str| -> Result<metal::ComputePipelineState> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| anyhow::anyhow!("{name} pso: {e}"))
        };
        let bytes = (width as usize) * (height as usize) * 3;
        let opt = MTLResourceOptions::StorageModeShared;
        let tiles = 16u32;
        let hist_bytes = (tiles * tiles * 256 * 4) as u64;
        Ok(Self {
            queue: device.new_command_queue(),
            wb_sum: pso("wb_sum")?,
            wb_apply: pso("wb_apply")?,
            hist: pso("clahe_hist")?,
            lut: pso("clahe_lut")?,
            apply: pso("clahe_apply")?,
            blend: pso("blend_orig")?,
            bgr: device.new_buffer(bytes as u64, opt),
            orig: device.new_buffer(bytes as u64, opt),
            sums: device.new_buffer(12, opt),
            hist_buf: device.new_buffer(hist_bytes, opt),
            lut_buf: device.new_buffer((tiles * tiles * 256) as u64, opt),
            info_buf: device.new_buffer(std::mem::size_of::<GpuInfo>() as u64, opt),
            device,
            width,
            height,
            bytes,
            last_ptr: 0,
        })
    }

    fn invalidate(&mut self) {
        self.last_ptr = 0;
    }

    fn realloc(&mut self, width: u32, height: u32) {
        use metal::MTLResourceOptions;
        let bytes = (width as usize) * (height as usize) * 3;
        if bytes == self.bytes && width == self.width {
            return;
        }
        let opt = MTLResourceOptions::StorageModeShared;
        self.bgr = self.device.new_buffer(bytes as u64, opt);
        self.orig = self.device.new_buffer(bytes as u64, opt);
        self.width = width;
        self.height = height;
        self.bytes = bytes;
        self.last_ptr = 0;
    }

    fn write(buf: &metal::Buffer, src: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut u8, src.len());
        }
    }

    fn zero(buf: &metal::Buffer, len: usize) {
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0, len);
        }
    }

    fn view(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.bgr.contents() as *const u8, self.bytes) }
    }

    fn run(&mut self, src: &BgrImage, cfg: &EnhanceCfg) -> Result<&[u8]> {
        let ptr = src.data.as_ptr() as usize;
        if ptr == self.last_ptr && src.width == self.width && src.data.len() == self.bytes {
            return Ok(self.view());
        }
        self.realloc(src.width, src.height);
        Self::write(&self.bgr, &src.data);
        if cfg.blend < 1.0 {
            Self::write(&self.orig, &src.data);
        }
        self.dispatch(cfg)?;
        self.last_ptr = ptr;
        Ok(self.view())
    }

    fn run_in_place(
        &mut self,
        data: &mut [u8],
        width: u32,
        height: u32,
        cfg: &EnhanceCfg,
    ) -> Result<()> {
        self.realloc(width, height);
        Self::write(&self.bgr, data);
        if cfg.blend < 1.0 {
            Self::write(&self.orig, data);
        }
        self.dispatch(cfg)?;
        // Unified memory: the caller may keep using `data` only if it is this buffer.
        // CoreML path uses `run` + shared view; this copies into the Metal buffer only.
        let out = self.view();
        data.copy_from_slice(out);
        self.last_ptr = 0;
        Ok(())
    }

    fn dispatch(&mut self, cfg: &EnhanceCfg) -> Result<()> {
        let n = self.width * self.height;
        let mut info = make_info(self.width, self.height, cfg, 1.0, 1.0, 1.0);
        self.write_info(&info);

        if cfg.wb {
            Self::zero(&self.sums, 12);
            self.encode(&self.wb_sum, n, |enc| {
                enc.set_buffer(0, Some(&self.bgr), 0);
                enc.set_buffer(1, Some(&self.sums), 0);
                enc.set_buffer(2, Some(&self.info_buf), 0);
            });
            let sums = unsafe { std::slice::from_raw_parts(self.sums.contents() as *const u32, 3) };
            let inv = 1.0 / n.max(1) as f32;
            let mb = sums[0] as f32 * inv;
            let mg = sums[1] as f32 * inv;
            let mr = sums[2] as f32 * inv;
            let gray = (mb + mg + mr) / 3.0;
            let gain = |m: f32| gray / m.max(1e-3);
            info.gb = gain(mb);
            info.gg = gain(mg);
            info.gr = gain(mr);
            self.write_info(&info);
            self.encode(&self.wb_apply, n, |enc| {
                enc.set_buffer(0, Some(&self.bgr), 0);
                enc.set_buffer(1, Some(&self.info_buf), 0);
            });
        }

        if cfg.he != HeMode::Off {
            let hist_len = (info.tiles_x * info.tiles_y * 256 * 4) as usize;
            Self::zero(&self.hist_buf, hist_len);
            self.encode(&self.hist, n, |enc| {
                enc.set_buffer(0, Some(&self.bgr), 0);
                enc.set_buffer(1, Some(&self.hist_buf), 0);
                enc.set_buffer(2, Some(&self.info_buf), 0);
            });
            let ntiles = info.tiles_x * info.tiles_y;
            self.encode(&self.lut, ntiles, |enc| {
                enc.set_buffer(0, Some(&self.hist_buf), 0);
                enc.set_buffer(1, Some(&self.lut_buf), 0);
                enc.set_buffer(2, Some(&self.info_buf), 0);
            });
            self.encode(&self.apply, n, |enc| {
                enc.set_buffer(0, Some(&self.bgr), 0);
                enc.set_buffer(1, Some(&self.lut_buf), 0);
                enc.set_buffer(2, Some(&self.info_buf), 0);
            });
        }

        if cfg.blend < 1.0 {
            self.encode(&self.blend, n, |enc| {
                enc.set_buffer(0, Some(&self.bgr), 0);
                enc.set_buffer(1, Some(&self.orig), 0);
                enc.set_buffer(2, Some(&self.info_buf), 0);
            });
        }
        Ok(())
    }

    fn write_info(&self, info: &GpuInfo) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                info as *const GpuInfo as *const u8,
                self.info_buf.contents() as *mut u8,
                std::mem::size_of::<GpuInfo>(),
            );
        }
    }

    fn encode(
        &self,
        pso: &metal::ComputePipelineState,
        n: u32,
        bind: impl FnOnce(&metal::ComputeCommandEncoderRef),
    ) {
        use metal::MTLSize;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        bind(enc);
        let tgs = 256u64.min(pso.max_total_threads_per_threadgroup());
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tgs, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
struct CudaEnhance {
    dev: std::sync::Arc<cudarc::driver::CudaDevice>,
    wb_sum: cudarc::driver::CudaFunction,
    wb_apply: cudarc::driver::CudaFunction,
    hist: cudarc::driver::CudaFunction,
    lut: cudarc::driver::CudaFunction,
    apply: cudarc::driver::CudaFunction,
    blend: cudarc::driver::CudaFunction,
    orig: cudarc::driver::CudaSlice<u8>,
    sums: cudarc::driver::CudaSlice<u32>,
    hist_buf: cudarc::driver::CudaSlice<u32>,
    lut_buf: cudarc::driver::CudaSlice<u8>,
    width: u32,
    height: u32,
    bytes: usize,
    host_out: Vec<u8>,
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
impl CudaEnhance {
    fn new(width: u32, height: u32) -> Result<Self> {
        use cudarc::driver::CudaDevice;
        use cudarc::nvrtc::compile_ptx;

        let ptx = compile_ptx(CUDA_SRC).map_err(|e| anyhow::anyhow!("nvrtc: {e}"))?;
        let dev = CudaDevice::new(0).map_err(|e| anyhow::anyhow!("cuda device: {e}"))?;
        dev.load_ptx(
            ptx,
            "enhance",
            &[
                "wb_sum",
                "wb_apply",
                "clahe_hist",
                "clahe_lut",
                "clahe_apply",
                "blend_orig",
            ],
        )
        .map_err(|e| anyhow::anyhow!("load ptx: {e}"))?;
        let get = |n: &str| {
            dev.get_func("enhance", n)
                .ok_or_else(|| anyhow::anyhow!("missing {n}"))
        };
        let bytes = (width as usize) * (height as usize) * 3;
        let tiles = 16usize;
        Ok(Self {
            wb_sum: get("wb_sum")?,
            wb_apply: get("wb_apply")?,
            hist: get("clahe_hist")?,
            lut: get("clahe_lut")?,
            apply: get("clahe_apply")?,
            blend: get("blend_orig")?,
            orig: dev.alloc_zeros(bytes).map_err(|e| anyhow::anyhow!("{e}"))?,
            sums: dev.alloc_zeros(3).map_err(|e| anyhow::anyhow!("{e}"))?,
            hist_buf: dev
                .alloc_zeros(tiles * tiles * 256)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            lut_buf: dev
                .alloc_zeros(tiles * tiles * 256)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            dev,
            width,
            height,
            bytes,
            host_out: Vec::new(),
        })
    }

    fn launch_n<A>(f: &cudarc::driver::CudaFunction, n: u32, args: A) -> Result<()>
    where
        A: cudarc::driver::LaunchArgs,
    {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let cfg = LaunchConfig::for_num_elems(n);
        unsafe { f.clone().launch(cfg, args) }.map_err(|e| anyhow::anyhow!("launch: {e}"))
    }

    fn wrap_host(&self, data: &mut [u8]) -> cudarc::driver::CudaSlice<u8> {
        unsafe {
            self.dev
                .upgrade_device_ptr(data.as_mut_ptr() as u64, data.len())
        }
    }

    fn dispatch_on(&mut self, bgr: &cudarc::driver::CudaSlice<u8>, cfg: &EnhanceCfg) -> Result<()> {
        let n = self.width * self.height;
        let mut info = make_info(self.width, self.height, cfg, 1.0, 1.0, 1.0);
        if cfg.wb {
            self.dev
                .memset_zeros(&mut self.sums)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Self::launch_n(&self.wb_sum, n, (bgr, &self.sums, info))?;
            self.dev.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut sums = [0u32; 3];
            self.dev
                .dtoh_sync_copy_into(&self.sums, &mut sums)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let inv = 1.0 / n.max(1) as f32;
            let mb = sums[0] as f32 * inv;
            let mg = sums[1] as f32 * inv;
            let mr = sums[2] as f32 * inv;
            let gray = (mb + mg + mr) / 3.0;
            let gain = |m: f32| gray / m.max(1e-3);
            info.gb = gain(mb);
            info.gg = gain(mg);
            info.gr = gain(mr);
            Self::launch_n(&self.wb_apply, n, (bgr, info))?;
        }
        if cfg.he != HeMode::Off {
            self.dev
                .memset_zeros(&mut self.hist_buf)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Self::launch_n(&self.hist, n, (bgr, &self.hist_buf, info))?;
            let ntiles = info.tiles_x * info.tiles_y;
            Self::launch_n(&self.lut, ntiles, (&self.hist_buf, &self.lut_buf, info))?;
            Self::launch_n(&self.apply, n, (bgr, &self.lut_buf, info))?;
        }
        if cfg.blend < 1.0 {
            Self::launch_n(&self.blend, n, (bgr, &self.orig, info))?;
        }
        self.dev.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn run(&mut self, src: &BgrImage, cfg: &EnhanceCfg) -> Result<&[u8]> {
        self.host_out.clear();
        self.host_out.extend_from_slice(&src.data);
        let (w, h) = (src.width, src.height);
        // Split borrow: dispatch uses other fields while host_out is the working buffer.
        let ptr = self.host_out.as_mut_ptr();
        let len = self.host_out.len();
        let data = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        self.run_in_place(data, w, h, cfg)?;
        Ok(&self.host_out)
    }

    fn run_in_place(
        &mut self,
        data: &mut [u8],
        width: u32,
        height: u32,
        cfg: &EnhanceCfg,
    ) -> Result<()> {
        if width != self.width || height != self.height || data.len() != self.bytes {
            bail!("cuda enhance size mismatch");
        }
        if cfg.blend < 1.0 {
            self.dev
                .htod_sync_copy_into(data, &mut self.orig)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let bgr = self.wrap_host(data);
        let result = self.dispatch_on(&bgr, cfg);
        std::mem::forget(bgr);
        result
    }
}

#[cfg(all(test, feature = "gpu", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::enhance::enhance_bgr;

    fn ramp() -> BgrImage {
        let (w, h) = (64u32, 64u32);
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = (20 + (x * 3 + y) % 80) as u8;
                data.extend_from_slice(&[v, v.saturating_add(10), v.saturating_add(20)]);
            }
        }
        BgrImage {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn metal_matches_cpu_within_2() {
        let src = ramp();
        let cfg = EnhanceCfg::default();
        let cpu = enhance_bgr(&src, &cfg);
        let mut gpu = GpuEnhance::new(src.width, src.height, cfg).unwrap();
        let got = gpu.run(&src).unwrap();
        let max = cpu
            .data
            .iter()
            .zip(got.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(max <= 2, "max abs {max}");
    }
}
