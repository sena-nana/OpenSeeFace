//! CPU vs GPU I/O for ONNX Runtime sessions.
//!
//! Host path (CPU EP and CoreML) binds host tensors. CUDA path uses pinned host
//! buffers so ORT can DMA without an extra pageable copy. Intermediate device
//! tensors for the detect→landmark pipeline live in `gpu_pre`.

use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use half::f16;
use ort::logging::LogLevel;
use ort::memory::Allocator;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, Session};
use ort::value::{Tensor, TensorRef, ValueType};
use ort::{api, AsPointer};

use crate::decode::TensorF16;

#[cfg(feature = "gpu")]
use ort::ep::ExecutionProvider;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

/// `cpu` or `gpu` (CoreML on Apple, CUDA on NVIDIA).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => bail!("unknown device {s:?}, expected cpu|gpu"),
        }
    }
}

pub(crate) fn oe(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

pub(crate) fn gpu_eps() -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
    #[cfg(not(feature = "gpu"))]
    bail!("GPU requested; rebuild with `--features gpu`");

    #[cfg(feature = "gpu")]
    {
        #[allow(unused_mut)]
        let mut eps = Vec::new();
        #[cfg(target_os = "macos")]
        if ort::ep::CoreML::default().is_available().unwrap_or(false) {
            eps.push(
                ort::ep::CoreML::default()
                    .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU)
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .build()
                    .error_on_failure(),
            );
        }
        #[cfg(not(target_os = "macos"))]
        if ort::ep::CUDA::default().is_available().unwrap_or(false) {
            eps.push(ort::ep::CUDA::default().build().error_on_failure());
        }
        if eps.is_empty() {
            bail!("GPU requested but no GPU execution provider is available");
        }
        Ok(eps)
    }
}

pub(crate) fn make_session(
    path: &Path,
    threads: usize,
    device: Device,
    batch: i64,
    dims: &[(&str, i64)],
) -> Result<(Session, f64)> {
    let start = Instant::now();
    let mut builder = Session::builder()
        .map_err(oe)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(oe)?
        .with_intra_threads(threads.max(1))
        .map_err(oe)?
        .with_inter_threads(1)
        .map_err(oe)?
        .with_parallel_execution(false)
        .map_err(oe)?
        .with_memory_pattern(true)
        .map_err(oe)?
        .with_log_level(LogLevel::Error)
        .map_err(oe)?;
    if device == Device::Gpu {
        builder = builder
            .with_dimension_override("batch_size", batch.max(1))
            .map_err(oe)?
            .with_execution_providers(gpu_eps()?)
            .map_err(oe)?;
        for (name, size) in dims {
            builder = builder.with_dimension_override(*name, *size).map_err(oe)?;
        }
    }
    let session = builder.commit_from_file(path).map_err(oe)?;
    Ok((session, start.elapsed().as_secs_f64() * 1000.0))
}

struct BoundIo {
    binding: IoBinding,
    input: Tensor<f16>,
    input_shape: Vec<i64>,
}

pub struct OrtModel {
    session: Session,
    bound: Option<BoundIo>,
    pub input_name: String,
    pub load_ms: f64,
    device: Device,
}

impl OrtModel {
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Self> {
        Self::open(path, threads, Device::Cpu, 1)
    }

    pub fn open(
        path: impl AsRef<Path>,
        threads: usize,
        device: Device,
        batch: i64,
    ) -> Result<Self> {
        Self::open_dims(path, threads, device, batch, &[])
    }

    pub fn open_dims(
        path: impl AsRef<Path>,
        threads: usize,
        device: Device,
        batch: i64,
        dims: &[(&str, i64)],
    ) -> Result<Self> {
        let path = path.as_ref();
        let (session, load_ms) = make_session(path, threads, device, batch, dims)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".into());
        Ok(Self {
            session,
            bound: None,
            input_name,
            load_ms,
            device,
        })
    }

    fn allocs(&self) -> Result<(Allocator, Allocator)> {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        if self.device == Device::Gpu {
            let pin_in = Allocator::new(
                &self.session,
                MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    0,
                    AllocatorType::Device,
                    MemoryType::CPUInput,
                )
                .map_err(oe)?,
            );
            let pin_out = Allocator::new(
                &self.session,
                MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    0,
                    AllocatorType::Device,
                    MemoryType::CPUOutput,
                )
                .map_err(oe)?,
            );
            if let (Ok(i), Ok(o)) = (pin_in, pin_out) {
                return Ok((i, o));
            }
        }
        let _ = self.device;
        Ok((Allocator::default(), Allocator::default()))
    }

    fn output_specs(&self, batch: i64) -> Result<Vec<(String, Vec<i64>)>> {
        let mut specs = Vec::new();
        for o in self.session.outputs() {
            let ValueType::Tensor { shape, .. } = o.dtype() else {
                bail!("expected tensor output {}", o.name());
            };
            let mut dims: Vec<i64> = shape.iter().copied().collect();
            if dims.first().is_some_and(|d| *d <= 0) {
                dims[0] = batch;
            }
            specs.push((o.name().to_string(), dims));
        }
        Ok(specs)
    }

    fn ensure_bound(&mut self, shape: &[i64]) -> Result<()> {
        if self.bound.as_ref().is_some_and(|b| b.input_shape == shape) {
            return Ok(());
        }
        let batch = shape.first().copied().unwrap_or(1);
        let mut specs = self.output_specs(batch)?;
        if specs.iter().any(|(_, s)| s.iter().any(|d| *d <= 0)) {
            let dummy = TensorF16::zeros(shape.to_vec());
            let probe = self.run_unbound(&dummy)?;
            specs = self
                .session
                .outputs()
                .iter()
                .zip(&probe)
                .map(|(o, t)| (o.name().to_string(), t.shape.clone()))
                .collect();
        }
        let (in_alloc, out_alloc) = self.allocs()?;
        let input_t = Tensor::<f16>::new(&in_alloc, shape).map_err(oe)?;
        let mut binding = self.session.create_binding().map_err(oe)?;
        binding
            .bind_input(self.input_name.clone(), &input_t)
            .map_err(oe)?;
        for (name, out_shape) in specs {
            binding
                .bind_output(name, Tensor::<f16>::new(&out_alloc, out_shape).map_err(oe)?)
                .map_err(oe)?;
        }
        self.bound = Some(BoundIo {
            binding,
            input: input_t,
            input_shape: shape.to_vec(),
        });
        Ok(())
    }

    fn refill_input(&mut self, prep: impl FnOnce(&mut [f16])) -> Result<()> {
        let name = self.input_name.clone();
        let bound = self.bound.as_mut().context("unbound")?;
        {
            let (_, buf) = bound.input.extract_tensor_mut();
            prep(buf);
        }
        bound.binding.bind_input(name, &bound.input).map_err(oe)
    }

    fn run_unbound(&mut self, input: &TensorF16) -> Result<Vec<TensorF16>> {
        let tensor = TensorRef::from_array_view((input.shape.as_slice(), input.data.as_slice()))
            .map_err(oe)?;
        collect_f16(
            &self
                .session
                .run(ort::inputs![self.input_name.as_str() => tensor])
                .map_err(oe)?,
        )
    }

    pub fn infer(&mut self) -> Result<()> {
        let OrtModel { session, bound, .. } = self;
        let binding = &bound.as_ref().context("infer() needs run() first")?.binding;
        let status = unsafe {
            (api().RunWithBinding)(session.ptr().cast_mut(), core::ptr::null(), binding.ptr())
        };
        unsafe { ort::Error::result_from_status(status) }.map_err(oe)
    }

    /// Write preprocess into the bound input, run, then decode from output slices.
    pub fn run_prep<R>(
        &mut self,
        shape: &[i64],
        prep: impl FnOnce(&mut [f16]),
        then: impl FnOnce(&[&[f16]]) -> Result<R>,
    ) -> Result<R> {
        self.ensure_bound(shape)?;
        self.refill_input(prep)?;
        let OrtModel { session, bound, .. } = self;
        let outs = session
            .run_binding(&bound.as_ref().context("unbound")?.binding)
            .map_err(oe)?;
        let result = decode_outputs(&outs, then)?;
        Ok(result)
    }

    pub fn run(&mut self, input: &TensorF16) -> Result<Vec<TensorF16>> {
        self.ensure_bound(&input.shape)?;
        self.refill_input(|buf| buf.copy_from_slice(&input.data))?;
        let OrtModel { session, bound, .. } = self;
        collect_f16(
            &session
                .run_binding(&bound.as_ref().context("unbound")?.binding)
                .map_err(oe)?,
        )
    }
}

pub(crate) fn collect_f16(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<TensorF16>> {
    let mut out = Vec::with_capacity(outputs.len());
    for i in 0..outputs.len() {
        let (shape, data) = outputs[i].try_extract_tensor::<f16>().map_err(oe)?;
        out.push(TensorF16 {
            shape: shape.iter().copied().collect(),
            data: data.to_vec(),
        });
    }
    if out.is_empty() {
        bail!("empty model output");
    }
    Ok(out)
}

pub(crate) fn output_f16<'s>(
    outputs: &'s ort::session::SessionOutputs<'_>,
    i: usize,
) -> Result<&'s [f16]> {
    let (_, data) = outputs[i].try_extract_tensor::<f16>().map_err(oe)?;
    Ok(data)
}

fn decode_outputs<R>(
    outs: &ort::session::SessionOutputs<'_>,
    then: impl FnOnce(&[&[f16]]) -> Result<R>,
) -> Result<R> {
    let a = output_f16(outs, 0)?;
    if outs.len() == 1 {
        return then(&[a]);
    }
    let b = output_f16(outs, 1)?;
    then(&[a, b])
}
