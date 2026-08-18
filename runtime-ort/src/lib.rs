//! ort inference for the same ONNX files used by the Python tracker.

mod decode;
mod gpu_pre;
mod metrics;
mod preprocess;
mod session;

pub use decode::{decode_landmarks, detect_faces, LmSpec, TensorF16};
pub use gpu_pre::GpuTracker;
pub use metrics::{cosine, max_abs, mean_abs, model_path, read_f32_le, rss, Latency, Rss};
pub use preprocess::{
    crop_box, crop_img, imagenet_nchw, iou, nchw, retina_nchw, BgrImage, ColorNorm,
};
pub use session::{Device, OrtModel};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(b: u8, g: u8, r: u8) -> BgrImage {
        BgrImage {
            width: 2,
            height: 2,
            data: vec![b, g, r].repeat(4),
        }
    }

    #[test]
    fn imagenet_lut_matches_affine() {
        let n = ColorNorm::IMAGENET;
        for v in [0u8, 1, 17, 128, 255] {
            for c in 0..3 {
                let got = n.lut[c][v as usize].to_f32();
                let exp = v as f32 * n.scale[c] + n.bias[c];
                assert!((got - exp).abs() < 2e-3, "c={c} v={v} {got} vs {exp}");
            }
        }
    }

    #[test]
    fn imagenet_swaps_bgr_to_rgb() {
        let red = imagenet_nchw(&solid(0, 0, 255), 2);
        let n = ColorNorm::IMAGENET;
        let exp_r = 255.0 * n.scale[0] + n.bias[0];
        let exp_g = n.bias[1];
        let exp_b = n.bias[2];
        assert!((red.data[0].to_f32() - exp_r).abs() < 2e-3);
        assert!((red.data[4].to_f32() - exp_g).abs() < 2e-3);
        assert!((red.data[8].to_f32() - exp_b).abs() < 2e-3);
    }

    #[test]
    fn retina_keeps_bgr() {
        let blue = retina_nchw(&BgrImage {
            width: 640,
            height: 640,
            data: vec![200u8, 10, 3].repeat(640 * 640),
        });
        assert!((blue.data[0].to_f32() - (200.0 - 104.0)).abs() < 2e-3);
        assert!((blue.data[640 * 640].to_f32() - (10.0 - 117.0)).abs() < 2e-3);
        assert!((blue.data[2 * 640 * 640].to_f32() - (3.0 - 123.0)).abs() < 2e-3);
    }
}
