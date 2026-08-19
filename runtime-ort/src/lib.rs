//! ORT inference and the OpenSeeFace tracker (`facetracker` binary).

mod adaptive;
mod capture;
mod crop;
mod decode;
mod enhance;
mod enhance_gpu;
mod ext;
mod features;
mod filter;
mod gaze;
mod geom;
mod glasses;
mod gpu_pre;
mod metrics;
mod pnp;
mod preprocess;
mod retinaface;
mod session;
mod tracker;
mod udp;
mod viz;
mod vmc;
mod vrm;

pub use adaptive::{
    center_2x, det_window, face_on_224, nme, pick_lm, AdaptiveCfg, AdaptiveState, DetWindow,
    FAST_LM,
};
pub use capture::{list_cameras, mirror_bgr, InputSource, PipedInput, VideoOut};
pub use crop::{stable_landmark_bbox, CropSmoothState, CropTrack};
pub use decode::{
    decode_landmarks, decode_landmarks_data, detect_faces, detect_faces_n, landmark_bbox,
    mean_conf, LmSpec, TensorF16, EYE_IDX,
};
pub use enhance::{enhance_bgr, enhance_bgr_in_place, EnhanceCfg, HeMode};
pub use ext::{ExtListener, ExtState, VISEME_COUNT, VISEME_NAMES};
pub use features::{
    FeatureVec, FEATURE_COUNT, FEATURE_NAMES, FEAT_CHEEK_PUFF, FEAT_JAW_OPEN, FEAT_MOUTH_FUNNEL,
    FEAT_MOUTH_OFFSET_X, FEAT_MOUTH_PRESS_LIP_OPEN, FEAT_MOUTH_PUCKER,
};
pub use filter::{unwrap_deg, FilterCfg, FilterKind, FilterQuality, OutputFilter};
pub use geom::xywh_iou;
pub use glasses::{ear_2d, paint_synthetic_glasses};
pub use gpu_pre::GpuTracker;
pub use metrics::{cosine, max_abs, mean_abs, model_path, read_f32_le, rss, Latency, Rss};
pub use pnp::Camera;
pub use preprocess::{
    crop_box, crop_box_pad, crop_img, face_crop, imagenet_nchw, imagenet_nchw_roi_into, iou, nchw,
    paste_bgr, resize_bgr, retina_nchw, synth_canvas, BgrImage, ColorNorm,
};
pub use session::{Device, OrtModel};
pub use tracker::{model_base_path, FaceInfo, Tracker, TrackerConfig};
pub use udp::{
    encode_face, encode_faces, encode_faces_into, FacePacket, PACKET_FRAME_SIZE,
    PACKET_FRAME_SIZE_LEGACY,
};
pub use viz::{draw_tracking, dump_symmetric_points, VizWindow};
pub use vmc::encode_vmc;
pub use vrm::{VrmCfg, VrmDriver, VrmFrame};

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

    #[test]
    fn resize_bgr_identity() {
        let im = solid(9, 10, 11);
        let got = resize_bgr(&im, 2, 2);
        assert_eq!(got.width, 2);
        assert_eq!(got.data, im.data);
    }

    #[test]
    fn resize_roi_matches_crop() {
        let mut im = BgrImage::zeros(40, 30);
        for i in 0..im.data.len() {
            im.data[i] = (i % 251) as u8;
        }
        let crop = crop_img(&im, 5, 7, 21, 23);
        let via_crop = resize_bgr(&crop, 16, 16);
        let mut direct = vec![0u8; 16 * 16 * 3];
        crate::preprocess::resize_roi_into(&im.data, 40, 30, 5, 7, 21, 23, 16, 16, &mut direct);
        assert_eq!(via_crop.data, direct);
    }

    #[test]
    fn nchw_roi_matches_crop() {
        let mut im = BgrImage::zeros(40, 30);
        for i in 0..im.data.len() {
            im.data[i] = (i % 251) as u8;
        }
        let crop = crop_img(&im, 5, 7, 21, 23);
        let a = imagenet_nchw(&crop, 16);
        let mut b = vec![half::f16::ZERO; 3 * 16 * 16];
        imagenet_nchw_roi_into(&im, 5, 7, 21, 23, 16, &mut b);
        for (x, y) in a.data.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn flip_h_reverses_rows() {
        let im = BgrImage {
            width: 2,
            height: 1,
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let got = im.flip_h();
        assert_eq!(got.data, vec![4, 5, 6, 1, 2, 3]);
    }
}
