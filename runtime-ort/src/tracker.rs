//! Full OpenSeeFace tracking loop (`tracker.py` `Tracker.predict`).

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::crop::{stable_landmark_bbox, CropSmoothState};
use crate::decode::{decode_landmarks, detect_faces_n, LmSpec};
use crate::features::FeatureExtractor;
use crate::filter::{FilterCfg, FilterKind, FilterQuality, OutputFilter};
use crate::gaze::get_eye_state;
use crate::geom::{clamp_to_im, group_rects};
use crate::pnp::{adjust_3d, estimate_depth, Camera, CONTOUR_PTS, CONTOUR_PTS_T, FACE_3D};
use crate::preprocess::{crop_img, imagenet_nchw, BgrImage};
use crate::retinaface::RetinaFace;
use crate::session::OrtModel;

const MAP30: [usize; 66] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 21, 21,
    21, 22, 23, 23, 23, 23, 23, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 24, 25, 25, 25, 26,
    26, 27, 27, 27, 24, 24, 28, 28, 28, 26, 29, 29, 29,
];

pub fn model_base_path(model_dir: Option<&Path>) -> PathBuf {
    if let Some(d) = model_dir {
        return d.to_path_buf();
    }
    let candidates = [
        PathBuf::from("models"),
        PathBuf::from("../models"),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("models")))
            .unwrap_or_else(|| PathBuf::from("models")),
    ];
    for c in candidates {
        if c.join("mnv3_detection_opt.onnx").is_file() {
            return c;
        }
    }
    PathBuf::from("models")
}

#[derive(Clone)]
pub struct FaceInfo {
    pub id: i32,
    pub conf: f32,
    pub lms: Vec<[f32; 3]>,
    pub eye_state: [[f32; 4]; 2],
    pub rotation: Option<[f32; 3]>,
    pub translation: [f32; 3],
    pub success: bool,
    pub quaternion: [f32; 4],
    pub euler: [f32; 3],
    pub pnp_error: f32,
    pub pts_3d: [[f32; 3]; 70],
    pub eye_blink: [f32; 2],
    pub bbox: [f32; 4],
    pub current_features: [f32; 14],
    pub face_3d: Vec<[f32; 3]>,
    pub contour: Vec<[f32; 3]>,
    pub alive: bool,
    coord: Option<[f32; 2]>,
    frame_count: i32,
    fail_count: i32,
    update_counts: [[f32; 2]; 66],
    features: FeatureExtractor,
    crop_smooth: CropSmoothState,
    output_filter: OutputFilter,
}

impl FaceInfo {
    fn new(id: i32, model_type: i32, max_feature_updates: f32, filter: FilterCfg) -> Self {
        let mut s = Self {
            id,
            conf: 0.0,
            lms: Vec::new(),
            eye_state: [[1.0, 0.0, 0.0, 0.0]; 2],
            rotation: None,
            translation: [0.0; 3],
            success: false,
            quaternion: [0.0, 0.0, 0.0, 1.0],
            euler: [0.0; 3],
            pnp_error: 0.0,
            pts_3d: [[0.0; 3]; 70],
            eye_blink: [1.0, 1.0],
            bbox: [0.0; 4],
            current_features: [0.0; 14],
            face_3d: FACE_3D.to_vec(),
            contour: Vec::new(),
            alive: false,
            coord: None,
            frame_count: -1,
            fail_count: 0,
            update_counts: [[0.0; 2]; 66],
            features: FeatureExtractor::new(max_feature_updates),
            crop_smooth: CropSmoothState::default(),
            output_filter: OutputFilter::new(filter),
        };
        s.update_contour(model_type);
        s
    }

    fn contour_idx(model_type: i32) -> &'static [usize] {
        if model_type == -1 {
            &CONTOUR_PTS_T
        } else {
            &CONTOUR_PTS
        }
    }

    fn update_contour(&mut self, model_type: i32) {
        self.contour = Self::contour_idx(model_type)
            .iter()
            .map(|&i| self.face_3d[i])
            .collect();
    }

    fn reset(&mut self, model_type: i32, max_feature_updates: f32) {
        self.alive = false;
        self.conf = 0.0;
        self.lms.clear();
        self.eye_state = [[1.0, 0.0, 0.0, 0.0]; 2];
        self.rotation = None;
        self.translation = [0.0; 3];
        self.success = false;
        self.quaternion = [0.0, 0.0, 0.0, 1.0];
        self.euler = [0.0; 3];
        self.pnp_error = 0.0;
        self.pts_3d = [[0.0; 3]; 70];
        self.eye_blink = [1.0, 1.0];
        self.bbox = [0.0; 4];
        self.current_features = [0.0; 14];
        if max_feature_updates < 1.0 {
            self.features = FeatureExtractor::new(0.0);
        }
        self.update_contour(model_type);
        self.fail_count = 0;
        self.coord = None;
        self.crop_smooth.reset();
        self.output_filter.reset();
    }

    fn update_det(
        &mut self,
        conf: f32,
        lms: Vec<[f32; 3]>,
        eye: [[f32; 4]; 2],
        coord: [f32; 2],
        frame_count: i32,
        model_type: i32,
        max_feature_updates: f32,
    ) {
        self.frame_count = frame_count;
        self.conf = conf;
        self.lms = lms;
        self.eye_state = eye;
        self.coord = Some(coord);
        self.alive = true;
        let _ = (model_type, max_feature_updates);
    }
}

pub struct Tracker {
    pub width: u32,
    pub height: u32,
    spec: LmSpec,
    threshold: f32,
    detection_threshold: f32,
    max_faces: usize,
    discard_after: i32,
    scan_every: i32,
    silent: bool,
    try_hard: bool,
    no_gaze: bool,
    use_retinaface: bool,
    static_model: bool,
    feature_level: i32,
    max_feature_updates: f32,
    cam: Camera,
    det: OrtModel,
    lm: OrtModel,
    gaze: Option<OrtModel>,
    retina: Option<RetinaFace>,
    retina_scan: Option<RetinaFace>,
    faces: Vec<[f32; 4]>,
    face_info: Vec<FaceInfo>,
    detected: usize,
    discard: i32,
    wait_count: i32,
    frame_count: i32,
    model_dir: PathBuf,
    last_tick: Option<std::time::Instant>,
}

pub struct TrackerConfig {
    pub width: u32,
    pub height: u32,
    pub model_type: i32,
    pub detection_threshold: f32,
    pub threshold: Option<f32>,
    pub max_faces: usize,
    pub discard_after: i32,
    pub scan_every: i32,
    pub max_threads: usize,
    pub silent: bool,
    pub model_dir: Option<PathBuf>,
    pub no_gaze: bool,
    pub use_retinaface: bool,
    pub max_feature_updates: f32,
    pub static_model: bool,
    pub try_hard: bool,
    pub filter: FilterKind,
    pub filter_mincutoff: f32,
    pub filter_beta: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            model_type: 3,
            detection_threshold: 0.6,
            threshold: None,
            max_faces: 1,
            discard_after: 10,
            scan_every: 3,
            max_threads: 4,
            silent: false,
            model_dir: None,
            no_gaze: false,
            use_retinaface: false,
            max_feature_updates: 0.0,
            static_model: true,
            try_hard: false,
            filter: FilterKind::OneEuro,
            filter_mincutoff: 1.0,
            filter_beta: 0.007,
        }
    }
}

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Result<Self> {
        let spec = LmSpec::from_type(cfg.model_type)?;
        let mut threshold = cfg.threshold.unwrap_or(0.6);
        if cfg.threshold.is_none() && cfg.model_type < 0 {
            threshold = 0.87;
        }
        let dir = model_base_path(cfg.model_dir.as_deref());
        let threads = cfg.max_threads.max(1);
        let det = OrtModel::load(dir.join("mnv3_detection_opt.onnx"), threads.min(4))?;
        let lm = OrtModel::load(dir.join(spec.file), threads.min(4))?;
        let gaze = if cfg.no_gaze {
            None
        } else {
            OrtModel::load(dir.join("mnv3_gaze32_split_opt.onnx"), 1).ok()
        };
        let retina = if cfg.use_retinaface || cfg.try_hard {
            RetinaFace::load(
                dir.join("retinaface_640x640_opt.onnx"),
                dir.join("priorbox_640x640.json"),
                threads.max(4),
                cfg.max_faces,
            )
            .ok()
        } else {
            None
        };
        let retina_scan = if cfg.use_retinaface {
            RetinaFace::load(
                dir.join("retinaface_640x640_opt.onnx"),
                dir.join("priorbox_640x640.json"),
                2,
                cfg.max_faces,
            )
            .ok()
        } else {
            None
        };
        let max_faces = cfg.max_faces.max(1);
        let mut feature_level = 2;
        if cfg.model_type == -1 {
            feature_level = 1;
        }
        let filter_cfg = FilterCfg::new(cfg.filter, cfg.filter_mincutoff, cfg.filter_beta);
        let face_info = (0..max_faces)
            .map(|id| {
                FaceInfo::new(
                    id as i32,
                    cfg.model_type,
                    cfg.max_feature_updates,
                    filter_cfg,
                )
            })
            .collect();
        Ok(Self {
            width: cfg.width,
            height: cfg.height,
            spec,
            threshold,
            detection_threshold: cfg.detection_threshold,
            max_faces,
            discard_after: cfg.discard_after,
            scan_every: cfg.scan_every,
            silent: cfg.silent,
            try_hard: cfg.try_hard,
            no_gaze: cfg.no_gaze || gaze.is_none(),
            use_retinaface: cfg.use_retinaface,
            static_model: cfg.static_model,
            feature_level,
            max_feature_updates: cfg.max_feature_updates,
            cam: Camera::from_frame(cfg.width, cfg.height),
            det,
            lm,
            gaze,
            retina,
            retina_scan,
            faces: Vec::new(),
            face_info,
            detected: 0,
            discard: 0,
            wait_count: 0,
            frame_count: 0,
            model_dir: dir,
            last_tick: None,
        })
    }

    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.cam = Camera::from_frame(width, height);
    }

    pub fn models_dir(&self) -> &Path {
        &self.model_dir
    }

    fn remap_lms(&self, conf: f32, mut lms: Vec<[f32; 3]>) -> (f32, Vec<[f32; 3]>) {
        if self.spec.model_type != -1 {
            return (conf, lms);
        }
        let mapped: Vec<[f32; 3]> = MAP30
            .iter()
            .map(|&i| lms.get(i).copied().unwrap_or([0.0; 3]))
            .collect();
        lms = mapped;
        let mut cs: Vec<f32> = lms.iter().map(|p| p[2]).collect();
        cs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let part_avg = if cs.len() >= 3 {
            (cs[0] + cs[1] + cs[2]) / 3.0
        } else {
            conf
        };
        let conf = if part_avg < 0.65 { part_avg } else { conf };
        (conf, lms)
    }

    fn detect_heatmap(&mut self, frame: &BgrImage) -> Vec<[f32; 4]> {
        let im = imagenet_nchw(frame, 224);
        let Ok(out) = self.det.run(&im) else {
            return Vec::new();
        };
        if out.len() < 2 {
            return Vec::new();
        }
        detect_faces_n(
            &out[0],
            &out[1],
            frame.width,
            frame.height,
            self.detection_threshold,
            self.max_faces,
        )
        .into_iter()
        .map(|d| [d[0], d[1], d[2], d[3]])
        .collect()
    }

    fn assign_face_info(&mut self, results: Vec<(f32, Vec<[f32; 3]>, [[f32; 4]; 2], f32)>) {
        if self.max_faces == 1 && results.len() == 1 {
            let (conf, lms, eye, _) = results.into_iter().next().unwrap();
            let coord = mean_xy(&lms);
            self.face_info[0].update_det(
                conf,
                lms,
                eye,
                coord,
                self.frame_count,
                self.spec.model_type,
                self.max_feature_updates,
            );
            return;
        }
        let coords: Vec<[f32; 2]> = results.iter().map(|r| mean_xy(&r.1)).collect();
        let max_dist = 2.0 * ((self.width * self.width + self.height * self.height) as f32).sqrt();
        let n_res = results.len();
        let mut used_res = vec![false; n_res];
        let mut used_face = vec![false; self.max_faces];
        let mut found = 0;
        while found < n_res {
            let mut best = (f32::MAX, 0usize, 0usize);
            for i in 0..self.max_faces {
                if used_face[i] {
                    continue;
                }
                for j in 0..n_res {
                    if used_res[j] {
                        continue;
                    }
                    let d = match self.face_info[i].coord {
                        None => max_dist,
                        Some(c) => (c[0] - coords[j][0]).hypot(c[1] - coords[j][1]),
                    };
                    if d < best.0 {
                        best = (d, i, j);
                    }
                }
            }
            if best.0 == f32::MAX {
                break;
            }
            let (_, fi, ri) = best;
            let (conf, lms, eye, _) = results[ri].clone();
            self.face_info[fi].update_det(
                conf,
                lms,
                eye,
                coords[ri],
                self.frame_count,
                self.spec.model_type,
                self.max_feature_updates,
            );
            used_res[ri] = true;
            used_face[fi] = true;
            found += 1;
        }
        for fi in &mut self.face_info {
            if fi.frame_count != self.frame_count {
                fi.reset(self.spec.model_type, self.max_feature_updates);
                fi.frame_count = self.frame_count;
            }
        }
    }

    pub fn predict(&mut self, frame: &BgrImage) -> Vec<FaceInfo> {
        self.frame_count += 1;
        self.wait_count += 1;
        let now = std::time::Instant::now();
        let dt = self
            .last_tick
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(1.0 / 30.0)
            .clamp(1.0 / 240.0, 0.25);
        self.last_tick = Some(now);
        let mut new_faces: Vec<[f32; 4]> = self.faces.clone();
        let bonus_cutoff = new_faces.len();
        if self.detected == 0 {
            if (self.use_retinaface || self.try_hard) && self.retina.is_some() {
                if let Ok(d) = self.retina.as_mut().unwrap().detect(frame) {
                    new_faces.extend(d);
                }
            }
            if !self.use_retinaface || self.try_hard {
                new_faces.extend(self.detect_heatmap(frame));
            }
            if self.try_hard {
                new_faces.push([0.0, 0.0, self.width as f32, self.height as f32]);
            }
            self.wait_count = 0;
        } else if self.detected < self.max_faces {
            if self.use_retinaface {
                if let Some(s) = self.retina_scan.as_mut() {
                    new_faces.extend(s.get_results());
                }
            }
            if self.wait_count >= self.scan_every {
                if self.use_retinaface {
                    if let Some(s) = self.retina_scan.as_mut() {
                        s.background_detect(frame);
                    }
                } else {
                    new_faces.extend(self.detect_heatmap(frame));
                    self.wait_count = 0;
                }
            }
        } else {
            self.wait_count = 0;
        }

        if new_faces.is_empty() {
            if !self.silent {
                eprintln!("Took 0.00ms");
            }
            return Vec::new();
        }

        let res = self.spec.size as f32;
        let mut crops = Vec::new();
        for (j, box4) in new_faces.iter().enumerate() {
            let (x, y, w, h) = (box4[0], box4[1], box4[2], box4[3]);
            let (x1, y1) = clamp_to_im(
                x - (w * 0.1) as i32 as f32,
                y - (h * 0.125) as i32 as f32,
                self.width as f32,
                self.height as f32,
            );
            let (x2, y2) = clamp_to_im(
                x + w + (w * 0.1) as i32 as f32,
                y + h + (h * 0.125) as i32 as f32,
                self.width as f32,
                self.height as f32,
            );
            if x2 - x1 < 4 || y2 - y1 < 4 {
                continue;
            }
            let scale_x = (x2 - x1) as f32 / res;
            let scale_y = (y2 - y1) as f32 / res;
            let crop = crop_img(frame, x1, y1, x2, y2);
            let tensor = imagenet_nchw(&crop, self.spec.size);
            let bonus = if j >= bonus_cutoff { 0.0 } else { 0.1 };
            crops.push((tensor, [x1 as f32, y1 as f32, scale_x, scale_y], bonus));
        }

        let mut raw = Vec::new();
        for (tensor, crop, bonus) in &crops {
            let Ok(out) = self.lm.run(tensor) else {
                continue;
            };
            let (conf, lms) = decode_landmarks(&out[0], *crop, self.spec);
            let (conf, lms) = self.remap_lms(conf, lms);
            if conf <= self.threshold {
                continue;
            }
            let eye = if let Some(g) = self.gaze.as_mut() {
                get_eye_state(g, frame, &lms, self.no_gaze).unwrap_or([[1.0, 0.0, 0.0, 0.0]; 2])
            } else {
                [[1.0, 0.0, 0.0, 0.0]; 2]
            };
            raw.push((conf, lms, eye, *bonus, *crop));
        }

        let bbs: Vec<[f32; 4]> = raw
            .iter()
            .map(|(_, lms, _, _, _)| {
                let (mut x1, mut y1, mut x2, mut y2) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for p in lms {
                    x1 = x1.min(p[0]);
                    y1 = y1.min(p[1]);
                    x2 = x2.max(p[0]);
                    y2 = y2.max(p[1]);
                }
                [x1, y1, x2 - x1, y2 - y1]
            })
            .collect();
        let groups = group_rects(&bbs);
        let mut best: Vec<Option<(f32, usize)>> = vec![None; raw.len().max(1)];
        for (i, (conf, _, _, bonus, _)) in raw.iter().enumerate() {
            let g = groups.get(i).copied().unwrap_or(i);
            if g >= best.len() {
                best.resize(g + 1, None);
            }
            let score = conf + bonus;
            let replace = best[g].map(|(s, _)| score > s).unwrap_or(true);
            if *conf > self.threshold && replace {
                best[g] = Some((score, i));
            }
        }
        let mut picked: Vec<(f32, Vec<[f32; 3]>, [[f32; 4]; 2], f32)> = best
            .into_iter()
            .flatten()
            .map(|(_, i)| {
                let (c, l, e, b, _) = raw[i].clone();
                (c, l, e, b)
            })
            .collect();
        picked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        picked.truncate(self.max_faces);
        self.assign_face_info(picked);

        let mut results = Vec::new();
        let mut detected = Vec::new();
        let model_type = self.spec.model_type;
        let static_model = self.static_model;
        let feature_level = self.feature_level;
        let cam = self.cam;
        let max_fu = self.max_feature_updates;
        for fi in &mut self.face_info {
            if !(fi.alive && fi.conf > self.threshold) {
                continue;
            }
            let prev = fi.rotation.map(|r| (r, fi.translation));
            let depth = estimate_depth(
                &fi.lms,
                &fi.eye_state,
                &fi.face_3d,
                FaceInfo::contour_idx(model_type),
                &cam,
                prev,
            );
            fi.success = depth.success;
            fi.quaternion = depth.quaternion;
            fi.euler = depth.euler;
            fi.pnp_error = depth.pnp_error;
            fi.pts_3d = depth.pts_3d;
            fi.lms = depth.lms;
            fi.rotation = Some(depth.rotation);
            fi.translation = depth.translation;
            if depth.pnp_error > 300.0 {
                fi.fail_count += 1;
                if fi.fail_count > 5 {
                    if !self.silent {
                        eprintln!(
                            "Detected anomaly when 3D fitting face {}. Resetting.",
                            fi.id
                        );
                    }
                    fi.face_3d = FACE_3D.to_vec();
                    fi.rotation = None;
                    fi.translation = [0.0; 3];
                    fi.update_counts = [[0.0; 2]; 66];
                    fi.update_contour(model_type);
                    fi.crop_smooth.reset();
                    fi.output_filter.reset();
                }
            } else {
                fi.fail_count = 0;
            }
            adjust_3d(
                &mut fi.face_3d,
                &mut fi.pts_3d,
                &fi.lms,
                fi.euler,
                depth.rotation,
                depth.translation,
                &cam,
                fi.conf,
                fi.pnp_error,
                static_model,
                model_type,
                &mut fi.update_counts,
                feature_level,
                &mut fi.features,
                &mut fi.current_features,
                &mut fi.eye_blink,
            );
            fi.update_contour(model_type);
            let mut x1 = f32::MAX;
            let mut y1 = f32::MAX;
            let mut x2 = f32::MIN;
            let mut y2 = f32::MIN;
            for p in fi.lms.iter().take(66) {
                x1 = x1.min(p[0]);
                y1 = y1.min(p[1]);
                x2 = x2.max(p[0]);
                y2 = y2.max(p[1]);
            }
            fi.bbox = [y1, x1, y2 - y1, x2 - x1];
            fi.crop_smooth.seed_size(fi.bbox);
            let next = stable_landmark_bbox(&fi.lms, Some(&mut fi.crop_smooth))
                .map(|b| [b[0], b[1], b[2], b[3]])
                .or_else(|| fi.crop_smooth.last_box())
                .unwrap_or([0.0, 0.0, 1.0, 1.0]);
            detected.push(next);
            let mut out = fi.clone();
            fi.output_filter.apply(
                &mut out.euler,
                &mut out.translation,
                &mut out.quaternion,
                &mut out.lms,
                &mut out.pts_3d,
                dt,
                FilterQuality {
                    conf: fi.conf,
                    pnp_error: fi.pnp_error,
                    success: fi.success,
                },
            );
            results.push(out);
            let _ = max_fu;
        }

        if !detected.is_empty() {
            self.detected = detected.len();
            self.faces = detected;
            self.discard = 0;
        } else {
            self.detected = 0;
            self.discard += 1;
            if self.discard > self.discard_after {
                self.faces.clear();
            }
        }
        self.faces.retain(|b| b.iter().all(|v| !v.is_nan()));
        self.detected = self.faces.len();
        results.sort_by_key(|f| f.id);
        results
    }
}

fn mean_xy(lms: &[[f32; 3]]) -> [f32; 2] {
    if lms.is_empty() {
        return [0.0, 0.0];
    }
    let n = lms.len() as f32;
    [
        lms.iter().map(|p| p[0]).sum::<f32>() / n,
        lms.iter().map(|p| p[1]).sum::<f32>() / n,
    ]
}

impl Clone for FeatureExtractor {
    fn clone(&self) -> Self {
        FeatureExtractor::new(0.0)
    }
}
