//! PnP, 3D face model, and pose — matching `Tracker.estimate_depth` / `FaceInfo.adjust_3d`.

use nalgebra::{Matrix3, Vector3};
use rand::Rng;

use crate::decode::{EYE_IDX, mean_conf};
use crate::geom::matrix_to_quaternion;

pub const FACE_3D: [[f32; 3]; 70] = [
    [0.4551769692672, 0.300895790030204, -0.764429433974752],
    [0.448998827123556, 0.166995837790733, -0.765143004071253],
    [0.437431554952677, 0.022655479179981, -0.739267175112735],
    [0.415033422928434, -0.088941454648772, -0.747947437846473],
    [0.389123587370091, -0.232380029794684, -0.704788385327458],
    [0.334630113904382, -0.361265387599081, -0.615587579236862],
    [0.263725112132858, -0.460009725616771, -0.491479221041573],
    [0.16241621322721, -0.558037146073869, -0.339445180872282],
    [0.0, -0.621079019321682, -0.287294770748887],
    [-0.16241621322721, -0.558037146073869, -0.339445180872282],
    [-0.263725112132858, -0.460009725616771, -0.491479221041573],
    [-0.334630113904382, -0.361265387599081, -0.615587579236862],
    [-0.389123587370091, -0.232380029794684, -0.704788385327458],
    [-0.415033422928434, -0.088941454648772, -0.747947437846473],
    [-0.437431554952677, 0.022655479179981, -0.739267175112735],
    [-0.448998827123556, 0.166995837790733, -0.765143004071253],
    [-0.4551769692672, 0.300895790030204, -0.764429433974752],
    [0.385529968662985, 0.402800553948697, -0.310031082540741],
    [0.322196658344302, 0.464439136821772, -0.250558059367669],
    [0.25409760441282, 0.46420381416882, -0.208177722146526],
    [0.186875436782135, 0.44706071961879, -0.145299823706503],
    [0.120880983543622, 0.423566314072968, -0.110757158774771],
    [-0.120880983543622, 0.423566314072968, -0.110757158774771],
    [-0.186875436782135, 0.44706071961879, -0.145299823706503],
    [-0.25409760441282, 0.46420381416882, -0.208177722146526],
    [-0.322196658344302, 0.464439136821772, -0.250558059367669],
    [-0.385529968662985, 0.402800553948697, -0.310031082540741],
    [0.0, 0.293332603215811, -0.137582088779393],
    [0.0, 0.194828701837823, -0.069158109325951],
    [0.0, 0.103844017393155, -0.009151819844964],
    [0.0, 0.0, 0.0],
    [0.080626352317973, -0.041276068128093, -0.134161035564826],
    [0.046439347377934, -0.057675223874769, -0.102990627164664],
    [0.0, -0.068753126205604, -0.090545348482397],
    [-0.046439347377934, -0.057675223874769, -0.102990627164664],
    [-0.080626352317973, -0.041276068128093, -0.134161035564826],
    [0.315905195966084, 0.298337502555443, -0.285107407636464],
    [0.275252345439353, 0.312721904921771, -0.244558251170671],
    [0.176394511553111, 0.311907184376107, -0.219205360345231],
    [0.131229723798772, 0.284447361805627, -0.234239149487417],
    [0.184124948330084, 0.260179585304867, -0.226590776513707],
    [0.279433549294448, 0.267363071770222, -0.248441437111633],
    [-0.131229723798772, 0.284447361805627, -0.234239149487417],
    [-0.176394511553111, 0.311907184376107, -0.219205360345231],
    [-0.275252345439353, 0.312721904921771, -0.244558251170671],
    [-0.315905195966084, 0.298337502555443, -0.285107407636464],
    [-0.279433549294448, 0.267363071770222, -0.248441437111633],
    [-0.184124948330084, 0.260179585304867, -0.226590776513707],
    [0.121155252430729, -0.208988660580347, -0.160606287940521],
    [0.041356305910044, -0.194484199722098, -0.096159882202821],
    [0.0, -0.205180167345702, -0.083299217789729],
    [-0.041356305910044, -0.194484199722098, -0.096159882202821],
    [-0.121155252430729, -0.208988660580347, -0.160606287940521],
    [-0.132325402795928, -0.290857984604968, -0.187067868218105],
    [-0.064137791831655, -0.325377847425684, -0.158924039726607],
    [0.0, -0.343742581679188, -0.113925986025684],
    [0.064137791831655, -0.325377847425684, -0.158924039726607],
    [0.132325402795928, -0.290857984604968, -0.187067868218105],
    [0.181481567104525, -0.243239316141725, -0.231284988892766],
    [0.083999507750469, -0.239717753728704, -0.155256465640701],
    [0.0, -0.256058040176369, -0.0950619498899],
    [-0.083999507750469, -0.239717753728704, -0.155256465640701],
    [-0.181481567104525, -0.243239316141725, -0.231284988892766],
    [-0.074036069749345, -0.250689938345682, -0.177346470406188],
    [0.0, -0.264945854681568, -0.112349967428413],
    [0.074036069749345, -0.250689938345682, -0.177346470406188],
    [0.257990002632141, 0.276080012321472, -0.219998998939991],
    [-0.257990002632141, 0.276080012321472, -0.219998998939991],
    [0.257990002632141, 0.276080012321472, -0.324570998549461],
    [-0.257990002632141, 0.276080012321472, -0.324570998549461],
];

pub const CONTOUR_PTS: [usize; 14] = [0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35];
pub const CONTOUR_PTS_T: [usize; 8] = [0, 2, 8, 14, 16, 27, 30, 33];

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl Camera {
    pub fn from_frame(width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        Self {
            fx: w,
            fy: w,
            cx: w / 2.0,
            cy: h / 2.0,
        }
    }

    pub fn matrix(&self) -> Matrix3<f32> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn inverse(&self) -> Matrix3<f32> {
        self.matrix()
            .try_inverse()
            .unwrap_or_else(Matrix3::identity)
    }
}

pub fn rodrigues(rvec: [f32; 3]) -> Matrix3<f32> {
    let r = Vector3::new(rvec[0], rvec[1], rvec[2]);
    let theta = r.norm();
    if theta < 1e-12 {
        return Matrix3::identity();
    }
    let k = r / theta;
    let c = theta.cos();
    let s = theta.sin();
    let oc = 1.0 - c;
    Matrix3::new(
        c + k.x * k.x * oc,
        k.x * k.y * oc - k.z * s,
        k.x * k.z * oc + k.y * s,
        k.y * k.x * oc + k.z * s,
        c + k.y * k.y * oc,
        k.y * k.z * oc - k.x * s,
        k.z * k.x * oc - k.y * s,
        k.z * k.y * oc + k.x * s,
        c + k.z * k.z * oc,
    )
}

pub fn project_points(
    pts: &[[f32; 3]],
    rvec: [f32; 3],
    tvec: [f32; 3],
    cam: &Camera,
) -> Vec<[f32; 2]> {
    let r = rodrigues(rvec);
    let t = Vector3::new(tvec[0], tvec[1], tvec[2]);
    pts.iter()
        .map(|p| {
            let x = r * Vector3::new(p[0], p[1], p[2]) + t;
            let z = if x.z.abs() < 1e-8 { 1e-8 } else { x.z };
            [cam.fx * x.x / z + cam.cx, cam.fy * x.y / z + cam.cy]
        })
        .collect()
}

fn rodrigues_f64(rvec: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    if theta < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let k = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta];
    let c = theta.cos();
    let s = theta.sin();
    let oc = 1.0 - c;
    [
        [
            c + k[0] * k[0] * oc,
            k[0] * k[1] * oc - k[2] * s,
            k[0] * k[2] * oc + k[1] * s,
        ],
        [
            k[1] * k[0] * oc + k[2] * s,
            c + k[1] * k[1] * oc,
            k[1] * k[2] * oc - k[0] * s,
        ],
        [
            k[2] * k[0] * oc - k[1] * s,
            k[2] * k[1] * oc + k[0] * s,
            c + k[2] * k[2] * oc,
        ],
    ]
}

fn residuals_f64(params: &[f64; 6], obj: &[[f32; 3]], img: &[[f32; 2]], cam: &Camera) -> Vec<f64> {
    let rotation = rodrigues_f64([params[0], params[1], params[2]]);
    let t = [params[3], params[4], params[5]];
    let fx = cam.fx as f64;
    let fy = cam.fy as f64;
    let cx = cam.cx as f64;
    let cy = cam.cy as f64;
    let mut r = Vec::with_capacity(obj.len() * 2);
    for (p, q) in obj.iter().zip(img) {
        let x = rotation[0][0] * p[0] as f64
            + rotation[0][1] * p[1] as f64
            + rotation[0][2] * p[2] as f64
            + t[0];
        let y = rotation[1][0] * p[0] as f64
            + rotation[1][1] * p[1] as f64
            + rotation[1][2] * p[2] as f64
            + t[1];
        let z = rotation[2][0] * p[0] as f64
            + rotation[2][1] * p[1] as f64
            + rotation[2][2] * p[2] as f64
            + t[2];
        let z = if z.abs() < 1e-8 { 1e-8 } else { z };
        r.push(q[0] as f64 - (fx * x / z + cx));
        r.push(q[1] as f64 - (fy * y / z + cy));
    }
    r
}

/// OpenSeeFace stores a landmark as `(row, col, conf)`. Image x is the column.
fn image_xy(lm: [f32; 3]) -> [f32; 2] {
    [lm[1], lm[0]]
}

fn reprojection_rms(residual: &[f64], points: usize) -> f64 {
    let sum: f64 = residual.iter().map(|v| v * v).sum();
    (sum / points.max(1) as f64).sqrt()
}

/// Iterative PnP (Gauss–Newton), OpenCV `SOLVEPNP_ITERATIVE` stand-in.
pub fn solve_pnp(
    obj: &[[f32; 3]],
    img: &[[f32; 2]],
    cam: &Camera,
    guess: Option<([f32; 3], [f32; 3])>,
) -> Option<([f32; 3], [f32; 3])> {
    if obj.len() < 4 || obj.len() != img.len() {
        return None;
    }
    // Zero rotation is the failed fit. 180° about Z is upright (+X on image
    // left). 180° about X mirrors left and right and must not win a tie.
    let mut seeds = Vec::with_capacity(4);
    if let Some(seed) = guess {
        seeds.push(seed);
    }
    let translation_seed = init_pose(obj, img, cam);
    seeds.push(translation_seed);
    seeds.push(([std::f32::consts::PI, 0.0, 0.0], translation_seed.1));
    seeds.push(([0.0, 0.0, std::f32::consts::PI], translation_seed.1));
    let mut best: Option<([f32; 3], [f32; 3], f32)> = None;
    for seed in seeds {
        let solved = refine_pose(obj, img, cam, seed);
        if solved.2.is_finite() && best.is_none_or(|(_, _, error)| solved.2 < error) {
            best = Some(solved);
        }
    }
    let (rvec, tvec, error) = best?;
    let tlen = (tvec[0] * tvec[0] + tvec[1] * tvec[1] + tvec[2] * tvec[2]).sqrt();
    if tvec[2].abs() < 0.1 || tlen > 1e6 || error > 250.0 {
        return None;
    }
    Some((rvec, tvec))
}

fn refine_pose(
    obj: &[[f32; 3]],
    img: &[[f32; 2]],
    cam: &Camera,
    seed: ([f32; 3], [f32; 3]),
) -> ([f32; 3], [f32; 3], f32) {
    let (r0, t0) = seed;
    let mut params = [
        r0[0] as f64,
        r0[1] as f64,
        r0[2] as f64,
        t0[0] as f64,
        t0[1] as f64,
        t0[2].abs().max(0.5) as f64,
    ];
    let n = obj.len() * 2;
    let eps = 1e-6;
    let mut lambda = 1e-3_f64;
    for _ in 0..40 {
        let r = residuals_f64(&params, obj, img, cam);
        let rms = reprojection_rms(&r, obj.len());
        let mut j = vec![vec![0.0f64; 6]; n];
        for k in 0..6 {
            let mut p2 = params;
            p2[k] += eps;
            let r2 = residuals_f64(&p2, obj, img, cam);
            for i in 0..n {
                j[i][k] = (r2[i] - r[i]) / eps;
            }
        }
        let mut jt_j = [[0.0f64; 6]; 6];
        let mut jt_r = [0.0f64; 6];
        for i in 0..n {
            for k in 0..6 {
                jt_r[k] += j[i][k] * r[i];
                for m in 0..6 {
                    jt_j[k][m] += j[i][k] * j[i][m];
                }
            }
        }
        for k in 0..6 {
            jt_j[k][k] *= 1.0 + lambda;
        }
        let Some(dx) = solve6(&jt_j, &jt_r) else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
            continue;
        };
        let step = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !step.is_finite() || step > 2.0 {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
            continue;
        }
        let mut trial = params;
        for k in 0..6 {
            trial[k] -= dx[k];
        }
        let trial_rms = reprojection_rms(&residuals_f64(&trial, obj, img, cam), obj.len());
        if trial_rms < rms {
            params = trial;
            lambda = (lambda * 0.3).max(1e-8);
            if trial_rms < 0.05 || (rms - trial_rms) < 1e-4 {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    let rvec = [params[0] as f32, params[1] as f32, params[2] as f32];
    let tvec = [params[3] as f32, params[4] as f32, params[5] as f32];
    let error = reprojection_rms(&residuals_f64(&params, obj, img, cam), obj.len()) as f32;
    (rvec, tvec, error)
}

fn init_pose(obj: &[[f32; 3]], img: &[[f32; 2]], cam: &Camera) -> ([f32; 3], [f32; 3]) {
    let n = obj.len() as f32;
    let mut oc = [0.0f32; 3];
    let mut ic = [0.0f32; 2];
    for (o, i) in obj.iter().zip(img) {
        oc[0] += o[0];
        oc[1] += o[1];
        oc[2] += o[2];
        ic[0] += i[0];
        ic[1] += i[1];
    }
    oc[0] /= n;
    oc[1] /= n;
    oc[2] /= n;
    ic[0] /= n;
    ic[1] /= n;
    let mut os = 0.0f32;
    let mut is_ = 0.0f32;
    for (o, i) in obj.iter().zip(img) {
        os += (o[0] - oc[0]).hypot(o[1] - oc[1]);
        is_ += (i[0] - ic[0]).hypot(i[1] - ic[1]);
    }
    let s = (is_ / os.max(1e-6)).max(1.0);
    let z = (cam.fx / s).abs().max(0.5);
    let tx = (ic[0] - cam.cx) * z / cam.fx;
    let ty = (ic[1] - cam.cy) * z / cam.fy;
    ([0.0, 0.0, 0.0], [tx, ty, z])
}

fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let mut m = nalgebra::SMatrix::<f64, 6, 6>::zeros();
    let mut v = nalgebra::SVector::<f64, 6>::zeros();
    for i in 0..6 {
        v[i] = b[i];
        for j in 0..6 {
            m[(i, j)] = a[i][j];
        }
    }
    m.lu()
        .solve(&v)
        .map(|x| [x[0], x[1], x[2], x[3], x[4], x[5]])
}

fn wrap_degrees(degrees: f32) -> f32 {
    (degrees + 180.0).rem_euclid(360.0) - 180.0
}

/// Packet degrees. Pitch is chin-up positive, yaw is nose-to-image-left
/// positive, and roll is a tilt toward the user's right. The upright solve
/// sits near roll ±180; `adjust_3d` still uses the raw solve.
fn published_euler(solved: [f32; 3]) -> [f32; 3] {
    [
        wrap_degrees(solved[0]),
        wrap_degrees(solved[1]),
        wrap_degrees(-wrap_degrees(solved[2] + 180.0)),
    ]
}

/// Same `Rz * Ry * Rx` matrix the output filter rebuilds from packet euler.
fn quaternion_from_packet_euler(euler: [f32; 3]) -> [f32; 4] {
    let (x, y, z) = (
        euler[0].to_radians(),
        euler[1].to_radians(),
        euler[2].to_radians(),
    );
    let (cx, sx) = (x.cos(), x.sin());
    let (cy, sy) = (y.cos(), y.sin());
    let (cz, sz) = (z.cos(), z.sin());
    let r = Matrix3::new(
        cy * cz,
        cz * sx * sy - cx * sz,
        sx * sz + cx * cz * sy,
        cy * sz,
        cx * cz + sx * sy * sz,
        cx * sy * sz - cz * sx,
        -sy,
        cy * sx,
        cx * cy,
    );
    matrix_to_quaternion(&r)
}

pub fn euler_from_rmat(r: &Matrix3<f32>) -> [f32; 3] {
    let sy = (r[(0, 0)] * r[(0, 0)] + r[(1, 0)] * r[(1, 0)]).sqrt();
    if sy > 1e-6 {
        [
            r[(2, 1)].atan2(r[(2, 2)]).to_degrees(),
            (-r[(2, 0)]).atan2(sy).to_degrees(),
            r[(1, 0)].atan2(r[(0, 0)]).to_degrees(),
        ]
    } else {
        [
            (-r[(1, 2)]).atan2(r[(1, 1)]).to_degrees(),
            (-r[(2, 0)]).atan2(sy).to_degrees(),
            0.0,
        ]
    }
}

pub struct DepthResult {
    pub success: bool,
    pub quaternion: [f32; 4],
    pub euler: [f32; 3],
    pub pnp_error: f32,
    pub pts_3d: [[f32; 3]; 70],
    pub lms: Vec<[f32; 3]>,
    pub rotation: [f32; 3],
    pub translation: [f32; 3],
}

pub fn estimate_depth(
    lms66: &[[f32; 3]],
    eye_state: &[[f32; 4]; 2],
    face_3d: &[[f32; 3]],
    contour_idx: &[usize],
    cam: &Camera,
    prev: Option<([f32; 3], [f32; 3])>,
) -> DepthResult {
    let mut lms = lms66.to_vec();
    lms.push([eye_state[0][1], eye_state[0][2], eye_state[0][3]]);
    lms.push([eye_state[1][1], eye_state[1][2], eye_state[1][3]]);

    let obj: Vec<[f32; 3]> = contour_idx
        .iter()
        .map(|&i| face_3d.get(i).copied().unwrap_or([0.0; 3]))
        .collect();
    // Landmarks are (row, col). The pinhole model takes image (x, y) = (col, row).
    let img: Vec<[f32; 2]> = contour_idx.iter().map(|&i| image_xy(lms[i])).collect();

    let fail = DepthResult {
        success: false,
        quaternion: [0.0; 4],
        euler: [0.0; 3],
        pnp_error: 99999.0,
        pts_3d: [[0.0; 3]; 70],
        lms: lms.clone(),
        rotation: [0.0; 3],
        translation: [0.0; 3],
    };

    let Some((rotation, translation)) = solve_pnp(&obj, &img, cam, prev) else {
        return fail;
    };

    let rmat = rodrigues(rotation);
    let Some(inv_r) = rmat.try_inverse() else {
        return fail;
    };
    let inv_cam = cam.inverse();
    let t = Vector3::new(translation[0], translation[1], translation[2]);

    let mut t_reference = Vec::with_capacity(face_3d.len());
    for p in face_3d {
        let mut x = rmat * Vector3::new(p[0], p[1], p[2]) + t;
        x = cam.matrix() * x;
        t_reference.push(x);
    }
    let mut pts_3d = [[0.0f32; 3]; 70];
    for i in 0..66.min(lms.len()) {
        let mut depth = t_reference[i].z;
        if depth == 0.0 {
            depth = 1e-6;
        }
        let xy = image_xy(lms[i]);
        let p = Vector3::new(xy[0] * depth, xy[1] * depth, depth);
        let world = inv_r * (inv_cam * p - t);
        pts_3d[i] = [world.x, world.y, world.z];
    }

    let mut pnp_error = 0.0f32;
    for i in 0..17 {
        let z = if t_reference[i].z.abs() < 1e-8 {
            1e-8
        } else {
            t_reference[i].z
        };
        let px = t_reference[i].x / z;
        let py = t_reference[i].y / z;
        let xy = image_xy(lms[i]);
        pnp_error += (xy[0] - px).powi(2) + (xy[1] - py).powi(2);
    }
    {
        let z = if t_reference[30].z.abs() < 1e-8 {
            1e-8
        } else {
            t_reference[30].z
        };
        let xy = image_xy(lms[30]);
        pnp_error += (xy[0] - t_reference[30].x / z).powi(2) + (xy[1] - t_reference[30].y / z).powi(2);
    }
    if pnp_error.is_nan() {
        pnp_error = 9_999_999.0;
    }

    // Pupils + eyeball centres (indices 66..70 in face_3d)
    for i in 0..4 {
        if i == 2 {
            let c = [
                (pts_3d[36][0] + pts_3d[39][0]) / 2.0,
                (pts_3d[36][1] + pts_3d[39][1]) / 2.0,
                (pts_3d[36][2] + pts_3d[39][2]) / 2.0,
            ];
            let d = dist3(pts_3d[36], pts_3d[39]);
            pts_3d[68] = [c[0], c[1], c[2] - 0.385 * d];
            continue;
        }
        if i == 3 {
            let c = [
                (pts_3d[42][0] + pts_3d[45][0]) / 2.0,
                (pts_3d[42][1] + pts_3d[45][1]) / 2.0,
                (pts_3d[42][2] + pts_3d[45][2]) / 2.0,
            ];
            let d = dist3(pts_3d[42], pts_3d[45]);
            pts_3d[69] = [c[0], c[1], c[2] - 0.385 * d];
            continue;
        }
        let (d1, d2, a, b) = if i == 0 {
            (
                dist2(lms[66], lms[36]),
                dist2(lms[66], lms[39]),
                pts_3d[36],
                pts_3d[39],
            )
        } else {
            (
                dist2(lms[67], lms[42]),
                dist2(lms[67], lms[45]),
                pts_3d[42],
                pts_3d[45],
            )
        };
        let d = d1 + d2;
        if d < 1e-8 {
            continue;
        }
        let pt = [
            (a[0] * d1 + b[0] * d2) / d,
            (a[1] * d1 + b[1] * d2) / d,
            (a[2] * d1 + b[2] * d2) / d,
        ];
        let mut reference = rmat * Vector3::new(pt[0], pt[1], pt[2]) + t;
        reference = cam.matrix() * reference;
        let depth = reference.z;
        let xy = image_xy(lms[66 + i]);
        let p = Vector3::new(xy[0] * depth, xy[1] * depth, depth);
        let world = inv_r * (inv_cam * p - t);
        pts_3d[66 + i] = [world.x, world.y, world.z];
    }
    for p in pts_3d.iter_mut() {
        if p.iter().any(|v| v.is_nan()) {
            *p = [0.0, 0.0, 0.0];
        }
    }

    pnp_error = (pnp_error / (2.0 * img.len() as f32)).sqrt();
    let euler = published_euler(euler_from_rmat(&rmat));
    DepthResult {
        success: true,
        quaternion: quaternion_from_packet_euler(euler),
        euler,
        pnp_error,
        pts_3d,
        lms,
        rotation,
        translation,
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub fn normalize_pts3d(pts: &mut [[f32; 3]; 70], face_3d: &[[f32; 3]]) {
    let base_v: [f32; 3] = [
        face_3d[27][1] - face_3d[28][1],
        face_3d[28][1] - face_3d[29][1],
        face_3d[29][1] - face_3d[30][1],
    ];
    let base_h = [
        (face_3d[0][0] - face_3d[16][0]).abs(),
        (face_3d[36][0] - face_3d[39][0]).abs(),
        (face_3d[42][0] - face_3d[45][0]).abs(),
    ];
    let nose = [pts[30][0], pts[30][1]];
    for p in pts.iter_mut() {
        p[0] -= nose[0];
        p[1] -= nose[1];
    }
    let a = crate::geom::angle([pts[30][0], pts[30][1]], [pts[27][0], pts[27][1]])
        - 90.0f32.to_radians();
    let (c, s) = (a.cos(), a.sin());
    for p in pts.iter_mut() {
        let x = p[0];
        let y = p[1];
        p[0] = x * c + y * s;
        p[1] = -x * s + y * c;
    }
    // Python: (pts_3d - pts_3d[30])[:,0:2].dot(R) + pts_3d[30]
    // After subtracting nose, pts[30] xy is 0. Rotation applied. Re-add? Python adds pts[30,0:2]
    // which after subtract is 0. Then later scales.
    let mean_v = ((pts[27][1] - pts[28][1]) / base_v[0]
        + (pts[28][1] - pts[29][1]) / base_v[1]
        + (pts[29][1] - pts[30][1]) / base_v[2])
        / 3.0;
    if mean_v.abs() > 1e-8 {
        for p in pts.iter_mut() {
            p[1] /= mean_v;
        }
    }
    let mean_h = ((pts[0][0] - pts[16][0]).abs() / base_h[0]
        + (pts[36][0] - pts[39][0]).abs() / base_h[1]
        + (pts[42][0] - pts[45][0]).abs() / base_h[2])
        / 3.0;
    if mean_h.abs() > 1e-8 {
        for p in pts.iter_mut() {
            p[0] /= mean_h;
        }
    }
}

const RIGHT_LOCK: [usize; 28] = [
    0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 56, 57, 58,
    59, 65,
];
const LEFT_LOCK: [usize; 28] = [
    9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 34, 35, 42, 43, 44, 45, 46, 47, 51, 52, 53,
    54, 61, 62, 63,
];
const LEFT_ELIGIBLE: &[usize] = &[
    8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 42, 43, 44, 45,
    46, 47, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64,
];
const RIGHT_ELIGIBLE: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 27, 28, 29, 31, 32, 33, 36, 37, 38, 39, 40, 41,
    48, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65,
];

/// Raw solve. Positive yaw is the user's right and locks that side.
fn model_update(solved: [f32; 3]) -> ModelUpdate {
    let pitch = solved[0];
    let yaw = solved[1];
    let roll = wrap_degrees(solved[2] + 180.0);
    if !(-15.0..=35.0).contains(&pitch) {
        return ModelUpdate::Skip;
    }
    if (-20.0..10.0).contains(&yaw) {
        return ModelUpdate::FrontalDepth;
    }
    if roll.abs() > 30.0 {
        return ModelUpdate::Skip;
    }
    ModelUpdate::Turned { lock_right: yaw > 10.0 }
}

enum ModelUpdate {
    Skip,
    FrontalDepth,
    Turned { lock_right: bool },
}

pub fn adjust_3d(
    face_3d: &mut [[f32; 3]],
    pts_3d: &mut [[f32; 3]; 70],
    lms: &[[f32; 3]],
    rotation: [f32; 3],
    translation: [f32; 3],
    cam: &Camera,
    conf: f32,
    pnp_error: f32,
    static_model: bool,
    model_type: i32,
    update_counts: &mut [[f32; 2]; 66],
    feature_level: i32,
    features: &mut crate::features::FeatureExtractor,
    current_features: &mut crate::features::FeatureVec,
    eye_blink: &mut [f32; 2],
) {
    if conf < 0.4 || pnp_error > 300.0 {
        normalize_pts3d(pts_3d, face_3d);
        apply_features(
            pts_3d,
            feature_level,
            features,
            current_features,
            eye_blink,
            mean_conf(lms, &EYE_IDX),
        );
        return;
    }
    if model_type != -1 && !static_model {
        let mut rng = rand::thread_rng();
        let mut eligible: Vec<usize> = (0..66).filter(|i| *i != 30).collect();
        let mut update_type: i32 = -1;
        let mut r = [[1.0f32; 3]; 66];
        for row in r.iter_mut() {
            for v in row.iter_mut() {
                *v = 1.0 + rng.gen::<f32>() * 0.02 - 0.01;
            }
        }
        r[30] = [1.0, 1.0, 1.0];
        let solved = euler_from_rmat(&rodrigues(rotation));
        let decision = model_update(solved);
        if let ModelUpdate::FrontalDepth = decision {
            for row in r.iter_mut() {
                row[2] = 1.0;
            }
            update_type = 0;
        } else if let ModelUpdate::Turned { lock_right } = decision {
            for row in r.iter_mut() {
                row[0] = 1.0;
                row[1] = 1.0;
            }
            update_type = 1;
            if lock_right {
                for &i in &RIGHT_LOCK {
                    r[i][2] = 1.0;
                }
                eligible = LEFT_ELIGIBLE.to_vec();
            } else {
                for &i in &LEFT_LOCK {
                    r[i][2] = 1.0;
                }
                eligible = RIGHT_ELIGIBLE.to_vec();
            }
        }
        if !matches!(decision, ModelUpdate::Skip) {
            let ut = if update_type < 0 {
                0
            } else {
                update_type as usize
            };
            let other = 1 - ut;
            eligible.retain(|&i| update_counts[i][ut] < update_counts[i][other] + 75.0);
            if !eligible.is_empty() {
                let mut updated: Vec<[f32; 3]> = face_3d[..66].to_vec();
                let o_proj = project_points(face_3d, rotation, translation, cam);
                let scaled: Vec<[f32; 3]> = updated
                    .iter()
                    .enumerate()
                    .map(|(i, p)| [p[0] * r[i][0], p[1] * r[i][1], p[2] * r[i][2]])
                    .collect();
                let c_proj = project_points(&scaled, rotation, translation, cam);
                let mut changed = false;
                for &i in &eligible {
                    let observed = image_xy(lms[i]);
                    let d_o = ((o_proj[i][0] - observed[0]).powi(2)
                        + (o_proj[i][1] - observed[1]).powi(2))
                    .sqrt();
                    let d_c = ((c_proj[i][0] - observed[0]).powi(2)
                        + (c_proj[i][1] - observed[1]).powi(2))
                    .sqrt();
                    if d_c < d_o {
                        update_counts[i][ut] += 1.0;
                        updated[i] = scaled[i];
                        changed = true;
                    }
                }
                if changed {
                    for i in 0..66 {
                        if update_counts[i][ut] > 7500.0 {
                            continue;
                        }
                        let mut w = lms[i][2];
                        if w > 0.7 {
                            w = 1.0;
                        }
                        w = 1.0 - w;
                        face_3d[i][0] = face_3d[i][0] * w + updated[i][0] * (1.0 - w);
                        face_3d[i][1] = face_3d[i][1] * w + updated[i][1] * (1.0 - w);
                        face_3d[i][2] = face_3d[i][2] * w + updated[i][2] * (1.0 - w);
                    }
                }
            }
        }
    }
    normalize_pts3d(pts_3d, face_3d);
    apply_features(
        pts_3d,
        feature_level,
        features,
        current_features,
        eye_blink,
        mean_conf(lms, &EYE_IDX),
    );
}

fn apply_features(
    pts_3d: &[[f32; 3]; 70],
    feature_level: i32,
    features: &mut crate::features::FeatureExtractor,
    current_features: &mut crate::features::FeatureVec,
    eye_blink: &mut [f32; 2],
    eye_conf: Option<f32>,
) {
    if feature_level >= 1 {
        *current_features = features.update_ex(pts_3d, feature_level == 2, eye_conf);
        eye_blink[0] = 1.0 - (-current_features[1]).clamp(0.0, 1.0);
        eye_blink[1] = 1.0 - (-current_features[0]).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_image(yaw_degrees: f32, pitch_degrees: f32) -> (Camera, Vec<[f32; 3]>, Vec<[f32; 2]>) {
        let cam = Camera::from_frame(640, 480);
        let rotation = model_rotation(yaw_degrees, pitch_degrees);
        let translation = Vector3::new(0.0, 0.0, 8.0);
        let obj: Vec<[f32; 3]> = CONTOUR_PTS.iter().map(|&i| FACE_3D[i]).collect();
        let img = obj
            .iter()
            .map(|p| {
                let x = rotation * Vector3::new(p[0], p[1], p[2]) + translation;
                [cam.fx * x.x / x.z + cam.cx, cam.fy * x.y / x.z + cam.cy]
            })
            .collect();
        (cam, obj, img)
    }

    fn model_rotation(yaw_degrees: f32, pitch_degrees: f32) -> Matrix3<f32> {
        rodrigues([std::f32::consts::PI, 0.0, 0.0])
            * rodrigues([pitch_degrees.to_radians(), yaw_degrees.to_radians(), 0.0])
    }

    fn assert_recovered(yaw: f32, pitch: f32, guess_zero: bool) {
        let (cam, obj, img) = face_image(yaw, pitch);
        let guess = guess_zero.then(|| init_pose(&obj, &img, &cam));
        let (rvec, tvec) = solve_pnp(&obj, &img, &cam, guess)
            .unwrap_or_else(|| panic!("facing face yaw {yaw} pitch {pitch} did not solve"));
        let euler = euler_from_rmat(&rodrigues(rvec));
        let truth = euler_from_rmat(&model_rotation(yaw, pitch));
        for (got, expected) in euler.iter().zip(truth) {
            let delta = (*got - expected + 180.0).rem_euclid(360.0) - 180.0;
            assert!(
                delta.abs() < 2.0,
                "recovered {euler:?} truth {truth:?} for yaw {yaw} pitch {pitch}"
            );
        }
        let proj = project_points(&obj, rvec, tvec, &cam);
        let mut error = 0.0f32;
        for (p, q) in proj.iter().zip(&img) {
            error += (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
        }
        error = (error / img.len() as f32).sqrt();
        assert!(
            error < 1.0,
            "reprojection {error} px for yaw {yaw} pitch {pitch}"
        );
    }

    #[test]
    fn facing_yaw_is_recovered_instead_of_the_zero_rotation() {
        assert_recovered(18.0, 0.0, false);
    }

    #[test]
    fn zero_guess_still_recovers_facing_yaw_and_pitch() {
        assert_recovered(18.0, -14.0, true);
    }

    #[test]
    fn larger_turn_still_reaches_the_facing_pose() {
        assert_recovered(35.0, 20.0, true);
    }

    fn project_model(rotation: Matrix3<f32>, point: [f32; 3]) -> [f32; 2] {
        let cam = Camera::from_frame(640, 480);
        let x = rotation * Vector3::new(point[0], point[1], point[2]) + Vector3::new(0.0, 0.0, 8.0);
        [cam.fx * x.x / x.z + cam.cx, cam.fy * x.y / x.z + cam.cy]
    }

    fn solve_upright(extra: Matrix3<f32>) -> (Matrix3<f32>, DepthResult) {
        // Subject's right (+X) is image-left and model up (+Y) is image-up.
        let upright = Matrix3::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
        let rotation = upright * extra;
        let cam = Camera::from_frame(640, 480);
        let translation = Vector3::new(0.0, 0.0, 8.0);
        let mut lms = vec![[0.0f32; 3]; 66];
        for (i, p) in FACE_3D.iter().take(66).enumerate() {
            let x = rotation * Vector3::new(p[0], p[1], p[2]) + translation;
            let px = cam.fx * x.x / x.z + cam.cx;
            let py = cam.fy * x.y / x.z + cam.cy;
            lms[i] = [py, px, 1.0];
        }
        let eyes = [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        let depth = estimate_depth(&lms, &eyes, &FACE_3D, &CONTOUR_PTS, &cam, None);
        (rotation, depth)
    }

    fn assert_usable(got: [f32; 3], expected: [f32; 3], label: &str) {
        for (axis, (value, want)) in got.iter().zip(expected).enumerate() {
            let delta = wrap_degrees(value - want);
            assert!(
                delta.abs() < 1.5,
                "{label} axis {axis}: usable {got:?} want {expected:?}"
            );
        }
    }

    #[test]
    fn upright_pose_publishes_pitch_yaw_roll() {
        let (rest_rotation, rest) = solve_upright(Matrix3::identity());
        assert!(
            rest.success && rest.pnp_error < 1.0,
            "rest fit {} err {}",
            rest.success,
            rest.pnp_error
        );
        let rest_right = project_model(rest_rotation, FACE_3D[0]);
        let rest_left = project_model(rest_rotation, FACE_3D[16]);
        let rest_chin = project_model(rest_rotation, FACE_3D[8]);
        let rest_brow = project_model(rest_rotation, FACE_3D[27]);
        assert!(
            rest_right[0] < rest_left[0] - 20.0,
            "subject right jaw is image-left of subject left: {rest_right:?} {rest_left:?}"
        );
        assert!(
            rest_chin[1] > rest_brow[1] + 20.0,
            "chin is below the brow: {rest_chin:?} {rest_brow:?}"
        );
        assert_usable(rest.euler, [0.0, 0.0, 0.0], "rest");

        let yaw_extra = rodrigues([0.0, 18.0_f32.to_radians(), 0.0]);
        let (yaw_rotation, yaw) = solve_upright(yaw_extra);
        let nose = [0.0f32, 0.0, 0.2];
        let rest_nose = project_model(rest_rotation, nose);
        let yaw_nose = project_model(yaw_rotation, nose);
        assert!(
            yaw_nose[0] < rest_nose[0] - 3.0,
            "positive model yaw moves the nose to image left: {rest_nose:?} -> {yaw_nose:?}"
        );
        assert!(yaw.success && yaw.pnp_error < 1.0, "yaw err {}", yaw.pnp_error);
        // Image left is the user's right, published as positive yaw.
        assert_usable(yaw.euler, [0.0, 18.0, 0.0], "yaw");

        let pitch_extra = rodrigues([14.0_f32.to_radians(), 0.0, 0.0]);
        let (pitch_rotation, pitch) = solve_upright(pitch_extra);
        let rest_chin = project_model(rest_rotation, FACE_3D[8]);
        let pitch_chin = project_model(pitch_rotation, FACE_3D[8]);
        assert!(
            pitch_chin[1] < rest_chin[1] - 3.0,
            "positive model pitch lifts the chin: {rest_chin:?} -> {pitch_chin:?}"
        );
        assert!(
            pitch.success && pitch.pnp_error < 1.0,
            "pitch err {}",
            pitch.pnp_error
        );
        assert_usable(pitch.euler, [14.0, 0.0, 0.0], "pitch");

        let roll_extra = rodrigues([0.0, 0.0, 11.0_f32.to_radians()]);
        let (roll_rotation, roll) = solve_upright(roll_extra);
        let rest_brow = project_model(rest_rotation, FACE_3D[27]);
        let roll_brow = project_model(roll_rotation, FACE_3D[27]);
        assert!(
            roll_brow[0] > rest_brow[0] + 3.0,
            "positive model roll moves the brow to image right: {rest_brow:?} -> {roll_brow:?}"
        );
        assert!(roll.success && roll.pnp_error < 1.0, "roll err {}", roll.pnp_error);
        // Brow moving to image right is a tilt toward the user's left.
        assert_usable(roll.euler, [0.0, 0.0, -11.0], "roll");
    }

    #[test]
    fn model_update_follows_the_raw_solve() {
        assert!(matches!(
            model_update([0.0, 0.0, -180.0]),
            ModelUpdate::FrontalDepth
        ));
        assert!(matches!(
            model_update([0.0, 18.0, -180.0]),
            ModelUpdate::Turned { lock_right: true }
        ));
        assert!(matches!(
            model_update([0.0, -25.0, 180.0]),
            ModelUpdate::Turned { lock_right: false }
        ));
        assert!(matches!(model_update([-40.0, 0.0, -180.0]), ModelUpdate::Skip));
        assert!(matches!(
            model_update([0.0, 18.0, 90.0]),
            ModelUpdate::Skip
        ));
    }

    #[test]
    fn estimate_depth_unprojects_a_turned_face_back_to_the_model() {
        let yaw = 18.0;
        let pitch = -14.0;
        let cam = Camera::from_frame(640, 480);
        let rotation = model_rotation(yaw, pitch);
        let translation = Vector3::new(0.0, 0.0, 8.0);
        let mut lms = vec![[0.0f32; 3]; 66];
        for (i, p) in FACE_3D.iter().take(66).enumerate() {
            let x = rotation * Vector3::new(p[0], p[1], p[2]) + translation;
            let px = cam.fx * x.x / x.z + cam.cx;
            let py = cam.fy * x.y / x.z + cam.cy;
            lms[i] = [py, px, 1.0];
        }
        let eyes = [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        let depth = estimate_depth(&lms, &eyes, &FACE_3D, &CONTOUR_PTS, &cam, None);
        assert!(depth.success, "fit failed, error {}", depth.pnp_error);
        let truth = euler_from_rmat(&rotation);
        let solved = euler_from_rmat(&rodrigues(depth.rotation));
        for (got, expected) in solved.iter().zip(truth) {
            let delta = (*got - expected + 180.0).rem_euclid(360.0) - 180.0;
            assert!(
                delta.abs() < 2.0,
                "geometric euler {solved:?} truth {truth:?}"
            );
        }
        for i in [0usize, 8, 30, 48, 54] {
            let delta = dist3(depth.pts_3d[i], FACE_3D[i]);
            assert!(
                delta < 0.02,
                "point {i} stayed in camera space: {:?} vs {:?} ({delta})",
                depth.pts_3d[i],
                FACE_3D[i]
            );
        }
    }
}
