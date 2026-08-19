//! PnP, 3D face model, and pose — matching `Tracker.estimate_depth` / `FaceInfo.adjust_3d`.

use nalgebra::{Matrix3, Vector3};
use rand::Rng;

use crate::decode::{mean_conf, EYE_IDX};
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

fn residuals(params: &[f64; 6], obj: &[[f32; 3]], img: &[[f32; 2]], cam: &Camera) -> Vec<f64> {
    let rvec = [params[0] as f32, params[1] as f32, params[2] as f32];
    let tvec = [params[3] as f32, params[4] as f32, params[5] as f32];
    let proj = project_points(obj, rvec, tvec, cam);
    let mut r = Vec::with_capacity(obj.len() * 2);
    for (p, q) in proj.iter().zip(img) {
        r.push(q[0] as f64 - p[0] as f64);
        r.push(q[1] as f64 - p[1] as f64);
    }
    r
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
    let (r0, t0) = guess.unwrap_or_else(|| init_pose(obj, img, cam));
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
    for _ in 0..25 {
        let r = residuals(&params, obj, img, cam);
        let mut j = vec![vec![0.0f64; 6]; n];
        for k in 0..6 {
            let mut p2 = params;
            p2[k] += eps;
            let r2 = residuals(&p2, obj, img, cam);
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
            jt_j[k][k] += 1e-4;
        }
        let Some(dx) = solve6(&jt_j, &jt_r) else {
            break;
        };
        let mut nrm = 0.0;
        for k in 0..6 {
            params[k] += dx[k];
            nrm += dx[k] * dx[k];
        }
        if nrm.sqrt() < 1e-8 {
            break;
        }
    }
    let tlen = (params[3] * params[3] + params[4] * params[4] + params[5] * params[5]).sqrt();
    let rvec = [params[0] as f32, params[1] as f32, params[2] as f32];
    let tvec = [params[3] as f32, params[4] as f32, params[5] as f32];
    let proj = project_points(obj, rvec, tvec, cam);
    let mut e = 0.0f32;
    for (p, q) in proj.iter().zip(img) {
        e += (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
    }
    e = (e / img.len().max(1) as f32).sqrt();
    if params[5].abs() < 0.1 || tlen > 1e6 || e > 250.0 {
        let (r1, t1) = init_pose(obj, img, cam);
        return Some((r1, t1));
    }
    Some((rvec, tvec))
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
    let img: Vec<[f32; 2]> = contour_idx
        .iter()
        .map(|&i| [lms[i][0], lms[i][1]])
        .collect();

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
        let p = Vector3::new(lms[i][0] * depth, lms[i][1] * depth, depth);
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
        pnp_error += (lms[i][0] - px).powi(2) + (lms[i][1] - py).powi(2);
    }
    {
        let z = if t_reference[30].z.abs() < 1e-8 {
            1e-8
        } else {
            t_reference[30].z
        };
        pnp_error += (lms[30][0] - t_reference[30].x / z).powi(2)
            + (lms[30][1] - t_reference[30].y / z).powi(2);
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
        let p = Vector3::new(lms[66 + i][0] * depth, lms[66 + i][1] * depth, depth);
        let world = inv_r * (inv_cam * p - t);
        pts_3d[66 + i] = [world.x, world.y, world.z];
    }
    for p in pts_3d.iter_mut() {
        if p.iter().any(|v| v.is_nan()) {
            *p = [0.0, 0.0, 0.0];
        }
    }

    pnp_error = (pnp_error / (2.0 * img.len() as f32)).sqrt();
    DepthResult {
        success: true,
        quaternion: matrix_to_quaternion(&rmat),
        euler: euler_from_rmat(&rmat),
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

pub fn adjust_3d(
    face_3d: &mut [[f32; 3]],
    pts_3d: &mut [[f32; 3]; 70],
    lms: &[[f32; 3]],
    euler: [f32; 3],
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
    current_features: &mut [f32; 14],
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
        let mut skip = false;
        if euler[0] > -165.0 && euler[0] < 145.0 {
            skip = true;
        } else if euler[1] > -10.0 && euler[1] < 20.0 {
            for row in r.iter_mut() {
                row[2] = 1.0;
            }
            update_type = 0;
        } else {
            for row in r.iter_mut() {
                row[0] = 1.0;
                row[1] = 1.0;
            }
            if euler[2] > 120.0 || euler[2] < 60.0 {
                skip = true;
            } else if euler[1] < -10.0 {
                update_type = 1;
                for &i in &RIGHT_LOCK {
                    r[i][2] = 1.0;
                }
                eligible = LEFT_ELIGIBLE.to_vec();
            } else {
                update_type = 1;
                for &i in &LEFT_LOCK {
                    r[i][2] = 1.0;
                }
                eligible = RIGHT_ELIGIBLE.to_vec();
            }
        }
        if !skip {
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
                    let d_o = ((o_proj[i][0] - lms[i][0]).powi(2)
                        + (o_proj[i][1] - lms[i][1]).powi(2))
                    .sqrt();
                    let d_c = ((c_proj[i][0] - lms[i][0]).powi(2)
                        + (c_proj[i][1] - lms[i][1]).powi(2))
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
    current_features: &mut [f32; 14],
    eye_blink: &mut [f32; 2],
    eye_conf: Option<f32>,
) {
    if feature_level >= 1 {
        *current_features = features.update_ex(pts_3d, feature_level == 2, eye_conf);
        eye_blink[0] = 1.0 - (-current_features[1]).clamp(0.0, 1.0);
        eye_blink[1] = 1.0 - (-current_features[0]).clamp(0.0, 1.0);
    }
}
