# Teacher-agreement benchmark: this fork vs upstream OpenSeeFace

Date: 2026-09-28 (initial run + post-crop-fix rerun). Harness and full data:
`../TrackingTest/benchmarks/teacher_agreement/` (`report.md` / `report.json`,
sampling in `sample.json`, landmark map in `nv_map.json`).

## Question

Is the fork's tracking algorithm more accurate than the original upstream
OpenSeeFace, with the NVIDIA Maxine AR SDK (126-landmark model) as the teacher?

## Setup

- **Teacher**: NVIDIA Maxine AR SDK v1.2, FaceExpressions feature, 126 landmarks,
  already recorded per-frame (`nv.bin`) for the TrackingTest session corpus
  (frame-index aligned by the shared ffmpeg bgr24 pipe).
- **Original**: upstream Python source at the fork point `85aa70f` (restored via
  `git worktree ../OpenSeeFace-orig`), models byte-identical to the VTube Studio
  v1.20.2 bundle (verified per-file; only `priorbox_640x640.json` differs).
  Ran as a labeling teacher (`TrackingTest/teachers/osf_orig_label`) with the
  same settings as the fork teacher: model 3, scan_every 1, try_hard 1,
  detection threshold 0.6, static 3D model, discard_after 10.
  Cross-checked against the distributed `facetracker.exe` via its `--raw-rgb`
  stdin mode + `--log-data` CSV on 3 clips: found-frame sequences identical
  (164/164, 301/301, 416/416) and landmark differences at float round-trip
  noise (median ≤ 2.1e-5 px) — the benchmarked original is the shipped binary.
- **Fork**: via `teachers/osf_label` (Rust, `osf_ort`), two runs:
  `--filter none` (same condition as the original, which has no output filter)
  and `--filter one_euro` (shipped default).
- **Data**: 30 clips stratified over 10 sessions (webcam `auto_*`, `pose_*`,
  LPM animation `lpm_*`), 11,990 NV-found frames.
- **Landmark correspondence**: NV126 -> OSF66 derived empirically (ICP + Hungarian
  on 186 frames, `tools/derive_nv_map.py`); 33 stable pairs survive, covering
  brows/eyes/nose/mouth/chin. OSF's jaw contour is quasi-3D by design (upstream
  README) and diverges from NV's outline-following contour under yaw, so contour
  points are excluded from NME. Mapping visually verified on overlay renders.
- **Metrics** (per NV-found frame):
  - coverage = P(tracker found | NV found)
  - NME = mean matched-point error / NV inter-ocular distance
  - score = 100 · mean( found ? exp(-NME/0.10) : 0 ) — a miss scores 0, so
    coverage and accuracy combine into one number.

## Results (30 clips, 11,990 NV-found frames)

| tracker | coverage | NME median | NME mean | NME p95 | **score** |
|---|---|---|---|---|---|
| original (85aa70f) | 99.5% | 0.0614 | 2.28 | 1.81 | **38.6** |
| fork pre-fix, filter=none | 75.3% | 0.0521 | 1.11 | 1.25 | 38.9 |
| **fork fixed, filter=none** | **99.5%** | 0.0615 | 2.28 | 1.81 | **38.6** |
| fork fixed, one-euro (shipped) | 99.5% | 0.0834 | 1.29 | 1.88 | 32.4 |

**Verdict: after the crop fix the fork matches the original overall (38.6 vs
38.6), winning 7 of 10 strata by small margins (pose 57.4 vs 57.3, lpm_skip
53.8 vs 53.3, lpm_speech 39.4 vs 39.3, auto webcam strata +0.1–0.7).**

Two things the aggregate hides:

1. The pre-fix "+0.3" lead was an artifact: it scored only the 75% of frames
   the fork managed to keep, comparing against the original scored on all
   frames. On matched easy strata the pre-fix fork did show a genuinely lower
   NME (e.g. 0.0504 vs 0.0577), which came from its then-different crop boxes
   (occasionally pupil-inflated hulls) — the same mechanism that exploded on
   strong yaw. That fragile advantage is deliberately not chased.
2. The pre-fix coverage hole was catastrophic on motion strata (pose 46.5,
   lpm_head_pose 21.0, lpm_skip 36.4 — all double-digit score points below the
   original). Post-fix all strata recover to original level or above.

The one-euro output filter lowers teacher agreement (32.4): it removes jitter
(see filter-eval.md) but adds lag on moving faces, which shows up as instant
error against a per-frame teacher. Coverage is unaffected, as expected. Tuning
mincutoff/beta against teacher agreement is future work.

## The strong-yaw tracking-loss bug (found by this bench, fixed)

On pose_20260916/20260916_162851 (repeated ~80° yaw) the fork found 14/164
frames where the original found 164/164. Root cause chain, confirmed by
instrumented runs and a unit regression test with the real frame's landmarks:

1. `stable_landmark_bbox` computed its size-reference hull with
   `landmark_bbox(fi.lms)` over **all 68 points — including the two gaze-model
   pupils**. On hard poses the gaze output can place a pupil near a frame
   corner (observed box 836x614 with the corner at (1211,751)); the pupil is
   not a face landmark and the original never includes it in its bbox
   (`lms[0:66].min/max` upstream).
2. `place_box`'s grow-only max rule latched the inflated size, the oversized
   crop dropped landmark confidence to ~0.35–0.48 < the 0.6 accept threshold,
   and the (upstream-inherited) stale-crop path suppressed re-detection for
   `discard_after` frames — producing the observed 1-found-per-~12-frames
   cycle (986x656 by the next frame).
3. The eyes+nose Umeyama fit itself was not the trigger (its scale was a sane
   ~152 on the failing frame), but under foreshortening it can also collapse —
   that is now guarded separately.

Fix in `runtime-ort/src/crop.rs` (all 79 unit tests pass, incl. the real-data
regression test `yaw_degenerate_fit_stays_bounded`):

- the hull is taken over the 66 model landmarks only, confidence-gated
  (`aabb_pts`), so wild pupils and dead refs cannot move it;
- `place_box` size growth is capped at `GROW_CAP` = 1.5x the current hull —
  one outlier frame cannot latch a giant crop;
- a fit whose template box covers less than `COVERAGE_FRAC` = 0.55x the hull
  extent, or whose center drifts more than half the hull span, is treated as
  degenerate and the hull box is used (measured tmpl/hull is 0.72–0.94 on
  healthy frames, 0.42 on the failing frame);
- the no-fit fallback prefers the full-landmark hull (upstream-equivalent,
  covers mouth and chin) over the interior subset, then holds the last box.

Verification: cold-start at frames 0/14/60/120 of the failing clip 1/10 →
10/10 found; full clip 14/164 → 164/164; the 30-clip rerun above.

## Caveats

- Teacher-relative, not ground-truth: on lpm_gaze the original tracks 99.7% of
  frames but sits ~6 IOD from NV (NME median 5.9) — both trackers diverge from
  the teacher there; on stylized LPM faces NV itself has no ground truth.
- Webcam clips (`auto_*`, `pose_*`) are a single person / single room; LPM
  clips are 3D-rendered avatars.
- With the crop rules now equivalent, fork and original landmarks agree to
  ~0.03 px median on webcam clips — the models and decode are identical, so
  the remaining measurable differences come from the output filter and the
  fork's extra features (gaze robustness, expressions), which this NME metric
  does not score.
