# Output filter evaluation

UDP pose and 2D landmarks are post-processed after PnP. Crop EMA and expression EMA stay in the tracking loop. Features/blink were tried with an extra One Euro pass and rejected.

Default One Euro scales `mincutoff` by landmark confidence and PnP stability (`stab = 1/(1 + pnp_error/200)`, floor 0.25 when PnP fails). High quality matches speed-only One Euro.

`osf-bench --suite filter --model 3 --threads 4 --frames 50`. Same raw `Tracker::predict` sequence. Teacher = clean-frame tracker.

| Filter | Static center jitter | vs none | Noisy NME vs teacher | Step extra lag | Cost |
|---|---|---|---|---|---|
| none | 39.5 px | — | 0.265 | 0 | 0 |
| ema | 31.1 px | −21% | 0.250 | 0 | ~1 µs |
| speed-only one-euro | 13.7 px | −65% | 0.230 | 0 | 1.5 µs |
| kalman | 6.1 px | −84% | 0.204 | +3 frames | ~1 µs |
| **one-euro (default)** | **12.5 px** | **−68%** | **0.230** | **0** | **1.8 µs** |

Quality scale vs speed-only: −9% static jitter, NME unchanged, no extra step lag. Kalman lags a head-turn; EMA missed a 30% jitter gate.

**Keep:** `one-euro` (default) and `none`. Live filter writes only the result clone (not crop/PnP).
