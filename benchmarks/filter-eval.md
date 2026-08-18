# Output filter evaluation

UDP pose and 2D landmarks were post-processed after PnP. Crop EMA and expression EMA were left in the tracking loop. Candidates: none, speed-adaptive EMA, One Euro, 1D constant-velocity Kalman. Features/blink were tried with an extra One Euro pass and rejected.

`osf-bench --suite filter --model 3 --threads 4 --frames 50` (synthetic static/wander/far/noisy/step). Same raw `Tracker::predict` sequence for every filter. Teacher = clean-frame tracker.

| Filter | Static center jitter | vs none | Noisy NME vs teacher | Step extra lag | Cost |
|---|---|---|---|---|---|
| none | 39.5 px | — | 0.265 | 0 | 0 |
| ema | 31.1 px | −21% | 0.250 | 0 | ~1 µs |
| **one-euro** | **13.7 px** | **−65%** | **0.230** | **0** | **1–4 µs** |
| kalman | 6.1 px | −84% | 0.204 | +3 frames | ~1 µs |

Blink + One Euro needed 8 frames to reach 90% closed (threshold was 3). Expression channels stay on the existing `alpha=0.2` EMA.

**Keep:** `one-euro` (default) and `none`. Kalman is smoother but lags a head-turn; EMA missed the 30% jitter gate. Do not filter features a second time. Live filter writes only the result clone (not crop/PnP). Unity `OpenSeeIKTarget.smooth` should drop to 0–0.1 if the tracker filter is on.
