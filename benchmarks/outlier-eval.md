# Outlier rejection evaluation

Tried after PnP, before One Euro, on the UDP clone only (crop / PnP / expression EMA unchanged): per-landmark velocity gates, confidence hold, constant-velocity trajectory residual, and `all` (those plus spatial majority vote so a whole-face step is not treated as an outlier). Also stacked with the default One Euro filter.

Same raw `Tracker::predict` sequence as `--suite filter` (model 3, 4 threads, 50 frames/scene). Teacher = clean-frame tracker on noisy/dark/static/spike.

| Kind | Static center jitter | vs none | Noisy NME | Dark NME | Spike NME | Step extra lag | Cost |
|---|---|---|---|---|---|---|---|
| none | 39.5 px | — | 0.265 | 0.464 | 0.007 | 0 | 0 |
| one-euro (default) | 12.5 px | −68% | 0.230 | 0.449 | 0.115 | 0 | 1.3 µs |
| vel | 21.0 px | −47% | 0.266 | 0.470 | 0.103 | +3 | 2.1 µs |
| conf | 21.3 px | −46% | 0.265 | 0.465 | 0.102 | 0 | 2.2 µs |
| traj | 9.2 px | −77% | 0.280 | 0.467 | 0.160 | +3 | 2.4 µs |
| all | 21.7 px | −45% | 0.268 | 0.464 | 0.116 | 0 | 2.4 µs |
| all+one-euro | 6.8 px | −83% | 0.245 | 0.458 | 0.170 | +1 | 3.6 µs |

Majority vote does what it was for: `vel`/`traj` alone add the same +3 step lag that sank Kalman; `all` does not. Cost is negligible.

Hold still lags points that should move. Clean wander NME is 0.114 vs 0 for `all`. Injected 1–2 frame spikes on five jaw/nose points barely move mean NME (0.007); `all` does not beat that, it just adds the same hold lag as wander. Stacked jitter beats One Euro on static (6.8 vs 12.5 px) and dark NME is tied, but noisy NME is worse (0.245 vs 0.230), far NME is worse (0.399 vs 0.231), spike NME is worse, and step lag is +1.

**Reject.** One Euro stays the output post-process. The outlier module was not kept.
