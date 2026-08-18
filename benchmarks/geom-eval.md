# Geometric refine evaluation

Tried after PnP, before One Euro, on the UDP clone only (crop / PnP / expression EMA unchanged): face symmetry, rigid relative-distance clamps, 3D model projection blend, and a 3-iter PnP polish. Also stacked with the default One Euro filter.

Same raw `Tracker::predict` sequence as `--suite filter` (model 3, 4 threads, 50 frames/scene). Teacher = clean-frame tracker on noisy/dark/static/left-noise.

| Kind | Static center jitter | vs none | Noisy NME | Dark NME | Step extra lag | Cost |
|---|---|---|---|---|---|---|
| none | 39.5 px | — | 0.265 | 0.464 | 0 | 0 |
| one-euro (default) | 12.5 px | −68% | 0.230 | 0.449 | 0 | 1.3 µs |
| sym | 35.5 px | −10% | 0.268 | 0.517 | 0 | 0.2 µs |
| rel | 39.7 px | +0% | 0.270 | 0.458 | 0 | 0.4 µs |
| proj | 39.5 px | 0% | 0.265 | 0.466 | 0 | ~0.05 µs |
| proj-pnp | 39.5 px | 0% | 0.265 | 0.466 | 0 | 5.1 µs |
| all | 36.6 px | −8% | 0.266 | 0.523 | 0 | 0.6 µs |
| all+one-euro | 10.5 px | −73% | 0.227 | 0.515 | 0 | 1.7 µs |

Projection barely moves high-confidence points. `proj-pnp` adds ~0.6–1.2° euler jitter. Symmetry/relative pull clean wander landmarks off the measurement (NME 0.07 vs 0). Left-side noise injection: `all` left NME 0.186 vs none 0.036 (over-corrects). Asymmetric wink eye-gap keep 0.93.

Cost and step lag were fine. Stacked jitter beat One Euro on static (10.5 vs 12.5 px) and noisy NME was tied, but dark NME got worse (0.515 vs 0.449) and clean frames were distorted.

**Reject.** One Euro stays the output post-process. The refine module was not kept.
