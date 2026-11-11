# Pinned official ASCENT execution

Repository: https://github.com/NextG-Wireless-Lab-Mason/Context-aware-Spectrum-Coexistence-Analyzer

Commit: `356c09f4caabfed772c66e2498a1aa8c2223a8c1`.

The downloaded numerical sources are byte-for-byte copies. Their SHA-256
digests are in `upstream_source_manifest.json` and enforced by `official_ascent.py`.
No certificate/private-key files, live weather calls, public HTTP listener,
database service, or unexplained serialized Python objects are required.

## License boundary

The repository has an MIT license in `DSA/LICENSE`, credited to Cognitive Radio
Open Source Software (2021). No root-level license explicitly covering
`Simulator.py` was found. Do not silently relabel all upstream files as local
code or assume that the subdirectory license grants rights to every file.
The final project can distribute the local adapter and execution evidence,
with this fixed-commit acquisition mechanism, rather than repackaging the entire
upstream repository.

## Reproduce

Install the pinned dependencies and run from the project directory:

```powershell
python -m pip install -r requirements.txt
python official_ascent.py
python official_ascent.py --full-case
```

The historical matplotlib/shapely versions in the upstream requirements do not
support this Python 3.13 environment. Only the geometry package and polygon
library are needed; the reviewed adapter injects ordinary scientific libraries
and loads function/class definitions via AST. It removes Flask registration
decorators, imports with weather/network side effects, and application startup.
Physical function bodies, original angle calculations, channel tests, gains,
power units, and propagation losses are retained in `upstream_exact`.

## Input and execution boundary

The static request is upstream `DSA/response.json`: sunny Blacksburg, 33 base
stations, FSS at (37.2025, -80.43444), 5 km inclusion zone, 1–1000 m UE radius,
and stored exclusion radius 1099.5412479883555 m. The request originally asks for
three repetitions. The separately preserved `upstream_exact.json` is the initial
single-snapshot evidence. `--full-case` runs all three repetitions into a new
`full_case_3snapshots.json`, keeping the initial evidence unchanged. It seeds
the simulator once, then shares the original LOS/angle caches and continuous
random stream across all repetitions.
The numeric angle cache is the upstream `t0p0.pkl`, accepted only after the
fixed SHA-256 check and a restricted loader permitting NumPy scalar/dtype
constructors. Every entry must be a finite numeric two-tuple key/value pair;
arbitrary class resolution is blocked.

The full-case preflight finds all 2970 optimization queries cached for snapshots
one and two. Snapshot three has 2061 unique missing angle pairs. These are
computed by the unmodified upstream `max_gain_5g_parameters`, including its
original `beam_pattern_5g` and SciPy brute/fmin algorithm, in four independent
processes. No closed-form replacement, approximation, or interference-target
fitting is used. The resulting numeric cache and its hash are preserved. The
actual unmodified simulator checks every queried angle exists and stops on an
unexpected miss rather than starting an unbounded optimization loop.

The original `run_simulator` reduction averages the linear per-BS INR ratios
over both stations and repetitions before taking dB. It is not the aggregate
INR sum and not an average of dB. The full-case report explicitly preserves
that original reduction, plus a separately named physically aggregated sum.
The first repeated snapshot is compared with the initial preserved evidence.

Original `Building` constructs geometry from the 8644-feature GeoJSON. The
adapter uses a fixed seed to construct any unspecified heights and then starts
the simulator with seed 10, matching the cached-building branch of the upstream
runner. It does not load `buildings.pkl` or arbitrary LOS pickle caches.

## Important upstream behaviors preserved

- The checked-in simulator actively computes UMi only; UMa/RMa interference
  branches are commented out even though their output slots remain.
- Per-beam Tx power is -6.77 in the W/dBW power convention, followed by 4 dB
  body loss. It is not a direct 33 dBm-per-gNB ICC configuration.
- gNB gain combines an 8 dBi element pattern and 16-by-16 array factor.
- FSS reception uses the WBES piecewise pattern: 20 dBi near boresight and
  -2 dBi at 48–180 degrees; this is not the ICC Bessel T1/T2 model.
- Noise is kTB with T=200 K and B=240 MHz, about -91.7868 dBm.
- Thirty interfering beams are sampled from the entire network UE pool for
  each BS, rather than restricted to that BS's associated UE pool.
- Input station activity/exclusion information is not applied to the upstream
  physical `simulate()` sum. Its per-BS results are evaluated separately by
  the DSA controller.
- `implement_simulator_feedback()` overwrites the supplied feedback with a
  literal 33-value vector. The exact controller keeps that behavior; the
  separately named `checked_controller` removes only that assignment and
  passes the actually simulated per-BS INR. Both use the original per-MBS
  threshold gate, not a guarantee on aggregate interference.
- UMi pathloss is overwritten on every tested wall, so a later clear wall can
  reset a previous NLOS pathloss to LOS while the `line_of_sight` flag remains
  false. The channel-overlap test also mistakenly constructs the FSS range
  from BS channel boundaries. These are evidence of implementation defects,
  not recommended physics to reproduce silently.

`output/official_ascent/upstream_exact.json` records source hashes, call counts,
input scope and exact results; `checked_controller.json` is a separate
controller-only comparison. They must not be mixed with corrected ICC model
results or described as reproducing this ICC paper's Frankfurt experiment.

Optional additional dependencies are listed in root `requirements-official.txt`.
The usual ICC simulation does not require Geometry3D. Saved official evidence
can be checked without rerunning geometry using `python official_ascent.py
--check-saved`.
