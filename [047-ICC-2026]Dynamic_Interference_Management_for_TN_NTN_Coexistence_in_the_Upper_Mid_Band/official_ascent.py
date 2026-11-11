"""Run pinned upstream ASCENT numerical code offline, with provenance.

The upstream Simulator.py is kept byte-for-byte. Only its function/class AST
nodes are loaded: HTTP decorators, weather imports, and server startup are not
executed. This module does not present upstream behavior as corrected physics.
"""
from __future__ import annotations

import argparse
import ast
import base64
import cmath
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import pickle
import random
import sys
import time
import types
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

COMMIT = "356c09f4caabfed772c66e2498a1aa8c2223a8c1"
REPOSITORY = "https://github.com/NextG-Wireless-Lab-Mason/Context-aware-Spectrum-Coexistence-Analyzer"
SOURCE_ROOT = Path(__file__).resolve().parent / "external_sources" / "ASCENT"
EXPECTED_HASHES = {
    "Simulator.py": "57215df16557e3b679b15da726a86b216f1c25347b9aa0d526d192f5b1de16bf",
    "DSA/controllers/ASCENTController.py": "de99bb80d06f1a67dc059d1aa85dfe52e13e3bb7e5dd152804fe4cb96f2f92a1",
    "DSA/db/sim_settings.json": "fead917341ab6cee51cbef5ecec8990bf7d7e1749615e9a00b277f62b14da9f1",
    "DSA/response.json": "1d5373f56358689704ef657047a8a6733692c46b4d6cb6ffd881eecafe983f26",
    "data/export (1).geojson": "7d6425bbbb2534eb7996afc95ab3d9f16c8c2f53ed257c6ca4b740bd7d203658",
    "t0p0.pkl": "8f2da80b46d53989254e082a0afc8d9de1dadd31c3a806f00ce843a4dfff8a4f",
    "DSA/LICENSE": "20789cc746a442de668cb557a426d4d6d926b84fc0264560be78cfc6b9f40cfc",
    "DSA/settings/settings.py": "9197f71a99059e248d4ccf676d6712ec97f64a364a393e6b12fd516f3abad5bc",
}


def ensure_sources(root=SOURCE_ROOT):
    """Fetch only reviewed numerical sources/data; never keys or services."""
    root = Path(root)
    for name, expected in EXPECTED_HASHES.items():
        target = root / name
        if not target.is_file():
            url = f"https://raw.githubusercontent.com/{REPOSITORY.split('github.com/')[1]}/{COMMIT}/{urllib.parse.quote(name)}"
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"Downloaded upstream hash mismatch: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Upstream source changed: {name}")
    return root


def load_upstream(require_geometry=False, root=SOURCE_ROOT):
    """Return an isolated namespace of unchanged upstream definitions.

    This is also the entry point for like-for-like budget diagnostics. Without
    Geometry3D, antenna functions and pathloss with an empty building list work.
    """
    root = ensure_sources(root)
    dependency_path = str(root / "runtime_deps")
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    import pandas as pd
    namespace = {"__name__": "ascent_upstream_numerical", "__file__": str(root / "Simulator.py"),
        "cmath": cmath, "json": json, "math": math, "os": __import__("os"), "pickle": pickle,
        "random": random, "Path": Path, "Tuple": tuple, "matplotlib": matplotlib,
        "plt": plt, "base64": base64, "io": io, "np": np, "pd": pd,
        "optimize": optimize, "tqdm": lambda iterable: iterable}
    try:
        import Geometry3D
        from shapely import geometry
        namespace.update({name: getattr(Geometry3D, name) for name in
                          ("Point", "Segment", "ConvexPolygon", "intersection")})
        namespace["geometry"] = geometry
    except ImportError:
        if require_geometry:
            raise RuntimeError("Install Geometry3D==0.2.4 and shapely==2.1.2; see official ASCENT run notes")
    tree = ast.parse((root / "Simulator.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.decorator_list = []  # registration only; function bodies unchanged
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root / "Simulator.py"), "exec"), namespace)
    return namespace


class _NumericCacheUnpickler(pickle.Unpickler):
    """Allow only the two NumPy numeric constructors found in the pinned cache."""
    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return np._core.multiarray.scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise pickle.UnpicklingError(f"Forbidden cached object: {module}.{name}")


def load_numeric_cache(root=SOURCE_ROOT):
    raw = (Path(root) / "t0p0.pkl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_HASHES["t0p0.pkl"]:
        raise RuntimeError("Numeric cache hash differs")
    data = _NumericCacheUnpickler(io.BytesIO(raw)).load()
    if not isinstance(data, dict):
        raise ValueError("Expected angle dictionary")
    result = {}
    for key, value in data.items():
        if not (isinstance(key, tuple) and len(key) == 2 and isinstance(value, tuple) and len(value) == 2):
            raise ValueError("Invalid angle cache entry")
        numbers = np.asarray((*key, *value), dtype=float)
        if not np.isfinite(numbers).all():
            raise ValueError("Nonfinite angle cache entry")
        result[tuple(map(float, key))] = tuple(map(float, value))
    return result


def build_context(namespace, case, seed=10, root=SOURCE_ROOT):
    """Recreate cached-buildings branch of upstream run_simulator setup."""
    rng_before = random.getstate()
    random.seed(seed)
    features = json.loads((Path(root) / "data/export (1).geojson").read_text())["features"]
    buildings, skipped = [], 0
    for feature in features:
        coordinates = feature["geometry"]["coordinates"]
        for coords in coordinates:
            try:
                buildings.append(namespace["Building"](coords, feature["properties"].get("height"),
                                                       case["lat_FSS"], case["lon_FSS"]))
            except Exception:
                skipped += 1  # same behavior as upstream construction loop
    random.setstate(rng_before)
    ctx = namespace["Context"]()
    ctx.rain, ctx.rain_rate = case["rain"], case["rain_rate"]
    ctx.lat_FSS, ctx.lon_FSS = case["lat_FSS"], case["lon_FSS"]
    ctx.R = 6.371e6
    ctx.x_FSS = ctx.R * math.cos(math.radians(ctx.lat_FSS)) * math.cos(math.radians(ctx.lon_FSS))
    ctx.y_FSS = ctx.R * math.cos(math.radians(ctx.lat_FSS)) * math.sin(math.radians(ctx.lon_FSS))
    ctx.z_FSS = ctx.R * math.sin(math.radians(ctx.lat_FSS))
    ctx.x, ctx.y, ctx.z = 0, 0, 4.5
    ctx.radius = case["radius"]
    ctx.base_station_count = case["base_station_count"]
    ctx.bs_ue_min_radius, ctx.bs_ue_max_radius = case["bs_ue_min_radius"], case["bs_ue_max_radius"]
    ctx.Noise_W = 1.38064852e-23 * 200 * 240e6
    ctx.data_within_zone = namespace["pd"].DataFrame(case["base_stations"])
    ctx.saved_tp, ctx.saved_los = load_numeric_cache(root), {}
    ctx.FSS_phi = {"UMi": 15, "UMa": 48, "RMa": 5}
    ctx.buildings = buildings
    return ctx, {"geojson_features": len(features), "constructed_buildings": len(buildings),
                 "skipped_coordinate_groups": skipped, "height_random_seed": seed,
                 "building_mode": "rebuild geometry once; then use upstream cached-building RNG branch"}


def _controller_result(source, case, values, checked=False):
    """Execute upstream feedback body; checked removes its literal overwrite."""
    import pandas as pd
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ASCENTController")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "implement_simulator_feedback")
    if checked:
        if not (isinstance(fn.body[0], ast.Assign) and isinstance(fn.body[0].targets[0], ast.Name)
                and fn.body[0].targets[0].id == "feedback"):
            raise RuntimeError("Unexpected controller overwrite layout")
        fn.body = fn.body[1:]
    ns = {"settings": types.SimpleNamespace(INR_THRESHOLD={"default": -8.5, "rain": -12})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), ns)
    controller = types.SimpleNamespace(BS_COUNT=case["base_station_count"], rain=case["rain"],
        BASE_STATIONS=pd.DataFrame(case["base_stations"]).copy(),
        EXCLUSION_ZONE_RADIUS=case["exclusion_zone_radius"], configure_simulator_settings=lambda: None)
    result = ns["implement_simulator_feedback"](controller, {"Interference_values_UMi_each_Bs": values})
    mask = controller.BASE_STATIONS["status"].to_numpy(dtype=bool)
    post_ratio = (np.sum(np.power(10.0, np.asarray(values)[mask] / 10.0))
                  if len(values) == len(mask) else None)
    return {"message": result, "active_mask": controller.BASE_STATIONS["status"].tolist(),
            "active_count": int(np.count_nonzero(controller.BASE_STATIONS["status"])),
            "exclusion_zone_radius_m": controller.EXCLUSION_ZONE_RADIUS,
            "feedback_overwrite_removed": checked, "threshold_db": -12 if case["rain"] else -8.5,
            "input_feedback_length": len(values),
            "measured_post_control_aggregate_inr_db": (float(10 * np.log10(max(post_ratio, 1e-300)))
                                                       if post_ratio is not None else None),
            "gate": "per-MBS INR threshold, not aggregate INR threshold"}


def run_case(output=None, seed=10):
    root = ensure_sources()
    out = Path(output or Path(__file__).resolve().parent / "output/official_ascent")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logging.getLogger("Geometry3D").setLevel(logging.ERROR)
    namespace = load_upstream(require_geometry=True)
    case = json.loads((root / "DSA/response.json").read_text())
    (out / "input_upstream_case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    print("Preparing original building geometry", flush=True)
    ctx, geometry_info = build_context(namespace, case, seed)
    print(json.dumps(geometry_info), flush=True)
    calls, cache_hits, cache_misses = Counter(), 0, 0
    monitored = ("simulate", "path_loss_UMi", "Interface_UMi_1", "gain_5g", "gain_fss_wbes_b", "beam_pattern_5g")
    for name in monitored:
        original = namespace[name]
        def counted(*args, _function=original, _name=name, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)
        namespace[name] = counted
    original_scan = namespace["max_gain_5g_parameters"]
    def track_scan(theta, phi, context, coarse=True, rounding_precision=0):
        nonlocal cache_hits, cache_misses
        key = (round(theta, rounding_precision), round(phi, rounding_precision)) if coarse else (theta, phi)
        if key in context.saved_tp:
            cache_hits += 1
        else:
            cache_misses += 1
            if cache_misses % 10 == 1:
                print(f"Original beam optimization cache miss {cache_misses}; hits {cache_hits}", flush=True)
        return original_scan(theta, phi, context, coarse, rounding_precision)
    namespace["max_gain_5g_parameters"] = track_scan
    random.seed(seed)
    np.random.seed(seed)
    log = io.StringIO()
    print("Executing unchanged upstream simulate: 33 base stations, one snapshot", flush=True)
    with contextlib.redirect_stdout(log):
        result = namespace["simulate"](output=False, ctx=ctx)
    (out / "upstream_console.log").write_text(log.getvalue(), encoding="utf-8")
    distances, ratios, los = np.asarray(result[4]), np.asarray(result[5]), np.asarray(result[6])
    inr = 10 * np.log10(np.maximum(ratios, 1e-300))
    exact = {"repository": REPOSITORY, "commit": COMMIT, "random_seed": seed,
        "case": "upstream static sunny Blacksburg case, one simulator snapshot",
        "simulation_count_in_original_request": case["simulation_count"], "executed_snapshots": 1,
        "input_active_count": sum(bs["status"] != 0 for bs in case["base_stations"]),
        "simulator_applies_input_activity_mask": False,
        "geometry": geometry_info, "noise_w": ctx.Noise_W, "noise_dbm": 10 * math.log10(ctx.Noise_W) + 30,
        "per_bs_inr_db": inr.tolist(), "per_bs_distance_m": distances.tolist(), "per_bs_los": los.tolist(),
        "aggregate_inr_db": float(10 * np.log10(np.sum(ratios))),
        "calls": dict(calls), "angle_cache_hits": cache_hits, "angle_cache_misses": cache_misses,
        "controller": _controller_result(root / "DSA/controllers/ASCENTController.py", case, inr.tolist()),
        "source_sha256": EXPECTED_HASHES, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "unchanged_upstream_model": True,
        "compatibility_changes": ["AST-load definitions only; no HTTP decorators/startup/weather import",
            "static request JSON instead of HTTP", "tqdm progress display disabled",
            "Geometry3D 0.2.4 with modern shapely 2.1.2; no old matplotlib downgrade",
            "numeric angle-cache restricted loader; no arbitrary pickle class resolution",
            "rebuild GeoJSON geometry once and use upstream cached-building RNG branch",
            "one snapshot instead of original request three; UMa/RMa inactive upstream branches retained"]}
    (out / "upstream_exact.json").write_text(json.dumps(exact, indent=2), encoding="utf-8")
    checked = {"commit": COMMIT, "scope": "controller-only correction; same unchanged physical simulator output",
        "corrections": ["remove hardcoded 33-value feedback overwrite; pass actual simulated per-MBS INR"],
        "controller": _controller_result(root / "DSA/controllers/ASCENTController.py", case, inr.tolist(), checked=True)}
    (out / "checked_controller.json").write_text(json.dumps(checked, indent=2), encoding="utf-8")
    with (out / "per_bs_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bs_index", "distance_m", "line_of_sight", "inr_db"])
        writer.writerows(zip(range(len(inr)), distances, los, inr))
    print(json.dumps({"aggregate_inr_db": exact["aggregate_inr_db"], "calls": exact["calls"],
                      "angle_cache_misses": cache_misses, "elapsed_seconds": exact["elapsed_seconds"],
                      "upstream_controller_active": exact["controller"]["active_count"],
                      "checked_controller_active": checked["controller"]["active_count"]}), flush=True)
    write_execution_checks(out)
    return exact, checked


def write_execution_checks(output):
    """Recheck saved numerical evidence without rerunning the costly geometry."""
    out = Path(output)
    exact = json.loads((out / "upstream_exact.json").read_text())
    checked = json.loads((out / "checked_controller.json").read_text())
    values = np.asarray(exact["per_bs_inr_db"])
    aggregate = float(10 * np.log10(np.sum(10 ** (values / 10))))
    checks = {"base_station_count_33": len(values) == 33,
        "finite_individual_inr": bool(np.isfinite(values).all()),
        "aggregate_is_linear_power_sum": bool(np.isclose(aggregate, exact["aggregate_inr_db"], atol=1e-10)),
        "original_simulator_actually_called": exact["calls"]["simulate"] == 1,
        "original_pathloss_actually_called": exact["calls"]["path_loss_UMi"] == 33,
        "original_beam_link_calculation_actually_called": exact["calls"]["Interface_UMi_1"] == 990,
        "no_original_beam_optimization_needed": exact["angle_cache_misses"] == 0,
        "exact_controller_keeps_feedback_overwrite": not exact["controller"]["feedback_overwrite_removed"],
        "checked_controller_removes_only_feedback_overwrite": checked["controller"]["feedback_overwrite_removed"]}
    if not all(checks.values()):
        raise RuntimeError("Official execution evidence failed its consistency checks")
    (out / "execution_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    lines = ["Pinned upstream ASCENT numerical execution", f"Commit: {exact['commit']}",
        "Source function bodies retained; HTTP/weather startup excluded by the reviewed AST loader.",
        "Original console can be empty because simulate(output=False) was selected.",
        f"Executed snapshot count: {exact['executed_snapshots']}",
        f"Recorded elapsed seconds: {exact['elapsed_seconds']}",
        f"Actual original calls: {json.dumps(exact['calls'], sort_keys=True)}",
        f"Aggregate INR before controller: {exact['aggregate_inr_db']:.9f} dB",
        f"Exact active BS count: {exact['controller']['active_count']}",
        f"Checked active BS count: {checked['controller']['active_count']}",
        f"Independent saved-evidence checks: {json.dumps(checks, sort_keys=True)}"]
    (out / "execution_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks


def planned_angle_keys(case, snapshots, seed=10):
    """Plan optimization-cache keys using the upstream RNG/angle expressions.

    This computes no physical result. The real simulator is subsequently run
    unchanged and checks every key against the resulting numerical cache.
    """
    rng = random.Random(seed)
    radius = 6.371e6
    lat, lon = case["lat_FSS"], case["lon_FSS"]
    xf = radius * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    yf = radius * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    stations = np.asarray([(radius * math.cos(math.radians(bs["latitude"])) * math.cos(math.radians(bs["longitude"])) - xf,
                            radius * math.cos(math.radians(bs["latitude"])) * math.sin(math.radians(bs["longitude"])) - yf)
                           for bs in case["base_stations"]])
    planned = []
    for _ in range(snapshots):
        channels = [rng.randint(0, 1) for _ in range(8)]
        users = []
        for bx, by in stations:
            for sector in range(3):
                for _ in range(10):
                    angle = rng.uniform(120 * sector, 120 * (sector + 1))
                    distance = rng.uniform(case["bs_ue_min_radius"], case["bs_ue_max_radius"])
                    users.append((bx + distance * math.cos(math.radians(angle)), by + distance * math.sin(math.radians(angle))))
                    rng.randint(1, 5)
        users = np.asarray(users)
        keys = []
        for bx, by in stations:
            for user in rng.sample(range(len(users)), 30):
                if not any(channels):
                    continue
                dx, dy = users[user] - [bx, by]
                for height in (10, 25, 35):
                    # Deliberately retain upstream atan/quadrant/elevation issues.
                    theta = float(np.degrees(np.arctan(dy / dx)) % 360)
                    phi = float(np.degrees(np.sqrt(dx * dx + dy * dy) / (1.5 - height)) % 360)
                    keys.append((round(theta), round(phi)))
        planned.append(keys)
    return planned


_BEAM_WORKER = None


def _initialize_beam_worker():
    global _BEAM_WORKER
    _BEAM_WORKER = load_upstream(require_geometry=False)


def _compute_original_beam_optimum(key):
    """Execute original brute/fmin optimization, not an approximate shortcut."""
    context = types.SimpleNamespace(saved_tp={})
    value = _BEAM_WORKER["max_gain_5g_parameters"](*key, context)
    return [float(key[0]), float(key[1]), float(value[0]), float(value[1])]


def run_full_case(output=None, snapshots=None, seed=10, workers=4):
    """Run the complete static request, continuous RNG, shared original caches."""
    root = ensure_sources()
    out = Path(output or Path(__file__).resolve().parent / "output/official_ascent")
    out.mkdir(parents=True, exist_ok=True)
    case = json.loads((root / "DSA/response.json").read_text())
    count = int(case["simulation_count"] if snapshots is None else snapshots)
    if count < 1:
        raise ValueError("snapshots must be positive")
    started = time.perf_counter()
    base_cache = load_numeric_cache(root)
    plans = planned_angle_keys(case, count, seed)
    missing = sorted(set(key for plan in plans for key in plan) - base_cache.keys())
    preflight = [{"snapshot": i + 1, "queries": len(plan), "cached_queries": sum(key in base_cache for key in plan),
                  "unique_missing": len(set(plan) - base_cache.keys())} for i, plan in enumerate(plans)]
    print(json.dumps({"planned_snapshots": count, "preflight": preflight, "unique_original_optimizations_needed": len(missing)}), flush=True)
    cache_file = out / f"additional_original_angle_cache_{count}snapshots.json"
    cache_rows = []
    if cache_file.exists():
        saved = json.loads(cache_file.read_text())
        if saved.get("commit") == COMMIT and saved.get("original_source_sha256") == EXPECTED_HASHES["Simulator.py"]:
            cache_rows = saved["numeric_rows"]
    existing = {tuple(row[:2]) for row in cache_rows}
    remaining = [key for key in missing if key not in existing]
    cache_started = time.perf_counter()
    if remaining:
        print(f"Computing {len(remaining)} original beam optimizations using {workers} processes", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_beam_worker) as pool:
            for index, row in enumerate(pool.map(_compute_original_beam_optimum, remaining, chunksize=16)):
                cache_rows.append(row)
                if (index + 1) % 256 == 0:
                    print(f"Original optimizations complete: {index + 1}/{len(remaining)}", flush=True)
    extra = {tuple(row[:2]): tuple(row[2:]) for row in cache_rows}
    if any(key not in extra for key in missing):
        raise RuntimeError("Planned numerical cache incomplete")
    cache_record = {"commit": COMMIT, "original_source_sha256": EXPECTED_HASHES["Simulator.py"],
        "method": "original max_gain_5g_parameters and original beam_pattern_5g; independent deterministic tasks",
        "cache_planning_only": preflight, "original_uncached_calls": len(cache_rows),
        "parallel_workers": workers, "cache_generation_seconds_this_run": round(time.perf_counter() - cache_started, 3),
        "numeric_rows": cache_rows}
    cache_file.write_text(json.dumps(cache_record, indent=2), encoding="utf-8")
    print("Preparing original full-case geometry", flush=True)
    logging.getLogger("Geometry3D").setLevel(logging.ERROR)
    namespace = load_upstream(require_geometry=True)
    ctx, geometry_info = build_context(namespace, case, seed, root)
    ctx.saved_tp.update(extra)
    calls, hits, misses = Counter(), 0, 0
    for name in ("simulate", "path_loss_UMi", "Interface_UMi_1", "gain_5g", "gain_fss_wbes_b", "beam_pattern_5g"):
        original = namespace[name]
        def counted(*args, _function=original, _name=name, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)
        namespace[name] = counted
    scan = namespace["max_gain_5g_parameters"]
    def scan_checked(theta, phi, context, coarse=True, rounding_precision=0):
        nonlocal hits, misses
        key = (round(theta, rounding_precision), round(phi, rounding_precision)) if coarse else (theta, phi)
        if key not in context.saved_tp:
            misses += 1
            raise RuntimeError("Unexpected beam-cache miss during full simulation; stop to quantify")
        hits += 1
        return scan(theta, phi, context, coarse, rounding_precision)
    namespace["max_gain_5g_parameters"] = scan_checked
    random.seed(seed)
    np.random.seed(seed)
    rows, ratio_rows = [], []
    log = io.StringIO()
    for index in range(count):
        calls_before, hits_before = dict(calls), hits
        snapshot_started = time.perf_counter()
        print(f"Original simulate snapshot {index + 1}/{count}", flush=True)
        with contextlib.redirect_stdout(log):
            result = namespace["simulate"](output=False, ctx=ctx)
        ratios = np.asarray(result[5])
        ratio_rows.append(ratios)
        rows.append({"snapshot": index + 1, "per_bs_inr_linear": ratios.tolist(),
            "per_bs_inr_db": (10 * np.log10(np.maximum(ratios, 1e-300))).tolist(),
            "aggregate_inr_db": float(10 * np.log10(np.sum(ratios))),
            "line_of_sight": np.asarray(result[6]).tolist(),
            "calls": {name: amount - calls_before.get(name, 0) for name, amount in calls.items()},
            "cache_hits": hits - hits_before, "elapsed_seconds": round(time.perf_counter() - snapshot_started, 3)})
        partial = {"commit": COMMIT, "completed_snapshots": index + 1, "requested_snapshots": count, "snapshots": rows}
        (out / f"full_case_{count}snapshots_progress.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        print(f"Completed original snapshot {index + 1}: INR {rows[-1]['aggregate_inr_db']:.6f} dB", flush=True)
    matrix = np.asarray(ratio_rows)
    # Exact upstream run_simulator reductions (not arithmetic mean of dB).
    per_bs_flat_db = (10 * np.log10(np.maximum(matrix.ravel(), 1e-300))).tolist()
    linear_average_all_bs_snapshots_db = float(10 * np.log10(np.mean(matrix)))
    per_bs_mean_linear = np.mean(matrix, axis=0)
    first_evidence = out / "upstream_exact.json"
    first_matches = None
    if first_evidence.exists() and seed == 10:
        old = json.loads(first_evidence.read_text())
        first_matches = bool(np.allclose(rows[0]["per_bs_inr_db"], old["per_bs_inr_db"], rtol=0, atol=1e-10))
        if not first_matches:
            raise RuntimeError("First full-case snapshot differs from preserved one-snapshot evidence")
    record = {"commit": COMMIT, "case": "complete upstream static sunny request", "seed": seed,
        "requested_simulation_count": case["simulation_count"], "executed_snapshots": count,
        "complete_original_request": count == case["simulation_count"], "continuous_rng_without_reseeding": True,
        "shared_original_los_and_angle_cache": True, "geometry": geometry_info,
        "snapshots": rows, "upstream_flat_per_bs_snapshots_db": per_bs_flat_db,
        "upstream_linear_mean_over_bs_and_snapshots_db": linear_average_all_bs_snapshots_db,
        "aggregate_mean_linear_over_snapshots_db": float(10 * np.log10(np.sum(per_bs_mean_linear))),
        "per_bs_mean_linear_inr_db": (10 * np.log10(per_bs_mean_linear)).tolist(),
        "mean_definition_note": "Upstream averages the linear per-BS ratios across all BS/snapshot entries before dB; aggregate sum also supplied separately",
        "calls": dict(calls), "cache_hits": hits, "unplanned_cache_misses": misses,
        "additional_cache_file": cache_file.name, "additional_cache_sha256": hashlib.sha256(cache_file.read_bytes()).hexdigest(),
        "first_snapshot_matches_preserved_single_snapshot": first_matches,
        "source_sha256": EXPECTED_HASHES, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "simulator_function_body_unchanged": True,
        "control_scope": "Physical simulator only; DSA controller is separate and per-MBS, with known feedback overwrite",
        "compatibility_changes": "Same documented AST/offline/modern-runtime setup as the preserved single snapshot; only deterministic angle optimizations precomputed with original code in parallel"}
    controller_source = root / "DSA/controllers/ASCENTController.py"
    record["controller_exact_flat_feedback"] = _controller_result(controller_source, case, per_bs_flat_db)
    record["controller_checked_flat_feedback"] = _controller_result(controller_source, case, per_bs_flat_db, checked=True)
    record["controller_checked_grouped_feedback"] = _controller_result(controller_source, case, record["per_bs_mean_linear_inr_db"], checked=True)
    record["feedback_shape_note"] = "Unchanged run_simulator flattens BS/snapshot outputs; 99 values conflict with controller BS_COUNT=33. Grouped feedback is an explicit checked adapter, not the original interface."
    (out / f"full_case_{count}snapshots.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / f"full_case_{count}snapshots_console.log").write_text(log.getvalue(), encoding="utf-8")
    print(json.dumps({"complete_original_request": record["complete_original_request"],
        "snapshots": count, "upstream_linear_mean_db": linear_average_all_bs_snapshots_db,
        "aggregate_mean_db": record["aggregate_mean_linear_over_snapshots_db"],
        "first_snapshot_matches": first_matches, "calls": dict(calls), "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    return record


def check_full_case_saved(output):
    """Validate full-request evidence without repeating expensive geometry."""
    out = Path(output)
    record = json.loads((out / "full_case_3snapshots.json").read_text(encoding="utf-8"))
    matrix = np.array([10 ** (np.asarray(row["per_bs_inr_db"]) / 10) for row in record["snapshots"]])
    aggregate = float(10 * np.log10(np.sum(np.mean(matrix, axis=0))))
    mean = float(10 * np.log10(np.mean(matrix)))
    cache = out / record["additional_cache_file"]
    checks = {
        "full_request_three_snapshots": record["executed_snapshots"] == record["requested_simulation_count"] == 3,
        "shape_three_by_33": matrix.shape == (3, 33),
        "finite_full_case_data": bool(np.isfinite(matrix).all()),
        "unchanged_original_simulator_three_calls": record["calls"]["simulate"] == 3,
        "original_pathloss_99_calls": record["calls"]["path_loss_UMi"] == 99,
        "first_snapshot_matches_single_evidence": record["first_snapshot_matches_preserved_single_snapshot"],
        "aggregate_linear_mean_correct": bool(np.isclose(aggregate, record["aggregate_mean_linear_over_snapshots_db"], atol=1e-10)),
        "upstream_per_station_mean_correct": bool(np.isclose(mean, record["upstream_linear_mean_over_bs_and_snapshots_db"], atol=1e-10)),
        "original_optimizer_cache_hash_matches": hashlib.sha256(cache.read_bytes()).hexdigest() == record["additional_cache_sha256"],
        "no_unplanned_optimization_miss": record["unplanned_cache_misses"] == 0,
        "checked_controller_rejects_flat_99_values": record["controller_checked_flat_feedback"]["message"]["status"] == "fail",
        "explicit_grouped_adapter_accepts_33_values": record["controller_checked_grouped_feedback"]["message"]["status"] == "success",
    }
    if not all(checks.values()):
        raise RuntimeError("Full upstream case consistency check failed")
    (out / "full_case_execution_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--check-saved", action="store_true", help="核验已有官方运行证据，不重跑建筑几何")
    parser.add_argument("--snapshots", type=int, help="运行完整请求或指定重复数；不覆盖既有单快照证据")
    parser.add_argument("--full-case", action="store_true", help="按原请求的simulation_count完整运行")
    parser.add_argument("--workers", type=int, default=4, help="缺失角度缓存的原算法并行计算进程数")
    args = parser.parse_args()
    if args.check_saved:
        saved_output = args.output or Path(__file__).resolve().parent / "output/official_ascent"
        checks = {"single": write_execution_checks(saved_output)}
        if (saved_output / "full_case_3snapshots.json").exists():
            checks["full"] = check_full_case_saved(saved_output)
        print(json.dumps(checks, indent=2))
    elif args.full_case or args.snapshots is not None:
        run_full_case(args.output, args.snapshots, args.seed, args.workers)
    else:
        run_case(args.output, args.seed)
