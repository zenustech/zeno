"""Record, inspect and replay a static OptiX camera/streaming session.

The normal UI also provides Start/Stop camera recording in Option > Ray Tracing.
Replay preserves the complete prelude; it does not reconstruct an arbitrary
initial warm cache or promise deterministic CUDA/IO scheduling.
"""
import argparse
import copy
import csv
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(directory, partial=False):
    directory = Path(directory).resolve()
    schema = json.loads((directory / "schema.json").read_text())
    if schema["version"] != 1 or len(set(schema["fields"])) != len(schema["fields"]):
        raise ValueError("Unsupported camera recording schema")
    files = list(directory.glob("camera-*.csv"))
    if len(files) != 1:
        raise ValueError("Expected one camera CSV in the recording directory")
    text = files[0].read_text()
    footer = [line for line in text.splitlines() if line.startswith("# dropped_batches=")]
    if not footer and not partial:
        raise ValueError("Recording has not been closed; stop recording before replay")
    if footer and int(footer[-1].split("=", 1)[1]) != 0:
        raise ValueError("Recording dropped frames; a faithful replay is unavailable")
    reader = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    if reader.fieldnames != ["sequence", "timestamp_ms", "duration_ms", *schema["fields"]]:
        raise ValueError("CSV fields disagree with the schema")
    rows = []
    for raw in reader:
        try:
            row = {key: float(value) for key, value in raw.items()}
            if len(row) != len(schema["fields"]) + 3 or not all(math.isfinite(x) for x in row.values()):
                raise ValueError("Non-finite or incomplete camera row")
        except (ValueError, TypeError):
            if partial and not footer and next(reader, None) is None:
                break
            raise ValueError("Malformed camera recording row") from None
        if row["sequence"] != len(rows) + 1 or (rows and row["timestamp_ms"] < rows[-1]["timestamp_ms"]):
            raise ValueError("Camera recording has missing/out-of-order frames")
        rows.append(row)
    if not rows:
        raise ValueError("Camera recording has no rendered frames")
    status_file = directory / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if not status["complete"] or status["frames"] != len(rows):
            raise ValueError("Recording did not finish cleanly: " + status.get("error", ""))
    elif not partial:
        raise ValueError("Recording status is missing; use --allow-partial only for crash recovery")
    return schema, rows


def summary(directory, rows):
    times = sorted(row["duration_ms"] for row in rows)
    slow = sorted(rows, key=lambda row: row["duration_ms"], reverse=True)[:20]
    queues = [key for key in rows[0] if key.startswith("texture_")]
    return dict(frames=len(rows), seconds=(rows[-1]["timestamp_ms"]+rows[-1]["duration_ms"]-rows[0]["timestamp_ms"])/1000,
                medianMs=statistics.median(times), p95Ms=times[int(.95*(len(times)-1))],
                p99Ms=times[int(.99*(len(times)-1))], over100=sum(t >= 100 for t in times),
                over200=sum(t >= 200 for t in times), cameraBytes=sum(p.stat().st_size for p in Path(directory).glob("camera-*.csv")),
                queueRanges={key:dict(first=rows[0][key],last=rows[-1][key],peak=max(row[key] for row in rows)) for key in queues},
                slowFrames=[dict(
                    {key: row[key] for key in ["sequence", "duration_ms", "renderer_ms", "geometry_ms", "launch_ms", "request_ms", "readback_ms", "marker", "eye_x", "eye_y", "eye_z", *queues]},
                    before_renderer_ms=row["render_start_ms"]-row["timestamp_ms"],
                    after_renderer_measurement_ms=row["duration_ms"]-row["renderer_ms"]-(row["render_start_ms"]-row["timestamp_ms"]))
                    for row in slow])


def make_script(schema, rows, pacing="recorded", freeze="none", freeze_after=None):
    if freeze != "none" and (freeze_after is None or not 1 <= freeze_after <= len(rows)):
        raise ValueError("Choose --freeze-after after the residency warm-up")
    first = rows[0]["timestamp_ms"]
    steps = []
    for row in rows:
        eye = [row["eye_"+axis] for axis in "xyz"]
        front = [row["front_"+axis] for axis in "xyz"]
        after = freeze_after is not None and row["sequence"] >= freeze_after
        steps.append(dict(ticks=1, recordedStartMs=row["timestamp_ms"]-first,
            cameraRecord=[row[name] for name in schema["fields"]],
            view=dict(position=eye, lookat=[a+b for a,b in zip(eye, front)], up=[row["up_"+axis] for axis in "xyz"],
                      fov=math.degrees(2*math.atan(row["sensor_height"]/(2*row["focal_length"])))),
            pixels=[int(row["window_width"]), int(row["window_height"])],
            freezeTextures=bool(row["freeze_textures"]) or (after and freeze in ["textures", "both"]),
            freezeGeometry=bool(row["freeze_geometry"]) or (after and freeze in ["geometry", "both"])))
    return dict(cameraSessionVersion=1, cameraSessionFields=schema["fields"], paced=pacing == "recorded",
                resourceCheckpoints=False, logSteps=False, steps=steps)


def scene_metadata(directory, rows):
    ids = {int(row["scene_id"]) for row in rows}
    if len(ids) != 1:
        raise ValueError("This replay supports one static scene; record a new session after changing/reloading its assets")
    return json.loads((directory / f"scene-{ids.pop()}.json").read_text())


def replay(args):
    args.output = args.output.resolve()
    directory = args.directory.resolve()
    schema, rows = load(directory, args.allow_partial)
    if args.stop_after is not None:
        if not 1 <= args.stop_after <= len(rows):
            raise ValueError("--stop-after must be a frame inside the recording")
        rows = rows[:args.stop_after]
    metadata = scene_metadata(directory, rows)
    if metadata.get("started_with_resident_scene"):
        print("Recording started with resident data. Cold replay cannot reconstruct that initial cache; include the full warm-up when recording.", flush=True)
    graph_path = args.graph or directory / "graph.zsg"
    graph = json.loads(graph_path.read_text())
    graph_info = directory / "graph-info.json"
    project = Path(json.loads(graph_info.read_text())["project_directory"]) if graph_info.exists() else graph_path.parent
    assets = []
    for subgraph in graph["graph"].values():
        for node in subgraph["nodes"].values():
            if node.get("name") == "LoadPrebakedAsset":
                assets.append(node)
            for socket in node.get("inputs", {}).values():
                value = socket.get("default-value")
                if socket.get("type") == "readpath" and isinstance(value, str) and value and not Path(value).is_absolute():
                    socket["default-value"] = str((project/value).resolve())
    if len(assets) != 1:
        raise ValueError("Expected the captured graph's single prebaked asset")
    manifest = Path(metadata["manifest"])
    if not manifest.is_file():
        raise ValueError(f"Captured geometry manifest is missing: {manifest}")
    inputs = assets[0]["inputs"]
    wanted = dict(manifest=str(manifest), geometryBudgetMiB=metadata["geometry_budget_bytes"]//2**20,
                  clasBudgetMiB=metadata["clas_budget_bytes"]//2**20, textureBudgetMiB=metadata["texture_budget_mib"],
                  textureHostMiB=metadata["host_budget_mib"], loadsPerFrame=metadata["loads_per_frame"], lodPixelError=metadata["pixel_error"])
    for key, value in wanted.items():
        inputs[key]["default-value"] = value
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"gray.zsg").exists():
        raise ValueError("Use a fresh output directory")
    script = make_script(schema, rows, args.pacing, args.freeze, args.freeze_after)
    motion = args.output/"camera-replay.json"
    motion.write_text(json.dumps(script, separators=(",", ":")))
    preferences = args.output/"preferences.ini"
    preferences.write_text("[General]\nrayTracingMaxBounce=4\nrayTracingDenoise=false\nenableSharc=false\n")
    spec = importlib.util.spec_from_file_location("camera_replay_validation", Path(__file__).with_name("validate_editor.py"))
    validation = importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)
    validation.graph = lambda _args: copy.deepcopy(graph)
    original_popen = validation.subprocess.Popen
    def popen(*pos, **kwargs):
        env = kwargs.get("env", os.environ).copy()
        for key, value in metadata.get("environment", {}).items():
            if value is None:env.pop(key, None)
            else:env[key] = value
        env.pop("ZENO_OPTIX_TEST_RANDOM_SEED", None)
        env["ZENO_OPTIX_RENDER_TIMING"] = "0"
        # The replay itself is re-recorded cheaply for camera/seed/quality checks.
        env["ZENO_OPTIX_CAMERA_RECORD"] = str(args.output/"observed")
        if args.diagnostics:
            env.update(ZENO_FRAME_CAPTURE=str(args.output/"diagnostics"), ZENO_FRAME_CAPTURE_ALL="1")
        else:
            env.pop("ZENO_FRAME_CAPTURE", None)
        kwargs["env"] = env
        if args.sanitizer and "compute-sanitizer" in str(pos[0][0]).lower():
            pos=([pos[0][0],"--port","47793",*pos[0][1:]],*pos[1:])
        return original_popen(*pos, **kwargs)
    validation.subprocess.Popen = popen
    sys.argv = ["validate_editor", "--editor", str(args.editor), "--manifest", str(manifest), "--output", str(args.output),
                "--viewport-script", str(motion), "--samples", "1", "--camera-samples", "1", "--interactive-only",
                "--pixels", f"{int(rows[0]['window_width'])}x{int(rows[0]['window_height'])}", "--test-preferences-file", str(preferences), "--timeout", str(args.timeout)]
    if args.sanitizer:sys.argv += ["--sanitizer", str(args.sanitizer)]
    else:sys.argv += ["--performance"]
    if args.optix_cache:sys.argv += ["--optix-cache", str(args.optix_cache)]
    validation.main()
    _, observed = load(args.output/"observed")
    observed_metadata = scene_metadata(args.output/"observed", observed)
    for key in ["manifest", "revision", "geometry_budget_bytes", "clas_budget_bytes", "texture_budget_mib", "host_budget_mib", "loads_per_frame", "pixel_error"]:
        if metadata[key] != observed_metadata[key]:
            raise ValueError(f"Replay scene settings changed: {key}")
    if len(observed) != len(rows):
        raise ValueError(f"Replay frame count changed: {len(rows)} -> {len(observed)}")
    quality_fields = ["width", "height", "window_width", "window_height", "render_ratio", "samples", "seed", "sample_start", "history_start",
                      "simple_render", "denoise", "max_bounce", "sharc", "debug_view", "adaptive_lod", "lod_pixel_error", "keep_history", "history_limit", "two_sided",
                      "moving", "picking", "clear_textures", "show_background", "need_aov", "sharc_spacing", "sharc_scale", "sharc_accumulation", "sharc_stale",
                      "sharc_roughness", "sharc_debug", "sharc_demodulation", "sharc_sh", "physical_aperture", "shutter_speed", "iso", "aces", "exposure", "panorama", "vr180", "pupil_distance"]
    pose_fields = [name for name in schema["fields"] if name.startswith(("eye_", "right_", "up_", "front_"))] + ["focal_length", "sensor_width", "sensor_height", "horizontal_shift", "vertical_shift", "pitch", "yaw", "aperture", "focal_distance"]
    for old, new in zip(rows, observed):
        different = [key for key in quality_fields+pose_fields if old[key] != new[key]]
        if different:
            raise ValueError(f"Replay did not preserve camera/quality at frame {int(old['sequence'])}: {different}")
    result = summary(args.output/"observed", observed)
    result.update(pacing=args.pacing, freeze=args.freeze, freezeAfter=args.freeze_after, cameraAndQualityMatched=True)
    (args.output/"replay-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def record(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"graph.zsg").write_bytes(args.graph.read_bytes())
    (args.output/"graph-info.json").write_text(json.dumps(dict(original_graph=str(args.graph.resolve()), project_directory=str(args.graph.resolve().parent), snapshot_kind="saved_graph"), indent=2))
    env = os.environ.copy();env["ZENO_OPTIX_CAMERA_RECORD"] = str(args.output.resolve())
    env.pop("ZENO_FRAME_CAPTURE", None);env["ZENO_OPTIX_RENDER_TIMING"] = "0"
    print(f"Recording to {args.output.resolve()}. Open/run {args.graph}; stop in Option > Ray Tracing or close Zeno.", flush=True)
    subprocess.run([str(args.editor.resolve())], cwd=args.editor.resolve().parent, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect");inspect.add_argument("directory", type=Path);inspect.add_argument("--allow-partial", action="store_true")
    run = commands.add_parser("replay");run.add_argument("directory", type=Path);run.add_argument("--output", type=Path, required=True)
    run.add_argument("--graph", type=Path);run.add_argument("--pacing", choices=["recorded", "fast"], default="recorded")
    run.add_argument("--freeze", choices=["none", "textures", "geometry", "both"], default="none");run.add_argument("--freeze-after", type=int)
    run.add_argument("--stop-after", type=int, help="Replay the full prelude through this frame, then stop")
    run.add_argument("--allow-partial", action="store_true");run.add_argument("--diagnostics", action="store_true")
    run.add_argument("--optix-cache", type=Path);run.add_argument("--sanitizer", type=Path);run.add_argument("--timeout", type=int, default=3600)
    capture = commands.add_parser("record");capture.add_argument("--graph", type=Path, required=True);capture.add_argument("--output", type=Path, required=True)
    for p in [run, capture]:p.add_argument("--editor", type=Path, default=ROOT/"build/bin/zenoedit.exe")
    args = parser.parse_args()
    if args.command == "record":record(args)
    elif args.command == "replay":replay(args)
    else:
        _, rows = load(args.directory, args.allow_partial);print(json.dumps(summary(args.directory, rows), indent=2))


if __name__ == "__main__":
    main()
