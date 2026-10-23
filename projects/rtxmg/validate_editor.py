"""Exercise the actual graph -> descriptor cache -> OptiX render child path.

Outputs, frame caches and OptiX disk caches stay under --output. The source
asset is only referenced by the graph and is never modified by this script.
"""
import argparse
import array
import copy
import json
import math
import os
import re
import statistics
from pathlib import Path
import subprocess
import tempfile
import time


def normalize(vector):
    length = math.sqrt(sum(x * x for x in vector))
    if length < 1e-10:
        raise ValueError("Camera view/up vectors must be nonzero and independent")
    return [x / length for x in vector]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def layout_nodes(nodes):
    # Expanded panels are substantially taller than 200 graph units. Keep
    # asset setup above the left-to-right camera chain, with room for inputs.
    positions={"geometry": [0,0], "scene": [850,0], "camera": [0,850],
               "physical-camera": [850,850], "diagnostic-index": [-650,0],
               "mirror-read":[0,2000],"mirror-bind":[850,2000],"mirror-scene":[1700,2000],
               "mirror-material":[0,2850]}
    for index,(identifier,node) in enumerate(nodes.items()):
        node["uipos"]=positions.get(identifier,[1700,index*850])
    xs=[node["uipos"][0] for node in nodes.values()]
    ys=[node["uipos"][1] for node in nodes.values()]
    return {"x":min(xs)-80,"y":min(ys)-80,"width":max(xs)-min(xs)+960,"height":max(ys)-min(ys)+900}


def graph(args):
    eye, lookat, up, fov = args.eye, args.lookat, [0., 1., 0.], args.fov
    if args.camera_scene:
        view = json.loads(args.camera_scene.read_text())["view"]
        eye, lookat, up, fov = view["position"], view["lookat"], view["up"], view["fov"]
    camera_path = json.loads(args.camera_path.read_text())["views"] if args.camera_path else []
    if camera_path:
        view = camera_path[0]
        eye, lookat, up, fov = view["position"], view["lookat"], view["up"], view["fov"]
    view = normalize([b - a for a, b in zip(eye, lookat)])
    up = normalize(cross(normalize(cross(view, up)), view))
    nodes, descriptions = {}, {}

    def add(identifier, name, fields, output, category, params=None):
        controls = {"vec3f": "Vector 3", "vec3i":"Vector 3", "float": "Float", "int": "Integer", "bool": "Check Box", "readpath": "read path", "string": "String", "": "None"}
        nodes[identifier] = {
            "name": name,
            "inputs": {key: {"link": None, "type": kind, "default-value": value, "control": {"name": controls[kind]}} for kind, key, value in fields},
            "params": params or {}, "outputs": {output: {"type": ""}, "DST": {"type": ""}},
            "uipos": [0,0], "options": ["VIEW"],
        }
        descriptions[name] = {
            "inputs": [[kind, key, json.dumps(value) if isinstance(value, list) else str(value)] for kind, key, value in fields] + [["", "SRC", ""]],
            "params": [], "outputs": [["", output, ""], ["", "DST", ""]], "categories": [category],
        }

    if args.ordinary:
        add("geometry", "CreateSphere", [
            ("vec3f", "position", [0., 0., 0.]), ("vec3f", "scaleSize", [1., 1., 1.]),
            ("float", "radius", 1.), ("vec3f", "rotate", [0., 0., 0.]),
            ("bool", "hasNormal", True), ("bool", "hasVertUV", False), ("bool", "isFlipFace", False),
            ("int", "rows", 24), ("int", "columns", 48), ("bool", "quads", False), ("bool", "SphereRT", False),
        ], "prim", "create", {"EulerRotationOrder": "XYZ", "EulerAngleMeasure": "Degree"})
        nodes["geometry"]["options"] = []
        add("scene", "MakeSceneNode_v2", [
            ("", "prim", None), ("string", "stampMode", "UnChanged"), ("string", "ObjectName", "RegressionSphere"),
            ("string", "type", "dynamic"), ("string", "matrixMode", "TotalChange"), ("bool", "centralize", False),
        ], "scene", "Scene")
        nodes["scene"]["inputs"]["prim"]["link"] = "main:geometry:[node]/outputs/prim"
    else:
        add("geometry", "LoadPrebakedAsset", [
            ("readpath", "manifest", args.manifest.resolve().as_posix()), ("string", "objectName", "ZorahGray"),
            ("int", "geometryBudgetMiB", args.pool), ("int", "clasBudgetMiB", args.pool),
            ("float", "lodPixelError", args.error), ("int", "loadsPerFrame", args.loads),
        ], "asset", "read")
    if args.mirror_obj:
        add("mirror-read","MustReadObjPrim",[("readpath","path",args.mirror_obj.resolve().as_posix())],
            "prim","primitive",{"triangulate":"1"})
        descriptions["MustReadObjPrim"]["params"]=[["bool","triangulate","1"]]
        nodes["mirror-read"]["params"]={"triangulate":{"value":True,"control":{"name":"Check Box"},"type":"bool"}}
        nodes["mirror-read"]["options"]=[]
        add("mirror-bind","BindMaterial",[("","object",None),("string","mtlid","ReflectionTestMetal")],"object","shader")
        nodes["mirror-bind"]["options"]=[]
        nodes["mirror-bind"]["inputs"]["object"]["link"]="main:mirror-read:[node]/outputs/prim"
        add("mirror-scene","MakeSceneNode_v2",[("","prim",None),("string","stampMode","TotalChange"),
            ("string","ObjectName","SourceMirror"),("string","type","dynamic"),
            ("string","matrixMode","TotalChange"),("bool","centralize",False)],"scene","Scene")
        nodes["mirror-scene"]["inputs"]["prim"]["link"]="main:mirror-bind:[node]/outputs/object"
        add("mirror-material","ShaderFinalize",[("float","base",1.),("vec3f","basecolor",[1.,1.,1.]),
            ("float","metallic",1.),("vec3f","metalColor",[1.,1.,1.]),("float","specular",1.),
            ("float","roughness",args.mirror_roughness),("float","smoothness",0.),("float","doubleSide",1.),
            ("float","opacity",0.),("float","shadowReceiver",0.),("string","sssType","Fixed"),
            ("string","mtlid","ReflectionTestMetal"),("vec3i","mask_value",[255,0,0])],"mtl","shader",
            {"backend":"CUDA","caustic":"none"})
        descriptions["ShaderFinalize"]["params"]=[["enum CUDA","backend","CUDA"],["enum none caster receiver","caustic","none"]]
        nodes["mirror-material"]["params"]={key:{"value":value,"control":{"name":"Enum","items":items},
            "type":"enum "+" ".join(items)} for key,value,items in
            [("backend","CUDA",["CUDA"]),("caustic","none",["none","caster","receiver"])]}
    add("camera", "MakeCamera", [
        ("vec3f", "pos", eye), ("vec3f", "view", view), ("vec3f", "up", up),
        ("float", "near", .01), ("float", "far", 20000.), ("float", "fov", fov),
        ("float", "aperture", 0.), ("float", "focalPlaneDistance", math.sqrt(sum((b-a)**2 for a,b in zip(eye,lookat)))),
    ], "camera", "shader")
    nodes["camera"]["options"] = []
    # The CUDA backend uses the physical-camera sample count for recording.
    add("physical-camera", "SetPhysicalCamera", [
        ("", "camera", None), ("float", "aperture", 0.), ("float", "shutter_speed", .04),
        ("float", "iso", 150.), ("int", "renderRatio", 1), ("bool", "aces", False),
        ("bool", "exposure", False), ("bool", "panorama_camera", False), ("bool", "panorama_vr180", False),
        ("float", "pupillary_distance", .06), ("int", "num_samples", args.camera_samples if args.camera_samples is not None else args.samples),
        ("int", "ray_bounce", 4), ("bool", "up2x", False),
    ], "camera", "shader")
    nodes["physical-camera"]["inputs"]["camera"]["link"] = "main:camera:[node]/outputs/camera"
    if camera_path:
        # Reevaluate one graph with distinct camera nodes. Only the selected
        # camera reaches SetPhysicalCamera; the asset reference stays identical.
        inputs = []
        cameras = {}
        for index, pose in enumerate(camera_path):
            identity = json.dumps(pose, sort_keys=True)
            identifier = cameras.get(identity)
            if identifier is None:
                identifier = "camera" if not cameras else f"camera-{len(cameras)}"
                cameras[identity] = identifier
                if identifier != "camera":
                    nodes[identifier] = copy.deepcopy(nodes["camera"])
                values = {"pos": pose["position"], "fov": pose["fov"]}
                values["view"] = normalize([b-a for a,b in zip(pose["position"],pose["lookat"])])
                values["up"] = normalize(cross(normalize(cross(values["view"],pose["up"])),values["view"]))
                values["focalPlaneDistance"] = math.dist(pose["position"],pose["lookat"])
                for key, value in values.items():
                    nodes[identifier]["inputs"][key]["default-value"] = value
            inputs.append(("",f"obj{index}",None))
        add("camera-list","MakeSmallList",inputs,"list","list")
        add("camera-frame","GetFrameNum",[],"FrameNum","frame")
        add("camera-select","ListGetItem",[("","list",None),("int","index",0)],"object","list")
        for index,pose in enumerate(camera_path):
            identifier = cameras[json.dumps(pose,sort_keys=True)]
            nodes["camera-list"]["inputs"][f"obj{index}"]["link"] = f"main:{identifier}:[node]/outputs/camera"
        nodes["camera-select"]["inputs"]["list"]["link"] = "main:camera-list:[node]/outputs/list"
        nodes["camera-select"]["inputs"]["index"]["link"] = "main:camera-frame:[node]/outputs/FrameNum"
        nodes["physical-camera"]["inputs"]["camera"]["link"] = "main:camera-select:[node]/outputs/object"
        for identifier in ("camera-list","camera-frame","camera-select"):
            nodes[identifier]["options"] = []
    return {"graph": {"main": {"nodes": nodes, "view_rect": layout_nodes(nodes)}}, "descs": descriptions,
            "views": {"timeline": {"start-frame": 0, "end-frame": args.frames-1, "curr-frame": 0, "always": False}},
            "version": "v2.5", "settings": {}}


def summarize_viewport(log, args):
    """Keep render, active host work, queue waits and completed GPU tasks distinct."""
    def stats(values):
        if not values:
            return None
        ordered = sorted(values)
        return {"count": len(values), "mean": statistics.mean(values),
                "median": statistics.median(values), "p95": ordered[int((len(ordered)-1)*.95)],
                "max": ordered[-1]}

    number = r"([\d.eE+-]+)"
    viewport = [float(v) for v in re.findall(r"OptiX viewport: .*?total="+number+r" ms", log)]
    update_pattern = r"CLAS update: "+number+r" ms, .*?settled=(\d+), pending=(\d+)"
    pattern = (r"CLAS GPU: LOD="+number+r" ms, CLAS="+number+r" ms, GAS="+number+
               r" ms, IAS="+number+r" ms, scene="+number+r" ms, upload="+number+
               r" ms; host staging="+number+r" ms, background prepare="+number+
               r" ms, background IO="+number+r" ms, version=(\d+)")
    wait_pattern = (r"CLAS scheduling: render submission wait="+number+
                    r" ms, background submission wait="+number+r" ms, render event wait="+number+r" ms")
    updates, gpu, waits, host_tasks = [], [], [], []
    latest_update = latest_gpu = latest_wait = None
    latest_host = 0
    for line in log.splitlines():
        match = re.search(update_pattern, line)
        if match:
            latest_update = match.groups()
        match = re.search(pattern, line)
        if match:
            latest_gpu = tuple(map(float, match.groups()))
        match = re.search(wait_pattern, line)
        if match:
            latest_wait = match.groups()
        match = re.search(r"CLAS completed task: active host work="+number+r" ms", line)
        if match:
            latest_host = float(match.group(1))
        if 'OptiX viewport:' in line:
            if latest_update is not None:
                updates.append(latest_update)
            if latest_gpu is not None:
                gpu.append(latest_gpu)
            if latest_wait is not None:
                waits.append(latest_wait)
            if latest_host > 0:
                host_tasks.append(latest_host)
            latest_update = latest_gpu = latest_wait = None
            latest_host = 0
    motion = args.viewport_iterations
    result = {"configuration": {"pixels": args.pixels, "viewport_samples": args.samples,
              "motion_ticks": motion, "pan_step": args.viewport_pan_step,
              "hold_ticks": args.viewport_settle_iterations, "pixel_error": args.error,
              "geometry_pool_mib": args.pool, "clas_pool_mib": args.pool},
              "motion_viewport_ms": stats(viewport[:motion]), "all_viewport_ms": stats(viewport)}
    if args.ordinary:
        return result
    if len(viewport) != len(updates) or len(viewport) != len(gpu):
        raise RuntimeError("Incomplete per-frame timing records")
    completed = [row for row in gpu if row[4] > 0]
    result.update(motion_render_thread_update_ms=stats([float(row[0]) for row in updates[:motion]]),
                  gpu_phases_ms={key: stats([row[i] for row in completed])
                                 for i, key in enumerate(["lod", "clas", "gas", "ias", "scene", "upload"])},
                  host_staging_ms=stats([row[6] for row in gpu if row[6] > 0]),
                  worker_prepare_ms=stats([row[7] for row in gpu if row[7] > 0]),
                  read_decode_ms=stats([row[8] for row in gpu if row[8] > 0]))
    result["completed_task_active_host_work_ms"] = stats(host_tasks)
    if waits:
        result["scheduling_ms"] = {key: stats([float(row[i]) for row in waits]) for i, key in enumerate(
            ["render_submission_wait", "background_submission_wait", "render_event_wait"])}
    first = next((i for i in range(motion, len(updates)) if updates[i][1:] == ('1', '0')), None)
    if first is not None:
        result.update(settle_ticks_after_motion=first-motion+1,
                      settle_ms=sum(viewport[motion:first+1]), settled_viewport_ms=stats(viewport[first+1:]))
    result["final_status"] = re.findall(r"CLAS update: .*", log)[-1]
    result["final_cut"] = re.findall(r"^\[CLAS\] version=.*", log, re.M)[-1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", type=Path, default=Path(__file__).resolve().parents[2] / "out/build/x64-Release/bin/zenoedit.exe")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--manifest", type=Path)
    scope.add_argument("--ordinary", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-scene", type=Path)
    parser.add_argument("--mirror-obj",type=Path,help="Add a source-resolution ordinary reflector to the actual material/BSDF path")
    parser.add_argument("--mirror-roughness",type=float,default=0.,help="Requested constant roughness of the white conductor test reflector")
    parser.add_argument("--camera-path", type=Path, help="Reevaluate the graph while selecting camera nodes from a views array (up to six snapshots)")
    parser.add_argument("--eye", nargs=3, type=float, default=[0., 3., 8.])
    parser.add_argument("--lookat", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--fov", type=float, default=45.)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--camera-samples", type=int, help="Physical-camera count independent of viewport/record control")
    parser.add_argument("--viewport-iterations", type=int, default=0, help="Exercise the actual viewport worker after recording")
    parser.add_argument("--viewport-pan-step",type=float,default=0.,help="Move camera right by this many world units per viewport tick")
    parser.add_argument("--viewport-dolly-step",type=float,default=0.,help="Move camera backward along its view axis per viewport tick")
    parser.add_argument("--viewport-end-camera",type=Path,help="Slowly interpolate the viewport pose to this camera scene during motion ticks")
    parser.add_argument("--viewport-script",type=Path,help="Repeated camera, FOV, resolution, reset and reload steps in one renderer process")
    parser.add_argument("--linear-capture",action="store_true",help="Also export the existing offline linear accumulation as a PFM file")
    parser.add_argument("--capture-motion",action="store_true")
    parser.add_argument("--interactive-only",action="store_true",help="Render reevaluated graph snapshots directly through the viewport, without an offline settle first")
    parser.add_argument("--camera-reset-test",action="store_true",help="Exercise Alt+G reset and restore the saved camera without reloading the graph")
    parser.add_argument("--camera-reset-ticks",type=int,default=300)
    parser.add_argument("--viewport-settle-iterations",type=int,default=0,help="Hold the final camera position for this many additional viewport ticks")
    parser.add_argument("--pixels", default="256x256")
    parser.add_argument("--hide-background",action="store_true",help="Hide the primary-ray environment background while retaining environment illumination")
    parser.add_argument("--pool", type=int, default=16)
    parser.add_argument("--loads", type=int, default=32)
    parser.add_argument("--error", type=float, default=2.)
    parser.add_argument("--debug-view", type=int, choices=range(8), default=0, help="0=shaded, 1=LOD, 2=clusters, 3=coverage, 4=geometric normal, 5=shading normal, 6=facing, 7=normal agreement")
    parser.add_argument("--cycle-debug-views", action="store_true", help="Check views 0-7 on eight identical static scene snapshots")
    parser.add_argument("--runtime-switch-test", action="store_true", help="Check quality transitions and the live Preferences worker slots on one scene snapshot")
    parser.add_argument("--keep-clas-history", action="store_true", help="Enable bounded interactive LOD history (offline recording remains strict)")
    parser.add_argument("--use-saved-clas-settings", action="store_true", help="Do not override persisted CLAS preferences with test environment flags")
    parser.add_argument("--test-preferences-file", type=Path, help="Read worker preferences through QSettings from an isolated INI file")
    parser.add_argument("--expect-clas-history", type=int, choices=[0, 1], help="Assert the first and all later shaded viewport frames use this history policy")
    parser.add_argument("--history-limit", type=int, default=16)
    parser.add_argument("--disable-sharc", action="store_true")
    parser.add_argument("--denoise", action="store_true", help="Exercise the denoiser and its accumulated normal/albedo guides")
    parser.add_argument("--history-regression", choices=["refinement", "maintenance"], help="Assert history behavior after motion or during forced AS-only compaction")
    parser.add_argument("--capture-history", action="store_true")
    parser.add_argument("--sample-sequence-start", type=int, help="Seed the first viewport sample ordinal for boundary testing")
    parser.add_argument("--random-seed",type=int,help="Use matched random numbers for controlled LOD image comparisons")
    parser.add_argument("--performance", action="store_true", help="Disable OptiX validation and enable CPU/GPU timings")
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument("--optix-cache", type=Path, help="Reuse an OptiX shader cache across diagnostic runs")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.mirror_obj and not args.mirror_obj.is_file():
        parser.error("Mirror OBJ must exist")
    if args.mirror_obj:
        args.linear_capture=True
    if not math.isfinite(args.mirror_roughness) or not 0<=args.mirror_roughness<=1:
        parser.error("Mirror roughness must be in [0,1]")
    if args.random_seed is not None and not 0<=args.random_seed<=0xffffffff:
        parser.error("Random seed must fit in 32 bits")
    if args.use_saved_clas_settings and args.history_regression:
        parser.error("History regression uses explicit controls; test saved preferences separately")
    if args.test_preferences_file and not args.test_preferences_file.is_file():
        parser.error("Test preferences file must exist")
    if args.expect_clas_history is not None and (not args.interactive_only or args.debug_view or args.cycle_debug_views):
        parser.error("History expectation needs an interactive-only shaded viewport test")
    if args.camera_path:
        views = json.loads(args.camera_path.read_text())["views"]
        if args.camera_scene or args.cycle_debug_views or not isinstance(views,list) or not 2 <= len(views) <= 6:
            parser.error("Camera path needs 2-6 views and cannot use another camera source or debug sweep")
        args.frames = len(views)
    if args.cycle_debug_views:
        if args.ordinary:
            parser.error("Diagnostic view cycling requires a cluster asset")
        args.frames = 8
    if args.camera_reset_test and (args.viewport_iterations or args.runtime_switch_test or args.frames != 1):
        parser.error("Camera reset test needs one frame and no other viewport test")
    if args.interactive_only and (not (args.viewport_iterations or args.viewport_script) or args.runtime_switch_test or args.camera_reset_test):
        parser.error("Interactive-only test needs viewport iterations and no other runtime test")
    if args.viewport_end_camera and (not args.viewport_iterations or args.viewport_pan_step or args.viewport_dolly_step):
        parser.error("End-camera interpolation needs viewport iterations and no pan/dolly step")
    if not 1 <= args.camera_reset_ticks <= 1000 or not all(math.isfinite(x) for x in (args.viewport_pan_step,args.viewport_dolly_step)):
        parser.error("Invalid camera-test step or tick count")
    if min(args.frames, args.samples, args.pool, args.loads) < 1:
        parser.error("Frame/sample counts and streaming settings must be positive")
    if args.camera_samples is not None and args.camera_samples<1:
        parser.error("Physical-camera samples must be positive")
    if not 4 <= args.history_limit <= 256:
        parser.error("History limit must be in [4, 256]")
    if args.history_regression and (args.ordinary or not args.viewport_iterations or args.frames != 1 or args.runtime_switch_test or args.debug_view != 0):
        parser.error("History regression needs one shaded cluster frame and viewport iterations")
    if args.history_regression == "refinement" and not args.viewport_settle_iterations:
        parser.error("Refinement regression needs hold ticks after camera movement")
    if not 0<=args.viewport_iterations<=1000 or not 0<=args.viewport_settle_iterations<=1000:
        parser.error("Viewport iterations must be in [0, 1000]")
    editor, output = args.editor.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "gray.zsg"
    graph_path.write_text(json.dumps(graph(args), indent=2))
    cache_parent = Path(tempfile.mkdtemp(prefix="frames-", dir=output))
    env = os.environ.copy()
    if args.performance and args.sanitizer:
        parser.error("Performance measurement must run without Compute Sanitizer")
    env.update(OPTIX_CACHE_PATH=str(args.optix_cache.resolve() if args.optix_cache else output / "optix-cache"), ZENO_OPTIX_VALIDATE="0" if args.performance else "1", ZENO_OPTIX_LOG_LEVEL="3")
    if args.performance or args.viewport_iterations or args.runtime_switch_test or args.camera_reset_test or args.camera_path or args.viewport_script:
        env["ZENO_OPTIX_RENDER_TIMING"] = "1"
    env["ZENO_OPTIX_VIEWPORT_BENCHMARK_FRAMES"] = str(args.viewport_iterations)
    env["ZENO_OPTIX_VIEWPORT_PAN_STEP"] = str(args.viewport_pan_step)
    env["ZENO_OPTIX_VIEWPORT_DOLLY_STEP"] = str(args.viewport_dolly_step)
    env["ZENO_OPTIX_VIEWPORT_CAMERA_END"] = json.dumps(json.loads(args.viewport_end_camera.read_text())["view"]) if args.viewport_end_camera else ""
    env["ZENO_OPTIX_VIEWPORT_SCRIPT"] = str(args.viewport_script.resolve()) if args.viewport_script else ""
    env["ZENO_OPTIX_LINEAR_CAPTURE"] = str(int(args.linear_capture))
    if args.random_seed is not None:
        env["ZENO_OPTIX_TEST_RANDOM_SEED"]=str(args.random_seed)
    else:
        env.pop("ZENO_OPTIX_TEST_RANDOM_SEED",None)
    env["ZENO_OPTIX_VIEWPORT_CAPTURE_MOTION"] = str(int(args.capture_motion))
    env["ZENO_OPTIX_VIEWPORT_SKIP_RECORD_CAPTURE"] = str(int(args.interactive_only))
    env["ZENO_OPTIX_VIEWPORT_TEST_CAMERA_RESET"] = str(int(args.camera_reset_test))
    env["ZENO_OPTIX_VIEWPORT_RESET_TICKS"] = str(args.camera_reset_ticks)
    env["ZENO_OPTIX_VIEWPORT_SETTLE_FRAMES"] = str(args.viewport_settle_iterations)
    env.update(ZENO_OPTIX_CLAS_DEBUG_VIEW=str(args.debug_view),
               ZENO_OPTIX_CLAS_DIAGNOSTIC_SWEEP=str(int(args.cycle_debug_views)))
    env["ZENO_OPTIX_VIEWPORT_TEST_RUNTIME_SWITCHES"] = str(int(args.runtime_switch_test))
    env.update(ZENO_OPTIX_CLAS_KEEP_HISTORY=str(int(args.keep_clas_history)),
               ZENO_OPTIX_CLAS_HISTORY_LIMIT=str(args.history_limit),
               ZENO_OPTIX_SHARC=str(int(not args.disable_sharc)),
               ZENO_OPTIX_VIEWPORT_CAPTURE_HISTORY=str(int(args.capture_history)))
    if args.use_saved_clas_settings:
        for key in ("ZENO_OPTIX_CLAS_DEBUG_VIEW", "ZENO_OPTIX_CLAS_KEEP_HISTORY", "ZENO_OPTIX_CLAS_HISTORY_LIMIT"):
            env.pop(key, None)
    if args.test_preferences_file:
        env["ZENO_OPTIX_TEST_PREFERENCES_FILE"] = str(args.test_preferences_file.resolve())
    if args.history_regression == "maintenance":
        env["ZENO_CLAS_FORCE_GAS_COMPACTION"] = "1"
    if args.sample_sequence_start is not None:
        env["ZENO_OPTIX_TEST_SAMPLE_SEQUENCE_START"] = str(args.sample_sequence_start)
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0

    def run(arguments, log_path):
        with log_path.open("wb") as log:
            process = subprocess.Popen(arguments, cwd=editor.parent, env=env, startupinfo=startup,
                                       stdout=log, stderr=subprocess.STDOUT)
            began, samples = time.monotonic(), []
            try:
                while process.poll() is None:
                    if time.monotonic()-began > args.timeout:
                        raise TimeoutError(f"Process timed out; inspect {log_path}")
                    if args.viewport_script and os.name == "nt":
                        import ctypes
                        from ctypes import wintypes
                        class Counters(ctypes.Structure):
                            _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [
                                (key,ctypes.c_size_t) for key in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage",
                                "QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage",
                                "PeakPagefileUsage","PrivateUsage")]
                        kernel = ctypes.WinDLL("kernel32",use_last_error=True)
                        kernel.OpenProcess.restype = wintypes.HANDLE
                        handle = kernel.OpenProcess(0x1000,False,process.pid)
                        if handle:
                            counters= Counters(); counters.cb=ctypes.sizeof(counters)
                            query=ctypes.WinDLL("psapi").GetProcessMemoryInfo
                            query.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
                            if query(handle,ctypes.byref(counters),counters.cb):
                                samples.append({"seconds":time.monotonic()-began,"working_set":counters.WorkingSetSize,
                                    "peak_working_set":counters.PeakWorkingSetSize,"private_bytes":counters.PrivateUsage})
                            kernel.CloseHandle.argtypes=[wintypes.HANDLE]; kernel.CloseHandle(handle)
                    time.sleep(1 if args.viewport_script else .1)
            finally:
                if process.poll() is None: process.terminate()
                process.wait(timeout=15)
            if samples: (output/"host-resources.json").write_text(json.dumps(samples))
        if process.returncode:
            raise RuntimeError(f"Process exited {process.returncode}; inspect {log_path}")
        return log_path.read_text(errors="replace")

    common = ["--zsg", str(graph_path), "--frame", str(args.frames), "--sframe", "0", "--sample", str(args.samples),
              "--pixel", args.pixels, "--path", str(output), "--cacheautorm", "0", "--optix", "1", "--video", "0", "--needDenoise", str(int(args.denoise))]
    if args.mirror_obj:
        common += ["--aov","1"]
    # This stage only evaluates the graph and writes descriptor caches. Avoid
    # initializing/loading the full OptiX asset twice; exercise it in the
    # explicitly logged render child below.
    cache_arguments = common.copy()
    cache_arguments[cache_arguments.index("--optix") + 1] = "0"
    cache_command = [str(editor), "--record", "true", *cache_arguments, "--cachePath", str(cache_parent), "--exitWhenRecordFinish", "true"]
    cache_log = output / "graph-record.log"
    with cache_log.open("wb") as log_file:
        process = subprocess.Popen(cache_command, cwd=editor.parent, env=env, startupinfo=startup,
                                   stdout=log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + args.timeout
        try:
            while True:
                folders = list(cache_parent.glob("zeno__cache__file__storage__path_*"))
                ready = len(folders) == 1 and all(
                    (folders[0] / f"{frame:06d}" / name).is_file()
                    for frame in range(args.frames)
                    for name in ("normalObj.zencache", "lightCameraObj.zencache", "materialObj.zencache", "matrixObj.zencache", "runInfo.txt"))
                # Runner exit confirms all cache writes and locks completed.
                # Raster recording has a UI completion race after that point;
                # only the generated cache is needed for this stage.
                runner_done = "runner process normally exited with 0" in cache_log.read_text(errors="replace")
                exit_code = process.poll()
                if ready and (runner_done or exit_code == 0):
                    break
                if exit_code is not None:
                    raise RuntimeError(f"Cache generation exited before completion; inspect {cache_log}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Cache generation timed out; inspect {cache_log}")
                time.sleep(.1)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
    caches = list(cache_parent.glob("zeno__cache__file__storage__path_*"))
    if len(caches) != 1:
        raise RuntimeError(f"Expected one generated descriptor cache under {cache_parent}")
    command = [str(editor), "--optixcmd", "0", *common, "--cachePath", str(caches[0]), "--optixShowBackground", str(int(not args.hide_background))]
    if args.sanitizer:
        command = [str(args.sanitizer.resolve()), "--tool", "memcheck", "--check-optix", "--error-exitcode", "99", *command]
    log = run(command, output / "optix-child.log")
    if '"result" : 0' not in log:
        raise RuntimeError("The render child did not report completion")
    if args.sanitizer and "ERROR SUMMARY: 0 errors" not in log:
        raise RuntimeError("Compute Sanitizer did not report zero errors")
    if args.mirror_obj and not args.interactive_only:
        captures=[]
        for mask in sorted((output/"P").glob("*.mask.pfm")):
            with mask.open("rb") as stream:
                if stream.readline()!=b"PF\n": raise RuntimeError("Invalid reflector mask capture")
                w,h=map(int,stream.readline().split())
                if stream.readline()!=b"-1.0\n": raise RuntimeError("Unsupported reflector mask byte order")
                values=array.array("f"); values.frombytes(stream.read())
            if len(values)!=w*h*3: raise RuntimeError("Incomplete reflector mask capture")
            pixels=sum(r>.99 and g==0 and b==0 for r,g,b in zip(values[::3],values[1::3],values[2::3]))
            if pixels<100: raise RuntimeError("Test reflector is invisible or its metal material is not bound")
            captures.append({"path":str(mask),"interior_mirror_pixels":pixels})
        if len(captures)!=args.frames: raise RuntimeError("Missing reflector material-mask captures")
        (output/"reflector-validation.json").write_text(json.dumps({"requested_roughness":args.mirror_roughness,
            "renderer_roughness":min(.99,max(.01,args.mirror_roughness)),"captures":captures},indent=2))
    if args.expect_clas_history is not None:
        applied = list(map(int, re.findall(r"CLAS history: .*?keep=(\d+)", log)))
        expected_count = args.frames * (args.viewport_iterations + args.viewport_settle_iterations)
        if len(applied) != expected_count or any(value != args.expect_clas_history for value in applied):
            raise RuntimeError("The first or a later viewport frame ignored the expected history policy")
        stored = re.findall(r"CLAS stored history: type=(\w+), bool=(\d+), int=(-?\d+), envOverride=(\d+)", log)
        if not stored or (args.use_saved_clas_settings and any(row[3] != "0" for row in stored)):
            raise RuntimeError("Saved-preference test did not exercise the startup settings reader")
        (output / "preferences-validation.json").write_text(json.dumps({
            "expected_history": args.expect_clas_history, "viewport_frames": len(applied),
            "first_frame_history": applied[0], "stored_values": stored,
            "environment_override": not args.use_saved_clas_settings}, indent=2))
    if args.runtime_switch_test:
        def section(label):
            return log.split("[viewport regression] " + label, 1)[1].split("[viewport regression]", 1)[0]
        quality = [section(label) for label in ("quality baseline", "quality simple ratio=1", "quality idle ratio=1")]
        subframes = [int(n) for part in quality for n in re.findall(r"OptiX render: .*?subframe=(\d+)", part)]
        if len(subframes) != 6 or subframes != list(range(subframes[0], subframes[0] + 6)):
            raise RuntimeError("The unchanged-resolution quality transition reset accumulation")
        for part in quality[1:]:
            if re.findall(r"OptiX camera: changed=(\d+), resized=(\d+)", part) != [("0", "0")] * 2:
                raise RuntimeError("The idle transition unnecessarily updated the camera or resolution")
        debug = section("runtime debug views")
        if list(map(int, re.findall(r"OptiX render: .*?debugView=(\d+)", debug))) != list(range(8)):
            raise RuntimeError("Runtime debug switches did not reach all eight render modes")
        if any(row != ("0", "0") for row in re.findall(r"OptiX camera: changed=(\d+), resized=(\d+)", debug)):
            raise RuntimeError("Runtime diagnostics unnecessarily updated the camera")
        plans = re.findall(r"CLAS update: .*?plans=(\d+)", debug)
        if plans and len(set(plans)) != 1:
            raise RuntimeError("Runtime debug changes unnecessarily replanned CLAS residency")
        restored = section("restore debug view")
        if re.findall(r"OptiX render: .*?debugView=(\d+)", restored) != [str(args.debug_view)]:
            raise RuntimeError("Restoring the shaded view did not reach the renderer")
        history_switch = section("runtime history options")
        applied = re.findall(r"CLAS history: .*?keep=(\d+)", history_switch)
        if args.debug_view == 0 and applied != [str(int(not args.keep_clas_history)), str(int(args.keep_clas_history))]:
            raise RuntimeError("Runtime history settings did not reach the renderer")
        sidedness = section("runtime opaque sidedness")
        sidedness_values = re.findall(r"CLAS material sidedness: twoSidedOpaque=(\d+), history reset=1", sidedness)
        initial_sidedness = re.findall(r"CLAS startup settings: .*?twoSidedOpaque=(\d+)", log)
        if not initial_sidedness or sidedness_values != [str(1-int(initial_sidedness[0])), initial_sidedness[0]]:
            raise RuntimeError("Runtime opaque sidedness did not toggle and restore the launch policy")
        if re.findall(r"OptiX camera: changed=(\d+), resized=(\d+)", sidedness) != [("0", "0")] * 2:
            raise RuntimeError("Runtime opaque sidedness unnecessarily updated the camera")
        sidedness_plans = re.findall(r"CLAS update: .*?plans=(\d+)", sidedness)
        if plans and sidedness_plans and set(sidedness_plans) != {plans[-1]}:
            raise RuntimeError("Runtime opaque sidedness unnecessarily replanned CLAS residency")
        w, h = map(int, args.pixels.split("x"))
        for label, size in (("quality simple ratio=2", f"{max(w//2,1)}x{max(h//2,1)}"), ("quality idle ratio=2", args.pixels)):
            part = section(label)
            if re.findall(r"OptiX viewport: (\d+x\d+),", part) != [size, size]:
                raise RuntimeError("Actual quality resolution changes were not applied")
        camera_positions = re.findall(r"OptiX camera: .*?pos=([^,]+,[^,]+,[^,]+), simple=", log.split("[viewport regression] quality baseline", 1)[1])
        if len(set(camera_positions)) != 1:
            raise RuntimeError("The fixed camera changed during runtime switches")
        (output / "runtime-switch-validation.json").write_text(json.dumps({"same_resolution_subframes":subframes,
            "camera_positions":list(set(camera_positions)),"debug_modes":list(range(8)),"clas_plan_versions":sorted(set(plans)),
            "history_policy_switches":list(map(int, applied)),
            "opaque_sidedness_switches":list(map(int, sidedness_values)),
            "checks":"quality accumulation, real resize, camera stability, runtime modes, shaded-view restore, history and opaque sidedness policies passed"}, indent=2))
    if args.viewport_iterations:
        selections=re.findall(r"OptiX sample selection: mode=viewport, viewportSamples=(\d+), cameraSamples=(\d+), actualSamples=(\d+)",log)
        expected_camera=args.camera_samples if args.camera_samples is not None else args.samples
        if len(selections)!=args.frames*(args.viewport_iterations+args.viewport_settle_iterations) or any(
                tuple(map(int,row))!=(args.samples,expected_camera,args.samples) for row in selections):
            raise RuntimeError("Viewport did not honor its sample control; inspect the sample-selection log")
        recordings=re.findall(r"OptiX sample selection: mode=record, viewportSamples=(\d+), cameraSamples=(\d+), actualSamples=(\d+)",log)
        if len(recordings)!=(0 if args.interactive_only else args.frames) or any(
                tuple(map(int,row))!=(args.samples,expected_camera,expected_camera) for row in recordings):
            raise RuntimeError("Recording did not retain the physical-camera sample setting")
        expected_resolution=args.pixels
        viewport_rows=re.findall(r"OptiX viewport: (\d+x\d+),",log)
        if len(viewport_rows)!=len(selections) or any(size!=expected_resolution for size in viewport_rows):
            raise RuntimeError("Viewport benchmark did not run at the requested resolution")
        if args.viewport_settle_iterations and not args.ordinary:
            statuses=re.findall(r"CLAS update: .*?settled=(\d+), pending=(\d+)",log)
            if not statuses or statuses[-1]!=("1","0"):
                raise RuntimeError("CLAS streaming did not settle after camera movement stopped")
    if not args.ordinary:
        if "[CLAS] version=" not in log:
            raise RuntimeError("No cluster update was exercised")
        reloads = 0
        if args.viewport_script:
            script = json.loads(args.viewport_script.read_text())
            reloads = args.frames * script.get("cycles",1) * sum(step.get("reload",False) for step in script["steps"])
            reloads -= int(args.interactive_only and script["steps"][0].get("reload",False))
        if log.count("[CLAS] GPU resident-cut pipeline") != 1+reloads:
            raise RuntimeError("Cluster residency was reinitialized between graph frames")
        for frame in caches[0].glob("[0-9]*"):
            payload = frame / "normalObj.zencache"
            # The optional tiny ordinary reflector intentionally uses the cache;
            # streamed Zorah geometry must still remain outside it.
            limit=65536+(2*args.mirror_obj.stat().st_size if args.mirror_obj else 0)
            if payload.exists() and payload.stat().st_size > limit:
                raise RuntimeError(f"Geometry unexpectedly entered the descriptor cache: {payload}")
    if args.history_regression:
        keys = ["sceneChanged", "cutChanged", "sceneVersion", "cutVersion", "keep", "bounded", "sampleStart", "historyStart", "historySamples", "recovery", "settled", "sharcClears"]
        def history_rows(part):
            return [dict(zip(keys, map(int, row))) for row in re.findall(
                r"CLAS history: " + ", ".join(key + r"=(\d+)" for key in keys), part)]
        benchmark = log.split("[viewport benchmark] motion", 1)[1]
        all_rows = history_rows(benchmark)
        if len(all_rows) != args.viewport_iterations + args.viewport_settle_iterations:
            raise RuntimeError("Incomplete history diagnostics")
        if any(row["keep"] != int(args.keep_clas_history) for row in all_rows):
            raise RuntimeError("Viewport did not apply the requested history policy")
        if args.disable_sharc and any(row["sharcClears"] for row in all_rows):
            raise RuntimeError("Viewport did not disable SHARC")
        recording = history_rows(log.split("[viewport benchmark] motion", 1)[0])
        if not recording or any(row["keep"] or row["bounded"] for row in recording):
            raise RuntimeError("Offline recording did not use strict accumulation")
        if args.history_regression == "maintenance":
            changes = [row for row in all_rows if row["sceneChanged"]]
            if len(changes) < 3 or any(row["cutChanged"] for row in all_rows) or len({row["cutVersion"] for row in all_rows}) != 1:
                raise RuntimeError("AS-only compaction was not exercised with a stable cut")
            held = all_rows
        else:
            held = history_rows(benchmark.split("[viewport benchmark] hold", 1)[1])
            changes = [row for row in held if row["cutChanged"]]
            if not changes:
                raise RuntimeError("No late LOD refinement occurred while holding the camera")
        if args.keep_clas_history or args.history_regression == "maintenance":
            sequence = [row["sampleStart"] for row in held]
            if any(b != a + args.samples for a,b in zip(sequence,sequence[1:])):
                raise RuntimeError("Geometry publication restarted the sample sequence")
            if not args.disable_sharc and len({row["sharcClears"] for row in held}) != 1:
                raise RuntimeError("Geometry publication cleared SHARC history")
        elif not all(row["sampleStart"] == 0 for row in changes):
            raise RuntimeError("Strict mode did not restart history after a real geometry change")
        if args.keep_clas_history and args.history_regression == "refinement":
            bounded = [row for row in held if row["bounded"]]
            if not bounded or any(row["historySamples"] > args.history_limit + args.samples - 1 for row in bounded):
                raise RuntimeError("Refinement history did not respect its effective sample bound")
            if not held[-1]["settled"] or held[-1]["bounded"] or held[-1]["historySamples"] <= args.history_limit:
                raise RuntimeError("Settled view did not resume progressive averaging")
        if args.history_regression == "maintenance" and any(row["bounded"] for row in held):
            raise RuntimeError("AS-only compaction unnecessarily limited history")
        if args.sample_sequence_start is not None and max(row["sampleStart"] for row in all_rows) <= 65535:
            raise RuntimeError("The widened sample sequence did not cross the old 16-bit boundary")
        (output / "history-validation.json").write_text(json.dumps({"mode":args.history_regression,
            "keep_history":args.keep_clas_history,"sharc_enabled":not args.disable_sharc,"history_limit":args.history_limit,
            "late_or_maintenance_publications":len(changes),"hold":held,"checks_passed":True},indent=2))
    if args.performance and args.viewport_iterations:
        report = output / "performance.json"
        report.write_text(json.dumps(summarize_viewport(log, args), indent=2))
        print(f"Viewport timing report: {report}")
    print(f"Render child passed; inspect the images in {output / 'P'}")
    print(f"Generated graph: {graph_path}")
    for line in log.splitlines():
        if line.startswith("[CLAS]") or "ERROR SUMMARY" in line:
            print(line)


if __name__ == "__main__":
    main()
