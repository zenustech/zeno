"""Compare native GPU instance TRS import against explicit ordinary nodes.

Uses the supplied tiny Zorah cube as a read-only geometry source, including
rotation, nonuniform scale, mirroring and a transformed parent.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baker", type=Path, default=Path(__file__).resolve().parents[2] / "out/build/x64-Release/bin/zeno_cluster_bake.exe")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_text())
    for buffer in source["buffers"]:
        if "uri" in buffer:
            shutil.copyfile(args.source.parent / buffer["uri"], root / buffer["uri"])
    translations = [[(i % 4 - 1.5) * 2.5, 0., (i // 4 - .5) * 3.] for i in range(8)]
    rotations = [[0., math.sin(i * .17), 0., math.cos(i * .17)] for i in range(8)]
    scales = [[-1. if i == 4 else 1., .7 + i * .1, 1.3] for i in range(8)]
    parent = {"translation": [1., 0., -1.], "rotation": [0., math.sin(.15), 0., math.cos(.15)], "scale": [1.1, .9, 1.2]}
    ordinary = copy.deepcopy(source)
    ordinary["nodes"] = [{**parent, "children": list(range(1, 9))}] + [
        {"mesh": 0, "translation": t, "rotation": r, "scale": s} for t, r, s in zip(translations, rotations, scales)
    ]
    ordinary["scenes"] = [{"nodes": [0]}]
    native = copy.deepcopy(source)
    data = bytearray()
    attributes = {}
    for name, values, width in [("TRANSLATION", translations, 3), ("ROTATION", rotations, 4), ("SCALE", scales, 3)]:
        offset = len(data)
        for value in values:
            data.extend(struct.pack("<" + "f" * width, *value))
        view = len(native["bufferViews"])
        native["bufferViews"].append({"buffer": len(native["buffers"]), "byteOffset": offset, "byteLength": len(data) - offset})
        attributes[name] = len(native["accessors"])
        native["accessors"].append({"bufferView": view, "componentType": 5126, "count": 8, "type": f"VEC{width}"})
    (root / "instances.bin").write_bytes(data)
    native["buffers"].append({"uri": "instances.bin", "byteLength": len(data)})
    native["nodes"] = [{**parent, "mesh": 0, "extensions": {"EXT_mesh_gpu_instancing": {"attributes": attributes}}}]
    native["scenes"] = [{"nodes": [0]}]
    native.setdefault("extensionsUsed", []).append("EXT_mesh_gpu_instancing")
    for name, scene in [("ordinary", ordinary), ("native", native)]:
        scene_path = root / f"{name}.gltf"
        scene_path.write_text(json.dumps(scene))
        subprocess.run([str(args.baker.resolve()), "--source", str(scene_path), "--output", str(root / name), "--include-instances", "1"], check=True)
    reference = json.loads((root / "ordinary/asset.json").read_text())
    result = json.loads((root / "native/asset.json").read_text())
    assert len(reference["nodes"]) == len(result["nodes"]) == 8
    maximum = max(abs(a-b) for expected, actual in zip(reference["nodes"], result["nodes"]) for a, b in zip(expected["matrix"], actual["matrix"]))
    assert maximum < 1e-5, maximum
    print(f"Eight instance transforms match explicit nodes; maximum matrix difference {maximum:g}")
    print(f"Native test manifest: {root / 'native/asset.json'}")


if __name__ == "__main__":
    main()
