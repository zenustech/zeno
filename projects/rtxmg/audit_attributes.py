"""Read-only native attribute readiness audit; never modifies source caches.

Checks every selected cluster header and samples decoded UVs across all LODs.
This audits available native data, not GPU interpolation or material shading.
"""
import argparse
import collections
import json
import math
import mmap
from pathlib import Path
import re
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-depth", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.samples_per_depth <= 32:
        parser.error("samples-per-depth must be in [1,32]")
    asset = json.loads(args.manifest.read_text())
    source_path = Path(asset["source"])
    if not source_path.is_absolute(): source_path = args.manifest.parent / source_path
    source = json.loads(source_path.read_text())
    reports = []
    for geometry, mesh in enumerate(asset["meshes"]):
        path = Path(mesh["metadata"])
        if not path.is_absolute(): path = args.manifest.parent/path
        meta = json.loads(path.read_text())
        if meta.get("format") != "rtxmg.shard.v19":
            raise ValueError("This audit requires native v19 shard metadata")
        by_depth = collections.defaultdict(list)
        for group, entry in enumerate(meta["groups"]): by_depth[entry["depth"]].append(group)
        sampled = set()
        for groups in by_depth.values():
            for i in range(args.samples_per_depth):
                sampled.add(groups[(len(groups)-1)*i//max(args.samples_per_depth-1,1)])
        counts = collections.Counter()
        uv_bounds = [[math.inf,math.inf,-math.inf,-math.inf] for _ in range(2)]
        material_ids = set()
        slots = len(source["meshes"][mesh["sourceMesh"]]["primitives"])
        payload = Path(meta["pages"])
        if not payload.is_absolute(): payload = path.parent/payload
        with payload.open("rb") as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
            for group_id, group in enumerate(meta["groups"]):
                start, size = group["offset"], group["bytes"]
                if start < 0 or start+size > len(data): raise ValueError("Group outside shard")
                group_data = memoryview(data)[start:start+size]
                count = struct.unpack_from("<I",group_data,8)[0] >> 16
                if count != len(group["refined"]): raise ValueError("Cluster count mismatch")
                for i in range(count):
                    c = 32 + 16*i
                    shape, attrs, vertex, index = struct.unpack_from("<4I",group_data,c)
                    tri, vertices = (shape&255)+1, ((shape>>8)&255)+1
                    flags, material, encoding = attrs&255,(attrs>>8)&255,attrs>>24
                    vp, ip = c+vertex,c+index
                    if tri>128 or vertices>128 or ip+tri*3>size: raise ValueError("Invalid cluster range")
                    counts["clusters"] += 1
                    for bit,name in [(1,"normals"),(2,"tangents"),(4,"uv0"),(8,"uv1")]:
                        if flags&bit: counts[name+"_clusters"] += 1
                    if material == 255:
                        if ip+tri*4>size: raise ValueError("Material bytes outside group")
                        ids = {group_data[ip+tri*3+t]&63 for t in range(tri)}
                        counts["mixed_material_clusters"] += 1
                    else: ids = {material}
                    if any(value>=slots for value in ids): raise ValueError("Material slot outside source primitive range")
                    material_ids.update(ids)
                    if group_id not in sampled: continue
                    counts["sampled_clusters"] += 1
                    offset = vp + vertices*12 + (vertices*4 if flags&1 else 0)
                    for channel in range(2):
                        if not flags&(4<<channel): continue
                        if flags&(32<<channel): raise ValueError("Compressed UV format requires a new audit decoder")
                        offset = (offset+7)&~7
                        quantized = bool(encoding&(1<<channel))
                        bytes_needed = 16+vertices*4 if quantized else vertices*8
                        if offset+bytes_needed > size: raise ValueError("UV array outside group")
                        if quantized:
                            base_u,base_v,step_u,step_v = struct.unpack_from("<4f",group_data,offset)
                            if not all(math.isfinite(x) for x in (base_u,base_v,step_u,step_v)) or min(step_u,step_v)<=0:
                                raise ValueError("Invalid quantized UV header")
                            values = [(base_u+(x&65535)*step_u,base_v+(x>>16)*step_v)
                                      for (x,) in struct.iter_unpack("<I",group_data[offset+16:offset+bytes_needed])]
                            counts[f"quantized_uv{channel}_sampled_clusters"] += 1
                        else:
                            values = struct.iter_unpack("<2f",group_data[offset:offset+bytes_needed])
                        bound = uv_bounds[channel]
                        for u,v in values:
                            if not math.isfinite(u) or not math.isfinite(v): raise ValueError("Non-finite native UV")
                            bound[:] = [min(bound[0],u),min(bound[1],v),max(bound[2],u),max(bound[3],v)]
                            counts[f"uv{channel}_sampled_vertices"] += 1
                        offset += bytes_needed
                group_data.release()
        reports.append({"source_mesh":mesh["sourceMesh"],"groups":len(meta["groups"]),
            "sampled_depths":sorted(by_depth),"counts":dict(counts),"material_slots":slots,
            "observed_local_material_ids":sorted(material_ids),
            "uv_bounds":[b if all(math.isfinite(x) for x in b) else None for b in uv_bounds]})
    vertex_source = (Path(__file__).parent/"ClusterDevice.h").read_text()
    vertex = re.search(r"struct Vertex\s*\{([^}]+)\}",vertex_source).group(1)
    fields = re.findall(r"([A-Za-z_]\w*)\s*\[",vertex)
    report = {"native_payload_audit":"passed","scope":"All selected cluster headers; sampled finite UV decode per LOD depth; material slot range checks",
        "meshes":reports,"gpu_vertex_fields":fields,"gpu_uv_tangent_material_validation":"not available in current gray runtime",
        "material_readiness":False,"remaining":"GPU attribute transport, barycentric interpolation, tangent handedness and per-instance material binding tests"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(f"Native attribute audit passed for {len(reports)} meshes; GPU attribute validation is not implemented")


if __name__ == "__main__": main()
