#pragma once
namespace zeno_cluster_device {
// Positions are build-only and live in staging until the CLAS build completes.
// Hit positions come from optixGetTriangleVertexData; only shading normals
// need persistent vertex storage. The on-disk Vertex layout is unchanged.
struct VertexNormal {
    float normal[3];
};
struct ShadingVertex {
    float uv[2];
    // Low 10 bits: signed tangent-frame angle; bit 30: UV; bit 31: tangent.
    unsigned int tangent;
};
// Shared host/NVRTC layout; keep ClusterSmoke's standalone shader in sync.
struct Cluster {
    const VertexNormal *normals;
    const unsigned char *indices;
    unsigned int lod, geometry, group, triangles;
    const ShadingVertex* attributes;
    const unsigned short* materials; // Source primitive slot, per triangle.
};
enum DebugView {
    Shaded = 0, Lod = 1, Clusters = 2, FlatGray = 3,
    GeometricNormal = 4, ShadingNormal = 5, GeometricFacing = 6, NormalAgreement = 7
};
} // namespace zeno_cluster_device
