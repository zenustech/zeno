#pragma once
namespace zeno_cluster_device {
struct Vertex {
    float position[3], normal[3];
};
// Shared host/NVRTC layout; keep ClusterSmoke's standalone shader in sync.
struct Cluster {
    const Vertex *vertices;
    const unsigned char *indices;
    unsigned int lod, geometry, group, triangles;
};
enum DebugView {
    Shaded = 0, Lod = 1, Clusters = 2, FlatGray = 3,
    GeometricNormal = 4, ShadingNormal = 5, GeometricFacing = 6, NormalAgreement = 7
};
} // namespace zeno_cluster_device
