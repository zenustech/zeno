#pragma once
#include "ClusterDevice.h"
#include "ClusterLod.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <optix.h>

namespace zeno::cluster::gpu {
constexpr uint32_t Invalid = ~0u, MaxRequests = 16384;
struct Group {
    float bounds[5];
    uint32_t refinedOffset, count, geometry, root, depth;
};
struct Geometry {
    uint32_t groupOffset, groupCount, residentOffset, residentCount;
    uint32_t placementOffset, placementCount, workOffset, workCount;
};
struct Placement {
    float transform[12], scale;
    uint32_t geometry;
};
struct Work {
    uint32_t placement, group, resident, mask[8];
};
struct Range {
    uint32_t offset, count;
};
struct Request {
    uint32_t group;
    float priority;
};
struct Result {
    uint32_t requests, overflow, changed, clusters;
    unsigned long long triangles;
    uint32_t minLod, maxLod, emptyPlacements, maxPerGas;
    uint32_t uniqueGases, uniqueReferences, cutChanged, padding;
    uint32_t newGases, newReferences, newMaxPerGas, padding2;
};
struct DescriptorPatch {
    uint32_t slot;
    zeno_cluster_device::Cluster cluster;
};
struct GroupPatch {
    uint32_t group, slot;
};
struct PageBuild {
    uint32_t first, count;
    CUdeviceptr output;
    uint64_t bytes;
    uint32_t slot;
};
struct Traversal {
    const Group *groups;
    const int32_t *refined;
    const Geometry *geometries;
    uint32_t geometryCount;
    const Placement *placements;
    const uint32_t *placementOrder;
    const uint32_t *residents;
    const uint32_t *groupSlots;
    const uint64_t *handles;
    const zeno_cluster_device::Cluster *descriptors;
    uint32_t *stamps;
    uint32_t stamp;
    Work *work;
    uint32_t workCount;
    uint32_t *counts;
    uint32_t *offsets;
    uint32_t *slots;
    uint64_t *renderHandles;
    uint32_t capacity;
    Range *ranges;
    uint32_t placementCount;
    const Range *previousRanges;
    const uint32_t *previousSlots;
    const uint32_t *previousTokens;
    uint32_t *same;
    bool rebuildAll;
    uint32_t *wanted;
    Request *requests;
    Result *result;
    float eye[3], projection, error;
    ZoomView zoom;
};
size_t scanBytes(uint32_t count);
size_t sortBytes(uint32_t count);
void traverse(const Traversal &, void *scanScratch, size_t bytes, cudaStream_t stream);
void canonicalize(const Traversal &, uint64_t *, uint64_t *, uint32_t *, uint32_t *, uint32_t *, uint32_t *,
                  void *, size_t, cudaStream_t);
void patch(const DescriptorPatch *, uint32_t, zeno_cluster_device::Cluster *, const GroupPatch *, uint32_t,
           uint32_t *, cudaStream_t);
void destinations(PageBuild *, uint32_t, const uint32_t *, uint64_t *, Result *, cudaStream_t);
void scatterHandles(const PageBuild *, uint32_t, const uint64_t *, uint64_t *, cudaStream_t);
void guardCount(const Result *, uint32_t *, uint32_t, cudaStream_t);
void gasArgs(const Range *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, CUdeviceptr,
             OptixClusterAccelBuildInputClustersArgs *, cudaStream_t);
void instances(const Placement *, uint32_t, const uint64_t *, const uint32_t *, OptixInstance *,
               cudaStream_t);
void publishGasHandles(uint32_t, const uint64_t *, const uint32_t *, uint64_t *, cudaStream_t);
} // namespace zeno::cluster::gpu
