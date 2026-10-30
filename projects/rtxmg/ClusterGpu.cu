#include "ClusterGpu.h"
#include <cstdio>
#include <cstdlib>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <stdexcept>

namespace zeno::cluster::gpu {
static void check(cudaError_t r) {
    if (r)
        throw std::runtime_error(cudaGetErrorString(r));
}
__device__ float metric(const Group &g, const Placement &p, const Traversal &t) {
    const float *m = p.transform;
    const float *b = g.bounds;
    float x = m[0] * b[0] + m[1] * b[1] + m[2] * b[2] + m[3] - t.eye[0];
    float y = m[4] * b[0] + m[5] * b[1] + m[6] * b[2] + m[7] - t.eye[1];
    float z = m[8] * b[0] + m[9] * b[1] + m[10] * b[2] + m[11] - t.eye[2];
    const float distance = sqrtf(x * x + y * y + z * z), radius = b[3] * p.scale;
    return b[4] * p.scale / fmaxf(distance - radius, .01f) *
           projectionForSphere(t.zoom, x, y, z, radius, distance, t.projection);
}
// One warp per resident group/placement. A resident group can only be admitted
// after all its parents, and parents cannot be retired while children remain.
// Thus the metric and its reverse select a complete cut even with missing pages.
__global__ void initializeWork(Traversal t) {
    const uint32_t p = blockIdx.x;
    const Geometry geo = t.geometries[t.placements[p].geometry];
    for (uint32_t local = threadIdx.x; local < geo.residentCount; local += blockDim.x) {
        const uint32_t resident = geo.residentOffset + local;
        Work &work = t.work[t.ranges[p].offset + local];
        work.group = t.residents[resident];
        work.resident = resident;
        work.placement = p;
    }
}
__global__ void countWork(Traversal t) {
    const uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (w >= t.workCount)
        return;
    const uint32_t residentIndex = t.work[w].resident, id = t.work[w].group, p = t.work[w].placement;
    const Group group = t.groups[id];
    const Placement placement = t.placements[p];
    const bool traversed =
        __shfl_sync(~0u, lane == 0 && (group.root || metric(group, placement, t) > t.error), 0) != 0;
    if (lane == 0) {
        if (traversed)
            atomicOr(t.wanted + residentIndex, 1u);
    }
    uint32_t total = 0;
    for (uint32_t part = 0; part < (group.count + 31) / 32; ++part) {
        uint32_t c = part * 32 + lane;
        bool selected = false;
        if (traversed && c < group.count) {
            const int32_t child = t.refined[group.refinedOffset + c];
            if (child < 0)
                selected = true;
            else {
                const float error = metric(t.groups[child], placement, t);
                const bool refine = error > t.error;
                const bool available = t.groupSlots[child] != Invalid;
                selected = !refine || !available;
                const uint32_t peers = __match_any_sync(__activemask(), child);
                if (refine && !available && lane == uint32_t(__ffs(peers) - 1) &&
                    atomicExch(t.stamps + child, t.stamp) != t.stamp) {
                    uint32_t offset = atomicAdd(&t.result->requests, 1u);
                    if (offset < MaxRequests)
                        t.requests[offset] = {uint32_t(child), error};
                    else
                        atomicOr(&t.result->overflow, 2u);
                }
            }
        }
        const uint32_t mask = __ballot_sync(~0u, selected);
        if (lane == 0)
            t.work[w].mask[part] = mask;
        total += __popc(mask);
    }
    if (lane == 0) {
        t.counts[w] = total;
        if (total)
            atomicOr(t.wanted + residentIndex, 2u);
    }
}
__global__ void scatterWork(Traversal t) {
    uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (w >= t.workCount)
        return;
    const Work work = t.work[w];
    const auto group = t.groups[work.group];
    uint32_t offset = t.offsets[w];
    for (uint32_t part = 0; part < (group.count + 31) / 32; ++part) {
        const uint32_t mask = work.mask[part], c = part * 32 + lane;
        if (mask & (1u << lane)) {
            uint32_t out = offset + __popc(mask & ((1u << lane) - 1));
            if (out >= t.capacity)
                atomicOr(&t.result->overflow, 1u);
            else {
                uint32_t slot = t.groupSlots[work.group] + c;
                t.slots[out] = slot;
                t.renderHandles[out] = t.handles[slot];
            }
        }
        offset += __popc(mask);
    }
}
__global__ void placementRanges(Traversal t) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= t.placementCount)
        return;
    const Geometry g = t.geometries[t.placements[p].geometry];
    // placementOrder is sorted by geometry, and each placement stores its local
    // index in the trailing range table initialized by the host.
    const uint32_t first = t.ranges[p].offset;
    const uint32_t begin = t.offsets[first], end = t.offsets[first + g.residentCount];
    t.ranges[p] = {begin, end - begin};
    if (begin == end)
        atomicAdd(&t.result->emptyPlacements, 1u);
    atomicMax(&t.result->maxPerGas, end - begin);
    if (!p)
        t.result->clusters = t.offsets[t.workCount];
}
__global__ void compareCuts(Traversal t) {
    uint32_t p = blockIdx.x;
    if (p >= t.placementCount)
        return;
    if (t.result->overflow & 1) {
        if (!threadIdx.x)
            t.same[p] = 0;
        return;
    }
    Range r = t.ranges[p];
    if (!t.previousRanges) {
        if (!threadIdx.x) {
            atomicExch(&t.result->changed, 1u);
            atomicExch(&t.result->cutChanged, 1u);
            t.same[p] = 0;
        }
        return;
    }
    Range old = t.previousRanges[p];
    if (r.count != old.count) {
        if (!threadIdx.x) {
            atomicExch(&t.result->changed, 1u);
            atomicExch(&t.result->cutChanged, 1u);
            t.same[p] = 0;
        }
        return;
    }
    if (r.offset + r.count > t.capacity) {
        if (!threadIdx.x)
            t.same[p] = 0;
        return;
    }
    bool different = false;
    for (uint32_t i = threadIdx.x; i < r.count; i += blockDim.x)
        if (t.slots[r.offset + i] != t.previousSlots[old.offset + i]) {
            different = true;
            break;
        }
    different = __syncthreads_or(different) != 0;
    if (!threadIdx.x) {
        // Bank compaction rebuilds GAS ownership without changing geometry.
        // Still compare the exact cut so image history can survive that swap.
        t.same[p] = !different && !t.rebuildAll;
        if (different) {
            atomicExch(&t.result->cutChanged, 1u);
            atomicExch(&t.result->changed, 1u);
        } else if (t.rebuildAll)
            atomicExch(&t.result->changed, 1u);
    }
}
size_t scanBytes(uint32_t n) {
    size_t bytes = 0;
    check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, (uint32_t *)nullptr, (uint32_t *)nullptr, n));
    return bytes;
}
size_t sortBytes(uint32_t n) {
    size_t bytes = 0;
    check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, (uint64_t *)nullptr, (uint64_t *)nullptr,
                                          (uint32_t *)nullptr, (uint32_t *)nullptr, n));
    return bytes;
}
void traverse(const Traversal &t, void *scratch, size_t bytes, cudaStream_t stream) {
    const auto stage = [&](const char *name) {
        if (std::getenv("ZENO_CLAS_SYNC_STAGES")) {
            std::fprintf(stderr, "[GPU stage] %s\n", name);
            check(cudaStreamSynchronize(stream));
        }
    };
    check(cudaMemsetAsync(t.result, 0, sizeof(Result), stream));
    check(cudaMemsetAsync(&t.result->minLod, 0xff, sizeof(uint32_t), stream));
    initializeWork<<<t.placementCount, 128, 0, stream>>>(t);
    countWork<<<(t.workCount * 32 + 127) / 128, 128, 0, stream>>>(t);
    stage("count");
    check(cudaMemsetAsync(t.counts + t.workCount, 0, sizeof(uint32_t), stream));
    check(cub::DeviceScan::ExclusiveSum(scratch, bytes, t.counts, t.offsets, t.workCount + 1, stream));
    stage("scan");
    scatterWork<<<(t.workCount * 32 + 127) / 128, 128, 0, stream>>>(t);
    stage("scatter");
    placementRanges<<<(t.placementCount + 127) / 128, 128, 0, stream>>>(t);
    stage("ranges");
    compareCuts<<<t.placementCount, 128, 0, stream>>>(t);
    check(cudaGetLastError());
    stage("compare");
}
__global__ void patchDescriptors(const DescriptorPatch *p, uint32_t n, zeno_cluster_device::Cluster *d) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d[p[i].slot] = p[i].cluster;
}
__global__ void patchGroups(const GroupPatch *p, uint32_t n, uint32_t *d) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d[p[i].group] = p[i].slot;
}
void patch(const DescriptorPatch *p, uint32_t n, zeno_cluster_device::Cluster *d, const GroupPatch *g,
           uint32_t m, uint32_t *slots, cudaStream_t s) {
    if (n)
        patchDescriptors<<<(n + 127) / 128, 128, 0, s>>>(p, n, d);
    if (m)
        patchGroups<<<(m + 127) / 128, 128, 0, s>>>(g, m, slots);
    check(cudaGetLastError());
}
__global__ void pageDestinations(PageBuild *pages, uint32_t n, const uint32_t *sizes, uint64_t *dest,
                                 Result *result) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    const auto page = pages[p];
    uint64_t offset = 0;
    for (uint32_t i = 0; i < page.count; ++i) {
        offset = (offset + 127) & ~127ull;
        if (page.output)
            dest[page.first + i] = page.output + offset;
        offset += sizes[page.first + i];
    }
    if (page.output && offset > page.bytes)
        atomicOr(&result->overflow, 4u);
    pages[p].bytes = (offset + 127) & ~127ull;
}
void destinations(PageBuild *p, uint32_t n, const uint32_t *sizes, uint64_t *dest, Result *result,
                  cudaStream_t s) {
    pageDestinations<<<(n + 127) / 128, 128, 0, s>>>(p, n, sizes, dest, result);
    check(cudaGetLastError());
}
__global__ void copyPageHandles(const PageBuild *pages, const uint64_t *handles, uint64_t *slots) {
    const auto p = pages[blockIdx.x];
    for (uint32_t i = threadIdx.x; i < p.count; i += blockDim.x)
        slots[p.slot + i] = handles[p.first + i];
}
void scatterHandles(const PageBuild *p, uint32_t n, const uint64_t *handles, uint64_t *slots,
                    cudaStream_t s) {
    copyPageHandles<<<n, 128, 0, s>>>(p, handles, slots);
    check(cudaGetLastError());
}
__global__ void guardedCount(const Result *result, uint32_t *count, uint32_t n) {
    *count = result->overflow ? 0 : n;
}
void guardCount(const Result *result, uint32_t *count, uint32_t n, cudaStream_t s) {
    guardedCount<<<1, 1, 0, s>>>(result, count, n);
    check(cudaGetLastError());
}
__device__ uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
__global__ void hashCuts(Traversal t, uint64_t *hashes, uint32_t *ids) {
    uint32_t p = blockIdx.x, lane = threadIdx.x % 32;
    Range r = t.ranges[p];
    uint64_t hash = 0;
    if (t.result->overflow & 1) {
        if (!threadIdx.x) {
            hashes[p] = 0;
            ids[p] = p;
        }
        return;
    }
    unsigned long long triangles = 0;
    uint32_t minLod = Invalid, maxLod = 0;
    for (uint32_t i = threadIdx.x; i < r.count; i += blockDim.x) {
        const uint32_t slot = t.slots[r.offset + i];
        hash += mix(slot);
        const auto cluster = t.descriptors[slot];
        triangles += cluster.triangles;
        minLod = min(minLod, cluster.lod);
        maxLod = max(maxLod, cluster.lod);
    }
    for (uint32_t delta = 16; delta; delta /= 2) {
        hash += __shfl_down_sync(~0u, hash, delta);
        triangles += __shfl_down_sync(~0u, triangles, delta);
        minLod = min(minLod, __shfl_down_sync(~0u, minLod, delta));
        maxLod = max(maxLod, __shfl_down_sync(~0u, maxLod, delta));
    }
    __shared__ uint64_t partial[4];
    __shared__ unsigned long long trianglePartial[4];
    __shared__ uint32_t minPartial[4], maxPartial[4];
    if (!lane) {
        partial[threadIdx.x / 32] = hash;
        trianglePartial[threadIdx.x / 32] = triangles;
        minPartial[threadIdx.x / 32] = minLod;
        maxPartial[threadIdx.x / 32] = maxLod;
    }
    __syncthreads();
    if (!threadIdx.x) {
        hash = partial[0] + partial[1] + partial[2] + partial[3];
        hashes[p] =
            t.result->changed ? mix(hash ^ mix(uint64_t(t.placements[p].geometry) << 32 | r.count)) : 0;
        ids[p] = p;
        // Reduce diagnostics per placement rather than contending on three
        // global atomics from every resident group/placement warp.
        atomicAdd(&t.result->triangles,
                  trianglePartial[0] + trianglePartial[1] + trianglePartial[2] + trianglePartial[3]);
        atomicMin(&t.result->minLod,
                  min(minPartial[0], min(minPartial[1], min(minPartial[2], minPartial[3]))));
        atomicMax(&t.result->maxLod,
                  max(maxPartial[0], max(maxPartial[1], max(maxPartial[2], maxPartial[3]))));
    }
}
__global__ void findCanonical(Traversal t, const uint64_t *hashes, const uint32_t *ids, uint32_t *canonical) {
    const uint32_t sorted = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x % 32;
    if (sorted >= t.placementCount)
        return;
    uint32_t p = ids[sorted], low = 0, high = sorted;
    const uint64_t hash = hashes[sorted];
    if ((t.result->overflow & 1) || !t.result->changed) {
        if (!lane) {
            canonical[p] = p;
            t.counts[p] = 0;
        }
        return;
    }
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (hashes[mid] < hash)
            low = mid + 1;
        else
            high = mid;
    }
    const Range r = t.ranges[p];
    uint32_t result = p;
    // Hashes only find candidates. Exact ordered slot equality determines
    // sharing, including the collision case. Never alias merely on a hash.
    for (uint32_t j = low; j < sorted; ++j) {
        uint32_t c = ids[j];
        Range other = t.ranges[c];
        if (t.placements[c].geometry != t.placements[p].geometry || other.count != r.count)
            continue;
        bool equal = true;
        for (uint32_t i = lane; i < r.count; i += 32)
            if (t.slots[r.offset + i] != t.slots[other.offset + i]) {
                equal = false;
                break;
            }
        if (__all_sync(~0u, equal)) {
            result = c;
            break;
        }
    }
    if (lane)
        return;
    canonical[p] = result;
    uint32_t unique = result == p;
    t.counts[p] = unique;
    if (unique) {
        atomicAdd(&t.result->uniqueGases, 1u);
        atomicAdd(&t.result->uniqueReferences, r.count);
    }
}
__global__ void reuseCuts(Traversal t, const uint32_t *canonical, uint32_t *reuse) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < t.placementCount && t.previousTokens && t.same[p] && !t.rebuildAll)
        atomicMin(reuse + canonical[p], t.previousTokens[p]);
}
__global__ void newGasFlags(Traversal t, const uint32_t *canonical, const uint32_t *reuse) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= t.placementCount)
        return;
    uint32_t build = canonical[p] == p && reuse[p] == Invalid && !(t.result->overflow & 1);
    t.counts[p] = build;
    if (build) {
        atomicAdd(&t.result->newGases, 1u);
        atomicAdd(&t.result->newReferences, t.ranges[p].count);
        atomicMax(&t.result->newMaxPerGas, t.ranges[p].count);
    }
}
void canonicalize(const Traversal &t, uint64_t *keys, uint64_t *sortedKeys, uint32_t *ids,
                  uint32_t *sortedIDs, uint32_t *canonical, uint32_t *reuse, void *scratch, size_t bytes,
                  cudaStream_t s) {
    hashCuts<<<t.placementCount, 128, 0, s>>>(t, keys, ids);
    check(cub::DeviceRadixSort::SortPairs(scratch, bytes, keys, sortedKeys, ids, sortedIDs, t.placementCount,
                                          0, 64, s));
    findCanonical<<<(t.placementCount * 32 + 127) / 128, 128, 0, s>>>(t, sortedKeys, sortedIDs, canonical);
    check(cudaMemsetAsync(reuse, 0xff, t.placementCount * sizeof(uint32_t), s));
    reuseCuts<<<(t.placementCount + 127) / 128, 128, 0, s>>>(t, canonical, reuse);
    newGasFlags<<<(t.placementCount + 127) / 128, 128, 0, s>>>(t, canonical, reuse);
    check(cudaMemsetAsync(t.counts + t.placementCount, 0, 4, s));
    check(cub::DeviceScan::ExclusiveSum(scratch, bytes, t.counts, t.offsets, t.placementCount + 1, s));
    check(cudaGetLastError());
}
__global__ void fillGasArgs(const Range *ranges, uint32_t n, const uint32_t *canonical,
                            const uint32_t *offsets, const uint32_t *reuse, CUdeviceptr handles,
                            OptixClusterAccelBuildInputClustersArgs *args) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n && canonical[p] == p && reuse[p] == Invalid) {
        OptixClusterAccelBuildInputClustersArgs a{};
        a.clusterHandlesCount = ranges[p].count;
        a.clusterHandlesBuffer = handles + uint64_t(ranges[p].offset) * sizeof(uint64_t);
        args[offsets[p]] = a;
    }
}
void gasArgs(const Range *r, uint32_t n, const uint32_t *c, const uint32_t *o, const uint32_t *reuse,
             CUdeviceptr h, OptixClusterAccelBuildInputClustersArgs *a, cudaStream_t s) {
    fillGasArgs<<<(n + 127) / 128, 128, 0, s>>>(r, n, c, o, reuse, h, a);
    check(cudaGetLastError());
}
__global__ void fillInstances(const Placement *placements, uint32_t n, const uint64_t *handles,
                              const uint32_t *tokens, OptixInstance *out) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) {
        OptixInstance i{};
        for (uint32_t k = 0; k < 12; ++k)
            i.transform[k] = placements[p].transform[k];
        i.instanceId = p;
        i.visibilityMask = 255;
        i.traversableHandle = handles[tokens[p]];
        out[p] = i;
    }
}
void instances(const Placement *p, uint32_t n, const uint64_t *h, const uint32_t *tokens, OptixInstance *out,
               cudaStream_t s) {
    fillInstances<<<(n + 127) / 128, 128, 0, s>>>(p, n, h, tokens, out);
    check(cudaGetLastError());
}
__global__ void scatterGasHandles(uint32_t count, const uint64_t *handles, const uint32_t *tokens,
                                  uint64_t *table) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        table[tokens[i]] = handles[i];
}
void publishGasHandles(uint32_t n, const uint64_t *h, const uint32_t *tokens, uint64_t *table,
                       cudaStream_t s) {
    if (n)
        scatterGasHandles<<<(n + 127) / 128, 128, 0, s>>>(n, h, tokens, table);
    check(cudaGetLastError());
}
} // namespace zeno::cluster::gpu
