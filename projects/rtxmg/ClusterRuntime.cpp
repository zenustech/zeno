#include "ClusterRuntime.h"
#include "ClusterAdaptiveLod.h"
#include "ClusterGpu.h"
#include "ClusterProfiling.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <optix_stubs.h>
#include <set>

namespace zeno::cluster {
std::recursive_mutex &renderSubmissionMutex() {
    static std::recursive_mutex mutex;
    return mutex;
}
namespace {
using Clock = std::chrono::steady_clock;
thread_local RuntimeStats *renderStats = nullptr;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct StatsScope {
    RuntimeStats *previous;
    explicit StatsScope(RuntimeStats &stats) : previous(renderStats) { renderStats = &stats; }
    ~StatsScope() { renderStats = previous; }
};
struct SubmissionScope {
    std::unique_lock<std::recursive_mutex> lock;
    explicit SubmissionScope(RuntimeStats &stats) : lock(renderSubmissionMutex(), std::defer_lock) {
        const auto start = Clock::now();
        lock.lock();
        stats.submissionWaitMs += elapsed(start);
    }
};
void check(cudaError_t r) {
    if (r)
        throw std::runtime_error(cudaGetErrorString(r));
}
void check(OptixResult r) {
    if (r)
        throw std::runtime_error("Cluster OptiX error " + std::to_string(r));
}
template <class T> T *ptr(CUdeviceptr p) { return reinterpret_cast<T *>(p); }
uint64_t align(uint64_t n, uint64_t a = 128) { return (n + a - 1) & ~(a - 1); }
struct Event {
    cudaEvent_t value{};
    explicit Event(bool timing = false) {
        check(cudaEventCreateWithFlags(&value, timing ? cudaEventDefault : cudaEventDisableTiming));
    }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event &) = delete;
    bool ready() const {
        auto r = cudaEventQuery(value);
        if (r == cudaErrorNotReady)
            return false;
        check(r);
        return true;
    }
    void record(cudaStream_t s) { check(cudaEventRecord(value, s)); }
    void wait() {
        const auto start = Clock::now();
        check(cudaEventSynchronize(value));
        if (renderStats) {
            ++renderStats->syncCalls;
            renderStats->syncMs += elapsed(start);
        }
    }
};
double milliseconds(const Event &a, const Event &b) {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a.value, b.value));
    return ms;
}
struct Buffer {
    CUdeviceptr p{};
    uint64_t bytes{};
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    ~Buffer() {
        if (p)
            cudaFree(ptr<void>(p));
    }
    bool reserve(uint64_t n, cudaStream_t stream) {
        if (bytes >= n)
            return false;
        n = align(std::max(n, bytes + bytes / 2), 256);
        void *next{};
        check(cudaMallocAsync(&next, n, stream));
        if (p)
            check(cudaFreeAsync(ptr<void>(p), stream));
        p = reinterpret_cast<CUdeviceptr>(next);
        bytes = n;
        return true;
    }
};
struct Arena {
    uint64_t capacity{}, used{};
    void *host{};
    Buffer device;
    ~Arena() {
        if (host)
            cudaFreeHost(host);
    }
    void init(uint64_t n, cudaStream_t stream) {
        capacity = n;
        check(cudaMallocHost(&host, n));
        device.reserve(n, stream);
    }
    CUdeviceptr add(const void *p, uint64_t n) {
        used = align(used);
        if (n > capacity - used)
            throw std::runtime_error("CLAS upload task exceeds staging capacity");
        const auto offset = used;
        used += n;
        if (n)
            std::memcpy(static_cast<char *>(host) + offset, p, n);
        return device.p + offset;
    }
    template <class T> CUdeviceptr add(const std::vector<T> &v) {
        return add(v.data(), v.size() * sizeof(T));
    }
    void submit(cudaStream_t s) {
        if (used)
            check(cudaMemcpyAsync(ptr<void>(device.p), host, used, cudaMemcpyHostToDevice, s));
    }
};
// CPU suballocator over fixed GPU storage. Released spans are always fenced.
struct Pool {
    using SizeIndex = std::multimap<uint64_t, uint64_t>;
    struct Span {
        uint64_t bytes;
        SizeIndex::iterator sizeEntry;
    };
    Buffer buffer;
    SizeIndex bySize;
    std::map<uint64_t, Span> free;
    uint64_t used{};
    void insert(uint64_t offset, uint64_t bytes) {
        auto entry = bySize.emplace(bytes, offset);
        free.emplace(offset, Span{bytes, entry});
    }
    void erase(std::map<uint64_t, Span>::iterator entry) {
        bySize.erase(entry->second.sizeEntry);
        free.erase(entry);
    }
    void init(uint64_t bytes, cudaStream_t s) {
        buffer.reserve(bytes, s);
        insert(0, bytes);
    }
    bool fits(uint64_t bytes) const { return bySize.lower_bound(align(bytes)) != bySize.end(); }
    uint64_t allocate(uint64_t bytes) {
        bytes = align(bytes);
        auto entry = bySize.lower_bound(bytes);
        if (entry == bySize.end())
            throw std::runtime_error("CLAS pool has no free range");
        const auto size = entry->first, offset = entry->second;
        erase(free.find(offset));
        if (size > bytes)
            insert(offset + bytes, size - bytes);
        used += bytes;
        return offset;
    }
    void release(uint64_t offset, uint64_t bytes) {
        bytes = align(bytes);
        used -= bytes;
        auto next = free.lower_bound(offset);
        if (next != free.end() && offset + bytes == next->first) {
            bytes += next->second.bytes;
            erase(next);
        }
        next = free.lower_bound(offset);
        if (next != free.begin()) {
            auto previous = std::prev(next);
            if (previous->first + previous->second.bytes == offset) {
                offset = previous->first;
                bytes += previous->second.bytes;
                erase(previous);
            }
        }
        insert(offset, bytes);
    }
};
struct PoolChunk;
struct Resident {
    PoolChunk *geometryChunk{},*clasChunk{};
    uint64_t geometryOffset{}, geometryBytes{}, clasOffset{}, clasBytes{};
    uint32_t slot{}, count{}, children{}, wanted{1};
    bool permanent{}, retiring{}, ready{};
};
struct Prepared {
    uint32_t id;
    Page page;
};
struct Batch {
    std::vector<Prepared> pages;
    double readMs{};
};
struct UploadJob {
    CUdeviceptr args{}, pages{}, descriptors{};
    uint32_t clusters{}, pageCount{}, descriptorCount{}, triangles{}, vertices{};
    struct Copy {
        CUdeviceptr source, dest;
        uint64_t bytes;
    };
    std::vector<Copy> copies;
    std::vector<uint32_t> ids;
};
// Resident groups and GAS batches keep stable addresses until their render
// fences release them. Commit storage in chunks within each configured budget
// instead of eagerly allocating every pool's complete ceiling. An empty chunk
// can be returned without relocating any live geometry or acceleration data.
struct ResidencyHeap {
    cudaMemPool_t memory{};
    uint64_t committed{};
    void init() {
        int device{};check(cudaGetDevice(&device));
        cudaMemPoolProps props{};
        props.allocType=cudaMemAllocationTypePinned;
        props.location.type=cudaMemLocationTypeDevice;
        props.location.id=device;
        check(cudaMemPoolCreate(&memory,&props));
    }
    void updateReleaseThreshold() {
        // CUDA's threshold includes live allocations, rather than just cache.
        uint64_t threshold=committed+(128ull<<20);
        check(cudaMemPoolSetAttribute(memory,cudaMemPoolAttrReleaseThreshold,&threshold));
    }
    CUdeviceptr allocate(uint64_t bytes,cudaStream_t stream) {
        void* address{};check(cudaMallocFromPoolAsync(&address,bytes,memory,stream));
        committed+=bytes;
        try { updateReleaseThreshold(); }
        catch(...) { committed-=bytes;cudaFreeAsync(address,stream);throw; }
        return reinterpret_cast<CUdeviceptr>(address);
    }
    void release(CUdeviceptr address,uint64_t bytes,cudaStream_t stream) {
        check(cudaFreeAsync(ptr<void>(address),stream));
        committed-=bytes;updateReleaseThreshold();
    }
};
struct PoolChunk {
    Pool storage;
    cudaStream_t stream;
    ResidencyHeap* heap;
    PoolChunk(uint64_t bytes,cudaStream_t s,ResidencyHeap* h):stream(s),heap(h) {
        storage.buffer.p=heap->allocate(bytes,s);
        storage.buffer.bytes=bytes;
        storage.insert(0,bytes);
    }
    ~PoolChunk() {
        if(storage.buffer.p) {
            heap->release(storage.buffer.p,storage.buffer.bytes,stream);
            storage.buffer.p=0;
        }
    }
};
struct ChunkPool {
    uint64_t capacity{},reserved{},used{};
    cudaStream_t stream{};
    ResidencyHeap* memoryPool{};
    std::vector<std::unique_ptr<PoolChunk>> chunks;
    void init(uint64_t bytes,cudaStream_t s,ResidencyHeap* pool) { capacity=bytes&~uint64_t(255);stream=s;memoryPool=pool; }
    bool fits(uint64_t bytes) const {
        bytes=align(bytes);
        for(const auto& chunk:chunks)if(chunk->storage.fits(bytes))return true;
        return bytes<=capacity-reserved;
    }
    std::pair<PoolChunk*,uint64_t> allocate(uint64_t bytes) {
        bytes=align(bytes);
        PoolChunk* chosen=nullptr;
        for(const auto& chunk:chunks)if(chunk->storage.fits(bytes)){chosen=chunk.get();break;}
        if(!chosen) {
            if(bytes>capacity-reserved)throw std::runtime_error("Resident storage pool is full");
            const uint64_t size=std::min(align(bytes,64ull<<20),capacity-reserved);
            chunks.push_back(std::make_unique<PoolChunk>(size,stream,memoryPool));
            chosen=chunks.back().get();reserved+=size;
            if(renderStats)++renderStats->allocationCalls;
        }
        const auto offset=chosen->storage.allocate(bytes);used+=bytes;
        return {chosen,offset};
    }
    void release(PoolChunk* chunk,uint64_t offset,uint64_t bytes) {
        chunk->storage.release(offset,bytes);used-=align(bytes);
        if(chunk->storage.used)return;
        reserved-=chunk->storage.buffer.bytes;
        const auto it=std::find_if(chunks.begin(),chunks.end(),[&](const auto& p){return p.get()==chunk;});
        chunks.erase(it);
        if(renderStats)++renderStats->freeCalls;
    }
};
struct GasBatch {
    ChunkPool *pool{};
    PoolChunk *chunk{};
    uint64_t offset{}, bytes{};
    ~GasBatch() {
        if (pool)
            pool->release(chunk,offset,bytes);
    }
    CUdeviceptr address() const { return chunk->storage.buffer.p+offset; }
};
struct GasToken {
    std::shared_ptr<GasBatch> owner;
    uint32_t references{};
};
struct Scene {
    Buffer work, counts, offsets, scan, slots, handles, ranges, wanted, result, requests;
    Buffer args, gasHandles, instances, ias, scratch, hashes, sortedHashes, ids, sortedIDs, canonical, same,
        reuse, tokens, newTokens;
    Event phaseDone, lastUse, uploadBegin{true}, uploadEnd{true}, clasBegin{true}, clasEnd{true},
        lodBegin{true}, lodEnd{true}, gasBegin{true}, gasEnd{true}, iasBegin{true}, iasEnd{true};
    Event gasUploadBegin{true}, gasUploadEnd{true};
    enum Phase { Idle, Sizing, Traversal, Build, Ready, Unchanged, Retry } phase = Idle;
    bool retiring{};
    uint32_t workCount{};
    uint32_t gasPoolIndex{};
    bool rebuildAllGases{};
    OptixTraversableHandle traversable{};
    std::array<float, 3> camera{};
    float projection{}, error{};
    ZoomView zoom;
    Clock::time_point submitted{};
    UploadJob uploadJob;
    std::vector<gpu::GroupPatch> patches;
    double sizingMs{}, uploadPriorMs{};
    double pageReadMs{}, stagingMs{}, prepareMs{};
    std::vector<uint32_t> residentIDs, removed;
    std::vector<uint32_t> usedTokens;
    gpu::Result *receipt{};
    gpu::Result *clasReceipt{};
    gpu::Request *requestReceipt{};
    uint32_t *wantedReceipt{};
    gpu::PageBuild *pageReceipt{};
    uint32_t *canonicalReceipt{};
    uint32_t *reuseReceipt{};
    ~Scene() {
        if (receipt)
            cudaFreeHost(receipt);
        if (clasReceipt)
            cudaFreeHost(clasReceipt);
        if (requestReceipt)
            cudaFreeHost(requestReceipt);
        if (wantedReceipt)
            cudaFreeHost(wantedReceipt);
        if (pageReceipt)
            cudaFreeHost(pageReceipt);
        if (canonicalReceipt)
            cudaFreeHost(canonicalReceipt);
        if (reuseReceipt)
            cudaFreeHost(reuseReceipt);
    }
    void init(uint32_t placements, uint32_t descriptorSlots) {
        check(cudaMallocHost(reinterpret_cast<void **>(&receipt), sizeof(gpu::Result)));
        check(cudaMallocHost(reinterpret_cast<void **>(&clasReceipt), sizeof(gpu::Result)));
        check(cudaMallocHost(reinterpret_cast<void **>(&requestReceipt),
                             gpu::MaxRequests * sizeof(gpu::Request)));
        // Every resident group owns at least one descriptor slot. Match the
        // receipt capacity to the table instead of imposing a second fixed cap.
        check(cudaMallocHost(reinterpret_cast<void **>(&wantedReceipt), uint64_t(descriptorSlots) * sizeof(uint32_t)));
        check(cudaMallocHost(reinterpret_cast<void **>(&pageReceipt), 128 * sizeof(gpu::PageBuild)));
        check(cudaMallocHost(reinterpret_cast<void **>(&canonicalReceipt), placements * sizeof(uint32_t)));
        check(cudaMallocHost(reinterpret_cast<void **>(&reuseReceipt), placements * sizeof(uint32_t)));
    }
};
struct Retirement {
    std::vector<uint32_t> pages;
    std::unique_ptr<Event> done;
};
} // namespace
struct Runtime::Impl {
    Asset asset;
    RuntimeSettings settings;
    RuntimeStats counters, frameStats;
    uint64_t mainUpdates{};
    std::future<void> preparing;
    cudaStream_t uploadStream{}, buildStream{};
    ResidencyHeap residencyHeap;
    bool initialized{}, allocationPressure{};
    uint32_t sbt{}, stamp{}, totalGroups{}, slotCount = 1u << 20, capacity = 4u << 20;
    int front = -1, candidate = -1;
    bool traversalPending = false;
    uint64_t residencyVersion{}, traversedResidency{};
    float error{};
    AdaptiveLod adaptive;
    LodControl lodControl;
    float requestedError{};
    Clock::time_point policyTime{};
    uint32_t unusedGroups{}, refinableGroups{};
    bool budgetBlocked{};
    std::array<float, 3> lastCamera{};
    float lastProjection{};
    ZoomView lastZoom;
    bool hasView{};
    std::array<float, 3> requestedCamera{};
    float requestedProjection{};
    ZoomView requestedZoom;
    bool hasRequestedView{};
    ChunkPool geometryPool,clasPool,gasPools[2];
    Pool descriptorPool;
    uint32_t activeGasPool{};
    bool compactGasPool{};
    Buffer gasHandleTable;
    std::vector<GasToken> gasTokens;
    std::vector<uint32_t> freeGasTokens;
    Buffer groups, refined, geometries, placements, placementOrder, residentList, groupSlots, slotHandles,
        descriptors, stamps;
    Buffer clasSizes, clasDest, clasScratch, clasStatus, clasCount;
    Arena upload;
    Scene scenes[2];
    std::vector<Retirement> retired;
    std::vector<uint32_t> geometryOffsets, parentOffsets, parentIDs;
    std::vector<gpu::Geometry> geometryHost;
    std::vector<gpu::Placement> placementHost;
    std::vector<uint32_t> order;
    std::vector<std::pair<uint32_t, uint32_t>> slotOwners;
    std::map<uint32_t, Resident> residents;
    std::vector<gpu::Request> requests;
    std::future<Batch> reading;
    std::vector<uint32_t> readingIDs, pendingRemoval;
    // Only the serialized reader task accesses this bounded file cache.
    std::map<uint32_t, std::pair<std::unique_ptr<std::ifstream>, uint64_t>> pageFiles;
    uint64_t readTick{};
    CUdeviceptr pendingGroupPatch{};

    explicit Impl(const std::filesystem::path &path, RuntimeSettings s)
        : asset(loadAsset(path)), settings(s), error(s.pixelError) {
        requestedError = s.pixelError;
        adaptive.reset(requestedError);
        if(settings.materials)for(const auto& geometry:asset.geometries)
            if(!geometry.hasAttributes)throw std::runtime_error("Textured CLAS needs attributes on every mesh; rebake custom meshes with --attributes 1");
        if (asset.geometries.empty() || asset.placements.empty())
            throw std::runtime_error("CLAS asset has no renderable geometry");
        if (!s.geometryBytes || !s.clasBytes || !(s.pixelError > 0) || !std::isfinite(s.pixelError) ||
            !s.loadsPerFrame)
            throw std::runtime_error("Invalid CLAS streaming settings");
        // The original million-slot table was sized for a 2 GiB geometry pool.
        // It otherwise becomes the limiting pool even when the user supplies
        // more geometry/CLAS memory. Grow all descriptor-indexed storage together.
        const uint64_t requestedSlots=s.geometryBytes/2048+(s.geometryBytes%2048!=0);
        if(requestedSlots>(1u<<24))throw std::runtime_error("CLAS geometry budget exceeds supported descriptor capacity");
        while(slotCount<requestedSlots)slotCount*=2;
        if (const char *value = std::getenv("ZENO_CLAS_INITIAL_REFERENCES")) {
            capacity = uint32_t(std::stoul(value));
            if (capacity < 16 || capacity > (32u << 20))
                throw std::runtime_error("Invalid initial CLAS reference capacity");
        }
    }
    ~Impl() {
        if (preparing.valid())
            preparing.wait();
        if (reading.valid())
            reading.wait();
        if (uploadStream || initialized)
            cudaStreamSynchronize(buildStream);
        if (uploadStream)
            cudaStreamSynchronize(uploadStream);
        if (initialized)
            cudaStreamSynchronize(0);
        // PoolChunk uses stream-ordered freeing; destroy its owners while the
        // build stream is still valid, after all scene readers have finished.
        gasTokens.clear();
        geometryPool.chunks.clear();
        clasPool.chunks.clear();
        if(buildStream)cudaStreamSynchronize(buildStream);
        if(residencyHeap.memory)cudaMemPoolDestroy(residencyHeap.memory);
        if (uploadStream)
            cudaStreamDestroy(uploadStream);
        if (buildStream)
            cudaStreamDestroy(buildStream);
    }
    std::pair<uint32_t, uint32_t> key(uint32_t id) const {
        const uint32_t g = uint32_t(std::upper_bound(geometryOffsets.begin(), geometryOffsets.end(), id) -
                                    geometryOffsets.begin() - 1);
        return {g, id - geometryOffsets[g]};
    }
    const Group &group(uint32_t id) const {
        auto [g, i] = key(id);
        return asset.geometries[g].groups[i];
    }
    uint64_t geometryBytes(uint32_t id) const {
        const auto &g = group(id);
        const uint64_t core=uint64_t(g.vertices)*sizeof(zeno_cluster_device::VertexNormal)+uint64_t(g.triangles)*3;
        return settings.materials?align(core,4)+uint64_t(g.vertices)*sizeof(zeno_cluster_device::ShadingVertex)+uint64_t(g.triangles)*2:core;
    }
    OptixClusterAccelBuildInput triangleInput(uint32_t count, uint32_t triangles, uint32_t vertices) const {
        OptixClusterAccelBuildInput in{};
        in.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES;
        auto &t = in.triangles;
        t.flags = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE;
        t.maxArgCount = count;
        t.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        t.maxSbtIndexValue = sbt;
        t.maxUniqueSbtIndexCountPerArg = 1;
        t.maxTriangleCountPerArg = MaxTriangles;
        t.maxVertexCountPerArg = MaxVertices;
        t.maxTotalTriangleCount = triangles;
        t.maxTotalVertexCount = vertices;
        return in;
    }
    void reserve(Buffer &b, uint64_t n) {
        const bool hadStorage = b.p != 0;
        if (b.reserve(n, buildStream)) {
            ++counters.allocationCalls;
            counters.freeCalls += hadStorage;
        }
    }
    template <class T> void initialUpload(Buffer &buffer, const std::vector<T> &data) {
        reserve(buffer, data.size() * sizeof(T));
        check(cudaMemcpyAsync(ptr<void>(buffer.p), data.data(), data.size() * sizeof(T),
                              cudaMemcpyHostToDevice, buildStream));
    }
    void initialize(OptixDeviceContext ctx, uint32_t material) {
        const auto start = Clock::now();
        sbt = material;
        unsigned int flags = 0, vertices = 0, triangles = 0;
        check(optixDeviceContextGetProperty(ctx, OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL, &flags, sizeof(flags)));
        check(optixDeviceContextGetProperty(ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES, &vertices,
                                            sizeof(vertices)));
        check(optixDeviceContextGetProperty(ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES,
                                            &triangles, sizeof(triangles)));
        if (!(flags & OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL_FLAG_STANDARD) || vertices < MaxVertices ||
            triangles < MaxTriangles)
            throw std::runtime_error("Device does not support these CLAS limits");
        check(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking));
        // The candidate owns immutable AS storage, and publication waits for
        // phaseDone. Retired geometry/GAS spans remain fenced by render lastUse.
        // Keep candidate builds off stream 0 so preview rays and readback do
        // not sit behind another scene's sizing/build/receipt work.
        check(cudaStreamCreateWithFlags(&buildStream,cudaStreamNonBlocking));
        // The renderer's default async allocator retains a large working set.
        // Keep residency recycling separate so retired chunks can actually
        // return VRAM after the existing worker completion events, while a
        // small cache avoids physical allocation churn for replacement GASes.
        residencyHeap.init();
        geometryPool.init(settings.geometryBytes, buildStream,&residencyHeap);
        clasPool.init(settings.clasBytes, buildStream,&residencyHeap);
        const uint64_t gasBankBytes=std::max<uint64_t>(1ull<<30,settings.geometryBytes/2);
        for (auto &pool : gasPools)
            pool.init(gasBankBytes, buildStream,&residencyHeap);
        gasTokens.resize(asset.placements.size() * 3 + 128);
        reserve(gasHandleTable, gasTokens.size() * sizeof(uint64_t));
        for (uint32_t i = uint32_t(gasTokens.size()); i > 0; --i)
            freeGasTokens.push_back(i - 1);
        descriptorPool.insert(0, uint64_t(slotCount) * 128);
        reserve(descriptors, uint64_t(slotCount) * sizeof(zeno_cluster_device::Cluster));
        reserve(slotHandles, uint64_t(slotCount) * sizeof(uint64_t));
        slotOwners.resize(slotCount, {gpu::Invalid, 0});
        upload.init(64ull << 20, buildStream);
        for (auto &scene : scenes)
            scene.init(uint32_t(asset.placements.size()),slotCount);
        std::vector<gpu::Group> hostGroups;
        std::vector<int32_t> hostRefined;
        geometryOffsets.push_back(0);
        geometryHost.resize(asset.geometries.size());
        for (uint32_t g = 0; g < asset.geometries.size(); ++g) {
            auto &geo = geometryHost[g];
            geo.groupOffset = uint32_t(hostGroups.size());
            geo.groupCount = uint32_t(asset.geometries[g].groups.size());
            std::set<uint32_t> roots(asset.geometries[g].roots.begin(), asset.geometries[g].roots.end());
            for (uint32_t id = 0; id < geo.groupCount; ++id) {
                const auto &source = asset.geometries[g].groups[id];
                gpu::Group next{};
                std::copy(source.bounds.begin(), source.bounds.end(), next.bounds);
                if (source.refined.size() > 256)
                    throw std::runtime_error("Group exceeds GPU cut mask capacity");
                next.refinedOffset = uint32_t(hostRefined.size());
                next.count = uint32_t(source.refined.size());
                next.geometry = g;
                next.depth = source.depth;
                next.root = roots.count(id) != 0;
                for (int32_t child : source.refined)
                    hostRefined.push_back(child < 0 ? -1 : int32_t(geo.groupOffset) + child);
                hostGroups.push_back(next);
            }
            geometryOffsets.push_back(uint32_t(hostGroups.size()));
        }
        totalGroups = uint32_t(hostGroups.size());
        parentOffsets.resize(totalGroups + 1);
        std::vector<int32_t> children;
        const auto forChildren = [&](uint32_t id, const auto &visit) {
            children.clear();
            const auto &g = hostGroups[id];
            for (uint32_t c = 0; c < g.count; ++c)
                if (hostRefined[g.refinedOffset + c] >= 0)
                    children.push_back(hostRefined[g.refinedOffset + c]);
            std::sort(children.begin(), children.end());
            children.erase(std::unique(children.begin(), children.end()), children.end());
            for (auto child : children)
                visit(uint32_t(child));
        };
        for (uint32_t id = 0; id < totalGroups; ++id)
            forChildren(id, [&](uint32_t child) { ++parentOffsets[child + 1]; });
        std::partial_sum(parentOffsets.begin(), parentOffsets.end(), parentOffsets.begin());
        parentIDs.resize(parentOffsets.back());
        auto cursor = parentOffsets;
        for (uint32_t id = 0; id < totalGroups; ++id)
            forChildren(id, [&](uint32_t child) { parentIDs[cursor[child]++] = id; });
        std::vector<std::vector<uint32_t>> byGeometry(asset.geometries.size());
        placementHost.resize(asset.placements.size());
        for (uint32_t p = 0; p < asset.placements.size(); ++p) {
            const auto &source = asset.placements[p];
            auto &target = placementHost[p];
            target.geometry = source.geometry;
            byGeometry[source.geometry].push_back(p);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    target.transform[r * 4 + c] = source.transform[c * 4 + r];
            const auto m = glm::make_mat4(source.transform.data());
            float bound = 0;
            for (int i = 0; i < 3; ++i) {
                float sum = 0;
                for (int j = 0; j < 3; ++j)
                    sum += std::abs(glm::dot(glm::vec3(m[i]), glm::vec3(m[j])));
                bound = std::max(bound, sum);
            }
            target.scale = std::sqrt(bound) * (1 + 4 * std::numeric_limits<float>::epsilon());
        }
        for (uint32_t g = 0; g < geometryHost.size(); ++g) {
            geometryHost[g].placementOffset = uint32_t(order.size());
            geometryHost[g].placementCount = uint32_t(byGeometry[g].size());
            order.insert(order.end(), byGeometry[g].begin(), byGeometry[g].end());
        }
        initialUpload(groups, hostGroups);
        initialUpload(refined, hostRefined);
        initialUpload(placements, placementHost);
        initialUpload(placementOrder, order);
        reserve(groupSlots, uint64_t(totalGroups) * sizeof(uint32_t));
        check(cudaMemsetAsync(ptr<void>(groupSlots.p), 0xff, groupSlots.bytes, buildStream));
        reserve(stamps, uint64_t(totalGroups) * sizeof(uint32_t));
        check(cudaMemsetAsync(ptr<void>(stamps.p), 0, stamps.bytes, buildStream));
        reserve(clasStatus, sizeof(gpu::Result));
        reserve(clasCount, sizeof(uint32_t));
        check(cudaStreamSynchronize(buildStream));
        // FLT_MAX marks a group whose coarser representation is forbidden.
        // Pin those groups and their ancestors with the roots, so even the
        // first published cut respects the asset's minimum coverage policy.
        // Near-camera refinement below that minimum remains asynchronous.
        std::vector<bool> permanent(totalGroups);
        for (uint32_t id = 0; id < totalGroups; ++id)
            permanent[id] = hostGroups[id].root || hostGroups[id].bounds[4] == std::numeric_limits<float>::max();
        for (uint32_t id = 0; id < totalGroups; ++id)
            if (permanent[id])
                for (uint32_t p = parentOffsets[id]; p < parentOffsets[id + 1]; ++p)
                    permanent[parentIDs[p]] = true;
        // Startup is explicitly blocking: establish complete safe coverage.
        Batch batch;
        const auto flush = [&] {
            upload.used = 0;
            std::vector<gpu::GroupPatch> patches;
            auto job = stagePages(ctx, batch, true, patches);
            pendingGroupPatch = upload.add(patches);
            upload.submit(uploadStream);
            check(cudaStreamSynchronize(uploadStream));
            submitPageSizes(ctx, job);
            check(cudaStreamSynchronize(buildStream));
            std::vector<gpu::PageBuild> actual(job.pageCount);
            check(cudaMemcpy(actual.data(), ptr<void>(job.pages), actual.size() * sizeof(gpu::PageBuild),
                             cudaMemcpyDeviceToHost));
            allocatePages(job, actual.data(), true);
            check(cudaMemcpyAsync(ptr<void>(job.pages), actual.data(), actual.size() * sizeof(gpu::PageBuild),
                                  cudaMemcpyHostToDevice, buildStream));
            submitPages(ctx, job, patches);
            check(cudaStreamSynchronize(buildStream));
            checkClasStatus();
            batch.pages.clear();
        };
        for (uint32_t g = 0; g < asset.geometries.size(); ++g)
            for (uint32_t id = uint32_t(asset.geometries[g].groups.size()); id-- > 0;) {
                if (!permanent[geometryOffsets[g] + id]) continue;
                batch.pages.push_back({geometryOffsets[g] + id, readPage(asset.geometries[g], id)});
                if (batch.pages.size() == 128)
                    flush();
            }
        if (!batch.pages.empty())
            flush();
        initialized = true;
        counters.planningMs = elapsed(start);
        std::cout << "[CLAS] GPU resident-cut pipeline geometries=" << asset.geometries.size()
                  << " placements=" << asset.placements.size() << " groups=" << totalGroups
                  << " coarsePages=" << residents.size() << " descriptorSlots=" << slotCount
                  << " gasBankBytes=" << gasBankBytes << std::endl;
    }
    bool parentsResident(uint32_t id) const {
        for (uint32_t i = parentOffsets[id]; i < parentOffsets[id + 1]; ++i) {
            auto found = residents.find(parentIDs[i]);
            if (found == residents.end() || found->second.retiring || !found->second.ready)
                return false;
        }
        return true;
    }
    UploadJob stagePages(OptixDeviceContext ctx, const Batch &batch, bool permanent,
                         std::vector<gpu::GroupPatch> &patches) {
        TimelineRange timeline("CLAS: stage geometry");
        UploadJob job;
        std::vector<OptixClusterAccelBuildInputTrianglesArgs> args;
        std::vector<gpu::PageBuild> pages;
        std::vector<gpu::DescriptorPatch> descriptorPatches;
        for (const auto &prepared : batch.pages) {
            uint32_t id = prepared.id;
            if (residents.count(id))
                continue;
            const auto &page = prepared.page;
            const auto [geometry, local] = key(id);
            const auto bytes = geometryBytes(id);
            if (!permanent && !parentsResident(id))
                continue;
            if (!geometryPool.fits(bytes) || !descriptorPool.fits(page.clusters.size() * 128)) {
                if (permanent)
                    throw std::runtime_error("Permanent coarse coverage exceeds streaming pools");
                ++counters.deferredUpdates;
                allocationPressure = true;
                continue;
            }
            Resident resident;
            const auto geometryAllocation=geometryPool.allocate(bytes);
            resident.geometryChunk=geometryAllocation.first;
            resident.geometryOffset=geometryAllocation.second;
            resident.geometryBytes = align(bytes);
            resident.count = uint32_t(page.clusters.size());
            resident.slot = uint32_t(descriptorPool.allocate(resident.count * 128) / 128);
            resident.permanent = permanent;
            const auto base = resident.geometryChunk->storage.buffer.p + resident.geometryOffset;
            std::vector<std::array<float,3>> positions(page.vertices.size());
            std::vector<zeno_cluster_device::VertexNormal> normals(page.vertices.size());
            for(size_t i=0;i<page.vertices.size();++i) {
                std::memcpy(positions[i].data(),page.vertices[i].position,sizeof(positions[i]));
                std::memcpy(normals[i].normal,page.vertices[i].normal,sizeof(normals[i]));
            }
            // The upload arena is not reused until the candidate's build fence
            // completes. CLAS input positions never enter the resident pool.
            const auto positionData=upload.add(positions);
            const auto normalBytes=normals.size()*sizeof(normals[0]);
            job.copies.push_back({upload.add(normals),base,normalBytes});
            job.copies.push_back({upload.add(page.indices), base + normalBytes, page.indices.size()});
            const auto attributeBase=base+align(normalBytes+page.indices.size(),4);
            const auto materialBase=attributeBase+page.vertices.size()*sizeof(zeno_cluster_device::ShadingVertex);
            if(settings.materials) {
                if(page.attributes.size()!=page.vertices.size()||page.materials.size()*3!=page.indices.size())
                    throw std::runtime_error("Incomplete cluster shading attributes");
                job.copies.push_back({upload.add(page.attributes),attributeBase,page.attributes.size()*sizeof(page.attributes[0])});
                job.copies.push_back({upload.add(page.materials),materialBase,page.materials.size()*sizeof(page.materials[0])});
            }
            pages.push_back({uint32_t(args.size()), resident.count, 0, 0, resident.slot});
            for (uint32_t c = 0; c < resident.count; ++c) {
                const auto &cluster = page.clusters[c];
                const uint32_t slot = resident.slot + c;
                OptixClusterAccelBuildInputTrianglesArgs a{};
                a.clusterId = slot;
                a.vertexCount = cluster.vertexCount;
                a.triangleCount = cluster.triangleCount;
                a.indexFormat = OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_8BIT;
                a.basePrimitiveInfo.sbtIndex = sbt;
                a.basePrimitiveInfo.primitiveFlags = settings.materials?0:OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_ANYHIT;
                a.vertexBuffer = positionData + cluster.vertexOffset * sizeof(positions[0]);
                a.vertexBufferStrideInBytes = sizeof(positions[0]);
                a.indexBuffer = base + normalBytes + cluster.indexOffset;
                args.push_back(a);
                descriptorPatches.push_back({slot,
                                             {ptr<const zeno_cluster_device::VertexNormal>(base+cluster.vertexOffset*sizeof(normals[0])),
                                              ptr<const unsigned char>(a.indexBuffer), group(id).depth,
                                              geometry, local, cluster.triangleCount,
                                              settings.materials?ptr<const zeno_cluster_device::ShadingVertex>(attributeBase+cluster.vertexOffset*sizeof(zeno_cluster_device::ShadingVertex)):nullptr,
                                              settings.materials?ptr<const unsigned short>(materialBase+cluster.indexOffset/3*2):nullptr}});
                slotOwners[slot] = {id, c};
                job.triangles += cluster.triangleCount;
                job.vertices += cluster.vertexCount;
            }
            for (uint32_t i = parentOffsets[id]; i < parentOffsets[id + 1]; ++i)
                ++residents.at(parentIDs[i]).children;
            patches.push_back({id, resident.slot});
            residents.emplace(id, resident);
            job.ids.push_back(id);
            ++residencyVersion;
        }
        job.pageCount = uint32_t(pages.size());
        job.clusters = uint32_t(args.size());
        job.descriptorCount = uint32_t(descriptorPatches.size());
        job.args = upload.add(args);
        job.pages = upload.add(pages);
        job.descriptors = upload.add(descriptorPatches);
        return job;
    }
    void checkClasStatus() {
        gpu::Result status{};
        check(cudaMemcpy(&status, ptr<void>(clasStatus.p), sizeof(status), cudaMemcpyDeviceToHost));
        if (status.overflow)
            throw std::runtime_error("CLAS explicit destinations exceed reserved page size");
    }
    bool allocatePages(UploadJob &job, gpu::PageBuild *actual, bool permanent) {
        bool admitted = true;
        for (size_t i = 0; i < job.ids.size(); ++i) {
            auto &resident = residents.at(job.ids[i]);
            const auto bytes = actual[i].bytes;
            if (!bytes || !clasPool.fits(bytes)) {
                admitted = false;
                break;
            }
            const auto clasAllocation=clasPool.allocate(bytes);
            resident.clasChunk=clasAllocation.first;
            resident.clasOffset=clasAllocation.second;
            resident.clasBytes = bytes;
            actual[i].output = resident.clasChunk->storage.buffer.p + resident.clasOffset;
        }
        if (admitted) {
            for (uint32_t id : job.ids)
                residents.at(id).ready = true;
            counters.loads += job.ids.size();
            allocationPressure = false;
            return true;
        }
        if (permanent)
            throw std::runtime_error("Exact permanent CLAS coverage exceeds pool capacity");
        for (uint32_t id : job.ids) {
            const auto resident = residents.at(id);
            if (resident.clasBytes)
                clasPool.release(resident.clasChunk,resident.clasOffset,resident.clasBytes);
            geometryPool.release(resident.geometryChunk,resident.geometryOffset,resident.geometryBytes);
            descriptorPool.release(uint64_t(resident.slot) * 128, uint64_t(resident.count) * 128);
            for (uint32_t i = parentOffsets[id]; i < parentOffsets[id + 1]; ++i)
                --residents.at(parentIDs[i]).children;
            residents.erase(id);
        }
        allocationPressure = true;
        ++counters.deferredUpdates;
        ++residencyVersion;
        return false;
    }
    void submitPageSizes(OptixDeviceContext ctx, const UploadJob &job) {
        check(cudaMemsetAsync(ptr<void>(clasStatus.p), 0, sizeof(gpu::Result), buildStream));
        for (const auto &c : job.copies)
            check(cudaMemcpyAsync(ptr<void>(c.dest), ptr<void>(c.source), c.bytes, cudaMemcpyDeviceToDevice,
                                  buildStream));
        const auto input = triangleInput(job.clusters, job.triangles, job.vertices);
        OptixAccelBufferSizes sizes{};
        check(optixClusterAccelComputeMemoryUsage(ctx, OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES, &input,
                                                  &sizes));
        reserve(clasScratch, sizes.tempSizeInBytes);
        reserve(clasSizes, uint64_t(job.clusters) * sizeof(uint32_t));
        reserve(clasDest, uint64_t(job.clusters) * sizeof(uint64_t));
        OptixClusterAccelBuildModeDesc desc{};
        desc.mode = OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES;
        desc.getSize.outputSizesBuffer = clasSizes.p;
        desc.getSize.tempBuffer = clasScratch.p;
        desc.getSize.tempBufferSizeInBytes = clasScratch.bytes;
        check(optixClusterAccelBuild(ctx, buildStream, &desc, &input, job.args, 0, 0));
        gpu::destinations(ptr<gpu::PageBuild>(job.pages), job.pageCount, ptr<const uint32_t>(clasSizes.p),
                          ptr<uint64_t>(clasDest.p), ptr<gpu::Result>(clasStatus.p), buildStream);
    }
    void submitPages(OptixDeviceContext ctx, const UploadJob &job,
                     const std::vector<gpu::GroupPatch> &patches) {
        check(cudaMemsetAsync(ptr<void>(clasStatus.p), 0, sizeof(gpu::Result), buildStream));
        if (job.clusters) {
            const auto input = triangleInput(job.clusters, job.triangles, job.vertices);
            OptixAccelBufferSizes build{};
            check(optixClusterAccelComputeMemoryUsage(
                ctx, OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS, &input, &build));
            reserve(clasScratch, build.tempSizeInBytes);
            gpu::destinations(ptr<gpu::PageBuild>(job.pages), job.pageCount, ptr<const uint32_t>(clasSizes.p),
                              ptr<uint64_t>(clasDest.p), ptr<gpu::Result>(clasStatus.p), buildStream);
            gpu::guardCount(ptr<const gpu::Result>(clasStatus.p), ptr<uint32_t>(clasCount.p), job.clusters,
                            buildStream);
            OptixClusterAccelBuildModeDesc desc{};
            desc.mode = OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS;
            desc.explicitDest.tempBuffer = clasScratch.p;
            desc.explicitDest.tempBufferSizeInBytes = clasScratch.bytes;
            desc.explicitDest.destAddressesBuffer = clasDest.p;
            desc.explicitDest.destAddressesStrideInBytes = 8;
            desc.explicitDest.outputHandlesBuffer = clasDest.p;
            desc.explicitDest.outputHandlesStrideInBytes = 8;
            desc.explicitDest.outputSizesBuffer = clasSizes.p;
            desc.explicitDest.outputSizesStrideInBytes = 4;
            check(optixClusterAccelBuild(ctx, buildStream, &desc, &input, job.args, clasCount.p, 0));
            gpu::scatterHandles(ptr<const gpu::PageBuild>(job.pages), job.pageCount,
                                ptr<const uint64_t>(clasDest.p), ptr<uint64_t>(slotHandles.p), buildStream);
        }
        gpu::patch(ptr<const gpu::DescriptorPatch>(job.descriptors), job.descriptorCount,
                   ptr<zeno_cluster_device::Cluster>(descriptors.p),
                   ptr<const gpu::GroupPatch>(pendingGroupPatch), uint32_t(patches.size()),
                   ptr<uint32_t>(groupSlots.p), buildStream);
    }
    void retireCompleted() {
        for (auto &scene : scenes)
            if (scene.retiring && scene.lastUse.ready()) {
                releaseSceneGases(scene);
                scene.retiring = false;
            }
        for (auto it = retired.begin(); it != retired.end();) {
            if (!it->done->ready()) {
                ++it;
                continue;
            }
            for (uint32_t id : it->pages) {
                auto found = residents.find(id);
                if (found == residents.end())
                    throw std::runtime_error("Retired CLAS page lost ownership");
                const auto resident = found->second;
                if (resident.children)
                    throw std::runtime_error("Retiring a CLAS parent with live descendants");
                for (uint32_t i = parentOffsets[id]; i < parentOffsets[id + 1]; ++i)
                    --residents.at(parentIDs[i]).children;
                geometryPool.release(resident.geometryChunk,resident.geometryOffset,resident.geometryBytes);
                clasPool.release(resident.clasChunk,resident.clasOffset,resident.clasBytes);
                descriptorPool.release(uint64_t(resident.slot) * 128, uint64_t(resident.count) * 128);
                residents.erase(found);
                ++counters.evictions;
            }
            it = retired.erase(it);
        }
        counters.gasBytes=gasPools[0].used+gasPools[1].used;
        counters.gasReservedBytes=gasPools[0].reserved+gasPools[1].reserved;
        counters.geometryReservedBytes=geometryPool.reserved;
        counters.clasReservedBytes=clasPool.reserved;
    }
    void retireRemoved(std::vector<uint32_t> pages) {
        if (pages.empty())
            return;
        Retirement r;
        r.pages = std::move(pages);
        r.done = std::make_unique<Event>();
        r.done->record(0);
        retired.push_back(std::move(r));
    }
    void chooseEvictions() {
        // Remove leaves in a candidate cut before retiring their storage.
        uint32_t count = 0;
        for (auto &[id, resident] : residents) {
            if (resident.permanent || resident.retiring || resident.children || resident.wanted)
                continue;
            resident.retiring = true;
            pendingRemoval.push_back(id);
            ++residencyVersion;
            if (++count == 128)
                break;
        }
    }
    void startReading() {
        if (reading.valid() || requests.empty() || budgetBlocked)
            return;
        std::set<uint32_t> needed;
        const auto ancestors = [&](auto &&self, uint32_t id) -> void {
            auto found = residents.find(id);
            if (found != residents.end() && !found->second.retiring)
                return;
            if (!needed.insert(id).second)
                return;
            for (uint32_t i = parentOffsets[id]; i < parentOffsets[id + 1]; ++i)
                self(self, parentIDs[i]);
        };
        std::sort(requests.begin(), requests.end(), [](auto a, auto b) { return a.priority > b.priority; });
        readingIDs.clear();
        uint64_t bytes = 0;
        const uint32_t limit = std::min(settings.loadsPerFrame, 128u);
        for (const auto &request : requests) {
            needed.clear();
            ancestors(ancestors, request.group);
            std::vector<uint32_t> chain(needed.begin(), needed.end());
            std::sort(chain.begin(), chain.end(),
                      [&](auto a, auto b) { return group(a).depth > group(b).depth; });
            for (uint32_t id : chain) {
                if (residents.count(id) || !parentsResident(id) ||
                    std::find(readingIDs.begin(), readingIDs.end(), id) != readingIDs.end())
                    continue;
                const auto n = geometryBytes(id);
                if (bytes + n > std::min<uint64_t>(
                                    16ull << 20, std::min(settings.geometryBytes, settings.clasBytes) / 8) &&
                    !readingIDs.empty())
                    continue;
                readingIDs.push_back(id);
                bytes += n;
                if (readingIDs.size() == limit)
                    break;
            }
            if (readingIDs.size() == limit)
                break;
        }
        if (readingIDs.empty())
            return;
        auto ids = readingIDs;
        std::sort(ids.begin(), ids.end());
        reading = std::async(std::launch::async, [this, ids] {
            const auto start = Clock::now();
            Batch batch;
            for (uint32_t id : ids) {
                const auto [g, i] = key(id);
                auto file = pageFiles.find(g);
                if (file == pageFiles.end()) {
                    if (pageFiles.size() >= 64) {
                        auto oldest = std::min_element(
                            pageFiles.begin(), pageFiles.end(),
                            [](const auto &a, const auto &b) { return a.second.second < b.second.second; });
                        pageFiles.erase(oldest);
                    }
                    auto stream =
                        std::make_unique<std::ifstream>(asset.geometries[g].pages, std::ios::binary);
                    if (!*stream)
                        throw std::runtime_error("Cannot open CLAS page file");
                    file = pageFiles.emplace(g, std::make_pair(std::move(stream), uint64_t(0))).first;
                }
                file->second.second = ++readTick;
                batch.pages.push_back({id, readPage(asset.geometries[g], i, *file->second.first)});
            }
            batch.readMs = elapsed(start);
            return batch;
        });
    }
    void layout(Scene &scene, std::vector<gpu::Range> &ranges) {
        scene.residentIDs.clear();
        for (const auto &[id, r] : residents)
            if (!r.retiring && r.ready)
                scene.residentIDs.push_back(id);
        ranges.resize(asset.placements.size());
        uint64_t work = 0;
        size_t cursor = 0;
        for (uint32_t g = 0; g < geometryHost.size(); ++g) {
            auto &geo = geometryHost[g];
            geo.residentOffset = uint32_t(cursor);
            while (cursor < scene.residentIDs.size() && scene.residentIDs[cursor] < geometryOffsets[g + 1])
                ++cursor;
            geo.residentCount = uint32_t(cursor) - geo.residentOffset;
            geo.workOffset = uint32_t(work);
            geo.workCount = geo.residentCount * geo.placementCount;
            for (uint32_t i = 0; i < geo.placementCount; ++i)
                ranges[order[geo.placementOffset + i]] = {uint32_t(work) + i * geo.residentCount, 0};
            work += geo.workCount;
        }
        if (work > std::numeric_limits<uint32_t>::max() / 32)
            throw std::runtime_error("CLAS resident dispatch capacity exceeded");
        scene.workCount = uint32_t(work);
        reserve(scene.work, work * sizeof(gpu::Work));
        reserve(scene.counts, (work + 1) * sizeof(uint32_t));
        reserve(scene.offsets, (work + 1) * sizeof(uint32_t));
        reserve(scene.scan,
                std::max(gpu::scanBytes(uint32_t(work + 1)), gpu::sortBytes(uint32_t(placementHost.size()))));
        reserve(scene.slots, uint64_t(capacity) * sizeof(uint32_t));
        reserve(scene.handles, uint64_t(capacity) * sizeof(uint64_t));
        reserve(scene.ranges, ranges.size() * sizeof(gpu::Range));
        reserve(scene.wanted, scene.residentIDs.size() * sizeof(uint32_t));
        reserve(scene.result, sizeof(gpu::Result));
        reserve(scene.requests, gpu::MaxRequests * sizeof(gpu::Request));
        reserve(residentList, scene.residentIDs.size() * sizeof(uint32_t));
        reserve(geometries, geometryHost.size() * sizeof(gpu::Geometry));
        reserve(scene.hashes, placementHost.size() * 8);
        reserve(scene.sortedHashes, placementHost.size() * 8);
        reserve(scene.ids, placementHost.size() * 4);
        reserve(scene.sortedIDs, placementHost.size() * 4);
        reserve(scene.canonical, placementHost.size() * 4);
        reserve(scene.same, placementHost.size() * 4);
        reserve(scene.reuse, placementHost.size() * 4);
        reserve(scene.tokens, placementHost.size() * 4);
        reserve(scene.newTokens, placementHost.size() * 4);
    }
    void startTraversal(OptixDeviceContext ctx, Scene &scene) {
        auto &job = scene.uploadJob;
        pendingGroupPatch = upload.add(scene.patches);
        std::vector<gpu::Range> ranges;
        layout(scene, ranges);
        const auto residentData = upload.add(scene.residentIDs), geometryData = upload.add(geometryHost),
                   rangeData = upload.add(ranges);
        scene.uploadBegin.record(uploadStream);
        upload.submit(uploadStream);
        if (job.pageCount)
            check(cudaMemcpyAsync(ptr<void>(job.pages), scene.pageReceipt,
                                  job.pageCount * sizeof(gpu::PageBuild), cudaMemcpyHostToDevice,
                                  uploadStream));
        scene.uploadEnd.record(uploadStream);
        SubmissionScope submission(counters);
        check(cudaStreamWaitEvent(buildStream, scene.uploadEnd.value, 0));
        scene.clasBegin.record(buildStream);
        submitPages(ctx, job, scene.patches);
        scene.clasEnd.record(buildStream);
        check(cudaMemcpyAsync(ptr<void>(residentList.p), ptr<void>(residentData),
                              scene.residentIDs.size() * sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                              buildStream));
        check(cudaMemcpyAsync(ptr<void>(geometries.p), ptr<void>(geometryData),
                              geometryHost.size() * sizeof(gpu::Geometry), cudaMemcpyDeviceToDevice,
                              buildStream));
        check(cudaMemcpyAsync(ptr<void>(scene.ranges.p), ptr<void>(rangeData),
                              ranges.size() * sizeof(gpu::Range), cudaMemcpyDeviceToDevice, buildStream));
        check(cudaMemsetAsync(ptr<void>(scene.wanted.p), 0, scene.residentIDs.size() * sizeof(uint32_t),
                              buildStream));
        if (++stamp == 0) {
            check(cudaMemsetAsync(ptr<void>(stamps.p), 0, stamps.bytes, buildStream));
            ++stamp;
        }
        gpu::Traversal t{};
        t.groups = ptr<const gpu::Group>(groups.p);
        t.refined = ptr<const int32_t>(refined.p);
        t.geometries = ptr<const gpu::Geometry>(geometries.p);
        t.geometryCount = uint32_t(geometryHost.size());
        t.placements = ptr<const gpu::Placement>(placements.p);
        t.placementOrder = ptr<const uint32_t>(placementOrder.p);
        t.residents = ptr<const uint32_t>(residentList.p);
        t.groupSlots = ptr<const uint32_t>(groupSlots.p);
        t.handles = ptr<const uint64_t>(slotHandles.p);
        t.descriptors = ptr<const zeno_cluster_device::Cluster>(descriptors.p);
        t.stamps = ptr<uint32_t>(stamps.p);
        t.stamp = stamp;
        t.work = ptr<gpu::Work>(scene.work.p);
        t.workCount = scene.workCount;
        t.counts = ptr<uint32_t>(scene.counts.p);
        t.offsets = ptr<uint32_t>(scene.offsets.p);
        t.slots = ptr<uint32_t>(scene.slots.p);
        t.renderHandles = ptr<uint64_t>(scene.handles.p);
        t.capacity = capacity;
        t.ranges = ptr<gpu::Range>(scene.ranges.p);
        t.placementCount = uint32_t(placementHost.size());
        t.wanted = ptr<uint32_t>(scene.wanted.p);
        t.requests = ptr<gpu::Request>(scene.requests.p);
        t.result = ptr<gpu::Result>(scene.result.p);
        t.same = ptr<uint32_t>(scene.same.p);
        t.rebuildAll = compactGasPool ||
                       gasPools[activeGasPool].used > gasPools[activeGasPool].capacity * .70 ||
                       (front >= 0 && std::getenv("ZENO_CLAS_FORCE_GAS_COMPACTION"));
        scene.rebuildAllGases = t.rebuildAll;
        if (front >= 0) {
            t.previousRanges = ptr<const gpu::Range>(scenes[front].ranges.p);
            t.previousSlots = ptr<const uint32_t>(scenes[front].slots.p);
            t.previousTokens = ptr<const uint32_t>(scenes[front].tokens.p);
        }
        std::copy(scene.camera.begin(), scene.camera.end(), t.eye);
        t.projection = scene.projection;
        t.zoom = scene.zoom;
        t.error = error;
        scene.lodBegin.record(buildStream);
        gpu::traverse(t, ptr<void>(scene.scan.p), scene.scan.bytes, buildStream);
        gpu::canonicalize(t, ptr<uint64_t>(scene.hashes.p), ptr<uint64_t>(scene.sortedHashes.p),
                          ptr<uint32_t>(scene.ids.p), ptr<uint32_t>(scene.sortedIDs.p),
                          ptr<uint32_t>(scene.canonical.p), ptr<uint32_t>(scene.reuse.p),
                          ptr<void>(scene.scan.p), scene.scan.bytes, buildStream);
        scene.lodEnd.record(buildStream);
        check(cudaMemcpyAsync(scene.receipt, ptr<void>(scene.result.p), sizeof(gpu::Result),
                              cudaMemcpyDeviceToHost, buildStream));
        check(cudaMemcpyAsync(scene.clasReceipt, ptr<void>(clasStatus.p), sizeof(gpu::Result),
                              cudaMemcpyDeviceToHost, buildStream));
        check(cudaMemcpyAsync(scene.requestReceipt, ptr<void>(scene.requests.p),
                              gpu::MaxRequests * sizeof(gpu::Request), cudaMemcpyDeviceToHost, buildStream));
        check(cudaMemcpyAsync(scene.wantedReceipt, ptr<void>(scene.wanted.p),
                              scene.residentIDs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                              buildStream));
        check(cudaMemcpyAsync(scene.canonicalReceipt, ptr<void>(scene.canonical.p), placementHost.size() * 4,
                              cudaMemcpyDeviceToHost, buildStream));
        check(cudaMemcpyAsync(scene.reuseReceipt, ptr<void>(scene.reuse.p), placementHost.size() * 4,
                              cudaMemcpyDeviceToHost, buildStream));
        scene.phaseDone.record(buildStream);
        scene.phase = Scene::Traversal;
        scene.error = error;
        traversedResidency = residencyVersion;
        counters.loadingMs = counters.hostStreamingMs;
        ++counters.plans;
    }
    bool beginTraversal(OptixDeviceContext ctx, const std::array<float, 3> &camera, float projection, bool blocking, const ZoomView &zoom) {
        TimelineRange timeline("CLAS: begin traversal");
        const int index = front < 0 ? 0 : 1 - front;
        Scene &scene = scenes[index];
        if (scene.retiring) {
            if (!blocking && !scene.lastUse.ready()) {
                ++counters.deferredUpdates;
                return false;
            }
            if (blocking)
                scene.lastUse.wait();
            releaseSceneGases(scene);
            scene.retiring = false;
        }
        const auto start = Clock::now();
        upload.used = 0;
        scene.patches.clear();
        for (uint32_t id : pendingRemoval)
            scene.patches.push_back({id, gpu::Invalid});
        scene.removed = std::move(pendingRemoval);
        pendingRemoval.clear();
        Batch batch;
        if (reading.valid() &&
            (blocking || reading.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
            batch = reading.get();
            readingIDs.clear();
            counters.pageReadMs = batch.readMs;
        }
        scene.uploadJob = stagePages(ctx, batch, false, scene.patches);
        scene.camera = camera;
        scene.projection = projection;
        scene.zoom = zoom;
        scene.submitted = start;
        scene.sizingMs = scene.uploadPriorMs = 0;
        scene.pageReadMs = batch.readMs;
        scene.prepareMs = 0;
        candidate = index;
        if (scene.uploadJob.clusters) {
            scene.uploadBegin.record(uploadStream);
            upload.submit(uploadStream);
            scene.uploadEnd.record(uploadStream);
            SubmissionScope submission(counters);
            check(cudaStreamWaitEvent(buildStream, scene.uploadEnd.value, 0));
            scene.clasBegin.record(buildStream);
            submitPageSizes(ctx, scene.uploadJob);
            scene.clasEnd.record(buildStream);
            check(cudaMemcpyAsync(scene.pageReceipt, ptr<void>(scene.uploadJob.pages),
                                  scene.uploadJob.pageCount * sizeof(gpu::PageBuild), cudaMemcpyDeviceToHost,
                                  buildStream));
            scene.phaseDone.record(buildStream);
            scene.phase = Scene::Sizing;
        } else
            startTraversal(ctx, scene);
        counters.hostStreamingMs += elapsed(start);
        scene.stagingMs = elapsed(start);
        counters.loadingMs = counters.hostStreamingMs;
        return true;
    }
    void releaseSceneGases(Scene &scene) {
        for (uint32_t token : scene.usedTokens) {
            auto &entry = gasTokens.at(token);
            if (!entry.references)
                throw std::runtime_error("GAS token retired twice");
            if (--entry.references == 0) {
                entry.owner.reset();
                freeGasTokens.push_back(token);
            }
        }
        scene.usedTokens.clear();
    }
    bool buildScene(OptixDeviceContext ctx, Scene &scene) {
        const auto start = Clock::now();
        scene.gasPoolIndex = scene.rebuildAllGases ? 1 - activeGasPool : activeGasPool;
        auto &gasPool = gasPools[scene.gasPoolIndex];
        if (scene.rebuildAllGases && gasPool.used) {
            compactGasPool = true;
            traversedResidency = ~0ull;
            ++counters.deferredUpdates;
            return false;
        }
        const uint32_t count = uint32_t(placementHost.size()), newCount = scene.receipt->newGases;
        reserve(scene.args, uint64_t(newCount) * sizeof(OptixClusterAccelBuildInputClustersArgs));
        reserve(scene.gasHandles, uint64_t(newCount) * sizeof(uint64_t));
        reserve(scene.instances, uint64_t(count) * sizeof(OptixInstance));
        OptixClusterAccelBuildInput input{};
        input.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
        input.clusters.flags = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
        input.clusters.maxArgCount = newCount;
        input.clusters.maxTotalClusterCount = scene.receipt->newReferences;
        input.clusters.maxClusterCountPerArg = scene.receipt->newMaxPerGas;
        OptixAccelBufferSizes sizes{};
        std::shared_ptr<GasBatch> batch;
        if (newCount) {
            check(optixClusterAccelComputeMemoryUsage(
                ctx, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS, &input, &sizes));
            if (sizes.outputSizeInBytes > gasPool.capacity)
                throw std::runtime_error("Complete GAS batch exceeds its storage bank capacity");
            if (!gasPool.fits(sizes.outputSizeInBytes) || freeGasTokens.size() < newCount) {
                ++counters.deferredUpdates;
                compactGasPool = true;
                traversedResidency = ~0ull;
                return false;
            }
            batch = std::make_shared<GasBatch>();
            batch->bytes = align(sizes.outputSizeInBytes);
            const auto allocation=gasPool.allocate(batch->bytes);
            batch->chunk=allocation.first;
            batch->offset=allocation.second;
            batch->pool = &gasPool;
            reserve(scene.scratch, sizes.tempSizeInBytes);
        }
        std::vector<uint32_t> tokens(count, gpu::Invalid), newTokens;
        newTokens.reserve(newCount);
        for (uint32_t p = 0; p < count; ++p)
            if (scene.canonicalReceipt[p] == p) {
                uint32_t token = scene.reuseReceipt[p];
                if (token == gpu::Invalid) {
                    token = freeGasTokens.back();
                    freeGasTokens.pop_back();
                    if (gasTokens[token].references || gasTokens[token].owner)
                        throw std::runtime_error("Live GAS token reached free list");
                    gasTokens[token].owner = batch;
                    newTokens.push_back(token);
                } else if (token >= gasTokens.size() || !gasTokens[token].references)
                    throw std::runtime_error("GPU attempted to reuse a retired GAS");
                tokens[p] = token;
            }
        if (newTokens.size() != newCount)
            throw std::runtime_error("GPU/host new GAS count mismatch");
        for (uint32_t p = 0; p < count; ++p) {
            const uint32_t canonical = scene.canonicalReceipt[p];
            if (canonical >= count || tokens[canonical] == gpu::Invalid)
                throw std::runtime_error("Invalid canonical GAS placement");
            tokens[p] = tokens[canonical];
        }
        scene.usedTokens = tokens;
        std::sort(scene.usedTokens.begin(), scene.usedTokens.end());
        scene.usedTokens.erase(std::unique(scene.usedTokens.begin(), scene.usedTokens.end()),
                               scene.usedTokens.end());
        for (uint32_t token : scene.usedTokens)
            ++gasTokens[token].references;
        // The previous scene's GAS tokens remain referenced until lastUse.
        // New immutable batches share the fixed AS pool, so unchanged objects
        // survive without a copy, relocation, or per-object allocation.
        upload.used = 0;
        const auto tokenData = upload.add(tokens), newData = upload.add(newTokens);
        scene.gasUploadBegin.record(uploadStream);
        upload.submit(uploadStream);
        scene.gasUploadEnd.record(uploadStream);
        SubmissionScope submission(counters);
        check(cudaStreamWaitEvent(buildStream, scene.gasUploadEnd.value, 0));
        check(cudaMemcpyAsync(ptr<void>(scene.tokens.p), ptr<void>(tokenData), tokens.size() * 4,
                              cudaMemcpyDeviceToDevice, buildStream));
        if (newCount)
            check(cudaMemcpyAsync(ptr<void>(scene.newTokens.p), ptr<void>(newData), newTokens.size() * 4,
                                  cudaMemcpyDeviceToDevice, buildStream));
        scene.gasBegin.record(buildStream);
        if (newCount) {
            gpu::gasArgs(ptr<const gpu::Range>(scene.ranges.p), count, ptr<const uint32_t>(scene.canonical.p),
                         ptr<const uint32_t>(scene.offsets.p), ptr<const uint32_t>(scene.reuse.p),
                         scene.handles.p, ptr<OptixClusterAccelBuildInputClustersArgs>(scene.args.p),
                         buildStream);
            OptixClusterAccelBuildModeDesc desc{};
            desc.mode = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
            desc.implicitDest.outputBuffer = batch->address();
            desc.implicitDest.outputBufferSizeInBytes = batch->bytes;
            desc.implicitDest.tempBuffer = scene.scratch.p;
            desc.implicitDest.tempBufferSizeInBytes = scene.scratch.bytes;
            desc.implicitDest.outputHandlesBuffer = scene.gasHandles.p;
            check(optixClusterAccelBuild(ctx, buildStream, &desc, &input, scene.args.p, 0, 0));
            gpu::publishGasHandles(newCount, ptr<const uint64_t>(scene.gasHandles.p),
                                   ptr<const uint32_t>(scene.newTokens.p), ptr<uint64_t>(gasHandleTable.p),
                                   buildStream);
        }
        scene.gasEnd.record(buildStream);
        gpu::instances(ptr<const gpu::Placement>(placements.p), count, ptr<const uint64_t>(gasHandleTable.p),
                       ptr<const uint32_t>(scene.tokens.p), ptr<OptixInstance>(scene.instances.p),
                       buildStream);
        OptixBuildInput in{};
        in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        in.instanceArray.instances = scene.instances.p;
        in.instanceArray.numInstances = count;
        OptixAccelBuildOptions options{};
        options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        check(optixAccelComputeMemoryUsage(ctx, &options, &in, 1, &sizes));
        reserve(scene.ias, sizes.outputSizeInBytes);
        reserve(scene.scratch, sizes.tempSizeInBytes);
        scene.iasBegin.record(buildStream);
        check(optixAccelBuild(ctx, buildStream, &options, &in, 1, scene.scratch.p, scene.scratch.bytes,
                              scene.ias.p, scene.ias.bytes, &scene.traversable, nullptr, 0));
        scene.iasEnd.record(buildStream);
        scene.phaseDone.record(buildStream);
        scene.phase = Scene::Build;
        counters.gasBuilds += newCount;
        ++counters.iasBuilds;
        counters.accelerationMs += elapsed(start);
        return true;
    }
    void acceptView(const Scene &scene) {
        const auto &result = *scene.receipt;
        counters.effectivePixelError = scene.error;
        counters.renderClusters = result.clusters;
        counters.renderTriangles = result.triangles;
        counters.minLod = result.minLod;
        counters.maxLod = result.maxLod;
        lastCamera = scene.camera;
        lastProjection = scene.projection;
        lastZoom = scene.zoom;
        hasView = true;
    }
    void profileScene(const Scene &scene, bool built) {
        counters.gpuUploadMs = scene.uploadPriorMs + milliseconds(scene.uploadBegin, scene.uploadEnd);
        counters.gpuClasMs = scene.sizingMs + milliseconds(scene.clasBegin, scene.clasEnd);
        counters.gpuLodMs = milliseconds(scene.lodBegin, scene.lodEnd);
        counters.gpuGasMs = built ? milliseconds(scene.gasBegin, scene.gasEnd) : 0;
        counters.gpuIasMs = built ? milliseconds(scene.iasBegin, scene.iasEnd) : 0;
        if (built) {
            counters.gpuUploadMs += milliseconds(scene.gasUploadBegin, scene.gasUploadEnd);
            counters.sharedGases = scene.receipt->uniqueGases;
            counters.rebuiltGases = scene.receipt->newGases;
            counters.reusedGases = scene.receipt->uniqueGases - scene.receipt->newGases;
        }
        counters.sceneLatencyMs = elapsed(scene.submitted);
        counters.hostTaskMs = scene.pageReadMs + scene.stagingMs + scene.prepareMs;
    }
    void prepareCandidate(OptixDeviceContext ctx) {
        TimelineRange timeline("CLAS: prepare candidate");
        auto &scene = scenes[candidate];
        const auto initialQueueWait = counters.submissionWaitMs;
        double activeMs = 0;
        scene.phaseDone.wait();
        if (scene.phase == Scene::Sizing) {
            const auto start = Clock::now();
            scene.sizingMs = milliseconds(scene.clasBegin, scene.clasEnd);
            scene.uploadPriorMs = milliseconds(scene.uploadBegin, scene.uploadEnd);
            if (!allocatePages(scene.uploadJob, scene.pageReceipt, false)) {
                const std::set<uint32_t> rejected(scene.uploadJob.ids.begin(), scene.uploadJob.ids.end());
                scene.patches.erase(std::remove_if(scene.patches.begin(), scene.patches.end(),
                                                   [&](auto p) { return rejected.count(p.group) != 0; }),
                                    scene.patches.end());
                scene.uploadJob = {};
            }
            startTraversal(ctx, scene);
            activeMs += elapsed(start);
            scene.phaseDone.wait();
        }
        if (scene.phase == Scene::Traversal) {
            const auto start = Clock::now();
            const auto &result = *scene.receipt;
            if (scene.clasReceipt->overflow)
                throw std::runtime_error("CLAS page build exceeded its reserved range");
            if (result.overflow & 1) {
                if (capacity >= (32u << 20))
                    throw std::runtime_error("CLAS render reference capacity exhausted");
                capacity *= 2;
                traversedResidency = ~0ull;
                scene.phase = Scene::Retry;
                return;
            }
            if (result.emptyPlacements)
                throw std::runtime_error("Resident cut left a placement without geometry");
            requests.assign(scene.requestReceipt,
                            scene.requestReceipt + std::min(result.requests, gpu::MaxRequests));
            size_t flag = 0;
            unusedGroups = refinableGroups = 0;
            for (auto &[id, resident] : residents)
                if (flag < scene.residentIDs.size() && id == scene.residentIDs[flag]) {
                    resident.wanted = scene.wantedReceipt[flag++] & 1;
                    if (!resident.permanent && !resident.retiring) {
                        if (resident.wanted) ++refinableGroups;
                        else ++unusedGroups;
                    }
                }
            if (flag != scene.residentIDs.size())
                throw std::runtime_error("CLAS residency receipt lost its owner");
            // A rejected fixed-quality load waits for a view/control change or
            // reclaimed storage. Do not repeatedly decode an unadmittable page.
            if (!lodControl.adaptive && allocationPressure) budgetBlocked = true;
            else startReading();
            if (result.changed) {
                if (!buildScene(ctx, scene)) {
                    scene.phase = Scene::Retry;
                    return;
                }
            } else
                scene.phase = Scene::Unchanged;
            activeMs += elapsed(start);
            if (scene.phase == Scene::Build) {
                scene.phaseDone.wait();
                scene.phase = Scene::Ready;
            }
        }
        counters.backgroundSubmissionWaitMs = counters.submissionWaitMs - initialQueueWait;
        counters.backgroundPrepareMs = std::max(0., activeMs - counters.backgroundSubmissionWaitMs);
        scene.prepareMs = counters.backgroundPrepareMs;
    }
    void startPreparation(OptixDeviceContext ctx, const std::array<float, 3> &camera, float projection, const ZoomView &zoom) {
        if (preparing.valid() || (candidate < 0 && !traversalPending))
            throw std::runtime_error("Invalid CLAS preparation ownership");
        CUcontext current{};
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || !current)
            throw std::runtime_error("CLAS preparation needs a current CUDA context");
        const bool begin=traversalPending;
        traversalPending=false;
        preparing = std::async(std::launch::async, [this, ctx, current, begin, camera, projection, zoom] {
            if (cuCtxPushCurrent(current) != CUDA_SUCCESS)
                throw std::runtime_error("Cannot bind CLAS build context");
            struct Pop {
                ~Pop() {
                    CUcontext old{};
                    cuCtxPopCurrent(&old);
                }
            } pop;
            // This worker owns candidate/residency preparation exclusively.
            // It never changes front, the root IAS, or render launch state.
            if(begin)beginTraversal(ctx,camera,projection,false,zoom);
            if(candidate>=0)prepareCandidate(ctx);
        });
    }
    bool finishCandidate(OptixDeviceContext ctx, bool blocking) {
        if (candidate < 0)
            return false;
        auto &scene = scenes[candidate];
        if (scene.phase == Scene::Sizing || scene.phase == Scene::Traversal || scene.phase == Scene::Build) {
            if (!blocking)
                return false;
            prepareCandidate(ctx);
        }
        if (scene.phase == Scene::Retry) {
            pendingRemoval.insert(pendingRemoval.end(), scene.removed.begin(), scene.removed.end());
            scene.phase = Scene::Idle;
            candidate = -1;
            return false;
        }
        if (scene.phase != Scene::Ready && scene.phase != Scene::Unchanged)
            throw std::runtime_error("CLAS scene was published before preparation completed");
        const bool changed = scene.phase == Scene::Ready;
        acceptView(scene);
        profileScene(scene, changed);
        if (changed) {
            if (front >= 0) {
                scenes[front].lastUse.record(0);
                scenes[front].retiring = true;
            }
            retireRemoved(std::move(scene.removed));
            front = candidate;
            activeGasPool = scene.gasPoolIndex;
            if (scene.rebuildAllGases) {
                compactGasPool = false;
                ++counters.gasCompactions;
            }
            ++counters.sceneVersion;
            if (scene.receipt->cutChanged)
                ++counters.cutVersion;
        } else
            retireRemoved(std::move(scene.removed));
        candidate = -1;
        scene.phase = Scene::Idle;
        return changed;
    }
    float poolLoad() const {
        return std::max({float(double(geometryPool.used)/settings.geometryBytes),
                         float(double(clasPool.used)/settings.clasBytes),
                         float(double(descriptorPool.used)/(uint64_t(slotCount)*128))});
    }
    bool policyPending() const {
        const float load = poolLoad();
        return (budgetBlocked && !retired.empty()) ||
            (unusedGroups && load > (error > requestedError ? .70f : .85f)) ||
            adaptive.pending(requestedError, load, lodControl.adaptive,
                             refinableGroups || !requests.empty(), allocationPressure);
    }
    bool tick(OptixDeviceContext ctx, const std::array<float, 3> &camera, float projection, bool blocking, const ZoomView &zoom) {
        const auto geometryBefore=geometryPool.used, clasBefore=clasPool.used;
        retireCompleted();
        if(geometryPool.used<geometryBefore || clasPool.used<clasBefore)
            allocationPressure=budgetBlocked=false;
        bool changed = finishCandidate(ctx, blocking);
        if (candidate >= 0)
            return changed;
        const float load = poolLoad();
        if (allocationPressure || load > (error > requestedError ? .70f : .85f))
            chooseEvictions();
        const bool reclaiming = !pendingRemoval.empty() || !retired.empty();
        const auto now = Clock::now();
        const double seconds = !blocking && policyTime != Clock::time_point{} ?
            std::chrono::duration<double>(now-policyTime).count() : 1./60.;
        policyTime = now;
        const float nextError = adaptive.advance(requestedError, load, seconds, lodControl.adaptive,
            !reclaiming && (refinableGroups || !requests.empty()), allocationPressure);
        if (nextError != error) {
            error = nextError;
            traversedResidency = ~0ull;
            budgetBlocked = false;
        }
        if (reclaiming) budgetBlocked = false;
        if (!lodControl.adaptive && allocationPressure && !reclaiming && !reading.valid())
            budgetBlocked = true;
        startReading();
        const bool ioReady = reading.valid() && (blocking || reading.wait_for(std::chrono::seconds(0)) ==
                                                                 std::future_status::ready);
        const bool different = !hasView || camera != lastCamera || projection != lastProjection || zoom != lastZoom;
        if (different || residencyVersion != traversedResidency || ioReady || !pendingRemoval.empty()) {
            if(blocking)beginTraversal(ctx,camera,projection,true,zoom);
            else traversalPending=true;
        }
        return changed;
    }
    std::vector<ReferenceCut> referenceCuts() const {
        if (front < 0)
            return {};
        const auto &scene = scenes[front];
        std::vector<gpu::Range> ranges(asset.placements.size());
        check(cudaMemcpy(ranges.data(), ptr<void>(scene.ranges.p), ranges.size() * sizeof(gpu::Range),
                         cudaMemcpyDeviceToHost));
        uint32_t total = 0;
        for (auto range : ranges)
            total = std::max(total, range.offset + range.count);
        std::vector<uint32_t> slots(total);
        check(cudaMemcpy(slots.data(), ptr<void>(scene.slots.p), slots.size() * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));
        std::vector<ReferenceCut> cuts;
        for (uint32_t p = 0; p < ranges.size(); ++p) {
            ReferenceCut cut;
            cut.geometry = asset.placements[p].geometry;
            cut.placement = p;
            const auto range = ranges[p];
            for (uint32_t i = 0; i < range.count; ++i) {
                const auto [id, c] = slotOwners.at(slots.at(range.offset + i));
                const auto [g, local] = key(id);
                if (g != cut.geometry)
                    throw std::runtime_error("Published CLAS descriptor belongs to another geometry");
                cut.groups[local].push_back(c);
            }
            cuts.push_back(std::move(cut));
        }
        return cuts;
    }
    void validateCut() const {
        if (front < 0)
            throw std::runtime_error("No complete CLAS scene published");
        const auto &scene = scenes[front];
        const auto actual = referenceCuts();
        std::set<uint32_t> available(scene.residentIDs.begin(), scene.residentIDs.end());
        uint64_t checked = 0;
        for (uint32_t p = 0; p < asset.placements.size(); ++p) {
            const uint32_t geometry = asset.placements[p].geometry;
            const auto &source = asset.geometries[geometry];
            const auto &placement = placementHost[p];
            const auto *m = placement.transform;
            std::map<uint32_t, std::vector<uint32_t>> expected;
            std::set<uint32_t> visited;
            const auto refine = [&](uint32_t child) {
                if (!available.count(geometryOffsets[geometry] + child))
                    return false;
                const auto &b = source.groups[child].bounds;
                const float x = m[0] * b[0] + m[1] * b[1] + m[2] * b[2] + m[3] - scene.camera[0],
                            y = m[4] * b[0] + m[5] * b[1] + m[6] * b[2] + m[7] - scene.camera[1],
                            z = m[8] * b[0] + m[9] * b[1] + m[10] * b[2] + m[11] - scene.camera[2];
                const float distance = std::sqrt(x * x + y * y + z * z), radius = b[3] * placement.scale;
                return b[4] * placement.scale / std::max(distance - radius, .01f) *
                           projectionForSphere(scene.zoom, x, y, z, radius, distance, scene.projection) >
                       scene.error;
            };
            const auto visit = [&](auto &&self, uint32_t id) -> void {
                if (!visited.insert(id).second)
                    return;
                const auto &group = source.groups[id];
                for (uint32_t c = 0; c < group.refined.size(); ++c) {
                    int child = group.refined[c];
                    if (child >= 0 && refine(uint32_t(child)))
                        self(self, uint32_t(child));
                    else
                        expected[id].push_back(c);
                }
            };
            for (uint32_t root : source.roots)
                visit(visit, root);
            if (expected != actual[p].groups)
                throw std::runtime_error(
                    "GPU cut differs from independent resident DAG traversal, placement " +
                    std::to_string(p));
            for (const auto &[_, cs] : expected)
                checked += cs.size();
        }
        std::cout << "[reference cut] version=" << counters.sceneVersion
                  << " placements=" << asset.placements.size() << " clusters=" << checked << " complete=1\n";
    }
};
Runtime::Runtime(const std::filesystem::path &path, RuntimeSettings settings)
    : impl(std::make_unique<Impl>(path, settings)) {}
Runtime::~Runtime() = default;
bool Runtime::update(OptixDeviceContext ctx, uint32_t sbt, const std::array<float, 3> &camera,
                     float projection, bool interactive, const ZoomView &zoom, LodControl control) {
    TimelineRange timeline("CLAS: render update");
    auto &s = *impl;
    const auto start = Clock::now();
    ++s.mainUpdates;
    if (!(projection > 0) || !std::isfinite(projection) ||
        !std::all_of(camera.begin(), camera.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("Invalid CLAS view");
    if (!(control.pixelError >= 0.f) || !std::isfinite(control.pixelError))
        throw std::runtime_error("Invalid requested CLAS pixel error");
    if (zoom.peripheralProjection != 0.f) {
        if (!(zoom.peripheralProjection > 0.f) || !std::isfinite(zoom.peripheralProjection) ||
            !(zoom.feather > 0.f) || !std::isfinite(zoom.feather))
            throw std::runtime_error("Invalid CLAS zoom projection");
        for (const auto &plane : zoom.planes)
            for (float v : plane)
                if (!std::isfinite(v)) throw std::runtime_error("Invalid CLAS zoom plane");
    }
    bool completedPreparation = false;
    if (s.preparing.valid()) {
        if (interactive && s.preparing.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            auto &stats = s.frameStats;
            stats.updates = s.mainUpdates;
            stats.updateMs = elapsed(start);
            stats.loadingMs = stats.planningMs = stats.accelerationMs = stats.hostStreamingMs =
                stats.pageReadMs = stats.backgroundPrepareMs = 0;
            stats.gpuSceneMs = stats.gpuLodMs = stats.gpuClasMs = stats.gpuGasMs = stats.gpuIasMs =
                stats.gpuUploadMs = 0;
            stats.allocationCalls = stats.freeCalls = stats.syncCalls = 0;
            stats.syncMs = 0;
            stats.submissionWaitMs = stats.backgroundSubmissionWaitMs = 0;
            stats.hostTaskMs = 0;
            stats.rebuiltGases = stats.reusedGases = 0;
            stats.planningPending = true;
            stats.settled = false;
            return false;
        }
        s.preparing.get();
        completedPreparation = true;
    }
    const float requestedError = control.pixelError > 0.f ? control.pixelError : s.settings.pixelError;
    if (control.adaptive != s.lodControl.adaptive || requestedError != s.requestedError) {
        s.lodControl = control;
        s.requestedError = requestedError;
        s.adaptive.reset(requestedError);
        s.error = requestedError;
        s.budgetBlocked = s.allocationPressure = false;
        s.traversedResidency = ~0ull;
    }
    // A new view may recover beyond the discrete-LOD limit learned for the
    // previous view. Keep its effective error; recover gradually with headroom.
    if (!s.hasRequestedView || camera != s.requestedCamera || projection != s.requestedProjection || zoom != s.requestedZoom) {
        s.adaptive.viewChanged(requestedError);
        s.budgetBlocked = s.allocationPressure = false;
        s.requestedCamera = camera;
        s.requestedProjection = projection;
        s.requestedZoom = zoom;
        s.hasRequestedView = true;
    }
    s.counters.updates = s.mainUpdates;
    if (!completedPreparation) {
        s.counters.backgroundPrepareMs = 0;
        s.counters.backgroundSubmissionWaitMs = 0;
    }
    s.counters.submissionWaitMs = 0;
    s.counters.hostTaskMs = 0;
    s.counters.updateMs = s.counters.loadingMs = s.counters.planningMs = s.counters.accelerationMs =
        s.counters.hostStreamingMs = s.counters.pageReadMs = 0;
    s.counters.gpuSceneMs = s.counters.gpuLodMs = s.counters.gpuClasMs = s.counters.gpuGasMs =
        s.counters.gpuIasMs = s.counters.gpuUploadMs = 0;
    s.counters.allocationCalls = s.counters.freeCalls = s.counters.syncCalls = 0;
    s.counters.syncMs = 0;
    s.counters.rebuiltGases = s.counters.reusedGases = 0;
    StatsScope statsScope(s.counters);
    if (!s.initialized)
        s.initialize(ctx, sbt);
    else if (s.sbt != sbt)
        throw std::runtime_error("CLAS SBT binding changed; reload asset");
    const auto settled = [&] {
        return s.front >= 0 && s.candidate < 0 && !s.reading.valid() && (s.requests.empty() || s.budgetBlocked) &&
               s.pendingRemoval.empty() && s.traversedResidency == s.residencyVersion && s.hasView &&
               camera == s.lastCamera && projection == s.lastProjection && zoom == s.lastZoom;
    };
    const auto policyStats = [&] {
        s.counters.requestedPixelError = s.requestedError;
        s.counters.adaptiveLod = s.lodControl.adaptive;
        s.counters.poolLoad = s.poolLoad();
        s.counters.smoothedPoolLoad = s.adaptive.load();
        s.counters.qualityLimited = s.budgetBlocked || s.error > s.requestedError * 1.0001f;
        if (!interactive && s.budgetBlocked)
            throw std::runtime_error("Fixed CLAS quality exceeds streaming budgets; increase geometry/CLAS budgets or enable adaptive LOD");
    };
    // The opt-in stress test also exercises AS-only publications at a fixed
    // camera. Normal static views retain the zero-work early exit below.
    if (interactive && settled() && std::getenv("ZENO_CLAS_FORCE_GAS_COMPACTION")) {
        s.compactGasPool = true;
        s.traversedResidency = ~0ull;
    }
    if (settled() && !s.policyPending()) {
        s.retireCompleted();
        ++s.counters.staticSkips;
        s.counters.updateMs = elapsed(start);
        s.counters.settled = true;
        policyStats();
        s.frameStats = s.counters;
        return false;
    }
    bool changed = false;
    const bool blocking = !interactive || s.front < 0;
    for (uint32_t iterations = 0;; ++iterations) {
        changed |= s.tick(ctx, camera, projection, blocking, zoom);
        if (blocking && s.candidate >= 0)
            changed |= s.finishCandidate(ctx, true);
        if (interactive && s.front >= 0)
            break;
        if (settled()) {
            if (s.policyPending()) continue;
            break;
        }
        if (iterations > 20000)
            throw std::runtime_error("CLAS residency failed to converge");
    }
    s.counters.pendingGroups = uint32_t(s.requests.size() + s.readingIDs.size());
    s.counters.residentGroups = uint32_t(s.residents.size());
    s.counters.geometryBytes = s.geometryPool.used;
    s.counters.clasBytes = s.clasPool.used;
    s.counters.gasBytes = s.gasPools[0].used + s.gasPools[1].used;
    s.counters.gasReservedBytes = s.gasPools[0].reserved + s.gasPools[1].reserved;
    s.counters.geometryReservedBytes=s.geometryPool.reserved;
    s.counters.clasReservedBytes=s.clasPool.reserved;
    s.counters.planningPending = s.candidate >= 0 || s.traversalPending || s.reading.valid();
    s.counters.settled = settled() && !s.policyPending();
    policyStats();
    s.counters.updateMs = elapsed(start);
    s.counters.gpuSceneMs =
        s.counters.gpuClasMs + s.counters.gpuLodMs + s.counters.gpuGasMs + s.counters.gpuIasMs;
    if (changed || s.counters.settled)
        std::cout << "[CLAS] version=" << s.counters.sceneVersion
                  << " cutVersion=" << s.counters.cutVersion
                  << " residentGroups=" << s.counters.residentGroups
                  << " renderClusters=" << s.counters.renderClusters << " loads=" << s.counters.loads
                  << " evictions=" << s.counters.evictions << " sharedGases=" << s.counters.sharedGases
                  << " geometryBytes=" << s.counters.geometryBytes << " clasBytes=" << s.counters.clasBytes
                  << " lodError=" << s.counters.effectivePixelError
                  << " requestedError=" << s.counters.requestedPixelError
                  << " adaptive=" << s.counters.adaptiveLod
                  << " poolLoad=" << s.counters.poolLoad
                  << " smoothedLoad=" << s.counters.smoothedPoolLoad
                  << " qualityLimited=" << s.counters.qualityLimited
                  << " pendingGroups=" << s.counters.pendingGroups << " settled=" << s.counters.settled
                  << " hostMs=" << s.counters.hostStreamingMs << " gpuLodMs=" << s.counters.gpuLodMs
                  << " gpuClasMs=" << s.counters.gpuClasMs << " gpuGasMs=" << s.counters.gpuGasMs
                  << " gpuIasMs=" << s.counters.gpuIasMs << std::endl;
    if (changed && std::getenv("ZENO_CLAS_VERIFY_CUTS"))
        s.validateCut();
    // Publish host-visible statistics before handing exclusive preparation
    // ownership to the worker. Pending viewport calls read only this snapshot
    // and the stable front scene; they never wait on the worker's CUDA event.
    s.frameStats = s.counters;
    if (interactive && (s.candidate >= 0 || s.traversalPending))
        s.startPreparation(ctx,camera,projection,zoom);
    s.frameStats.updateMs = elapsed(start);
    return changed;
}
OptixTraversableHandle Runtime::handle() const {
    return impl->front < 0 ? 0 : impl->scenes[impl->front].traversable;
}
const zeno_cluster_device::Cluster *Runtime::descriptors() const {
    return ptr<const zeno_cluster_device::Cluster>(impl->descriptors.p);
}
RuntimeStats Runtime::stats() const { return impl->frameStats; }
std::array<uint64_t,2> Runtime::residencyMemory() const {
    std::array<uint64_t,2> bytes{};
    if(impl->residencyHeap.memory) {
        check(cudaMemPoolGetAttribute(impl->residencyHeap.memory,cudaMemPoolAttrReservedMemCurrent,&bytes[0]));
        check(cudaMemPoolGetAttribute(impl->residencyHeap.memory,cudaMemPoolAttrUsedMemCurrent,&bytes[1]));
    }
    return bytes;
}
std::vector<ReferenceCut> Runtime::referenceCut() const { return impl->referenceCuts(); }
void Runtime::validateCut() const { impl->validateCut(); }
} // namespace zeno::cluster
