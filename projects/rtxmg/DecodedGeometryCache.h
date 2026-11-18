#pragma once
#include "ClusterCache.h"
#include "HostTextureCache.h"
#include <atomic>
#include <functional>
#include <mutex>

namespace zeno::cluster {
struct DecodedGeometryStats {
    HostBlockStats blocks;
    uint64_t bypasses{},restoredBytes{},storedBytes{};
    double restoreMs{},storeMs{};
};
// Runtime-local immutable decoded pages. Only ordinary RAM is used. The
// caller keeps its reusable Page buffers; leases never cross GPU preparation.
class DecodedGeometryCache {
    uint64_t capacity;
    std::shared_ptr<HostCacheUsage> usage=std::make_shared<HostCacheUsage>();
    HostBlockCache blocks;
    std::atomic<uint64_t> bypasses{0},restoredBytes{0},storedBytes{0};
    std::atomic<uint64_t> restoreNs{0},storeNs{0};
    std::mutex reuseMutex;
    std::vector<uint64_t> seenKeys;
    size_t keyCount=0;
public:
    // With a dense key count, admit only pages read more than once. The bounded
    // bitset costs one bit per asset group and keeps cold scans out of the cache.
    explicit DecodedGeometryCache(uint64_t bytes,size_t keys=0);
    // Keys are scoped to this asset/runtime. Oversized pages bypass the cache.
    bool read(uint32_t key,const Group& group,bool attributes,Page& output,
              const std::function<void(Page&)>& decode,bool allowFill=true);
    DecodedGeometryStats stats() const;
    static uint64_t pageBytes(const Group&,bool attributes);
};
}
