#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace zeno::cluster {
// Payload accounting shared by the decoded workspace and tile cache. Limits
// do not include container metadata or the reader's compressed input buffer.
struct HostCacheUsage {
    std::atomic<uint64_t> bytes{0},peak{0};
    void add(uint64_t n);
    void remove(uint64_t n);
};
struct HostBlockStats {
    uint64_t bytes{},peak{},hits{},misses{},reloads{},sharedWaits{},spaceWaits{},evictions{};
};
// Immutable decoded blocks, shared by overlapping tile requests. Fill and IO
// run outside the lock. Active leases cannot be evicted; waiters are notified
// on release and failed fills, including when every workspace slot is busy.
class HostBlockCache {
    struct State;
    std::shared_ptr<State> state;
public:
    class Lease {
        friend class HostBlockCache;
        struct Entry;
        std::shared_ptr<State> state;
        std::shared_ptr<Entry> entry;
        Lease(std::shared_ptr<State>,std::shared_ptr<Entry>);
        void release();
    public:
        Lease(Lease&&) noexcept;
        Lease& operator=(Lease&&) noexcept;
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
        ~Lease();
        const char* data() const;
        size_t size() const;
    };
    HostBlockCache(uint64_t bytes,std::shared_ptr<HostCacheUsage>,bool trackReloads=false,bool reuseStorage=false);
    ~HostBlockCache();
    Lease get(const std::string& key,size_t bytes,const std::function<void(char*)>& fill);
    // Lookup without allocating, waiting for a fill, or admitting new work.
    std::optional<Lease> tryGet(const std::string& key,size_t bytes);
    // Used during source registration, before readers start.
    void setCapacity(uint64_t bytes);
    // Reclaim idle blocks without waiting for active readers. Returns the
    // effective capacity when live leases prevent reaching the target.
    uint64_t trimCapacity(uint64_t target);
    void clear();
    HostBlockStats stats() const;
};

struct HostTileKey {
    uint32_t image{},level{},x{},y{},width{},height{},blockBytes{};
    bool operator==(const HostTileKey&) const;
};
struct BcMipView {
    const char* data{};
    size_t bytes{};
    uint32_t width{},height{},blockBytes{};
};
void copyBcTile(const HostTileKey&,const BcMipView&,char* destination);
struct HostTileStats {
    uint64_t bytes{},peak{},hits{},misses{},demandFills{},prefetched{},prefetchHits{},
        evictions{},prefetchEvictions{},prefetchSkipped{},neighborBytes{};
};
// Three queues distinguish unused neighbors, recent demand and reused demand.
// Neighbor population uses decoded bytes only, never extra IO. It is limited
// to 1/8 of the pool and cannot evict demand tiles. Demand evicts neighbors
// first; a 3/4 cap on the reused queue leaves room for newly visited regions.
class HostTileCache {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    HostTileCache(uint64_t bytes,std::shared_ptr<HostCacheUsage>);
    ~HostTileCache();
    bool read(const HostTileKey&,char* destination);
    // Writes the requested tile first, then retains it and up to eight valid
    // neighbors. BC blocks, partial edge blocks and zero padding are exact.
    void fill(const HostTileKey&,const BcMipView&,char* destination);
    // Growth is safe while readers are active. Clear before shrinking.
    void setCapacity(uint64_t bytes);
    void clear();
    HostTileStats stats() const;
};
}
