#include "HostTextureCache.h"
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace zeno::cluster {
namespace {
void highWater(std::atomic<uint64_t>& peak,uint64_t bytes) {
    auto old=peak.load(std::memory_order_relaxed);
    while(old<bytes&&!peak.compare_exchange_weak(old,bytes,std::memory_order_relaxed)){}
}
}
void HostCacheUsage::add(uint64_t n){highWater(peak,bytes.fetch_add(n,std::memory_order_relaxed)+n);}
void HostCacheUsage::remove(uint64_t n){bytes.fetch_sub(n,std::memory_order_relaxed);}

struct HostBlockCache::Lease::Entry {
    std::vector<char> data;
    std::list<std::string>::iterator age;
    bool loading=true;
};
struct HostBlockCache::State {
    std::mutex mutex;
    std::condition_variable ready;
    std::map<std::string,std::shared_ptr<Lease::Entry>> entries;
    std::list<std::string> lru;
    std::unordered_set<std::string> history;
    std::shared_ptr<HostCacheUsage> usage;
    uint64_t limit;
    bool trackReloads;
    bool reuseStorage;
    std::atomic<uint64_t> bytes{0},peak{0},hits{0},misses{0},reloads{0},sharedWaits{0},spaceWaits{0},evictions{0};
    State(uint64_t n,std::shared_ptr<HostCacheUsage> u,bool trace,bool reuse):usage(std::move(u)),limit(n),trackReloads(trace),reuseStorage(reuse){}
    ~State(){usage->remove(bytes.load());}
};
HostBlockCache::Lease::Lease(std::shared_ptr<State> s,std::shared_ptr<Entry> e):state(std::move(s)),entry(std::move(e)){}
HostBlockCache::Lease::Lease(Lease&&) noexcept=default;
HostBlockCache::Lease& HostBlockCache::Lease::operator=(Lease&& other) noexcept {
    if(this!=&other){release();state=std::move(other.state);entry=std::move(other.entry);}return *this;
}
void HostBlockCache::Lease::release(){
    if(!entry)return;
    {std::lock_guard<std::mutex> lock(state->mutex);entry.reset();}
    state->ready.notify_all();state.reset();
}
HostBlockCache::Lease::~Lease(){release();}
const char* HostBlockCache::Lease::data() const{return entry->data.data();}
size_t HostBlockCache::Lease::size() const{return entry->data.size();}
HostBlockCache::HostBlockCache(uint64_t n,std::shared_ptr<HostCacheUsage> usage,bool trace,bool reuse):state(std::make_shared<State>(n,std::move(usage),trace,reuse)){}
HostBlockCache::~HostBlockCache()=default;
std::optional<HostBlockCache::Lease> HostBlockCache::tryGet(const std::string& key,size_t bytes) {
    auto s=state;std::lock_guard<std::mutex> lock(s->mutex);auto it=s->entries.find(key);
    if(it==s->entries.end()||it->second->loading)return std::nullopt;
    if(it->second->data.size()!=bytes)throw std::runtime_error("Host cache key changed size");
    ++s->hits;s->lru.splice(s->lru.end(),s->lru,it->second->age);
    return Lease(s,it->second);
}
HostBlockCache::Lease HostBlockCache::get(const std::string& key,size_t bytes,const std::function<void(char*)>& fill) {
    auto s=state;std::unique_lock<std::mutex> lock(s->mutex);
    std::vector<char> reusable;
    for(;;) {
        if(bytes>s->limit)throw std::runtime_error("Texture mip exceeds the host workspace budget");
        auto it=s->entries.find(key);
        if(it!=s->entries.end()) {
            if(it->second->loading){++s->sharedWaits;s->ready.wait(lock);continue;}
            if(it->second->data.size()!=bytes)throw std::runtime_error("Host cache key changed size");
            ++s->hits;s->lru.splice(s->lru.end(),s->lru,it->second->age);return Lease(s,it->second);
        }
        if(s->bytes+bytes>s->limit) {
            auto oldest=s->entries.end();
            for(auto age=s->lru.begin();age!=s->lru.end();++age) {
                auto i=s->entries.find(*age);
                if(!i->second->loading&&i->second.use_count()==1){oldest=i;break;}
            }
            if(oldest==s->entries.end()){++s->spaceWaits;s->ready.wait(lock);continue;}
            const auto n=oldest->second->data.size();
            if(s->reuseStorage && n==bytes)reusable=std::move(oldest->second->data);
            s->lru.erase(oldest->second->age);s->entries.erase(oldest);
            s->bytes-=n;s->usage->remove(n);++s->evictions;continue;
        }
        auto entry=std::make_shared<Lease::Entry>();entry->data=std::move(reusable);entry->data.resize(bytes);
        s->lru.push_back(key);entry->age=std::prev(s->lru.end());
        s->entries.emplace(key,entry);++s->misses;
        if(s->trackReloads){if(!s->history.insert(key).second)++s->reloads;}
        s->bytes+=bytes;highWater(s->peak,s->bytes.load());s->usage->add(bytes);lock.unlock();
        try{fill(entry->data.data());}
        catch(...) {
            lock.lock();s->lru.erase(entry->age);s->entries.erase(key);s->bytes-=bytes;s->usage->remove(bytes);
            s->ready.notify_all();throw;
        }
        lock.lock();entry->loading=false;s->ready.notify_all();return Lease(s,std::move(entry));
    }
}
void HostBlockCache::setCapacity(uint64_t bytes){
    std::lock_guard<std::mutex> lock(state->mutex);
    if(!state->entries.empty())throw std::logic_error("Register host workspace sizes before loading textures");
    state->limit=bytes;
}
uint64_t HostBlockCache::trimCapacity(uint64_t target){
    auto& s=*state;std::lock_guard<std::mutex> lock(s.mutex);
    while(s.bytes>target){
        auto victim=s.entries.end();
        for(auto age=s.lru.begin();age!=s.lru.end();++age){
            auto it=s.entries.find(*age);
            if(!it->second->loading&&it->second.use_count()==1){victim=it;break;}
        }
        if(victim==s.entries.end())break;
        const auto n=victim->second->data.size();s.lru.erase(victim->second->age);s.entries.erase(victim);
        s.bytes-=n;s.usage->remove(n);++s.evictions;
    }
    s.limit=std::max(target,s.bytes.load());s.ready.notify_all();return s.limit;
}
void HostBlockCache::clear(){
    auto& s=*state;std::lock_guard<std::mutex> lock(s.mutex);
    for(const auto& e:s.entries)if(e.second->loading||e.second.use_count()!=1)
        throw std::logic_error("Drain texture readers before clearing their workspace");
    s.usage->remove(s.bytes.load());s.entries.clear();s.lru.clear();s.history.clear();
    s.bytes=0;s.peak=0;s.hits=0;s.misses=0;s.reloads=0;s.sharedWaits=0;s.spaceWaits=0;s.evictions=0;
}
HostBlockStats HostBlockCache::stats() const {
    auto& s=*state;return {s.bytes.load(),s.peak.load(),s.hits.load(),s.misses.load(),s.reloads.load(),
        s.sharedWaits.load(),s.spaceWaits.load(),s.evictions.load()};
}

bool HostTileKey::operator==(const HostTileKey& b) const {
    return image==b.image&&level==b.level&&x==b.x&&y==b.y&&width==b.width&&height==b.height&&blockBytes==b.blockBytes;
}
namespace {
struct TileHash {
    size_t operator()(const HostTileKey& k) const {
        uint64_t h=14695981039346656037ull;
        for(auto n:{k.image,k.level,k.x,k.y,k.width,k.height,k.blockBytes}){h^=n;h*=1099511628211ull;}
        return size_t(h);
    }
};
size_t tileBytes(const HostTileKey& k){
    if(!k.width||!k.height||k.width%4||k.height%4||(k.blockBytes!=8&&k.blockBytes!=16))
        throw std::runtime_error("Invalid host BC tile dimensions");
    const uint64_t bytes=uint64_t(k.width/4)*(k.height/4)*k.blockBytes;
    if(bytes>(1u<<20))throw std::runtime_error("Host BC tile exceeds 1 MiB");
    return size_t(bytes);
}
void extract(const HostTileKey& k,const BcMipView& m,char* out){
    const auto width=(uint64_t(m.width)+3)/4,height=(uint64_t(m.height)+3)/4;
    const auto tw=k.width/4,th=k.height/4;
    const uint64_t x=uint64_t(k.x)*tw,y=uint64_t(k.y)*th;
    std::memset(out,0,tileBytes(k));
    if(x<width&&y<height)for(uint64_t row=0;row<std::min(uint64_t(th),height-y);++row)
        std::memcpy(out+row*tw*k.blockBytes,m.data+((y+row)*width+x)*k.blockBytes,
                    size_t(std::min(uint64_t(tw),width-x))*k.blockBytes);
}
}
struct HostTileCache::Impl {
    enum Queue {Neighbor,Recent,Hot};
    struct Entry {std::vector<char> data;std::list<HostTileKey>::iterator age;Queue queue;};
    std::mutex mutex;
    std::unordered_map<HostTileKey,Entry,TileHash> entries;
    std::list<HostTileKey> queues[3];
    std::shared_ptr<HostCacheUsage> usage;
    uint64_t limit,hotBytes=0;
    std::atomic<uint64_t> bytes{0},peak{0},hits{0},misses{0},demandFills{0},prefetched{0},prefetchHits{0},
        evictions{0},prefetchEvictions{0},prefetchSkipped{0},neighborBytes{0};
    Impl(uint64_t n,std::shared_ptr<HostCacheUsage> u):usage(std::move(u)),limit(n){}
    ~Impl(){usage->remove(bytes.load());}
    void move(Entry& e,Queue queue){
        if(e.queue==Neighbor)neighborBytes-=e.data.size();
        if(e.queue==Hot)hotBytes-=e.data.size();
        queues[queue].splice(queues[queue].end(),queues[e.queue],e.age);e.queue=queue;
        if(queue==Neighbor)neighborBytes+=e.data.size();
        if(queue==Hot)hotBytes+=e.data.size();
        while(hotBytes>limit/4*3&&!queues[Hot].empty()){
            const auto old=queues[Hot].front();auto& item=entries.at(old);
            hotBytes-=item.data.size();queues[Recent].splice(queues[Recent].end(),queues[Hot],item.age);item.queue=Recent;
        }
    }
    std::vector<char> evict(Queue queue){
        const auto key=queues[queue].front();auto it=entries.find(key);auto data=std::move(it->second.data);
        queues[queue].pop_front();entries.erase(it);bytes-=data.size();usage->remove(data.size());++evictions;
        if(queue==Neighbor){neighborBytes-=data.size();++prefetchEvictions;}
        if(queue==Hot)hotBytes-=data.size();
        return data;
    }
    void store(const HostTileKey& key,const BcMipView& mip,bool demand){
        const auto n=tileBytes(key);std::lock_guard<std::mutex> lock(mutex);
        auto it=entries.find(key);
        if(it!=entries.end()){
            if(demand)move(it->second,it->second.queue==Neighbor?Recent:Hot);return;
        }
        if(n>limit||(!demand&&n>limit/8)){if(!demand)++prefetchSkipped;return;}
        std::vector<char> storage;
        while((bytes+n>limit)||(!demand&&neighborBytes+n>limit/8)){
            Queue victim=Neighbor;
            if(queues[victim].empty()){
                if(!demand){++prefetchSkipped;return;}
                victim=queues[Recent].empty()?Hot:Recent;
            }
            storage=evict(victim);
        }
        storage.resize(n);extract(key,mip,storage.data());
        const auto queue=demand?Recent:Neighbor;queues[queue].push_back(key);
        entries.emplace(key,Entry{std::move(storage),std::prev(queues[queue].end()),queue});
        bytes+=n;usage->add(n);highWater(peak,bytes.load());
        if(demand)++demandFills;else {++prefetched;neighborBytes+=n;}
    }
};
HostTileCache::HostTileCache(uint64_t n,std::shared_ptr<HostCacheUsage> usage):impl(std::make_unique<Impl>(n,std::move(usage))){}
HostTileCache::~HostTileCache()=default;
bool HostTileCache::read(const HostTileKey& key,char* destination){
    auto& s=*impl;std::lock_guard<std::mutex> lock(s.mutex);const auto it=s.entries.find(key);
    if(it==s.entries.end()){++s.misses;return false;}
    ++s.hits;const auto queue=it->second.queue;
    if(queue==Impl::Neighbor)++s.prefetchHits;
    s.move(it->second,queue==Impl::Neighbor?Impl::Recent:Impl::Hot);
    std::memcpy(destination,it->second.data.data(),it->second.data.size());return true;
}
void copyBcTile(const HostTileKey& key,const BcMipView& mip,char* destination){
    tileBytes(key);
    if(!mip.data||!mip.width||!mip.height||mip.blockBytes!=key.blockBytes||
       uint64_t((uint64_t(mip.width)+3)/4)*((uint64_t(mip.height)+3)/4)*mip.blockBytes!=mip.bytes)
        throw std::runtime_error("Host BC mip view has an invalid size");
    extract(key,mip,destination);
}
void HostTileCache::fill(const HostTileKey& key,const BcMipView& mip,char* destination){
    copyBcTile(key,mip,destination);impl->store(key,mip,true);
    const auto nx=(uint64_t(mip.width)+key.width-1)/key.width,ny=(uint64_t(mip.height)+key.height-1)/key.height;
    for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){
        if(!x&&!y)continue;const int64_t tx=int64_t(key.x)+x,ty=int64_t(key.y)+y;
        if(tx<0||ty<0||uint64_t(tx)>=nx||uint64_t(ty)>=ny)continue;
        auto next=key;next.x=uint32_t(tx);next.y=uint32_t(ty);impl->store(next,mip,false);
    }
}
void HostTileCache::setCapacity(uint64_t bytes){
    auto& s=*impl;std::lock_guard<std::mutex> lock(s.mutex);
    if(bytes<s.limit&&!s.entries.empty())throw std::logic_error("Clear host tiles before shrinking their capacity");s.limit=bytes;
}
void HostTileCache::clear(){
    auto& s=*impl;std::lock_guard<std::mutex> lock(s.mutex);s.usage->remove(s.bytes.load());s.entries.clear();
    for(auto& q:s.queues)q.clear();s.hotBytes=0;s.bytes=0;s.peak=0;s.hits=0;s.misses=0;s.demandFills=0;s.prefetched=0;
    s.prefetchHits=0;s.evictions=0;s.prefetchEvictions=0;s.prefetchSkipped=0;s.neighborBytes=0;
}
HostTileStats HostTileCache::stats() const {
    auto& s=*impl;return {s.bytes.load(),s.peak.load(),s.hits.load(),s.misses.load(),s.demandFills.load(),s.prefetched.load(),
        s.prefetchHits.load(),s.evictions.load(),s.prefetchEvictions.load(),s.prefetchSkipped.load(),s.neighborBytes.load()};
}
}
