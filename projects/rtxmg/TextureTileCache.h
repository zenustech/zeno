#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace zeno::cluster::texturecache {
constexpr uint32_t TileSide = 256;
struct Source {
    uint32_t format{}, width{}, height{}, levels{}, blockBytes{};
};
struct ReadStats { uint64_t bytes{}, readUs{}, decodeUs{}; };
struct BakeStats { uint64_t sourceBytes{}, decodedBytes{}, tiles{}, fileBytes{}; };
Source inspect(const std::filesystem::path& source);
uint64_t estimate(const Source&,const std::vector<uint32_t>& levels);
std::filesystem::path cachePath(const std::filesystem::path& root,const std::filesystem::path& source);
// Separate disposable cache. Original BC blocks, mip levels and metadata stay
// unchanged. Publication replaces only this source's cache file atomically.
BakeStats bake(const std::filesystem::path& source,const std::filesystem::path& root,
               const std::vector<uint32_t>& levels);
class Reader {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit Reader(std::unique_ptr<Impl>);
  public:
    ~Reader();
    // Missing, stale or malformed caches fall back to the original KTX path.
    static std::unique_ptr<Reader> open(const std::filesystem::path& root,
                                      const std::filesystem::path& source,const Source& expected);
    bool contains(uint32_t level) const;
    uint32_t tileBytes() const;
    ReadStats read(uint32_t level,uint32_t x,uint32_t y,char* output) const;
};
} // namespace zeno::cluster::texturecache
