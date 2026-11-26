#pragma once
#include <cstdint>
#include <filesystem>
namespace zeno::cluster {
// Extract BC block rows without allocating a complete decoded mip. Zstd input
// remains sequential; the caller's bounded RAM cache shares/reuses tile fills.
struct TextureBlockReadStats {uint64_t sourceBytes{},decodedBytes{};};
TextureBlockReadStats readTextureBlockTile(const std::filesystem::path&,uint64_t offset,uint64_t encoded,uint64_t decoded,
    unsigned int scheme,unsigned int width,unsigned int height,unsigned int blockBytes,
    unsigned int tileX,unsigned int tileY,unsigned int tileWidth,unsigned int tileHeight,
    uint64_t windowBudget,char* destination);
}
