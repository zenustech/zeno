#pragma once
#include <zeno/utils/api.h>
#include <filesystem>

namespace zeno {
// Returns an immutable baked manifest. Raw glTF preparation runs only when its
// source dependency stamp or the explicit bake revision changes.
ZENO_API std::filesystem::path prepareMegaGeometryAsset(const std::filesystem::path& source,
    const std::filesystem::path& cacheDirectory,const std::filesystem::path& baker,
    unsigned jobs=2,unsigned revision=0);
}
