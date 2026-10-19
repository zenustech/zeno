#pragma once
#include "ClusterCache.h"

namespace zeno::cluster {
// Read-only adapter for RTXMG's little-endian, uncompressed v19 shard layout.
// The returned metadata points at the original shard; it contains no payload
// copy.
nlohmann::json indexRtxmgShard(const std::filesystem::path &path, uint64_t sourceHash,
                               uint64_t configHash);
void validateRtxmgShard(const std::filesystem::path &path, uint64_t sourceHash,
                        uint64_t configHash);
} // namespace zeno::cluster
