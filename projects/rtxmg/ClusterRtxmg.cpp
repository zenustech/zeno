#include "ClusterRtxmg.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zeno::cluster {
namespace {
constexpr uint64_t ShardMagic = 0x006f65676e73766eull;
struct Bytes {
    const uint8_t *data;
    uint64_t size;
    void range(uint64_t offset, uint64_t count) const {
        if (offset > size || count > size - offset)
            throw std::runtime_error("RTXMG shard range outside file/group");
    }
    template <class T> T at(uint64_t offset) const {
        range(offset, sizeof(T));
        T result;
        std::memcpy(&result, data + offset, sizeof(T));
        return result;
    }
};
// Mapping faults only the metadata pages touched by the importer, rather than
// reading the scene's entire geometry payload into host memory.
class Mapping {
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
#else
    std::vector<uint8_t> storage;
#endif
  public:
    Bytes bytes{nullptr, 0};
    explicit Mapping(const std::filesystem::path &path) {
#ifdef _WIN32
        file = CreateFileW(path.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Cannot open RTXMG shard: " + path.string());
        LARGE_INTEGER size{};
        if (GetFileSizeEx(file, &size) && size.QuadPart > 0)
            mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping)
            bytes.data =
                static_cast<const uint8_t *>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
        if (!bytes.data) {
            if (mapping)
                CloseHandle(mapping);
            CloseHandle(file);
            throw std::runtime_error("Cannot map RTXMG shard: " + path.string());
        }
        bytes.size = uint64_t(size.QuadPart);
#else
        std::ifstream stream(path, std::ios::binary);
        storage.resize(std::filesystem::file_size(path));
        stream.read(reinterpret_cast<char *>(storage.data()), storage.size());
        if (!stream)
            throw std::runtime_error("Cannot read RTXMG shard: " + path.string());
        bytes = {storage.data(), storage.size()};
#endif
    }
    ~Mapping() {
#ifdef _WIN32
        UnmapViewOfFile(bytes.data);
        CloseHandle(mapping);
        CloseHandle(file);
#endif
    }
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
};
void header(Bytes b, uint64_t size, uint64_t sourceHash, uint64_t configHash) {
    if (b.at<uint64_t>(0) != ShardMagic || b.at<uint32_t>(8) != 19 || b.at<uint32_t>(12) != 208 ||
        b.at<uint32_t>(16) != 3 || b.at<uint64_t>(24) != 16)
        throw std::runtime_error("Unsupported RTXMG shard layout (requires v19, "
                                 "config v3, 64-bit layout)");
    if (b.at<uint64_t>(32) != configHash || b.at<uint64_t>(40) != sourceHash || size < 80 ||
        b.at<uint64_t>(64) != size - 80)
        throw std::runtime_error(
            "RTXMG shard identity or size changed; regenerate the scene index");
}
uint64_t align16(uint64_t n) { return (n + 15) & ~uint64_t(15); }
struct Span {
    uint64_t offset, count;
};
Span span(Bytes b, uint64_t &cursor, uint64_t stride) {
    const auto count = b.at<uint64_t>(cursor);
    cursor += 16;
    if (count > b.size / stride)
        throw std::runtime_error("Invalid RTXMG span count");
    b.range(cursor, count * stride);
    Span result{cursor, count};
    cursor += align16(count * stride);
    return result;
}
void normal(float *out, uint32_t packed) {
    float x = float(packed & 2047) * (2.f / 2047.f) - 1.f;
    float y = float((packed >> 11) & 2047) * (2.f / 2047.f) - 1.f;
    const float z = 1.f - std::abs(x) - std::abs(y);
    if (z < 0) {
        const float oldX = x;
        x = (1.f - std::abs(y)) * (oldX >= 0 ? 1.f : -1.f);
        y = (1.f - std::abs(oldX)) * (y >= 0 ? 1.f : -1.f);
    }
    const float invLength = 1.f / std::sqrt(x * x + y * y + z * z);
    out[0] = x * invLength;
    out[1] = y * invLength;
    out[2] = z * invLength;
}
} // namespace

void validateRtxmgShard(const std::filesystem::path &path, uint64_t sourceHash,
                        uint64_t configHash) {
    std::array<uint8_t, 80> data{};
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(data.data()), data.size());
    if (!stream)
        throw std::runtime_error("Cannot read RTXMG shard header: " + path.string());
    header({data.data(), data.size()}, std::filesystem::file_size(path), sourceHash, configHash);
}

nlohmann::json indexRtxmgShard(const std::filesystem::path &path, uint64_t sourceHash,
                               uint64_t configHash) {
    Mapping mapping(path);
    const Bytes b = mapping.bytes;
    header(b, b.size, sourceHash, configHash);
    if (b.at<uint32_t>(84) > MaxVertices || b.at<uint32_t>(88) > MaxTriangles)
        throw std::runtime_error("RTXMG clusters exceed OptiX integration limits");
    uint64_t cursor = 160; // ShardHeader(80) + aligned GeometryBase(76).
    const auto data = span(b, cursor, 1), infos = span(b, cursor, 32), levels = span(b, cursor, 24);
    if (!infos.count || !levels.count || levels.count != b.at<uint32_t>(92) ||
        infos.count > INT32_MAX)
        throw std::runtime_error("Invalid RTXMG geometry counts");
    nlohmann::json groups = nlohmann::json::array(), roots = nlohmann::json::array();
    for (uint32_t id = 0; id < infos.count; ++id) {
        const uint64_t info = infos.offset + id * 32, bits = b.at<uint64_t>(info);
        const uint64_t offset = bits & ((1ull << 42) - 1), size = bits >> 42;
        const uint32_t count = b.at<uint8_t>(info + 13);
        if (b.at<uint64_t>(info + 24) >> 42)
            throw std::runtime_error("Compressed RTXMG shards are not supported by this reader");
        if (!count || offset > data.count || size > data.count - offset)
            throw std::runtime_error("Invalid RTXMG group range");
        const Bytes group{b.data + data.offset + offset, size};
        if ((group.at<uint32_t>(8) >> 16) != count)
            throw std::runtime_error("RTXMG group cluster count mismatch");
        std::array<float, 5> bounds;
        for (int i = 0; i < 5; ++i) {
            bounds[i] = group.at<float>(12 + i * 4);
            if (!std::isfinite(bounds[i]))
                throw std::runtime_error("Invalid RTXMG LOD metric");
        }
        std::vector<int32_t> refined;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t child = group.at<int32_t>(32 + count * 16 + i * 4);
            if (child < -1 || child >= int32_t(id))
                throw std::runtime_error("Invalid RTXMG refinement DAG");
            refined.push_back(child);
        }
        groups.push_back({{"offset", data.offset + offset},
                          {"bytes", size},
                          {"vertices", b.at<uint16_t>(info + 8)},
                          {"triangles", b.at<uint16_t>(info + 10)},
                          {"depth", b.at<uint8_t>(info + 12)},
                          {"bounds", bounds},
                          {"refined", refined}});
    }
    // Use all unreferenced groups as roots. A disconnected component can have
    // terminated before the final LOD level; it must remain visible as well.
    std::vector<bool> referenced(infos.count);
    for (const auto &group : groups)
        for (int32_t child : group.at("refined"))
            if (child >= 0)
                referenced[child] = true;
    for (uint32_t id = 0; id < infos.count; ++id)
        if (!referenced[id])
            roots.push_back(id);
    return {{"version", CacheVersion},
            {"format", "rtxmg.shard.v19"},
            {"pages", std::filesystem::absolute(path).string()},
            {"pageBytes", b.size},
            {"sourceHash", sourceHash},
            {"configHash", configHash},
            {"sourceTriangles", b.at<uint32_t>(96)},
            {"groups", std::move(groups)},
            {"roots", roots}};
}

Page readRtxmgPage(const Geometry &geometry, uint32_t id, std::istream &stream) {
    const auto &meta = geometry.groups.at(id);
    if (meta.bytes > 4u * 1024u * 1024u)
        throw std::runtime_error("RTXMG group exceeds size limit");
    std::vector<uint8_t> data(meta.bytes);
    stream.clear();
    stream.seekg(meta.offset);
    stream.read(reinterpret_cast<char *>(data.data()), data.size());
    if (!stream)
        throw std::runtime_error("RTXMG group read failed");
    const Bytes b{data.data(), data.size()};
    const uint32_t count = b.at<uint32_t>(8) >> 16;
    if (!count || count > 256 || count != meta.refined.size())
        throw std::runtime_error("RTXMG group cluster count changed");
    Page page;
    page.vertices.reserve(meta.vertices);
    page.indices.reserve(meta.triangles * 3);
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t c = 32 + i * 16;
        const uint32_t counts = b.at<uint32_t>(c), attr = b.at<uint32_t>(c + 4) & 255;
        const uint32_t triangles = (counts & 255) + 1, vertices = ((counts >> 8) & 255) + 1;
        if (triangles > MaxTriangles || vertices > MaxVertices || (attr & (16 | 128)))
            throw std::runtime_error("Unsupported RTXMG cluster position layout");
        const uint64_t vp = c + b.at<uint32_t>(c + 8), ip = c + b.at<uint32_t>(c + 12);
        b.range(vp, vertices * ((attr & 1) ? 16 : 12));
        b.range(ip, triangles * 3);
        const int32_t refined = b.at<int32_t>(32 + count * 16 + i * 4);
        if (refined != meta.refined[i])
            throw std::runtime_error("RTXMG refinement data changed");
        const Cluster cluster{uint32_t(page.vertices.size()), uint32_t(page.indices.size()),
                              uint16_t(vertices), uint16_t(triangles), refined};
        page.clusters.push_back(cluster);
        for (uint32_t v = 0; v < vertices; ++v) {
            Vertex vertex{};
            for (int j = 0; j < 3; ++j) {
                vertex.position[j] = b.at<float>(vp + v * 12 + j * 4);
                if (!std::isfinite(vertex.position[j]))
                    throw std::runtime_error("Invalid RTXMG position");
            }
            if (attr & 1)
                normal(vertex.normal, b.at<uint32_t>(vp + vertices * 12 + v * 4));
            page.vertices.push_back(vertex);
        }
        for (uint32_t t = 0; t < triangles * 3; ++t) {
            const uint8_t index = b.at<uint8_t>(ip + t);
            if (index >= vertices)
                throw std::runtime_error("RTXMG triangle index out of range");
            page.indices.push_back(index);
        }
        if (!(attr & 1)) { // Smooth geometric fallback for sources without normals.
            for (uint32_t t = 0; t < triangles; ++t) {
                Vertex *v[3];
                for (int j = 0; j < 3; ++j)
                    v[j] = &page.vertices[cluster.vertexOffset +
                                          page.indices[cluster.indexOffset + t * 3 + j]];
                float a[3], d[3], n[3];
                for (int j = 0; j < 3; ++j) {
                    a[j] = v[1]->position[j] - v[0]->position[j];
                    d[j] = v[2]->position[j] - v[0]->position[j];
                }
                n[0] = a[1] * d[2] - a[2] * d[1];
                n[1] = a[2] * d[0] - a[0] * d[2];
                n[2] = a[0] * d[1] - a[1] * d[0];
                for (auto vertex : v)
                    for (int j = 0; j < 3; ++j)
                        vertex->normal[j] += n[j];
            }
            for (uint32_t v = 0; v < vertices; ++v) {
                auto &n = page.vertices[cluster.vertexOffset + v].normal;
                const float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (length > 0)
                    for (float &value : n)
                        value /= length;
            }
        }
    }
    if (page.vertices.size() != meta.vertices || page.indices.size() != meta.triangles * 3)
        throw std::runtime_error("RTXMG group geometry totals changed");
    return page;
}
} // namespace zeno::cluster
