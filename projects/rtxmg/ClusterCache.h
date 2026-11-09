#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include <tinygltf/json.hpp>
#include "ClusterDevice.h"

namespace zeno::cluster {
constexpr uint32_t CacheVersion = 1;
constexpr uint32_t PageMagic = 0x31474d5a; // ZMG1, little endian.
constexpr uint32_t MaxVertices = 128;
constexpr uint32_t MaxTriangles = 128;
struct Vertex { float position[3], normal[3]; };
static_assert(sizeof(Vertex) == 24);
struct Cluster {
    uint32_t vertexOffset, indexOffset;
    uint16_t vertexCount, triangleCount;
    int32_t refined;
};
static_assert(sizeof(Cluster) == 16);
struct PageHeader {
    uint32_t magic, version, clusterCount, vertexCount;
    uint32_t indexBytes, encodedVertexBytes;
};
struct Page {
    std::vector<Cluster> clusters;
    std::vector<Vertex> vertices;
    std::vector<uint8_t> indices;
    std::vector<zeno_cluster_device::ShadingVertex> attributes;
    std::vector<uint16_t> materials;
};
// Reader-owned scratch. Reusing it and Page capacities avoids large process-
// heap allocation/free bursts on the asynchronous streaming path.
struct Geometry;
struct PageReadWorkspace {
    std::vector<uint8_t> body, compressed, packed;
    PageReadWorkspace();
    ~PageReadWorkspace();
    PageReadWorkspace(PageReadWorkspace&&) noexcept;
    PageReadWorkspace& operator=(PageReadWorkspace&&) noexcept;
    PageReadWorkspace(const PageReadWorkspace&) = delete;
    PageReadWorkspace& operator=(const PageReadWorkspace&) = delete;
private:
    struct Decoder;
    std::unique_ptr<Decoder> decoder;
    friend void readPage(const Geometry&, uint32_t, std::istream&, Page&, PageReadWorkspace&);
};
struct Group {
    uint64_t offset{}, bytes{};
    uint32_t vertices{}, triangles{}, depth{};
    std::array<float, 5> bounds{};
    std::vector<int32_t> refined;
};
struct Geometry {
    std::filesystem::path pages;
    bool rtxmgShard = false;
    bool hasAttributes = false;
    std::vector<Group> groups;
    std::vector<uint32_t> roots;
};
struct Placement {
    uint32_t geometry{}, sourceNode{};
    std::array<float, 16> transform{}; // glTF column-major affine matrix.
};
struct Asset {
    std::filesystem::path manifest;
    std::vector<Geometry> geometries;
    std::vector<Placement> placements;
};
nlohmann::json readJson(const std::filesystem::path& path);
void writeJson(const std::filesystem::path& path, const nlohmann::json& value);
Asset loadAsset(const std::filesystem::path& manifest);
Page readPage(const Geometry& geometry, uint32_t group);
Page readPage(const Geometry& geometry, uint32_t group, std::istream& stream);
void readPage(const Geometry& geometry, uint32_t group, std::istream& stream, Page& page, PageReadWorkspace& workspace);
Page readRtxmgPage(const Geometry& geometry, uint32_t group, std::istream& stream);
void readRtxmgPage(const Geometry& geometry, uint32_t group, std::istream& stream, Page& page, std::vector<uint8_t>& bytes);
uint64_t writePage(std::ostream& stream, const Page& page);
}
