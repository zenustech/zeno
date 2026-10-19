#include "ClusterCache.h"
#include "ClusterRtxmg.h"
#include <meshoptimizer.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <zstd.h>

namespace zeno::cluster {
namespace {
struct PackedVertex {float position[3];int16_t normal[2];};
static_assert(sizeof(PackedVertex)==16);
void packNormal(int16_t* out,const float* n){
    for(int i=0;i<3;++i)if(!std::isfinite(n[i]))throw std::runtime_error("Non-finite cluster normal");
    const float length=std::abs(n[0])+std::abs(n[1])+std::abs(n[2]);
    if(!std::isfinite(length))throw std::runtime_error("Invalid cluster normal magnitude");
    if(length==0){out[0]=out[1]=INT16_MIN;return;}
    float x=n[0]/length,y=n[1]/length;
    if(n[2]<0){const float oldX=x;x=std::copysign(1.f-std::abs(y),oldX);y=std::copysign(1.f-std::abs(oldX),y);}
    out[0]=int16_t(std::lround(std::clamp(x,-1.f,1.f)*32767.f));
    out[1]=int16_t(std::lround(std::clamp(y,-1.f,1.f)*32767.f));
}
void unpackNormal(float* out,const int16_t* n){
    if(n[0]==INT16_MIN&&n[1]==INT16_MIN){out[0]=out[1]=out[2]=0;return;}
    if(n[0]==INT16_MIN||n[1]==INT16_MIN)throw std::runtime_error("Invalid packed cluster normal");
    float x=float(n[0])/32767.f,y=float(n[1])/32767.f,z=1.f-std::abs(x)-std::abs(y);
    const float t=std::clamp(-z,0.f,1.f);x-=std::copysign(t,x);y-=std::copysign(t,y);
    const float length=std::sqrt(x*x+y*y+z*z);out[0]=x/length;out[1]=y/length;out[2]=z/length;
}
}
nlohmann::json readJson(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open cluster metadata: " + path.string());
    nlohmann::json value; stream >> value; return value;
}
void writeJson(const std::filesystem::path& path, const nlohmann::json& value) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << value.dump() << '\n';
    if (!stream) throw std::runtime_error("Cannot write cluster metadata: " + path.string());
}
Asset loadAsset(const std::filesystem::path& manifest) {
    const auto meta = readJson(manifest);
    if (meta.value("schema", "") != "zeno.rtxmg.asset" || meta.value("version", 0u) != CacheVersion)
        throw std::runtime_error("Unsupported prebaked cluster asset schema");
    Asset asset; asset.manifest = std::filesystem::absolute(manifest);
    for (const auto& entry : meta.at("meshes")) {
        const auto mesh = readJson(asset.manifest.parent_path() / entry.at("metadata").get<std::string>());
        if (mesh.at("version").get<uint32_t>() != CacheVersion) throw std::runtime_error("Cluster mesh version mismatch");
        Geometry geometry;
        geometry.rtxmgShard = mesh.value("format", "") == "rtxmg.shard.v19";
        geometry.pages = asset.manifest.parent_path() / mesh.at("pages").get<std::string>();
        if (geometry.rtxmgShard) validateRtxmgShard(geometry.pages, mesh.at("sourceHash").get<uint64_t>(), mesh.at("configHash").get<uint64_t>());
        const auto fileBytes = std::filesystem::file_size(geometry.pages);
        if (geometry.rtxmgShard && mesh.at("pageBytes").get<uint64_t>() != fileBytes)
            throw std::runtime_error("RTXMG shard size changed; regenerate the scene index");
        for (const auto& group : mesh.at("groups")) {
            Group g;
            g.offset = group.at("offset").get<uint64_t>(); g.bytes = group.at("bytes").get<uint64_t>();
            g.vertices = group.at("vertices").get<uint32_t>(); g.triangles = group.at("triangles").get<uint32_t>();
            g.depth = group.at("depth").get<uint32_t>();
            g.bounds = group.at("bounds").get<std::array<float, 5>>();
            g.refined = group.at("refined").get<std::vector<int32_t>>();
            if (g.offset > fileBytes || g.bytes > fileBytes - g.offset || g.bytes < sizeof(PageHeader))
                throw std::runtime_error("Cluster group file range is invalid");
            for (float v : g.bounds) if (!std::isfinite(v)) throw std::runtime_error("Invalid cluster bounds");
            geometry.groups.push_back(std::move(g));
        }
        geometry.roots = mesh.at("roots").get<std::vector<uint32_t>>();
        for (uint32_t root : geometry.roots) if (root >= geometry.groups.size()) throw std::runtime_error("Invalid LOD root");
        for (size_t id = 0; id < geometry.groups.size(); ++id)
            for (int32_t child : geometry.groups[id].refined)
                if (child < -1 || child >= int32_t(id)) throw std::runtime_error("Invalid or cyclic LOD refinement");
        asset.geometries.push_back(std::move(geometry));
    }
    for (const auto& node : meta.at("nodes")) {
        Placement p; p.geometry = node.at("mesh").get<uint32_t>(); p.sourceNode = node.at("sourceNode").get<uint32_t>();
        p.transform = node.at("matrix").get<std::array<float, 16>>();
        if (p.geometry >= asset.geometries.size()) throw std::runtime_error("Invalid cluster placement");
        for (float v : p.transform) if (!std::isfinite(v)) throw std::runtime_error("Invalid placement transform");
        asset.placements.push_back(p);
    }
    return asset;
}
uint64_t writePage(std::ostream& stream, const Page& page) {
    // Preserve positions exactly; store unit normals as signed 16-bit octahedral
    // coordinates rather than duplicating three full floats per cluster vertex.
    std::vector<PackedVertex> packed(page.vertices.size());
    for(size_t i=0;i<packed.size();++i){std::memcpy(packed[i].position,page.vertices[i].position,12);packNormal(packed[i].normal,page.vertices[i].normal);}
    std::vector<uint8_t> encoded(meshopt_encodeVertexBufferBound(packed.size(), sizeof(PackedVertex)));
    const auto count = meshopt_encodeVertexBuffer(encoded.data(), encoded.size(), packed.data(), packed.size(), sizeof(PackedVertex));
    if (!count || count > UINT32_MAX) throw std::runtime_error("Cluster vertex encoding failed");
    PageHeader h{PageMagic, 3, uint32_t(page.clusters.size()), uint32_t(page.vertices.size()),
                 uint32_t(page.indices.size()), uint32_t(count)};
    const size_t clusterBytes=page.clusters.size()*sizeof(Cluster);
    std::vector<uint8_t> body(clusterBytes+count+page.indices.size());
    std::memcpy(body.data(),page.clusters.data(),clusterBytes);
    std::memcpy(body.data()+clusterBytes,encoded.data(),count);
    std::memcpy(body.data()+clusterBytes+count,page.indices.data(),page.indices.size());
    std::vector<uint8_t> compressed(ZSTD_compressBound(body.size()));
    const auto compressedBytes=ZSTD_compress(compressed.data(),compressed.size(),body.data(),body.size(),1);
    if(ZSTD_isError(compressedBytes))throw std::runtime_error("Cluster page compression failed");
    if(compressedBytes<body.size()) {h.version=4;compressed.resize(compressedBytes);body=std::move(compressed);}
    stream.write(reinterpret_cast<const char*>(&h), sizeof(h));
    stream.write(reinterpret_cast<const char*>(body.data()),body.size());
    if (!stream) throw std::runtime_error("Cluster page write failed");
    return sizeof(h)+body.size();
}
Page readPage(const Geometry& geometry, uint32_t group) {
    std::ifstream stream(geometry.pages, std::ios::binary);
    return readPage(geometry,group,stream);
}
Page readPage(const Geometry& geometry,uint32_t group,std::istream& stream){
    if(geometry.rtxmgShard)return readRtxmgPage(geometry,group,stream);
    const auto& meta = geometry.groups.at(group);
    stream.clear();
    stream.seekg(meta.offset);
    PageHeader h{}; stream.read(reinterpret_cast<char*>(&h), sizeof(h));
    const uint64_t bytes = sizeof(h) + uint64_t(h.clusterCount) * sizeof(Cluster) + h.encodedVertexBytes + h.indexBytes;
    const bool compressed=h.version==2||h.version==4;
    if (!stream || h.magic != PageMagic || h.version<1 || h.version>4 ||
        (!compressed && bytes != meta.bytes) || bytes>4u*1024u*1024u ||
        (compressed && (meta.bytes<=sizeof(h) || meta.bytes>=bytes)) ||
        h.clusterCount != meta.refined.size() || h.vertexCount != meta.vertices || h.indexBytes != uint64_t(meta.triangles) * 3 ||
        !h.clusterCount || !h.vertexCount || !h.encodedVertexBytes ||
        h.clusterCount > 256 || h.vertexCount > h.clusterCount * MaxVertices)
        throw std::runtime_error("Invalid cluster page header");
    Page p; p.clusters.resize(h.clusterCount); p.vertices.resize(h.vertexCount); p.indices.resize(h.indexBytes);
    std::vector<uint8_t> body(bytes-sizeof(h));
    if(compressed) {
        std::vector<uint8_t> compressed(meta.bytes-sizeof(h));
        stream.read(reinterpret_cast<char*>(compressed.data()),compressed.size());
        const auto decoded=ZSTD_decompress(body.data(),body.size(),compressed.data(),compressed.size());
        if(ZSTD_isError(decoded)||decoded!=body.size())throw std::runtime_error("Cluster page decompression failed");
    }else stream.read(reinterpret_cast<char*>(body.data()),body.size());
    const size_t clusterBytes=p.clusters.size()*sizeof(Cluster);
    std::memcpy(p.clusters.data(),body.data(),clusterBytes);
    std::memcpy(p.indices.data(),body.data()+clusterBytes+h.encodedVertexBytes,p.indices.size());
    if(h.version>=3){
        std::vector<PackedVertex> packed(h.vertexCount);
        if(!stream||meshopt_decodeVertexBuffer(packed.data(),packed.size(),sizeof(PackedVertex),body.data()+clusterBytes,h.encodedVertexBytes))
            throw std::runtime_error("Cluster page decode failed");
        for(size_t i=0;i<packed.size();++i){std::memcpy(p.vertices[i].position,packed[i].position,12);unpackNormal(p.vertices[i].normal,packed[i].normal);}
    }else if (!stream || meshopt_decodeVertexBuffer(p.vertices.data(), p.vertices.size(), sizeof(Vertex), body.data()+clusterBytes,h.encodedVertexBytes))
        throw std::runtime_error("Cluster page decode failed");
    for (size_t i = 0; i < p.clusters.size(); ++i) {
        const auto& c = p.clusters[i];
        if (!c.vertexCount || c.vertexCount > MaxVertices || !c.triangleCount || c.triangleCount > MaxTriangles ||
            uint64_t(c.vertexOffset) + c.vertexCount > p.vertices.size() || uint64_t(c.indexOffset) + c.triangleCount * 3 > p.indices.size() ||
            c.refined != meta.refined[i]) throw std::runtime_error("Invalid cluster page geometry range");
        for (size_t j = c.indexOffset; j < c.indexOffset + c.triangleCount * 3; ++j)
            if (p.indices[j] >= c.vertexCount) throw std::runtime_error("Cluster triangle index out of range");
    }
    for (const auto& v : p.vertices) {
        for (float x : v.position) if (!std::isfinite(x)) throw std::runtime_error("Non-finite cluster vertex");
        for (float x : v.normal) if (!std::isfinite(x)) throw std::runtime_error("Non-finite cluster normal");
    }
    return p;
}
}
