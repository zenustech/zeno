#include "ClusterRtxmg.h"
#include "ClusterSource.h"
#include <functional>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

using namespace zeno::cluster;
namespace {
std::string decodeUri(const std::string &uri) {
    std::string result;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < uri.size(); ++i) {
        if (uri[i] == '%' && i + 2 < uri.size() && hex(uri[i + 1]) >= 0 && hex(uri[i + 2]) >= 0) {
            const char c = char(hex(uri[i + 1]) * 16 + hex(uri[i + 2]));
            if (!c)
                throw std::runtime_error("NUL in glTF buffer URI");
            result += c;
            i += 2;
        } else
            result += uri[i];
    }
    return result;
}
uint64_t hashBytes(uint64_t h, const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
struct Identity {
    uint64_t key, size;
    int64_t mtime;
};
std::string nativeGeometryKey(const Json &mesh) {
    // RTXMG aliases meshes by their POSITION, NORMAL, index and TEXCOORD_0
    // accessor identities, retaining the first mesh's material assignment.
    // Resolve that representative before looking up its source key in index.bin.
    Json key = Json::array();
    for (const auto &p : mesh.at("primitives"))
        if (p.value("mode", 4) == 4) {
            const auto &a = p.at("attributes");
            key.push_back({a.value("POSITION", -1), a.value("NORMAL", -1), p.value("indices", -1),
                           a.value("TEXCOORD_0", -1)});
        }
    return key.dump();
}
Identity sourceIdentity(const Json &scene, size_t mesh, const fs::path &source) {
    std::set<std::tuple<std::string, uint64_t, uint64_t>> ranges;
    std::set<fs::path> files;
    auto consider = [&](size_t accessor) {
        const auto &a = scene.at("accessors").at(accessor);
        if (!a.contains("bufferView"))
            throw std::runtime_error("Sparse/implicit glTF accessors are not supported");
        const auto &v = scene.at("bufferViews").at(a.at("bufferView").get<size_t>());
        const auto extensions = v.value("extensions", Json::object());
        const auto &range = extensions.contains("EXT_meshopt_compression")
                                ? extensions.at("EXT_meshopt_compression")
                                : v;
        auto uri = scene.at("buffers").at(range.at("buffer").get<size_t>()).value("uri", "");
        if (uri.rfind("data:", 0) == 0)
            uri.clear();
        uri = decodeUri(uri);
        ranges.emplace(uri, range.value("byteOffset", uint64_t(0)),
                       range.at("byteLength").get<uint64_t>());
        files.insert(uri.empty() ? source : source.parent_path() / uri);
    };
    const auto &primitives = scene.at("meshes").at(mesh).at("primitives");
    for (const auto &p : primitives) {
        if (p.contains("indices"))
            consider(p.at("indices"));
        for (auto it = p.at("attributes").begin(); it != p.at("attributes").end(); ++it)
            consider(it.value());
    }
    uint64_t hash = 14695981039346656037ull;
    for (const auto &[uri, offset, size] : ranges) {
        hash = hashBytes(hash, uri.data(), uri.size());
        hash = hashBytes(hash, &offset, sizeof(offset));
        hash = hashBytes(hash, &size, sizeof(size));
    }
    for (const auto &p : primitives) {
        if (p.value("mode", 4) != 4 || p.at("attributes").empty())
            continue;
        const uint32_t material = p.value("material", UINT32_MAX);
        const auto m = material == UINT32_MAX ? Json::object() : scene.at("materials").at(material);
        const std::string alpha = m.value("alphaMode", "OPAQUE");
        const uint32_t mode = alpha == "MASK"    ? 1
                              : alpha == "BLEND" ? 2
                                                 : 0,
                       twoSided = m.value("doubleSided", false);
        hash = hashBytes(hash, &material, 4);
        hash = hashBytes(hash, &mode, 4);
        hash = hashBytes(hash, &twoSided, 4);
    }
    Identity result{hash, 0, 0};
    for (const auto &path : files) {
        result.size += fs::file_size(path);
        result.mtime =
            std::max(result.mtime, int64_t(fs::last_write_time(path).time_since_epoch().count()));
    }
    return result;
}
template <class T> T read(std::istream &stream) {
    T result{};
    stream.read(reinterpret_cast<char *>(&result), sizeof(result));
    if (!stream)
        throw std::runtime_error("Truncated RTXMG index.bin");
    return result;
}
struct Record {
    uint64_t hash, size;
    int64_t mtime;
};
std::string hexHash(uint64_t hash) {
    std::ostringstream s;
    s << std::hex << std::setw(16) << std::setfill('0') << hash;
    return s.str();
}
} // namespace

int main(int argc, char **argv) try {
    fs::path source, cache, output;
    int meshOnly = -1, maxMeshes = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 == argc)
            throw std::runtime_error("Missing import argument");
        if (arg == "--source")
            source = argv[++i];
        else if (arg == "--cache")
            cache = argv[++i];
        else if (arg == "--output")
            output = argv[++i];
        else if (arg == "--mesh")
            meshOnly = std::stoi(argv[++i]);
        else if (arg == "--max-meshes")
            maxMeshes = std::stoi(argv[++i]);
        else
            throw std::runtime_error("Unknown import option: " + arg);
    }
    if (source.empty() || cache.empty() || output.empty())
        throw std::runtime_error("Usage: zeno_cluster_import_rtxmg --source scene.gltf --cache "
                                 "_nvsngeocache --output scene-index-directory [--mesh id] "
                                 "[--max-meshes n]");
    source = fs::absolute(source);
    cache = fs::absolute(cache);
    output = fs::absolute(output);
    // Never place generated metadata into NVIDIA's source/cache directories.
    const auto resolvedOutput = fs::weakly_canonical(output),
               resolvedSource = fs::weakly_canonical(source.parent_path());
    auto src = resolvedSource.begin(), dst = resolvedOutput.begin();
    for (; src != resolvedSource.end() && dst != resolvedOutput.end() && *src == *dst;
         ++src, ++dst) {
    }
    if (src == resolvedSource.end())
        throw std::runtime_error("Scene index output must be outside the source asset directory");
    const auto config = readJson(cache / "bake_config.json");
    if (config.value("useCompressedData", true))
        throw std::runtime_error("Compressed RTXMG shard groups are not supported yet");
    std::ifstream index(cache / "index.bin", std::ios::binary);
    if (read<uint64_t>(index) != 0x736e6764696e6d76ull || read<uint32_t>(index) != 3 ||
        read<uint32_t>(index) != 112)
        throw std::runtime_error("Unsupported RTXMG source index format (requires v3)");
    const auto count = read<uint64_t>(index), configHash = read<uint64_t>(index);
    if (count > (fs::file_size(cache / "index.bin") - 112) / 32)
        throw std::runtime_error("Invalid RTXMG index record count");
    index.seekg(112);
    std::map<uint64_t, Record> records;
    for (uint64_t i = 0; i < count; ++i) {
        const auto key = read<uint64_t>(index), hash = read<uint64_t>(index),
                   size = read<uint64_t>(index);
        const auto mtime = read<int64_t>(index);
        if (!records.emplace(key, Record{hash, size, mtime}).second)
            throw std::runtime_error("Duplicate RTXMG source key");
    }
    const Json scene = readJson(source);
    Source input{scene, source.parent_path(), {}};
    std::map<std::string, size_t> canonical;
    std::vector<size_t> representative(scene.at("meshes").size());
    for (size_t i = 0; i < representative.size(); ++i) {
        auto entry = canonical.emplace(nativeGeometryKey(scene.at("meshes")[i]), i);
        representative[i] = entry.first->second;
    }
    Json asset{{"schema", "zeno.rtxmg.asset"},
               {"version", CacheVersion},
               {"source", source.string()},
               {"nativeCache", cache.string()},
               {"gray", true},
               {"gpuInstancing", true},
               {"meshes", Json::array()},
               {"nodes", Json::array()}};
    std::set<size_t> selected;
    for (const auto &node : scene.at("nodes"))
        if (node.contains("mesh")) {
            const size_t mesh = node.at("mesh");
            if (meshOnly < 0 || mesh == size_t(meshOnly))
                selected.insert(mesh);
        }
    std::map<size_t, size_t> geometryIndex;
    std::map<uint64_t, size_t> shared;
    fs::create_directories(output);
    for (size_t mesh : selected) {
        const auto identity = sourceIdentity(scene, representative.at(mesh), source);
        const auto record = records.find(identity.key);
        if (record == records.end())
            throw std::runtime_error("RTXMG index has no shard for source mesh " +
                                     std::to_string(mesh));
        if (record->second.size != identity.size || record->second.mtime != identity.mtime)
            throw std::runtime_error("RTXMG shard is stale for source mesh " +
                                     std::to_string(mesh));
        const uint64_t hash = record->second.hash;
        if (shared.count(hash)) {
            geometryIndex[mesh] = shared.at(hash);
            continue;
        }
        if (maxMeshes > 0 && shared.size() >= size_t(maxMeshes))
            continue;
        const auto base = "mesh-" + std::to_string(mesh) + ".json";
        auto metadata = indexRtxmgShard(cache / (hexHash(hash) + ".shard"), hash, configHash);
        metadata["sourceMesh"] = mesh;
        writeJson(output / base, metadata);
        geometryIndex[mesh] = shared[hash] = asset["meshes"].size();
        asset["meshes"].push_back({{"sourceMesh", mesh}, {"metadata", base}});
        if (shared.size() % 100 == 0)
            std::cout << "[RTXMG import] indexed " << shared.size() << " geometries" << std::endl;
    }
    const auto &nodes = scene.at("nodes");
    std::vector<int> parents(nodes.size(), -1), state(nodes.size());
    std::vector<glm::mat4> world(nodes.size(), glm::mat4(1));
    for (size_t i = 0; i < nodes.size(); ++i)
        for (size_t child : nodes[i].value("children", std::vector<size_t>{})) {
            if (child >= nodes.size() || parents[child] != -1)
                throw std::runtime_error("Invalid glTF parent hierarchy");
            parents[child] = int(i);
        }
    std::function<glm::mat4(size_t)> transform = [&](size_t i) -> glm::mat4 {
        if (state[i] == 2)
            return world[i];
        if (state[i] == 1)
            throw std::runtime_error("Cyclic glTF hierarchy");
        state[i] = 1;
        const auto &node = nodes[i];
        glm::mat4 local(1);
        if (node.contains("matrix")) {
            const auto m = node["matrix"].get<std::array<float, 16>>();
            std::memcpy(glm::value_ptr(local), m.data(), 64);
        } else {
            const auto t = node.value("translation", std::array<float, 3>{0, 0, 0}),
                       s = node.value("scale", std::array<float, 3>{1, 1, 1});
            const auto r = node.value("rotation", std::array<float, 4>{0, 0, 0, 1});
            local = glm::translate(glm::mat4(1), glm::vec3(t[0], t[1], t[2])) *
                    glm::mat4_cast(glm::quat(r[3], r[0], r[1], r[2])) *
                    glm::scale(glm::mat4(1), glm::vec3(s[0], s[1], s[2]));
        }
        world[i] = parents[i] < 0 ? local : transform(parents[i]) * local;
        state[i] = 2;
        return world[i];
    };
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].contains("mesh"))
            continue;
        const size_t mesh = nodes[i].at("mesh");
        if (!geometryIndex.count(mesh))
            continue;
        const auto matrix = transform(i);
        auto append = [&](const glm::mat4 &placed, int64_t instance) {
            std::array<float, 16> values;
            std::memcpy(values.data(), glm::value_ptr(placed), 64);
            asset["nodes"].push_back({{"mesh", geometryIndex.at(mesh)},
                                      {"sourceMesh", mesh},
                                      {"sourceNode", i},
                                      {"sourceInstance", instance},
                                      {"matrix", values}});
        };
        const auto ext = nodes[i].value("extensions", Json::object());
        if (!ext.contains("EXT_mesh_gpu_instancing")) {
            append(matrix, -1);
            continue;
        }
        const auto &attrs = ext.at("EXT_mesh_gpu_instancing").at("attributes");
        size_t n = 0;
        for (auto it = attrs.begin(); it != attrs.end(); ++it) {
            const size_t c = scene.at("accessors").at(it.value().get<size_t>()).at("count");
            if (n && c != n)
                throw std::runtime_error("Instance accessor counts differ");
            n = c;
        }
        if (!n)
            throw std::runtime_error("Empty GPU instancing attributes");
        input.decoded.clear();
        const auto t = attrs.contains("TRANSLATION") ? input.vectors(attrs.at("TRANSLATION"))
                                                     : std::vector<glm::vec3>(n, glm::vec3(0));
        const auto s = attrs.contains("SCALE") ? input.vectors(attrs.at("SCALE"))
                                               : std::vector<glm::vec3>(n, glm::vec3(1));
        const auto r = attrs.contains("ROTATION")
                           ? input.rotations(attrs.at("ROTATION"))
                           : std::vector<glm::quat>(n, glm::quat(1, 0, 0, 0));
        for (size_t j = 0; j < n; ++j)
            append(matrix * glm::translate(glm::mat4(1), t[j]) * glm::mat4_cast(r[j]) *
                       glm::scale(glm::mat4(1), s[j]),
                   int64_t(j));
    }
    if (asset["nodes"].empty())
        throw std::runtime_error("No mesh placements selected");
    writeJson(output / "asset.json.tmp", asset);
    if (fs::exists(output / "asset.json"))
        fs::remove(output / "asset.json");
    fs::rename(output / "asset.json.tmp", output / "asset.json");
    const auto checked = loadAsset(output / "asset.json");
    for (const auto &geometry : checked.geometries)
        for (uint32_t root : geometry.roots)
            readPage(geometry, root);
    std::cout << "[RTXMG import] asset=" << output / "asset.json"
              << " geometries=" << checked.geometries.size()
              << " placements=" << checked.placements.size()
              << " (native payloads referenced in place)" << std::endl;
    return 0;
} catch (const std::exception &e) {
    std::cerr << "[RTXMG import] " << e.what() << std::endl;
    return 1;
}
