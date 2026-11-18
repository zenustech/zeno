#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include <tinygltf/json.hpp>

namespace zeno::cluster::layers {
inline constexpr unsigned PolicyVersion = 2;
// Offline only. The renderer and native CLAS payload format are unchanged.
class Scene {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    Scene(const std::filesystem::path& source, uint64_t cacheBytes, unsigned workers = 0);
    ~Scene();
    void setAliases(const std::unordered_map<size_t, size_t>& aliases);
    std::string dependencyKey(size_t mesh);
    class Context;
    std::unique_ptr<Context> constrain(size_t mesh, size_t primitive,
        const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices);
};

class Scene::Context {
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class Scene;
    friend size_t simplify(unsigned int*, const unsigned int*, size_t, const float*, size_t, size_t,
        const float*, size_t, const float*, size_t, const unsigned char*, size_t, float, unsigned int, float*);
public:
    explicit Context(std::unique_ptr<Impl> value);
    ~Context();
    const std::vector<unsigned char>& locks() const;
    nlohmann::json statistics() const;
};

// Same ABI as meshopt_simplifyWithAttributes. ClusterBake routes only the
// cluster-LOD generator through this adapter, leaving the submodule untouched.
size_t simplify(unsigned int* destination, const unsigned int* indices, size_t indexCount,
    const float* positions, size_t vertexCount, size_t positionStride,
    const float* attributes, size_t attributeStride, const float* weights, size_t attributeCount,
    const unsigned char* locks, size_t targetCount, float targetError, unsigned int options, float* resultError);
} // namespace zeno::cluster::layers
