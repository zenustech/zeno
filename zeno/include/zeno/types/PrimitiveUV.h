#pragma once
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <tinygltf/json.hpp>
#include <limits>
#include <cmath>

namespace zeno {

// Accept the SDK's historical uvNs and the uvs_N spelling used upstream.
inline int primitiveUVIndex(const std::string &name) {
    if (name == "uvs") {
        return 0;
    }
    std::string digits;
    if (name.rfind("uvs_", 0) == 0) {
        digits = name.substr(4);
    } else if (name.size() > 3 && name.rfind("uv", 0) == 0 && name.back() == 's') {
        digits = name.substr(2, name.size() - 3);
    } else {
        return -1;
    }
    unsigned index = 0;
    if (digits.empty()) {
        return -1;
    }
    for (char c : digits) {
        if (c < '0' || c > '9' || index > (unsigned(std::numeric_limits<int>::max()) - 9) / 10) {
            return -1;
        }
        index = index * 10 + unsigned(c - '0');
    }
    return index ? int(index) : -1;
}
inline std::string primitiveUVLoopName(unsigned index) {
    return index ? "uvs_" + std::to_string(index) : "uvs";
}
inline std::string primitiveUVTriangleName(unsigned index) {
    return "auv_" + std::to_string(index);
}
inline std::string primitiveCornerName(const std::string &source, int corner) {
    return "__corner_" + source + "_" + std::to_string(corner);
}
inline const std::vector<int> *primitiveUVLoopIndices(const PrimitiveObject &prim, unsigned index) {
    const auto name = primitiveUVLoopName(index);
    if (prim.loops.attr_is<int>(name)) {
        return &prim.loops.attr<int>(name);
    }
    const auto old = "uv" + std::to_string(index) + "s";
    if (index && prim.loops.attr_is<int>(old)) {
        return &prim.loops.attr<int>(old);
    }
    return nullptr;
}
inline const AttrVector<vec2f> *primitiveUVPalette(const PrimitiveObject &prim, unsigned index) {
    if (!index) {
        return &prim.uvs;
    }
    return index <= prim.append_uvs.size() ? &prim.append_uvs[index - 1] : nullptr;
}
inline bool primitiveUVTriangleValue(const PrimitiveObject &prim, unsigned set, size_t triangle, int corner,
                                     vec2f &value) {
    if (triangle >= prim.tris.size() || corner < 0 || corner > 2) {
        return false;
    }
    if (!set) {
        const auto key = "uv" + std::to_string(corner);
        if (!prim.tris.attr_is<vec3f>(key)) {
            return false;
        }
        const auto &values = prim.tris.attr<vec3f>(key);
        if (triangle >= values.size()) {
            return false;
        }
        const auto v = values[triangle];
        value = {v[0], v[1]};
        return true;
    }
    int index = -1;
    const auto packed = primitiveUVTriangleName(set);
    const auto old = "uv" + std::to_string(set) + std::to_string(corner);
    if (prim.tris.attr_is<vec3i>(packed)) {
        const auto &ids = prim.tris.attr<vec3i>(packed);
        if (triangle >= ids.size()) {
            return false;
        }
        index = ids[triangle][corner];
    } else if (prim.tris.attr_is<int>(old)) {
        const auto &ids = prim.tris.attr<int>(old);
        if (triangle >= ids.size()) {
            return false;
        }
        index = ids[triangle];
    }
    const auto palette = primitiveUVPalette(prim, set);
    if (!palette || index < 0 || size_t(index) >= palette->size()) {
        return false;
    }
    value = (*palette)[index];
    return true;
}

// Source names are metadata, not shader expressions. Index 0 remains the
// existing primary UV; imported extras retain their file order, even above 8.
struct PrimitiveUVSet {
    unsigned index;
    std::string name, source, domain;
    int palette = -1;
    std::vector<std::string> aliases;
};
inline std::vector<PrimitiveUVSet> primitiveUVSets(const PrimitiveObject &prim) {
    std::vector<PrimitiveUVSet> sets;
    if (!prim.userData().has("UVSets")) {
        return sets;
    }
    for (const auto &j : nlohmann::json::parse(prim.userData().get2<std::string>("UVSets"))) {
        sets.push_back({j.at("index").get<unsigned>(), j.at("name").get<std::string>(),
                        j.at("source").get<std::string>(), j.at("domain").get<std::string>(),
                        j.value("palette", -1), j.value("aliases", std::vector<std::string>{})});
    }
    return sets;
}
inline void setPrimitiveUVSets(PrimitiveObject &prim, const std::vector<PrimitiveUVSet> &sets) {
    auto metadata = nlohmann::json::array();
    for (const auto &set : sets) {
        metadata.push_back({{"index", set.index},
                            {"name", set.name},
                            {"source", set.source},
                            {"domain", set.domain},
                            {"palette", set.palette},
                            {"aliases", set.aliases}});
    }
    prim.userData().set2("UVSets", metadata.dump());
}
template <class T>
inline bool primitiveUVAttribute(const T &container, const std::string &source, size_t index, vec2f &value) {
    if (index >= container.size()) {
        return false;
    }
    if (container.template attr_is<vec2f>(source)) {
        const auto &data = container.template attr<vec2f>(source);
        if (index >= data.size()) {
            return false;
        }
        value = data[index];
    } else if (container.template attr_is<vec3f>(source)) {
        const auto &data = container.template attr<vec3f>(source);
        if (index >= data.size()) {
            return false;
        }
        value = {data[index][0], data[index][1]};
    } else {
        return false;
    }
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}
inline bool primitiveUVIndexedValue(const PrimitiveObject &prim, int palette, int index, vec2f &value) {
    const auto values = palette >= 0 ? primitiveUVPalette(prim, palette) : nullptr;
    if (!values || index < 0 || size_t(index) >= values->size()) {
        return false;
    }
    value = (*values)[index];
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}
inline bool primitiveUVPointValue(const PrimitiveObject &prim, const PrimitiveUVSet &set, size_t point,
                                  vec2f &value) {
    if (point >= prim.verts.size()) {
        return false;
    }
    if (set.palette >= 0 && prim.verts.attr_is<int>(set.source)) {
        const auto &ids = prim.verts.attr<int>(set.source);
        return point < ids.size() && primitiveUVIndexedValue(prim, set.palette, ids[point], value);
    }
    return primitiveUVAttribute(prim.verts, set.source, point, value);
}
inline bool primitiveUVLoopValue(const PrimitiveObject &prim, const PrimitiveUVSet &set, size_t loop,
                                 vec2f &value) {
    if (loop >= prim.loops.size()) {
        return false;
    }
    if (set.domain == "point") {
        return prim.loops[loop] >= 0 && primitiveUVPointValue(prim, set, prim.loops[loop], value);
    }
    if (set.palette >= 0) {
        const auto ids = primitiveUVLoopIndices(prim, set.palette);
        return ids && loop < ids->size() && primitiveUVIndexedValue(prim, set.palette, (*ids)[loop], value);
    }
    return primitiveUVAttribute(prim.loops, set.source, loop, value);
}
inline bool primitiveUVValue(const PrimitiveObject &prim, const PrimitiveUVSet &set, size_t triangle,
                             int corner, vec2f &value) {
    if (triangle >= prim.tris.size() || corner < 0 || corner > 2) {
        return false;
    }
    if (set.domain == "point") {
        return prim.tris[triangle][corner] >= 0 &&
               primitiveUVPointValue(prim, set, prim.tris[triangle][corner], value);
    }
    if (set.domain == "face") {
        return primitiveUVAttribute(prim.tris, set.source, triangle, value);
    }
    if (set.palette >= 0) {
        return primitiveUVTriangleValue(prim, set.palette, triangle, corner, value) &&
               std::isfinite(value[0]) && std::isfinite(value[1]);
    }
    const auto source = primitiveCornerName(set.source, corner);
    return primitiveUVAttribute(prim.tris,
                                prim.tris.has_attr(source) ? source : set.source + std::to_string(corner),
                                triangle, value);
}
// Merge palette storage and rebase its loop/triangle indices after geometry merge.
ZENO_API void primMergeUVs(PrimitiveObject &out, const std::vector<PrimitiveObject *> &inputs);

} // namespace zeno
