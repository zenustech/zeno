#pragma once

namespace zeno {

// These are the existing shader semantics. Exposing additional alternatives in
// graph nodes requires a tested adapter for their type and coordinate space.
enum class MeshAttributeSemantic : unsigned { UV, Normal, Color, Tangent, Count };
inline constexpr unsigned MeshAttributeAlterCount=8;
static_assert(unsigned(MeshAttributeSemantic::Count)*MeshAttributeAlterCount<=32,"Alternative attribute mask width");
enum class MeshAttributeDomain : unsigned char { Point, Face, Corner };
enum class MeshAttributeScalar : unsigned char { Float32, Int32 };
struct AlternativeAttributeView {
    unsigned long long values;
    unsigned count;
    unsigned short elementSize;
    MeshAttributeDomain domain;
    MeshAttributeScalar scalar;
};
// Mask bit semantic*8+alter-1 identifies a view. Views follow set-bit order,
// so neither the semantic nor the logical Alter ID needs a descriptor field.
static_assert(sizeof(AlternativeAttributeView)==16,"Alternative attribute host/device ABI");

} // namespace zeno

#if !defined(__CUDACC__)
#include <zeno/types/PrimitiveObject.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace zeno {
using AlternativeAttributeKey = std::pair<MeshAttributeSemantic,int>;
inline MeshAttributeSemantic meshAttributeSemantic(const std::string &name) {
    if (name=="uv") return MeshAttributeSemantic::UV;
    if (name=="nrm") return MeshAttributeSemantic::Normal;
    if (name=="clr") return MeshAttributeSemantic::Color;
    if (name=="tang") return MeshAttributeSemantic::Tangent;
    throw std::runtime_error("Unknown shader attribute semantic: " + name);
}
inline std::string alternativeAttributeName(const std::string &attr,int alter) {
    return "__alternative_" + attr + "_" + std::to_string(alter);
}

// Payload transport and domain remapping do not depend on the semantic or
// vector width. Shader adapters establish the expected physical type.
struct AlternativeAttribute {
    std::vector<unsigned char> values;
    unsigned elementSize{};
    MeshAttributeDomain domain=MeshAttributeDomain::Point;
    MeshAttributeScalar scalar=MeshAttributeScalar::Float32;
    size_t count() const { return elementSize ? values.size()/elementSize : 0; }
    template<class T> void assign(const std::vector<T> &source) {
        constexpr bool integer=std::is_same_v<T,int> || std::is_same_v<T,vec2i> || std::is_same_v<T,vec3i> || std::is_same_v<T,vec4i>;
        scalar=integer ? MeshAttributeScalar::Int32 : MeshAttributeScalar::Float32;
        elementSize=sizeof(T); values.resize(source.size()*sizeof(T));
        if (!values.empty()) std::memcpy(values.data(),source.data(),values.size());
    }
};
using AlternativeAttributes = std::map<AlternativeAttributeKey,AlternativeAttribute>;
inline std::string alternativeTopologyKey(const PrimitiveObject &prim) {
    uint64_t hash=14695981039346656037ull;
    auto add=[&](uint64_t v){hash^=v;hash*=1099511628211ull;};
    add(prim.verts.size());add(prim.tris.size());add(prim.quads.size());add(prim.polys.size());
    for(const auto &tri:prim.tris)for(int k=0;k<3;++k)add(uint32_t(tri[k]));
    return std::to_string(hash);
}
} // namespace zeno
#endif
