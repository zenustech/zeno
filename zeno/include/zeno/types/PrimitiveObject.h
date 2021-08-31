#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/AttrVector.h>
#include <zeno/utils/vec.h>
#include <variant>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace zeno {

struct PrimitiveObject : IObjectClone<PrimitiveObject> {

    AttrVector<vec3f> verts;
    AttrVector<int> points;
    AttrVector<vec2i> lines;
    AttrVector<vec3i> tris;
    AttrVector<vec4i> quads;

    std::vector<int> loops;
    AttrVector<std::pair<int, int>> polys;

    template <class T>
    [[deprecated("use prim->verts.somefunc() instead")]] auto &add_attr(std::string const &name) {
        return verts.add_attr(name);
    }

    template <class T>
    [[deprecated("use prim->verts.somefunc() instead")]] auto &add_attr(std::string const &name, T const &value) {
        return verts.add_attr(name, value);
    }

    template <class T>
    [[deprecated("use prim->verts.somefunc() instead")]] auto const &attr(std::string const &name) const {
        return verts.attr(name);
    }

    template <class T>
    [[deprecated("use prim->verts.somefunc() instead")]] auto &attr(std::string const &name) {
        return verts.attr(name);
    }

    [[deprecated("use prim->verts.somefunc() instead")]] auto const &attr(std::string const &name) const {
        return verts.attr(name);
    }

    [[deprecated("use prim->verts.somefunc() instead")]] auto &attr(std::string const &name) {
        return verts.attr(name);
    }

    [[deprecated("use prim->verts.somefunc() instead")]] bool has_attr(std::string const &name) const {
        return verts.has_attr(name);
    }

    template <class T>
    [[deprecated("use prim->verts.somefunc() instead")]] bool attr_is(std::string const &name) const {
        return verts.attr_is<T>(name);
    }

    [[deprecated("use prim->verts.somefunc() instead")]] size_t size() const {
        return verts.size();
    }

    [[deprecated("use prim->verts.somefunc() instead")]] void resize(size_t size) {
        return verts.resize(size);
    }

#ifndef ZENO_APIFREE
    ZENO_API virtual void dumpfile(std::string const &path) override;
#else
    virtual void dumpfile(std::string const &path) override {}
#endif
};

} // namespace zeno
