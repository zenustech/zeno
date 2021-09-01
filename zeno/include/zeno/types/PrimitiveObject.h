#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/AttrVector.h>
#include <zeno/utils/vec.h>
#include <optional>
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

    AttrVector<int> loops;
    AttrVector<std::pair<int, int>> polys;

    // deprecated:
    template <class F>
    void foreach_attr(F const &f) {
        verts.foreach_attr(f);
        std::string pos_name = "pos";
        f(pos_name, verts.values);
    }

    template <class F>
    void foreach_attr(F const &f) const {
        verts.foreach_attr(f);
        std::string const pos_name = "pos";
        f(pos_name, verts.values);
    }

    size_t num_attrs() const {
        return verts.num_attrs();
    }

    template <class T>
    auto &add_attr(std::string const &name) {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        }
        return verts.add_attr<T>(name);
    }

    template <class T>
    auto &add_attr(std::string const &name, T const &value) {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        }
        return verts.add_attr<T>(name, value);
    }

    template <class T>
    auto const &attr(std::string const &name) const {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        }
        return verts.attr<T>(name);
    }

    template <class T>
    auto &attr(std::string const &name) {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        }
        return verts.attr<T>(name);
    }

    auto const &attr(std::string const &name) const {
        //if (name == "pos") return verts.values;
        return verts.attr(name);
    }

    auto &attr(std::string const &name) {
        //if (name == "pos") return verts.values;
        return verts.attr(name);
    }

    bool has_attr(std::string const &name) const {
        return verts.has_attr(name);
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        return verts.attr_is<T>(name);
    }

    size_t size() const {
        return verts.size();
    }

    void resize(size_t size) {
        return verts.resize(size);
    }
    // end of deprecated

#ifndef ZENO_APIFREE
    ZENO_API virtual void dumpfile(std::string const &path) override;
#else
    virtual void dumpfile(std::string const &path) override {}
#endif
};

} // namespace zeno
