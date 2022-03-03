#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/AttrVector.h>
#include <zeno/utils/type_traits.h>
#include <zeno/utils/vec.h>
#include <optional>
#include <variant>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace zeno {

struct MaterialObject;

struct PrimitiveObject : IObjectClone<PrimitiveObject> {
    AttrVector<vec3f> verts;
    AttrVector<int> points;
    AttrVector<vec2i> lines;
    AttrVector<vec3i> tris;
    AttrVector<vec4i> quads;

    AttrVector<int> loops;
    AttrVector<std::pair<int, int>> polys;

    std::shared_ptr<MaterialObject> mtl;

    // deprecated:
    template <class Accept = std::tuple<vec3f, float>, class F>
    void foreach_attr(F &&f) {
        std::string pos_name = "pos";
        f(pos_name, verts.values);
        verts.foreach_attr<Accept>(std::move(f));
    }

    template <class Accept = std::tuple<vec3f, float>, class F>
    void foreach_attr(F &&f) const {
        std::string const pos_name = "pos";
        f(pos_name, verts.values);
        verts.foreach_attr<Accept>(std::move(f));
    }

    size_t num_attrs() const {
        return 1 + verts.num_attrs();
    }

    auto attr_keys() const {
        auto keys = verts.attr_keys();
        keys.insert(keys.begin(), "pos");
        return keys;
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

    template <class Accept = std::tuple<vec3f, float>, class F>
    auto attr_visit(std::string const &name, F const &f) const {
        if (name == "pos") {
            return f(verts.values);
        } else {
            return verts.attr_visit<Accept>(name, f);
        }
    }

    template <class Accept = std::tuple<vec3f, float>, class F>
    auto attr_visit(std::string const &name, F const &f) {
        if (name == "pos") {
            return f(verts.values);
        } else {
            return verts.attr_visit<Accept>(name, f);
        }
    }

    bool has_attr(std::string const &name) const {
        if (name == "pos") return true;
        return verts.has_attr(name);
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return true;
        } else {
            if (name == "pos") return false;
        }
        return verts.attr_is<T>(name);
    }

    size_t size() const {
        return verts.size();
    }

    void resize(size_t size) {
        return verts.resize(size);
    }
    // end of deprecated
};
static void addIndividualPrimitive(PrimitiveObject* dst, const PrimitiveObject* src, size_t index)
        {
            for(auto key:src->attr_keys())
            {
                //using T = std::decay_t<decltype(src->attr(key)[0])>;
                if (key != "pos") {
                std::visit([index, &key, dst](auto &&src) {
                    using SrcT = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;
                    std::get<SrcT>(dst->attr(key)).emplace_back(src[index]);
                }, src->attr(key));
                // dst->attr(key).emplace_back(src->attr(key)[index]);
                } else {
                    dst->attr<vec3f>(key).emplace_back(src->attr<vec3f>(key)[index]);
                }
            }
            dst->resize(dst->attr<zeno::vec3f>("pos").size());
        }
} // namespace zeno
