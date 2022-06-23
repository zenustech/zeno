#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {
namespace {

struct PrimToList : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        auto attr = get_input2<std::string>("attr");
        auto lst = std::make_shared<ListObject>();

        if (attr.empty()) {
            if (type == "verts") {
                lst->arr.resize(prim->verts.size());
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(prim->verts[i]);
                }
            } else if (type == "points") {
                lst->arr.resize(prim->points.size());
                for (size_t i = 0; i < prim->points.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(prim->points[i]);
                }
            } else if (type == "lines") {
                lst->arr.resize(prim->lines.size());
                for (size_t i = 0; i < prim->lines.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(prim->lines[i]);
                }
            } else if (type == "tris") {
                lst->arr.resize(prim->tris.size());
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(prim->tris[i]);
                }
            } else if (type == "quads") {
                lst->arr.resize(prim->quads.size());
                for (size_t i = 0; i < prim->quads.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(prim->quads[i]);
                }
            } else if (type == "polys") {
                lst->arr.resize(prim->polys.size());
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    auto [base, len] = prim->polys[i];
                    lst->arr[i] = std::make_shared<NumericObject>(vec2i(base, len));
                }
            } else if (type == "loops") {
                lst->arr.resize(prim->loops.size());
                for (size_t i = 0; i < prim->loops.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(prim->loops[i]);
                }
            } else {
                throw makeError("invalid type " + type);
            }
        } else {
            auto fun = [&] (auto const &arr) {
                lst->arr.resize(arr.size());
                for (size_t i = 0; i < arr.size(); i++) {
                    lst->arr[i] = std::make_shared<NumericObject>(arr[i]);
                }
            };
            if (type == "verts") {
                prim->verts.attr_visit(attr, fun);
            } else if (type == "points") {
                prim->points.attr_visit(attr, fun);
            } else if (type == "lines") {
                prim->lines.attr_visit(attr, fun);
            } else if (type == "tris") {
                prim->tris.attr_visit(attr, fun);
            } else if (type == "quads") {
                prim->quads.attr_visit(attr, fun);
            } else if (type == "polys") {
                prim->polys.attr_visit(attr, fun);
            } else if (type == "loops") {
                prim->loops.attr_visit(attr, fun);
            } else {
                throw makeError("invalid type " + type);
            }
        }

        set_output("list", std::move(lst));
    }
};

ZENO_DEFNODE(PrimToList)({
    {
        {"prim"},
    },
    {
        {"list"},
    },
    {},
    {"primitive"},
});

struct PrimUpdateFromList : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto lst = get_input<ListObject>("list");
        auto type = get_input2<std::string>("type");
        auto attr = get_input2<std::string>("attr");

        if (attr.empty()) {
            if (type == "verts") {
                prim->verts.resize(lst->arr.size());
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    prim->verts[i] = objectToLiterial<vec3f>(lst->arr[i]);
                }
            } else if (type == "points") {
                prim->points.resize(lst->arr.size());
                for (size_t i = 0; i < prim->points.size(); i++) {
                    prim->points[i] = objectToLiterial<int>(lst->arr[i]);
                }
            } else if (type == "lines") {
                prim->lines.resize(lst->arr.size());
                for (size_t i = 0; i < prim->lines.size(); i++) {
                    prim->lines[i] = objectToLiterial<vec2i>(lst->arr[i]);
                }
            } else if (type == "tris") {
                prim->tris.resize(lst->arr.size());
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    prim->tris[i] = objectToLiterial<vec3i>(lst->arr[i]);
                }
            } else if (type == "quads") {
                prim->quads.resize(lst->arr.size());
                for (size_t i = 0; i < prim->quads.size(); i++) {
                    prim->quads[i] = objectToLiterial<vec4i>(lst->arr[i]);
                }
            } else if (type == "polys") {
                prim->polys.resize(lst->arr.size());
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    auto v = objectToLiterial<vec2i>(lst->arr[i]);
                    prim->polys[i] = {v[0], v[1]};
                }
            } else if (type == "loops") {
                prim->loops.resize(lst->arr.size());
                for (size_t i = 0; i < prim->loops.size(); i++) {
                    prim->loops[i] = objectToLiterial<int>(lst->arr[i]);
                }
            } else {
                throw makeError("invalid type " + type);
            }
        } else {
            auto fun = [&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                for (size_t i = 0; i < arr.size(); i++) {
                    arr[i] = objectToLiterial<T>(lst->arr[i]);
                }
            };
            if (type == "verts") {
                prim->verts.resize(lst->arr.size());
                prim->verts.attr_visit(attr, fun);
            } else if (type == "points") {
                prim->points.resize(lst->arr.size());
                prim->points.attr_visit(attr, fun);
            } else if (type == "lines") {
                prim->lines.resize(lst->arr.size());
                prim->lines.attr_visit(attr, fun);
            } else if (type == "tris") {
                prim->tris.resize(lst->arr.size());
                prim->tris.attr_visit(attr, fun);
            } else if (type == "quads") {
                prim->quads.resize(lst->arr.size());
                prim->quads.attr_visit(attr, fun);
            } else if (type == "polys") {
                prim->polys.resize(lst->arr.size());
                prim->polys.attr_visit(attr, fun);
            } else if (type == "loops") {
                prim->loops.resize(lst->arr.size());
                prim->loops.attr_visit(attr, fun);
            } else {
                throw makeError("invalid type " + type);
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimUpdateFromList)({
    {
        {"prim"},
        {"list"},
    },
    {
        {"prim"},
    },
    {},
    {"primitive"},
});

}
}
