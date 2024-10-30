#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {
namespace {

struct PrimFlattenTris : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        AttrVector<vec3f> new_verts(prim->tris.size());
        for (int i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            new_verts[i*3+0] = prim->verts[ind[0]];
            new_verts[i*3+1] = prim->verts[ind[1]];
            new_verts[i*3+2] = prim->verts[ind[2]];
        }
        prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &new_arr = new_verts.add_attr<T>(key);
            for (int i = 0; i < prim->tris.size(); i++) {
                auto ind = prim->tris[i];
                new_arr[i*3+0] = arr[ind[0]];
                new_arr[i*3+1] = arr[ind[1]];
                new_arr[i*3+2] = arr[ind[2]];
            }
        });
        std::swap(new_verts, prim->verts);
        prim->points.clear();
        prim->lines.clear();
        prim->tris.clear();
        prim->quads.clear();
        prim->polys.clear();
        prim->loops.clear();
    }
};

ZENO_DEFNODE(PrimFlattenTris)({
    {
        "prim",
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});
struct PrimFlattenLines : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    AttrVector<vec3f> new_verts(2 * prim->lines.size());
    for (int i = 0; i < prim->lines.size(); i++) {
      auto ind = prim->lines[i];
      new_verts[i*2+0] = prim->verts[ind[0]];
      new_verts[i*2+1] = prim->verts[ind[1]];
    }
    prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
      using T = std::decay_t<decltype(arr[0])>;
      auto &new_arr = new_verts.add_attr<T>(key);
      for (int i = 0; i < prim->lines.size(); i++) {
        auto ind = prim->lines[i];
        new_arr[i*2+0] = arr[ind[0]];
        new_arr[i*2+1] = arr[ind[1]];
      }
    });
    for (int i = 0; i < prim->lines.size(); i++) {
      prim->lines[i] = zeno::vec2i(2*i, 2*i+1);
    }
    std::swap(new_verts, prim->verts);
    prim->points.clear();
    prim->tris.clear();
    prim->quads.clear();
    prim->polys.clear();
    prim->loops.clear();
    set_output("prim", std::move(prim));
  }
};

ZENO_DEFNODE(PrimFlattenLines)({
    {
        "prim",
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

struct PrimFlattenPolys : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    size_t vertNum = 0;
    for(size_t i=0; i<prim->polys.size();i++)
    {
      vertNum += prim->polys[i][1];
    }
    AttrVector<vec3f> new_verts(vertNum);
    size_t v_count = 0;
    for (int i = 0; i < prim->polys.size(); i++) {
      auto num_vert_per_poly = prim->polys[i][1];
      for(int j=0;j<num_vert_per_poly;j++)
      {
        auto vidx = prim->loops[prim->polys[i][0] + j];
        new_verts[v_count] = prim->verts[vidx];
        v_count++;
      }
    }
    prim->loops.resize(vertNum);
    for(size_t i=0;i<vertNum;i++)
    {
      prim->loops[i] = i;
    }
    std::swap(new_verts, prim->verts);
    prim->points.clear();
    prim->tris.clear();
    prim->quads.clear();
    set_output("prim", std::move(prim));
  }
};

ZENO_DEFNODE(PrimFlattenPolys)({
    {
        "prim",
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});
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
                    lst->arr[i] = std::make_shared<NumericObject>(prim->polys[i]);
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
        {"enum verts points lines tris quads polys loops", "type", "verts"},
        {"string", "attr", "pos"},
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
                    prim->polys[i] = objectToLiterial<vec2i>(lst->arr[i]);
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
        {"enum verts points lines tris quads polys loops", "type", "verts"},
        {"string", "attr", "pos"},
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
