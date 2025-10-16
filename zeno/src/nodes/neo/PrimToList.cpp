#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {
namespace {

struct PrimFlattenTris : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
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
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
    },
    {
{gParamType_Primitive, "prim"},
},
    {},
    {"primitive"},
});
struct PrimFlattenLines : INode {
  virtual void apply() override {
    auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
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
    ZImpl(set_output("prim", std::move(prim)));
  }
};

ZENO_DEFNODE(PrimFlattenLines)({
    {
        {gParamType_Primitive, "prim"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"primitive"},
});

struct PrimFlattenPolys : INode {
  virtual void apply() override {
    auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
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
    ZImpl(set_output("prim", std::move(prim)));
  }
};

ZENO_DEFNODE(PrimFlattenPolys)({
    {
        {gParamType_Primitive, "prim"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"primitive"},
});
struct PrimToList : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto type = ZImpl(get_input2<std::string>("type"));
        auto attr = ZImpl(get_input2<std::string>("attr"));
        auto lst = create_ListObject();

        if (attr.empty()) {
            if (type == "verts") {
                lst->m_impl->resize(prim->verts.size());
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->verts[i]));
                }
            } else if (type == "points") {
                lst->m_impl->resize(prim->points.size());
                for (size_t i = 0; i < prim->points.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->points[i]));
                }
            } else if (type == "lines") {
                lst->m_impl->resize(prim->lines.size());
                for (size_t i = 0; i < prim->lines.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->lines[i]));
                }
            } else if (type == "tris") {
                lst->m_impl->resize(prim->tris.size());
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->tris[i]));
                }
            } else if (type == "quads") {
                lst->m_impl->resize(prim->quads.size());
                for (size_t i = 0; i < prim->quads.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->quads[i]));
                }
            } else if (type == "polys") {
                lst->m_impl->resize(prim->polys.size());
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->polys[i]));
                }
            } else if (type == "loops") {
                lst->m_impl->resize(prim->loops.size());
                for (size_t i = 0; i < prim->loops.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(prim->loops[i]));
                }
            } else {
                throw makeError("invalid type " + type);
            }
        } else {
            auto fun = [&] (auto const &arr) {
                lst->m_impl->resize(arr.size());
                for (size_t i = 0; i < arr.size(); i++) {
                    lst->m_impl->set(i, std::make_unique<NumericObject>(arr[i]));
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

        ZImpl(set_output("list", std::move(lst)));
    }
};

ZENO_DEFNODE(PrimToList)({
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {"enum verts points lines tris quads polys loops", "type", "verts"},
        {gParamType_String, "attr", "pos"},
    },
    {
        {gParamType_List, "list"},
    },
    {},
    {"primitive"},
});

struct PrimUpdateFromList : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto lst = ZImpl(get_input<ListObject>("list"));
        auto type = ZImpl(get_input2<std::string>("type"));
        auto attr = ZImpl(get_input2<std::string>("attr"));

        if (attr.empty()) {
            if (type == "verts") {
                prim->verts.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    prim->verts[i] = objectToLiterial<vec3f>(lst->m_impl->get(i)->clone());
                }
            } else if (type == "points") {
                prim->points.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->points.size(); i++) {
                    prim->points[i] = objectToLiterial<int>(lst->m_impl->get(i)->clone());
                }
            } else if (type == "lines") {
                prim->lines.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->lines.size(); i++) {
                    prim->lines[i] = objectToLiterial<vec2i>(lst->m_impl->get(i)->clone());
                }
            } else if (type == "tris") {
                prim->tris.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    prim->tris[i] = objectToLiterial<vec3i>(lst->m_impl->get(i)->clone());
                }
            } else if (type == "quads") {
                prim->quads.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->quads.size(); i++) {
                    prim->quads[i] = objectToLiterial<vec4i>(lst->m_impl->get(i)->clone());
                }
            } else if (type == "polys") {
                prim->polys.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    prim->polys[i] = objectToLiterial<vec2i>(lst->m_impl->get(i)->clone());
                }
            } else if (type == "loops") {
                prim->loops.resize(lst->m_impl->size());
                for (size_t i = 0; i < prim->loops.size(); i++) {
                    prim->loops[i] = objectToLiterial<int>(lst->m_impl->get(i)->clone());
                }
            } else {
                throw makeError("invalid type " + type);
            }
        } else {
            auto fun = [&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                for (size_t i = 0; i < arr.size(); i++) {
                    arr[i] = objectToLiterial<T>(lst->m_impl->get(i)->clone());
                }
            };
            if (type == "verts") {
                prim->verts.resize(lst->m_impl->size());
                prim->verts.attr_visit(attr, fun);
            } else if (type == "points") {
                prim->points.resize(lst->m_impl->size());
                prim->points.attr_visit(attr, fun);
            } else if (type == "lines") {
                prim->lines.resize(lst->m_impl->size());
                prim->lines.attr_visit(attr, fun);
            } else if (type == "tris") {
                prim->tris.resize(lst->m_impl->size());
                prim->tris.attr_visit(attr, fun);
            } else if (type == "quads") {
                prim->quads.resize(lst->m_impl->size());
                prim->quads.attr_visit(attr, fun);
            } else if (type == "polys") {
                prim->polys.resize(lst->m_impl->size());
                prim->polys.attr_visit(attr, fun);
            } else if (type == "loops") {
                prim->loops.resize(lst->m_impl->size());
                prim->loops.attr_visit(attr, fun);
            } else {
                throw makeError("invalid type " + type);
            }
        }

        ZImpl(set_output("prim", std::move(prim)));
    }
};

ZENO_DEFNODE(PrimUpdateFromList)({
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {"enum verts points lines tris quads polys loops", "type", "verts"},
        {gParamType_String, "attr", "pos"},
        {gParamType_List, "list"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"primitive"},
});

}
}
