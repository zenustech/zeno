#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/ChangeBackground.h>
#include <zeno/VDBGrid.h>

namespace zeno {
namespace {

struct VDBChangeBackground : INode{
  virtual void apply() override {
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));
    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get()); p) {
        openvdb::tools::changeBackground(p->m_grid->tree(), get_input2_float("background"));
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get()); p) {
        vec3f bg = toVec3f(get_input2_vec3f("background"));
        openvdb::tools::changeBackground(p->m_grid->tree(), vec_to_other<openvdb::Vec3f>(bg));
    }

    set_output("grid", std::move(grid));
  }
};
ZENO_DEFNODE(VDBChangeBackground)(
     { /* inputs: */ {
     {gParamType_VDBGrid, "grid", "", zeno::Socket_ReadOnly},
     {gParamType_Float, "background", ""},
     }, /* outputs: */ {
         {gParamType_VDBGrid, "grid"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBGetBackground : INode{
  virtual void apply() override {
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));
    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get()); p) {
        set_output_float("background", p->m_grid->background());
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get()); p) {
        set_output_vec3f("background", toAbiVec3f(other_to_vec<3>(p->m_grid->background())));
    }

    set_output("grid", std::move(grid));
  }
};
ZENO_DEFNODE(VDBGetBackground)(
     { /* inputs: */ {
     {gParamType_VDBGrid, "grid", "", zeno::Socket_ReadOnly},
     }, /* outputs: */ {
     {gParamType_Float, "background"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBInvertSDF : INode{
  virtual void apply() override {
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));

    auto visitor = [&] (auto &grid) {
        auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
            for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
                iter.modifyValue([&](auto &v) {
                    v = -v;
                });
            }
        };
        auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
        velman.foreach(wrangler);
        openvdb::tools::changeBackground(grid->tree(), -grid->background());
        openvdb::tools::prune(grid->tree());
    };

    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get())) {
        visitor(p->m_grid);
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get())) {
        visitor(p->m_grid);
    }

    set_output("grid", std::move(grid));
  }
};
ZENO_DEFNODE(VDBInvertSDF)(
     { /* inputs: */ {
     {gParamType_Unknown, "grid", "", zeno::Socket_ReadOnly},
     }, /* outputs: */ {
         {gParamType_VDBGrid, "grid"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBPruneFootprint : INode{
  virtual void apply() override {
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));

    auto visitor = [&] (auto &grid) {
        openvdb::tools::prune(grid->tree());
    };

    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get())) {
        visitor(p->m_grid);
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get())) {
        visitor(p->m_grid);
    }

    set_output("grid", std::move(grid));
  }
};

ZENO_DEFNODE(VDBPruneFootprint)(
     { /* inputs: */ {
         {gParamType_VDBGrid, "grid", "", zeno::Socket_ReadOnly},
     }, /* outputs: */ {
         {gParamType_VDBGrid, "grid"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
}
