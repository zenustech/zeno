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
    auto grid = get_input<VDBGrid>("grid");
    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid); p) {
        openvdb::tools::changeBackground(p->m_grid->tree(), get_input2<float>("background"));
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid); p) {
        openvdb::tools::changeBackground(p->m_grid->tree(), vec_to_other<openvdb::Vec3f>(get_input2<vec3f>("background")));
    }

    set_output("grid", get_input("grid"));
  }
};
ZENO_DEFNODE(VDBChangeBackground)(
     { /* inputs: */ {
     "grid", {"float", "background"},
     }, /* outputs: */ {
     "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBGetBackground : INode{
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("grid");
    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid); p) {
        set_output2("background", p->m_grid->background());
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid); p) {
        set_output2("background", other_to_vec<3>(p->m_grid->background()));
    }

    set_output("grid", get_input("grid"));
  }
};
ZENO_DEFNODE(VDBGetBackground)(
     { /* inputs: */ {
     "grid",
     }, /* outputs: */ {
     {"float", "background"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBInvertSDF : INode{
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("grid");

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

    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid)) {
        visitor(p->m_grid);
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid)) {
        visitor(p->m_grid);
    }

    set_output("grid", get_input("grid"));
  }
};
ZENO_DEFNODE(VDBInvertSDF)(
     { /* inputs: */ {
     "grid",
     }, /* outputs: */ {
     "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBPruneFootprint : INode{
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("grid");

    auto visitor = [&] (auto &grid) {
        openvdb::tools::prune(grid->tree());
    };

    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid)) {
        visitor(p->m_grid);
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid)) {
        visitor(p->m_grid);
    }

    set_output("grid", get_input("grid"));
  }
};

ZENO_DEFNODE(VDBPruneFootprint)(
     { /* inputs: */ {
     "grid",
     }, /* outputs: */ {
     "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
}
