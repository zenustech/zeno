#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <vector>
#include <zeno/VDBGrid.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>

namespace zeno {

struct VDBNoiseSDF : zeno::INode {
  virtual void apply() override {
    auto inoutSDF = get_input<VDBFloatGrid>("inoutSDF");
    auto grid = inoutSDF->m_grid;
    auto depth = get_input<NumericObject>("depth")->get<float>();
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.modifyValue([&](auto &v) {
                v += depth;
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);
    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

ZENO_DEFNODE(VDBNoiseSDF)(
     { /* inputs: */ {
     "inoutSDF", {"float", "strength"}, 
     }, /* outputs: */ {
       "inoutSDF",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});


}
