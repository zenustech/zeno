#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <openvdb/points/PointScatter.h>
#include <zeno/VDBGrid.h>
#include <zeno/core/Graph.h>
#include <random>
#include <zeno/utils/interfaceutil.h>


namespace zeno {
namespace {
#if 0
struct VDBPointScatter : INode{
  virtual void apply() override {
    auto grid = safe_dynamic_cast<VDBFloatGrid>(get_input("grid"));
    openvdb::Index64 count = get_input2_int("count");
    float spread = get_input2_float("spread");
    int seed = get_input2_int("seed");
    auto isuniform = zsString2Std(get_input2_string("method")) != "NonUniform";
    auto issdf = zsString2Std(get_input2_string("gridtype")) == "SDF";
    auto istotal = zsString2Std(get_input2_string("counttype")) == "Total";
    if (issdf) {
        grid = std::static_pointer_cast<VDBFloatGrid>(
            getThisGraph()->callTempNode("SDFToFog", {{"SDF", grid},
                {"inplace", std::make_shared<NumericObject>((int)0)}}).at("oSDF"));
    }
    if (seed == -1) seed = std::random_device{}();

    if (!istotal && isuniform) {
        count *= grid->m_grid->tree().activeVoxelCount();
    }
    if (istotal && !isuniform) {
        auto avc = grid->m_grid->tree().activeVoxelCount();
        count += avc - 1;
        count /= avc;
    }

    auto points = std::make_shared<VDBPointsGrid>(isuniform ?
        openvdb::points::uniformPointScatter(*grid->m_grid, count, seed, spread) :
        openvdb::points::nonUniformPointScatter(*grid->m_grid, count, seed, spread)
        );

    set_output("points", std::move(points));
  }
};
ZENO_DEFNODE(VDBPointScatter)(
     { /* inputs: */ {
     {gParamType_VDBGrid,"grid", "", zeno::Socket_ReadOnly},
     {gParamType_Int, "count", "4"},
     {gParamType_Float, "spread", "1"},
     {gParamType_Int, "seed", "-1"},
     {"enum Fog SDF", "gridtype", "Fog"},
     {"enum PerVoxel Total", "counttype", "PerVoxel"},
     {"enum Uniform NonUniform", "method", "Uniform"},
     }, /* outputs: */ {
         {gParamType_VDBGrid,"points"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});
#endif

}
}
