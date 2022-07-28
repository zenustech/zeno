#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <openvdb/points/PointScatter.h>
#include <zeno/VDBGrid.h>
#include <zeno/core/Graph.h>
#include <random>

namespace zeno {
namespace {

struct VDBPointScatter : INode{
  virtual void apply() override {
    auto grid = get_input<VDBFloatGrid>("grid");
    int count = get_input2<int>("count");
    float spread = get_input2<float>("spread");
    int seed = get_input2<int>("seed");
    auto uniform = get_input2<std::string>("method") != "NonUniform";
    auto issdf = get_input2<std::string>("gridtype") == "SDF";
    if (issdf) {
        grid = std::static_pointer_cast<VDBFloatGrid>(
            getThisGraph()->callTempNode("SDFToFog", {{"SDF", grid},
                {"inplace", std::make_shared<NumericObject>((int)0)}}).at("oSDF"));
    }
    if (seed == -1) seed = std::random_device{}();

    auto points = std::make_shared<VDBPointsGrid>(uniform ?
        openvdb::points::uniformPointScatter(*grid->m_grid, openvdb::Index64(count), seed, spread) :
        openvdb::points::nonUniformPointScatter(*grid->m_grid, openvdb::Index64(count), seed, spread)
        );

    set_output("points", std::move(points));
  }
};
ZENO_DEFNODE(VDBPointScatter)(
     { /* inputs: */ {
     "grid",
     {"int", "count", "100"},
     {"float", "spread", "1"},
     {"int", "seed", "-1"},
     {"enum Fog SDF", "gridtype", "Fog"},
     {"enum Uniform NonUniform", "method", "Uniform"},
     }, /* outputs: */ {
     "points",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
}
