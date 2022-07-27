#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <openvdb/points/PointScatter.h>
#include <zeno/VDBGrid.h>
#include <random>

namespace zeno {
namespace {

struct VDBPointScatter : INode{
  virtual void apply() override {
    auto sdf = get_input<VDBFloatGrid>("SDF");
    int count = get_input2<int>("count");
    float spread = get_input2<float>("spread");
    int seed = get_input2<int>("seed");
    auto uniform = get_input2<std::string>("type") != "NonUniform";
    if (seed == -1) seed = std::random_device{}();

    auto points = std::make_shared<VDBPointsGrid>(uniform ?
        openvdb::points::uniformPointScatter(*sdf->m_grid, openvdb::Index64(count), seed, spread) :
        openvdb::points::nonUniformPointScatter(*sdf->m_grid, openvdb::Index64(count), seed, spread)
        );

    set_output("points", std::move(points));
  }
};
ZENO_DEFNODE(VDBPointScatter)(
     { /* inputs: */ {
     "SDF",
     {"int", "count", "100"},
     {"float", "spread", "1"},
     {"int", "seed", "-1"},
     {"enum Uniform NonUniform", "type", "Uniform"},
     }, /* outputs: */ {
     "points",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
}
