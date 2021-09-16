#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <openvdb/tools/LevelSetAdvect.h>

namespace zeno {

struct SDFAdvect : zeno::INode {
    virtual void apply() override {
        auto inSDF = get_input("InoutSDF")->as<VDBFloatGrid>();
        auto vecField = get_input("VecField")->as<VDBFloat3Grid>();
        auto grid = inSDF->m_grid;
        auto field = vecField->m_grid;
        auto timeStep = std::get<float>(get_param("TimeStep"));
        auto velField = openvdb::tools::DiscreteField<openvdb::Vec3SGrid>(*field);
        auto advection = openvdb::tools::LevelSetAdvection<openvdb::FloatGrid, decltype(velField)>(*grid, velField);
        advection.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advection.setTemporalScheme(openvdb::math::TVD_RK2);
        advection.setNormCount(std::get<int>(get_param("ReNormalizeStep")));
        advection.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advection.setTrackerTemporalScheme(openvdb::math::TVD_RK1);
        advection.advect(0.0, timeStep);
        set_output("InoutSDF", get_input("InoutSDF"));
    }
};

ZENO_DEFNODE(SDFAdvect)(
    { /* inputs: */ {
        "InoutSDF",
        "VecField",
    }, /* outputs: */ {
        "InoutSDF"
    }, /* params: */ {
        {"float", "TimeStep", "0.04"},
        {"int", "ReNormalizeStep ", "3"},
    }, /* category: */ {
        "openvdb",
    } }
);

}