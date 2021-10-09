#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <openvdb/tools/LevelSetAdvect.h>

namespace zeno {

struct SDFAdvect : zeno::INode {
    virtual void apply() override {
        auto inSDF = get_input("InoutSDF")->as<VDBFloatGrid>();
        auto vecField = get_input("VecField")->as<VDBFloat3Grid>();
        auto grid = inSDF->m_grid;
        auto field = vecField->m_grid;
        auto timeStep = get_input<NumericObject>("TimeStep")->get<float>();
        auto velField = openvdb::tools::DiscreteField<openvdb::Vec3SGrid>(*field);
        auto advection = openvdb::tools::LevelSetAdvection<openvdb::FloatGrid, decltype(velField)>(*grid, velField);

        auto spatialScheme = get_input<StringObject>("SpatialScheme")->get();
        if (spatialScheme == std::string("Order_1")) {
            advection.setSpatialScheme(openvdb::math::FIRST_BIAS);
        }
        else if (spatialScheme == std::string("Order_2")) {
            advection.setSpatialScheme(openvdb::math::SECOND_BIAS);
        }
        else if (spatialScheme == std::string("Order_3")) {
            advection.setSpatialScheme(openvdb::math::THIRD_BIAS);
        }
        else if (spatialScheme == std::string("Order_5_HJ_WENO")) {
            advection.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        }
        else if (spatialScheme == std::string("Order_5_WENO")) {
            advection.setSpatialScheme(openvdb::math::WENO5_BIAS);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for SpatialScheme: " + spatialScheme);
        }

        auto temporalScheme = get_input<StringObject>("TemporalScheme")->get();
        if (temporalScheme == std::string("Explicit_Euler")) {
            advection.setTemporalScheme(openvdb::math::TVD_RK1);
        }
        else if (temporalScheme == std::string("Order_2_Runge_Kuta")) {
            advection.setTemporalScheme(openvdb::math::TVD_RK2);
        }
        else if (temporalScheme == std::string("Order_3_Runge_Kuta")) {
            advection.setTemporalScheme(openvdb::math::TVD_RK3);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for TemporalScheme: " + temporalScheme);
        }

        advection.setNormCount(get_input<NumericObject>("RenormalizeStep")->get<int>());

        auto trackerSpatialScheme = get_input<StringObject>("TrackerSpatialScheme")->get();
        if (trackerSpatialScheme == std::string("Order_1")) {
            advection.setTrackerSpatialScheme(openvdb::math::FIRST_BIAS);
        }
        else if (trackerSpatialScheme == std::string("Order_2")) {
            advection.setTrackerSpatialScheme(openvdb::math::SECOND_BIAS);
        }
        else if (trackerSpatialScheme == std::string("Order_3")) {
            advection.setTrackerSpatialScheme(openvdb::math::THIRD_BIAS);
        }
        else if (trackerSpatialScheme == std::string("Order_5_HJ_WENO")) {
            advection.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        }
        else if (trackerSpatialScheme == std::string("Order_5_WENO")) {
            advection.setTrackerSpatialScheme(openvdb::math::WENO5_BIAS);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for TrackerSpatialScheme: " + trackerSpatialScheme);
        }

        auto trackerTemporalScheme = get_input<StringObject>("TrackerTemporalScheme")->get();
        if (trackerTemporalScheme == std::string("Explicit_Euler")) {
            advection.setTrackerTemporalScheme(openvdb::math::TVD_RK1);
        }
        else if (trackerTemporalScheme == std::string("Order_2_Runge_Kuta")) {
            advection.setTrackerTemporalScheme(openvdb::math::TVD_RK2);
        }
        else if (trackerTemporalScheme == std::string("Order_3_Runge_Kuta")) {
            advection.setTrackerTemporalScheme(openvdb::math::TVD_RK3);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for TrackerTemporalScheme: " + trackerTemporalScheme);
        }

        advection.advect(0.0, timeStep);
        set_output("InoutSDF", get_input("InoutSDF"));
    }
};

ZENO_DEFNODE(SDFAdvect)(
    { /* inputs: */ {
        "InoutSDF",
        "VecField",
        {"float", "TimeStep", "0.04"},
        {"enum Order_1 Order_2 Order_3 Order_5_WENO Order_5_HJ_WENO", "SpatialScheme", "Order_5_HJ_WENO"},
        {"enum Explicit_Euler Order_2_Runge_Kuta Order_3_Runge_Kuta", "TemporalScheme", "Order_2_Runge_Kuta"},
        {"int", "RenormalizeStep", "3"},
        {"enum Order_1 Order_2 Order_3 Order_5_WENO Order_5_HJ_WENO", "TrackerSpatialScheme", "Order_5_HJ_WENO"},
        {"enum Explicit_Euler Order_2_Runge_Kuta Order_3_Runge_Kuta", "TrackerTemporalScheme", "Explicit_Euler"},
    }, /* outputs: */ {
        "InoutSDF"
    }, /* params: */ {
    }, /* category: */ {
        "openvdb",
    } }
);

}