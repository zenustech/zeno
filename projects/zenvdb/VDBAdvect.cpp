#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <zeno/zeno.h>
#include <zeno/utils/interfaceutil.h>
#include <zeno/VDBGrid.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/VolumeAdvect.h>
namespace zeno {

struct SDFAdvect : zeno::INode {
    virtual void apply() override {
        auto inSDF = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("InoutSDF"));
        auto vecField = safe_dynamic_cast<VDBFloat3Grid>(get_input("VecField"));
        auto grid = inSDF->m_grid;
        auto field = vecField->m_grid;
        auto timeStep = get_input2_float("TimeStep");
        auto velField = openvdb::tools::DiscreteField<openvdb::Vec3SGrid>(*field);
        auto advection = openvdb::tools::LevelSetAdvection<openvdb::FloatGrid, decltype(velField)>(*grid, velField);

        auto spatialScheme = zsString2Std(get_input2_string("SpatialScheme"));
        if (spatialScheme == "Order_1") {
            advection.setSpatialScheme(openvdb::math::FIRST_BIAS);
        }
        else if (spatialScheme == "Order_2") {
            advection.setSpatialScheme(openvdb::math::SECOND_BIAS);
        }
        else if (spatialScheme == "Order_3") {
            advection.setSpatialScheme(openvdb::math::THIRD_BIAS);
        }
        else if (spatialScheme == "Order_5_HJ_WENO") {
            advection.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        }
        else if (spatialScheme == "Order_5_WENO") {
            advection.setSpatialScheme(openvdb::math::WENO5_BIAS);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for SpatialScheme: " + spatialScheme);
        }

        auto temporalScheme = zsString2Std(get_input2_string("TemporalScheme"));
        if (temporalScheme == "Explicit_Euler") {
            advection.setTemporalScheme(openvdb::math::TVD_RK1);
        }
        else if (temporalScheme == "Order_2_Runge_Kuta") {
            advection.setTemporalScheme(openvdb::math::TVD_RK2);
        }
        else if (temporalScheme == "Order_3_Runge_Kuta") {
            advection.setTemporalScheme(openvdb::math::TVD_RK3);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for TemporalScheme: " + temporalScheme);
        }

        advection.setNormCount(get_input2_int("RenormalizeStep"));

        auto trackerSpatialScheme = zsString2Std(get_input2_string("TrackerSpatialScheme"));
        if (trackerSpatialScheme == "Order_1") {
            advection.setTrackerSpatialScheme(openvdb::math::FIRST_BIAS);
        }
        else if (trackerSpatialScheme == "Order_2") {
            advection.setTrackerSpatialScheme(openvdb::math::SECOND_BIAS);
        }
        else if (trackerSpatialScheme == "Order_3") {
            advection.setTrackerSpatialScheme(openvdb::math::THIRD_BIAS);
        }
        else if (trackerSpatialScheme == "Order_5_HJ_WENO") {
            advection.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        }
        else if (trackerSpatialScheme == "Order_5_WENO") {
            advection.setTrackerSpatialScheme(openvdb::math::WENO5_BIAS);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for TrackerSpatialScheme: " + trackerSpatialScheme);
        }

        auto trackerTemporalScheme = zsString2Std(get_input2_string("TrackerTemporalScheme"));
        if (trackerTemporalScheme == "Explicit_Euler") {
            advection.setTrackerTemporalScheme(openvdb::math::TVD_RK1);
        }
        else if (trackerTemporalScheme == "Order_2_Runge_Kuta") {
            advection.setTrackerTemporalScheme(openvdb::math::TVD_RK2);
        }
        else if (trackerTemporalScheme == "Order_3_Runge_Kuta") {
            advection.setTrackerTemporalScheme(openvdb::math::TVD_RK3);
        }
        else {
            throw zeno::Exception("SDFAdvect Node: wrong parameter for TrackerTemporalScheme: " + trackerTemporalScheme);
        }

        advection.advect(0.0, timeStep);
        set_output("InoutSDF", std::move(inSDF));
    }
};

ZENO_DEFNODE(SDFAdvect)(
    { /* inputs: */ {
        {gParamType_VDBGrid,"InoutSDF", "", zeno::Socket_ReadOnly},
        {gParamType_VDBGrid,"VecField", "", zeno::Socket_ReadOnly},
        {gParamType_Float, "TimeStep", "0.04"},
        {"enum Order_1 Order_2 Order_3 Order_5_WENO Order_5_HJ_WENO", "SpatialScheme", "Order_5_HJ_WENO"},
        {"enum Explicit_Euler Order_2_Runge_Kuta Order_3_Runge_Kuta", "TemporalScheme", "Order_2_Runge_Kuta"},
        {gParamType_Int, "RenormalizeStep", "3"},
        {"enum Order_1 Order_2 Order_3 Order_5_WENO Order_5_HJ_WENO", "TrackerSpatialScheme", "Order_5_HJ_WENO"},
        {"enum Explicit_Euler Order_2_Runge_Kuta Order_3_Runge_Kuta", "TrackerTemporalScheme", "Explicit_Euler"},
    }, /* outputs: */ {
        {gParamType_VDBGrid,"InoutSDF"}
    }, /* params: */ {
    }, /* category: */ {
        "openvdb",
    } }
);


struct VolumeAdvect : zeno::INode {
    virtual void apply() override {
        //
        //auto inSDF = safe_dynamic_cast<VDBFloatGrid>(get_input("InoutField"));
        auto vecField = safe_dynamic_cast<VDBFloat3Grid>(get_input("VecField"));
        //auto grid = inSDF->m_grid;
        auto field = vecField->m_grid;
        auto timeStep = get_input2_float("TimeStep");
        //auto velField = openvdb::tools::DiscreteField<openvdb::Vec3SGrid>(*field);
        using VolumeAdvection =
        openvdb::tools::VolumeAdvection< openvdb::Vec3fGrid, true>;
        VolumeAdvection advection(*field);
        
        auto spatialScheme = zsString2Std(get_input2_string("Integrator"));
        if (spatialScheme == "SemiLagrangian") {
            advection.setIntegrator(openvdb::tools::Scheme::SemiLagrangian::SEMI);
        }
        else if (spatialScheme == "MidPoint") {
            advection.setIntegrator(openvdb::tools::Scheme::SemiLagrangian::MID);
        }
        else if (spatialScheme == "RK3") {
            advection.setIntegrator(openvdb::tools::Scheme::SemiLagrangian::RK3);
        }
        else if (spatialScheme == "RK4") {
            advection.setIntegrator(openvdb::tools::Scheme::SemiLagrangian::RK4);
        }
        else if (spatialScheme == "MacCormack") {
            advection.setIntegrator(openvdb::tools::Scheme::SemiLagrangian::MAC);
        }
        else if (spatialScheme == "BFECC") {
            advection.setIntegrator(openvdb::tools::Scheme::SemiLagrangian::BFECC);
        }
        else {
            throw zeno::Exception("VolumeAdvect Node: wrong parameter for Integrator: " + spatialScheme);
        }

        auto temporalScheme = zsString2Std(get_input2_string("Limiter"));
        if (temporalScheme == "None") {
            advection.setLimiter(openvdb::tools::Scheme::Limiter::NO_LIMITER);
        }
        else if (temporalScheme == "Clamp") {
            advection.setLimiter(openvdb::tools::Scheme::Limiter::CLAMP);
        }
        else if (temporalScheme == "Revert") {
            advection.setLimiter(openvdb::tools::Scheme::Limiter::REVERT);
        }
        else {
            throw zeno::Exception("VolumeAdvect Node: wrong parameter for Limitter: " + temporalScheme);
        }

        advection.setSubSteps(get_input2_int("SubSteps"));
        
        if (safe_dynamic_cast<VDBGrid>(get_input("InField"))->getType()=="FloatGrid")
        {
            
            auto f = safe_dynamic_cast<VDBFloatGrid>(get_input("InField"));
            auto f2 = f->m_grid->deepCopy();
            //auto result = std::make_shared<VDBFloatGrid>();
            auto res = advection.template advect<openvdb::FloatGrid,
                    openvdb::tools::Sampler<1, false>>(*f2, timeStep);
            f->m_grid = res->deepCopy();
            //set_output("outField", get_input("InField"));
        }
        else if(safe_dynamic_cast<VDBGrid>(get_input("InField"))->getType()=="Vec3fGrid")
        {
            auto f = safe_dynamic_cast<VDBFloat3Grid>(get_input("InField"));
            auto f2 = f->m_grid->deepCopy();
            auto res = advection.template advect<openvdb::Vec3fGrid,
                    openvdb::tools::Sampler<1, true>>(*f2, timeStep);
            f->m_grid = res->deepCopy();
            //set_output("outField", get_input("InField"));
        }
        //advection.advect(0.0, timeStep);
        auto layers_to_ext = std::make_unique<zeno::NumericObject>();
        layers_to_ext->set<int>(advection.getMaxDistance(*field, timeStep));
        set_output("extend", std::move(layers_to_ext));
    }
};

ZENO_DEFNODE(VolumeAdvect)(
    { /* inputs: */ {
        {gParamType_VDBGrid,"InField", "", zeno::Socket_ReadOnly},
        {gParamType_VDBGrid,"VecField", "", zeno::Socket_ReadOnly},
        {gParamType_Float, "TimeStep", "0.04"},
        {gParamType_Int, "SubSteps", "1"},
        {"enum SemiLagrangian  MidPoint RK3 RK4 MacCormack BFECC", "Integrator", "BFECC"},
        {"enum None Clamp Revert", "Limiter", "Revert"},
    }, /* outputs: */ {
        //"outField"
        {gParamType_Int, "extend"},
    }, /* params: */ {
    }, /* category: */ {
        "openvdb",
    } }
);

}