#include "../vdb_velocity_extrapolator.h"
#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/types/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/ZenoInc.h>
#include <zeno/zeno.h>

/*
static void solve_viscosity(
    packed_FloatGrid3 &velocity,
    packed_FloatGrid3 &viscous_velocity,
    openvdb::FloatGrid::Ptr &liquid_sdf,
    openvdb::FloatGrid::Ptr &solid_sdf,
    openvdb::Vec3fGrid::Ptr &solid_velocity,
    float density, float viscosity, float dt);
*/

namespace zeno {

struct SolveViscousTerm : zeno::INode {
    static constexpr float eps = 10 * std::numeric_limits<float>::epsilon();

    void apply() override {
        auto n = get_param<int>("VelExtraLayer");
        auto dt = get_input2<float>("dt");
        auto dx = get_param<float>("dx");
        if (has_input("Dx")) {
            dx = get_input2<float>("Dx");
        }
        auto density = get_input2<float>("Density");
        auto viscosity = get_input2<float>("Viscosity");
        auto velocity = get_input<VDBFloat3Grid>("Velocity");
        auto velocity_viscous = get_input<VDBFloat3Grid>("ViscousVelocity");
        auto liquid_sdf = get_input<VDBFloatGrid>("LiquidSDF");
        auto solid_sdf = get_input<VDBFloatGrid>("SolidSDF");
        auto solid_velocity = get_input<VDBFloat3Grid>("SolidVelocity");

        if (viscosity > eps) {
            auto viscosity_grid = openvdb::FloatGrid::create(viscosity);

            packed_FloatGrid3 packed_velocity, packed_viscous_vel;
            packed_velocity.from_vec3(velocity->m_grid);
            packed_viscous_vel.from_vec3(velocity_viscous->m_grid);

            FLIP_vdb::solve_viscosity(packed_velocity, packed_viscous_vel, liquid_sdf->m_grid, solid_sdf->m_grid,
                                      solid_velocity->m_grid, viscosity_grid, density, dt);

            vdb_velocity_extrapolator::union_extrapolate(n, packed_viscous_vel.v[0], packed_viscous_vel.v[1],
                                                         packed_viscous_vel.v[2], &(liquid_sdf->m_grid->tree()));

            packed_viscous_vel.to_vec3(velocity_viscous->m_grid);
        } else {
            velocity_viscous->m_grid = velocity->m_grid->deepCopy();
            velocity_viscous->setName("Velocity_Viscous");
        }
    }
};

ZENDEFNODE(SolveViscousTerm, {
                                 /* inputs: */
                                 {{gParamType_Float, "dt"},
                                  {gParamType_Float, "Dx"},
                                  {gParamType_Float, "Density", "1000.0"},
                                  {gParamType_Float, "Viscosity", "0.0"},
                                  "Velocity",
                                  "ViscousVelocity",
                                  "LiquidSDF",
                                  "SolidSDF",
                                  "SolidVelocity"},
                                 /* outputs: */
                                 {},
                                 /* params: */
                                 {{gParamType_Float, "dx", "0.0"}, {gParamType_Int, "VelExtraLayer", "3"}},
                                 /* category: */
                                 {"FLIPSolver"},
                             });

struct SolveVariationalViscosity : zeno::INode {
    static constexpr float eps = 10 * std::numeric_limits<float>::epsilon();

    void apply() override {
        auto n = get_param<int>("VelExtraLayer");
        auto dt = get_input2<float>("dt");
        auto dx = get_param<float>("dx");
        if (has_input("Dx")) {
            dx = get_input2<float>("Dx");
        }
        auto density = get_input2<float>("Density");
        auto viscosity_grid = get_input2<VDBFloatGrid>("ViscosityGrid");
        auto velocity = get_input<VDBFloat3Grid>("Velocity");
        auto velocity_viscous = get_input<VDBFloat3Grid>("ViscousVelocity");
        auto liquid_sdf = get_input<VDBFloatGrid>("LiquidSDF");
        auto solid_sdf = get_input<VDBFloatGrid>("SolidSDF");
        auto solid_velocity = get_input<VDBFloat3Grid>("SolidVelocity");

        packed_FloatGrid3 packed_velocity, packed_viscous_vel;
        packed_velocity.from_vec3(velocity->m_grid);
        packed_viscous_vel.from_vec3(velocity_viscous->m_grid);

        FLIP_vdb::solve_viscosity(packed_velocity, packed_viscous_vel, liquid_sdf->m_grid, solid_sdf->m_grid,
                                  solid_velocity->m_grid, viscosity_grid->m_grid, density, dt);

        vdb_velocity_extrapolator::union_extrapolate(n, packed_viscous_vel.v[0], packed_viscous_vel.v[1],
                                                     packed_viscous_vel.v[2], &(liquid_sdf->m_grid->tree()));

        packed_viscous_vel.to_vec3(velocity_viscous->m_grid);
    }
};

ZENDEFNODE(SolveVariationalViscosity, {
                                          /* inputs: */
                                          {{gParamType_Float, "dt"},
                                           {gParamType_Float, "Dx"},
                                           {gParamType_Float, "Density", "1000.0"},
                                           "ViscosityGrid",
                                           "Velocity",
                                           "ViscousVelocity",
                                           "LiquidSDF",
                                           "SolidSDF",
                                           "SolidVelocity"},
                                          /* outputs: */
                                          {},
                                          /* params: */
                                          {{gParamType_Float, "dx", "0.0"}, {gParamType_Int, "VelExtraLayer", "3"}},
                                          /* category: */
                                          {"FLIPSolver"},
                                      });

} // namespace zeno
