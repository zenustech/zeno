#include "FLIP_vdb.h"
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>
#include <zeno/ConditionObject.h>

namespace zeno {

struct ParticleEmitter : zeno::INode {
  virtual void apply() override {
    auto particles = get_input("Particles")->as<VDBPointsGrid>();
    auto shape = get_input("ShapeSDF")->as<VDBFloatGrid>();
    float vx = get_param<float>("vx");
    float vy = get_param<float>("vy");
    float vz = get_param<float>("vz");

    openvdb::Vec3fGrid::Ptr velocityVolume = nullptr;
    if (has_input("VelocityVolume")) {
      if (!has_input<zeno::ConditionObject>("VelocityVolume")) {
          velocityVolume = get_input("VelocityVolume")->as<VDBFloat3Grid>()->m_grid;
      }
    }

    if (has_input("VelocityInit")) {
        auto vel = get_input<zeno::NumericObject>("VelocityInit")->get<zeno::vec3f>();
        vx = vel[0], vy = vel[1], vz = vel[2];
    }

    openvdb::FloatGrid::Ptr liquid_sdf = nullptr;
    if (has_input("LiquidSDF")) {
      liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>()->m_grid;
    }

    FLIP_vdb::emit_liquid(particles->m_grid, shape->m_grid,
                          velocityVolume, liquid_sdf, vx, vy,
                          vz);
    set_output("Particles", get_input("Particles"));
  }
};

static int defParticleEmitter = zeno::defNodeClass<ParticleEmitter>(
    "ParticleEmitter", {/* inputs: */ {
                            {gParamType_VDBGrid, "Particles"},
                            {gParamType_VDBGrid, "ShapeSDF"},
                            {gParamType_VDBGrid, "VelocityVolume"},
                            {gParamType_VDBGrid, "VelocityInit"},
                            {gParamType_VDBGrid, "LiquidSDF"},
                        },
                        /* outputs: */ {
                            {gParamType_VDBGrid, "Particles"},
                        },
                        /* params: */
                        {
                            {gParamType_Float, "vx", "0.0"},
                            {gParamType_Float, "vy", "0.0"},
                            {gParamType_Float, "vz", "0.0"},
                        },

                        /* category: */
                        {
                            "FLIPSolver",
                        }});

} // namespace zeno
