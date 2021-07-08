#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

namespace zeno {

struct ParticleEmitter : zeno::INode {
  virtual void apply() override {
    auto particles = get_input("Particles")->as<VDBPointsGrid>();
    auto shape = get_input("ShapeSDF")->as<VDBFloatGrid>();
    float vx = std::get<float>(get_param("vx"));
    float vy = std::get<float>(get_param("vy"));
    float vz = std::get<float>(get_param("vz"));

    openvdb::Vec3fGrid::Ptr velocityVolume;
    if (has_input("VelocityVolume")) {
      velocityVolume = get_input("VelocityVolume")->as<VDBFloat3Grid>()->m_grid;

    } else {
      velocityVolume = nullptr;
    }

    openvdb::FloatGrid::Ptr liquid_sdf;

    if (has_input("LiquidSDF")) {
      liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>()->m_grid;

    } else {
      liquid_sdf = nullptr;
    }
    FLIP_vdb::emit_liquid(particles->m_grid, shape->m_grid,
                          velocityVolume, liquid_sdf, vx, vy,
                          vz);
  }
};

static int defParticleEmitter = zeno::defNodeClass<ParticleEmitter>(
    "ParticleEmitter", {/* inputs: */ {
                            "Particles",
                            "ShapeSDF",
                            "VelocityVolume",
                            "LiquidSDF",
                        },
                        /* outputs: */ {},
                        /* params: */
                        {
                            {"float", "vx", "0.0"},
                            {"float", "vy", "0.0"},
                            {"float", "vz", "0.0"},
                        },

                        /* category: */
                        {
                            "FLIPSolver",
                        }});

} // namespace zeno