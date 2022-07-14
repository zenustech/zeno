#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

namespace zeno {

struct ParticleAddDV : zeno::INode {
  virtual void apply() override {
    auto particles = get_input("Particles")->as<VDBPointsGrid>();
    auto dv = get_input("dv")->as<zeno::NumericObject>()->get<zeno::vec3f>();

    float vx = std::get<float>(get_param("vx"));
    float vy = std::get<float>(get_param("vy"));
    float vz = std::get<float>(get_param("vz"));
    openvdb::Vec3R _dv = openvdb::Vec3R(dv[0], dv[1], dv[2]);
    FLIP_vdb::point_integrate_vector(particles->m_grid, _dv, "vel");
  }
};

static int defParticleAddDV = zeno::defNodeClass<ParticleAddDV>(
    "ParticleAddDV", {/* inputs: */ {
                          "Particles",
                          "dv",
                      },
                      /* outputs: */ {},
                      /* params: */
                      {
                          {"string", "channel", "vel"},
                          {"float", "vx", "0.0"},
                          {"float", "vy", "0.0"},
                          {"float", "vz", "0.0"},
                      },

                      /* category: */
                      {
                          "FLIPSolver",
                      }});

} // namespace zeno