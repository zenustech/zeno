#include "Sparse_buffer.h"
#include "array3_utils.h"
#include "fluidsim.h"
#include "volumeMeshTools.h"
#include <LaplaceBEM.h>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "../Parser/scene_parser.h"

using namespace BEM;

float timestep = 0.01f;
int frame = 0;

float grid_width = 2;

FluidSim sim;

using TV = FLUID::Vec3f;
using IV = FLUID::Vec3i;

float vdb_phi(const FLUID::Vec3f &position) {
  float phi = 1.f;
  for (auto &grid : boundaries) {
    openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
        grid->constTree(), grid->transform());
    openvdb::math::Vec3<float> P(position[0], position[1], position[2]);
    auto tmp = interpolator.wsSample(P); // ws denotes world space
    if (tmp < phi)
      phi = tmp;
    if (phi < 0.f)
      break;
  }
  return (float)phi;
}

// define vdbgrid;
float box_phi(const FLUID::Vec3f &position, const FLUID::Vec3f &centre,
              FLUID::Vec3f &b) {
  // vec3 d = abs(p) - b;
  // return min(max(d.x,max(d.y,d.z)),0.0) +
  //	length(max(d,0.0));

  FLUID::Vec3f p = position - centre;
  FLUID::Vec3f d = FLUID::fabs(p) - b;
  return min(max(d[0], max(d[1], d[2])), 0.0f) +
         dist(FLUID::Vec3f(max(d[0], 0.0f), max(d[1], 0.0f), max(d[2], 0.0f)),
              FLUID::Vec3f(0, 0, 0));
}

float boundary_phi(const FLUID::Vec3f &position) {
  //    return -sphere_phi(position, FLUID::Vec3f(0,0,0),0.7f);
  FLUID::Vec3f b(0.4f, 1.0f, 0.2f);
  return -box_phi(position, FLUID::Vec3f(0, 0, 0), b);
  // lookup vdbphi
}

float liquid_phi(const FLUID::Vec3f &position) {
  return sphere_phi(position, FLUID::Vec3f(0.5, 0.5, 0.5), 0.18f);
  // FLUID::Vec3f b(0.2f, 0.2f, 0.2f);
  // return box_phi(position, FLUID::Vec3f(-0.2,-0.2,0), b);
}
float dam0_liquid_phi(const FLUID::Vec3f &position) {
  return cuboid_phi(position, FLUID::Vec3f(0.196372, 0.0913078, 1.41758),
                    FLUID::Vec3f(1.03927, 2.24647, 9.58837));
}
float dam1_liquid_phi(const FLUID::Vec3f &position) {
  return cuboid_phi(position, FLUID::Vec3f(1.1025, 0.0913078, 0.226317),
                    FLUID::Vec3f(8.83574, 2.24647, 1.06922));
}

//
////Main testing code
int advection_type = 0;
////-------------
int main(int argc, char **argv) {
  // read vdb levelset

  std::vector<FLIP_particle> outParticles;

  openvdb::initialize();
  // wxl
  parse_scene(argv[1], sim);
  // end wxl
  std::vector<openvdb::Vec3s> points;
  std::vector<openvdb::Vec3I> triangles;
  if (argc != 2) {
    cerr << "The first parameter should be the folder to write the output "
            "liquid meshes into. (eg. ./output/)"
         << "advection_method" << endl;
    return 1;
  }

  // string outpath(argv[1]);
  string outpath(simConfigs.outputDir);
  // sscanf(argv[2],"%d", &advection_type);

  printf("Initializing liquid\n");
#if 0
   sim.set_liquid(FLUID::Vec3f(-0.2, -0.2, -0.2), FLUID::Vec3f(0.2, 0.2, 0.2),
                 liquid_phi);
#elif 0
  sim.set_liquid(FLUID::Vec3f(-0.2, -0.2, -0.2), FLUID::Vec3f(1.2, 1.2, 1.2),
                 liquid_phi); //< wxl
#elif 0 //< flood
  sim.set_liquid(FLUID::Vec3f(0.0, 0.0, 0.0), FLUID::Vec3f(2, 3, 10),
                 dam0_liquid_phi);
  sim.set_liquid(FLUID::Vec3f(1.0, 0.0, 0.0), FLUID::Vec3f(9, 3, 2),
                 dam1_liquid_phi);
#endif
  // std::cout << "particles:" << sim.particles.size() << std::endl;
//  sim.setSolverRegion(FLUID::Vec3f(-2.0, -2.0, -2.0),
//                      FLUID::Vec3f(2.0, 2.0, 2.0));
  // vdbToolsWapper::export_VDB(outpath, 0, sim.particles, sim.particle_radius,
  //                           points, triangles, sim.eulerian_fluids);

  //	sim.set_liquid(FLUID::Vec3f(0.15,0.18,0.025),FLUID::Vec3f(0.35,0.5,0.475));
  sim.init_domain();
  printf("Initializing boundary\n");

  printf("Exporting initial data\n");
  for (double T=0;T<simConfigs.T;T+=simConfigs.dt) {

    printf("--------------------\nFrame %d\n", frame);

    // Simulate
    sim.setGravity(simConfigs.g);
    printf("Simulating liquid\n");
    { sim.advance(simConfigs.dt, vdb_phi); } //< wxl
    sim.eulerian_fluids.write_bulk_obj(outpath, frame);
#if 0
    // particle2levelset
    auto ls =
        vdbToolsWapper::particleToLevelset(sim.particles, sim.particle_radius);
    // new particles, only get particle near surface
    std::vector<FLIP_particle> newParticles;
    newParticles.reserve(sim.particles.size());
    for (int i = 0; i < sim.particles.size(); i++) {
      float phi = vdb_phi(sim.particles[i].pos, ls);
      if (phi <= 0 && phi > -sim.dx)
        newParticles.push_back(sim.particles[i]);
    }

    vdbToolsWapper::export_VDB_Mesh(outpath, frame, sim.particles,
                                    sim.particle_radius, points, triangles,
                                    sim.eulerian_fluids);
#endif
    //if (frame % 10 == 0) {
    if (true) {
      vdbToolsWapper::outputBgeo(outpath, frame, sim.particles);
      vdbToolsWapper::outputBin(outpath, frame, sim.particles);
    } else {
      sim.find_surface_particles(outParticles);
      vdbToolsWapper::outputBgeo(outpath, frame, outParticles);
    }
    // vdbToolsWapper::outputBgeo(outpath, frame, newParticles);
    //	    vdbToolsWapper::outputBgeo(outpath,frame,points,
    // sim.eulerian_fluids);
    printf("Exporting particle data\n");
    frame++;
  }

  return 0;
}
