#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include "AlgebraicMultigrid.h"
#include "GeometricLevelGen.h"
#include "Sparse_buffer.h"
#include "levelset_util.h"
#include "pcg_solver.h"
#include "sparse_matrix.h"
#include "util.h"
#include "volumeMeshTools.h"
#include <tbb/concurrent_vector.h>


class FluidSim {
    using GridT =
    typename openvdb::Grid<typename openvdb::tree::Tree4<float, 5, 4, 3>::Type>;
    using TreeT = typename GridT::TreeType;
    using IterRange = openvdb::tree::IteratorRange<typename TreeT::LeafCIter>;

public:
  static float H(float r) {
    float res = 0;
    if (r >= -1 && r < 0)
      res = 1 + r;
    if (r >= 0 && r <= 1)
      res = 1 - r;
    return res;
  }
  static float compute_weight(FLUID::Vec3f gp, FLUID::Vec3f pp, float dx) {
    // k(x,y,z) = H(dx/hx)H(dy/hx)H(dz/hx)
    // H(r) = 1-r 0<=r<1  1+r -1<=r<0 0 else;
    FLUID::Vec3f dd = gp - pp;
    return H(dd[0] / dx) * H(dd[1] / dx) * H(dd[2] / dx);
  }
  struct levelsetEmitter {
      FLUID::Vec3f vel;
      typename GridT::Ptr ls;
  };
  struct boxEmitter {
    FLUID::Vec3f vel;
    FLUID::Vec3f bmin;
    FLUID::Vec3f bmax;
    boxEmitter() {
      vel = FLUID::Vec3f(0);
      bmin = FLUID::Vec3f(0);
      bmax = FLUID::Vec3f(0);
    }
    boxEmitter(const boxEmitter &e) {
      vel = e.vel;
      bmin = e.bmin;
      bmax = e.bmax;
    }
  };

  void initialize(double _dx);
  void set_boundary(float (*phi)(const FLUID::Vec3f &));
  void set_liquid(const FLUID::Vec3f &bmin, const FLUID::Vec3f &bmax,
                  std::function<float(const FLUID::Vec3f &)> phi);
  void init_domain();
  void resolveParticleBoundaryCollision();
  void setSolverRegion(const FLUID::Vec3f &bmin, const FLUID::Vec3f &bmax) {
    regionMin = bmin;
    regionMax = bmax;
  }
  void setEmitter(FLUID::Vec3f &bmin, FLUID::Vec3f &bmax, FLUID::Vec3f &vel) {
    boxEmitter e;
    e.bmin = bmin;
    e.bmax = bmax;
    e.vel = vel;
    emitters.push_back(e);
  }
  void advance(float dt, float (*phi)(const FLUID::Vec3f &));
  void setGravity(float g=-9.8) { gravity = g; }
  bool isIsolatedParticle(FLUID::Vec3f &pos);

  // Grid dimensions
  float dx;
  uint total_frame;

  // Eulerian Fluid
  sparse_fluid8x8x8 eulerian_fluids;
  sparse_fluid8x8x8 resample_field;
  vector<vector<uint>> particle_bulks;
  std::vector<FLIP_particle> particles;
  float particle_radius;
  float cfl_dt;
  float gravity = -9.81;

  SparseMatrixd matrix;
  std::vector<double> rhs;
  std::vector<double> Dofs;
  FLUID::Vec3f regionMin, regionMax;
  std::vector<boxEmitter> emitters;
  void project(float dt, std::vector<FLIP_particle> &p, float order_coef, float (*phi)(const FLUID::Vec3f &));
    bool inFluid(FLUID::Vec3f pos);
    bool inDomain(FLUID::Vec3f pos);
  FLUID::Vec3f getDelta(FLUID::Vec3f &pos)
  {
      return eulerian_fluids.get_delta_vel(pos);
  }
  void reorder_particles();
  void emitFluids(float dt, float (*phi)(const FLUID::Vec3f &));
static  float sampleEmitter(FLUID::Vec3f &pos, levelsetEmitter & lse);
static void emit(std::vector<FLIP_particle>& p, levelsetEmitter &lse, sparse_fluid8x8x8 &_eulerian_fluids, FluidSim* _emitter_sample_field, float gap, float (*phi)(const FLUID::Vec3f &));
  void emit(float (*phi)(const FLUID::Vec3f &));
  void emitRegion(float (*phi)(const FLUID::Vec3f &), float dt);
  void setEmitterSampleField(FluidSim* field)
  {
      EmitterSampleField = field;
  }
  FLUID::Vec3f getVelocity(FLUID::Vec3f pos)
  {
      if(eulerian_fluids.get_liquid_phi(pos)<=0)
      {
          return eulerian_fluids.get_velocity(pos);
      }
      else
      {
          if(EmitterSampleField!= nullptr)
          {
              return EmitterSampleField->getVelocity(pos);
          }
          else
          {
              return eulerian_fluids.get_velocity(pos);
          }
      }
  }
  // FLUID::Vec3f get_dvelocity(const FLUID::Vec3f & position);
  // void compute_delta(Array3f & u, Array3f &u_old, Array3f &u_temp);
  static void subset_particles(std::vector<FLIP_particle> &pin, std::vector<FLIP_particle> &pout, std::vector<char> &mask);
  void boundaryModel(float dt, float nu, std::vector<FLIP_particle> &_p, float (*phi)(const FLUID::Vec3f &));
  void particle_interpolate(float alpha);
  void FLIP_advection(float dt);
  void particle_to_grid();
  void particle_to_grid_mask();
  void find_surface_particles(std::vector<FLIP_particle> &outParticles);
  void particle_to_grid(sparse_fluid8x8x8 &_eulerian_fluid,
                        std::vector<FLIP_particle> &_particles, float dx);
  static void fusion_p2g_liquid_phi(sparse_fluid8x8x8 &_eulerian_fluid,
                             std::vector<FLIP_particle> &_particles, float _dx, float _particle_radius);
  void  postAdvBoundary();
  void extrapolate(sparse_fluid8x8x8 &_eulerian_fluid, int times);
  void remeshing();

  FLUID::Vec3f trace_rk3(const FLUID::Vec3f &position, float dt);

  float cfl();

  void advect_particles(float dt);

  void add_force(float dt);
  void project(float dt);
  void extrapolate(int times);
  void constrain_velocity();

  ////helpers for pressure projection
  void compute_weights();
  void solve_pressure(float dt);
  void solve_pressure_parallel_build(float dt);
  void compute_phi();
  ////void computeGradPhi(Array3f & u_temp, int dir);
  ////void advectPotential(float dt);
  void resampleVelocity(std::vector<FLIP_particle> &_particles, float _dx,
                        std::vector<FLIP_particle> &_resample_pos) {
    resample_field.initialize_bulks(_particles, _dx);
    particle_to_grid(resample_field, _particles, _dx);
    extrapolate(resample_field, 4);
    tbb::parallel_for(
        (size_t)0, (size_t)_resample_pos.size(), (size_t)1, [&](size_t index) {
          _resample_pos[index].vel =
              resample_field.get_velocity(_resample_pos[index].pos);
        });
  }
  void addEmitter(FLUID::Vec3f &vel, typename GridT::Ptr g)
  {
      levelsetEmitter e;
      e.vel = vel;
      e.ls = g;
      Emitters.push_back(e);
  }
  std::vector<levelsetEmitter> Emitters;
  FluidSim* EmitterSampleField = nullptr;
  std::vector<typename GridT::Ptr> fluidDomains;
  bool fillRegion = false;
};

#endif
