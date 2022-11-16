#pragma once
#include "Structures.hpp"

namespace zeno {

void compute_surface_neighbors(zs::CudaExecutionPolicy &pol, typename ZenoParticles::particles_t &tris,
                               typename ZenoParticles::particles_t &lines, typename ZenoParticles::particles_t &verts);

void compute_surface_edges(zs::CudaExecutionPolicy &pol, const ZenoParticles::particles_t &sfs,
                           ZenoParticles::particles_t *sesPtr, ZenoParticles::particles_t *svsPtr = nullptr);

} // namespace zeno