#pragma once
#include "Structures.hpp"

namespace zeno {

void compute_surface_neighbors(zs::CudaExecutionPolicy &pol, ZenoParticles::particles_t &tris,
                               ZenoParticles::particles_t &lines, ZenoParticles::particles_t &verts);

void update_surface_cell_normals(zs::CudaExecutionPolicy &pol, ZenoParticles::particles_t &verts,
                                 const zs::SmallString &xTag, std::size_t vOffset, ZenoParticles::particles_t &tris,
                                 const zs::SmallString &triNrmTag, ZenoParticles::particles_t &lines,
                                 const zs::SmallString &biNrmTag);


} // namespace zeno