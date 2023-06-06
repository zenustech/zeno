#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

namespace zeno {

template typename ZenoParticles::bv_t
ZenoParticles::computeBoundingVolume<zs::CudaExecutionPolicy>(zs::CudaExecutionPolicy &pol, zs::SmallString xtag) const;

template void ZenoParticles::updateElementIndices<zs::CudaExecutionPolicy>(zs::CudaExecutionPolicy &pol,
                                                                           typename ZenoParticles::particles_t &eles);

template void ZenoParticles::orderByMortonCode<zs::CudaExecutionPolicy, true>(zs::CudaExecutionPolicy &pol,
                                                                        const typename ZenoParticles::bv_t &, zs::wrapv<true>);

} // namespace zeno