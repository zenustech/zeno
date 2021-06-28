#include "SparsityCompute.tpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  template GeneralHashTable partition_for_particles<execspace_e::host>(GeneralParticles&, float, int);

/*
  GeneralHashTable partition_for_particles(GeneralParticles &particles, float dx, int blocklen) {
#if 0
    auto mh = match([](auto &p) { return p.handle(); })(particles);
    if (mh.onHost())
      return partition_for_particles_host_impl(particles, dx, blocklen);
    else if (mh.devid() >= 0)
      return partition_for_particles_cuda_impl(particles, dx, blocklen);
    else
      throw std::runtime_error(fmt::format("[partition_for_particles] scenario not implemented\n"));
#endif
  }
*/

}  // namespace zs