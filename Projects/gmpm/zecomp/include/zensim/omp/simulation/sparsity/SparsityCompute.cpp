#include "zensim/simulation/sparsity/SparsityCompute.tpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  template GeneralHashTable partition_for_particles<execspace_e::openmp>(GeneralParticles&, float, int);

}  // namespace zs