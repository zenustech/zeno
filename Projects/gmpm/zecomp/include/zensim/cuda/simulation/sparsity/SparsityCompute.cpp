#include "zensim/simulation/sparsity/SparsityCompute.tpp"

#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  template GeneralHashTable partition_for_particles<execspace_e::cuda>(GeneralParticles&, float,
                                                                       int);

}  // namespace zs