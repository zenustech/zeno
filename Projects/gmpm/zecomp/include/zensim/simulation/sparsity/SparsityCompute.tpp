#include "SparsityCompute.hpp"

namespace zs {

  template <execspace_e space>
  GeneralHashTable partition_for_particles(GeneralParticles& particles, float dx, int blocklen) {
    return match([&](auto &p) -> GeneralHashTable { 
      MemoryHandle mh = p.handle(); 
      std::size_t keyCnt = p.size();
      constexpr int dim = remove_cvref_t<decltype(p)>::dim;
      for (auto d = dim; d--;) keyCnt /= blocklen;

      HashTable<i32, dim, int> ret{keyCnt, mh.memspace(), mh.devid()};

      // exec_tags execTag = suggest_exec_space(mh);
      auto execTag = wrapv<space>{};
      auto execPol = par_exec(execTag);
      if constexpr (space == execspace_e::cuda || space == execspace_e::hip)
        execPol.device(mh.devid());

      execPol(range(ret._tableSize), CleanSparsity{execTag, ret});
      execPol(range(p.size()), ComputeSparsity{execTag, dx, blocklen, ret, p.X});
      return ret;
    })(particles);
  }

}