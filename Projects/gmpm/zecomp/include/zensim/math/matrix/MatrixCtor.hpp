#pragma once
#include "Matrix.hpp"

namespace zs {

  template <typename Pol, typename V, typename I>
  void spm_set_pattern(Pol &&pol, YaleSparseMatrix<V, I> &mat, const Vector<I> &rs,
                       const Vector<I> &cs);
  template <typename Pol, typename V, typename I>
  void spm_from_triplets(Pol &&pol, YaleSparseMatrix<V, I> &mat, const Vector<I> &rs,
                         const Vector<I> &cs, const Vector<V> &vs);

}  // namespace zs