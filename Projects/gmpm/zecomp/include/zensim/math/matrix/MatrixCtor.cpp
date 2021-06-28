#include "MatrixCtor.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename V, typename I> void spm_set_pattern(const SequentialExecutionPolicy &pol,
                                                         YaleSparseMatrix<V, I> &mat,
                                                         const Vector<I> &rs, const Vector<I> &cs) {
    assert_with_msg(mat.devid() == 0, "[spm_set_pattern](seq_exec): memory space not host!");
  }
  template <typename V, typename I>
  void spm_from_triplets(const SequentialExecutionPolicy &pol, YaleSparseMatrix<V, I> &mat,
                         const Vector<I> &rs, const Vector<I> &cs, const Vector<V> &vs) {
    assert_with_msg(mat.devid() == 0, "[spm_from_triplets](seq_exec): memory space not host!");
  }

#define INSTANTIATE_SPARSE_MATRIX_CTOR(V, I)                                                       \
  template void spm_set_pattern<V, I>(const SequentialExecutionPolicy &, YaleSparseMatrix<V, I> &, \
                                      const Vector<I> &, const Vector<I> &);                       \
  template void spm_from_triplets<V, I>(const SequentialExecutionPolicy &,                         \
                                        YaleSparseMatrix<V, I> &, const Vector<I> &,               \
                                        const Vector<I> &, const Vector<V> &);

  INSTANTIATE_SPARSE_MATRIX_CTOR(f32, i32)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f32, i64)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f32, u32)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f32, u64)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f64, i32)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f64, i64)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f64, u32)
  INSTANTIATE_SPARSE_MATRIX_CTOR(f64, u64)

}  // namespace zs