#pragma once
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolver_common.h>
#include <cusparse_v2.h>

#include "zensim/cuda/execution/CudaLibExecutionPolicy.cuh"
#include "zensim/math/matrix/Matrix.hpp"

namespace zs {

  template <typename ValueType, typename IndexType> struct CudaYaleSparseMatrix
      : YaleSparseMatrix<ValueType, IndexType>,
        MatrixAccessor<CudaYaleSparseMatrix<ValueType, IndexType>> {
    using base_t = YaleSparseMatrix<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    template <typename ExecPol>
    CudaYaleSparseMatrix(ExecPol &pol, memsrc_e mre, ProcID pid,
                         matrix_order_e order = matrix_order_e::rowMajor, index_type nrows = 1,
                         index_type ncols = 1) noexcept
        : YaleSparseMatrix<value_type, index_type>{mre, pid, order, nrows, ncols, 512},
          auxCholBuffer{mre, pid, 512} {
      pol.template call<culib_cusparse>(cusparseCreateMatDescr, &descr);
      pol.template call<culib_cusparse>(cusparseSetMatType, descr, CUSPARSE_MATRIX_TYPE_GENERAL);
      pol.template call<culib_cusparse>(cusparseSetMatIndexBase, descr, CUSPARSE_INDEX_BASE_ZERO);
      pol.template call<culib_cusolversp>(cusolverSpCreateCsrcholInfo, &cholInfo);
      //
#if 0
      pol.addListener(
          dtorEvent.createListener([&pol](cusparseMatDescr_t descr, csrcholInfo_t cholInfo) {
            pol.template call<culib_cusparse>(cusparseDestroyMatDescr, descr);
            pol.template call<culib_cusolversp>(cusolverSpDestroyCsrcholInfo, cholInfo);
          }));
#endif
    }
    ~CudaYaleSparseMatrix() noexcept;

    cusparseMatDescr_t descr{0};
    cusparseSpMatDescr_t spmDescr{0};
    csrcholInfo_t cholInfo{nullptr};

    void analyze_pattern(const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol);
    void factorize(const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol);
    void solve(Vector<value_type> &, const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol,
               const Vector<value_type> &);
    void cgsolve(Vector<value_type> &,
                 const CudaLibExecutionPolicy<culib_cusparse, culib_cublas, culib_cusolversp> &pol,
                 const Vector<value_type> &);
    void pcgsolve(const Vector<value_type> &, Vector<value_type> &,
                  const CudaLibExecutionPolicy<culib_cusparse, culib_cublas, culib_cusolversp> &pol,
                  const Vector<value_type> &);

    // Vector<char> auxSpmBuffer{};
    Vector<char> auxCholBuffer{};
  };

}  // namespace zs