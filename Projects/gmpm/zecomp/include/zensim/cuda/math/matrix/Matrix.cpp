#include "Matrix.hpp"

#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/profile/CudaTimers.cuh"
#include "zensim/tpls/fmt/color.h"

namespace zs {

  template <typename V, typename I> CudaYaleSparseMatrix<V, I>::~CudaYaleSparseMatrix() noexcept {
    cusparseDestroyMatDescr(descr);
    cusolverSpDestroyCsrcholInfo(cholInfo);
  }

  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::analyze_pattern(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{Cuda::context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};
    std::size_t sizeInternal, sizeChol;
    timer.tick();
    if (this->isRowMajor())
      pol.call(cusolverSpXcsrcholAnalysis, this->rows(), this->nnz(), this->descr,
               this->offsets.data(), this->indices.data(), cholInfo);
    timer.tock("[gpu] analyze pattern");
    // else
    //   pol.call(cusolverSpXcsccholAnalysis, this->cols(), this->nnz(), this->descr,
    //            this->offsets.data(), this->indices.data(), cholInfo);

    if constexpr (is_same_v<V, double>) {
      if (this->isRowMajor())
        pol.call(cusolverSpDcsrcholBufferInfo, this->rows(), this->nnz(), this->descr,
                 this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
                 &sizeInternal, &sizeChol);
      // else
      //   pol.call(cusolverSpDcsccholBufferInfo, this->cols(), this->nnz(), this->descr,
      //            this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
      //            &sizeInternal, &sizeChol);
    } else if constexpr (is_same_v<V, float>) {
      if (this->isRowMajor())
        pol.call(cusolverSpScsrcholBufferInfo, this->rows(), this->nnz(), this->descr,
                 this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
                 &sizeInternal, &sizeChol);
      // else
      //   pol.call(cusolverSpScsccholBufferInfo, this->cols(), this->nnz(), this->descr,
      //            this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
      //            &sizeInternal, &sizeChol);
    }
    auxCholBuffer.resize(sizeChol);
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::factorize(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    int singularity{-2};
    CudaTimer timer{Cuda::context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};
    if constexpr (is_same_v<V, double>) {
      timer.tick();
      pol.call(cusolverSpDcsrcholFactor, this->rows(), this->nnz(), this->descr, this->vals.data(),
               this->offsets.data(), this->indices.data(), cholInfo, auxCholBuffer.data());
      timer.tock("[gpu] cholesky factorization, A = L*L^T");
      pol.call(cusolverSpDcsrcholZeroPivot, cholInfo, 1e-8, &singularity);
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();
      pol.call(cusolverSpScsrcholFactor, this->rows(), this->nnz(), this->descr, this->vals.data(),
               this->offsets.data(), this->indices.data(), cholInfo, auxCholBuffer.data());
      timer.tock("[gpu] cholesky factorization, A = L*L^T");
      pol.call(cusolverSpScsrcholZeroPivot, cholInfo, 1e-6, &singularity);
    }
    if (0 <= singularity) {
      fmt::print(fg(fmt::color::yellow), "error [gpu] A is not invertible, singularity={}\n",
                 singularity);
    }
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::solve(
      zs::Vector<V> &x, const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol,
      const zs::Vector<V> &rhs) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{Cuda::context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};
    if constexpr (is_same_v<V, double>) {
      timer.tick();
      pol.call(cusolverSpDcsrcholSolve, this->rows(), rhs.data(), x.data(), cholInfo,
               auxCholBuffer.data());
      timer.tock("[gpu] system solve using cholesky factorization info");
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();
      pol.call(cusolverSpScsrcholSolve, this->rows(), rhs.data(), x.data(), cholInfo,
               auxCholBuffer.data());
      timer.tock("[gpu] system solve using cholesky factorization info");
    }
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::cgsolve(
      zs::Vector<V> &x,
      const CudaLibExecutionPolicy<culib_cusparse, culib_cublas, culib_cusolversp> &pol,
      const zs::Vector<V> &rhs) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");

    if (x.size() == 0 || rhs.size() == 0 || x.size() != rhs.size()) return;

    CudaTimer timer{Cuda::context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};

    if constexpr (is_same_v<V, double>) {
      timer.tick();
      const auto &sparse
          = static_cast<const CudaLibComponentExecutionPolicy<culib_cusparse> &>(pol);
      const auto &blas = static_cast<const CudaLibComponentExecutionPolicy<culib_cublas> &>(pol);
      // const auto &solversp
      //    = static_cast<const CudaLibComponentExecutionPolicy<culib_cusolversp> &>(pol);
      double r1;

      V a, b, na, dot;
      double alpha = 1.0;
      double alpham1 = -1.0;
      double beta = 0.0;
      double r0 = 0.;
      // zs::Vector<V> r = rhs, p = x, Ax = x;
      zs::Vector<V> r{rhs};
      zs::Vector<V> p{x};
      zs::Vector<V> Ax{x};
#if 0
      {
        fmt::print("ref x: size {}, addr {}\n", x.size(), x.self().address());
        {
          auto xx = x;
          fmt::print("assigned copied x: size {}, addr {}\n", xx.size(), xx.self().address());
        }
        {
          auto xx{x};
          fmt::print("ref copied x: size {}, addr {}\n", xx.size(), xx.self().address());
        }
        getchar();
      }
#endif
      /// descriptor for matA
      sparse.call(cusparseCreateCsr, &spmDescr, this->rows(), this->cols(), this->nnz(),
                  this->offsets.data(), this->indices.data(), this->vals.data(), CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
#if 0
      fmt::print("size: {}\n", x.size());
      fmt::print("x addr {}, rhs addr {}\n", (std::uintptr_t)x.data(), (std::uintptr_t)rhs.data());
      fmt::print("p addr {}\n", (std::uintptr_t)p.data());
      fmt::print("Ax addr {}\n", (std::uintptr_t)Ax.data());
#endif
      /// descriptor for vecx
      cusparseDnVecDescr_t vecx = NULL;
      sparse.call(cusparseCreateDnVec, &vecx, this->rows(), x.data(), CUDA_R_64F);
      /// descriptor for vecp
      cusparseDnVecDescr_t vecp = NULL;
      sparse.call(cusparseCreateDnVec, &vecp, this->rows(), p.data(), CUDA_R_64F);
      /// descriptor for vecAx
      cusparseDnVecDescr_t vecAx = NULL;
      sparse.call(cusparseCreateDnVec, &vecAx, this->rows(), Ax.data(), CUDA_R_64F);

      /* Allocate workspace for cuSPARSE */
      size_t bufferSize = 0;
      sparse.call(cusparseSpMV_bufferSize, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecx,
                  &beta, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
      auxCholBuffer.resize(bufferSize);

      /// matA * vecx -> vecAx
      sparse.call(cusparseSpMV, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecx, &beta,
                  vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, auxCholBuffer.data());

      /// r - Ax
      blas.call(cublasDaxpy, this->rows(), &alpham1, Ax.data(), 1, r.data(), 1);
      /// r * r -> r1
      blas.call(cublasDdot, this->rows(), r.data(), 1, r.data(), 1, &r1);

      int k = 1;

      const V tol = 1e-9;
      const int max_iter = 10000;
      while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
          b = r1 / r0;
          blas.call(cublasDscal, this->rows(), &b, p.data(), 1);
          blas.call(cublasDaxpy, this->rows(), &alpha, r.data(), 1, p.data(), 1);
          // cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
          // cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        } else {
          /// p = r
          blas.call(cublasDcopy, this->rows(), r.data(), 1, p.data(), 1);
          // cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        /// Ap = A * p
        sparse.call(cusparseSpMV, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecp, &beta,
                    vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, auxCholBuffer.data());

        /// dot = p.dot(Ap)
        blas.call(cublasDdot, this->rows(), p.data(), 1, Ax.data(), 1, &dot);
        // cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        /// alpha =
        a = r1 / dot;

        /// x = x + alpha * p
        blas.call(cublasDaxpy, this->rows(), &a, p.data(), 1, x.data(), 1);
        // cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        /// r = r - alpha * Ap
        blas.call(cublasDaxpy, this->rows(), &na, Ax.data(), 1, r.data(), 1);
        // cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        blas.call(cublasDdot, this->rows(), r.data(), 1, r.data(), 1, &r1);
        // cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
      }

      timer.tock("[gpu] system conjugate gradient solve");
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();

      timer.tock("[gpu] system conjugate gradient solve");
    }
  }

  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::pcgsolve(
      const zs::Vector<V> &mass, zs::Vector<V> &x,
      const CudaLibExecutionPolicy<culib_cusparse, culib_cublas, culib_cusolversp> &pol,
      const zs::Vector<V> &rhs) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");

    if (x.size() == 0 || rhs.size() == 0 || x.size() != rhs.size()) return;

    CudaTimer timer{Cuda::context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};

    if constexpr (is_same_v<V, double>) {
      timer.tick();
      const auto &sparse
          = static_cast<const CudaLibComponentExecutionPolicy<culib_cusparse> &>(pol);
      const auto &blas = static_cast<const CudaLibComponentExecutionPolicy<culib_cublas> &>(pol);
      // const auto &solversp
      //    = static_cast<const CudaLibComponentExecutionPolicy<culib_cusolversp> &>(pol);
      double r1;

      V a, b, na, dot;
      double alpha = 1.0;
      double alpham1 = -1.0;
      double beta = 0.0;
      double r0 = 0.;
      // zs::Vector<V> r = rhs, p = x, Ax = x;
      zs::Vector<V> r = rhs;
      zs::Vector<V> p = x;
      zs::Vector<V> Ax = x;
      zs::Vector<V> z = x;

      auto cudaPol = zs::cuda_exec().device(0);

#if 0
      {
        fmt::print("ref x: size {}, addr {}\n", x.size(), x.self().address());
        {
          auto xx = x;
          fmt::print("assigned copied x: size {}, addr {}\n", xx.size(), xx.self().address());
        }
        {
          auto xx{x};
          fmt::print("ref copied x: size {}, addr {}\n", xx.size(), xx.self().address());
        }
        getchar();
      }
#endif
      /// descriptor for matA
      sparse.call(cusparseCreateCsr, &spmDescr, this->rows(), this->cols(), this->nnz(),
                  this->offsets.data(), this->indices.data(), this->vals.data(), CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
#if 0
      fmt::print("size: {}\n", x.size());
      fmt::print("x addr {}, rhs addr {}\n", (std::uintptr_t)x.data(), (std::uintptr_t)rhs.data());
      fmt::print("p addr {}\n", (std::uintptr_t)p.data());
      fmt::print("Ax addr {}\n", (std::uintptr_t)Ax.data());
#endif
      /// descriptor for vecx
      cusparseDnVecDescr_t vecx = NULL;
      sparse.call(cusparseCreateDnVec, &vecx, this->rows(), x.data(), CUDA_R_64F);
      /// descriptor for vecp
      cusparseDnVecDescr_t vecp = NULL;
      sparse.call(cusparseCreateDnVec, &vecp, this->rows(), p.data(), CUDA_R_64F);
      /// descriptor for vecAx
      cusparseDnVecDescr_t vecAx = NULL;
      sparse.call(cusparseCreateDnVec, &vecAx, this->rows(), Ax.data(), CUDA_R_64F);
      /// descriptor for z
      cusparseDnVecDescr_t vecz = NULL;
      sparse.call(cusparseCreateDnVec, &vecz, this->rows(), z.data(), CUDA_R_64F);

      /* Allocate workspace for cuSPARSE */
      size_t bufferSize = 0;
      sparse.call(cusparseSpMV_bufferSize, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecx,
                  &beta, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
      auxCholBuffer.resize(bufferSize);

      /// matA * vecx -> vecAx
      sparse.call(cusparseSpMV, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecx, &beta,
                  vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, auxCholBuffer.data());

      /// rhs - Ax -> r
      blas.call(cublasDaxpy, this->rows(), &alpham1, Ax.data(), 1, r.data(), 1);
      /// Minv * r -> z
      cudaPol({z.size()}, [dim = r.size() / mass.size(), zz = proxy<execspace_e::cuda>(z),
                           rr = proxy<execspace_e::cuda>(r),
                           m = proxy<execspace_e::cuda>(mass)] __device__(int i) mutable {
        zz(i) = rr(i) / m(i / dim);
      });
      /// r * z -> rz
      blas.call(cublasDdot, this->rows(), r.data(), 1, z.data(), 1, &r1);

      int k = 1;

      const V tol = 1e-9;
      const int max_iter = 10000;
      while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
          b = r1 / r0;
          blas.call(cublasDscal, this->rows(), &b, p.data(), 1);
          blas.call(cublasDaxpy, this->rows(), &alpha, z.data(), 1, p.data(), 1);
          // cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
          // cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        } else {
          /// p = r
          blas.call(cublasDcopy, this->rows(), z.data(), 1, p.data(), 1);
          // cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        /// Ap = A * p
        sparse.call(cusparseSpMV, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecp, &beta,
                    vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, auxCholBuffer.data());

        /// dot = p.dot(Ap)
        blas.call(cublasDdot, this->rows(), p.data(), 1, Ax.data(), 1, &dot);
        // cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        /// alpha =
        a = r1 / dot;

        /// x = x + alpha * p
        blas.call(cublasDaxpy, this->rows(), &a, p.data(), 1, x.data(), 1);
        // cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        /// r = r - alpha * Ap
        blas.call(cublasDaxpy, this->rows(), &na, Ax.data(), 1, r.data(), 1);
        // cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        /// Minv * r -> z
        cudaPol({z.size()}, [dim = r.size() / mass.size(), zz = proxy<execspace_e::cuda>(z),
                             rr = proxy<execspace_e::cuda>(r),
                             m = proxy<execspace_e::cuda>(mass)] __device__(int i) mutable {
          zz(i) = rr(i) / m(i / dim);
        });

        r0 = r1;
        blas.call(cublasDdot, this->rows(), r.data(), 1, z.data(), 1, &r1);
        // cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
      }

      timer.tock("[gpu] system preconditioned conjugate gradient solve");
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();

      timer.tock("[gpu] system preconditioned conjugate gradient solve");
    }
  }

  template struct CudaYaleSparseMatrix<f32, i32>;
  template struct CudaYaleSparseMatrix<f64, i32>;

}  // namespace zs