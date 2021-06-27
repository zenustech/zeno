#pragma once
#include "Givens.hpp"

namespace zs {

  namespace math {

    /**
     \brief 2x2 polar decomposition.
     \param[in] A matrix.
     \param[out] R Robustly a rotation matrix in givens form
     \param[out] S_Sym Symmetric. Whole matrix is stored

     Whole matrix S is stored since its faster to calculate due to simd vectorization
     Polar guarantees negative sign is on the small magnitude singular value.
     S is guaranteed to be the closest one to identity.
     R is guaranteed to be the closest rotation to A.
     */
    template <typename T>
    constexpr void polarDecomposition(const T A[4], GivensRotation<T>& R, T S[4]) {
      double x[2] = {A[0] + A[3], A[1] - A[2]};
      double denominator = gcem::sqrt(x[0] * x[0] + x[1] * x[1]);
      R.c = (T)1, R.s = (T)0;
      if (denominator != 0) {
        /*
          No need to use a tolerance here because x(0) and x(1) always have
          smaller magnitude then denominator, therefore overflow never happens.
        */
        R.c = x[0] / denominator;
        R.s = -x[1] / denominator;
      }
      for (int i = 0; i < 4; ++i) S[i] = A[i];
      R.template matRotation<2, T>(S);
    }

    /**
       \brief 2x2 polar decomposition.
       \param[in] A matrix.
       \param[out] R Robustly a rotation matrix.
       \param[out] S_Sym Symmetric. Whole matrix is stored

       Whole matrix S is stored since its faster to calculate due to simd vectorization
       Polar guarantees negative sign is on the small magnitude singular value.
       S is guaranteed to be the closest one to identity.
       R is guaranteed to be the closest rotation to A.
    */
    template <typename T>
    constexpr void polarDecomposition(const T A[4], const T R[4], const T S[4]) {
      GivensRotation<T> r(0, 1);
      polarDecomposition(A, r, S);
      r.fill<2>(R);
    }

  }  // namespace math

}  // namespace zs
