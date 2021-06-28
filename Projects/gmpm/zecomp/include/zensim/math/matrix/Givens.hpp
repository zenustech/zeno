/**
Givens rotation
*/
#pragma once
#include <type_traits>

#include "zensim/tpls/gcem/gcem.hpp"

namespace zs {

  namespace math {
    /**
            Class for givens rotation.
            Row rotation G*A corresponds to something like
            c -s  0
            ( s  c  0 ) A
            0  0  1
            Column rotation A G' corresponds to something like
            c -s  0
            A ( s  c  0 )
            0  0  1

            c and s are always computed so that
            ( c -s ) ( a )  =  ( * )
            s  c     b       ( 0 )

            Assume rowi<rowk.
            */
    template <typename T> struct GivensRotation {
    public:
      int rowi;
      int rowk;
      T c;
      T s;

      constexpr GivensRotation(int rowi_in, int rowk_in)
          : rowi(rowi_in), rowk(rowk_in), c(1), s(0) {}

      constexpr GivensRotation(T a, T b, int rowi_in, int rowk_in) : rowi(rowi_in), rowk(rowk_in) {
        compute(a, b);
      }

      ~GivensRotation() = default;

      constexpr void setIdentity() {
        c = 1;
        s = 0;
      }

      constexpr void transposeInPlace() { s = -s; }

      /**
              Compute c and s from a and b so that
              ( c -s ) ( a )  =  ( * )
              s  c     b       ( 0 )
              */
      template <typename TT>
      constexpr std::enable_if_t<std::is_same<TT, float>::value, void> compute(const TT a,
                                                                               const TT b) {
        TT d = a * a + b * b;
        c = 1;
        s = 0;
        TT sqrtd = gcem::sqrt(d);
        // T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
          TT t = 1 / sqrtd;
          c = a * t;
          s = -b * t;
        }
      }

      template <typename TT>
      constexpr std::enable_if_t<std::is_same<TT, double>::value, void> compute(const TT a,
                                                                                const TT b) {
        TT d = a * a + b * b;
        c = 1;
        s = 0;
        TT sqrtd = gcem::sqrt(d);
        // T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
          TT t = 1 / sqrtd;
          c = a * t;
          s = -b * t;
        }
      }

      /**
              This function computes c and s so that
              ( c -s ) ( a )  =  ( 0 )
              s  c     b       ( * )
              */
      template <typename TT>
      constexpr std::enable_if_t<std::is_same<TT, float>::value, void> computeUnconventional(
          const TT a, const TT b) {
        TT d = a * a + b * b;
        c = 0;
        s = 1;
        TT sqrtd = gcem::sqrt(d);
        // T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
          TT t = 1 / sqrtd;
          s = a * t;
          c = b * t;
        }
      }

      template <typename TT>
      constexpr std::enable_if_t<std::is_same<TT, double>::value, void> computeUnconventional(
          const TT a, const TT b) {
        TT d = a * a + b * b;
        c = 0;
        s = 1;
        TT sqrtd = gcem::sqrt(d);
        // T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
          TT t = 1 / sqrtd;
          s = a * t;
          c = b * t;
        }
      }

      /**
        Fill the R with the entries of this rotation
              */
      template <int Dim, typename T1> constexpr void fill(T1 A[Dim * Dim]) const {
        for (int i = 0; i < Dim * Dim; ++i) A[i] = 0;
        for (int i = 0; i < Dim * Dim; i += Dim + 1) A[i] = 1;
        A[rowi + rowi * Dim] = c;
        A[rowk + rowi * Dim] = -s;
        A[rowi + rowk * Dim] = s;
        A[rowk + rowk * Dim] = c;
      }

      /**
              This function does something like Q^T A -> A
              [ c -s  0 ]
              [ s  c  0 ] A -> A
              [ 0  0  1 ]
              It only affects row i and row k of A.
              */
      template <int Dim, typename T1> constexpr void matRotation(T1 A[Dim * Dim]) const {
        for (int d = 0; d < Dim; d++) {
          T1 tau1 = A[rowi + d * Dim];
          T1 tau2 = A[rowk + d * Dim];
          A[rowi + d * Dim] = c * tau1 - s * tau2;
          A[rowk + d * Dim] = s * tau1 + c * tau2;
        }
      }

      template <int Dim, typename T1> constexpr void vecRotation(T1 A[Dim]) const {
        T1 tau1 = A[rowi];
        T1 tau2 = A[rowk];
        A[rowi] = c * tau1 - s * tau2;
        A[rowk] = s * tau1 + c * tau2;
      }

      /**
        Multiply givens must be for same row and column
        **/
      constexpr void operator*=(const GivensRotation<T>& A) {
        T new_c = c * A.c - s * A.s;
        T new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
      }

      /**
        Multiply givens must be for same row and column
        **/
      constexpr GivensRotation<T> operator*(const GivensRotation<T>& A) const {
        GivensRotation<T> r(*this);
        r *= A;
        return r;
      }
    };

  }  // namespace math

}  // namespace zs
