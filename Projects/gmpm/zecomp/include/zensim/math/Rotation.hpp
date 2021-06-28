#pragma once
#include "Vec.h"
#include "zensim/tpls/gcem/gcem.hpp"

namespace zs {

  /**
     Imported from ZIRAN, wraps Eigen's Rotation2D (in 2D) and Quaternion (in 3D).
   */
  template <typename T, int dim> struct Rotation : vec<T, dim, dim> {
    using TV = vec<T, dim>;
    using TM = vec<T, dim, dim>;

    constexpr auto &self() noexcept { return static_cast<TM &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const TM &>(*this); }

    constexpr Rotation() noexcept : TM{} {
      for (int d = 0; d < dim; ++d) (*this)(d, d) = (T)1;
    }
    constexpr Rotation(const vec<T, 4> &q) noexcept : TM{} {
      if constexpr (dim == 2) {
        /// Construct a 2D counter clock wise rotation from the angle \a a in
        /// radian.
        T sinA = gcem::sin(q(0)), cosA = gcem::cos(q(0));
        (*this)(0, 0) = cosA;
        (*this)(0, 1) = -sinA;
        (*this)(1, 0) = sinA;
        (*this)(1, 1) = cosA;
      } else if constexpr (dim == 3) {
        /// The quaternion is required to be normalized, otherwise the result is
        /// undefined.
        self() = quaternion2matrix(q);
      }
    }
    constexpr Rotation(const TV &a, const TV &b) noexcept : TM{} {
      if constexpr (dim == 2) {
        TV aa = a.normalized();
        TV bb = b.normalized();
        (*this)(0, 0) = aa(0) * bb(0) + aa(1) * bb(1);
        (*this)(0, 1) = -(aa(0) * bb(1) - bb(0) * aa(1));
        (*this)(1, 0) = aa(0) * bb(1) - bb(0) * aa(1);
        (*this)(1, 1) = aa(0) * bb(0) + aa(1) * bb(1);
      } else if constexpr (dim == 3) {
        T k_cos_theta = a.dot(b);
        T k = gcem::sqrt(a.l2NormSqr() * b.l2NormSqr());
        vec<T, 4> q{};
        if (k_cos_theta / k == -1) {
          // 180 degree rotation around any orthogonal vector
          q(3) = 0;
          auto c = a.orthogonal().normalized();
          q(0) = c(0);
          q(1) = c(1);
          q(2) = c(2);
        } else {
          q(3) = k_cos_theta + k;
          auto c = a.cross(b);
          q(0) = c(0);
          q(1) = c(1);
          q(2) = c(2);
          q = q.normalized();
        }
        self() = quaternion2matrix(q);
      }
    }

    template <int d = dim, enable_if_t<d == 3> = 0>
    static constexpr vec<T, d, d> quaternion2matrix(const vec<T, 4> &q) noexcept {
      /// (0, 1, 2, 3)
      /// (x, y, z, w)
      const T tx = T(2) * q(0);
      const T ty = T(2) * q(1);
      const T tz = T(2) * q(2);
      const T twx = tx * q(3);
      const T twy = ty * q(3);
      const T twz = tz * q(3);
      const T txx = tx * q(0);
      const T txy = ty * q(0);
      const T txz = tz * q(0);
      const T tyy = ty * q(1);
      const T tyz = tz * q(1);
      const T tzz = tz * q(2);
      vec<T, d, d> rot{};
      rot(0, 0) = T(1) - (tyy + tzz);
      rot(0, 1) = txy - twz;
      rot(0, 2) = txz + twy;
      rot(1, 0) = txy + twz;
      rot(1, 1) = T(1) - (txx + tzz);
      rot(1, 2) = tyz - twx;
      rot(2, 0) = txz - twy;
      rot(2, 1) = tyz + twx;
      rot(2, 2) = T(1) - (txx + tyy);
      return rot;
    }
  };

  template <class T, int dim> struct AngularVelocity;

  template <class T> struct AngularVelocity<T, 2> {
    using TV = vec<T, 2>;
    T omega{0};
    constexpr AngularVelocity operator+(const AngularVelocity &o) const noexcept {
      return AngularVelocity{omega + o.omega};
    }
    constexpr AngularVelocity operator*(T alpha) const noexcept {
      return AngularVelocity{omega * alpha};
    }
    constexpr TV cross(const TV &x) const noexcept { return TV{-omega * x(1), omega * x(0)}; }
  };

  template <class T> struct AngularVelocity<T, 3> {
    using TV = vec<T, 3>;
    TV omega{0, 0, 0};
    constexpr AngularVelocity operator+(const AngularVelocity &o) const noexcept {
      return AngularVelocity{omega + o.omega};
    }
    constexpr AngularVelocity operator*(T alpha) const noexcept {
      return AngularVelocity{omega * alpha};
    }
    friend constexpr AngularVelocity operator*(T alpha, const AngularVelocity &o) noexcept {
      return AngularVelocity{o.omega * alpha};
    }
    constexpr TV cross(const TV &x) const noexcept { return omega.cross(x); }
    friend constexpr TV cross(const TV &x, const AngularVelocity &o) noexcept {
      return x.cross(o.omega);
    }
  };

}  // namespace zs
