#pragma once
#include "LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum class analytic_geometry_e { Plane, Cuboid, Sphere, Torus };

  template <analytic_geometry_e geomT, typename DataType, int d> struct AnalyticLevelSet;

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Plane, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Plane, DataType, d>, DataType, d> {
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV origin, TV normal)
        : _origin{origin}, _normal{normal.normalized()} {}

    constexpr T getSignedDistance(const TV &X) const noexcept { return _normal.dot(X - _origin); }
    constexpr TV getNormal(const TV &X) const noexcept { return _normal; }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return std::make_tuple(_origin, _origin);
    }

    TV _origin{}, _normal{};
  };

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Cuboid, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Cuboid, DataType, d>, DataType, d> {
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV min, TV max) : _min{min}, _max{max} {}
    constexpr AnalyticLevelSet(TV center, T len)
        : _min{center - (len / 2)}, _max{center + (len / 2)} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {
      TV center = (_min + _max) / 2;
      TV point = (X - center).abs() - (_max - _min) / 2;
      T max = point.max();
      for (int i = 0; i < dim; ++i)
        if (point(i) < 0) point(i) = 0;  ///< inside the box
      return (max < 0 ? max : 0) + point.length();
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i < dim; i++) {
        v1 = X;
        v2 = X;
        v1(i) = X(i) + eps;
        v2(i) = X(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return std::make_tuple(_min, _max);
    }

    TV _min{}, _max{};
  };

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Sphere, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Sphere, DataType, d>, DataType, d> {
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV center, T radius) : _center{center}, _radius{radius} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {
      return (X - _center).length() - _radius;
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV outward_normal = X - _center;
      if (outward_normal.l2NormSqr() < (T)1e-7) return TV::zeros();
      return outward_normal.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return std::make_tuple(_center - _radius, _center + _radius);
    }

    TV _center{};
    T _radius{};
  };

  template <typename T, int dim> using GenericAnalyticLevelSet
      = variant<AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>>;

}  // namespace zs
