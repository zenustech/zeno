#pragma once
#include "GenericLevelSet.h"
#include "zensim/math/Rotation.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum class collider_e { Sticky, Slip, Separate };

  template <typename LS> struct Collider {
    using T = typename LS::T;
    static constexpr int dim = LS::dim;
    using TV = vec<T, dim>;

    constexpr void setCollisionType(collider_e ct) noexcept { type = ct; }
    constexpr void setTranslation(TV b_in, TV dbdt_in) noexcept {
      b = b_in;
      dbdt = dbdt_in;
    }
    constexpr void setRotation(Rotation<T, dim> R_in, AngularVelocity<T, dim> omega_in) noexcept {
      R = R_in;
      omega = omega_in;
    }
    constexpr bool queryInside(const TV &x) const noexcept {
      TV x_minus_b = x - b;
      T one_over_s = 1 / s;
      TV X = R.transpose() * x_minus_b * one_over_s;  // material space
      return levelset.getSignedDistance(X) < 0;
    }
    constexpr bool resolveCollision(const TV &x, TV &v, T erosion = 0) const noexcept {
      /** derivation:
          x = \phi(X,t) = R(t)s(t)X+b(t)
          X = \phi^{-1}(x,t) = (1/s) R^{-1} (x-b)
          V(X,t) = \frac{\partial \phi}{\partial t}
                = R'sX + Rs'X + RsX' + b'
          v(x,t) = V(\phi^{-1}(x,t),t)
                = R'R^{-1}(x-b) + (s'/s)(x-b) + RsX' + b'
                = omega \cross (x-b) + (s'/s)(x-b) +b'
      */
      /// collision
      TV x_minus_b = x - b;
      T one_over_s = 1 / s;
      TV X = R.transpose() * x_minus_b * one_over_s;  // material space
      if (levelset.getSignedDistance(X) < -erosion) {
        TV v_object = omega.cross(x_minus_b) + (dsdt * one_over_s) * x_minus_b
                      + R * s * levelset.getMaterialVelocity(X) + dbdt;
        if (type == collider_e::Sticky)
          v = TV::zeros();
        else {
          v -= v_object;
          TV n = R * levelset.getNormal(X);
          T proj = n.dot(v);
          if ((type == collider_e::Separate && proj < 0) || type == collider_e::Slip) v -= proj * n;
          v += v_object;
        }
        return true;
      }
      return false;
    }

    Collider() = default;
    constexpr Collider(LS &&ls, collider_e t = collider_e::Sticky)
        : levelset{std::move(ls)}, type{t} {}
    constexpr Collider(const LS &ls, collider_e t = collider_e::Sticky) : levelset{ls}, type{t} {}

    // levelset
    LS levelset;
    collider_e type{collider_e::Sticky};  ///< runtime
    /** scale **/
    T s{1};
    T dsdt{0};
    /** rotation **/
    Rotation<T, dim> R{};
    AngularVelocity<T, dim> omega{};
    /** translation **/
    TV b{TV::zeros()};
    TV dbdt{TV::zeros()};
  };

  template <typename Ls, typename... Args> Collider(Ls, Args...) -> Collider<Ls>;

  template <typename T, int dim> using GenericCollider
      = variant<Collider<AnalyticLevelSet<analytic_geometry_e::Plane, T, dim>>,
                Collider<AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>>,
                Collider<AnalyticLevelSet<analytic_geometry_e::Sphere, T, dim>>,
                Collider<LevelSet<T, dim>>, Collider<HeightField<T>>>;

  template <typename LS> struct LevelSetBoundary {
    using T = typename LS::value_type;
    static constexpr int dim = LS::dim;
    using TV = vec<T, dim>;

    LevelSetBoundary() = default;
    constexpr LevelSetBoundary(LS &&ls, collider_e t = collider_e::Sticky)
        : levelset{std::move(ls)}, type{t} {}
    constexpr LevelSetBoundary(const LS &ls, collider_e t = collider_e::Sticky)
        : levelset{ls}, type{t} {}

    constexpr void setCollisionType(collider_e ct) noexcept { type = ct; }
    constexpr void setTranslation(TV b_in, TV dbdt_in) noexcept {
      b = b_in;
      dbdt = dbdt_in;
    }
    constexpr void setRotation(Rotation<T, dim> R_in, AngularVelocity<T, dim> omega_in) noexcept {
      R = R_in;
      omega = omega_in;
    }

    // levelset
    LS levelset;
    collider_e type{collider_e::Sticky};  ///< runtime
    /** scale **/
    T s{1};
    T dsdt{0};
    /** rotation **/
    Rotation<T, dim> R{};
    AngularVelocity<T, dim> omega{};
    /** translation **/
    TV b{TV::zeros()};
    TV dbdt{TV::zeros()};
  };

  template <typename Ls, typename... Args> LevelSetBoundary(Ls, Args...) -> LevelSetBoundary<Ls>;

  template <typename> struct is_levelset_boundary : std::false_type {};
  template <typename LS> struct is_levelset_boundary<LevelSetBoundary<LS>> : std::true_type {};

  using GeneralBoundary = variant<LevelSetBoundary<SparseLevelSet<3>>,
                                  Collider<AnalyticLevelSet<analytic_geometry_e::Plane, f32, 3>>,
                                  Collider<AnalyticLevelSet<analytic_geometry_e::Cuboid, f32, 3>>,
                                  Collider<AnalyticLevelSet<analytic_geometry_e::Sphere, f32, 3>>,
                                  Collider<LevelSet<f32, 3>>,
                                  Collider<AnalyticLevelSet<analytic_geometry_e::Cuboid, f32, 2>>,
                                  Collider<AnalyticLevelSet<analytic_geometry_e::Sphere, f32, 2>>,
                                  Collider<LevelSet<f32, 2>>, Collider<HeightField<f32>>>;

}  // namespace zs
