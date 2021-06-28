#pragma once
#include "BoundingVolumeInterface.hpp"

namespace zs {

  template <typename Derived, typename T, int dim> struct LevelSetInterface
      : BoundingVolumeInterface<Derived, T, dim> {
    using base_t = BoundingVolumeInterface<Derived, T, dim>;
    using TV = typename base_t::TV;

    constexpr T getSignedDistance(const TV &X) const noexcept {
      return selfPtr()->getSignedDistance(X);
    }
    constexpr TV getNormal(const TV &X) const noexcept { return selfPtr()->getNormal(X); }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {
      return selfPtr()->getMaterialVelocity(X);
    }
    using base_t::getBoundingBox;
    using base_t::getBoxCenter;
    using base_t::selfPtr;
  };

}  // namespace zs
