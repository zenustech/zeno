#pragma once
#include <type_traits>
#include <algorithm>

#include "zensim/math/Vec.h"

namespace zs {

  // enum struct bv_e : char { aabb, obb, sphere, convex };
  template <typename Derived, typename T, int dim> struct BoundingVolumeInterface {
    static_assert(std::is_floating_point_v<T>, "T for bounding volume should be floating point!");

    using TV = vec<T, dim>;

    constexpr std::tuple<TV, TV> getBoundingBox() const noexcept {
      return selfPtr()->do_getBoundingBox();
    }
    constexpr TV getBoxCenter() const noexcept { return selfPtr()->do_getBoxCenter(); }
    constexpr TV getBoxSideLengths() const noexcept { return selfPtr()->do_getBoxSideLengths(); }
    constexpr TV getUniformCoord(const TV& pos) const noexcept { return selfPtr()->do_getUniformCoord(pos); }

  protected:
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return std::make_tuple(TV::zeros(), TV::zeros());
    }
    constexpr TV do_getBoxCenter() const noexcept {
      auto &&[lo, hi] = getBoundingBox();
      return (lo + hi) / 2;
    }
    constexpr TV do_getBoxSideLengths() const noexcept {
      auto &&[lo, hi] = getBoundingBox();
      return hi - lo;
    }
    constexpr TV do_getUniformCoord(const TV &pos) const noexcept {
      auto &&[lo, offset] = getBoundingBox();
      const auto lengths = offset - lo;
      offset = pos - lo;
      for (int d = 0; d < dim; ++d)
        offset[d] = std::clamp(offset[d], (T)0, lengths[d]) / lengths[d];
      return offset;
    }

    constexpr Derived *selfPtr() noexcept { return static_cast<Derived *>(this); }
    constexpr const Derived *selfPtr() const noexcept { return static_cast<const Derived *>(this); }
  };

}  // namespace zs