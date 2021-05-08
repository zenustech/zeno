// vim: sw=2 sts=2 ts=2
#pragma once


#include "MathVec.h"


namespace fdb {

template <int max_index>
struct Transform {
  float dx;

  static constexpr auto half_max_index = max_index >> 1;

  explicit Transform(float dx)
    : dx(dx)
  {}

  Vec3f indexToLocal(Vec3I const &ipos) {
    Vec3i signed_idx = (Vec3i)ipos - half_max_index;
    Vec3f lpos = (Vec3f)signed_idx * dx;
    return lpos;
  }

  Vec3I localToIndex(Vec3f const &lpos) {
    Vec3i signed_idx(lpos / dx);
    Vec3I ipos(signed_idx + half_max_index);
    return ipos;
  }
};

}
