// vim: sw=2 sts=2 ts=2
#pragma once


#include <cstddef>
#include "MathVec.h"


namespace fdb {

struct Transform {
  float dx;
  Vec3d center;
  unsigned int max_index;

  Vec3f indexToLocal(Vec3I const &idx) {
    auto half_max_index = max_index >> 1;
    Vec3i signed_idx = (Vec3i)idx - half_max_index;
    Vec3f local = signed_idx * (dx / half_max_index);
    return local;
  }
};

}
