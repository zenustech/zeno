#pragma once
#include "zensim/geometry/Collider.h"
#include "ZensimMesh.h"
#include <zeno/zeno.h>

namespace zeno {

struct ZenoExplicitTimeIntegrator : zeno::IObject {
  using value_type = typename ZenoFEMMesh::value_type;
  using size_type = typename ZenoFEMMesh::size_type;
  using vec3 = typename ZenoFEMMesh::vec3;

  value_type _dt;
  vec3 _gravity;
  zs::Vector<vec3> _f;
};

} // namespace zeno