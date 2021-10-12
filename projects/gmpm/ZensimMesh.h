#pragma once
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/Matrix.hpp"
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ZenoFEMMesh : zeno::IObject {

  using value_type = float;
  using spmat_type = zs::YaleSparseMatrix<value_type, int>;
  using allocator_type = zs::ZSPmrAllocator<>;

  using vec3 = zs::vec<value_type, 3>;
  using mat3 = zs::vec<value_type, 3, 3>;
  using mat4 = zs::vec<value_type, 4, 4>;
  using mat_9_12 = zs::vec<value_type, 9, 12>;

  std::shared_ptr<zeno::PrimitiveObject> _mesh;
  zs::Vector<int> _bouDoFs;
  zs::Vector<int> _freeDoFs;
  zs::Vector<int> _DoF2FreeDoF;
  zs::Vector<int> _SpMatFreeDoFs;
  zs::Vector<zs::vec<int, 12, 12>> _elmSpIndices;

  zs::Vector<value_type> _elmMass;
  zs::Vector<value_type> _elmVolume;
  zs::Vector<zs::vec<value_type, 9, 12>> _elmdFdx;
  zs::Vector<mat4> _elmMinv;

  zs::Vector<value_type> _elmYoungModulus;
  zs::Vector<value_type> _elmPossonRatio;
  zs::Vector<value_type> _elmDensity;

  spmat_type _connMatrix;
  spmat_type _freeConnMatrix;

  zs::Vector<mat3> _elmAct;
  zs::Vector<mat3> _elmOrient;
  zs::Vector<vec3> _elmWeight;

  void DoPreComputation() {
    auto nm_elms = _mesh->quads.size(); // num tets
    for (std::size_t elm_id = 0; elm_id != nm_elms; ++elm_id) {
      auto elm = _mesh->quads[elm_id];
      mat4 M{};
      for (std::size_t i = 0; i != 4; ++i) {
        const auto &vert = _mesh->verts[elm[i]];
        for (std::size_t d = 0; d != 3; ++d)
          M(d, i) = vert[d];
        M(3, i) = 1;
        // M.block(0, i, 3, 1) << vert[0], vert[1], vert[2];
      }
      // M.bottomRows(1).setConstant(1.0);
      _elmVolume[elm_id] = gcem::abs(zs::determinant(M)) / 6;
      _elmMass[elm_id] = _elmVolume[elm_id] * _elmDensity[elm_id];

      mat3 Dm{};
      for (size_t i = 1; i < 4; ++i) {
        const auto &vert = _mesh->verts[elm[i]];
        const auto &vert0 = _mesh->verts[elm[0]];

        for (std::size_t d = 0; d != 3; ++d)
          Dm(d, i - 1) = vert[d] - vert0[d];
        // Dm.col(i - 1) << vert[0] - vert0[0], vert[1] - vert0[1],
        //    vert[2] - vert0[2];
      }

      mat3 DmInv = inverse(Dm);
      _elmMinv[elm_id] = inverse(M);

      value_type m = DmInv(0, 0);
      value_type n = DmInv(0, 1);
      value_type o = DmInv(0, 2);
      value_type p = DmInv(1, 0);
      value_type q = DmInv(1, 1);
      value_type r = DmInv(1, 2);
      value_type s = DmInv(2, 0);
      value_type t = DmInv(2, 1);
      value_type u = DmInv(2, 2);

      value_type t1 = -m - p - s;
      value_type t2 = -n - q - t;
      value_type t3 = -o - r - u;

      _elmdFdx[elm_id] =
          mat_9_12{t1, 0, 0, m, 0, 0, p, 0, 0,  s, 0, 0, 0, t1, 0, 0, m, 0,
                   0,  p, 0, 0, s, 0, 0, 0, t1, 0, 0, m, 0, 0,  p, 0, 0, s,
                   t2, 0, 0, n, 0, 0, q, 0, 0,  t, 0, 0, 0, t2, 0, 0, n, 0,
                   0,  q, 0, 0, t, 0, 0, 0, t2, 0, 0, n, 0, 0,  q, 0, 0, t,
                   t3, 0, 0, o, 0, 0, r, 0, 0,  u, 0, 0, 0, t3, 0, 0, o, 0,
                   0,  r, 0, 0, u, 0, 0, 0, t3, 0, 0, o, 0, 0,  r, 0, 0, u};
    }
  }
};

} // namespace zeno