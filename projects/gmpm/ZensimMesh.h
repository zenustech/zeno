#pragma once
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/Matrix.hpp"
#include "zensim/math/matrix/Matrix.hpp"
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ZenoFEMMesh : zeno::IObject {

  using value_type = float;
  using spmat_type = zs::YaleSparseMatrix<value_type, int>;
  using allocator_type = zs::ZSPmrAllocator<>;

  std::shared_ptr<zeno::PrimitiveObject> _mesh;
  zs::Vector<int> _bouDoFs;
  zs::Vector<int> _freeDoFs;
  zs::Vector<int> _DoF2FreeDoF;
  zs::Vector<int> _SpMatFreeDoFs;
  zs::Vector<zs::vec<int, 12, 12>> _elmSpIndices;

  zs::Vector<value_type> _elmMass;
  zs::Vector<value_type> _elmVolume;
  zs::Vector<zs::vec<value_type, 9, 12>> _elmdFdx;
  zs::Vector<zs::vec<value_type, 4, 4>> _elmMinv;

  zs::Vector<value_type> _elmYoungModulus;
  zs::Vector<value_type> _elmPossonRatio;
  zs::Vector<value_type> _elmDensity;

  spmat_type _connMatrix;
  spmat_type _freeConnMatrix;

  zs::Vector<zs::vec<value_type, 3, 3>> _elmAct;
  zs::Vector<zs::vec<value_type, 3, 3>> _elmOrient;
  zs::Vector<zs::vec<value_type, 3>> _elmWeight;
};

} // namespace zeno