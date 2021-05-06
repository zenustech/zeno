// vim: sw=2 sts=2 ts=2
#include <cstdio>
#include "SparseGrid.h"
#include "Transform.h"
#include "MathVec.h"


int main(void) {
  fdb::PointsGrid grid;
  fdb::Transform<fdb::PointsGrid::MAX_INDEX> trans(0.1);

  fdb::Vec3f pos(0.1, 0.4, 0.5);
  grid.addPoint(trans.localToIndex(pos));

  for (auto const &ipos: grid.iterPoint()) {
    fdb::Vec3f pos = trans.indexToLocal(ipos);
    printf("%d %d %d\n", ipos.x, ipos.y, ipos.z);
    printf("%f %f %f\n", pos.x, pos.y, pos.z);
  }

  return 0;
}
