// vim: sw=2 sts=2 ts=2
#include "SparseGrid.h"
#include "MathVec.h"
#include <cstdio>
#include <cmath>


int main(void) {
  fdb::PointsGrid grid;
  grid.addPoint(fdb::Vec3I(3, 4, 5));
  for (auto const &pos: grid.iterPoint()) {
    printf("%d %d %d\n", pos.x, pos.y, pos.z);
  }
}
