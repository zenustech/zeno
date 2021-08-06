// vim: sw=2 sts=2 ts=2
#include <cstdio>
#include "SparseGrid.h"
#include "MathVec.h"


int main(void) {
  fdb::PointsGrid grid;
  float dx = 0.1;

  fdb::Vec3f pos(0.1, 0.4, 0.5);
  grid.addPoint(fdb::Vec3i(pos / dx));

  for (auto const &ipos: grid.iterPoint()) {
    fdb::Vec3f pos = (fdb::Vec3f)ipos * dx;
    printf("%f %f %f\n", pos.x, pos.y, pos.z);
  }

  return 0;
}
