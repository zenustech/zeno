// vim: sw=2 sts=2 ts=2
#include "SparseGrid.h"
#include "MathVec.h"
#include <cstdio>
#include <cmath>


int main(void) {
  Grid<float> grid;
  grid(2, 3, 4) = 3.14;
  printf("%d\n", grid.isActiveAt(2, 3, 4));
  printf("%f\n", grid(2, 3, 4));
  printf("%f\n", grid(3, 3, 4));

  Vec3f a(1, 2, 3);
  Vec3f b(2, 3, 4);
  auto c = a + b;
  c = sqrt(c);
  printf("%f %f %f\n", c.x, c.y, c.z);
}
