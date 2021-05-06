// vim: sw=2 sts=2 ts=2
#include "VolumeGrid.h"
#include "MathVec.h"
#include <cstdio>
#include <cmath>


int main(void) {
  fdb::volume::Grid<float> grid;
  grid(2, 3, 4) = 3.14;
  printf("%d\n", grid.isActiveAt(2, 3, 4));
  printf("%f\n", grid(2, 3, 4));
  printf("%f\n", grid(3, 3, 4));
}
