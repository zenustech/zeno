#pragma once
#include <cmath>
#include <cstdlib>

namespace zs {

  double PDF(int lambda, int k);
  int rand_p(double lambda);
  double anti_normal_PDF(double u, double o, int x);

  double PDF(double u, double o, int x);
  int rand_normal(double u, double o);
  int rand_anti_normal(double u, double o);
}  // namespace zs
