#include "func.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

float compute_sum(const std::vector<float> &vec) {
  float sum = 0.f;

#ifdef _OPENMP
  int N = vec.size();
  const float *arr = vec.data();
#pragma omp parallel for shared(arr) reduction(+ : sum)
  for (int i = 0; i < N; ++i)
    sum += arr[i];

#else
  for (auto &&v : vec)
    sum += v;
#endif

  return sum;
}