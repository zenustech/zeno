#include <Hg/SIMD/float4.hpp>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <omp.h>

using namespace hg::SIMD;

template <typename T>
void swap(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}

void blur1(const float *img, float *img_out, int nx)
{
#pragma omp parallel for simd
  for (int i = 0; i < nx; i++) {
    float xl = (img[i - 1]);
    float xr = (img[i + 1]);
    float res = xl + xr;
    img_out[i] = res;
  }
}

void blur2(const float *img, float *img_out, int nx)
{
  int mx = 256;
  int mt = 8 * omp_get_num_procs();
#pragma omp parallel for
  for (int x = 0; x < nx; x += mx) {
    for (int t = 0; t < mt; t++) {
      #pragma omp simd
      for (int i = x; i < x + mx; i++) {
        float xl = (img[i - 1]);
        float xr = (img[i + 1]);
        float res = xl + xr;
        img_out[i] = res;
      }
    }
  }
}

#define blur blur2

int main(void)
{
  int nx = 4096 * 4096 * 16;
  float *img = new float[nx];
  float *img_out = new float[nx];
  assert(img && img_out);

  for (int i = 0; i < nx; i++) {
    img[i] = drand48();
  }

  for (int i = 0; i < 8; i++) {
    printf("%d\n", i);
    blur(img, img_out, nx);
    swap(img, img_out);
  }

#if 0
  for (int i = 0; i < 64; i++) {
    printf("%f\n", img[i]);
  }
#endif

  return 0;
}
