#include <Hg/SIMD/float4.hpp>
#include <cassert>
#include <cstdio>
#include <cmath>

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
    float xc = (img[i]);
    float xr = (img[i + 1]);
    float res = xl + xc + xr;
    img_out[i] = res;
  }
}

void blur2(const float *img, float *img_out, int nx)
{
  int mx = 32;
#pragma omp parallel for simd
  for (int i = 0; i < nx; i += mx) {
    for (int t = 0; t < 2; t++) {
      for (int j = i; j < i + mx; j++) {
        float res = img[j - 2] + img[j - 1] + img[j] + img[j + 1] + img[j + 2];
        img_out[j] = res;
      }
    }
  }
}

int main(void)
{
  int nx = 4096 * 4096;
  float *img = new float[nx];
  float *img_out = new float[nx];
  assert(img && img_out);

  for (int i = 0; i < nx; i++) {
    img[i] = drand48();
  }

  for (int i = 0; i < 256; i++) {
    printf("%d\n", i);
    blur2(img, img_out, nx);
    swap(img, img_out);
  }

#if 0
  for (int i = 0; i < 64; i++) {
    printf("%f\n", img[i]);
  }
#endif

  return 0;
}
