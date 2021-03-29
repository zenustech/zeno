#include <Hg/SIMD/simd.hpp>
#if defined(HG_SIMD_FLOAT16)
#include <Hg/SIMD/float16.hpp>
#elif defined(HG_SIMD_FLOAT8)
#include <Hg/SIMD/float8.hpp>
#endif
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <omp.h>


using namespace hg::simd;


const long nx = 256, ny = 256, nz = 256, nq = 15;


const float niu = 0.005;
const float tau = 3.0 * niu + 0.5;
const float inv_tau = 1 / tau;
const long nl = nx * ny * nz;


glm::ivec3 directions[] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
float weights[] = {2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0};


float *f_old = new float[nx * ny * nz * nq];
float *f_new = new float[nx * ny * nz * nq];
float *rho = new float[nx * ny * nz];
glm::vec3 *vel = new glm::vec3[nx * ny * nz];


long linearXYZ(long x, long y, long z)
{
  return x + nx * (y + ny * z);
}

long linearLQ(long l, long q)
{
  return q + nq * l;
  //return l + nl * q;
}


std::tuple<long, long, long> unlinearXYZ(long i)
{
  long j = i / nx;
  return std::make_tuple(i % nx, j % ny, j / ny);
}


std::tuple<long, long> unlinearLQ(long i)
{
  return std::make_tuple(i / nq, i % nq);
  //return std::make_tuple(i % nl, i / nl);
}


float f_eq(long l, long q)
{
  float m = rho[l];
  glm::vec3 v = vel[l];
  float eu = glm::dot(v, (glm::vec3)directions[q]);
  float uv = glm::dot(v, v);
  float term = 1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv;
  float feq = weights[q] * m * term;
  return feq;
}


#if defined(HG_SIMD_FLOAT16)
float16 vectorized_f_eq(long l)
{
  float16 d_x{0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1, 0};
  float16 d_y{0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1, 0};
  float16 d_z{0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, 0};
  float16 wei{2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 0.0};
  glm::vec3 v = vel[l];
  float fac = (4.5 * inv_tau) * rho[l];
  float rhs = 1 / 4.5 - (1.5 / 4.5) * glm::dot(v, v);
  float16 eu = v.x * d_x + v.y * d_y + v.z * d_z;
  float16 term = (3 / 4.5 + eu) * eu + rhs;
  float16 feq = fac * (wei * term);
  return feq;
}
#elif defined(HG_SIMD_FLOAT8)
std::tuple<float8, float8> vectorized_f_eq(long l)
{
  float8 d1_x{0,  1, -1,  0,  0,  0,  0,  1};
  float8 d2_x{-1,  1, -1,  1, -1, -1,  1, 0};
  float8 d1_y{0,  0,  0,  1, -1,  0,  0,  1};
  float8 d2_y{-1,  1, -1, -1,  1,  1, -1, 0};
  float8 d1_z{0,  0,  0,  0,  0,  1, -1,  1};
  float8 d2_z{-1, -1,  1,  1, -1,  1, -1, 0};
  float8 wei1{2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/72.0};
  float8 wei2{1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 0.0};
  glm::vec3 v = vel[l];
  float fac = (4.5 * inv_tau) * rho[l];
  float rhs = 1 / 4.5 - (1.5 / 4.5) * glm::dot(v, v);
  float8 eu1 = v.x * d1_x + v.y * d1_y + v.z * d1_z;
  float8 eu2 = v.x * d2_x + v.y * d2_y + v.z * d2_z;
  float8 c_3o4_5{3 / 4.5};
  float8 term1 = (c_3o4_5 + eu1) * eu1 + rhs;
  float8 term2 = (c_3o4_5 + eu2) * eu2 + rhs;
  float8 feq1 = fac * (wei1 * term1);
  float8 feq2 = fac * (wei2 * term2);
  return std::make_tuple(feq1, feq2);
}
#endif


void substep()
{
  // feq / total = 1/12
  // simd profit = 300%
#pragma omp parallel for
  for (long l = 0; l < nx * ny * nz; l++) {
#if defined(HG_SIMD_FLOAT16)
    float arr_feq[16];
    float16::to(arr_feq[0]) = vectorized_f_eq(l);
#elif defined(HG_SIMD_FLOAT8)
    float arr_feq[16];
    auto [feq1, feq2] = vectorized_f_eq(l);
    float8::to(arr_feq[0]) = feq1;
    float8::to(arr_feq[8]) = feq2;
#endif
    for (long q = 0; q < nq; q++) {
      long lq = linearLQ(l, q);
      auto [x, y, z] = unlinearXYZ(l);
      auto md = glm::ivec3(x, y, z) - directions[q];
      md += glm::ivec3(nx, ny, nz);
      md %= glm::ivec3(nx, ny, nz);
      long lmd = linearXYZ(md.x, md.y, md.z);
      long lmdq = linearLQ(lmd, q);
#if defined(HG_SIMD_FLOAT16) || defined(HG_SIMD_FLOAT8)
      f_new[lq] = f_old[lmdq] * (1 - inv_tau) + arr_feq[q];
#else
      f_new[lq] = f_old[lmdq] * (1 - inv_tau) + f_eq(lmd, q) * inv_tau;
#endif
    }
  }
#pragma omp parallel for
  for (long l = 0; l < nx * ny * nz; l++) {
    float m = 0;
    glm::vec3 v(0);
    for (long q = 0; q < nq; q++) {
      long lq = linearLQ(l, q);
      float f = f_new[lq];
      v += f * (glm::vec3)directions[q];
      m += f;
    }
    rho[l] = m;
    vel[l] = v / std::max(m, (float)1e-6);
  }
  std::swap(f_new, f_old);
}


void initialize()
{
  for (long l = 0; l < nx * ny * nz; l++) {
    rho[l] = 1;
    vel[l] = glm::vec3(0);
  }

  for (long lq = 0; lq < nx * ny * nz * nq; lq++) {
    auto [l, q] = unlinearLQ(lq);
    f_old[lq] = f_eq(l, q);
  }
}

int main()
{
  initialize();

  for (int i = 0; i < 8; i++) {
    printf("%d\n", i);
    substep();
  }

  return 0;
}
