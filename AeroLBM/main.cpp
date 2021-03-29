#include <Hg/SIMD/float16.hpp>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <omp.h>


using namespace hg::simd;


const long nx = 128, ny = 128, nz = 128, nq = 15;


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


float16 vectorized_f_eq(long l)
{
  float16 d_x{0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1, 0};
  float16 d_y{0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1, 0};
  float16 d_z{0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, 0};
  float16 wei{2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 0.0};
  float m = rho[l];
  glm::vec3 v = vel[l];
  float uv = glm::dot(v, v);
  float16 eu = v.x * d_x + v.y * d_y + v.z * d_z;
  float16 term = 3 * eu + 4.5 * (eu * eu) + (1 - 1.5 * uv);
  float16 feq = m * (wei * term);
  return feq;
}


void substep2()
{
#pragma omp parallel for
  for (long l = 0; l < nx * ny * nz; l++) {
    float arr_feq[16];
    float16::to(*arr_feq) = vectorized_f_eq(l) * inv_tau;
    for (long q = 0; q < nq; q++) {
      long lq = linearLQ(l, q);
      auto [x, y, z] = unlinearXYZ(l);
      auto md = glm::ivec3(x, y, z) - directions[q];
      md += glm::ivec3(nx, ny, nz);
      md %= glm::ivec3(nx, ny, nz);
      long lmd = linearXYZ(md.x, md.y, md.z);
      long lmdq = linearLQ(lmd, q);
      f_new[lq] = f_old[lmdq] * (1 - inv_tau) + arr_feq[q];
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


void substep1()
{
#pragma omp parallel for
  for (long l = 0; l < nx * ny * nz; l++) {
    for (long q = 0; q < nq; q++) {
      long lq = linearLQ(l, q);
      auto [x, y, z] = unlinearXYZ(l);
      auto md = glm::ivec3(x, y, z) - directions[q];
      md += glm::ivec3(nx, ny, nz);
      md %= glm::ivec3(nx, ny, nz);
      long lmd = linearXYZ(md.x, md.y, md.z);
      long lmdq = linearLQ(lmd, q);
      f_new[lq] = f_old[lmdq] * (1 - inv_tau) + f_eq(lmd, q) * inv_tau;
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


int main()
{
  initialize();

  for (int i = 0; i < 32; i++) {
    printf("%d\n", i);
    substep2();
  }

  return 0;
}
