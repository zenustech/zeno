#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <omp.h>


const long nx = 64, ny = 64, nz = 64, nq = 15;


const float niu = 0.005;
const float tau = 3.0 * niu + 0.5;
const float inv_tau = 1 / tau;


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
  long nl = nx * ny * nz;
  return l + nl * q;
}


std::tuple<long, long, long> unlinearXYZ(long i)
{
  long j = i / nx;
  return std::make_tuple(i % nx, j % ny, j / ny);
}


std::tuple<long, long> unlinearLQ(long i)
{
  long nl = nx * ny * nz;
  return std::make_tuple(i % nl, i / nl);
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


void substep()
{
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


void apply_bc() {
/*
@ti.func
def apply_bc_core(outer, bc_type, bc_vel, ibc, jbc, kbc, inb, jnb, knb):
    if (outer == 1):  # handle outer boundary
        if bc_type == 0:
            vel[ibc, jbc, kbc] = bc_vel
        elif bc_type == 1:
            vel[ibc, jbc, kbc] = vel[inb, jnb, knb]
    rho[ibc, jbc, kbc] = rho[inb, jnb, knb]
    for l in range(direction_size):
        f_old[ibc, jbc, kbc, l] = f_eq(ibc, jbc, kbc, l) - f_eq(inb, jnb, knb, l) + f_old[inb, jnb, knb, l]


@ti.kernel
def apply_bc():
    for y, z in ti.ndrange((1, res[1] - 1), (1, res[2] - 1)):
        apply_bc_core(1, 0, [0.1, 0.0, 0.0],
                0, y, z, 1, y, z)
        apply_bc_core(1, 1, [0.0, 0.0, 0.0],
                res[0] - 1, y, z, res[0] - 2, y, z)
*/
}


int main()
{
  initialize();

  substep();
  apply_bc();

  return 0;
}
