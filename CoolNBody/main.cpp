#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#include <omp.h>
#include <sys/stat.h>
#include <x86intrin.h>


std::vector<double> mass;
std::vector<__m256d> pos;
std::vector<__m256d> vel;
std::vector<__m256d> acc;


const double DT = 0.004;
const double G = -0.75;
const double EPS = 0.01;


static inline __m256d _mm256_rsqrt_pd(__m256d s) {
  __m128 q = _mm256_cvtpd_ps(s);
  q = _mm_rsqrt_ps(q);
  __m256d x = _mm256_cvtps_pd(q);
  __m256d y = s * x * x;
  __m256d a = _mm256_mul_pd(y, _mm256_set1_pd(0.375));
  a = _mm256_mul_pd(a, y);
  __m256d b = _mm256_mul_pd(y, _mm256_set1_pd(1.25));
  b = _mm256_sub_pd(b, _mm256_set1_pd(1.875));
  y = _mm256_sub_pd(a, b);
  x = _mm256_mul_pd(x, y);
  return x;
}


void step() {
  acc.resize(vel.size());

#pragma omp parallel for
  for (size_t i = 0; i < pos.size(); i++) {
    acc[i] = _mm256_set1_pd(0.0);

    for (size_t j = 0; j < pos.size(); j += 4) {
      __m256d r0 = pos[i] - pos[j + 0];
      __m256d r1 = pos[i] - pos[j + 1];
      __m256d r2 = pos[i] - pos[j + 2];
      __m256d r3 = pos[i] - pos[j + 3];
      __m256d x0 = r0 * r0;
      __m256d x1 = r1 * r1;
      __m256d x2 = r2 * r2;
      __m256d x3 = r3 * r3;
      __m256d t0 = _mm256_hadd_pd(x0, x1);
      __m256d t1 = _mm256_hadd_pd(x2, x3);
      __m256d y0 = _mm256_permute2f128_pd(t0, t1, 0x21);
      __m256d y1 = _mm256_blend_pd(t0, t1, 0b1100);

      __m256d y = y0 + y1 + EPS;
      __m256d z = y * _mm256_sqrt_pd(y);
      __m256d fac = G * mass[i] / z;
      //__m256d z = _mm256_rsqrt_pd(y);
      //__m256d fac = G * mass[i] * z * z * z;

      acc[i] += fac[0] * r0 + fac[1] * r1 + fac[2] * r2 + fac[3] * r3;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    pos[i] += vel[i] * DT + acc[i] * (DT * DT / 2);
    vel[i] += acc[i] * DT;
  }
}


void load(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror(path);
  }
  assert(fp);
  size_t count;
  fscanf(fp, "#count %zd\n", &count);
  pos.resize(count);
  vel.resize(count);
  mass.resize(count);
  for (int i = 0; i < pos.size(); i++) {
    fscanf(fp, "#v_mass %lf\n", &mass[i]);
    fscanf(fp, "v %lf %lf %lf\n", &pos[i][0], &pos[i][1], &pos[i][2]);
    fscanf(fp, "#v_vel %lf %lf %lf\n", &vel[i][0], &vel[i][1], &vel[i][2]);
  }
  fclose(fp);
}


void dump(const char *path) {
  FILE *fp = fopen(path, "w");
  if (!fp) {
    perror(path);
  }
  assert(fp);
  fprintf(fp, "#count %zd\n", pos.size());
  for (int i = 0; i < pos.size(); i++) {
    fprintf(fp, "#v_mass %lf\n", mass[i]);
    fprintf(fp, "v %lf %lf %lf\n", pos[i][0], pos[i][1], pos[i][2]);
    fprintf(fp, "#v_vel %lf %lf %lf\n", vel[i][0], vel[i][1], vel[i][2]);
  }
  fclose(fp);
}


int main(void)
{
  /*load("solarsystem.obj");
  for (int i = 0; i < pos.size(); i++) {
    mass[i] *= 4 * M_PI * M_PI;
    vel[i] *= 365.24;
  }*/
  load("dubinski.obj");
  for (int i = 0; i < pos.size(); i++) {
    mass[i] *= 16000.0;
    vel[i] *= 8.0;
    pos[i] *= 1.5;
  }

  for (int i = 0; i < 250; i++) {
    for (int j = 0; j < 28; j++) {
      step();
    }

    char path[1024];
    sprintf(path, "/tmp/zenio/%06d/", i);
    mkdir(path, 0755);
    strcat(path, "result.obj");
    printf("dumping to %s\n", path);
    dump(path);
  }
}
