#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>
#include <sys/stat.h>
#include <x86intrin.h>


const double DT = 0.004;
const double G = -1.00;
const double EPS = 0.01;


std::vector<double> mass;
std::vector<__m256d> pos;
std::vector<__m256d> vel;
std::vector<__m256d> acc;


std::vector<__m256d> n_center;
std::vector<std::array<int, 8>> n_children;
std::vector<double> n_size;
std::vector<int> n_pid;


int alloc_node(__m256d center, double size, int pid) {
  int n = n_center.size();
  n_center.push_back(center);
  n_size.push_back(size);
  n_pid.push_back(pid);
  n_children.push_back({0, 0, 0, 0, 0, 0, 0, 0});
  return n;
}


void tree_add_particle(int i, int n = 0) {
  int chid = _mm256_movemask_pd(n_center[n] > pos[i]) & 7;
  int chn = n_children[n][chid];
  if (chn != 0) {
    tree_add_particle(i, chn);
    return;
  }
  __m256d ch_dir = _mm256_setr_pd(
      chid & 1 ? 1 : -1, chid & 2 ? 1 : -1, chid & 4 ? 1 : -1, 0.0);
  double ch_size = n_size[n] * 0.5;
  __m256d ch_center = n_center[n] + ch_dir * ch_size;
  n_children[n][chid] = alloc_node(ch_center, ch_size, i);
}


void build_tree() {
  __m256d bmin = pos[0];
  __m256d bmax = pos[0];
  for (int i = 1; i < pos.size(); i++) {
    bmin = _mm256_min_pd(bmin, pos[i]);
    bmax = _mm256_max_pd(bmax, pos[i]);
  }
  double vsize[4];
  _mm256_storeu_pd(vsize, bmax - bmin);
  double size = std::min(std::min(vsize[0], vsize[1]), vsize[2]);
  __m256d center = (bmin + bmax) * 0.5;

  n_size.resize(1);
  n_center.resize(1);
  n_children.resize(1);
  n_pid.resize(1);

  n_size[0] = size;
  n_center[0] = center;
  n_children[0] = {0, 0, 0, 0, 0, 0, 0, 0};
  n_pid[0] = -1;

  for (int i = 0; i < pos.size(); i++) {
    tree_add_particle(i);
  }
  printf("%d\n", n_center.size());
}


void step() {
  acc.resize(vel.size());

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    acc[i] = _mm256_set1_pd(0.0);

    for (int j = 0; j < pos.size(); j += 4) {
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
  int count;
  fscanf(fp, "#count %d\n", &count);
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
  fprintf(fp, "#count %d\n", pos.size());
  for (int i = 0; i < pos.size(); i++) {
    fprintf(fp, "#v_mass %lf\n", mass[i]);
    fprintf(fp, "v %lf %lf %lf\n", pos[i][0], pos[i][1], pos[i][2]);
    fprintf(fp, "#v_vel %lf %lf %lf\n", vel[i][0], vel[i][1], vel[i][2]);
  }
  fclose(fp);
}


int main(void)
{
  load("solarsystem.obj");
  for (int i = 0; i < pos.size(); i++) {
    mass[i] *= 4 * M_PI * M_PI;
    vel[i] *= 365.24;
  }
  /*load("dubinski.obj");
  for (int i = 0; i < pos.size(); i++) {
    mass[i] *= 16000.0 * 0.75;
    vel[i] *= 8.0;
    pos[i] *= 1.5;
  }*/

  build_tree();
  /*
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
  */
}
