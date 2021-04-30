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


struct Stars {
  std::vector<double> mass;
  std::vector<__m256d> pos;
  std::vector<__m256d> vel;
  std::vector<__m256d> acc;
};


struct Octree {
  std::vector<__m256d> center;
  std::vector<std::array<int, 8>> children;
  std::vector<double> size;
  std::vector<int> pid;
};


int tree_alloc_node(Octree &tree, __m256d center, double size, int pid) {
  int n = tree.center.size();
  tree.center.push_back(center);
  tree.size.push_back(size);
  tree.pid.push_back(pid);
  tree.children.push_back({0, 0, 0, 0, 0, 0, 0, 0});
  return n;
}


void tree_add_particle(Octree &tree, Stars &star, int i, int n = 0) {
  int chid = _mm256_movemask_pd(tree.center[n] > star.pos[i]) & 7;
  int chn = tree.children[n][chid];
  if (chn != 0) {
    tree_add_particle(tree, star, i, chn);
    return;
  }
  __m256d ch_dir = _mm256_setr_pd(
      chid & 1 ? 1 : -1, chid & 2 ? 1 : -1, chid & 4 ? 1 : -1, 0.0);
  double ch_size = tree.size[n] * 0.5;
  __m256d ch_center = tree.center[n] + ch_dir * ch_size;
  tree.children[n][chid] = tree_alloc_node(tree, ch_center, ch_size, i);
}


void reset_root_node(Octree &tree, Stars &star) {
  __m256d bmin = star.pos[0];
  __m256d bmax = star.pos[0];
  for (int i = 1; i < star.pos.size(); i++) {
    bmin = _mm256_min_pd(bmin, star.pos[i]);
    bmax = _mm256_max_pd(bmax, star.pos[i]);
  }
  double vsize[4];
  _mm256_storeu_pd(vsize, bmax - bmin);
  double size = std::min(std::min(vsize[0], vsize[1]), vsize[2]);
  __m256d center = (bmin + bmax) * 0.5;

  tree.size.resize(1);
  tree.center.resize(1);
  tree.children.resize(1);
  tree.pid.resize(1);

  tree.size[0] = size;
  tree.center[0] = center;
  tree.children[0] = {0, 0, 0, 0, 0, 0, 0, 0};
  tree.pid[0] = -1;
}


void build_tree(Stars &star, Octree &tree) {
  reset_root_node(tree, star);

  for (int i = 0; i < star.pos.size(); i++) {
    tree_add_particle(tree, star, i);
  }
}


void advect_particles(Stars &star) {
  star.acc.resize(star.vel.size());

#pragma omp parallel for
  for (int i = 0; i < star.pos.size(); i++) {
    star.acc[i] = _mm256_set1_pd(0.0);

    for (int j = 0; j < star.pos.size(); j += 4) {
      __m256d r0 = star.pos[i] - star.pos[j + 0];
      __m256d r1 = star.pos[i] - star.pos[j + 1];
      __m256d r2 = star.pos[i] - star.pos[j + 2];
      __m256d r3 = star.pos[i] - star.pos[j + 3];
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
      __m256d fac = G * star.mass[i] / z;
      //__m256d z = _mm256_rsqrt_pd(y);
      //__m256d fac = G * star.mass[i] * z * z * z;

      star.acc[i] += fac[0] * r0 + fac[1] * r1 + fac[2] * r2 + fac[3] * r3;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < star.pos.size(); i++) {
    star.pos[i] += star.vel[i] * DT + star.acc[i] * (DT * DT / 2);
    star.vel[i] += star.acc[i] * DT;
  }
}


void load_particles(Stars &star, const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror(path);
  }
  assert(fp);
  int count;
  fscanf(fp, "#count %d\n", &count);
  star.pos.resize(count);
  star.vel.resize(count);
  star.mass.resize(count);
  for (int i = 0; i < star.pos.size(); i++) {
    fscanf(fp, "#v_mass %lf\n", &star.mass[i]);
    fscanf(fp, "v %lf %lf %lf\n", &star.pos[i][0], &star.pos[i][1], &star.pos[i][2]);
    fscanf(fp, "#v_vel %lf %lf %lf\n", &star.vel[i][0], &star.vel[i][1], &star.vel[i][2]);
  }
  fclose(fp);
}


void dump_particles(Stars &star, const char *path) {
  FILE *fp = fopen(path, "w");
  if (!fp) {
    perror(path);
  }
  assert(fp);
  fprintf(fp, "#count %d\n", star.pos.size());
  for (int i = 0; i < star.pos.size(); i++) {
    fprintf(fp, "#v_mass %lf\n", star.mass[i]);
    fprintf(fp, "v %lf %lf %lf\n", star.pos[i][0], star.pos[i][1], star.pos[i][2]);
    fprintf(fp, "#v_vel %lf %lf %lf\n", star.vel[i][0], star.vel[i][1], star.vel[i][2]);
  }
  fclose(fp);
}


int main(void)
{
  Stars star;
  Octree tree;

  load_particles(star, "solarsystem.obj");
  for (int i = 0; i < star.pos.size(); i++) {
    star.mass[i] *= 4 * M_PI * M_PI;
    star.vel[i] *= 365.24;
  }
  /*load_particles(star, "dubinski.obj");
  for (int i = 0; i < star.pos.size(); i++) {
    star.mass[i] *= 16000.0 * 0.75;
    star.vel[i] *= 8.0;
    star.pos[i] *= 1.5;
  }*/

  build_tree(star, tree);
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
    dump_particles(star, path);
  }
  */
}
