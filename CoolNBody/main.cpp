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
const double LAM = 1.0;


struct Stars {
  std::vector<double> mass;
  std::vector<__m256d> pos;
  std::vector<__m256d> vel;
  std::vector<__m256d> acc;
};


struct Octree {
  std::vector<__m256d> center;
  std::vector<std::array<int, 8>> children;
  std::vector<__m256d> com;
  std::vector<double> mass;
  std::vector<double> size;
  std::vector<int> pid;
};


__m256d gravity_func(__m256d dist) {
  __m256d rr = dist * dist;
  double z = -1 / (rr[0] + rr[1] + rr[2]);
  return (z * z * z) * dist;
}


int tree_alloc_node(Octree &tree, __m256d center, double size, int pid) {
  int n = tree.center.size();
  tree.center.push_back(center);
  tree.com.push_back(_mm256_setzero_pd());
  tree.mass.push_back(0.0);
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
  tree.com[n] += star.mass[i] * star.pos[i];
  tree.mass[n] += star.mass[i];
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
  tree.com.resize(1);
  tree.mass.resize(1);
  tree.pid.resize(1);

  tree.size[0] = size;
  tree.center[0] = center;
  tree.com[0] = _mm256_setzero_pd();
  tree.mass[0] = 0.0;
  tree.children[0] = {0, 0, 0, 0, 0, 0, 0, 0};
  tree.pid[0] = -1;
}


void build_tree(Stars &star, Octree &tree) {
  printf("building octree...\n");
  reset_root_node(tree, star);

  for (int i = 0; i < star.pos.size(); i++) {
    tree_add_particle(tree, star, i);
  }
  printf("built octree with %d nodes\n", tree.center.size());

  for (int n = 0; n < tree.center.size(); n++) {
    if (tree.mass[n] != 0)
      tree.com[n] /= tree.mass[n];
    else
      tree.com[n] = tree.center[n];
  }
}


__m256d gravity_at(Octree &tree, __m256d pos, int n = 0) {
  __m256d dist = tree.com[n] - pos;
  __m256d rr = dist * dist;
  if (rr[0] + rr[1] + rr[2] > std::pow(LAM * tree.size[n], 2)) {
    return tree.mass[n] * gravity_func(dist);
  }

  __m256d acc = _mm256_setzero_pd();
  for (int chid = 0; chid < 8; chid++) {
    int chn = tree.children[n][chid];
    if (chn != 0)
      acc += gravity_at(tree, pos, chn);
  }
  return acc;
}


void compute_gravity(Stars &star, Octree &tree) {
  printf("computing gravity for %d stars...\n", star.pos.size());
  star.acc.resize(star.pos.size());
  for (int i = 0; i < star.pos.size(); i++) {
    star.acc[i] = gravity_at(tree, star.pos[i]);
    printf("%lf %lf %lf\n", star.acc[i][0], star.acc[i][1], star.acc[i][2]);
  }
  printf("computing gravity done\n");
}


void advect_particles(Stars &star) {
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
  printf("loading %d particles from %s\n", count, path);
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
  printf("dumping %d particles to %s\n", star.pos.size(), path);
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

#if 0
  load_particles(star, "solarsystem.obj");
  for (int i = 0; i < star.pos.size(); i++) {
    star.mass[i] *= 4 * M_PI * M_PI;
    star.vel[i] *= 365.24;
  }
#else
  load_particles(star, "dubinski.obj");
  for (int i = 0; i < star.pos.size(); i++) {
    star.mass[i] *= 16000.0 * 0.75;
    star.vel[i] *= 8.0;
    star.pos[i] *= 1.5;
  }
#endif

  for (int i = 0; i < 250; i++) {
    for (int j = 0; j < 28; j++) {
      build_tree(star, tree);
      compute_gravity(star, tree);
      advect_particles(star);
    }

    char path[1024];
    sprintf(path, "/tmp/zenio/%06d/", i);
    mkdir(path, 0755);
    strcat(path, "result.obj");
    printf("dumping to %s\n", path);
    dump_particles(star, path);
  }
}
