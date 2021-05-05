// vim: sw=2 sts=2 ts=2
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


glm::dvec3 gravity_func(glm::dvec3 dist) {
  double r = 1 / (glm::length(dist) + 0.001);
  return r * r * r * dist;
}


struct Stars {
  std::vector<double> mass;
  std::vector<glm::dvec3> pos;
  std::vector<glm::dvec3> vel;
  std::vector<glm::dvec3> acc;

  void get_bounds(glm::dvec3 &bmin, glm::dvec3 &bmax) {
    bmin = bmax = pos[0];
    for (int i = 1; i < pos.size(); i++) {
      bmin = glm::min(bmin, pos[i]);
      bmax = glm::max(bmax, pos[i]);
    }
  }
};


struct Neighbor {
  std::vector<std::vector<int>> table;
  glm::ivec3 res;

  glm::dvec3 bmin, bmax, bscale;

  void initialize(glm::dvec3 const &bmin_, glm::dvec3 const &bmax_) {
    bmin = bmin_ + 0.1;
    bmax = bmax_ + 0.1;
    bscale = glm::dvec3(1) / (bmax - bmin);

    table.clear();
    double dx = 10.0;
    res = (glm::ivec3)glm::ceil((bmax - bmin) / dx);
    printf("resize %d %d %d\n", res.x, res.y, res.z);
    table.resize(res.x * res.y * res.z);
  }

  int linearize(glm::ivec3 const &idx) {
    return idx.x + res.x * (idx.y + res.y * idx.z);
  }

  std::vector<int> &at(glm::ivec3 const &idx) {
    return table[linearize(idx)];
  }

  auto world_to_index(glm::dvec3 const &pos) {
    auto rel = (pos - bmin) * bscale;
    auto idx = (glm::ivec3)glm::floor(rel * (glm::dvec3)res);
    idx = glm::max(glm::ivec3(0), glm::min(res, idx));
    return idx;
  }

  std::vector<int> find_neighbors(glm::dvec3 const &pos) {
    auto idx = world_to_index(pos);
    return at(idx);
  }

  void add_particle(glm::dvec3 const &pos, int i) {
    auto idx = world_to_index(pos);
    at(idx).push_back(i);
  }
};


void build_neighbor(Stars &star, Neighbor &neighbor) {
  glm::dvec3 bmin, bmax;
  star.get_bounds(bmin, bmax);
  neighbor.initialize(bmin, bmax);

  for (int i = 0; i < star.pos.size(); i++) {
    neighbor.add_particle(star.pos[i], i);
  }
}


void compute_gravity(Stars &star, Neighbor &neighbor) {
  printf("computing gravity for %d stars...\n", star.pos.size());
  star.acc.clear();
  star.acc.resize(star.pos.size());
  #pragma omp parallel for simd
  for (int i = 0; i < star.pos.size(); i++) {
    for (int j: neighbor.find_neighbors(star.pos[i])) {
      star.acc[i] += star.mass[j] * gravity_func(star.pos[j] - star.pos[i]);
    }
  }
  printf("computing gravity done\n");
}


void compute_gravity(Stars &star) {
  printf("direct computing gravity for %d stars...\n", star.pos.size());
  star.acc.clear();
  star.acc.resize(star.pos.size());
  #pragma omp parallel for simd
  for (int i = 0; i < star.pos.size(); i++) {
    for (int j = 0; j < star.pos.size(); j++) {
      star.acc[i] += star.mass[j] * gravity_func(star.pos[j] - star.pos[i]);
    }
  }
  printf("direct computing gravity done\n");
}


void advect_particles(Stars &star, double dt) {
  double half_dt2 = 0.5 * dt * dt;
  for (int i = 0; i < star.pos.size(); i++) {
    star.pos[i] += star.vel[i] * dt + star.acc[i] * half_dt2;
    star.vel[i] += star.acc[i] * dt;
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
  Neighbor neighbor;

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
#if 0
      build_neighbor(star, neighbor);
      compute_gravity(star, neighbor);
#else
      compute_gravity(star);
#endif
      advect_particles(star, 0.005);
    }

    char path[1024];
    sprintf(path, "/tmp/zenio/%06d/", i);
    mkdir(path, 0755);
    strcat(path, "result.obj");
    printf("dumping to %s\n", path);
    dump_particles(star, path);
  }
  return 0;
}
