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


const double DT = 0.005;
const double G = 1.00;
const double EPS = 0.001;
const double LAM = 1.0;


struct Stars {
  std::vector<double> mass;
  std::vector<glm::dvec3> pos;
  std::vector<glm::dvec3> vel;
  std::vector<glm::dvec3> acc;
};


struct Field {
  std::vector<double> data;
};


glm::dvec3 gravity_func(glm::dvec3 dist) {
  double z = 1 / (glm::length(dist) + EPS);
  return (G * z * z * z) * dist;
}


glm::dvec3 field_gravity_at(Field &field, glm::dvec3 dist) {
  return glm::dvec3(0);
}


void compute_gravity(Stars &star, Field &field) {
  printf("computing gravity for %d stars...\n", star.pos.size());
  star.acc.resize(star.pos.size());
  for (int i = 0; i < star.pos.size(); i++) {
    star.acc[i] = field_gravity_at(field, star.pos[i]);
  }
  printf("computing gravity done\n");
}


void compute_gravity(Stars &star) {
  printf("direct computing gravity for %d stars...\n", star.pos.size());
  star.acc.clear();
  star.acc.resize(star.pos.size());
  for (int i = 0; i < star.pos.size(); i++) {
    for (int j = 0; j < star.pos.size(); j++) {
      star.acc[i] += star.mass[j] * gravity_func(star.pos[j] - star.pos[i]);
    }
  }
  printf("direct computing gravity done\n");
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
  Field field;

#if 1
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
      //build_field(star, field);
      //compute_gravity(star, tree);
      compute_gravity(star);
      advect_particles(star);
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
