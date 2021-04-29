#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
#include <omp.h>


std::vector<float> mass;
std::vector<glm::dvec3> pos;
std::vector<glm::dvec3> vel;
std::vector<glm::dvec3> acc;


const double DT = 0.01;
const double G = 0.01;


void step() {
  acc.resize(vel.size());

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    acc[i] = glm::dvec3(0);
    for (int j = 0; j < pos.size(); j++) {
      if (j == i) continue;
      glm::dvec3 r = pos[i] - pos[j];
      double x = 1 / std::sqrt(glm::dot(r, r) + 1e-8);
      double fac = G * x * x * x;
      acc[i] += fac * r;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    pos[i] += vel[i] * DT * 0.5 * acc[i] * DT * DT;
    vel[i] += acc[i] * DT;
  }
}


void load(const char *path) {
  FILE *fp = fopen(path, "r");
  assert(fp);
  size_t count;
  fscanf(fp, "#count %zd\n", &count);
  pos.resize(count);
  for (int i = 0; i < pos.size(); i++) {
    fscanf(fp, "#v_mass %lf\n", &mass[i]);
    fscanf(fp, "v %lf %lf %lf\n", &pos[i].x, &pos[i].y, &pos[i].z);
    fscanf(fp, "#v_vel %lf %lf %lf\n", &vel[i].x, &vel[i].y, &vel[i].z);
  }
  fclose(fp);
}


void dump(const char *path) {
  FILE *fp = fopen(path, "w");
  assert(fp);
  fprintf(fp, "#count %zd\n", pos.size());
  for (int i = 0; i < pos.size(); i++) {
    fprintf(fp, "#v_mass %lf\n", mass[i]);
    fprintf(fp, "v %lf %lf %lf\n", pos[i].x, pos[i].y, pos[i].z);
    fprintf(fp, "#v_vel %lf %lf %lf\n", vel[i].x, vel[i].y, vel[i].z);
  }
  fclose(fp);
}


int main(void)
{
  load("solarsystem.txt");
  for (int i = 0; i < pos.size(); i++) {
    mass[i] *= 4 * M_PI * M_PI;
    vel[i] *= 365.24;
  }
}
