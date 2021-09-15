#include "iostream"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/scalable_allocator.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace {
using namespace zeno;
// todo where to put this func???
float area(zeno::vec3f &p1, zeno::vec3f &p2, zeno::vec3f &p3) {
  zeno::vec3f e1 = p3 - p1;
  zeno::vec3f e2 = p2 - p1;
  zeno::vec3f areavec = zeno::cross(e1, e2);
  return 0.5 * sqrt(zeno::dot(areavec, areavec));
}
// todo where to put this func????
bool ptInTriangle(zeno::vec3f p, zeno::vec3f p0, zeno::vec3f p1,
                  zeno::vec3f p2) {
  // float A = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) +
  //                  p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]);
  // float sign = A < 0 ? -1.0f : 1.0f;
  // float s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] +
  //            (p0[0] - p2[0]) * p[1]) *
  //           sign;
  // float t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] +
  //            (p1[0] - p0[0]) * p[1]) *
  //           sign;

  // return s > 0 && t > 0 && ((s + t) < 2 * A * sign);
  p0-=p;
  p1-=p;
  p2-=p;
  auto u = zeno::cross(p1,p2);
  auto v = zeno::cross(p2,p0);
  auto w = zeno::cross(p0,p1);
  if(zeno::dot(u,v)<0)
  {return false;}
  if(zeno::dot(u,w)<0)
  {return false;}
  return true;
}

// to do where to put this func??
template <class T>
T baryCentricInterpolation(T &v1, T &v2, T &v3, zeno::vec3f &p,
                           zeno::vec3f &vert1, zeno::vec3f &vert2,
                           zeno::vec3f &vert3) {
  float a1 = area(p, vert2, vert3);
  float a2 = area(p, vert1, vert3);
  float a = area(vert1, vert2, vert3);
  float w1 = a1 / a;
  float w2 = a2 / a;
  float w3 = 1 - w1 - w2;
  return w1 * v1 + w2 * v2 + w3 * v3;
}

struct SprayParticles : zeno::INode {
  virtual void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto channel = std::get<std::string>(get_param("channel"));
    auto prim = get_input("TrianglePrim")->as<PrimitiveObject>();
    auto result = zeno::IObject::make<ParticlesObject>();
    tbb::concurrent_vector<std::tuple<zeno::vec3f,zeno::vec3f>> data(0);
    //tbb::concurrent_vector<zeno::vec3f> vel(0);
    size_t n = prim->tris.size();
    tbb::parallel_for((size_t)0, (size_t)n, (size_t)1, [&](size_t index)
    {
      zeno::vec3f a, b, c;
      zeno::vec3i vi = prim->tris[index];
      a = prim->attr<zeno::vec3f>("pos")[vi[0]];
      b = prim->attr<zeno::vec3f>("pos")[vi[1]];
      c = prim->attr<zeno::vec3f>("pos")[vi[2]];
      zeno::vec3f e1 = b - a;
      zeno::vec3f e2 = c - a;
      zeno::vec3f e3 = c - b;
      zeno::vec3f dir1 = e1 / zeno::length(e1);
      zeno::vec3f dir2 = e2 / zeno::length(e2);
      zeno::vec3f dir3 = e3 / zeno::length(e3);
      int in = zeno::length(e1) / (0.5 * dx) + 1;
      int jn = zeno::length(e2) / (0.5 * dx) + 1;
      int kn = zeno::length(e3) / (0.5 * dx) + 1;
      zeno::vec3f vel1 = prim->attr<zeno::vec3f>(channel)[vi[0]];
      zeno::vec3f vel2 = prim->attr<zeno::vec3f>(channel)[vi[1]];
      zeno::vec3f vel3 = prim->attr<zeno::vec3f>(channel)[vi[2]];
      // pos.emplace_back(a);
      // vel.emplace_back(vel1);
      // pos.emplace_back(b);
      // vel.emplace_back(vel2);
      // pos.emplace_back(c);
      // vel.emplace_back(vel3);
      data.emplace_back(std::make_tuple(a, vel1));
      data.emplace_back(std::make_tuple(b, vel2));
      data.emplace_back(std::make_tuple(c, vel3));
      for (int kk = 0; kk < kn; kk++) {
        zeno::vec3f vij = b + (float)kk * 0.5f * dx * dir3;
        if (ptInTriangle(vij, a, b, c)) {
              data.emplace_back(std::make_tuple(vij,baryCentricInterpolation(vel1, vel2, vel3, vij, a, b, c)));
        }
      }
      for (int ii = 0; ii < in; ii++) {
        zeno::vec3f vij = a + (float)ii * 0.5f * dx * dir1;
        if (ptInTriangle(vij, a, b, c)) {
          data.emplace_back(std::make_tuple(vij,baryCentricInterpolation(vel1, vel2, vel3, vij, a, b, c)));
        }
      }
      for (int jj = 0; jj < jn; jj++) {
        zeno::vec3f vij = a + (float)jj * 0.5f * dx * dir2;
        if (ptInTriangle(vij, a, b, c)) {
          data.emplace_back(std::make_tuple(vij,baryCentricInterpolation(vel1, vel2, vel3, vij, a, b, c)));
        }
      }
      for (int ii = 0; ii < in; ii++) {
        for (int jj = 0; jj < jn; jj++) {
          zeno::vec3f vij =
              a + (float)ii * 0.5f * dx * dir1 + (float)jj * 0.5f * dx * dir2;
          if (ptInTriangle(vij, a, b, c)) {
            data.emplace_back(std::make_tuple(vij,baryCentricInterpolation(vel1, vel2, vel3, vij, a, b, c)));
          }
        }
      }
    });
    result->pos.resize(data.size());
    result->vel.resize(data.size());
#pragma omp parallel for
    for (int index = 0; index < data.size(); index++) {
      result->pos[index] = zeno::vec_to_other<glm::vec3>(std::get<0>(data[index]));
      result->vel[index] = zeno::vec_to_other<glm::vec3>(std::get<1>(data[index]));
    }
    set_output("particles", result);
  }
};

static int defSprayParticles = zeno::defNodeClass<SprayParticles>(
    "SprayParticles", {/* inputs: */ {
                           "TrianglePrim", "Dx",
                       },
                       /* outputs: */
                       {
                           "particles",
                       },
                       /* params: */
                       {
                           {"float", "dx", "0.01"},
                           {"string", "channel", "vel"},
                       },
                       /* category: */
                       {
                           "FLIPSolver",
                       }});

} // namespace zeno
