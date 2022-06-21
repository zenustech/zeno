#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <iostream>
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

struct PrimSprayParticles : zeno::INode {
  virtual void apply() override {
    auto dx = get_input2<float>("Dx");
    auto prim = get_input("TrianglePrim")->as<PrimitiveObject>();
    auto result = zeno::IObject::make<PrimitiveObject>();
    tbb::concurrent_vector<zeno::vec3f> data(0);
    size_t n = prim->tris.size();
    log_debug("PrimSprayParticles got num tris: {}", n);
    tbb::parallel_for((size_t)0, (size_t)n, (size_t)1, [&](size_t index) {
      zeno::vec3f a, b, c;
      zeno::vec3i vi = prim->tris[index];
      a = prim->verts[vi[0]];
      b = prim->verts[vi[1]];
      c = prim->verts[vi[2]];
      zeno::vec3f e1 = b - a;
      zeno::vec3f e2 = c - a;
      zeno::vec3f e3 = c - b;
      auto le1 = length(e1);
      auto le2 = length(e2);
      auto le3 = length(e3);
      int in = std::max(int(le1 / dx), 1);
      int jn = std::max(int(le2 / dx), 1);
      int kn = std::max(int(le3 / dx), 1);
      zeno::vec3f dir1 = e1 / (in);
      zeno::vec3f dir2 = e2 / (jn);
      zeno::vec3f dir3 = e3 / (kn);
      data.push_back(a);
      data.push_back(b);
      data.push_back(c);
      /*for (int kk = 0; kk < kn; kk++) {
        zeno::vec3f vij = b + (0.5f + kk) * dir3;
        if (ptInTriangle(vij, a, b, c)) {
              data.push_back(vij);
        }
      }
      for (int ii = 0; ii < in; ii++) {
        zeno::vec3f vij = a + (0.5f + ii) * dir1;
        if (ptInTriangle(vij, a, b, c)) {
          data.push_back(vij);
        }
      }
      for (int jj = 0; jj < jn; jj++) {
        zeno::vec3f vij = a + (0.5f + jj) * dir2;
        if (ptInTriangle(vij, a, b, c)) {
          data.push_back(vij);
        }
      }*/
      for (int ii = 0; ii < in; ii++) {
        for (int jj = 0; jj < jn; jj++) {
          zeno::vec3f vij = a + (0.5f + ii) * dir1 + (0.5f + jj) * dir2;
          if (ptInTriangle(vij, a, b, c)) {
            data.push_back(vij);
          }
        }
      }
    });
    result->resize(data.size());
    for (int index = 0; index < data.size(); index++) {
      result->verts[index] = data[index];
    }
    set_output("particlesPrim", result);
  }
};

static int defPrimSprayParticles = zeno::defNodeClass<PrimSprayParticles>(
    "PrimSprayParticles", {/* inputs: */ {
                           "TrianglePrim", {"float", "Dx", "0.04"},
                       },
                       /* outputs: */
                       {
                           "particlesPrim",
                       },
                       /* params: */
                       {
                       },
                       /* category: */
                       {
                           "primitive",
                       }});

} // namespace zeno
