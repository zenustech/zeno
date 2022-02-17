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
bool pointInTriangle(const zeno::vec3f& query_point,
                     const zeno::vec3f& triangle_vertex_0,
                     const zeno::vec3f& triangle_vertex_1,
                     const zeno::vec3f& triangle_vertex_2,
                     zeno::vec3f &weights)
{
    // u=P2−P1
    zeno::vec3f u = triangle_vertex_1 - triangle_vertex_0;
    // v=P3−P1
    zeno::vec3f v = triangle_vertex_2 - triangle_vertex_0;
    // n=u×v
    zeno::vec3f n = zeno::cross(u,v);
    // w=P−P1
    zeno::vec3f w = query_point - triangle_vertex_0;
    // Barycentric coordinates of the projection P′of P onto T:
    // γ=[(u×w)⋅n]/n²
    float gamma = zeno::dot(n, zeno::cross(u,w)) / zeno::dot(n,n);
    // β=[(w×v)⋅n]/n²
    float beta = zeno::dot(zeno::cross(w, v),n) / zeno::dot(n,n);
    float alpha = 1 - gamma - beta;
    // The point P′ lies inside T if:
    return ((0 <= alpha) && (alpha <= 1) &&
            (0 <= beta)  && (beta  <= 1) &&
            (0 <= gamma) && (gamma <= 1));
}
// to do where to put this func??
template <class T>
T baryCentricInterpolation(T &v1, T &v2, T &v3, zeno::vec3f &p,
                           zeno::vec3f &vert1, zeno::vec3f &vert2,
                           zeno::vec3f &vert3) {
  float a1 = area(p, vert2, vert3);
  float a2 = area(p, vert1, vert3);
  float a = area(vert1, vert2, vert3);
  float w1 = a1 / (a+1e-7);
  float w2 = a2 / (a+1e-7);
  float w3 = 1 - w1 - w2;
  return w1 * v1 + w2 * v2 + w3 * v3;
}

struct SprayParticles : zeno::INode {
  virtual void apply() override {
    auto dx = get_input("Dx")->as<NumericObject>()->get<float>();
    
    //auto channel = std::get<std::string>(get_param("channel"));
    
    auto prim = get_input("TrianglePrim")->as<PrimitiveObject>();
    
    
    auto result = zeno::IObject::make<PrimitiveObject>();
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
      // pos.emplace_back(a);
      // vel.emplace_back(vel1);
      // pos.emplace_back(b);
      // vel.emplace_back(vel2);
      // pos.emplace_back(c);
      // vel.emplace_back(vel3);
      data.emplace_back(std::make_tuple(a, zeno::vec3f(vi[0],vi[1],vi[2])));
      data.emplace_back(std::make_tuple(b, zeno::vec3f(vi[0],vi[1],vi[2])));
      data.emplace_back(std::make_tuple(c, zeno::vec3f(vi[0],vi[1],vi[2])));
      for (int kk = 0; kk < kn; kk++) {
        zeno::vec3f vij = b + (float)kk * 0.5f * dx * dir3;
        if (ptInTriangle(vij, a, b, c)) {
              data.emplace_back(std::make_tuple(vij,zeno::vec3f(vi[0],vi[1],vi[2])));
        }
      }
      for (int ii = 0; ii < in; ii++) {
        zeno::vec3f vij = a + (float)ii * 0.5f * dx * dir1;
        if (ptInTriangle(vij, a, b, c)) {
          data.emplace_back(std::make_tuple(vij,zeno::vec3f(vi[0],vi[1],vi[2])));
        }
      }
      for (int jj = 0; jj < jn; jj++) {
        zeno::vec3f vij = a + (float)jj * 0.5f * dx * dir2;
        if (ptInTriangle(vij, a, b, c)) {
          data.emplace_back(std::make_tuple(vij,zeno::vec3f(vi[0],vi[1],vi[2])));
        }
      }
      for (int ii = 0; ii < in; ii++) {
        for (int jj = 0; jj < jn; jj++) {
          zeno::vec3f vij =
              a + (float)ii * 0.5f * dx * dir1 + (float)jj * 0.5f * dx * dir2;
          if (ptInTriangle(vij, a, b, c)) {
            data.emplace_back(std::make_tuple(vij,zeno::vec3f(vi[0],vi[1],vi[2])));
          }
        }
      }
    });
    result->resize(data.size());
    result->add_attr<zeno::vec3f>("TriIndex");
#pragma omp parallel for
    for (int index = 0; index < data.size(); index++) {
      result->verts[index] = std::get<0>(data[index]);
      result->attr<zeno::vec3f>("TriIndex")[index] = std::get<1>(data[index]);
    }
    set_output("particles", std::move(result));
  }
};

static int defSprayParticles = zeno::defNodeClass<SprayParticles>(
    "SprayParticles", {/* inputs: */ {
                           "TrianglePrim", 
                           {"float", "Dx", "0.01"},
                       },
                       /* outputs: */
                       {
                           "particles",
                       },
                       /* params: */
                       {
                       },
                       /* category: */
                       {
                           "primitive",
                       }});

static void BarycentricInterpPrimitive(PrimitiveObject* dst, const PrimitiveObject* src, int i, int v0, int v1, int v2,
zeno::vec3f &pdst, zeno::vec3f &pos0, zeno::vec3f &pos1, zeno::vec3f &pos2)
{
  for(auto key:src->attr_keys())
  {
      if (key!="TriIndex")
      std::visit([i, v0, v1, v2, &pdst, &pos0, &pos1, &pos2](auto &&dst, auto &&src) {
          using DstT = std::remove_cv_t<std::remove_reference_t<decltype(dst)>>;
          using SrcT = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;
          if constexpr (std::is_same_v<DstT, SrcT>) {
              auto val1 = src[v0];
              auto val2 = src[v1];
              auto val3 = src[v2];
              // auto val = baryCentricInterpolation(val1, val2, val3, pdst, 
              //              pos0, pos1, pos2);
              zeno::vec3f w;
              pointInTriangle(pdst, pos0, pos1, pos2, w);
              dst[i] = w[0]*val1 + w[1]*val2 + w[2]*val3;
          } else  {
              throw std::runtime_error("the same attr of both primitives are of different types.");
          }
      }, dst->attr(key), src->attr(key));
  }
}
struct InterpMeshBarycentric : INode{
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("MeshPrim");
    auto points = get_input<PrimitiveObject>("Particles");
    auto triIndex = points->attr<zeno::vec3f>("TriIndex");
    for(auto key:prim->attr_keys())
    { 
        if(key!="pos"&&key!="TriIndex")
       std::visit([&points, key](auto &&ref) {
                    using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                    points->add_attr<T>(key);
                }, prim->attr(key));
    }
//#pragma omp parallel for
  tbb::parallel_for((size_t)0, (size_t)(points->size()), (size_t)1, [&](size_t index) 
  {
      auto tidx = triIndex[index];
      int v0 = tidx[0], v1 = tidx[1], v2 = tidx[2];
      BarycentricInterpPrimitive(points.get(), prim.get(), index, v0, v1, v2, points->verts[index], prim->verts[v0], prim->verts[v1], prim->verts[v2]);
  });

    set_output("oParticles", get_input("Particles"));

  }
};

ZENDEFNODE(InterpMeshBarycentric, {
    {
      "MeshPrim",
      "Particles",
    },
    {
      "oParticles"
    },
    {},
    {"primitive"},
    });

struct PrimFindTriangle : INode{
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("MeshPrim");
    auto points = get_input<PrimitiveObject>("Particles");
    auto triIndex = points->add_attr<zeno::vec3f>("TriIndex");
    for(auto key:prim->attr_keys())
    { 
        if(key!="pos")
       std::visit([&points, key](auto &&ref) {
                    using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                    points->add_attr<T>(key);
                }, prim->attr(key));
    }
//#pragma omp parallel for
  tbb::parallel_for((size_t)0, (size_t)(points->size()), (size_t)1, [&](size_t index) 
  {
    float mind = 999999999.0;
    for(auto tidx:prim->tris ){
      int v0 = tidx[0], v1 = tidx[1], v2 = tidx[2];
      zeno::vec3f c = (prim->verts[v0] + prim->verts[v1] + prim->verts[v2])/3.0;
      float d = zeno::length(points->verts[index] - c);
      zeno::vec3f w;
      if(d<mind && pointInTriangle(points->verts[index], prim->verts[v0], prim->verts[v1], prim->verts[v2], w))
      {
        points->attr<zeno::vec3f>("TriIndex")[index] = zeno::vec3f(v0,v1,v2);
        mind = d;
      }
    }
  });

    set_output("oParticles", get_input("Particles"));

  }
};

ZENDEFNODE(PrimFindTriangle, {
    {
      "MeshPrim",
      "Particles",
    },
    {
      "oParticles"
    },
    {},
    {"primitive"},
    });
} // namespace zeno
