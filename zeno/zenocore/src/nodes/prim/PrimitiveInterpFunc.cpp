#include "iostream"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

namespace zeno{
// todo where to put this func???
static float area(zeno::vec3f &p1, zeno::vec3f &p2, zeno::vec3f &p3) {
  zeno::vec3f e1 = p3 - p1;
  zeno::vec3f e2 = p2 - p1;
  zeno::vec3f areavec = zeno::cross(e1, e2);
  return 0.5 * sqrt(zeno::dot(areavec, areavec));
}
// todo where to put this func????
static bool ptInTriangle(zeno::vec3f p, zeno::vec3f p0, zeno::vec3f p1,
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
static bool pointInTriangle(const zeno::vec3f& query_point,
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
    weights = vec3f(gamma, beta, alpha);
    return ((0 <= alpha) && (alpha <= 1) &&
            (0 <= beta)  && (beta  <= 1) &&
            (0 <= gamma) && (gamma <= 1));
}
// to do where to put this func??
static void baryCentricInterpolation(zeno::vec3f &p,
                           zeno::vec3f &vert1, zeno::vec3f &vert2,
                           zeno::vec3f &vert3, zeno::vec3f &weights) {
  float a1 = area(p, vert2, vert3);
  float a2 = area(p, vert1, vert3);
  float a = area(vert1, vert2, vert3);
  float w1 = a1 / (a+1e-7);
  float w2 = a2 / (a+1e-7);
  float w3 = 1 - w1 - w2;
  weights = zeno::vec3f(w1, w2, w3);
}
ZENO_API void BarycentricInterpPrimitive(PrimitiveObject* _dst, const PrimitiveObject* _src, int i, int v0, int v1, int v2,
zeno::vec3f &pdst, zeno::vec3f &pos0, zeno::vec3f &pos1, zeno::vec3f &pos2)
{
  for(auto key:_src->attr_keys())
  {
      if (key!="TriIndex"&&key!="pos"&&key!="InitWeight")
      std::visit([i, v0, v1, v2, &pdst, &pos0, &pos1, &pos2](auto &&dst, auto &&src) {
          using DstT = std::remove_cv_t<std::remove_reference_t<decltype(dst)>>;
          using SrcT = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;
          if constexpr (std::is_same_v<DstT, SrcT>) {
              auto val1 = src[v0];
              auto val2 = src[v1];
              auto val3 = src[v2];
              zeno::vec3f w;
              baryCentricInterpolation(pdst, pos0, pos1, pos2, w);
              auto val = w[0] * val1 + w[1] * val2 + w[2]*val3;
              dst[i] = val;
          } else  {
              throw std::runtime_error("the same attr of both primitives are of different types.");
          }
      }, _dst->attr(key), _src->attr(key));
  }
}
} // namespace zeno
