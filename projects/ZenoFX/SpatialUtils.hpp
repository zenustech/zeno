#pragma once
#include <zeno/utils/vec.h>

namespace zeno {

inline auto dist_pp_sqr(const vec3f &a, const vec3f &b, vec3f &ws) noexcept {
  ws[0] = 1.f;
  ws[1] = ws[2] = 0.f;
  return lengthSquared(b - a);
}

inline auto dist_pp(const vec3f &a, const vec3f &b, vec3f &ws) {
  return std::sqrt(dist_pp_sqr(a, b, ws));
}

//! point-edge

inline int dist_pe_category(const vec3f &p, const vec3f &e0,
                            const vec3f &e1) noexcept {
  const auto e = e1 - e0;
  auto indicator = dot(e, p - e0) / lengthSquared(e);
  return indicator < 0 ? 0 : (indicator > 1 ? 1 : 2);
}

inline auto dist_pe_sqr(const vec3f &p, const vec3f &e0, const vec3f &e1, vec3f &ws) {
  using T = float;
  T ret = std::numeric_limits<T>::max();
  switch (dist_pe_category(p, e0, e1)) {
  case 0:
    ret = dist_pp_sqr(p, e0, ws);
    ws[0] = 1.f;
    ws[1] = ws[2] = 0.f;
    break;
  case 1:
    ret = dist_pp_sqr(p, e1, ws);
    ws[1] = 1.f;
    ws[0] = ws[2] = 0.f;
    break;
  case 2: {
    auto dir = e1 - e0;
    auto dir2 = lengthSquared(dir);
    auto dirNorm = std::sqrt(dir2);
    dir /= dirNorm;
    ret = lengthSquared(cross(e0 - p, e1 - p)) / dir2;
    auto e0Proj = dot(p - e0, dir) / dirNorm;
    auto e1Proj = -(dot(p - e1, dir)) / dirNorm;
    ws[0] = e1Proj;
    ws[1] = e0Proj;
    ws[2] = 0.f;
    break;
  }
  default:
    break;
  }
  return ret;
}

inline auto dist_pe(const vec3f &p, const vec3f &e0, const vec3f &e1, vec3f &ws) {
  return std::sqrt(dist_pe_sqr(p, e0, e1, ws));
}

//! point-triangle
// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2022
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.06

// ref: https://www.geometrictools.com/GTE/Mathematics/DistPointTriangle.h
inline auto dist_pt_sqr(const vec3f &p, const vec3f &t0, const vec3f &t1,
                        const vec3f &t2, vec3f &ws) noexcept {
  using T = float;
  using TV = vec3f;
  TV diff = t0 - p;
  TV e0 = t1 - t0;
  TV e1 = t2 - t0;
  T a00 = dot(e0, e0);
  T a01 = dot(e0, e1);
  T a11 = dot(e1, e1);
  T b0 = dot(diff, e0);
  T b1 = dot(diff, e1);
  T det = std::max(a00 * a11 - a01 * a01, (T)0);
  T s = a01 * b1 - a11 * b0;
  T t = a01 * b0 - a00 * b1;

  if (s + t <= det) {
    if (s < (T)0) {
      if (t < (T)0) { // region 4
        if (b0 < (T)0) {
          t = (T)0;
          if (-b0 >= a00)
            s = (T)1;
          else
            s = -b0 / a00;
        } else {
          s = (T)0;
          if (b1 >= (T)0)
            t = (T)0;
          else if (-b1 >= a11)
            t = (T)1;
          else
            t = -b1 / a11;
        }
      } else { // region 3
        s = (T)0;
        if (b1 >= (T)0)
          t = (T)0;
        else if (-b1 >= a11)
          t = (T)1;
        else
          t = -b1 / a11;
      }
    } else if (t < (T)0) { // region 5
      t = (T)0;
      if (b0 >= (T)0)
        s = (T)0;
      else if (-b0 >= a00)
        s = (T)1;
      else
        s = -b0 / a00;
    } else { // region 0
             // minimum at interior point
      s /= det;
      t /= det;
    }
  } else {
    T tmp0{}, tmp1{}, numer{}, denom{};
    if (s < (T)0) { // region 2
      tmp0 = a01 + b0;
      tmp1 = a11 + b1;
      if (tmp1 > tmp0) {
        numer = tmp1 - tmp0;
        denom = a00 - (a01 + a01) + a11;
        if (numer >= denom) {
          s = (T)1;
          t = (T)0;
        } else {
          s = numer / denom;
          t = (T)1 - s;
        }
      } else {
        s = (T)0;
        if (tmp1 <= (T)0)
          t = (T)1;
        else if (b1 >= (T)0)
          t = (T)0;
        else
          t = -b1 / a11;
      }
    } else if (t < (T)0) { // region 6
      tmp0 = a01 + b1;
      tmp1 = a00 + b0;
      if (tmp1 > tmp0) {
        numer = tmp1 - tmp0;
        denom = a00 - (a01 + a01) + a11;
        if (numer >= denom) {
          t = (T)1;
          s = (T)0;
        } else {
          t = numer / denom;
          s = (T)1 - t;
        }
      } else {
        t = (T)0;
        if (tmp1 <= (T)0)
          s = (T)1;
        else if (b0 >= (T)0)
          s = (T)0;
        else
          s = -b0 / a00;
      }
    } else { // region 1
      numer = a11 + b1 - a01 - b0;
      if (numer <= (T)0) {
        s = (T)0;
        t = (T)1;
      } else {
        denom = a00 - (a01 + a01) + a11;
        if (numer >= denom) {
          s = (T)1;
          t = (T)0;
        } else {
          s = numer / denom;
          t = (T)1 - s;
        }
      }
    }
  }
  auto hitpoint = t0 + s * e0 + t * e1;
  ws[0] = 1 - s - t;
  ws[1] = s;
  ws[2] = t;
  return lengthSquared(p - hitpoint);
}

inline auto dist_pt(const vec3f &p, const vec3f &t0, const vec3f &t1,
                    const vec3f &t2, vec3f &ws) {
  return std::sqrt(dist_pt_sqr(p, t0, t1, t2, ws));
}

} // namespace zeno