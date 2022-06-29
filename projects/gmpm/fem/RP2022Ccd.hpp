#pragma once
#include "zensim/geometry/Geometry.hpp"
#include "zensim/math/Vec.h"

namespace zeno {

namespace rpccd {

template <typename VecT>
constexpr bool
ptccd(const VecT &p, const VecT &t0, const VecT &t1, const VecT &t2,
      const VecT &dp, const VecT &dt0, const VecT &dt1, const VecT &dt2,
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  T t = toc;
  auto pend = p + t * dp;
  auto t0end = t0 + t * dt0;
  auto t1end = t1 + t * dt1;
  auto t2end = t2 + t * dt2;
  while (vertexFaceCCD(p, t0, t1, t2, pend, t0end, t1end, t2end)) {
    t /= 2;
    pend = p + t * dp;
    t0end = t0 + t * dt0;
    t1end = t1 + t * dt1;
    t2end = t2 + t * dt2;
  }

  if (t == toc) {
    return false;
  } else {
    toc = t * (1 - eta);
    return true;
  }
}

#if 0
template <typename VecT>
constexpr bool
eeccd(const VecT &ea0, const VecT &ea1, const VecT &eb0, const VecT &eb1,
      const VecT &dea0, const VecT &dea1, const VecT &deb0, const VecT &deb1,
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  T t = toc;
  auto ea0end = ea0 + t * dea0;
  auto ea1end = ea1 + t * dea1;
  auto eb0end = eb0 + t * deb0;
  auto eb1end = eb1 + t * deb1;
  while (
      Intersect_EE_robust(ea0, ea1, eb0, eb1, ea0end, ea1end, eb0end, eb1end)) {
    t /= 2;
    ea0end = ea0 + t * dea0;
    ea1end = ea1 + t * dea1;
    eb0end = eb0 + t * deb0;
    eb1end = eb1 + t * deb1;
  }

  if (t == toc) {
    return false;
  } else {
    toc = t * (1 - eta);
    return true;
  }
}
#endif

} // namespace rpccd

} // namespace zeno