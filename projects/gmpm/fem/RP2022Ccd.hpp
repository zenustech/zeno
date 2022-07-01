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
  while (edgeEdgeCCD(ea0, ea1, eb0, eb1, ea0end, ea1end, eb0end, eb1end)) {
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

} // namespace rpccd

namespace accd {

template <typename VecT>
constexpr bool
ptccd(VecT p, VecT t0, VecT t1, VecT t2, VecT dp, VecT dt0, VecT dt1, VecT dt2,
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc, typename VecT::value_type tStart = 0) {
  using T = typename VecT::value_type;
  auto mov = (dt0 + dt1 + dt2 + dp) / 4;
  dt0 -= mov;
  dt1 -= mov;
  dt2 -= mov;
  dp -= mov;
  T dispMag2Vec[3] = {dt0.l2NormSqr(), dt1.l2NormSqr(), dt2.l2NormSqr()};
  T tmp = zs::limits<T>::lowest();
  for (int i = 0; i != 3; ++i)
    if (dispMag2Vec[i] > tmp)
      tmp = dispMag2Vec[i];
  T maxDispMag = dp.norm() + zs::sqrt(tmp);
  if (maxDispMag == 0)
    return false;

  T dist2_cur = dist2_pt_unclassified(p, t0, t1, t2);
  T dist_cur = zs::sqrt(dist2_cur);
  T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
  T toc_prev = toc;
  toc = tStart;
  while (true) {
    T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) /
                      ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn pt!\n");

    p += tocLowerBound * dp;
    t0 += tocLowerBound * dt0;
    t1 += tocLowerBound * dt1;
    t2 += tocLowerBound * dt2;
    dist2_cur = dist2_pt_unclassified(p, t0, t1, t2);
    dist_cur = zs::sqrt(dist2_cur);
    if (toc &&
        ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
      toc = toc_prev;
      return false;
    }
  }
  return true;
}
template <typename VecT>
constexpr bool
eeccd(VecT ea0, VecT ea1, VecT eb0, VecT eb1, VecT dea0, VecT dea1, VecT deb0,
      VecT deb1, typename VecT::value_type eta,
      typename VecT::value_type thickness, typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  auto mov = (dea0 + dea1 + deb0 + deb1) / 4;
  dea0 -= mov;
  dea1 -= mov;
  deb0 -= mov;
  deb1 -= mov;
  T maxDispMag = zs::sqrt(zs::max(dea0.l2NormSqr(), dea1.l2NormSqr())) +
                 zs::sqrt(zs::max(deb0.l2NormSqr(), deb1.l2NormSqr()));
  if (maxDispMag == 0)
    return false;

  T dist2_cur = dist2_ee_unclassified(ea0, ea1, eb0, eb1);
  T dFunc = dist2_cur - thickness * thickness;
  if (dFunc <= 0) {
    // since we ensured other place that all dist smaller than dHat are
    // positive, this must be some far away nearly parallel edges
    T dists[] = {(ea0 - eb0).l2NormSqr(), (ea0 - eb1).l2NormSqr(),
                 (ea1 - eb0).l2NormSqr(), (ea1 - eb1).l2NormSqr()};
    {
      dist2_cur = zs::limits<T>::max();
      for (const auto &dist : dists)
        if (dist < dist2_cur)
          dist2_cur = dist;
      // dist2_cur = *std::min_element(dists.begin(), dists.end());
    }
    dFunc = dist2_cur - thickness * thickness;
  }
  T dist_cur = zs::sqrt(dist2_cur);
  T gap = eta * dFunc / (dist_cur + thickness);
  T toc_prev = toc;
  toc = 0;
  while (true) {
    T tocLowerBound = (1 - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn ee!\n");

    ea0 += tocLowerBound * dea0;
    ea1 += tocLowerBound * dea1;
    eb0 += tocLowerBound * deb0;
    eb1 += tocLowerBound * deb1;
    dist2_cur = dist2_ee_unclassified(ea0, ea1, eb0, eb1);
    dFunc = dist2_cur - thickness * thickness;
    if (dFunc <= 0) {
      // since we ensured other place that all dist smaller than dHat are
      // positive, this must be some far away nearly parallel edges
      T dists[] = {(ea0 - eb0).l2NormSqr(), (ea0 - eb1).l2NormSqr(),
                   (ea1 - eb0).l2NormSqr(), (ea1 - eb1).l2NormSqr()};
      {
        dist2_cur = zs::limits<T>::max();
        for (const auto &dist : dists)
          if (dist < dist2_cur)
            dist2_cur = dist;
      }
      dFunc = dist2_cur - thickness * thickness;
    }
    dist_cur = zs::sqrt(dist2_cur);
    if (toc && (dFunc / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
      toc = toc_prev;
      return false;
    }
  }
  return true;
}

} // namespace accd

namespace ticcd {

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

  constexpr zs::vec<double, 3> err(-1, -1, -1);
  double ms = 1e-8;
  double toi{};
  const double tolerance = 1e-6;
  const double t_max = 1;
  const int max_itr = 1e6;
  double output_tolerance = 1e-6;
  while (vertexFaceCCD(p, t0, t1, t2, pend, t0end, t1end, t2end, err, ms, toi,
                       tolerance, t_max, max_itr, output_tolerance, true)) {
    t = zs::min(t / 2, toi);
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

  constexpr zs::vec<double, 3> err(-1, -1, -1);
  double ms = 1e-8;
  double toi{};
  const double tolerance = 1e-6;
  const double t_max = 1;
  const int max_itr = 1e6;
  double output_tolerance = 1e-6;
  while (edgeEdgeCCD(ea0, ea1, eb0, eb1, ea0end, ea1end, eb0end, eb1end, err,
                     ms, toi, tolerance, t_max, max_itr, output_tolerance,
                     true)) {
    t = zs::min(t / 2, toi);
    // t /= 2;
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
} // namespace ticcd

} // namespace zeno