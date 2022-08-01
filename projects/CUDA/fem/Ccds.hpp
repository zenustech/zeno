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
  int iter = 0;
  while (++iter < 20000) {
    // while (true) {
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
      typename VecT::value_type thickness, typename VecT::value_type &toc,
      typename VecT::value_type tStart = 0) {
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
  toc = tStart;
  int iter = 0;
  while (++iter < 20000) {
    // while (true) {
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
  bool earlyTerminate = false;
  double ms = 1e-8;
  double toi{};
  const double tolerance = 1e-6;
  const double t_max = 1;
  const int max_itr = 3e5;
  double output_tolerance = 1e-6;
  while (vertexFaceCCD(p, t0, t1, t2, pend, t0end, t1end, t2end, err, ms, toi,
                       tolerance, t_max, max_itr, output_tolerance,
                       earlyTerminate, true) &&
         !earlyTerminate) {
    t = zs::min(t / 2, toi);
    pend = p + t * dp;
    t0end = t0 + t * dt0;
    t1end = t1 + t * dt1;
    t2end = t2 + t * dt2;
  }

  if (earlyTerminate) {
    toc = t;
    if (accd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, toc)) {
      return true;
    }
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
  bool earlyTerminate = false;
  double ms = 1e-8;
  double toi{};
  const double tolerance = 1e-6;
  const double t_max = 1;
  const int max_itr = 3e5;
  double output_tolerance = 1e-6;
  while (edgeEdgeCCD(ea0, ea1, eb0, eb1, ea0end, ea1end, eb0end, eb1end, err,
                     ms, toi, tolerance, t_max, max_itr, output_tolerance,
                     earlyTerminate, true) &&
         !earlyTerminate) {
    t = zs::min(t / 2, toi);
    ea0end = ea0 + t * dea0;
    ea1end = ea1 + t * dea1;
    eb0end = eb0 + t * deb0;
    eb1end = eb1 + t * deb1;
  }

  if (earlyTerminate) {
    toc = t;
    if (accd::eeccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness,
                    toc)) {
      return true;
    }
  }

  if (t == toc) {
    return false;
  } else {
    toc = t * (1 - eta);
    return true;
  }
}
} // namespace ticcd

namespace deprecated {

template <typename VecT,
          zs::enable_if_all<VecT::dim == 1, VecT::extent == 3> = 0>
constexpr auto solve_quadratic(const zs::VecInterface<VecT> &c,
                               double eps = 1e-8) {
  using T = typename VecT::value_type;
  using RetT = typename VecT::template variant_vec<
      T, zs::integer_seq<typename VecT::index_type, 2>>;
  auto s = RetT::zeros();
  // make sure we have a d2 equation
  if (zs::abs(c[2]) < eps) {
    if (zs::abs(c[1]) < eps)
      return zs::make_tuple(0, s);
    s[0] = -c[0] / c[1];
    return zs::make_tuple(1, s);
  }

  T p{}, q{}, D{};
  // normal for: x^2 + px + q
  p = c[1] / (2 * c[2]);
  q = c[0] / c[2];
  D = p * p - q;

  if (zs::abs(D) < eps) {
    // one float root
    s[0] = s[1] = -p;
    return zs::make_tuple(1, s);
  }

  if (D < eps)
    // no real root
    return zs::make_tuple(0, s);

  else {
    // two real roots, s[0] < s[1]
    auto sqrt_D = zs::sqrt(D);
    s[0] = -sqrt_D - p;
    s[1] = sqrt_D - p;
    return zs::make_tuple(2, s);
  }
}
template <typename VecT, typename T,
          zs::enable_if_all<VecT::dim == 1, VecT::extent == 3,
                            std::is_floating_point_v<T>> = 0>
[[nodiscard("VF coplanarity cubic equation coefficients")]] constexpr auto
cubic_eqn_VF(const zs::VecInterface<VecT> a0, const zs::VecInterface<VecT> ad,
             const zs::VecInterface<VecT> b0, const zs::VecInterface<VecT> bd,
             const zs::VecInterface<VecT> c0, const zs::VecInterface<VecT> cd,
             const zs::VecInterface<VecT> p0, const zs::VecInterface<VecT> pd,
             T thickness) noexcept {
  auto dab = bd - ad, dac = cd - ad, dap = pd - ad;
  auto oab = b0 - a0, oac = c0 - a0, oap = p0 - a0;
  auto dabXdac = cross(dab, dac);
  auto dabXoac = cross(dab, oac);
  auto oabXdac = cross(oab, dac);
  auto oabXoac = cross(oab, oac);

  T a = dot(dap, dabXdac);
  T b = dot(oap, dabXdac) + dot(dap, dabXoac + oabXdac);
  T c = dot(dap, oabXoac) + dot(oap, dabXoac + oabXdac);
  T d = dot(oap, oabXoac);
  if (d > 0)
    d -= thickness;
  return zs::make_tuple(a, b, c, d);
}
template <typename VecT, typename T,
          zs::enable_if_all<VecT::dim == 1, VecT::extent == 3,
                            std::is_floating_point_v<T>> = 0>
[[nodiscard("EE coplanaritycubic equation coefficients")]] constexpr auto
cubic_eqn_EE(const zs::VecInterface<VecT> a0, const zs::VecInterface<VecT> ad,
             const zs::VecInterface<VecT> b0, const zs::VecInterface<VecT> bd,
             const zs::VecInterface<VecT> c0, const zs::VecInterface<VecT> cd,
             const zs::VecInterface<VecT> d0, const zs::VecInterface<VecT> dd,
             T thickness) {
  auto dba = bd - ad, ddc = dd - cd, dca = cd - ad;
  auto odc = d0 - c0, oba = b0 - a0, oca = c0 - a0;
  auto dbaXddc = cross(dba, ddc);
  auto dbaXodc = cross(dba, odc);
  auto obaXddc = cross(oba, ddc);
  auto obaXodc = cross(oba, odc);

  T a = dot(dca, dbaXddc);
  T b = dot(oca, dbaXddc) + dot(dca, dbaXodc + obaXddc);
  T c = dot(dca, obaXodc) + dot(oca, dbaXodc + obaXddc);
  T d = dot(oca, obaXodc);

  if (d > 0)
    d -= thickness;
  return zs::make_tuple(a, b, c, d);
}
template <typename VecT>
constexpr bool pt_ccd(VecT p, VecT t0, VecT t1, VecT t2, VecT dp, VecT dt0,
                      VecT dt1, VecT dt2, typename VecT::value_type thickness,
                      typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  auto [A, B, C, D] = cubic_eqn_VF(t0, dt0, t1, dt1, t2, dt2, p, dp, thickness);
  if (zs::abs(A) < 1e-8 && zs::abs(B) < 1e-8 && zs::abs(C) < 1e-8 &&
      zs::abs(D) < 1e-8)
    return false;
  // tmin, tmax
  auto [numSols, ts] = solve_quadratic(zs::vec<T, 3>{C, 2 * B, 3 * A});
  if (numSols != 2) {
    // no local minima & maxima
    return ptaccd(p, t0, t1, t2, dp, dt0, dt1, dt2, 0.1, thickness, toc);
  }
  // ensure ts[0] < ts[1]
  auto d = [&A = A, &B = B, &C = C, &D = D](T t) {
    auto t2 = t * 2;
    return A * t2 * t + B * t2 + C * t + D;
  };
  auto dDrv = [&A = A, &B = B, &C = C](T t) {
    return 3 * A * t * t + 2 * B * t + C;
  };
  if (d(0) < zs::limits<T>::min()) {
    printf("\n\n\n\n\nwhat the heck??? trange [%f, %f]; t0 %f d "
           "%f\n\n\n\n\n",
           (float)ts[0], (float)ts[1], 0.f, (float)d(0));
    return ptaccd(p, t0, t1, t2, dp, dt0, dt1, dt2, 0.1, thickness, toc);
  }
  T toi{toc};
  bool check = false;
  // coplanarity test
  // case 1
  if (A > zs::limits<T>::min()) {
    // ts[0] : tmax
    // ts[1] : tmin
    if (ts[1] > 0)
      if (d(ts[1]) <= 0) {
        // auto k = -B / (3 * A);
        if (d(ts[0]) < 0 && A * B < 0) {
          printf("\n\n\n\n\nwhat the heck??\n\n\n\n\n\n");
        }
        T st = 0;
        if (ts[0] > st)
          st = ts[0];
        toi = st + d(st) * 3 * A / B;
        check = true;
      }
  } else { // A < 0
    // ts[0] : tmin
    // ts[1] : tmax
    if (auto dmin = d(ts[0]); dmin > 0 || dmin <= 0 && ts[0] < 0) {
      toi = ts[1] + d(ts[1]) / -dDrv(toc);
      check = true;
    } else if (dmin <= 0 && ts[0] >= 0) {
      toi = d(0) / -dDrv(0);
      check = true;
    }
  }
  // inside test
  if (check) {
    if (d(toi) <= 0) {
      printf(
          "\n\n\n\n\nwhat the heck??? trange [%f, %f]; toi %f d %f\n\n\n\n\n",
          (float)ts[0], (float)ts[1], (float)toi, (float)d(toi));
    }
    if (toi > 0 && toi < toc) {
      if (d(toi) < zs::limits<T>::min())
        return ptaccd(p, t0, t1, t2, dp, dt0, dt1, dt2, 0.1, thickness, toc);
      toi *= 0.8;
      auto a_ = t0 + toi * dt0;
      auto b_ = t1 + toi * dt1;
      auto c_ = t2 + toi * dt2;
      auto p_ = p + toi * dp;
      auto n = cross(b_ - a_, c_ - a_);
      auto da = a_ - p_;
      auto db = b_ - p_;
      auto dc = c_ - p_;
#if 0
      if (dot(cross(db, dc), n) < 0)
        return false;
      if (dot(cross(dc, da), n) < 0)
        return false;
      if (dot(cross(da, db), n) < 0)
        return false;
#endif
      toc = toi;
      // return true;
      return ptaccd(p_, a_, b_, c_, dp, dt0, dt1, dt2, 0.1, thickness, toc,
                    toi);
    }
  }
  return false;
}
template <typename VecT>
constexpr bool ee_ccd(VecT ea0, VecT ea1, VecT eb0, VecT eb1, VecT dea0,
                      VecT dea1, VecT deb0, VecT deb1,
                      typename VecT::value_type thickness,
                      typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  auto [A, B, C, D] =
      cubic_eqn_EE(ea0, dea0, ea1, dea1, eb0, deb0, eb1, deb1, thickness);
  if (zs::abs(A) < 1e-8 && zs::abs(B) < 1e-8 && zs::abs(C) < 1e-8 &&
      zs::abs(D) < 1e-8)
    return false;
  // tmin, tmax
  auto [numSols, ts] = solve_quadratic(zs::vec<T, 3>{C, 2 * B, 3 * A});
  if (numSols != 2) {
    // no local minima & maxima
    return ee_accd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, 0.1, thickness,
                   toc);
  }
  auto d = [&A = A, &B = B, &C = C, &D = D](T t) {
    auto t2 = t * 2;
    return A * t2 * t + B * t2 + C * t + D;
  };
  auto dDrv = [&A = A, &B = B, &C = C](T t) {
    return 3 * A * t * t + 2 * B * t + C;
  };
  T toi{toc};
  bool check = false;
  // coplanarity test
  // case 1
  if (A > 0) {
    // ts[0] : tmax
    // ts[1] : tmin
    if (ts[1] > 0)
      if (d(ts[1]) <= 0) {
        // auto k = -B / (3 * A);
        if (d(ts[0]) < 0 && A * B < 0) {
          printf("\n\n\n\n\nwhat the heck??\n\n\n\n\n\n");
        }
        T st = 0;
        if (ts[0] > st)
          st = ts[0];
        toi = st + d(st) * 3 * A / B;
        check = true;
      }
  } else { // A < 0
    // ts[0] : tmin
    // ts[1] : tmax
    if (auto dmin = d(ts[0]); dmin > 0 || dmin <= 0 && ts[0] < 0) {
      toi = ts[1] + d(ts[1]) / -dDrv(toc);
      check = true;
    } else if (dmin <= 0 && ts[0] >= 0) {
      toi = d(0) / -dDrv(0);
      check = true;
    }
  }
  // inside test
  if (check) {
    if (d(toi) <= 0 || d(0) <= 0 || d(toc) <= 0) {
      printf("\n\n\n\n\nwhat the heck??? trange [%f, %f]; toi %f d %f; t0 %f d "
             "%f; t1 %f d "
             "%f\n\n\n\n\n",
             (float)ts[0], (float)ts[1], (float)toi, (float)d(toi), 0.f,
             (float)d(0), (float)toc, (float)d(toc));
    }
    if (toi > 0 && toi < toc) {
      auto p1 = ea0 + toi * dea0;
      auto p2 = ea1 + toi * dea1;
      auto p3 = eb0 + toi * deb0;
      auto p4 = eb1 + toi * deb1;
      auto p13 = p1 - p3;
      auto p43 = p4 - p3;
#if 0
      if (dot(cross(db, dc), n) < 0)
        return false;
      if (dot(cross(dc, da), n) < 0)
        return false;
      if (dot(cross(da, db), n) < 0)
        return false;
#endif
      toc = toi;
      return true;
    }
  }
  return true;
}

} // namespace deprecated

} // namespace zeno