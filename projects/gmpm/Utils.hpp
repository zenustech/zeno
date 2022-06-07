#pragma once
#include <string_view>
#include <vector>
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/zeno.h>

#include "Structures.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/BoundingVolumeInterface.hpp"
#include "zensim/geometry/Distance.hpp"

// only use this macro within a zeno::INode::apply()
#define RETRIEVE_OBJECT_PTRS(T, STR)                                           \
  ([this](const std::string_view str) {                                        \
    std::vector<T *> objPtrs{};                                                \
    if (has_input<T>(str.data()))                                              \
      objPtrs.push_back(get_input<T>(str.data()).get());                       \
    else if (has_input<zeno::ListObject>(str.data())) {                        \
      auto &objSharedPtrLists = *get_input<zeno::ListObject>(str.data());      \
      for (auto &&objSharedPtr : objSharedPtrLists.get())                      \
        if (auto ptr = dynamic_cast<T *>(objSharedPtr.get()); ptr != nullptr)  \
          objPtrs.push_back(ptr);                                              \
    }                                                                          \
    return objPtrs;                                                            \
  })(STR);

namespace zeno {

template <typename Pol, typename TileVecT, int codim = 3>
zs::Vector<typename ZenoParticles::lbvh_t::Box>
retrieve_bounding_volumes(Pol &pol, const TileVecT &vtemp,
                          const typename ZenoParticles::particles_t &eles,
                          zs::wrapv<codim> = {}, float thickness = 0.f,
                          const zs::SmallString &xTag = "xn") {
  using namespace zs;
  using bv_t = typename ZenoParticles::lbvh_t::Box;
  static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
  constexpr auto space = Pol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  Vector<bv_t> ret{eles.get_allocator(), eles.size()};
  pol(zs::range(eles.size()), [eles = proxy<space>({}, eles),
                               bvs = proxy<space>(ret),
                               vtemp = proxy<space>({}, vtemp),
                               codim_v = wrapv<codim>{}, xTag,
                               thickness] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = vtemp.template pack<3>(xTag, inds[0]);
    bv_t bv{x0, x0};
    for (int d = 1; d != dim; ++d)
      merge(bv, vtemp.template pack<3>(xTag, inds[d]));
    bv._min -= thickness / 2;
    bv._max += thickness / 2;
    bvs[ei] = bv;
  });
  return ret;
}

// for ccd
template <typename Pol, typename TileVecT0, typename TileVecT1, int codim = 3>
zs::Vector<typename ZenoParticles::lbvh_t::Box> retrieve_bounding_volumes(
    Pol &pol, const TileVecT0 &verts,
    const typename ZenoParticles::particles_t &eles, const TileVecT1 &vtemp,
    zs::wrapv<codim> = {}, float stepSize = 1.f, float thickness = 0.f,
    const zs::SmallString &xTag = "xn", const zs::SmallString &dirTag = "dir") {
  using namespace zs;
  using bv_t = typename ZenoParticles::lbvh_t::Box;
  static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
  constexpr auto space = Pol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  Vector<bv_t> ret{eles.get_allocator(), eles.size()};
  pol(zs::range(eles.size()), [eles = proxy<space>({}, eles),
                               bvs = proxy<space>(ret),
                               verts = proxy<space>({}, verts),
                               vtemp = proxy<space>({}, vtemp),
                               codim_v = wrapv<codim>{}, xTag, dirTag, stepSize,
                               thickness] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = verts.template pack<3>(xTag, inds[0]);
    auto dir0 = vtemp.template pack<3>(dirTag, inds[0]);
    bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
    for (int d = 1; d != dim; ++d) {
      auto x = verts.template pack<3>(xTag, inds[d]);
      auto dir = vtemp.template pack<3>(dirTag, inds[d]);
      merge(bv, x);
      merge(bv, x + stepSize * dir);
    }
    bv._min -= thickness / 2;
    bv._max += thickness / 2;
    bvs[ei] = bv;
  });
  return ret;
}

// for ccd
template <typename VecT>
constexpr bool pt_accd(VecT p, VecT t0, VecT t1, VecT t2, VecT dp, VecT dt0,
                       VecT dt1, VecT dt2, typename VecT::value_type eta,
                       typename VecT::value_type thickness,
                       typename VecT::value_type &toc) {
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
  toc = 0;
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
ee_accd(VecT ea0, VecT ea1, VecT eb0, VecT eb1, VecT dea0, VecT dea1, VecT deb0,
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

template <typename Pol, typename TileVecT, typename T>
bool find_self_intersection_free_stepsize(Pol &pol, ZenoParticles &zstets,
                                          const TileVecT &vtemp, T &stepSize,
                                          T xi) {
  using namespace zs;
  using bv_t = typename ZenoParticles::lbvh_t::Box;
  constexpr auto space = Pol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  const auto &verts = zstets.getParticles();

  const auto &surfaces = zstets[ZenoParticles::s_surfTriTag];

  const auto &surfEdges = zstets[ZenoParticles::s_surfEdgeTag];
  {
    if (!zstets.hasBvh(ZenoParticles::s_surfEdgeTag))
      zstets.bvh(ZenoParticles::s_surfEdgeTag)
          .build(pol, retrieve_bounding_volumes(pol, vtemp, surfEdges, vtemp,
                                                wrapv<2>{}, stepSize, xi * 2,
                                                "xn", "dir"));
    else
      zstets.bvh(ZenoParticles::s_surfEdgeTag)
          .refit(pol, retrieve_bounding_volumes(pol, vtemp, surfEdges, vtemp,
                                                wrapv<2>{}, stepSize, xi * 2,
                                                "xn", "dir"));
  }
  const auto &seBvh = zstets.bvh(ZenoParticles::s_surfEdgeTag);

  // query tri - edge intersection
  zs::Vector<int> intersected{vtemp.get_allocator(), 1};
  intersected.setVal(0);
  pol(Collapse{surfaces.size()}, [ses = proxy<space>({}, surfEdges),
                                  sts = proxy<space>({}, surfaces),
                                  verts = proxy<space>({}, verts),
                                  vtemp = proxy<space>({}, vtemp),
                                  intersected = proxy<space>(intersected),
                                  bvh = proxy<space>(seBvh),
                                  stepSize] ZS_LAMBDA(int sti) mutable {
    auto tri =
        sts.template pack<3>("inds", sti).template reinterpret_bits<int>();
    auto t0 = vtemp.pack<3>("xn", tri[0]);
    auto t1 = vtemp.pack<3>("xn", tri[1]);
    auto t2 = vtemp.pack<3>("xn", tri[2]);
    auto [mi, ma] = get_bounding_box(t0, t1);
    auto bv = bv_t{mi, ma};
    merge(bv, t2);
    bv._min = bv._min;
    bv._max = bv._max;
    bvh.iter_neighbors(bv, [&](int sei) {
      auto line =
          ses.template pack<2>("inds", sei).template reinterpret_bits<int>();
      if (tri[0] == line[0] || tri[0] == line[1] || tri[1] == line[0] ||
          tri[1] == line[1] || tri[2] == line[0] || tri[2] == line[1])
        return;
      // all affected by sticky boundary conditions
      if (reinterpret_bits<int>(verts("BCorder", tri[0])) == 3 &&
          reinterpret_bits<int>(verts("BCorder", tri[1])) == 3 &&
          reinterpret_bits<int>(verts("BCorder", tri[2])) == 3 &&
          reinterpret_bits<int>(verts("BCorder", line[0])) == 3 &&
          reinterpret_bits<int>(verts("BCorder", line[1])) == 3)
        return;
      // ccd
      if (et_intersected(vtemp.pack<3>("xn", line[0]),
                         vtemp.pack<3>("xn", line[1]), t0, t1, t2))
        atomic_cas(exec_cuda, &intersected[0], 0, 1);
    });
  });
  return intersected.getVal();
}

template <typename Pol, typename TileVecT, typename T>
void find_intersection_free_stepsize(Pol &pol, ZenoParticles &zstets,
                                     const TileVecT &vtemp, T &stepSize, T xi,
                                     T dHat, bool updateTets = true) {
  using namespace zs;
  using bv_t = typename ZenoParticles::lbvh_t::Box;
  constexpr auto space = Pol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  const auto &verts = zstets.getParticles();
  const auto &eles = zstets.getQuadraturePoints();

  const auto &surfaces = zstets[ZenoParticles::s_surfTriTag];
  if (updateTets) {
    if (!zstets.hasBvh(ZenoParticles::s_surfTriTag)) // build if bvh not exist
      zstets.bvh(ZenoParticles::s_surfTriTag)
          .build(pol, retrieve_bounding_volumes(pol, vtemp, surfaces, vtemp,
                                                wrapv<3>{}, stepSize, xi, "xn",
                                                "dir"));
    else
      zstets.bvh(ZenoParticles::s_surfTriTag)
          .refit(pol, retrieve_bounding_volumes(pol, vtemp, surfaces, vtemp,
                                                wrapv<3>{}, stepSize, xi, "xn",
                                                "dir"));
  }
  const auto &stBvh = zstets.bvh(ZenoParticles::s_surfTriTag);

  const auto &surfEdges = zstets[ZenoParticles::s_surfEdgeTag];
  if (updateTets) {
    if (!zstets.hasBvh(ZenoParticles::s_surfEdgeTag))
      zstets.bvh(ZenoParticles::s_surfEdgeTag)
          .build(pol, retrieve_bounding_volumes(pol, vtemp, surfEdges, vtemp,
                                                wrapv<2>{}, stepSize, xi, "xn",
                                                "dir"));
    else
      zstets.bvh(ZenoParticles::s_surfEdgeTag)
          .refit(pol, retrieve_bounding_volumes(pol, vtemp, surfEdges, vtemp,
                                                wrapv<2>{}, stepSize, xi, "xn",
                                                "dir"));
  }
  const auto &seBvh = zstets.bvh(ZenoParticles::s_surfEdgeTag);

  const auto &surfVerts = zstets[ZenoParticles::s_surfVertTag];

  // query pt
  zs::Vector<T> surfAlphas{surfVerts.get_allocator(), surfVerts.size()},
      finalAlpha{surfVerts.get_allocator(), 1};
  finalAlpha.setVal(stepSize);
  pol(Collapse{surfVerts.size()},
      [svs = proxy<space>({}, surfVerts), sts = proxy<space>({}, surfaces),
       verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
       surfAlphas = proxy<space>(surfAlphas),
       finalAlpha = proxy<space>(finalAlpha), bvh = proxy<space>(stBvh),
       stepSize, xi, thickness = xi / 2 + dHat] ZS_LAMBDA(int svi) mutable {
        auto vi = reinterpret_bits<int>(svs("inds", svi));
        auto p = vtemp.pack<3>("xn", vi);
        auto dp = vtemp.pack<3>("dir", vi);
        auto bv = bv_t{get_bounding_box(p, p + stepSize * dp)};
        bv._min -= thickness;
        bv._max += thickness;
        auto alpha = stepSize;
        bvh.iter_neighbors(bv, [&](int stI) {
          auto tri = sts.template pack<3>("inds", stI)
                         .template reinterpret_bits<int>();
          if (vi == tri[0] || vi == tri[1] || vi == tri[2])
            return;
          // all affected by sticky boundary conditions
          if (reinterpret_bits<int>(verts("BCorder", vi)) == 3 &&
              reinterpret_bits<int>(verts("BCorder", tri[0])) == 3 &&
              reinterpret_bits<int>(verts("BCorder", tri[1])) == 3 &&
              reinterpret_bits<int>(verts("BCorder", tri[2])) == 3)
            return;
          // ccd
          auto t0 = vtemp.pack<3>("xn", tri[0]);
          auto t1 = vtemp.pack<3>("xn", tri[1]);
          auto t2 = vtemp.pack<3>("xn", tri[2]);
          auto dt0 = vtemp.pack<3>("dir", tri[0]);
          auto dt1 = vtemp.pack<3>("dir", tri[1]);
          auto dt2 = vtemp.pack<3>("dir", tri[2]);
          pt_accd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, xi, alpha);
        });
        if (alpha < stepSize)
          atomic_min(exec_cuda, &finalAlpha[0], alpha);
      });
  // zs::reduce(pol, std::begin(surfAlphas), std::end(surfAlphas),
  // std::begin(finalAlpha), limits<T>::max(), getmin<T>{});
  auto surfAlpha = finalAlpha.getVal();
  fmt::print(fg(fmt::color::dark_cyan),
             "surface alpha: {}, default stepsize: {}\n", surfAlpha, stepSize);
  stepSize = surfAlpha;
  // query ee
  pol(Collapse{surfEdges.size()},
      [ses = proxy<space>({}, surfEdges), verts = proxy<space>({}, verts),
       vtemp = proxy<space>({}, vtemp), finalAlpha = proxy<space>(finalAlpha),
       bvh = proxy<space>(seBvh), stepSize, xi,
       thickness = xi / 2 + dHat] ZS_LAMBDA(int sei) mutable {
        auto edgeInds =
            ses.template pack<2>("inds", sei).template reinterpret_bits<int>();
        auto x0 = vtemp.pack<3>("xn", edgeInds[0]);
        auto x1 = vtemp.pack<3>("xn", edgeInds[1]);
        auto dea0 = vtemp.pack<3>("dir", edgeInds[0]);
        auto dea1 = vtemp.pack<3>("dir", edgeInds[1]);
        auto bv = bv_t{get_bounding_box(x0, x0 + stepSize * dea0)};
        merge(bv, x1);
        merge(bv, x1 + stepSize * dea1);
        bv._min -= thickness;
        bv._max += thickness;
        auto alpha = stepSize;
        bvh.iter_neighbors(bv, [&](int seI) {
          if (sei > seI)
            return;
          auto oEdgeInds = ses.template pack<2>("inds", seI)
                               .template reinterpret_bits<int>();
          if (edgeInds[0] == oEdgeInds[0] || edgeInds[0] == oEdgeInds[1] ||
              edgeInds[1] == oEdgeInds[0] || edgeInds[1] == oEdgeInds[1])
            return;
          // all affected by sticky boundary conditions
          if (reinterpret_bits<int>(verts("BCorder", edgeInds[0])) == 3 &&
              reinterpret_bits<int>(verts("BCorder", edgeInds[1])) == 3 &&
              reinterpret_bits<int>(verts("BCorder", oEdgeInds[0])) == 3 &&
              reinterpret_bits<int>(verts("BCorder", oEdgeInds[1])) == 3)
            return;
          // ccd
          auto eb0 = vtemp.pack<3>("xn", oEdgeInds[0]);
          auto eb1 = vtemp.pack<3>("xn", oEdgeInds[1]);
          auto deb0 = vtemp.pack<3>("dir", oEdgeInds[0]);
          auto deb1 = vtemp.pack<3>("dir", oEdgeInds[1]);
          ee_accd(x0, x1, eb0, eb1, dea0, dea1, deb0, deb1, (T)0.2, xi, alpha);
        });
        if (alpha < stepSize)
          atomic_min(exec_cuda, &finalAlpha[0], alpha);
      });
  stepSize = finalAlpha.getVal();
  fmt::print(fg(fmt::color::dark_cyan), "surf edge alpha: {}\n", stepSize);
}

template <typename Pol, typename TileVecT, typename T>
void find_boundary_intersection_free_stepsize(Pol &pol, ZenoParticles &zstets,
                                              const TileVecT &vtemp,
                                              ZenoParticles &zsboundary, T dt,
                                              T &stepSize,
                                              T xi) { // this has issues
  using namespace zs;
  constexpr T slackness = 0.8;
  using bv_t = typename ZenoParticles::lbvh_t::Box;
  constexpr auto space = Pol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  const auto &verts = zstets.getParticles();
  const auto &eles = zstets.getQuadraturePoints();

  const auto &surfaces = zstets[ZenoParticles::s_surfTriTag];
  const auto &surfEdges = zstets[ZenoParticles::s_surfEdgeTag];
  const auto &surfVerts = zstets[ZenoParticles::s_surfVertTag];

  ///
  xi /= slackness; // thicken for stability
  {
    auto bvs = retrieve_bounding_volumes(
        pol, zsboundary.getParticles(), zsboundary.getQuadraturePoints(),
        zsboundary.getParticles(), wrapv<3>{}, dt, xi, "x", "v");
    if (!zsboundary.hasBvh(ZenoParticles::s_elementTag))
      zsboundary.bvh(ZenoParticles::s_elementTag).build(pol, bvs);
    else
      zsboundary.bvh(ZenoParticles::s_elementTag).refit(pol, bvs);
  }
  const auto &triBvh = zsboundary.bvh(ZenoParticles::s_elementTag);

  {
    auto bvs = retrieve_bounding_volumes(
        pol, zsboundary.getParticles(),
        zsboundary[ZenoParticles::s_surfEdgeTag], zsboundary.getParticles(),
        wrapv<2>{}, dt, xi, "x", "v");
    if (!zsboundary.hasBvh(ZenoParticles::s_surfEdgeTag))
      zsboundary.bvh(ZenoParticles::s_surfEdgeTag).build(pol, bvs);
    else
      zsboundary.bvh(ZenoParticles::s_surfEdgeTag).refit(pol, bvs);
  }
  const auto &edgeBvh = zsboundary.bvh(ZenoParticles::s_surfEdgeTag);
  // query pt
  zs::Vector<T> surfAlphas{surfVerts.get_allocator(), surfVerts.size()},
      finalAlpha{surfVerts.get_allocator(), 1};

  finalAlpha.setVal(stepSize);
  pol(Collapse{surfVerts.size()},
      [svs = proxy<space>({}, surfVerts), vtemp = proxy<space>({}, vtemp),
       verts = proxy<space>({}, verts),
       // boundary
       bouVerts = proxy<space>({}, zsboundary.getParticles()),
       bouEles = proxy<space>({}, zsboundary.getQuadraturePoints()),
       surfAlphas = proxy<space>(surfAlphas),
       finalAlpha = proxy<space>(finalAlpha), bvh = proxy<space>(triBvh),
       stepSize, thickness = xi, dt] ZS_LAMBDA(int svi) mutable {
        auto vi = reinterpret_bits<int>(svs("inds", svi));
        // this vert affected by sticky boundary conditions
        if (reinterpret_bits<int>(verts("BCorder", vi)) == 3)
          return;
        auto p = vtemp.pack<3>("xn", vi);
        auto [mi, ma] = get_bounding_box(p - thickness / 2, p + thickness / 2);
        auto bv = bv_t{mi, ma};
        bvh.iter_neighbors(bv, [&](int stI) {
          auto tri = bouEles.template pack<3>("inds", stI)
                         .template reinterpret_bits<int>();
          // ccd
          auto alpha = stepSize;
          // surfAlphas[svi] = alpha;
          if (pt_accd(p, bouVerts.pack<3>("x", tri[0]).template cast<T>(),
                      bouVerts.pack<3>("x", tri[1]).template cast<T>(),
                      bouVerts.pack<3>("x", tri[2]).template cast<T>(),
                      vtemp.pack<3>("dir", vi),
                      bouVerts.pack<3>("v", tri[0]).template cast<T>() * dt,
                      bouVerts.pack<3>("v", tri[1]).template cast<T>() * dt,
                      bouVerts.pack<3>("v", tri[2]).template cast<T>() * dt,
                      (T)0.1, thickness, alpha))
            if (alpha < stepSize)
              // surfAlphas[svi] = alpha;
              atomic_min(exec_cuda, &finalAlpha[0], alpha);
        });
      });
  // zs::reduce(pol, std::begin(surfAlphas), std::end(surfAlphas),
  // std::begin(finalAlpha), limits<T>::max(), getmin<T>{});
  auto surfAlpha = finalAlpha.getVal();
  stepSize = surfAlpha;
  fmt::print(fg(fmt::color::dark_cyan),
             "surf alpha: {}, default stepsize: {}\n", surfAlpha, stepSize);
  // query ee
  zs::Vector<T> surfEdgeAlphas{surfEdges.get_allocator(), surfEdges.size()};
  pol(Collapse{surfEdges.size()},
      [ses = proxy<space>({}, surfEdges), vtemp = proxy<space>({}, vtemp),
       verts = proxy<space>({}, verts),
       // boundary
       bouVerts = proxy<space>({}, zsboundary.getParticles()),
       bouEles = proxy<space>({}, zsboundary.getQuadraturePoints()),
       surfEdgeAlphas = proxy<space>(surfEdgeAlphas),
       finalAlpha = proxy<space>(finalAlpha), bvh = proxy<space>(edgeBvh),
       stepSize, thickness = xi, dt] ZS_LAMBDA(int sei) mutable {
        auto edgeInds =
            ses.template pack<2>("inds", sei).template reinterpret_bits<int>();
        // both verts affected by sticky boundary conditions
        if (reinterpret_bits<int>(verts("BCorder", edgeInds[0])) == 3 &&
            reinterpret_bits<int>(verts("BCorder", edgeInds[1])) == 3)
          return;
        auto x0 = vtemp.pack<3>("xn", edgeInds[0]);
        auto x1 = vtemp.pack<3>("xn", edgeInds[1]);
        auto [mi, ma] = get_bounding_box(x0, x1);
        auto bv = bv_t{mi - thickness / 2, ma + thickness / 2};
        bvh.iter_neighbors(bv, [&](int seI) {
          if (sei > seI)
            return;
          auto oEdgeInds = ses.template pack<2>("inds", seI)
                               .template reinterpret_bits<int>();
          // ccd
          auto alpha = stepSize;
          // surfEdgeAlphas[sei] = alpha;
          if (ee_accd(
                  x0, x1,
                  bouVerts.pack<3>("x", oEdgeInds[0]).template cast<T>(),
                  bouVerts.pack<3>("x", oEdgeInds[1]).template cast<T>(),
                  vtemp.pack<3>("dir", edgeInds[0]),
                  vtemp.pack<3>("dir", edgeInds[1]),
                  bouVerts.pack<3>("v", oEdgeInds[0]).template cast<T>() * dt,
                  bouVerts.pack<3>("v", oEdgeInds[1]).template cast<T>() * dt,
                  (T)0.1, thickness, alpha))
            if (alpha < stepSize)
              // surfEdgeAlphas[sei] = alpha;
              atomic_min(exec_cuda, &finalAlpha[0], alpha);
        });
      });
#if 0
  zs::reduce(pol, std::begin(surfEdgeAlphas), std::end(surfEdgeAlphas),
             std::begin(finalAlpha), limits<T>::max(), getmin<T>{});
  stepSize = std::min(surfAlpha, finalAlpha.getVal());
#else
  stepSize = finalAlpha.getVal();
  fmt::print(fg(fmt::color::dark_cyan), "surf edge alpha: {}\n", surfAlpha);
#endif
}

template <typename ExecPol, typename TileVectorT, typename IndexBucketsT>
inline void spatial_hashing(ExecPol &pol, const TileVectorT &tvs,
                            const typename TileVectorT::value_type dx,
                            IndexBucketsT &ibs, bool init = true,
                            bool count_only = false) {
  using namespace zs;
  constexpr auto space = ExecPol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif

  auto allocator = tvs.get_allocator();
  auto mloc = allocator.location;
  if (init)
    ibs._dx = dx; // radius + radius;
  /// table
  auto &partition = ibs._table;
  using Partition = RM_CVREF_T(partition);
  if (init) {
    partition = Partition{tvs.size(), tvs.memspace(), tvs.devid()};

    // clean
    pol(range(partition._tableSize),
        [table = proxy<space>(partition)] ZS_LAMBDA(size_t i) mutable {
          table._table.keys[i] =
              Partition::key_t::uniform(Partition::key_scalar_sentinel_v);
          table._table.indices[i] = Partition::sentinel_v;
          table._table.status[i] = -1;
          if (i == 0)
            *table._cnt = 0;
        });
  }

  // compute sparsity
  pol(range(tvs.size()),
      [tvs = proxy<space>({}, tvs),
       ibs = proxy<space>(ibs)] ZS_LAMBDA(size_t pi) mutable {
        auto x = tvs.template pack<3>("x", pi);
        auto coord = ibs.bucketCoord(x);
        ibs.table.insert(coord);
      });
  auto numCells = partition.size() + 1;

  /// counts
  using index_type = typename IndexBucketsT::index_type;
  auto &counts = ibs._counts;
  if (init) {
    counts = counts.clone(mloc);
    counts.resize(numCells);
    zs::memset(mem_device, counts.data(), 0, sizeof(index_type) * numCells);
  } else {
    auto prevCounts = counts;

    counts.resize(numCells);
    zs::memset(mem_device, counts.data(), 0, sizeof(index_type) * numCells);

    zs::copy(mem_device, counts.data(), prevCounts.data(),
             sizeof(index_type) * prevCounts.size());
  }

  auto tmp = counts; // for index distribution later
  pol(range(tvs.size()),
      [tvs = proxy<space>({}, tvs),
       ibs = proxy<space>(ibs)] ZS_LAMBDA(size_t pi) mutable {
        auto pos = tvs.template pack<3>("x", pi);
        auto coord = ibs.bucketCoord(pos);
        atomic_add(wrapv<space>{},
                   (index_type *)&ibs.counts[ibs.table.query(coord)],
                   (index_type)1);
      });

  if (count_only)
    return;

  /// offsets
  auto &offsets = ibs._offsets;
  offsets = offsets.clone(mloc);
  offsets.resize(numCells);
  exclusive_scan(pol, std::begin(counts), std::end(counts),
                 std::begin(offsets));
  /// indices
  auto &indices = ibs._indices;
  indices = indices.clone(mloc);
  indices.resize(tvs.size());
  pol(range(tvs.size()),
      [tvs = proxy<space>({}, tvs), counts = proxy<space>(tmp),
       ibs = proxy<space>(ibs)] ZS_LAMBDA(size_t pi) mutable {
        auto pos = tvs.template pack<3>("x", pi);
        auto coord = ibs.bucketCoord(pos);
        auto cellno = ibs.table.query(coord);
        auto localno = atomic_add(wrapv<space>{}, (index_type *)&counts[cellno],
                                  (index_type)1);
        ibs.indices[ibs.offsets[cellno] + localno] = (index_type)pi;
      });
}

template <typename ExecPol, int side_length>
inline void identify_boundary_indices(ExecPol &pol, ZenoPartition &partition,
                                      zs::wrapv<side_length>) {
  using namespace zs;
  constexpr auto space = ExecPol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  if (!partition.hasTags())
    return;

  auto &table = partition.table;

  auto allocator = table.get_allocator();

  using Ti = typename ZenoPartition::Ti;
  using indices_t = typename ZenoPartition::indices_t;

  std::size_t numBlocks = table.size();
  indices_t marks{allocator, numBlocks + 1}, offsets{allocator, numBlocks + 1};

  pol(range(numBlocks), [table = proxy<space>(table),
                         marks = proxy<space>(marks)] ZS_LAMBDA(Ti bi) mutable {
    using table_t = RM_CVREF_T(table);
    auto bcoord = table._activeKeys[bi];
    using key_t = typename table_t::key_t;
    bool isBoundary =
        (table.query(bcoord + key_t{-side_length, 0, 0}) ==
             table_t::sentinel_v ||
         table.query(bcoord + key_t{side_length, 0, 0}) ==
             table_t::sentinel_v ||
         table.query(bcoord + key_t{0, -side_length, 0}) ==
             table_t::sentinel_v ||
         table.query(bcoord + key_t{0, side_length, 0}) ==
             table_t::sentinel_v ||
         table.query(bcoord + key_t{0, 0, -side_length}) ==
             table_t::sentinel_v ||
         table.query(bcoord + key_t{0, 0, side_length}) == table_t::sentinel_v);
    marks[bi] = isBoundary ? (Ti)1 : (Ti)0;
  });

  exclusive_scan(pol, std::begin(marks), std::end(marks), std::begin(offsets));
  auto bouCnt = offsets.getVal(numBlocks);

  auto &boundaryIndices = partition.getBoundaryIndices();
  boundaryIndices.resize(bouCnt);
  pol(range(numBlocks),
      [marks = proxy<space>(marks),
       boundaryIndices = proxy<space>(boundaryIndices),
       offsets = proxy<space>(offsets)] ZS_LAMBDA(Ti bi) mutable {
        if (marks[bi])
          boundaryIndices[offsets[bi]] = bi;
      });

  auto &tags = partition.getTags();
  tags.resize(bouCnt);
  tags.reset(0);
  return;
}

template <typename ExecPol>
inline void histogram_sort_primitives(ExecPol &pol, ZenoParticles &primitive,
                                      ZenoPartition &partition,
                                      ZenoGrid &zsgrid) {
  using namespace zs;
  constexpr auto space = ExecPol::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
  // ZS_LAMBDA -> __device__
  static_assert(space == execspace_e::cuda,
                "specialized policy and compiler not match");
#else
  static_assert(space != execspace_e::cuda,
                "specialized policy and compiler not match");
#endif
  using T = typename ZenoParticles::particles_t::value_type;
  using Ti = typename ZenoPartition::table_t::index_type;
  static_assert(std::is_signed_v<Ti>, "Ti should be a signed integer");
  // using Box = AABBBox<3, T>;
  // using TV = typename Box::TV;
  // using mc_t = u32;

  auto &grid = zsgrid.get();
  auto &table = partition.get();

  using grid_t = RM_CVREF_T(grid);
  using table_t = RM_CVREF_T(table);

  auto &pars = primitive.getParticles();
  auto allocator = pars.get_allocator();
  auto mloc = allocator.location;

  // morton codes
  const auto cnt = pars.size();
  constexpr auto blockSize = grid_t::block_size;
  constexpr auto sizeLength = grid_t::side_length;
  auto numBuckets = (Ti)table.size() * (Ti)blockSize;
  Vector<Ti> bucketCnts{(std::size_t)numBuckets, mloc.memspace(), mloc.devid()},
      bucketOffsets{(std::size_t)numBuckets, mloc.memspace(), mloc.devid()};
  bucketCnts.reset(0);
  // key: [blockno, cellno]
  pol(range(cnt), [dxinv = 1.f / grid.dx, bucketCnts = proxy<space>(bucketCnts),
                   prims = proxy<space>({}, pars), table = proxy<space>(table),
                   grid = proxy<space>(grid)] ZS_LAMBDA(Ti i) mutable {
    using grid_t = RM_CVREF_T(grid);
    auto pos = prims.template pack<3>("x", i);
    auto index = (pos * dxinv - 0.5f);
    typename table_t::key_t coord{};
    for (int d = 0; d != 3; ++d)
      coord[d] = lower_trunc(index[d]);
    auto ccoord = coord & (grid_t::side_length - 1);
    auto bcoord = coord - ccoord;
    if (auto bno = table.query(bcoord); bno != table_t::sentinel_v) {
      auto bucketNo = (Ti)bno * (Ti)grid_t::side_length +
                      (Ti)grid_t::coord_to_cellid(ccoord);
      atomic_add(wrapv<space>{}, &bucketCnts[bucketNo], (Ti)1);
    } else {
      printf("unable to sort primitives by histogram sort since no "
             "corresponding bucket exists");
    }
  });
  exclusive_scan(pol, std::begin(bucketCnts), std::end(bucketCnts),
                 std::begin(bucketOffsets));

  Vector<Ti> sortedIndices{cnt, mloc.memspace(), mloc.devid()};
  pol(range(cnt), [dxinv = 1.f / grid.dx, bucketCnts = proxy<space>(bucketCnts),
                   bucketOffsets = proxy<space>(bucketOffsets),
                   indices = proxy<space>(sortedIndices),
                   prims = proxy<space>({}, pars), table = proxy<space>(table),
                   grid = proxy<space>(grid)] ZS_LAMBDA(Ti i) mutable {
    using grid_t = RM_CVREF_T(grid);
    auto pos = prims.template pack<3>("x", i);
    auto index = (pos * dxinv - 0.5f);
    typename table_t::key_t coord{};
    for (int d = 0; d != 3; ++d)
      coord[d] = lower_trunc(index[d]);
    auto ccoord = coord & (grid_t::side_length - 1);
    auto bcoord = coord - ccoord;
    if (auto bno = table.query(bcoord); bno != table_t::sentinel_v) {
      auto bucketNo = (Ti)bno * (Ti)grid_t::side_length +
                      (Ti)grid_t::coord_to_cellid(ccoord);
      indices[bucketOffsets[bucketNo] +
              atomic_add(wrapv<space>{}, &bucketCnts[bucketNo], (Ti)-1) - 1] =
          i;
    }
  });

  {
    auto tmp = pars.clone(mloc);
    if (!pars.hasProperty("id"))
      pars.append_channels(pol, {{"id", 1}});
    pol(range(cnt),
        [prims = proxy<space>({}, pars), tmp = proxy<space>({}, tmp),
         indices = proxy<space>(sortedIndices)] ZS_LAMBDA(Ti i) mutable {
          auto o_id = indices[i];
          for (int chn = 0; chn != tmp.numChannels(); ++chn)
            prims(chn, i) = tmp(chn, o_id);
          prims("id", i) = o_id;
        });
  }

  if (primitive.isMeshPrimitive()) {
    auto &eles = primitive.getQuadraturePoints();
    const auto ecnt = eles.size();
    const int degree = primitive.numDegree();

    bucketCnts.reset(0);
    pol(range(ecnt),
        [dxinv = 1.f / grid.dx, bucketCnts = proxy<space>(bucketCnts),
         pars = proxy<space>({}, pars), eles = proxy<space>({}, eles),
         table = proxy<space>(table), grid = proxy<space>(grid),
         degree] ZS_LAMBDA(Ti ei) mutable {
          using grid_t = RM_CVREF_T(grid);
          auto pos = table_t::key_t::zeros();
          for (int d = 0; d != degree; ++d) {
            auto ind = reinterpret_bits<int>(eles("inds", d, ei));
            pos += pars.template pack<3>("x", ind);
          }
          pos /= degree;
          auto index = (pos * dxinv - 0.5f);
          typename table_t::key_t coord{};
          for (int d = 0; d != 3; ++d)
            coord[d] = lower_trunc(index[d]);
          auto ccoord = coord & (grid_t::side_length - 1);
          auto bcoord = coord - ccoord;
          if (auto bno = table.query(bcoord); bno != table_t::sentinel_v) {
            auto bucketNo = (Ti)bno * (Ti)grid_t::side_length +
                            (Ti)grid_t::coord_to_cellid(ccoord);
            atomic_add(wrapv<space>{}, &bucketCnts[bucketNo], (Ti)1);
          } else {
            printf("unable to sort primitives by histogram sort since no "
                   "corresponding bucket exists");
          }
        });
    exclusive_scan(pol, std::begin(bucketCnts), std::end(bucketCnts),
                   std::begin(bucketOffsets));

    Vector<Ti> elementIndices{ecnt, mloc.memspace(), mloc.devid()};
    pol(range(ecnt), [dxinv = 1.f / grid.dx,
                      bucketCnts = proxy<space>(bucketCnts),
                      bucketOffsets = proxy<space>(bucketOffsets),
                      indices = proxy<space>(elementIndices),
                      pars = proxy<space>({}, pars),
                      eles = proxy<space>({}, eles),
                      table = proxy<space>(table), grid = proxy<space>(grid),
                      degree] ZS_LAMBDA(Ti ei) mutable {
      using grid_t = RM_CVREF_T(grid);
      auto pos = table_t::key_t::zeros();
      for (int d = 0; d != degree; ++d) {
        auto ind = reinterpret_bits<int>(eles("inds", d, ei));
        pos += pars.template pack<3>("x", ind);
      }
      pos /= degree;
      auto index = (pos * dxinv - 0.5f);
      typename table_t::key_t coord{};
      for (int d = 0; d != 3; ++d)
        coord[d] = lower_trunc(index[d]);
      auto ccoord = coord & (grid_t::side_length - 1);
      auto bcoord = coord - ccoord;
      if (auto bno = table.query(bcoord); bno != table_t::sentinel_v) {
        auto bucketNo = (Ti)bno * (Ti)grid_t::side_length +
                        (Ti)grid_t::coord_to_cellid(ccoord);
        indices[bucketOffsets[bucketNo] +
                atomic_add(wrapv<space>{}, &bucketCnts[bucketNo], (Ti)-1) - 1] =
            ei;
      }
    });

    auto tmp = eles.clone(mloc);
    if (!eles.hasProperty("id"))
      eles.append_channels(pol, {{"id", 1}});
    pol(range(ecnt),
        [prims = proxy<space>({}, eles), tmp = proxy<space>({}, tmp),
         vertIndices = proxy<space>(sortedIndices),
         elementIndices = proxy<space>(elementIndices),
         degree] ZS_LAMBDA(Ti ei) mutable {
          auto o_eid = elementIndices[ei];
          for (int chn = 0; chn != tmp.numChannels(); ++chn)
            prims(chn, ei) = tmp(chn, o_eid);
          for (int d = 0; d != degree; ++d)
            prims("inds", d, ei) = reinterpret_bits<float>(
                vertIndices[reinterpret_bits<int>(prims("inds", d, ei))]);
          prims("id", ei) = o_eid;
        });
  }
  return;
}

} // namespace zeno