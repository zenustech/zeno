#pragma once
#include <string_view>
#include <vector>
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/zeno.h>

#include "Structures.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/BoundingVolumeInterface.hpp"

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

template <typename Pol, int codim = 3>
zs::Vector<typename ZenoParticles::lbvh_t::Box>
retrieve_bounding_volumes(Pol &pol,
                          const typename ZenoParticles::particles_t &verts,
                          const typename ZenoParticles::particles_t &eles,
                          zs::wrapv<codim> = {}, float thickness = 0.f) {
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
                               codim_v = wrapv<codim>{},
                               thickness] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = verts.pack<3>("x", inds[0]);
    bv_t bv{x0};
    for (int d = 1; d != dim; ++d)
      merge(bv, verts.pack<3>("x", inds[d]));
    bv._min -= thickness / 2;
    bv._max += thickness / 2;
    bvs[ei] = bv;
  });
  return ret;
}

// for ccd
template <typename Pol, int codim = 3>
zs::Vector<typename ZenoParticles::lbvh_t::Box> retrieve_bounding_volumes(
    Pol &pol, const typename ZenoParticles::particles_t &verts,
    const typename ZenoParticles::particles_t &eles,
    const typename ZenoParticles::particles_t &searchDir, zs::wrapv<codim> = {},
    float stepSize = 1.f, float thickness = 0.f,
    const zs::SmallString &dirTag = "dir") {
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
                               searchDir = proxy<space>({}, searchDir),
                               codim_v = wrapv<codim>{}, dirTag, stepSize,
                               thickness] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = verts.pack<3>("x", inds[0]);
    auto dir0 = searchDir.pack<3>(dirTag, inds[0]);
    auto [mi, ma] = get_bounding_box(x0, x0 + stepSize * dir0);
    bv_t bv{mi, ma};
    for (int d = 1; d != dim; ++d) {
      auto x = verts.pack<3>("x", inds[d]);
      auto dir = searchDir.pack<3>(dirTag, inds[d]);
      auto [mi, ma] = get_bounding_box(x, x + stepSize * dir);
      merge(bv, mi);
      merge(bv, ma);
    }
    bv._min -= thickness / 2;
    bv._max += thickness / 2;
    bvs[ei] = bv;
  });
  return ret;
}

// for ccd
template <typename Pol, typename T>
void find_intersection_free_stepsize(
    Pol &pol, ZenoParticles &zstets,
    const typename ZenoParticles::particles_t &vtemp, T &stepSize, T xi) {
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
  if (!zstets.hasBvh(ZenoParticles::s_surfTriTag)) // build if bvh not exist
    zstets.bvh(ZenoParticles::s_surfTriTag)
        .build(pol,
               retrieve_bounding_volumes(pol, verts, surfaces, wrapv<3>{}, xi));
  const auto &stBvh = zstets.bvh(ZenoParticles::s_surfTriTag);

  const auto &surfEdges = zstets[ZenoParticles::s_surfEdgeTag];
  if (!zstets.hasBvh(ZenoParticles::s_surfEdgeTag))
    zstets.bvh(ZenoParticles::s_surfEdgeTag)
        .build(pol, retrieve_bounding_volumes(pol, verts, surfEdges, wrapv<2>{},
                                              xi));
  const auto &seBvh = zstets.bvh(ZenoParticles::s_surfEdgeTag);

  const auto &surfVerts = zstets[ZenoParticles::s_surfVertTag];
#if 0
  if (!zstets.hasBvh(ZenoParticles::s_surfVertTag))
    zstets.bvh(ZenoParticles::s_surfVertTag)
        .build(pol, retrieve_bounding_volumes(pol, verts, surfVerts, wrapv<1>{},
                                              xi));
  const auto &svBvh = zstets.bvh(ZenoParticles::s_surfVertTag);
#endif
  // query pt
  pol(Collapse{surfVerts.size()}, [svs = proxy<space>({}, surfVerts),
                                   sts = proxy<space>({}, surfaces),
                                   verts = proxy<space>({}, verts),
                                   vtemp = proxy<space>({}, vtemp),
                                   bvh = proxy<space>(stBvh),
                                   thickness = xi] ZS_LAMBDA(int svi) mutable {
    auto vi = reinterpret_bits<int>(svs("inds", svi));
    auto p = vtemp.pack<3>("xn", vi);
    auto [mi, ma] = get_bounding_box(p - thickness / 2, p + thickness / 2);
    auto bv = bv_t{mi, ma};
    bvh.iter_neighbors(bv, [&](int stI) {
      auto tri =
          sts.template pack<3>("inds", stI).template reinterpret_bits<int>();
      if (vi == tri[0] || vi == tri[1] || vi == tri[2])
        return;
      // all affected by sticky boundary conditions
      if (reinterpret_bits<int>(verts("BCorder", vi)) == 3 &&
          reinterpret_bits<int>(verts("BCorder", tri[0])) == 3 &&
          reinterpret_bits<int>(verts("BCorder", tri[1])) == 3 &&
          reinterpret_bits<int>(verts("BCorder", tri[2])) == 3)
        return;
      // ccd
    });
  });
  // query ee
  pol(Collapse{surfEdges.size()}, [ses = proxy<space>({}, surfEdges),
                                   verts = proxy<space>({}, verts),
                                   vtemp = proxy<space>({}, vtemp),
                                   bvh = proxy<space>(seBvh),
                                   thickness = xi] ZS_LAMBDA(int sei) mutable {
    auto edgeInds =
        ses.template pack<2>("inds", sei).template reinterpret_bits<int>();
    auto x0 = vtemp.pack<3>("xn", edgeInds[0]);
    auto x1 = vtemp.pack<3>("xn", edgeInds[1]);
    auto [mi, ma] = get_bounding_box(x0, x1);
    auto bv = bv_t{mi - thickness / 2, ma + thickness / 2};
    bvh.iter_neighbors(bv, [&](int seI) {
      if (sei > seI)
        return;
      auto oEdgeInds =
          ses.template pack<2>("inds", seI).template reinterpret_bits<int>();
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
    });
  });
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