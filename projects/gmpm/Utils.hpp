#pragma once
#include <string_view>
#include <vector>
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/zeno.h>

#include "Structures.hpp"
#include "zensim/container/HashTable.hpp"

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
        auto x = tvs.template pack<3>("pos", pi);
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
        auto pos = tvs.template pack<3>("pos", pi);
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
        auto pos = tvs.template pack<3>("pos", pi);
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
  auto mloc = allocator.location;

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
    auto pos = prims.template pack<3>("pos", i);
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
    auto pos = prims.template pack<3>("pos", i);
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
    const int degree =
        primitive.category == ZenoParticles::curve
            ? 2
            : (ZenoParticles::surface ? 3 : (ZenoParticles::tet ? 4 : -1));

    bucketCnts.reset(0);
    pol(range(ecnt),
        [dxinv = 1.f / grid.dx, bucketCnts = proxy<space>(bucketCnts),
         pars = proxy<space>({}, pars), eles = proxy<space>({}, eles),
         table = proxy<space>(table), grid = proxy<space>(grid),
         degree] ZS_LAMBDA(Ti ei) mutable {
          using grid_t = RM_CVREF_T(grid);
          auto pos = table_t::key_t::zeros();
          for (int d = 0; d != degree; ++d) {
            auto ind = (Ti)eles("inds", d, ei);
            pos += pars.template pack<3>("pos", ind);
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
        auto ind = (Ti)eles("inds", d, ei);
        pos += pars.template pack<3>("pos", ind);
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
            prims("inds", d, ei) = vertIndices[(Ti)prims("inds", d, ei)];
          prims("id", ei) = o_eid;
        });
  }
  return;
}

} // namespace zeno