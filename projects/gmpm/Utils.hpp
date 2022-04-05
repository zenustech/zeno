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

} // namespace zeno