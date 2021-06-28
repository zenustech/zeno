#pragma once
#include "../transfer/Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/Structure.hpp"

namespace zs {

  template <typename GridBlocksT> struct CleanGridBlocks;
  template <typename TableT, typename GridBlocksT> struct PrintGridBlocks;
  template <transfer_scheme_e, typename GridBlocksT> struct ComputeGridBlockVelocity;
  template <typename ColliderT, typename TableT, typename GridBlocksT>
  struct ApplyBoundaryConditionOnGridBlocks;

  template <execspace_e space, typename GridBlocksT> CleanGridBlocks(wrapv<space>, GridBlocksT)
      -> CleanGridBlocks<GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, typename TableT, typename GridBlocksT>
  PrintGridBlocks(wrapv<space>, TableT, GridBlocksT)
      -> PrintGridBlocks<HashTableProxy<space, TableT>, GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, transfer_scheme_e scheme, typename GridBlocksT>
  ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridBlocksT, float dt, float gravity,
                           float *maxVel)
      -> ComputeGridBlockVelocity<scheme, GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, typename LevelsetT, typename TableT, typename GridBlocksT>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Collider<LevelsetT>, TableT, GridBlocksT)
      -> ApplyBoundaryConditionOnGridBlocks<Collider<LevelsetT>, HashTableProxy<space, TableT>,
                                            GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, typename TableT, typename GridBlocksT>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, LevelSetBoundary<SparseLevelSet<3>>, TableT,
                                     GridBlocksT)
      -> ApplyBoundaryConditionOnGridBlocks<Collider<SparseLevelSetProxy<space, SparseLevelSet<3>>>,
                                            HashTableProxy<space, TableT>,
                                            GridBlocksProxy<space, GridBlocksT>>;

  template <execspace_e space, typename GridBlocksT>
  struct CleanGridBlocks<GridBlocksProxy<space, GridBlocksT>> {
    static constexpr auto exectag = wrapv<space>{};
    using gridblocks_t = GridBlocksProxy<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit CleanGridBlocks(wrapv<space>, GridBlocksT &gridblocks)
        : gridblocks{proxy<space>(gridblocks)} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      block(0, cellid).asFloat() = static_cast<VT>(0);
      for (int d = 0; d < gridblocks_t::dim; ++d)
        block(d + 1, cellid).asFloat() = static_cast<VT>(0);
    }

    gridblocks_t gridblocks;
  };

  template <execspace_e space, typename TableT, typename GridBlocksT>
  struct PrintGridBlocks<HashTableProxy<space, TableT>, GridBlocksProxy<space, GridBlocksT>> {
    using partition_t = HashTableProxy<space, TableT>;
    using gridblocks_t = GridBlocksProxy<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit PrintGridBlocks(wrapv<space>, TableT &table, GridBlocksT &gridblocks)
        : partition{proxy<space>(table)}, gridblocks{proxy<space>(gridblocks)} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto blockkey = partition._activeKeys[blockid];
      auto checkedid = partition.query(blockkey);
      if (checkedid != blockid && cellid == 0)
        printf("%d-th block(%d, %d, %d) table index: %d\n", blockid, blockkey[0], blockkey[1],
               blockkey[2], checkedid);
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      if (blockid == 0 && cellid == 0) printf("block space: %d\n", gridblock_t::space());
      if (blockid == 0)  // || (float)block(1, cellid).asFloat() > 0)
        printf("(%d, %d): mass %e, (%e, %e, %e)\n", (int)blockid, (int)cellid,
               block(0, cellid).asFloat(), block(1, cellid).asFloat(), block(2, cellid).asFloat(),
               block(3, cellid).asFloat());
    }

    partition_t partition;
    gridblocks_t gridblocks;
  };

  template <execspace_e space, transfer_scheme_e scheme, typename GridBlocksT>
  struct ComputeGridBlockVelocity<scheme, GridBlocksProxy<space, GridBlocksT>> {
    static constexpr auto exectag = wrapv<space>{};
    using gridblocks_t = GridBlocksProxy<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    constexpr ComputeGridBlockVelocity() = default;
    ~ComputeGridBlockVelocity() = default;

    explicit ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridBlocksT &gridblocks,
                                      float dt, float gravity, float *maxVel)
        : gridblocks{proxy<space>(gridblocks)}, dt{dt}, gravity{gravity}, maxVel{maxVel} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      VT mass = block(0, cellid).asFloat();
      if (mass != (VT)0) {
        mass = (VT)1 / mass;
        vec<VT, gridblocks_t::dim> vel{};
        for (int d = 0; d < gridblocks_t::dim; ++d) {
          vel[d] = block(d + 1, cellid).asFloat() * mass;
        }
        vel[1] += gravity * dt;

        /// write back
        for (int d = 0; d < gridblocks_t::dim; ++d) block(d + 1, cellid).asFloat() = vel[d];

        /// cfl dt
        float ret{0.f};
        for (int d = 0; d < gridblocks_t::dim; ++d) ret += vel[d] * vel[d];
        atomic_max(exectag, maxVel, ret);
      }
    }

    gridblocks_t gridblocks;
    float dt;
    float gravity;
    float *maxVel;
  };

  template <execspace_e space, typename ColliderT, typename TableT, typename GridBlocksT>
  struct ApplyBoundaryConditionOnGridBlocks<ColliderT, HashTableProxy<space, TableT>,
                                            GridBlocksProxy<space, GridBlocksT>> {
    using collider_t = ColliderT;
    using partition_t = HashTableProxy<space, TableT>;
    using gridblocks_t = GridBlocksProxy<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    template <typename Boundary = ColliderT,
              enable_if_t<!is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &collider, TableT &table,
                                       GridBlocksT &gridblocks)
        : collider{collider},
          partition{proxy<space>(table)},
          gridblocks{proxy<space>(gridblocks)} {}
    template <typename Boundary, enable_if_t<is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &boundary, TableT &table,
                                       GridBlocksT &gridblocks)
        : collider{Collider{proxy<space>(boundary.levelset), boundary.type/*, boundary.s,
                            boundary.dsdt, boundary.R, boundary.omega, boundary.b, boundary.dbdt*/}},
          partition{proxy<space>(table)},
          gridblocks{proxy<space>(gridblocks)} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto blockkey = partition._activeKeys[blockid];
      auto &block = gridblocks[blockid];
      using VT = typename collider_t::T;
      VT dx = static_cast<VT>(gridblocks._dx.asFloat());

      if (block(0, cellid).asFloat() > 0) {
        vec<VT, gridblocks_t::dim> vel{},
            pos = (blockkey * gridblock_t::side_length() + gridblock_t::to_coord(cellid)) * dx;
        for (int d = 0; d < gridblocks_t::dim; ++d)
          vel[d] = static_cast<VT>(block(d + 1, cellid).asFloat());

        collider.resolveCollision(pos, vel);

        for (int d = 0; d < gridblocks_t::dim; ++d) block(d + 1, cellid).asFloat() = vel[d];
      }
    }

    collider_t collider;
    partition_t partition;
    gridblocks_t gridblocks;
  };

}  // namespace zs