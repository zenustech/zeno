#pragma once
#include <utility>

#include "zensim/container/HashTable.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/geometry/LevelSetInterface.h"

namespace zs {

  template <int dim_ = 3> struct SparseLevelSet {
    static constexpr int dim = dim_;
    static constexpr int lane_width = 32;
    using value_type = f32;
    using index_type = i64;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    using Affine = vec<value_type, dim + 1, dim + 1>;
    using tiles_t = TileVector<value_type, lane_width>;
    using table_t = HashTable<index_type, dim, i64>;

    constexpr SparseLevelSet() = default;

    SparseLevelSet clone(const MemoryHandle mh) const {
      SparseLevelSet ret{};
      ret._sideLength = _sideLength;
      ret._space = _space;
      ret._dx = _dx;
      ret._backgroundValue = _backgroundValue;
      ret._backgroundVecValue = _backgroundVecValue;
      ret._table = _table.clone(mh);
      ret._tiles = _tiles.clone(mh);
      ret._min = _min;
      ret._max = _max;
      ret._w2v = _w2v;
      return ret;
    }

    int _sideLength{8};  // tile side length
    int _space{512};     // voxels per tile
    value_type _dx{1};
    value_type _backgroundValue{0};
    TV _backgroundVecValue{TV::zeros()};
    table_t _table{};
    tiles_t _tiles{};
    TV _min{}, _max{};
    Affine _w2v{};
  };

  template <execspace_e, typename SparseLevelSetT, typename = void> struct SparseLevelSetProxy;

  template <execspace_e Space, typename SparseLevelSetT>
  struct SparseLevelSetProxy<Space, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetProxy<Space, SparseLevelSetT>,
                          typename SparseLevelSetT::value_type, SparseLevelSetT::dim> {
    using table_t = typename SparseLevelSetT::table_t;
    using tiles_t = typename SparseLevelSetT::tiles_t;
    using T = typename SparseLevelSetT::value_type;
    using Ti = typename SparseLevelSetT::index_type;
    using TV = typename SparseLevelSetT::TV;
    using IV = typename SparseLevelSetT::IV;
    using Affine = typename SparseLevelSetT::Affine;
    static constexpr int dim = SparseLevelSetT::dim;

    template <typename Val, std::size_t... Is>
    static constexpr auto arena_type_impl(index_seq<Is...>) {
      return vec<Val, (Is + 1 > 0 ? 2 : 2)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(std::make_index_sequence<d>{});
    }

    template <typename Val> using Arena = decltype(arena_type<Val, dim>());
    constexpr auto offset(i64 bid, IV cellCoord) const noexcept {
      auto res{cellCoord[0]};
      for (int d = 1; d < dim; ++d) res = res * _sideLength + cellCoord[d];
      return bid * _space + res;
    }

    constexpr SparseLevelSetProxy() = default;
    ~SparseLevelSetProxy() = default;
    constexpr SparseLevelSetProxy(const std::vector<SmallString> &tagNames, SparseLevelSetT &ls)
        : _sideLength{ls._sideLength},
          _space{ls._space},
          _dx{ls._dx},
          _backgroundValue{ls._backgroundValue},
          _backgroundVecValue{ls._backgroundVecValue},
          _unnamed{false},
          table{proxy<Space>(ls._table)},
          tiles{proxy<Space>(tagNames, ls._tiles)},
          _min{ls._min},
          _max{ls._max},
          _w2v{ls._w2v} {}
    constexpr SparseLevelSetProxy(SparseLevelSetT &ls)
        : _sideLength{ls._sideLength},
          _space{ls._space},
          _dx{ls._dx},
          _backgroundValue{ls._backgroundValue},
          _backgroundVecValue{ls._backgroundVecValue},
          _unnamed{true},
          table{proxy<Space>(ls._table)},
          unnamedTiles{proxy<Space>(ls._tiles)},
          _min{ls._min},
          _max{ls._max},
          _w2v{ls._w2v} {}

    constexpr T getSignedDistance(const TV &x) const noexcept {
      /// world to local
      auto arena = Arena<T>::uniform(_backgroundValue);
      IV loc{};
      TV X = x * _w2v;
      // TV X = x / _dx;
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor(X(d));
      // for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = X - loc;
      if constexpr (dim == 2) {
        for (auto &&[dx, dy] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            if (_unnamed)
              arena(dx, dy) = unnamedTiles.val(
                  0, offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
            else
              arena(dx, dy) = tiles.val(
                  "sdf", offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
          }
        }
      } else if constexpr (dim == 3) {
        for (auto &&[dx, dy, dz] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy, loc(2) + dz};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            if (_unnamed)
              arena(dx, dy, dz) = unnamedTiles.val(
                  0, offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
            else
              arena(dx, dy, dz) = tiles.val(
                  "sdf", offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
          }
        }
      }
      return trilinear_interop<0>(diff, arena);
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i < dim; i++) {
        v1 = X;
        v2 = X;
        v1(i) = X(i) + eps;
        v2(i) = X(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      if (tiles.numChannels() != 1 + dim) return TV::zeros();
      /// world to local
      auto arena = Arena<TV>::uniform(_backgroundVecValue);
      IV loc{};
      TV X = x * _w2v;
      // TV X = x / _dx;
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor(X(d));
      // for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = X - loc;
      if constexpr (dim == 2) {
        for (auto &&[dx, dy] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            if (_unnamed)
              arena(dx, dy) = unnamedTiles.template pack<dim>(
                  1, offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
            else
              arena(dx, dy) = tiles.template pack<dim>(
                  "vel", offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
          }
        }
      } else if constexpr (dim == 3) {
        for (auto &&[dx, dy, dz] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy, loc(2) + dz};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            if (_unnamed)
              arena(dx, dy, dz) = unnamedTiles.template pack<dim>(
                  1, offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
            else
              arena(dx, dy, dz) = tiles.template pack<dim>(
                  "vel", offset(blockno, coord - blockid * _sideLength));  //< bid + cellid
          }
        }
      }
      return trilinear_interop<0>(diff, arena);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return std::make_tuple(_min, _max); }

  protected:
    template <std::size_t d, typename Field, enable_if_t<(d == dim - 1)> = 0>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      return linear_interop(diff(d), arena(0), arena(1));
    }
    template <std::size_t d, typename Field, enable_if_t<(d != dim - 1)> = 0>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      return linear_interop(diff(d), trilinear_interop<d + 1>(diff, arena[0]),
                            trilinear_interop<d + 1>(diff, arena[1]));
    }

    int _sideLength;  // tile side length
    int _space;       // voxels per tile
    T _dx;
    T _backgroundValue;
    TV _backgroundVecValue;
    HashTableProxy<Space, table_t> table;
    bool _unnamed{false};
    TileVectorProxy<Space, tiles_t> tiles;
    TileVectorUnnamedProxy<Space, tiles_t> unnamedTiles;
    TV _min, _max;
    Affine _w2v;
  };

  template <execspace_e ExecSpace, int dim>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 SparseLevelSet<dim> &levelset) {
    return SparseLevelSetProxy<ExecSpace, SparseLevelSet<dim>>{tagNames, levelset};
  }

  template <execspace_e ExecSpace, int dim>
  constexpr decltype(auto) proxy(SparseLevelSet<dim> &levelset) {
    return SparseLevelSetProxy<ExecSpace, SparseLevelSet<dim>>{levelset};
  }

}  // namespace zs