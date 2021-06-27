#pragma once
#include "zensim/math/Vec.h"
#include "zensim/tpls/magic_enum/magic_enum.hpp"
#include "zensim/types/Iterator.h"

namespace zs {

  enum struct kernel_e { linear = 2, quadratic = 3, cubic = 4 };
  constexpr wrapv<kernel_e::linear> kernel_linear;
  constexpr wrapv<kernel_e::quadratic> kernel_quad;
  constexpr wrapv<kernel_e::cubic> kernel_cubic;

  template <typename Tn, int dim, typename Ti, typename Table>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength,
                                      const Table &table) {
    using IV = vec<Tn, dim>;
    IV blockCoord = coord;
    for (int d = 0; d < dim; ++d) blockCoord[d] += (coord[d] < 0 ? -sideLength + 1 : 0);
    blockCoord = blockCoord / sideLength;
    return std::make_tuple(table.query(blockCoord), coord - blockCoord * sideLength);
  }
  template <typename Tn, int dim, typename Ti>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength) {
    using IV = vec<Tn, dim>;
    IV blockCoord = coord;
    for (int d = 0; d < dim; ++d) blockCoord[d] += (coord[d] < 0 ? -sideLength + 1 : 0);
    blockCoord = blockCoord / sideLength;
    return std::make_tuple(blockCoord, coord - blockCoord * sideLength);
  }
  template <typename Tn, int dim, typename Ti, typename Table, typename Grid>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength, const Table &table,
                                      Grid &&grid) {
    using IV = vec<Tn, dim>;
    IV blockCoord = coord;
    for (int d = 0; d < dim; ++d) blockCoord[d] += (coord[d] < 0 ? -sideLength + 1 : 0);
    blockCoord = blockCoord / sideLength;
    return std::forward_as_tuple(grid[table.query(blockCoord)], coord - blockCoord * sideLength);
  }

  template <int dim_, kernel_e kt = kernel_e::quadratic, typename T = f32, typename Ti = int>
  struct LocalArena {
    using value_type = T;
    using index_type = Ti;
    static constexpr int dim = dim_;
    static constexpr index_type width = magic_enum::enum_integer(kt);
    using TV = vec<value_type, dim>;
    using TM = vec<value_type, dim, width>;
    using IV = vec<index_type, dim>;

    constexpr void init(const value_type dx_, const TV &pos) {
      dx = dx_;
      const auto dxInv = (T)1 / dx;
      for (int d = 0; d < dim; ++d) corner[d] = lower_trunc(pos[d] * dxInv + (T)0.5) - 1;
      localPos = pos - this->corner * dx;
      if constexpr (kt == kernel_e::quadratic)
        weights = bspline_weight(pos - corner * dx, (T)1 / dx);
    }

    constexpr auto range() const noexcept { return ndrange<dim>(width); }

  protected:
    template <typename... Tn, std::size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr auto weight_impl(const std::tuple<Tn...> &loc, index_seq<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= weights(Is, std::get<Is>(loc))), ...);
      return ret;
    }

  public:
    template <typename... Tn> constexpr IV offset(const std::tuple<Tn...> &loc) const noexcept {
      return make_vec<index_type>(loc);
    }

    template <typename... Tn,
              enable_if_all<(!is_std_tuple<Tn>() && ... && (sizeof...(Tn) == dim))> = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(std::forward_as_tuple(FWD(is)...));
    }

    template <typename... Tn> constexpr T weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, std::index_sequence_for<Tn...>{});
    }
    template <typename... Tn> constexpr TV diff(const std::tuple<Tn...> &pos) const noexcept {
      return offset(pos) * dx - localPos;
    }
    template <typename... Tn> constexpr IV coord(const std::tuple<Tn...> &pos) const noexcept {
      return offset(pos) + corner;
    }

    TV localPos{TV::zeros()};
    TM weights{TM::zeros()};
    IV corner{IV::zeros()};
    value_type dx{0};
  };

  template <kernel_e kt = kernel_e::quadratic, typename Ti = int, int dim = 3, typename T = f32,
            typename TT = T>
  constexpr LocalArena<dim, kt, T, Ti> make_local_arena(TT dx, const vec<T, dim> &pos) {
    LocalArena<dim, kt, T, Ti> ret{};
    ret.init(dx, pos);
    return ret;
  }

}  // namespace zs