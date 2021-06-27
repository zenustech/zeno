#pragma once
#include "zensim/TypeAlias.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/tpls/gcem/gcem_incl/pow.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <typename V = f32, typename I = i32, int d = 3> struct Mesh {
    using ElemT = I[d + 1];
    using TV = V[d];
    using TM = V[d][d];
    using TMAffine = V[d + 1][d + 1];
    static constexpr int dim = d;
    Vector<V> M;
    Vector<TV> X;
    Vector<ElemT> Elems;
    Vector<TM> F;
  };

  using GeneralMesh
      = variant<Mesh<f32, i32, 2>, Mesh<f32, i64, 2>, Mesh<f32, i32, 3>, Mesh<f32, i64, 3>,
                Mesh<f64, i32, 2>, Mesh<f64, i64, 2>, Mesh<f64, i32, 3>, Mesh<f64, i64, 3>>;

  /// sizeof(float) = 4
  /// bin_size = 64
  /// attrib_size = 16
  template <typename V = dat32, int dim_ = 3, int channel_bits = 4, int domain_bits = 2>
  struct GridBlock {
    using value_type = V;
    using size_type = int;
    static constexpr int dim = dim_;
    using IV = vec<size_type, dim>;
    static constexpr int num_chns() noexcept { return 1 << channel_bits; }
    static constexpr int side_length() noexcept { return 1 << domain_bits; }
    static constexpr int space() noexcept { return 1 << domain_bits * dim; }

    constexpr auto &operator()(int c, IV loc) noexcept { return _data[c][offset(loc)]; }
    constexpr auto operator()(int c, IV loc) const noexcept { return _data[c][offset(loc)]; }
    constexpr auto &operator()(int c, size_type cellid) noexcept { return _data[c][cellid]; }
    constexpr auto operator()(int c, size_type cellid) const noexcept { return _data[c][cellid]; }
    static constexpr IV to_coord(size_type cellid) {
      IV ret{IV::zeros()};
      for (int d = dim - 1; d >= 0 && cellid > 0; --d, cellid >>= domain_bits)
        ret[d] = cellid % side_length();
      return ret;
    }

  protected:
    constexpr int offset(const IV &loc) const noexcept {
      // using Seq = typename gen_seq<d>::template uniform_values_t<vseq_t, (1 << domain_bits)>;
      size_type ret{0};
      for (int d = 0; d < dim; ++d) ret = (ret << domain_bits) + loc[d];
      return ret;
    }
    V _data[num_chns()][space()];
  };

  template <typename Block> struct GridBlocks;
  template <typename V, int d, int chn_bits, int domain_bits>
  struct GridBlocks<GridBlock<V, d, chn_bits, domain_bits>> {
    using value_type = V;
    using block_t = GridBlock<V, d, chn_bits, domain_bits>;
    static constexpr int dim = block_t::dim;
    using IV = typename block_t::IV;
    using size_type = typename Vector<block_t>::size_type;

    constexpr GridBlocks(float dx = 1.f, std::size_t numBlocks = 0, memsrc_e mre = memsrc_e::host,
                         ProcID devid = -1, std::size_t alignment = 0)
        : blocks{numBlocks, mre, devid, alignment}, dx{dx} {}

    void resize(size_type newSize) { blocks.resize(newSize); }
    Vector<block_t> blocks;
    V dx;
  };

#if 0
  using GeneralGridBlocks
      = variant<GridBlocks<GridBlock<dat32, 2, 3, 2>>, GridBlocks<GridBlock<dat32, 3, 3, 2>>,
                GridBlocks<GridBlock<dat64, 2, 3, 2>>, GridBlocks<GridBlock<dat64, 3, 3, 2>>,
                GridBlocks<GridBlock<dat32, 2, 4, 2>>, GridBlocks<GridBlock<dat32, 3, 4, 2>>,
                GridBlocks<GridBlock<dat64, 2, 4, 2>>, GridBlocks<GridBlock<dat64, 3, 4, 2>>>;
#else
  using GeneralGridBlocks = variant<GridBlocks<GridBlock<dat32, 3, 2, 2>>>;
#endif

  template <execspace_e, typename GridBlocksT, typename = void> struct GridBlocksProxy;
  template <execspace_e space, typename GridBlocksT> struct GridBlocksProxy<space, GridBlocksT> {
    using value_type = typename GridBlocksT::value_type;
    using block_t = typename GridBlocksT::block_t;
    static constexpr int dim = block_t::dim;
    using IV = typename block_t::IV;
    using size_type = typename GridBlocksT::size_type;

    constexpr GridBlocksProxy() = default;
    ~GridBlocksProxy() = default;
    explicit constexpr GridBlocksProxy(GridBlocksT &gridblocks)
        : _gridBlocks{gridblocks.blocks.data()},
          _blockCount{gridblocks.blocks.size()},
          _dx{gridblocks.dx} {}

    constexpr block_t &operator[](size_type i) { return _gridBlocks[i]; }
    constexpr const block_t &operator[](size_type i) const { return _gridBlocks[i]; }

    block_t *_gridBlocks;
    size_type _blockCount;
    value_type _dx;
  };

  template <execspace_e ExecSpace, typename V, int d, int chn_bits, int domain_bits>
  constexpr decltype(auto) proxy(GridBlocks<GridBlock<V, d, chn_bits, domain_bits>> &blocks) {
    return GridBlocksProxy<ExecSpace, GridBlocks<GridBlock<V, d, chn_bits, domain_bits>>>{blocks};
  }

  ///
  template <typename I = i32, int d = 3> struct Nodes {
    using IV = I[d];
    Vector<IV> nodes;
    f32 dx;
  };

  using GeneralNodes = variant<Nodes<i32, 2>, Nodes<i64, 2>, Nodes<i32, 3>, Nodes<i64, 3>>;

}  // namespace zs