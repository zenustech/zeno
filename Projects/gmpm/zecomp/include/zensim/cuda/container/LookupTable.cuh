#pragma once
#include <mutex>

#include "zensim/cuda/Cuda.h"
#include "zensim/math/Vec.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  //
  // https://github.com/sparsehash/sparsehash/blob/1dffea3d917445d70d33d0c7492919fc4408fe5c/src/sparsehash/internal/sparsehashtable.h#L838
  //

  template <typename Index, int dim, auto Length> using lut_block_snode
      = ds::snode_t<ds::decorations<ds::soa, ds::sum_pow2_align>,
                    typename gen_seq<dim>::template uniform_values_t<ds::static_domain, Length>,
                    tuple<Index>, vseq_t<1>>;
  template <typename Index, typename Tn, int dim, auto SideLength> using lut_snode
      = ds::snode_t<ds::static_decorator<>,
                    ds::uniform_domain<0, Tn, dim, typename gen_seq<dim>::ascend>,
                    tuple<lut_block_snode<Index, dim, SideLength>>, vseq_t<1>>;
  template <typename Index, typename Tn, int dim, auto SideLength = 8> using lut_instance
      = ds::instance_t<ds::dense, lut_snode<Index, Tn, dim, SideLength>>;

  template <typename Value, typename Index, auto BlockLength, typename Indices> struct LutTableImpl;

  template <typename Ti = int, typename Index = int, int dim = 3, auto SideLength = 8>
  using LutTable = LutTableImpl<Ti, Index, SideLength, typename gen_seq<dim>::ascend>;

  template <typename Table> __global__ void reset_lookup_table(std::size_t count, Table tab) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    tab.entry(tab._activeKeys[idx]) = tab.sentinel_v;
  }

  template <typename Ti_, typename Tn_, auto SideLength, std::size_t... Is>
  struct LutTableImpl<Ti_, Tn_, SideLength, std::index_sequence<Is...>>
      : lut_instance<Ti_, Tn_, sizeof...(Is), SideLength> {
    using base_t = lut_instance<Ti_, Tn_, sizeof...(Is), SideLength>;
    using value_t = Ti_;
    using Tn = Tn_;
    static constexpr int dim = sizeof...(Is);
    using key_t = vec<Tn, dim>;

    // using memset_func = void *(void *, int, std::size_t);

    static constexpr value_t sentinel_v = (value_t)0;

    constexpr auto &entry(key_t const &key) noexcept {
      return (*this)((key[Is] / SideLength)...)((key[Is] % SideLength)...);
    }
    constexpr auto const &entry(key_t const &key) const noexcept {
      return (*this)((key[Is] / SideLength)...)((key[Is] % SideLength)...);
    }

    template <typename Allocator> LutTableImpl(Allocator allocator, key_t ndim)
        : base_t{buildInstance(allocator, ndim)},
          _cnt{(value_t *)allocator.allocate(sizeof(value_t))},
          _activeKeys{(key_t *)allocator.allocate(sizeof(key_t) * (ndim[Is] * ...))} {
      *_cnt = 0;
    }
    ~LutTableImpl() {}
    template <typename Device> void reset(Device &dev, void *stream) {
      resetTable(dev, stream);
      cudaMemsetAsync(_cnt, 0, sizeof(value_t), stream);
    }
    template <typename Device> void resetTable(Device &dev, void *stream) {
      dev.launch(stream, {((int)(*_cnt) + 127) / 128, 128}, reset_lookup_table<LutTableImpl>,
                 (std::size_t)(*_cnt), *this);
    }
    void resetH() {
      for (value_t i = 0; i < *_cnt; ++i) entry(_activeKeys[i]) = sentinel_v;
      *_cnt = 0;
    }
    __forceinline__ __device__ value_t insert(key_t const &key) noexcept {
      value_t &e = entry(key);
      value_t tag = atomicCAS(&e, sentinel_v, (value_t)-1);
      if (tag == sentinel_v) {
        value_t idx = atomicAdd(_cnt, 1);
        e = idx + 1;
        _activeKeys[idx] = key;  ///< created a record
        return idx;
      }
      return (value_t)-1;
    }
    __forceinline__ __host__ value_t insertH(key_t const &key) noexcept {
      static std::mutex mut{};
      value_t &e = entry(key);
      {
        std::scoped_lock lock{mut};
        if (e == sentinel_v) {
          value_t idx = (*_cnt)++;
          e = idx + 1;
          _activeKeys[idx] = key;  ///< created a record
          return idx;
        }
        return (value_t)-1;
      }
    }
    constexpr value_t query(key_t key) const noexcept { return entry(key) - 1; }

    value_t *_cnt;
    key_t *_activeKeys;

  protected:
    template <typename Allocator> constexpr auto buildInstance(Allocator allocator, key_t extent) {
      using namespace ds;
      uniform_domain<0, Tn, dim, typename gen_seq<dim>::ascend> dom{
          wrapv<0>{}, (Tn)((extent(Is) + SideLength - 1) / SideLength)...};
      auto node = snode{static_decorator<>{}, dom,
                        zs::make_tuple(lut_block_snode<value_t, dim, SideLength>{}), vseq_t<1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      inst.alloc(allocator, alignof(value_t));
      return inst;
    }

    // protected:
    //  void *(*mset)(void *, int, std::size_t);
  };

}  // namespace zs
