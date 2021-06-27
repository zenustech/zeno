#pragma once
#include "zensim/cuda/Cuda.h"
#include "zensim/math/Vec.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  /// chenyao lou
  /// https://preshing.com/20130605/the-worlds-simplest-lock-free-hash-table/
  ///

  /// 0: key 1: value 2: value list
  template <typename Tn, typename Key, typename Index> using matrix_hash_snode
      = ds::snode_t<ds::decorations<ds::soa>, ds::uniform_domain<0, Tn, 2, index_seq<0, 1>>,
                    tuple<wrapt<Key>, wrapt<Index>>, vseq_t<1, 1>>;

  template <typename Tn, typename Key, typename Index> using matrix_hash_instance
      = ds::instance_t<ds::dense, matrix_hash_snode<Tn, Key, Index>>;

  template <typename Tn_> struct MatrixHash : matrix_hash_instance<Tn_, Tn_, Tn_> {
    using base_t = matrix_hash_instance<Tn_, Tn_, Tn_>;
    using Tn = Tn_;

    static constexpr Tn sentinel_v = -1;

  protected:
    template <typename Allocator>
    constexpr auto buildInstance(Allocator allocator, Tn rowCnt, Tn neighborCnt) {
      using namespace ds;
      uniform_domain<0, Tn, 2, index_seq<0, 1>> dom{wrapv<0>{}, rowCnt, neighborCnt};
      matrix_hash_snode<Tn, Tn, Tn> node{decorations<soa>{}, dom,
                                         zs::make_tuple(wrapt<Tn>{}, wrapt<Tn>{}), vseq_t<1, 1>{}};
      auto inst = instance{wrapv<dense>{}, tuple{node}};
      inst.alloc(allocator, Cuda::alignment());
      return inst;
    }

  public:
    MatrixHash() : _tableSize{-1}, _rowCnt{-1}, _neighborCnt{-1}, _cnts{nullptr} {}
    template <typename Allocator> MatrixHash(Allocator allocator, Tn rowCnt, Tn neighborCnt)
        : base_t{buildInstance(allocator, rowCnt, neighborCnt)},
          _tableSize{rowCnt * neighborCnt},
          _rowCnt{rowCnt},
          _neighborCnt{neighborCnt},
          _cnts{(Tn*)allocator.allocate(sizeof(Tn) * rowCnt)} {}
    ~MatrixHash() {}

    void reset(void* stream) {
      resetTable(stream);
      cudaMemsetAsync(_cnts, 0, sizeof(Tn) * _rowCnt, stream);
    }
    void resetTable(void* stream) {
      cudaMemsetAsync(this->getHandles().template get<0>(), 0xff, this->node().size(),
                                 stream);
    }

    __forceinline__ __device__ Tn insert(const vec<Tn, 2>& key) {
      using namespace placeholders;
      const Tn hkey = key(1);
      Tn hashedentry = hkey % _neighborCnt;
      for (;;) {
        Tn storedKey{atomicCAS(&(*this)(_0, key(0), hashedentry), sentinel_v, hkey)};
        if ((*this)(_0, key(0), hashedentry) == hkey) {  ///< found hash entry
          if (storedKey == sentinel_v) {
            auto localno = atomicAdd(_cnts + key(0), 1);
            (*this)(_1, key(0), hashedentry) = localno;
            return localno;
          }
          break;
        }
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _neighborCnt) hashedentry = hashedentry % _neighborCnt;
      }
      return -1;
    }
    /// make sure no one else is inserting in the same time!
    constexpr Tn query(const vec<Tn, 2>& key) {
      using namespace placeholders;
      const Tn hkey = key(1);
      Tn hashedentry = hkey % _neighborCnt;
      Tn okey;
      while ((okey = (*this)(_0, key(0), hashedentry)) != hkey) {
        /// definitely not inserted
        if (okey == sentinel_v) return -1;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _neighborCnt) hashedentry = hashedentry % _neighborCnt;
      }
      return (*this)(_1, key(0), hashedentry);
    }

    Tn _tableSize, _rowCnt, _neighborCnt;
    Tn* _cnts;
  };

}  // namespace zs
