#pragma once
#include <utility>

#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <int dim_ = 3> struct IndexBuckets {
    static constexpr int dim = dim_;
    using value_type = f32;
    using index_type = i64;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    using table_t = HashTable<int, dim, index_type>;
    using vector_t = Vector<index_type>;

    constexpr IndexBuckets() = default;

    IndexBuckets clone(const MemoryHandle mh) const {
      IndexBuckets ret{};
      ret._table = _table.clone(mh);
      ret._indices = _indices.clone(mh);
      ret._offsets = _offsets.clone(mh);
      ret._counts = _counts.clone(mh);
      ret._dx = _dx;
      return ret;
    }

    constexpr auto numEntries() const noexcept { return _indices.size(); }
    constexpr auto numBuckets() const noexcept { return _counts.size() - 1; }

    table_t _table{};
    vector_t _indices{}, _offsets{}, _counts{};
    value_type _dx{1};
  };

  using GeneralIndexBuckets = variant<IndexBuckets<3>>;

  template <execspace_e Space, typename IndexBucketsT, typename = void> struct IndexBucketsProxy {
    using value_type = typename IndexBucketsT::value_type;
    using index_type = typename IndexBucketsT::index_type;
    using TV = typename IndexBucketsT::TV;
    using IV = typename IndexBucketsT::IV;
    using table_t = typename IndexBucketsT::table_t;
    using vector_t = typename IndexBucketsT::vector_t;

    constexpr IndexBucketsProxy() = default;
    ~IndexBucketsProxy() = default;
    constexpr IndexBucketsProxy(IndexBucketsT &ibs)
        : table{proxy<Space>(ibs._table)},
          indices{proxy<Space>(ibs._indices)},
          offsets{proxy<Space>(ibs._offsets)},
          counts{proxy<Space>(ibs._counts)},
          dx{ibs._dx} {}

    constexpr auto coord(const index_type bucketno) const noexcept {
      return table._activeKeys[bucketno];
    }

    HashTableProxy<Space, table_t> table;
    VectorProxy<Space, vector_t> indices, offsets, counts;
    value_type dx;
  };

  template <execspace_e ExecSpace, int dim>
  constexpr decltype(auto) proxy(IndexBuckets<dim> &indexBuckets) {
    return IndexBucketsProxy<ExecSpace, IndexBuckets<dim>>{indexBuckets};
  }

}  // namespace zs