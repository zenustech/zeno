#pragma once
#include "zensim/container/Vector.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/resource/Resource.h"

namespace zs {

  enum struct matrix_order_e : char { rowMajor = 0, colMajor };

  /// matrix base
  template <typename ValueType = float, typename IndexType = int> struct MatrixBase : MemoryHandle {
    using value_type = ValueType;
    using index_type = IndexType;

    constexpr index_type rows() const noexcept { return nrows; }
    constexpr index_type cols() const noexcept { return ncols; }
    constexpr index_type size() const noexcept { return rows() * cols(); }
    constexpr bool isVector() const noexcept { return rows() == 1 || cols() == 1; }
    constexpr bool isRowMajor() const noexcept { return order == matrix_order_e::rowMajor; }

    constexpr index_type outerSize() const noexcept {
      return isVector() ? 1 : isRowMajor() ? rows() : cols();
    }
    constexpr index_type innerSize() const noexcept {
      return isVector() ? size() : isRowMajor() ? cols() : rows();
    }

    index_type nrows{1}, ncols{1};
    matrix_order_e order{matrix_order_e::rowMajor};
  };

  /// matrix access
  template <typename Derived> struct MatrixAccessor {
    // using value_type = typename Derived::value_type;
    // using index_type = typename Derived::index_type;
    template <typename Ti> constexpr decltype(auto) coeff(Ti r, Ti c) const {
      return self().do_coeff(r, c);
    }

  protected:
    constexpr auto &self() noexcept { return static_cast<Derived &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const Derived &>(*this); }
  };

  /// matrix

  /// matrix
  template <typename ValueType = float, typename IndexType = int> struct IdentityMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<IdentityMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    constexpr value_type do_coeff(index_type r, index_type c) const {
      return r == c ? identity : 0;
    }
    value_type identity{1};
  };
  template <typename ValueType = float, typename IndexType = int> struct DiagonalMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<IdentityMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    constexpr value_type do_coeff(index_type r, index_type c) const {
      return r == c ? diagEntries[r] : 0;
    }
    Vector<value_type> diagEntries{};
  };
  template <typename ValueType = float, typename IndexType = int> struct YaleSparseMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<YaleSparseMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    YaleSparseMatrix() = delete;
    constexpr YaleSparseMatrix(memsrc_e mre, ProcID pid,
                               matrix_order_e order = matrix_order_e::rowMajor,
                               index_type nrows = 1, index_type ncols = 1,
                               std::size_t alignment = alignof(std::max_align_t))
        : MatrixBase<ValueType, IndexType>{{mre, pid}, nrows, ncols, order},
          offsets{mre, pid, alignment},
          indices{mre, pid, alignment},
          vals{mre, pid, alignment} {}

    constexpr auto nnz() const noexcept { return vals.size(); }
    constexpr value_type do_coeff(index_type r, index_type c) const {
      index_type i = c;
      index_type j = r;
      if (base_t::isRowMajor()) {
        i = r;
        j = c;
      }
      if ((index_type)offsets.size() <= i) return 0;
      for (index_type st = offsets[i], ed = offsets[i + 1]; st < ed; ++st)
        if (indices[st] == j) return vals[st];
      return 0;
    }

    Vector<index_type> offsets{}, indices{};
    Vector<value_type> vals{};
  };
  template <typename ValueType = float, typename IndexType = int> struct CooSparseMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<CooSparseMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    constexpr auto nnz() const noexcept { return vals.size(); }
    Vector<index_type> rowInds{}, colInds{};
    Vector<value_type> vals{};
  };
  template <typename ValueType = float, typename IndexType = int> struct DenseMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<DenseMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    constexpr auto nnz() const noexcept { return base_t::size(); }
    Vector<value_type> vals{};
  };

}  // namespace zs