#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>

#include "zensim/math/MathUtils.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Relationship.h"
#include "zensim/meta/Sequence.h"
#include "zensim/tpls/gcem/gcem.hpp"
#include "zensim/types/Tuple.h"

namespace zs {

  /// declarations
  template <typename> struct indexer_impl;
  template <typename T, typename Extents> struct vec_view;
  template <typename T, typename Extents> struct vec_impl;
#if 0
template <typename T, auto... Ns>
using vec =
    vec_impl<T,
             std::integer_sequence<std::common_type_t<decltype(Ns)...>, Ns...>>;
#else
  template <typename T, auto... Ns> using vec = vec_impl<T, std::integer_sequence<int, Ns...>>;
#endif

  template <typename T> struct is_vec : std::false_type {};
  template <typename T, auto... Ns> struct is_vec<vec<T, Ns...>> : std::true_type {};

  /// indexer
  template <typename Tn, Tn... Ns> struct indexer_impl<std::integer_sequence<Tn, Ns...>> {
    static constexpr auto dim = sizeof...(Ns);
    static constexpr auto extent = (Ns * ...);
    using index_type = Tn;
    using extents = std::integer_sequence<Tn, Ns...>;
    template <place_id I> static constexpr Tn range(std::integral_constant<place_id, I>) noexcept {
      return select_indexed_value<I, Ns...>::value;  // select_indexed_value<I,
                                                     // Tn, Ns...>::value;
    }
    template <std::size_t I> static constexpr Tn range() noexcept {
      return select_indexed_value<I, Ns...>::value;
    }
    template <std::size_t... Is>
    static constexpr Tn identity_offset_impl(std::index_sequence<Is...>, Tn i) {
      return (... + (i * excl_suffix_mul(Is, extents{})));
    }
    static constexpr Tn identity_offset(Tn i) {
      return identity_offset_impl(std::make_index_sequence<dim>{}, i);
    }
    template <std::size_t... Is, typename... Args>
    static constexpr Tn offset_impl(std::index_sequence<Is...>, Args &&...args) {
      return (... + (std::forward<Args>(args) * excl_suffix_mul(Is, extents{})));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    static constexpr Tn offset(Args &&...args) {
      return offset_impl(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
    }
  };
  template <typename Tn, Tn... Ns> using indexer = indexer_impl<std::integer_sequence<Tn, Ns...>>;

  /// vec without lifetime managing
  template <typename T, typename Tn, Tn... Ns> struct vec_view<T, std::integer_sequence<Tn, Ns...>>
      : indexer<Tn, Ns...> {
    using base_t = indexer<Tn, Ns...>;
    using base_t::dim;
    using base_t::extent;
    using value_type = T;
    using base_t::offset;
    using typename base_t::extents;
    using typename base_t::index_type;

    constexpr vec_view() = delete;
    constexpr explicit vec_view(T *ptr) : _data{ptr} {}

    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[base_t::offset(std::forward<Args>(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[base_t::offset(std::forward<Args>(args)...)];
    }
    // []
    template <typename Index,
              typename R
              = vec_view<T, gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index &&index) noexcept {
      return R{_data + base_t::offset(std::forward<Index>(index))};
    }
    template <typename Index,
              typename R
              = vec_view<std::add_const_t<T>,
                         gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index index) const noexcept {
      return R{_data + base_t::offset(std::forward<Index>(index))};
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr T &operator[](Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr const T &operator[](Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr T &val(Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr const T &val(Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }

  private:
    T *_data;
  };

  /// vec
  template <typename T, typename Tn, Tn... Ns>
  struct vec_impl<T, std::integer_sequence<Tn, Ns...>> {
    // static_assert(std::is_trivial<T>::value,
    //              "Vec element type is not trivial!\n");
    using base_t = indexer<Tn, Ns...>;
    static constexpr auto dim = sizeof...(Ns);
    static constexpr auto extent = (Ns * ...);
    using value_type = T;
    using index_type = Tn;
    using extents = std::integer_sequence<Tn, Ns...>;

    T _data[extent];

  public:
    /// expose internal
    constexpr auto data() noexcept -> T * { return _data; }
    constexpr auto data() volatile noexcept -> volatile T * { return (volatile T *)_data; }
    constexpr auto data() const noexcept -> const T * { return _data; }

    /// think this does not break rule of five
    /// https://github.com/kokkos/kokkos/issues/177
    constexpr vec_impl &operator=(const vec_impl &o) = default;
#if 0
    constexpr volatile vec_impl &operator=(const vec_impl &o) volatile {
      for (Tn i = 0; i < extent; ++i) data()[i] = o.data()[i];
      return *this;
    }
#endif
    template <typename... Args, std::size_t... Is, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const std::tuple<Args...> &tup, index_seq<Is...>) {
      vec_impl ret{};
      ((void)(ret.data()[Is] = std::get<Is>(tup)), ...);
      return ret;
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const std::tuple<Args...> &tup) {
      return from_tuple(tup, std::index_sequence_for<Args...>{});
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    constexpr vec_impl &operator=(const std::tuple<Args...> &tup) {
      *this = from_tuple(tup);
      return *this;
    }
    template <typename... Args, std::size_t... Is, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const zs::tuple<Args...> &tup, index_seq<Is...>) {
      vec_impl ret{};
      ((void)(ret.data()[Is] = zs::get<Is>(tup)), ...);
      return ret;
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const zs::tuple<Args...> &tup) {
      return from_tuple(tup, std::index_sequence_for<Args...>{});
    }

    static constexpr vec_impl from_array(const std::array<T, extent> &arr) noexcept {
      vec_impl r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = arr[i];
      return r;
    }
    constexpr std::array<T, extent> to_array() const noexcept {
      std::array<T, extent> r{};
      for (Tn i = 0; i < extent; ++i) r[i] = _data[i];
      return r;
    }
    static constexpr vec_impl uniform(T v) noexcept {
      vec_impl r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = v;
      return r;
    }
    static constexpr vec_impl zeros() noexcept { return uniform(0); }
    static constexpr vec_impl identity() noexcept {
      vec_impl r = zeros();
      for (Tn i = 0; i < vseq_t<Ns...>::reduce(getmax<Tn>{}); ++i)
        r.val(base_t::identity_offset(i)) = 1;
      return r;
    }
    constexpr void set(T val) noexcept {
      for (Tn idx = 0; idx < extent; ++idx) _data[idx] = val;
    }
    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[base_t::offset(std::forward<Args>(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[base_t::offset(std::forward<Args>(args)...)];
    }
    // []
    template <typename Index,
              typename R
              = vec_view<T, gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index &&index) noexcept {
      return R{_data + base_t::offset(std::forward<Index>(index))};
    }
    template <typename Index,
              typename R
              = vec_view<std::add_const_t<T>,
                         gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index index) const noexcept {
      return R{_data + base_t::offset(std::forward<Index>(index))};
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr T &operator[](Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr const T &operator[](Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr T &val(Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr const T &val(Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }
    ///

    template <typename TT> constexpr auto cast() const noexcept {
      vec_impl<TT, extents> r{};
      for (Tn idx = 0; idx < extent; ++idx) r.val(idx) = _data[idx];
      return r;
    }
    template <typename TT> constexpr operator vec_impl<TT, extents>() const noexcept {
      vec_impl<TT, extents> r{};
      for (Tn idx = 0; idx < extent; ++idx) r.val(idx) = _data[idx];
      return r;
    }
    /// compare
    template <typename TT>
    constexpr bool operator==(const vec_impl<TT, extents> &o) const noexcept {
      for (Tn i = 0; i < extent; ++i)
        if (_data[i] != o.val(i)) return false;
      return true;
    }
    template <typename TT>
    constexpr bool operator!=(const vec_impl<TT, extents> &&o) const noexcept {
      for (Tn i = 0; i < extent; ++i)
        if (_data[i] == o.val(i)) return false;
      return true;
    }

    /// linalg
    template <typename TT, typename R = std::common_type_t<T, TT>>
    constexpr R dot(vec_impl<TT, extents> const &o) const noexcept {
      R res{0};
      for (Tn i = 0; i < extent; ++i) res += _data[i] * o.val(i);
      return res;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>, std::size_t d = dim,
              std::size_t ext = extent, enable_if_all<d == 1, ext == 3> = 0>
    constexpr auto cross(const vec_impl<TT, extents> &o) const noexcept {
      vec_impl<R, extents> res{0};
      res.val(0) = _data[1] * o.val(2) - _data[2] * o.val(1);
      res.val(1) = _data[2] * o.val(0) - _data[0] * o.val(2);
      res.val(2) = _data[0] * o.val(1) - _data[1] * o.val(0);
      return res;
    }
    template <std::size_t d = dim, std::size_t ext = extent, enable_if_all<d == 1, ext == 3> = 0>
    constexpr vec_impl orthogonal() const noexcept {
      T x = gcem::abs(val(0));
      T y = gcem::abs(val(1));
      T z = gcem::abs(val(2));
      vec_impl other = x < y ? (x < z ? vec_impl{1, 0, 0} : vec_impl{0, 0, 1})
                             : (y < z ? vec_impl{0, 1, 0} : vec_impl{0, 0, 1});
      return cross(other);
    }
    template <std::size_t d = dim, enable_if_t<d == 2> = 0>
    constexpr auto transpose() const noexcept {
      constexpr auto N0 = select_indexed_value<0, Ns...>::value;
      constexpr auto N1 = select_indexed_value<1, Ns...>::value;
      using extentsT = std::integer_sequence<Tn, N1, N0>;
      vec_impl<T, extentsT> r{};
      for (Tn i = 0; i < N0; ++i)
        for (Tn j = 0; j < N1; ++j) r(j, i) = (*this)(i, j);
      return r;
    }
    constexpr T prod() const noexcept {
      T res{1};
      for (Tn i = 0; i < extent; ++i) res *= _data[i];
      return res;
    }
    constexpr T sum() const noexcept {
      T res{0};
      for (Tn i = 0; i < extent; ++i) res += _data[i];
      return res;
    }
    constexpr T l2NormSqr() const noexcept {
      T res{0};
      for (Tn i = 0; i < extent; ++i) res += _data[i] * _data[i];
      return res;
    }
    constexpr T infNormSqr() const noexcept {
      T res{0};
      for (Tn i = 0; i < extent; ++i) {
        T sqr = _data[i] * _data[i];
        if (sqr > res) res = sqr;
      }
      return res;
    }
    static constexpr T sqrtNewtonRaphson(T x, T curr, T prev) noexcept {
      return curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }
    constexpr T length() const noexcept {
      T sqrNorm = l2NormSqr();
      // return sqrtNewtonRaphson(sqrNorm, sqrNorm, (T)0);
      return gcem::sqrt(sqrNorm);
    }
    constexpr T norm() const noexcept {
      T sqrNorm = l2NormSqr();
      // return sqrtNewtonRaphson(sqrNorm, sqrNorm, (T)0);
      return gcem::sqrt(sqrNorm);
    }
    constexpr vec_impl normalized() const noexcept { return (*this) / length(); }
    constexpr vec_impl abs() const noexcept {
      vec_impl r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = _data[i] > 0 ? _data[i] : -_data[i];
      return r;
    }
    constexpr T max() const noexcept {
      T res{_data[0]};
      for (Tn i = 1; i < extent; ++i)
        if (_data[i] > res) res = _data[i];
      return res;
    }

    /// borrowed from
    /// https://github.com/cemyuksel/cyCodeBase/blob/master/cyIVector.h
    /// east const
    //!@name Unary operators
    constexpr vec_impl operator-() const noexcept {
      vec_impl r;
      for (Tn i = 0; i < extent; ++i) r.val(i) = -_data[i];
      return r;
    }

    //!@name Binary operators
    // scalar
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator+(vec_impl const &e, TT const v) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = e.val(i) + v;
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator+(TT const v, vec_impl const &e) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = e.val(i) + v;
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator-(vec_impl const &e, TT const v) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = e.val(i) - v;
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator-(TT const v, vec_impl const &e) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = v - e.val(i);
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator*(vec_impl const &e, TT const v) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = e.val(i) * v;
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator*(TT const v, vec_impl const &e) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = e.val(i) * v;
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator/(vec_impl const &e, TT const v) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = e.val(i) / v;
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    friend constexpr auto operator/(TT const v, vec_impl const &e) noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = v / e.val(i);
      return r;
    }
    // vector
    template <typename TT, typename R = std::common_type_t<T, TT>>
    constexpr auto operator+(vec_impl<TT, extents> const &o) const noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = _data[i] + o.val(i);
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    constexpr auto operator-(vec_impl<TT, extents> const &o) const noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = _data[i] - o.val(i);
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    constexpr auto operator*(vec_impl<TT, extents> const &o) const noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = _data[i] * o.val(i);
      return r;
    }
    template <typename TT, typename R = std::common_type_t<T, TT>>
    constexpr auto operator/(vec_impl<TT, extents> const &o) const noexcept {
      vec_impl<R, extents> r{};
      for (Tn i = 0; i < extent; ++i) r.val(i) = _data[i] / o.val(i);
      return r;
    }
    //!@name Assignment operators
    // scalar
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator+=(TT &&v) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] += std::forward<TT>(v);
      return *this;
    }
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator-=(TT &&v) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] -= std::forward<TT>(v);
      return *this;
    }
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator*=(TT &&v) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] *= std::forward<TT>(v);
      return *this;
    }
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator/=(TT &&v) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] /= std::forward<TT>(v);
      return *this;
    }
    // vector
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator+=(vec_impl<TT, extents> const &o) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] += o.val(i);
      return *this;
    }
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator-=(vec_impl<TT, extents> const &o) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] -= o.val(i);
      return *this;
    }
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator*=(vec_impl<TT, extents> const &o) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] *= o.val(i);
      return *this;
    }
    template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
    constexpr vec_impl &operator/=(vec_impl<TT, extents> const &o) noexcept {
      for (Tn i = 0; i < extent; ++i) _data[i] /= o.val(i);
      return *this;
    }

#if 0
  //!@name Bitwise operators
  vec_impl operator<<(vec_impl const &o) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] << o.val(i);
    return r;
  }
  vec_impl operator>>(vec_impl const &o) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] >> o.val(i);
    return r;
  }
  vec_impl operator&(vec_impl const &o) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] & o.val(i);
    return r;
  }
  vec_impl operator|(vec_impl const &o) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] | o.val(i);
    return r;
  }
  vec_impl operator^(vec_impl const &o) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] ^ o.val(i);
    return r;
  }
  vec_impl operator<<(T const v) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] << v;
    return r;
  }
  vec_impl operator>>(T const v) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] >> v;
    return r;
  }
  vec_impl operator&(T const v) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] & v;
    return r;
  }
  vec_impl operator|(T const v) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] | v;
    return r;
  }
  vec_impl operator^(T const v) const noexcept {
    vec_impl r;
    for (Tn i = 0; i < extent; ++i)
      r.val(i) = _data[i] ^ v;
    return r;
  }

  //!@name Bitwise Assignment operators
  vec_impl &operator<<=(vec_impl const &o) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] <<= o.val(i);
    return *this;
  }
  vec_impl &operator>>=(vec_impl const &o) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] >>= o.val(i);
    return *this;
  }
  vec_impl &operator&=(vec_impl const &o) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] &= o.val(i);
    return *this;
  }
  vec_impl &operator|=(vec_impl const &o) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] |= o.val(i);
    return *this;
  }
  vec_impl &operator^=(vec_impl const &o) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] ^= o.val(i);
    return *this;
  }
  vec_impl &operator<<=(T const v) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] <<= v;
    return *this;
  }
  vec_impl &operator>>=(T const v) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] >>= v;
    return *this;
  }
  vec_impl &operator&=(T const v) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] &= v;
    return *this;
  }
  vec_impl &operator|=(T const v) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] |= v;
    return *this;
  }
  vec_impl &operator^=(T const v) noexcept {
    for (Tn i = 0; i < extent; ++i)
      _data[i] ^= v;
    return *this;
  }
#endif
  };

  /// make vec
  template <typename... Args> constexpr auto make_vec(Args &&...args) {
    using Tn = std::common_type_t<std::decay_t<Args>...>;
    return vec<Tn, sizeof...(Args)>{std::forward<Args>(args)...};
  }
  /// make vec from std tuple
  template <typename T, typename... Ts, std::size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const std::tuple<Ts...> &tup, index_seq<Is...>) {
    return vec<T, (sizeof...(Ts))>{std::get<Is>(tup)...};
  }
  template <typename T, typename... Ts>
  constexpr auto make_vec(const std::tuple<Ts...> &tup) {
    return make_vec_impl<T>(tup, std::index_sequence_for<Ts...>{});
  }
  /// make vec from zs tuple
  template <typename T, typename... Ts, std::size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const tuple<Ts...> &tup, index_seq<Is...>) {
    return vec<T, (sizeof...(Ts))>{get<Is>(tup)...};
  }
  template <typename T, typename... Ts>
  constexpr auto make_vec(const tuple<Ts...> &tup) {
    return make_vec_impl<T>(tup, std::index_sequence_for<Ts...>{});
  }

  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T0, std::integer_sequence<Tn, N0, N1>> const &A,
                           vec_impl<T1, std::integer_sequence<Tn, N1>> const &x) noexcept {
    using R = std::common_type_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N0>> r{};
    for (Tn i = 0; i < N0; ++i) {
      r(i) = 0;
      for (Tn j = 0; j < N1; ++j) r(i) += A(i, j) * x(j);
    }
    return r;
  }
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T1, std::integer_sequence<Tn, N0>> const &x,
                           vec_impl<T0, std::integer_sequence<Tn, N0, N1>> const &A) noexcept {
    using R = std::common_type_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N1>> r{};
    for (Tn i = 0; i < N1; ++i) {
      r(i) = 0;
      for (Tn j = 0; j < N0; ++j) r(i) += A(j, i) * x(j);
    }
    return r;
  }
  /// affine transform
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T0, std::integer_sequence<Tn, N0, N1>> const &A,
                           vec_impl<T1, std::integer_sequence<Tn, N1 - 1>> const &x) noexcept {
    using R = std::common_type_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N0 - 1>> r{};
    for (Tn i = 0; i < N0 - 1; ++i) {
      r(i) = 0;
      for (Tn j = 0; j < N1; ++j) r(i) += A(i, j) * (j == N1 - 1 ? (T1)1 : x(j));
    }
    return r;
  }
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T1, std::integer_sequence<Tn, N0 - 1>> const &x,
                           vec_impl<T0, std::integer_sequence<Tn, N0, N1>> const &A) noexcept {
    using R = std::common_type_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N1 - 1>> r{};
    for (Tn i = 0; i < N1 - 1; ++i) {
      r(i) = 0;
      for (Tn j = 0; j < N0; ++j) r(i) += A(j, i) * (j == N0 - 1 ? (T1)1 : x(j));
    }
    return r;
  }

  /// affine map = linear map + translation matrix+(0, 0, 1) point(vec+{1})
  /// vector(vec+{0}) homogeneous coordinates

  template <typename... Args> constexpr auto make_array(Args &&...args) {
    return std::array<std::common_type_t<remove_cvref_t<Args>...>, sizeof...(Args)>{FWD(args)...};
  }
  template <typename RetT, typename... Args> constexpr auto make_array(Args &&...args) {
    return std::array<RetT, sizeof...(Args)>{FWD(args)...};
  }

}  // namespace zs
