#pragma once

#include <functional>

#include "Meta.h"
#include "Sequence.h"
#include "zensim/tpls/tl/function_ref.hpp"
#include "zensim/types/Function.h"
#include "zensim/types/Optional.h"

namespace zs {

  /// operator

  /// binary operation
  template <typename T> using plus = std::plus<T>;
  template <typename T> using minus = std::minus<T>;
  template <typename T> using logical_or = std::logical_or<T>;
  template <typename T> using logical_and = std::logical_and<T>;
  template <typename T> using multiplies = std::multiplies<T>;
  template <typename T> struct getmax {
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept {
      return lhs > rhs ? lhs : rhs;
    }
  };
  template <typename T> struct getmin {
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept {
      return lhs < rhs ? rhs : lhs;
    }
  };

  /// monoid operation for value sequence declaration
  template <typename BinaryOp> struct monoid_op;
  template <typename T> struct monoid_op<plus<T>> {
    static constexpr T e{0};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) + ...);
    }
  };
  template <typename T> struct monoid_op<multiplies<T>> {
    static constexpr T e{1};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) * ...);
    }
  };
  template <typename T> struct monoid_op<logical_or<T>> {
    static constexpr T e{false};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) || ...);
    }
  };
  template <typename T> struct monoid_op<logical_and<T>> {
    static constexpr T e{true};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) && ...);
    }
  };
  template <typename T> struct monoid_op<getmax<T>> {
    static constexpr T e{std::numeric_limits<T>::lowest()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res > args ? res : args), ...);
    }
  };
  template <typename T> struct monoid_op<getmin<T>> {
    static constexpr T e{std::numeric_limits<T>::max()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res < args ? res : args), ...);
    }
  };

  /// map operation
  struct count_leq {  ///< count less and equal
    template <typename... Tn> constexpr auto operator()(std::size_t M, Tn... Ns) {
      return ((Ns <= M ? 1 : 0) + ...);
    }
  };

  template <typename F, typename T> constexpr decltype(auto) fold(F &&f, T &&v) {
    return std::forward<T>(v);
  }

  template <typename F, typename T, typename... Ts>
  constexpr decltype(auto) fold(F &&f, T &&v, Ts &&...vs) {
    return std::forward<F>(f)(std::forward<T>(v),
                              fold(std::forward<F>(f), std::forward<Ts>(vs)...));
  }

  // template <template <typename> class Functor, typename T>
  // constexpr T join(Functor<T> &&m) {
  //  return {std::get<T>(m)};
  //}

  /// monad : construct, mbind
  // construct T -> M<T>
  template <template <typename> class Functor, typename T>
  constexpr Functor<T> construct(T &&value) {
    return {std::forward<T>(value)};
  }
  // mbind (M<T0>, T0 -> M<T1>) -> M<T1>
  template <typename T, typename F> constexpr auto mbind(const optional<T> &m, F &&f)
      -> decltype(std::invoke(f, m.value())) {
    if (m)
      return f(m.value());
    else
      return {};
  }

  /// functor: map, transform
  // Kleisli composition
  template <typename F, typename G> constexpr auto mcompose(F &&f, G &&g) {
    return [f = FWD_CAPTURE(f), g = FWD_CAPTURE(g)](auto value) {
      return mbind(f.template get<0>()(value), g.template get<0>());
    };
  }
  template <typename A, typename B, typename C>
  constexpr auto compose(std::function<B(A)> &f0, std::function<C(B)> &f1) {
    return [&f0, &f1](const A &x) -> C { return std::invoke(f1, std::invoke(f0, x)); };
  }
  template <typename A, typename B, typename... Fs>
  constexpr auto compose(std::function<B(A)> &f0, Fs &&...fs) {
    return [&f0, &fs...](const A &x) mutable {
      return std::invoke(compose(std::forward<Fs>(fs)...), std::invoke(f0, x));
    };
  }
  namespace view {
    template <template <typename> class Functor, typename T, typename R>
    constexpr auto map(const Functor<T> &functor, tl::function_ref<R(T)> f) -> Functor<R> {
      Functor<R> res{functor};
      for (auto &e : res) e = f(e);
      return res;
    }
  }  // namespace view

  namespace action {}  // namespace action

  /// result_of

  template <typename T> struct add_optional { using type = optional<T>; };
  template <typename T> struct add_optional<optional<T>> { using type = optional<T>; };

  template <typename T> using add_optional_t = typename add_optional<T>::type;

  /// zip, enumerate, map, iterator

#if 0
template <typename T, typename Func>
auto operator|(std::optional<T> const &opt, Func const &func)
    -> add_optionality<decltype(func(*opt))> {
  if (opt)
    return func(*opt);
  else
    return {};
}

template <
    class T, class F,
    std::enable_if_t<detail::is_optional<std::decay_t<T>>::value, int> = 0>
auto operator|(T &&t, F &&f) -> decltype(
    detail::void_or_nullopt<decltype(f(std::forward<T>(t).operator*()))>()) {
  using return_type = decltype(f(std::forward<T>(t).operator*()));
  if (t)
    return f(std::forward<T>(t).operator*());
  else
    return detail::void_or_nullopt<return_type>();
}
#endif

}  // namespace zs
