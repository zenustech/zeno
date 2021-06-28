#pragma once
#include <type_traits>
#include <variant>

#include "zensim/meta/Sequence.h"

namespace zs {

  /// https://github.com/SuperV1234/ndctechtown2020/blob/master/7_a_match.pdf
  template <typename... Fs> struct overload_set : Fs... {
    template <typename... Xs> constexpr overload_set(Xs &&...xs) : Fs{std::forward<Xs>(xs)}... {}
    using Fs::operator()...;
  };
  /// class template argument deduction
  template <typename... Xs> overload_set(Xs &&...xs) -> overload_set<remove_cvref_t<Xs>...>;

  template <typename... Fs> constexpr auto make_overload_set(Fs &&...fs) {
    return overload_set<std::decay_t<Fs>...>(std::forward<Fs>(fs)...);
  }

  template <typename... Ts> using variant = std::variant<Ts...>;

  template <typename... Fs> constexpr auto match(Fs &&...fs) {
#if 0
  return [visitor = overload_set{std::forward<Fs>(fs)...}](
             auto &&...vs) -> decltype(auto) {
    return std::visit(visitor, std::forward<decltype(vs)>(vs)...);
  };
#else
    return [visitor = make_overload_set(std::forward<Fs>(fs)...)](auto &&...vs) -> decltype(auto) {
      return std::visit(visitor, std::forward<decltype(vs)>(vs)...);
    };
#endif
  }

  template <typename> struct is_variant : std::false_type {};
  template <typename... Ts> struct is_variant<variant<Ts...>> : std::true_type {};

}  // namespace zs
