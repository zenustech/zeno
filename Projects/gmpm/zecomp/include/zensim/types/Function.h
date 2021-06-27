#pragma once
#include <functional>
#include <tuple>

#include "zensim/tpls/tl/function_ref.hpp"

namespace zs {

  namespace detail {
    /// currently only support free functions & lambdas/functors
    /// exclude member functions
    template <typename, typename = void> struct function_traits_impl;
    // free-function
    template <typename R, typename... Args> struct function_traits_impl<R(Args...)> {
      static constexpr std::size_t arity = sizeof...(Args);
      using return_t = R;
      using arguments_t = std::tuple<Args...>;
    };
    template <typename R, typename... Args> struct function_traits_impl<R (*)(Args...)>
        : function_traits_impl<R(Args...)> {};
    // member function
    template <typename C, typename R, typename... Args>  // function member pointer
    struct function_traits_impl<R (C::*)(Args...)> : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R>  // data member pointer
    struct function_traits_impl<R(C::*)> : function_traits_impl<R(C &)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const> : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) volatile>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const volatile>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) volatile noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const volatile noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    // lambda/ functor
    template <typename Functor>
    struct function_traits_impl<Functor, std::void_t<decltype(&Functor::operator())>> {
    protected:
      using calltype = function_traits_impl<decltype(&Functor::operator())>;
      template <typename... Ts, std::size_t... Is>
      static auto extract_arguments(std::tuple<Ts...>, std::index_sequence<Is...>)
          -> std::tuple<std::tuple_element_t<Is + 1, typename calltype::arguments_t>...>;

    public:
      static constexpr std::size_t arity = calltype::arity - 1;
      using return_t = typename calltype::return_t;
      using arguments_t = decltype(extract_arguments(std::declval<typename calltype::arguments_t>(),
                                                     std::make_index_sequence<arity>{}));
    };

  }  // namespace detail
  template <typename F> using function_traits = detail::function_traits_impl<std::decay_t<F>>;

  template <class R, class... Args> using function = std::function<R(Args...)>;

}  // namespace zs
