#pragma once

#include "Container.h"


namespace zenov2 {

namespace __details {
    template <class T>
    struct function_traits : function_traits<decltype(&T::operator())> {
    };

    // partial specialization for function type
    template <class R, class... Args>
    struct function_traits<R(Args...)> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    // partial specialization for function pointer
    template <class R, class... Args>
    struct function_traits<R (*)(Args...)> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    // partial specialization for std::function
    template <class R, class... Args>
    struct function_traits<std::function<R(Args...)>> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    // partial specialization for pointer-to-member-function (i.e., operator()'s)
    template <class T, class R, class... Args>
    struct function_traits<R (T::*)(Args...)> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    template <class T, class R, class... Args>
    struct function_traits<R (T::*)(Args...) const> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    template <class Tuple, class List, size_t ...Indices>
    auto impl_any_list_to_tuple(List &&list, std::index_sequence<Indices...>) {
        return std::make_tuple(
                container::smart_any_cast<std::tuple_element_t<Indices, Tuple>>
                (list.at(Indices))...);
    }

    template <class Tuple, class List>
    auto any_list_to_tuple(List &&list) {
        constexpr size_t N = std::tuple_size_v<Tuple>;
        return impl_any_list_to_tuple<Tuple>(
                std::forward<List>(list),
                std::make_index_sequence<N>{});
    }

    template <class Tuple, class List, size_t ...Indices>
    void impl_tuple_to_any_list(Tuple &&tuple, List &&list, std::index_sequence<Indices...>) {
        ((list.at(Indices) = std::get<Indices>(tuple), (void)0), ...);
    }

    template <class Tuple, class List>
    void tuple_to_any_list(Tuple &&tuple, List &&list) {
        constexpr size_t N = std::tuple_size_v<Tuple>;
        impl_tuple_to_any_list(
                std::forward<Tuple>(tuple),
                std::forward<List>(list),
                std::make_index_sequence<N>{});
    }

    template <class T>
    struct is_tuple : std::false_type {
    };

    template <class ...Ts>
    struct is_tuple<std::tuple<Ts...>> : std::true_type {
    };

    template <class T>
    auto tuple_if_not_tuple(T &&t) {
        if constexpr (is_tuple<T>::value) {
            return t;
        } else {
            return std::tuple<T>(t);
        }
    }

    template <class F, class T>
    auto call_with_wrap_tuple(F func, T &&args) {
        if constexpr (std::is_void_v<decltype(std::apply(func, std::move(args)))>) {
            std::apply(func, std::move(args));
            return std::tuple<>();
        } else {
            return tuple_if_not_tuple(std::apply(func, std::move(args)));
        }
    }
}

template <class F>
auto wrap_context_function(F func) {
    return [=] (backend::Context *ctx) {
        using Args = typename details::function_traits<F>::argument_types;
        auto rets = details::call_with_wrap_tuple(
                func, details::any_list_to_tuple<Args>(
                    static_cast<backend::Context const *>(ctx)->inputs));
        details::tuple_to_any_list(std::move(rets), ctx->outputs);
    };
}

}
