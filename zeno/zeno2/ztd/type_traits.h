#pragma once


#include <tuple>
#include <memory>
#include <variant>
#include <functional>


namespace zeno2::ztd {
inline namespace type_traits {

template <class F, class ...Ts>
using nocvref_invoke_result_t = std::remove_cvref_t<decltype(std::declval<F>()(
        std::declval<std::remove_cvref_t<Ts>>()...))>;

template <class ...Ts>
struct promoted {
};

template <class T, class ...Ts>
struct promoted<T, Ts...> {
    using type = decltype(0 ? std::declval<T>() : std::declval<typename promoted<Ts...>::type>());
};

template <class T>
struct promoted<T> {
    using type = T;
};

template <class ...Ts>
using promoted_t = std::remove_cvref_t<typename promoted<Ts...>::type>;

template <class T, class ...Ts>
using first_t = T;

template <class ...Ts>
static constexpr bool true_v = true;

template <class ...Ts>
static constexpr bool false_v = false;

template <class T>
struct is_tuple : std::false_type {
};

template <class ...Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {
};

template <typename T, typename Tuple>
struct tuple_contains;

template <typename T>
struct tuple_contains<T, std::tuple<>> : std::false_type {
};

template <typename T, typename U, typename... Ts>
struct tuple_contains<T, std::tuple<U, Ts...>> : tuple_contains<T, std::tuple<Ts...>> {
};

template <typename T, typename... Ts>
struct tuple_contains<T, std::tuple<T, Ts...>> : std::true_type {
};

template <class T>
struct is_variant : std::false_type {
};

template <class ...Ts>
struct is_variant<std::variant<Ts...>> : std::true_type {
    using tuple_type = std::tuple<Ts...>;
};

template <class T, class V>
static constexpr bool variant_contains_v = false;

template <class T, class ...Ts>
static constexpr bool variant_contains_v<T, std::variant<Ts...>> = (std::is_same_v<T, Ts> || ...);

template <class T>
struct variant_to_tuple {
};

template <class ...Ts>
struct variant_to_tuple<std::variant<Ts...>> {
    using type = std::tuple<Ts...>;
};

template <class T>
struct is_shared_ptr : std::false_type {
};

template <class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {
};

template <class T>
struct remove_shared_ptr {
    using type = T;
};

template <class T>
struct remove_shared_ptr<std::shared_ptr<T>> {
    using type = T;
};

template <class T>
struct is_unique_ptr : std::false_type {
};

template <class T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {
};

template <class T>
struct remove_unique_ptr {
    using type = T;
};

template <class T>
struct remove_unique_ptr<std::unique_ptr<T>> {
    using type = T;
};

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

template <int First, int Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}

}
}
