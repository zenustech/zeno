#pragma once

#include <tuple>
#include <memory>
#include <variant>
#include <functional>
#include <type_traits>
#include <utility>

namespace zeno {

template <class T, class...>
using first_t = T;

template <auto Val, class...>
static inline constexpr auto first_v = Val;


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


template <class T>
struct remove_cvref : std::remove_cv<std::remove_reference_t<T>> {
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

#define ZENO_RMCVREF(...) ::zeno::remove_cvref_t<decltype(__VA_ARGS__)>
#define ZENO_DECAY(...) std::decay_t<decltype(__VA_ARGS__)>
#define ZENO_FWD(x) std::forward<decltype(x)>(x)
#define ZENO_CRTP(Derived, Base, ...) Derived : Base<Derived __VA_ARGS__>
#define ZENO_MKFUNC(x, ...) [__VA_ARGS__] (auto &&...x) -> decltype(auto) { return f(ZENO_FWD(x)...); }


template <class T>
struct type_identity {
    using type = T;
};

template <class T>
using type_identity_t = typename type_identity<T>::type;

template <class T>
static inline constexpr type_identity<T> type_identity_v{};


template <class ...Ts>
struct type_identity_list {
    using type = std::tuple<type_identity<Ts>...>;
};

template <class ...Ts>
using type_identity_list_t = typename type_identity_list<Ts...>::type;

template <class ...Ts>
static inline constexpr type_identity_list<Ts...> type_identity_list_v{};


template <auto Val>
struct value_constant : std::integral_constant<decltype(Val), Val> {
};

template <auto Val>
static inline constexpr value_constant<Val> value_constant_v{};

template <auto Val>
constexpr std::integral_constant<decltype(Val), Val> make_integral_constant() {
    return {};
}


struct avoided_void {
    using type = void;
    constexpr void operator()() const {}
    constexpr operator bool() { return false; }
};

template <class T, class U = avoided_void>
struct avoid_void {
    using type = std::conditional_t<std::is_void_v<T>, U, T>;
};

template <class F, class ...Args>
constexpr decltype(auto) avoid_void_call(F &&f, Args &&...args) {
    if constexpr (std::is_void_v<std::invoke_result_t<F, Args...>>) {
        std::forward<F>(f)(std::forward<Args>(args)...);
        return avoided_void{};
    } else {
        return std::forward<F>(f)(std::forward<Args>(args)...);
    }
}

template <class T, class U = avoided_void>
using avoid_void_t = typename avoid_void<T, U>::type;


// usage example:
// static_for<0, n>([&] (auto index) {
//     std::array<int, index.value> arr;
//     return false;   // true to break the loop
// });
template <auto First, auto Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (avoid_void_call(f, make_integral_constant<First>())) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}


template <class To, class T>
inline constexpr To implicit_cast(T &&t) {
    return std::forward<T>(t);
}

template <class To, class T>
inline constexpr To braced_cast(T &&t) {
    return To{std::forward<T>(t)};
}

template <class To, class T>
inline constexpr To cstyle_cast(T &&t) {
    return To(std::forward<T>(t));
}

template <class T>
inline constexpr T as_copy(T const &t) {
    return t;
}

template <class T>
inline constexpr T &as_non_const(T const volatile &t) {
    return const_cast<T &>(t);
}

struct identity {
    template <class T>
    constexpr T &&operator()(T &&t) const noexcept {
        return std::forward<T>(t);
    }
};

}
