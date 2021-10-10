#pragma once


#include <zs/ztd/type_traits.h>
#include <functional>
#include <algorithm>
#include <concepts>
#include <utility>
#include <cstdint>
#include <cmath>


namespace zs::ztd {
inline namespace mathvec {


/* main class */

template <size_t N, class T>
    requires(N >= 1 && std::copy_constructible<T> && std::default_initializable<T>)
struct vec {
    union {
        T _M_data[N]{};
        struct {
            std::conditional_t<N >= 1, T, std::type_identity<T>> x;
            std::conditional_t<N >= 2, T, std::type_identity<T>> y;
            std::conditional_t<N >= 3, T, std::type_identity<T>> z;
            std::conditional_t<N >= 4, T, std::type_identity<T>> w;
        };
    };

    constexpr size_t size() const {
        return N;
    }

    constexpr T *data() {
        return _M_data;
    }

    constexpr T const *data() const {
        return _M_data;
    }

    constexpr vec() = default;
    constexpr vec(vec const &) = default;
    constexpr vec &operator=(vec const &) = default;
    constexpr vec(vec &&) = default;
    constexpr vec &operator=(vec &&) = default;

    constexpr vec(std::initializer_list<T> const &ts) {
        auto it = ts.begin();
        for (int i = 0; i < std::min(N, ts.size()); i++) {
            data()[i] = *it++;
        }
    }

    template <class ...Ts>
        requires(sizeof...(Ts) == N - 2)
    constexpr vec(T t1, T t2, Ts ...ts)
        : vec(std::initializer_list<T>{t1, t2, (T)ts...})
    {}

    constexpr explicit(N != 1) vec(T const &t) {
        for (int i = 0; i < N; i++) {
            data()[i] = t;
        }
    }

    template <class ...Ts>
    constexpr explicit(N != 1) operator T() {
        return data()[0];
    }

    constexpr T const &operator[](size_t i) const {
        return data()[i];
    }

    constexpr T &operator[](size_t i) {
        return data()[i];
    }

    template <class S>
        requires(std::convertible_to<T, S> || std::constructible_from<S, T>)
    constexpr explicit(!std::is_convertible_v<T, S>) operator vec<N, S>() const {
        vec<N, S> ret;
        for (int i = 0; i < N; i++) {
            if constexpr (std::is_convertible_v<T, S>) {
                ret[i] = data()[i];
            } else {
                ret[i] = S(data()[i]);
            }
        }
        return ret;
    }
};


/* type traits */

template <class T>
struct vec_traits : std::false_type {
    static constexpr size_t dim = 0;
    using type = T;
};

template <size_t N, class T>
struct vec_traits<vec<N, T>> : std::true_type {
    static constexpr size_t dim = N;
    using type = T;

    template <class S>
    constexpr bool operator==(vec_traits<S> const &that) const {
        return !value || !that.value || dim == that.dim;
    }
};

template <class ...Ts>
static constexpr bool vec_broadcastable = false;

template <class T>
static constexpr bool vec_broadcastable<T> = true;

template <class T1, class T2, class ...Ts>
static constexpr bool vec_broadcastable<T1, T2, Ts...> = vec_traits<T1>() == vec_traits<T2>() && vec_broadcastable<T2, Ts...>;


/* broadcasting */

template <class F, class ...Ts>
    requires (!vec_traits<Ts>::value && ...)
constexpr auto vapply(F func, Ts const &...ts) {
    return func(ts...);
}

template <class F, size_t N, class ...Ts>
    requires (!vec_traits<Ts>::value && ...)
constexpr auto vapply(F func, vec<N, Ts> const &...ts) {
    using T0 = nocvref_invoke_result_t<F, Ts...>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(ts[i]...);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto vapply(F func, vec<N, T1> const &t1, T2 const &t2) {
    using T0 = nocvref_invoke_result_t<F, T1, T2>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1[i], t2);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto vapply(F func, T1 const &t1, vec<N, T2> const &t2) {
    using T0 = nocvref_invoke_result_t<F, T1, T2>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1, t2[i]);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2, class T3>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value && !vec_traits<T3>::value)
constexpr auto vapply(F func, vec<N, T1> const &t1, T2 const &t2, T3 const &t3) {
    using T0 = nocvref_invoke_result_t<F, T1, T2, T3>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1[i], t2, t3);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2, class T3>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value && !vec_traits<T3>::value)
constexpr auto vapply(F func, T1 const &t1, vec<N, T2> const &t2, T3 const &t3) {
    using T0 = nocvref_invoke_result_t<F, T1, T2, T3>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1, t2[i], t3);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2, class T3>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value && !vec_traits<T3>::value)
constexpr auto vapply(F func, T1 const &t1, T2 const &t2, vec<N, T3> const &t3) {
    using T0 = nocvref_invoke_result_t<F, T1, T2, T3>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1, t2, t3[i]);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2, class T3>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value && !vec_traits<T3>::value)
constexpr auto vapply(F func, vec<N, T1> const &t1, vec<N, T2> const &t2, T3 const &t3) {
    using T0 = nocvref_invoke_result_t<F, T1, T2, T3>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1[i], t2[i], t3);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2, class T3>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value && !vec_traits<T3>::value)
constexpr auto vapply(F func, T1 const &t1, vec<N, T2> const &t2, vec<N, T3> const &t3) {
    using T0 = nocvref_invoke_result_t<F, T1, T2, T3>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1, t2[i], t3[i]);
    }
    return ret;
}

template <class F, size_t N, class T1, class T2, class T3>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value && !vec_traits<T3>::value)
constexpr auto vapply(F func, vec<N, T1> const &t1, T2 const &t2, vec<N, T3> const &t3) {
    using T0 = nocvref_invoke_result_t<F, T1, T2, T3>;
    vec<N, T0> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = func(t1[i], t2, t3[i]);
    }
    return ret;
}

template <class F, class T1, class T2>
constexpr auto &ivapply(F func, T1 &t1, T2 const &t2) {
    return (t1 = vapply(func, t1, t2));
}


/* casting utilities */

template <class S, size_t N, class T>
    requires (!vec_traits<S>::value && !vec_traits<T>::value && (std::convertible_to<T, S> || std::constructible_from<S, T>))
constexpr vec<N, S> vcast(vec<N, T> const &t) {
    if constexpr (std::is_convertible_v<T, S>) {
        return t;
    } else {
        return vec<N, S>(t);
    }
}

template <class S, class T>
    requires (!vec_traits<S>::value && !vec_traits<T>::value && (std::convertible_to<T, S> || std::constructible_from<S, T>))
constexpr S vcast(T const &t) {
    if constexpr (std::is_convertible_v<T, S>) {
        return t;
    } else {
        return S(t);
    }
}

template <class S, size_t N, class T>
constexpr S vec_to_other(vec<N, T> const &t) {
    return ([&] <size_t ...I> (std::index_sequence<I...>) {
        return S(t[I]...);
    })(std::make_index_sequence<N>());
}

template <size_t N, class S>
constexpr auto other_to_vec(S const &s) {
    return ([&] <size_t ...I> (std::index_sequence<I...>) {
        using T = std::remove_cvref_t<decltype(s[0])>;
        return vec<N, T>(s[I]...);
    })(std::make_index_sequence<N>());
}


/* operators */

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() + std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator+(T1 const &t1, T2 const &t2) {
    return vapply(std::plus{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() - std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator-(T1 const &t1, T2 const &t2) {
    return vapply(std::minus{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() * std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator*(T1 const &t1, T2 const &t2) {
    return vapply(std::multiplies{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() / std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator/(T1 const &t1, T2 const &t2) {
    return vapply(std::divides{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() % std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator%(T1 const &t1, T2 const &t2) {
    return vapply(std::modulus{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() == std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator==(T1 const &t1, T2 const &t2) {
    return vapply(std::equal_to{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() != std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator!=(T1 const &t1, T2 const &t2) {
    return vapply(std::not_equal_to{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() > std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator>(T1 const &t1, T2 const &t2) {
    return vapply(std::greater{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() < std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator<(T1 const &t1, T2 const &t2) {
    return vapply(std::less{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() >= std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator>=(T1 const &t1, T2 const &t2) {
    return vapply(std::greater_equal{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() <= std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator<=(T1 const &t1, T2 const &t2) {
    return vapply(std::less_equal{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() & std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator&(T1 const &t1, T2 const &t2) {
    return vapply(std::bit_and{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() | std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator|(T1 const &t1, T2 const &t2) {
    return vapply(std::bit_or{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() ^ std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator^(T1 const &t1, T2 const &t2) {
    return vapply(std::bit_xor{}, t1, t2);
}

template <class = void>
struct __bit_shr {
    template <class T1, class T2>
    constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
        return std::forward<T1>(t1) >> std::forward<T2>(t2);
    }
};

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() >> std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator>>(T1 const &t1, T2 const &t2) {
    return vapply(__bit_shr{}, t1, t2);
}

template <class = void>
struct __bit_shl {
    template <class T1, class T2>
    constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
        return std::forward<T1>(t1) << std::forward<T2>(t2);
    }
};

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() << std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator<<(T1 const &t1, T2 const &t2) {
    return vapply(__bit_shl{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() && std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator&&(T1 const &t1, T2 const &t2) {
    return vapply(std::logical_and{}, t1, t2);
}

template <class T1, class T2>
    requires ((vec_traits<T1>::value || vec_traits<T2>::value) && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() || std::declval<typename vec_traits<T2>::type>())>)
constexpr auto operator||(T1 const &t1, T2 const &t2) {
    return vapply(std::logical_or{}, t1, t2);
}

template <class T1>
    requires (vec_traits<T1>::value && true_v<decltype(!(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto operator!(T1 const &t1) {
    return vapply(std::logical_not{}, t1);
}

template <class = void>
struct __positive {
    template <class T1>
    constexpr decltype(auto) operator()(T1 &&t1) const {
        return +std::forward<T1>(t1);
    }
};

template <class T1>
    requires (vec_traits<T1>::value && true_v<decltype(+(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto operator+(T1 const &t1) {
    return vapply(__positive{}, t1);
}

template <class T1>
    requires (vec_traits<T1>::value && true_v<decltype(-(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto operator-(T1 const &t1) {
    return vapply(std::negate{}, t1);
}

template <class T1>
    requires (vec_traits<T1>::value && true_v<decltype(~(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto operator~(T1 const &t1) {
    return vapply(std::bit_not{}, t1);
}


/* inplace operators */

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() + std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator+=(T1 &t1, T2 const &t2) {
    return ivapply(std::plus{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() - std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator-=(T1 &t1, T2 const &t2) {
    return ivapply(std::minus{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() * std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator*=(T1 &t1, T2 const &t2) {
    return ivapply(std::multiplies{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() / std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator/=(T1 &t1, T2 const &t2) {
    return ivapply(std::divides{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() % std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator%=(T1 &t1, T2 const &t2) {
    return ivapply(std::modulus{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() & std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator&=(T1 &t1, T2 const &t2) {
    return ivapply(std::bit_and{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() | std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator|=(T1 &t1, T2 const &t2) {
    return ivapply(std::bit_or{}, t1, t2);
}

template <class T1, class T2>
    requires (vec_traits<T1>::value && vec_broadcastable<T1, T2>
        && true_v<decltype(std::declval<typename vec_traits<T1>::type>() ^ std::declval<typename vec_traits<T2>::type>())>)
constexpr auto &operator^=(T1 &t1, T2 const &t2) {
    return ivapply(std::bit_xor{}, t1, t2);
}


/* scalar math functions */

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::floor(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto floor(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::floor((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::ceil(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto ceil(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::ceil((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::sin(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto sin(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::sin((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::cos(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto cos(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::cos((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::tan(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto tan(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::tan((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::asin(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto asin(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::sin((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::acos(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto acos(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::acos((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::atan(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto atan(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::atan((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::sinh(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto sinh(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::sinh((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::cosh(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto cosh(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::cosh((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::tanh(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto tanh(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::tanh((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::asinh(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto asinh(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::sinh((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::acosh(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto acosh(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::acosh((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::atanh(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto atanh(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::atanh((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::abs(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto abs(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::abs((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::fabs(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto fabs(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::fabs((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::sqrt(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto sqrt(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::sqrt((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::exp(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto exp(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::exp((T0)t1);
    }, t1);
}

template <class T1, class T0 = typename vec_traits<T1>::type>
    requires (true_v<decltype(std::log(std::declval<typename vec_traits<T1>::type>()))>)
constexpr auto log(T1 const &t1) {
    return vapply([] (auto const &t1) -> T0 {
        return std::log((T0)t1);
    }, t1);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::fmod(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto fmod(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::fmod((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::fmin(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto fmin(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::fmin((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::fmax(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto fmax(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::fmax((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::min(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto min(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::min((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::max(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto max(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::max((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::atan2(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto atan2(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::atan2((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type>>
    requires (vec_broadcastable<T1, T2> && true_v<decltype(std::pow(std::declval<T0>(), std::declval<T0>()))>)
constexpr auto pow(T1 const &t1, T2 const &t2) {
    return vapply([] (auto const &t1, auto const &t2) -> T0 {
        return std::pow((T0)t1, (T0)t2);
    }, t1, t2);
}

template <class T1, class T2, class T3, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type, typename vec_traits<T3>::type>>
    requires (vec_broadcastable<T1, T2, T3> && true_v<decltype(std::fma(std::declval<T0>(), std::declval<T0>(), std::declval<T0>()))>)
constexpr auto fma(T1 const &t1, T2 const &t2, T3 const &t3) {
    return vapply([] (auto const &t1, auto const &t2, auto const &t3) -> T0 {
        return std::fma((T0)t1, (T0)t2, (T0)t3);
    }, t1, t2, t3);
}

template <class T1, class T2, class T3, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type, typename vec_traits<T3>::type>>
    requires (vec_broadcastable<T1, T2, T3> && true_v<decltype(std::clamp(std::declval<T0>(), std::declval<T0>(), std::declval<T0>()))>)
constexpr auto clamp(T1 const &t1, T2 const &t2, T3 const &t3) {
    return vapply([] (auto const &t1, auto const &t2, auto const &t3) -> T0 {
        return std::clamp((T0)t1, (T0)t2, (T0)t3);
    }, t1, t2, t3);
}

template <class T1, class T2, class T3, class T0 = promoted_t<typename vec_traits<T1>::type, typename vec_traits<T2>::type, typename vec_traits<T3>::type>>
    requires (vec_broadcastable<T1, T2, T3> && true_v<decltype(std::lerp(std::declval<T0>(), std::declval<T0>(), std::declval<T0>()))>)
constexpr auto lerp(T1 const &t1, T2 const &t2, T3 const &t3) {
    return vapply([] (auto const &t1, auto const &t2, auto const &t3) -> T0 {
        return std::lerp((T0)t1, (T0)t2, (T0)t3);
    }, t1, t2, t3);
}


/* vector math functions */

template <size_t N, class T>
    requires (!vec_traits<T>::value && (std::convertible_to<T, bool> || std::constructible_from<bool, T>))
constexpr bool vany(vec<N, T> const &t) {
    bool ret = false;
    for (size_t i = 0; i < N; i++) {
        ret = ret || (bool)t[i];
    }
    return ret;
}

template <class T>
    requires (!vec_traits<T>::value && (std::convertible_to<T, bool> || std::constructible_from<bool, T>))
constexpr bool vany(T const &t) {
    return (bool)t;
}

template <size_t N, class T>
    requires (!vec_traits<T>::value && (std::convertible_to<T, bool> || std::constructible_from<bool, T>))
constexpr bool vall(vec<N, T> const &t) {
    bool ret = true;
    for (size_t i = 0; i < N; i++) {
        ret = ret && (bool)t[i];
    }
    return ret;
}

template <class T>
    requires (!vec_traits<T>::value && (std::convertible_to<T, bool> || std::constructible_from<bool, T>))
constexpr bool vall(T const &t) {
    return (bool)t;
}

template <class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto dot(T1 const &t1, T2 const &t2) {
    return t1 * t2;
}

template <size_t N, class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto dot(vec<N, T1> const &t1, vec<N, T2> const &t2) {
    std::remove_cvref_t<decltype(t1[0] * t2[0])> res(0);
    for (size_t i = 0; i < N; i++) {
        res += t1[i] * t2[i];
    }
    return res;
}

template <size_t N, class T>
    requires (!vec_traits<T>::value)
constexpr auto length(vec<N, T> const &t) {
    T res(0);
    for (size_t i = 0; i < N; i++) {
        res += t[i] * t[i];
    }
    return (T)std::sqrt(res);
}

template <size_t N, class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto distance(vec<N, T1> const &t1, vec<N, T2> const &t2) {
    return length(t2 - t1);
}

template <size_t N, class T>
    requires (!vec_traits<T>::value)
constexpr auto normalize(vec<N, T> const &t) {
    return t * (T(1) / length(t));
}

template <class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto cross(vec<2, T1> const &t1, vec<2, T2> const &t2) {
    return t1[0] * t2[1] - t2[0] * t1[1];
}

template <class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto cross(vec<3, T1> const &t1, vec<3, T2> const &t2) {
    return vec<3, std::remove_cvref_t<decltype(t1[0] * t2[0])>>(
      t1[1] * t2[2] - t2[1] * t1[2],
      t1[2] * t2[0] - t2[2] * t1[0],
      t1[0] * t2[1] - t2[0] * t1[1]);
}

template <size_t N, class T>
    requires (!vec_traits<T>::value)
constexpr auto tovec(T const &t) {
  return vec<N, T>(t);
}

template <size_t N, class T>
    requires (!vec_traits<T>::value)
constexpr auto tovec(vec<N, T> const &t) {
    return t;
}


/* common type aliases */

using vec2f = vec<2, float>;
using vec2d = vec<2, double>;
using vec2i = vec<2, int32_t>;
using vec2l = vec<2, intptr_t>;
using vec2h = vec<2, int16_t>;
using vec2c = vec<2, int8_t>;
using vec2b = vec<2, bool>;
using vec2I = vec<2, uint32_t>;
using vec2L = vec<2, uintptr_t>;
using vec2Q = vec<2, uint64_t>;
using vec2H = vec<2, uint16_t>;
using vec2C = vec<2, uint8_t>;
using vec3f = vec<3, float>;
using vec3d = vec<3, double>;
using vec3i = vec<3, int32_t>;
using vec3l = vec<3, intptr_t>;
using vec3h = vec<3, int16_t>;
using vec3c = vec<3, int8_t>;
using vec3b = vec<3, bool>;
using vec3I = vec<3, uint32_t>;
using vec3L = vec<3, uintptr_t>;
using vec3Q = vec<3, uint64_t>;
using vec3H = vec<3, uint16_t>;
using vec3C = vec<3, uint8_t>;
using vec4f = vec<4, float>;
using vec4d = vec<4, double>;
using vec4i = vec<4, int32_t>;
using vec4l = vec<4, intptr_t>;
using vec4h = vec<4, int16_t>;
using vec4c = vec<4, int8_t>;
using vec4b = vec<4, bool>;
using vec4I = vec<4, uint32_t>;
using vec4L = vec<4, uintptr_t>;
using vec4Q = vec<4, uint64_t>;
using vec4H = vec<4, uint16_t>;
using vec4C = vec<4, uint8_t>;


}
}
