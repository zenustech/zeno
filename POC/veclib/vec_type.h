#pragma once


#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <algorithm>


namespace zeno::ztd {
inline namespace math {


template <size_t N, class T>
    requires(N >= 1 && std::copy_constructible<T> && std::default_initializable<T>)
struct vec {
    T _M_data[N]{};

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
    constexpr bool operator==(vec const &) const = default;
    constexpr bool operator<=>(vec const &) const = default;

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


}
}
