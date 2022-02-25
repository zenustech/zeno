
#include <sstream>
#include <tuple>
#include <type_traits>
#ifdef ZENO_ENABLE_MAGICENUM
#include <magic_enum.hpp>
#endif

namespace zeno {

struct __stream_bit_shl {
    template <class T0, class T1, decltype((std::declval<T0>() << std::declval<T1>()), 0) = 0>
    constexpr decltype(auto) operator()(T0 &&t0, T1 &&t1) const {
        return std::forward<T0>(t0) << std::forward<T1>(t1);
    }
};

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && (std::tuple_size<T>::value >= 1), int> = 0,
         std::size_t ...Is, decltype((to_stream(std::declval<Os &>(), std::get<0>(std::declval<T const &>())),
         ..., to_stream(std::declval<Os &>(), std::get<Is + 1>(std::declval<T const &>()))), 0) = 0>
void __helper_tuple_to_stream(Os &os, T const &t, std::index_sequence<Is...>) {
    os << '(';
    (to_stream(os, std::get<0>(t)), ..., (os << ' ', to_stream(os, std::get<Is + 1>(t))));
    os << ')';
}

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && (std::tuple_size<T>::value >= 1), int> = 0,
         decltype(__helper_tuple_to_stream(std::declval<T const &>(), std::make_index_sequence<std::tuple_size_v<T>>{}), 0) = 0>
void to_stream(Os &os, T const &t) {
    return __helper_tuple_to_stream(os, t, std::make_index_sequence<std::tuple_size_v<T> - 1>{});
}

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && (std::tuple_size<T>::value == 0), int> = 0>
void to_stream(Os &os, T const &t) {
    os << "()";
}

template <class Os, class T, std::enable_if_t<std::is_invocable_v<__stream_bit_shl, Os &, T const &>, int> = 0>
void to_stream(Os &os, T const &t) {
    os << t;
}

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && std::is_enum_v<T>, int> = 0>
void to_stream(Os &os, T const &t) {
#ifdef ZENO_ENABLE_MAGICENUM
    os << magic_enum::enum_name(t);
#else
    os << std::underlying_type_t<T>{t};
#endif
}

template <class T, std::enable_if_t<!std::is_convertible_v<T, std::string>, int> = 0>
std::string to_string(T const &t) {
    std::ostringstream ss;
    to_stream(ss, t);
    return ss.str();
}

template <class T, std::enable_if_t<std::is_convertible_v<T, std::string>, int> = 0>
std::string to_string(T const &t) {
    return t;
}

}
