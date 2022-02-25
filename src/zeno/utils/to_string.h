
#include <string_view>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <iomanip>
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
         std::size_t ...Is, decltype((to_stream(std::declval<Os &>(), std::get<0>(std::declval<T const &>()), {}),
         ..., to_stream(std::declval<Os &>(), std::get<Is + 1>(std::declval<T const &>()), {})), 0) = 0>
void __helper_tuple_to_stream(Os &os, T const &t, std::string_view &&fms, std::index_sequence<Is...>) {
    os << '(';
    (to_stream(os, std::get<0>(t), fms), ..., (os << ' ', to_stream(os, std::get<Is + 1>(t), fms)));
    os << ')';
}

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && (std::tuple_size<T>::value >= 1), int> = 0,
         decltype(__helper_tuple_to_stream(std::declval<T const &>(), std::make_index_sequence<std::tuple_size_v<T>>{}), 0) = 0>
void to_stream(Os &os, T const &t, std::string_view fms) {
    return __helper_tuple_to_stream(os, t, fms, std::make_index_sequence<std::tuple_size_v<T> - 1>{});
}

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && (std::tuple_size<T>::value == 0), int> = 0>
void to_stream(Os &os, T const &t, std::string_view fms) {
    os << "()";
}

template <class Os, class T, std::enable_if_t<std::is_invocable_v<__stream_bit_shl, Os &, T const &>, int> = 0>
void to_stream(Os &os, T const &t, std::string_view fms) {
    auto flgs = os.flags();
    if (fms.size() != 0) {
        if (fms.size() != 0 && fms[0] == '-') {
            fms = fms.substr(1);
            os << std::right;
        }
        if (fms.size() != 0 && fms[0] == '0') {
            fms = fms.substr(1);
            os << std::setfill('0');
        }
        int tmp = 0;
        while (fms.size() != 0 && '0' <= fms[0] && '9' >= fms[0]) {
            tmp *= 10;
            tmp += fms[0] - '0';
            fms = fms.substr(1);
        }
        if (tmp != 0)
            os << std::setw(tmp);
        if (fms.size() != 0) {
            switch (fms[0]) {
            case 'x': os << std::hex; break;
            case 'd': os << std::dec; break;
            case 'o': os << std::oct; break;
            };
        }
    }
    os << t;
    os.flags(flgs);
}

template <class Os, class T, std::enable_if_t<!std::is_invocable_v<__stream_bit_shl, Os &, T const &> && std::is_enum_v<T>
          && std::is_invocable_v<__stream_bit_shl, Os &, typename std::underlying_type<T>::type const &>, int> = 0>
void to_stream(Os &os, T const &t, std::string_view fms) {
#ifdef ZENO_ENABLE_MAGICENUM
    os << magic_enum::enum_name(t);
#else
    to_stream(os, std::underlying_type_t<T>{t});
#endif
}

template <class T>
std::string to_string(T const &t, std::string_view fms) {
    std::ostringstream ss;
    to_stream(ss, t, fms);
    return ss.str();
}

template <class T, std::enable_if_t<!std::is_convertible_v<T, std::string>, int> = 0>
std::string to_string(T const &t) {
    std::ostringstream ss;
    to_stream(ss, t, {});
    return ss.str();
}

template <class T, std::enable_if_t<std::is_convertible_v<T, std::string>, int> = 0>
std::string to_string(T const &t) {
    return t;
}

}
