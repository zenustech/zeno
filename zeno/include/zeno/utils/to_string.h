
#include <string_view>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <iomanip>
#include <iterator>
#ifdef ZENO_ENABLE_MAGICENUM
#include <magic_enum.hpp>
#endif

namespace zeno {

namespace _to_stream_details {

template <class T0, class T1, class = void>
struct _has_stream_bit_shl : std::false_type {};

template <class T0, class T1>
struct _has_stream_bit_shl<T0, T1, std::void_t<decltype(
    std::declval<T0>() << std::declval<T1>()
    )>> : std::true_type {};

template <class T, class = void>
struct _has_range_begin_end : std::false_type {};

template <class T>
struct _has_range_begin_end<T, std::void_t<decltype(
    std::begin(std::declval<T>()) != std::end(std::declval<T>())
    )>> : std::true_type {};

template <class T, class = void>
struct _has_tuple_size : std::false_type {};

template <class T>
struct _has_tuple_size<T, std::void_t<decltype(
    std::tuple_size<T>::value
    )>> : std::true_type {};

struct _to_stream_impl {
    template <class Os, class T, std::size_t ...Is>
    static void _helper_tuple_to_stream(Os &os, T const &t, std::string_view fms, std::index_sequence<Is...>) {
        os << '(';
        (to_stream(os, std::get<0>(t), fms), ..., (os << ' ', to_stream(os, std::get<Is + 1>(t), fms)));
        os << ')';
    }

    template <class Os, class T, std::enable_if_t<!_has_stream_bit_shl<Os &, T const &>::value
        && (std::tuple_size<T>::value >= 1), int> = 0>
    static void to_stream(Os &os, T const &t, std::string_view fms) {
        return _helper_tuple_to_stream(os, t, fms, std::make_index_sequence<std::tuple_size_v<T> - 1>{});
    }

    template <class Os, class T, std::enable_if_t<!_has_stream_bit_shl<Os &, T const &>::value
        && (std::tuple_size<T>::value == 0), int> = 0>
    static void to_stream(Os &os, T const &t, std::string_view fms) {
        os << "()";
    }

    template <class Os, class T, std::enable_if_t<!_has_stream_bit_shl<Os &, T const &>::value
        && !_has_tuple_size<T>::value && _has_range_begin_end<T>::value, int> = 0>
    static void to_stream(Os &os, T const &t, std::string_view fms) {
        auto it = std::begin(t);
        auto eit = std::end(t);
        os << '[';
        if (it != eit) {
            to_stream(os, *it, fms);
            ++it;
            for (; it != eit; ++it) {
                os << ' ';
                to_stream(os, *it, fms);
            }
        }
        os << ']';
    }

    template <class Os, class T, std::enable_if_t<_has_stream_bit_shl<Os &, T const &>::value && !std::is_enum<T>::value, int> = 0>
    static void to_stream(Os &os, T const &t, std::string_view fms) {
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
            {
                int tmp = 0;
                while (fms.size() != 0 && '0' <= fms[0] && '9' >= fms[0]) {
                    tmp *= 10;
                    tmp += fms[0] - '0';
                    fms = fms.substr(1);
                }
                if (tmp != 0)
                    os << std::setw(tmp);
            }
            if (fms.size() != 0 && fms[0] == '.') {
                fms = fms.substr(1);
                int tmp = 0;
                while (fms.size() != 0 && '0' <= fms[0] && '9' >= fms[0]) {
                    tmp *= 10;
                    tmp += fms[0] - '0';
                    fms = fms.substr(1);
                }
                os << std::setprecision(tmp);
            }
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

    template <class Os, class T, std::enable_if_t<std::is_enum<T>::value
              && _has_stream_bit_shl<Os &, typename std::underlying_type<T>::type const &>::value, int> = 0>
    static void to_stream(Os &os, T const &t, std::string_view fms) {
#ifdef ZENO_ENABLE_MAGICENUM
        os << magic_enum::enum_name(t);
#else
        os << std::underlying_type_t<T>{t};
#endif
    }
};

}

template <class Os, class T>
void to_stream(Os &os, T const &t, std::string_view fms) {
    _to_stream_details::_to_stream_impl::to_stream(os, t, fms);
}

template <class T>
std::string to_string(T const &t, std::string_view fms) {
    std::ostringstream ss;
    to_stream(ss, t, fms);
    return ss.str();
}

template <class T>
std::string to_string(T const &t) {
    if constexpr (std::is_convertible_v<T, std::string>) {
        return t;
    } else {
        std::ostringstream ss;
        to_stream(ss, t, {});
        return ss.str();
    }
}

}
