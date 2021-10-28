#include <type_traits>
#include <string_view>
#include <sstream>
#include <string>
#include <tuple>
#include <iostream>


template <class ...Args>
using format_string = std::string_view;

template <class ...Args>
inline void _format_impl(auto &os, std::string_view fmt, size_t idx, std::tuple<Args...> const &args) {
    if constexpr (sizeof...(Args) != 0) {
        size_t beg = fmt.find('{', idx);
        size_t end = fmt.find('}', beg);
        os << fmt.substr(idx, beg - idx);
        os << std::get<0>(args);
        return _format_impl(os, fmt, end + 1, ([]<size_t ...Is> (std::tuple<Args...> const &args, std::index_sequence<Is...>) {
            return std::tuple<std::tuple_element_t<1 + Is, std::tuple<Args...>>...>(std::get<1 + Is>(args)...);
        })(args, std::make_index_sequence<sizeof...(Args) - 1>()));
    } else {
        os << fmt.substr(idx);
    }
}

template <class ...Args>
inline void format_to(auto &os, std::string_view fmt, Args &&...args) {
    return _format_impl<Args...>(os, fmt, 0, std::tuple<Args...>(std::forward<Args>(args)...));
}

template <class ...Args>
inline std::string format(std::string_view fmt, Args &&...args) {
    std::ostringstream ss;
    vformat_to<Args...>(ss, fmt, std::forward<Args>(args)...);
    return ss.str();
}


int main() {
    std::cout << format("hello, {} world", 42) << std::endl;
    std::cout << format("hello, {}, {} world", "fucking", 42) << std::endl;
    return 0;
}
