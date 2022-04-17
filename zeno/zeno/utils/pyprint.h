
#include <string>
#include <sstream>
#include <iostream>

namespace zeno {

template <class T0, class ...Ts>
std::string pycatstr(T0 &&t0, Ts &&...ts) {
    std::ostringstream ss;
    ss << std::forward<T0>(t0);
    if constexpr (sizeof...(Ts))
        ((std::cout << std::forward<Ts>(ts) << ' '), ...);
    return ss.str();
}

template <class T0, class ...Ts>
void pyprint(T0 &&t0, Ts &&...ts) {  // zhouhang happy print
    std::cout << std::forward<T0>(t0);
    if constexpr (sizeof...(Ts))
        ((std::cout << std::forward<Ts>(ts) << ' '), ...);
    std::cout << std::endl;
}

}
