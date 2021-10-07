#pragma once

#include <z2/ztd/vec.h>


namespace zeno {

using namespace z2::ztd::mathvec;

inline auto alltrue(auto x) {
    return vall(x);
}

inline auto anytrue(auto x) {
    return vany(x);
}

inline auto tofloat(auto x) {
    return vcast<float>(x);
}

inline auto toint(auto x) {
    return vcast<int>(x);
}

inline auto mix(auto x, auto y, auto z) {
    return lerp(x, y, z);
}

template <class T>
static constexpr auto is_vec_n = std::max((size_t)1, vec_traits<T>::dim);

template <class T>
static constexpr auto is_vec_v = vec_traits<T>::value;

template <class T>
using decay_vec_t = typename vec_traits<T>::type;

}
