#pragma once

#include <gtest/gtest.h>
#include <zeno/math/vec.h>
#include <zeno/ztd/type_info.h>
#include <zeno/ztd/type_traits.h>

ZENO_NAMESPACE_BEGIN
namespace math {

template <size_t N, class T, class Stream>
    requires (ztd::tuple_contains<Stream, std::tuple<std::stringstream, std::ostream>>::value)
Stream &operator<<(Stream &os, math::vec<N, T> const &v) {
    os << "math::vec<" << N << ", " << ztd::cpp_type_name(typeid(T)) << ">(";
    os << v[0];
    for (int i = 1; i < N; i++) {
        os << ", " << v[i];
    }
    os << ")";
    return os;
}

#define SHOW_VAR(x, ...) #x " = " << (x)

}
ZENO_NAMESPACE_END
