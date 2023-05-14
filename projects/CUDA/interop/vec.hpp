#include <variant>
#include <zeno/core/IObject.h>
#include "zensim/math/Vec.h"

namespace zs {
#define DEFINE_VEC(dim, type, tail) using vec##dim##tail = zs::vec<type, dim>;
#define DEFINE_MAT(dim_x, dim_y, type, tail) using mat##dim_x##dim_y##tail = zs::vec<type, dim_x, dim_y>;
#define DEFINE_MAT_WITH_ROWS(rows, type, tail) \
    DEFINE_MAT(rows, 1, type, tail)            \
    DEFINE_MAT(rows, 2, type, tail) DEFINE_MAT(rows, 3, type, tail) DEFINE_MAT(rows, 4, type, tail)
#define DEFINE_SQR_MAT(dim, type, tail) using mat##dim##tail = zs::vec<type, dim, dim>;
#define DEFINE_MAT_VEC_TYPE(type, tail)                                                                       \
    DEFINE_VEC(1, type, tail)                                                                                 \
    DEFINE_VEC(2, type, tail)                                                                                 \
    DEFINE_VEC(3, type, tail)                                                                                 \
    DEFINE_VEC(4, type, tail) DEFINE_MAT_WITH_ROWS(1, type, tail) DEFINE_MAT_WITH_ROWS(2, type, tail)         \
        DEFINE_MAT_WITH_ROWS(3, type, tail) DEFINE_MAT_WITH_ROWS(4, type, tail) DEFINE_SQR_MAT(1, type, tail) \
            DEFINE_SQR_MAT(2, type, tail) DEFINE_SQR_MAT(3, type, tail) DEFINE_SQR_MAT(4, type, tail)
#define LIST_MAT_VEC_TYPE(tail)                                                                                 \
    mat11##tail, mat12##tail, mat13##tail, mat14##tail, mat21##tail, mat22##tail, mat23##tail, mat24##tail,     \
        mat31##tail, mat32##tail, mat33##tail, mat34##tail, mat41##tail, mat42##tail, mat43##tail, mat44##tail, \
        vec1##tail, vec2##tail, vec3##tail, vec4##tail

DEFINE_MAT_VEC_TYPE(int, i)
DEFINE_MAT_VEC_TYPE(float, f)
DEFINE_MAT_VEC_TYPE(double, d)

using SmallVecValue =
    std::variant<LIST_MAT_VEC_TYPE(i), LIST_MAT_VEC_TYPE(f), LIST_MAT_VEC_TYPE(d), int, float, double>;
} // namespace zs

namespace zeno {
// reference: NumericObject.h by archibate
struct SmallVecObject : IObjectClone<SmallVecObject> {
    zs::SmallVecValue value;

    SmallVecObject() = default;
    SmallVecObject(zs::SmallVecValue const &value) : value(value) {
    }

    zs::SmallVecValue &get() {
        return value;
    }

    zs::SmallVecValue const &get() const {
        return value;
    }

    template <class T>
    T get() const {
        return std::visit(
            [](auto const &val) -> T {
                using V = std::decay_t<decltype(val)>;
                if constexpr (!std::is_constructible_v<T, V>) {
                    throw makeError<TypeError>(typeid(T), typeid(V), "return::get<T>");
                } else {
                    return T(val);
                }
            },
            value);
    }

    template <class T>
    bool is() const {
        return std::holds_alternative<T>(value);
    }

    template <class T>
    void set(T const &x) {
        value = x;
    }
};
} // namespace zeno