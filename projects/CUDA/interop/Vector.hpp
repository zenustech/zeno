#pragma once
#include <variant>
#include <zeno/core/IObject.h>
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/py_interop/VectorView.hpp"
namespace zeno {
template <class T, bool isVirtual>
using vec_value_t = zs::Vector<T, zs::ZSPmrAllocator<isVirtual>>; 
using VectorViewLiteValue = std::variant<
vec_value_t<int, false>, vec_value_t<int, true>, 
vec_value_t<double, false>, vec_value_t<double, true>, 
vec_value_t<float, false>, vec_value_t<float, true>>; 
// C API: create & get data 
struct VectorViewLiteObject : IObjectClone<VectorViewLiteObject> {
    VectorViewLiteValue value;

    VectorViewLiteObject() = default; 
    VectorViewLiteObject(VectorViewLiteValue const &value) : value(value) {}

    VectorViewLiteValue &get() {
        return value; 
    }

    VectorViewLiteValue const &get() const {
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
}