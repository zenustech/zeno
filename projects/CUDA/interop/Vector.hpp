#pragma once
#include <variant>
#include <zeno/core/IObject.h>
#include "zs_object.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/py_interop/VectorView.hpp"

namespace zs
{
    template <class T, bool isVirtual>
    using vec_value_t = zs::Vector<T, zs::ZSPmrAllocator<isVirtual>>;
    using VectorViewLiteValue = std::variant<
        vec_value_t<int, false>, vec_value_t<int, true>,
        vec_value_t<double, false>, vec_value_t<double, true>,
        vec_value_t<float, false>, vec_value_t<float, true>>;
}

namespace zeno
{
    // C API: create & get data
    using VectorViewLiteObject = ZsObject<zs::VectorViewLiteValue>;
}