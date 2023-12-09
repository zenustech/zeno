#pragma once
#include <variant>
#include <zeno/core/IObject.h>
#include "zs_object.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/py_interop/TileVectorView.hpp"

namespace zs
{
    template <class T, bool isVirtual>
    using vec_value_t = zs::Vector<T, zs::ZSPmrAllocator<isVirtual>>;
    using VectorObject = std::variant<
        vec_value_t<int, false>, vec_value_t<int, true>,
        vec_value_t<double, false>, vec_value_t<double, true>,
        vec_value_t<float, false>, vec_value_t<float, true>>;
}

namespace zeno
{
    // C API: create & get data
    using ZsVectorObject = ZsObject<zs::VectorObject>;
}