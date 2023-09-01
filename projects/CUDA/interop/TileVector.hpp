#pragma once
#include <variant>
#include <zeno/core/IObject.h>
#include "zs_object.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/math/Vec.h"

namespace zs
{
    template <class T, size_t Length, bool isVirtual>
    using tv_value_t = zs::TileVector<T, Length, zs::ZSPmrAllocator<isVirtual>>;
#define LIST_TILEVECTOR(LENGTH) tv_value_t<int, LENGTH, true>, tv_value_t<int, LENGTH, false>,     \
                                tv_value_t<float, LENGTH, true>, tv_value_t<float, LENGTH, false>, \
                                tv_value_t<double, LENGTH, true>, tv_value_t<double, LENGTH, false>

    using TileVectorValue = std::variant<LIST_TILEVECTOR(8), LIST_TILEVECTOR(32),
                                         LIST_TILEVECTOR(64), LIST_TILEVECTOR(512)>;
}

namespace zeno
{
    // C API: create & get data
    using ZsTileVectorObject = ZsObject<zs::TileVectorValue>;
}