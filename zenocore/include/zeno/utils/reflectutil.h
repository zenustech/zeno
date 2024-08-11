#pragma once

#include <zeno/core/objectptrcontainer.h>
//#include "reflect/zenocore/zenoreflecttypes.cpp.generated.hpp"

using namespace zeno::reflect;

namespace zeno
{
    template<typename T>
    Any constructObject(std::shared_ptr<T> obj) {
        auto pContainer = new ObjectPtrContainer(obj);
        Any wtf = make_any_by_container(pContainer);
        return wtf;
    }
}