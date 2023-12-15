#pragma once

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>//todo: wip
#include <zeno/types/ListObject.h>

namespace zeno {

template <class Target = PrimitiveObject, class Func>
static bool _cihou_list_input(IObject *obj, Func const &func) {
    if (auto lst = dynamic_cast<ListObject *>(obj)) {
        for (auto const &obj: lst->arr)
            _cihou_list_input(obj.get(), func);
    } else if (auto tgt = dynamic_cast<Target *>(obj)) {
        return func(tgt);
    } else {
        throw Exception("invalid input `"
                + (std::string)typeid(*obj).name() +
                "` to be cihoued as list of prim");
    }
}

template <class Target = PrimitiveObject, class Func>
static bool _cihou_list_input(std::shared_ptr<IObject> obj, Func const &func) {
    if (auto lst = std::dynamic_pointer_cast<ListObject>(obj)) {
        for (auto const &obj: lst->arr)
            _cihou_list_input(obj, func);
    } else if (auto tgt = std::dynamic_pointer_cast<Target>(obj)) {
        return func(tgt);
    } else {
        throw Exception("invalid input `"
                + (std::string)typeid(*obj).name() +
                "` to be cihoued as list of prim");
    }
}

}
