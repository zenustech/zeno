#pragma once

#include <zeno/core/IObject.h>
#include <functional>
#include <memory>
#include <map>

namespace zeno {

struct FunctionObject : IObjectClone<FunctionObject> {
    using DictType = std::map<std::string, zany>;
    std::function<DictType(DictType const &)> func;

    FunctionObject() = default;
    explicit FunctionObject(std::function<DictType(DictType const &)> func_) : func(func_) {
    }

    inline DictType call(DictType const &args) {
        return func(args);
    }
};

}
