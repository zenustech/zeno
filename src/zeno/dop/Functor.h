#pragma once


#include <zeno/ztd/zany.h>
#include <zeno/ztd/map.h>
#include <functional>
#include <typeindex>
#include <vector>
#include <any>
#include <map>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct FuncContext {
    std::vector<ztd::zany> inputs;
    std::vector<ztd::zany> outputs;
};


using Functor = std::function<void(FuncContext *)>;
using FuncSignature = std::vector<std::type_index>;


struct FuncOverloads {
    std::map<FuncSignature, Functor> functors;

    Functor const &overload(FuncSignature const &sig) const;
};


ztd::map<std::string, FuncOverloads> &overloading_table();
void add_overloading(std::string const &kind, FuncSignature const &sig, Functor func);


}
ZENO_NAMESPACE_END
