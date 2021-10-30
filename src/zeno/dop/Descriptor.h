#pragma once


#include <type_traits>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <zeno/ztd/map.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct Node;


struct Descriptor {
    struct SocketInfo {
        std::string name;
    };

    struct CategoryInfo {
        std::string category;
        std::string documentation;
    };

    CategoryInfo cate;
    std::vector<SocketInfo> inputs;
    std::vector<SocketInfo> outputs;

    std::string name;

    std::unique_ptr<Node> create(std::string const &) const;
};


using FactoryFunctor = std::function<std::unique_ptr<Node>()>;


struct OverloadDesc {
    ztd::map<std::string, FactoryFunctor> factories;
};


void define(std::string const &kind, Descriptor desc);
void overload(std::string const &kind, std::string const &sig, FactoryFunctor const &fac);
ztd::map<std::string, Descriptor> &descriptor_table();
ztd::map<std::string, OverloadDesc> &overloads_table();


#define ZENO_DOP_DEFINE(name, ...) \
    static int _zeno_dop_define_##name = (ZENO_NAMESPACE::dop::define(#name, __VA_ARGS__), 1)
#define ZENO_DOP_OVERLOAD(name, Class) \
    static int _zeno_dop_overload_##name = (ZENO_NAMESPACE::dop::overload(#Class, desc), 1)


}
ZENO_NAMESPACE_END
