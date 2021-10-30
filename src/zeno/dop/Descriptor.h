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

    std::unique_ptr<Node> create(std::string const &sig) const;
};


using FactoryFunctor = std::function<std::unique_ptr<Node>()>;


struct Overloading {
    ztd::map<std::string, FactoryFunctor> factories;

    std::unique_ptr<Node> create(std::string const &sig) const;
};


void add_descriptor(std::string const &kind, Descriptor desc);
void add_overloading(std::string const &kind, std::string const &sig, FactoryFunctor const &fac);
ztd::map<std::string, Descriptor> &descriptor_table();
ztd::map<std::string, Overloading> &overloading_table();


#define ZENO_DOP_DESCRIPTOR(name, ...) \
    static int _zeno_dop_define_##name = (ZENO_NAMESPACE::dop::add_descriptor(#name, __VA_ARGS__), 1);
#define ZENO_DOP_OVERLOADING(name, sig, Class) \
    static int _zeno_dop_overload_##name = (ZENO_NAMESPACE::dop::add_overloading(#name, sig, std::make_unique<Class>), 1);
#define ZENO_DOP_DEFINE(name, ...) \
    ZENO_DOP_DESCRIPTOR(name, __VA_ARGS__) \
    ZENO_DOP_OVERLOADING(name, "", name)


}
ZENO_NAMESPACE_END
