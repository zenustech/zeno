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


using NodeFactory = std::function<std::unique_ptr<Node>()>;


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
    NodeFactory factory;

    std::unique_ptr<Node> create() const;
};


void add_descriptor(std::string const &kind, NodeFactory fac, Descriptor desc);
ztd::map<std::string, Descriptor> &descriptor_table();


#define ZENO_DOP_DEFINE(name, ...) \
    static int _zeno_dop_define_##name = (ZENO_NAMESPACE::dop::add_descriptor(#name, std::make_unique<Class>, __VA_ARGS__), 1);


}
ZENO_NAMESPACE_END
