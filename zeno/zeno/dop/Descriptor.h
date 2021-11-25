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
        std::string type;
        std::string defl;
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


void add_descriptor(const char *kind, NodeFactory fac, Descriptor desc) noexcept;
ztd::map<std::string, Descriptor> &descriptor_table();


}
ZENO_NAMESPACE_END
