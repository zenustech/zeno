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

    using FactoryFunc = std::function<std::unique_ptr<Node>()>;
    FactoryFunc factory;

    std::string name;

    std::unique_ptr<Node> create() const;
};


Descriptor make_desc(std::string const &kind, Descriptor desc, Descriptor::FactoryFunc factory);
/*ztd::map<std::string, Descriptor> &desc_table();
Descriptor &desc_of(std::string const &kind);*/


template <class T>
Descriptor make_desc(std::string const &kind, Descriptor desc) {
    static_assert(std::is_base_of_v<Node, T>);
    return make_desc(kind, std::move(desc), std::make_unique<T>);
}


#define ZENO_DOP_DEFINE(T, ...) static Descriptor desc##T = ZENO_NAMESPACE::dop::make_desc<T>(#T, __VA_ARGS__)


}
ZENO_NAMESPACE_END
