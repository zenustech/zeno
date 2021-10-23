#pragma once


#include <type_traits>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <zs/ztd/map.h>


namespace zeno::dop {


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
};


void define(std::string const &kind, Descriptor desc, Descriptor::FactoryFunc factory);
ztd::map<std::string, Descriptor> &desc_table();
Descriptor &desc_of(std::string const &kind);


template <class T>
int define(std::string const &kind, Descriptor desc) {
    static_assert(std::is_base_of_v<Node, T>);
    define(kind, std::move(desc), std::make_unique<T>);
    return 1;
}


#define ZENO_DOP_DEFINE(T, ...) static int def##T = ::zeno::dop::define<T>(#T, __VA_ARGS__)


}
