#pragma once


#include <type_traits>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <map>


namespace z2::dop {


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
Descriptor &desc_of(std::string const &kind);
std::vector<std::string> desc_names();


template <class T>
int define(std::string const &kind, Descriptor desc) {
    static_assert(std::is_base_of_v<Node, T>);
    define(kind, std::move(desc), std::make_unique<T>);
    return 1;
}


#define Z2_DOP_DEFINE(T, ...) static int def##T = ::z2::dop::define<T>(#T, __VA_ARGS__)


}
