#pragma once


#include <zeno/common.h>
#include <type_traits>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <map>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct Node;


using FactoryFunctor = std::function<std::unique_ptr<Node>()>;


struct CallSignature {
};


struct Descriptor {
    struct SocketInfo {
        std::string name;
    };

    struct CategoryInfo {
        std::string category;
        std::string documentation;
    };

    std::string name;

    CategoryInfo cate;
    std::vector<SocketInfo> inputs;
    std::vector<SocketInfo> outputs;

    std::map<CallSignature, FactoryFunctor> factories;

    Descriptor();
    ~Descriptor();
    Descriptor(Descriptor const &);
    Descriptor(Descriptor &&);
    Descriptor &operator=(Descriptor const &);
    Descriptor &operator=(Descriptor &&);

    std::unique_ptr<Node> create() const;
};


void define(std::string const &kind, Descriptor desc, FactoryFunctor factory);
ztd::map<std::string, Descriptor> &desc_table();
Descriptor &desc_of(std::string const &kind);


template <class T>
int define(std::string const &kind, Descriptor desc) {
    static_assert(std::is_base_of_v<Node, T>);
    define(kind, std::move(desc), std::make_unique<T>);
    return 1;
}


#define ZENO_DOP_DEFINE(T, ...) static int def##T = ZENO_NAMESPACE::dop::define<T>(#T, __VA_ARGS__)


}
ZENO_NAMESPACE_END
