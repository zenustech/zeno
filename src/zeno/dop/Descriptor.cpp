#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Node.h>
#include <zeno/ztd/map.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


Descriptor::Descriptor() = default;
Descriptor::~Descriptor() = default;
Descriptor::Descriptor(Descriptor const &) = default;
Descriptor::Descriptor(Descriptor &&) = default;
Descriptor &Descriptor::operator=(Descriptor const &) = default;
Descriptor &Descriptor::operator=(Descriptor &&) = default;


std::unique_ptr<Node> Descriptor::create() const {
    auto node = factory();
    node->desc = this;
    node->inputs.resize(inputs.size());
    node->outputs.resize(outputs.size());
    return node;
}


ztd::map<std::string, Descriptor> &desc_table() {
    static ztd::map<std::string, Descriptor> impl;
    return impl;
}


Descriptor &desc_of(std::string const &kind) {
    return desc_table().at(kind);
}


void define(std::string const &kind, Descriptor desc, Descriptor::FactoryFunc factory) {
    desc.name = kind;
    desc.factory = std::move(factory);
    desc.inputs.push_back({"SRC"});
    desc.outputs.push_back({"DST"});
    desc_table().emplace(kind, std::move(desc));
}


}
ZENO_NAMESPACE_END
