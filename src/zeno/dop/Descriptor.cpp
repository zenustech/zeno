#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Node.h>
#include <zeno/ztd/map.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::map<std::string, Descriptor> &descriptor_table() {
    static ztd::map<std::string, Descriptor> impl;
    return impl;
}


ztd::map<std::string, OverloadDesc> &overloads_table() {
    static ztd::map<std::string, OverloadDesc> impl;
    return impl;
}


std::unique_ptr<Node> Descriptor::create(std::string const &sig) const {
    auto const &overload = overloads_table().at(this->name);
    auto const &factory = overload.factories.at(sig);
    auto node = factory();
    node->desc = const_cast<Descriptor *>(this);
    node->inputs.resize(this->inputs.size());
    node->outputs.resize(this->outputs.size());
    return node;
}


void define(std::string const &kind, Descriptor desc) {
    desc.name = kind;
    desc.inputs.push_back({"SRC"});
    desc.outputs.push_back({"DST"});
    bool success = descriptor_table().emplace(kind, std::move(desc)).second;
    [[unlikely]] if (!success)
        printf("dop::define: redefined descriptor: kind=[%s]\n", kind.c_str());
}


void overload(std::string const &kind, std::string const &sig, FactoryFunctor const &fac)
{
    bool success = overloads_table()[kind].factories.emplace(sig, fac).second;
    [[unlikely]] if (!success)
        printf("dop::define: redefined overload: kind=[%s], sig=[%s]\n", kind.c_str(), sig.c_str());
}


}
ZENO_NAMESPACE_END
