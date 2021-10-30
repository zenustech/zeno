#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Node.h>
#include <zeno/ztd/map.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::map<std::string, Descriptor> &descriptor_table() {
    static ztd::map<std::string, Descriptor> impl;
    return impl;
}


ztd::map<std::string, Overloading> &overloading_table() {
    static ztd::map<std::string, Overloading> impl;
    return impl;
}


std::unique_ptr<Node> Overloading::create(Signature const &sig) const {
    return factories.at(sig)();
}


std::unique_ptr<Node> Descriptor::create(Signature const &sig) const {
    auto const &overload = overloading_table().at(this->name);
    auto node = overload.create(sig);
    node->desc = const_cast<Descriptor *>(this);
    node->inputs.resize(this->inputs.size());
    node->outputs.resize(this->outputs.size());
    return node;
}


void add_descriptor(std::string const &kind, Descriptor desc) {
    desc.name = kind;
    desc.inputs.push_back({"SRC"});
    desc.outputs.push_back({"DST"});
    bool success = descriptor_table().emplace(kind, std::move(desc)).second;
    [[unlikely]] if (!success)
        printf("[zeno] dop::define: redefined descriptor: kind=[%s]\n", kind.c_str());
}


void add_overloading(std::string const &kind, Signature const &sig, FactoryFunctor const &fac)
{
    bool success = overloading_table()[kind].factories.emplace(sig, fac).second;
    [[unlikely]] if (!success)
        printf("[zeno] dop::define: redefined overload: kind=[%s], sig=[%s]\n", kind.c_str(), sig.c_str());
}


}
ZENO_NAMESPACE_END
