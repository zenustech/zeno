#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Node.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::map<std::string, Descriptor> &descriptor_table() {
    static ztd::map<std::string, Descriptor> impl;
    return impl;
}


std::unique_ptr<Node> Descriptor::create() const {
    auto node = factory();
    node->desc = const_cast<Descriptor *>(this);
    node->inputs.resize(inputs.size());
    node->outputs.resize(outputs.size());
    return node;
}


void add_descriptor(std::string const &kind, NodeFactory fac, Descriptor desc) {
    desc.name = kind;
    desc.factory = std::move(fac);
    desc.inputs.push_back({"SRC"});
    desc.outputs.push_back({"DST"});
    bool success = descriptor_table().emplace(kind, std::move(desc)).second;
    [[unlikely]] if (!success)
        printf("[zeno] dop::define: redefined descriptor: kind=[%s]\n", kind.c_str());
}


}
ZENO_NAMESPACE_END
