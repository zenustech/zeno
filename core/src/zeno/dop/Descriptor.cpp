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
    //node->inputs.resize(inputs.size());
    //node->outputs.resize(outputs.size());
    return node;
}


void add_descriptor(const char *kind, NodeFactory fac, Descriptor desc) noexcept {
    desc.name = kind;
    desc.factory = std::move(fac);
    desc.inputs.push_back({"SRC"});
    desc.outputs.push_back({"DST"});
    bool success = descriptor_table().emplace(kind, std::move(desc)).second;
    [[unlikely]] if (!success)
        printf("[zeno-init] dop::define: redefined descriptor: kind=[%s]\n", kind);
}


}
ZENO_NAMESPACE_END
