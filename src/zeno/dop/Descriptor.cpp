#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Node.h>


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


static int match_signature(Signature const &lhs, Signature const &rhs) {
    int score = 0;
    for (int i = 0; i < std::min(lhs.size(), rhs.size()); i++) {
        if (rhs[i] == std::type_index(typeid(void))) {
            continue;
        }
        if (lhs[i] == rhs[i]) {
            score += 1;
        }
    }
    return score;
}


std::unique_ptr<Node> Overloading::create(Signature const &sig) const {
    std::vector<std::reference_wrapper<FactoryFunctor const>> matches;
    for (auto const &[key, factory]: factories) {
        if (int prio = match_signature(sig, key); prio != -1) {
            if (matches.size() < prio + 1) {
                matches.resize(prio + 1);
            }
            matches[prio] = std::cref(factory);
        }
    }
    if (matches.empty())
        throw ztd::error("no suitable overloading found");
    auto const &factory = matches[0].get();
    return factory();
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
        printf("[zeno] dop::define: redefined overload: kind=[%s]\n", kind.c_str());
}


}
ZENO_NAMESPACE_END
