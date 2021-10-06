#include <z2/dop/Descriptor.h>


namespace z2::dop {


static auto &desc_table() {
    static std::map<std::string, Descriptor> impl;
    return impl;
}


Descriptor &desc_of(std::string const &kind) {
    return desc_table().at(kind);
}


std::vector<std::string> desc_names() {
    std::vector<std::string> ret;
    for (auto const &[k, v]: desc_table()) {
        ret.push_back(k);
    }
    return ret;
}


void define(std::string const &kind, Descriptor &&desc, Descriptor::FactoryFunc &&factory) {
    desc.factory = factory;
    desc_table().emplace(kind, desc);
}


}
