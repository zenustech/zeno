#include <zeno/dop/Descriptor.h>
#include <zeno/ztd/map.h>


namespace zeno::dop {


ztd::map<std::string, Descriptor> &desc_table() {
    static ztd::map<std::string, Descriptor> impl;
    return impl;
}


Descriptor &desc_of(std::string const &kind) {
    return desc_table().at(kind);
}



void define(std::string const &kind, Descriptor desc, Descriptor::FactoryFunc factory) {
    desc.factory = std::move(factory);
    desc.inputs.push_back({"SRC"});
    desc.outputs.push_back({"DST"});
    desc_table().emplace(kind, std::move(desc));
}


}
