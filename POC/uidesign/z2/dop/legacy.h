#include <z2/dop/LegacyNode.h>
#include <z2/dop/Descriptor.h>
#include <z2/dop/execute.h>


namespace z2::dop::legacy {


std::any INode::get_input2(std::string const &name) const {
    for (int i = 0; i < desc->inputs.size(); i++) {
        if (desc->inputs[i].name == name) {
            return get_input(i);
        }
    }
    return {};
}


void INode::set_output2(std::string const &name, std::any &&val) {
    for (int i = 0; i < desc->inputs.size(); i++) {
        if (desc->outputs[i].name == name) {
            return set_output(i, std::move(val));
        }
    }
}


}
