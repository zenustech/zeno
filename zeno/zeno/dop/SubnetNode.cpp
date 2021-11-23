#include <zeno/dop/SubnetNode.h>
#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Executor.h>
#include <zeno/dop/macros.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


void SubnetNode::apply() {
    subnetIn->outputs.resize(inputs.size());
    subnetOut->inputs.resize(outputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        subnetIn->outputs[i] = get_input(i);
    }

    Executor exec;
    for (int i = 0; i < outputs.size(); i++) {
        outputs[i] = exec.evaluate({.node = subnetOut, .sockid = i});
    }

    /*inputs.resize(subins.size());
    for (size_t i = 0; i < subins.size(); i++) {
        subins[i]->inputs.at(i) = inputs[i];
    }

    Executor exec;
    outputs.resize(subouts.size());
    for (size_t i = 0; i < subouts.size(); i++) {
        outputs[i] = exec.evaluate({.node = subouts[i], .sockid = 0});
    }*/
    // TODO: FIXME: PLEASE IMPL THIS!
}

ZENO_DOP_DEFCLASS(SubnetNode, {{
    "misc", "a custom subnet to combine many nodes into one",
}, {
}, {
}});


void SubnetIn::apply() {
}


void SubnetOut::apply() {
}


}
ZENO_NAMESPACE_END
