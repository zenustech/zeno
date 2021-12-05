#include <zeno/dop/OverloadNode.h>
#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Functor.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


void OverloadNode::apply() {
    FuncContext ctx;

    ctx.inputs.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
        ctx.inputs.push_back(get_input(i));
    }

    ctx.outputs.resize(outputs.size());
    overloading_table().at(desc->name).invoke(&ctx);

    for (int i = 0; i < ctx.outputs.size(); i++) {
        set_output(i, std::move(ctx.outputs[i]));
    }
}


}
ZENO_NAMESPACE_END
