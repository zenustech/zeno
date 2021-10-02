#include "DopNode.h"
#include "DopContext.h"
#include "DopGraph.h"
#include "DopTable.h"


void DopNode::apply_func() {
    ztd::Vector<DopLazy> in(inputs.size());

    for (int i = 0; i < in.size(); i++) {
        in[i] = graph->resolve_value(inputs[i].value);
    }

    auto func = tab.lookup(kind);
    ztd::Vector<DopLazy> out(outputs.size());
    func(in, out);

    for (int i = 0; i < out.size(); i++) {
        outputs.at(i).result = std::move(out[i]);
    }
}


DopLazy DopNode::get_output_by_name(std::string name) {
    apply_func();
    for (int i = 0; i < outputs.size(); i++) {
        if (outputs[i].name == name)
            return outputs[i].result;
    }
    throw ztd::makeException("Bad output socket name: ", name);
}


void DopNode::serialize(std::ostream &ss) const {
    ss << "DopNode[" << '\n';
    ss << "  name=" << name << '\n';
    ss << "  kind=" << kind << '\n';
    ss << "  inputs=[" << '\n';
    for (auto const &input: inputs) {
        ss << "    ";
        input.serialize(ss);
        ss << '\n';
    }
    ss << "  ]" << '\n';
    ss << "  outputs=[" << '\n';
    for (auto const &output: outputs) {
        ss << "    ";
        output.serialize(ss);
        ss << '\n';
    }
    ss << "  ]" << '\n';
    ss << "]" << '\n';
}
