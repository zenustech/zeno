#include "DopNode.h"
#include "DopContext.h"
#include "DopVisited.h"
#include "DopGraph.h"
#include "DopTable.h"


void DopNode::apply_func(DopVisited *visited) {
    DopContext ctx;
    bool valid = visited->is_visited(this);
    for (auto const &input: inputs) {
        auto val = graph->resolve_value(visited, input.value, &valid);
        ctx.in.push_back(std::move(val));
    }
    if (valid)
        return;

    ctx.out.resize(outputs.size());
    auto func = tab.lookup(kind);
    func(&ctx);

    for (int i = 0; i < ctx.out.size(); i++) {
        outputs[i].result = std::move(ctx.out[i]);
    }
    visited->mark_visited(this);
}


std::any DopNode::get_output_by_name(DopVisited *visited, std::string name) {
    apply_func(visited);
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
