#include "DopNode.h"
#include "DopFunctor.h"
#include "DopGraph.h"
#include "DopTable.h"


std::any DopNode::get_input(int i, std::set<std::string> &visited) {
    return graph->resolve_value(inputs[i].value, visited);
}


void DopNode::set_output(int i, std::any val) {
    outputs.at(i).result = std::move(val);
}


void DopNode::_apply_func(std::set<std::string> &visited) {
    auto func = tab.lookup(kind);
    func(this, visited);
}


std::any DopNode::get_output_by_name(std::string name, std::set<std::string> &visited) {
    int n = -1;
    for (int i = 0; i < outputs.size(); i++) {
        if (outputs[i].name == name)
            n = i;
    }
    if (n == -1)
        throw ztd::makeException("Bad output socket name: ", name);

    _apply_func(visited);
    visited.insert(this->name);

    auto val = outputs[n].result;
    if (!val.has_value()) {
        throw ztd::makeException("no value returned at socket: ", name);
    }
    return val;
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


void DopNode::invalidate() {
    node_changed = true;
}
