#pragma once


#include "stdafx.h"


struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;
    bool applied = false;

    void apply_func();

    std::any get_output_by_name(std::string name) {
        apply_func();
        for (int i = 0; i < outputs.size(); i++) {
            if (outputs[i].name == name)
                return outputs[i].result;
        }
        throw ztd::makeException("Bad output socket name: ", name);
    }

    void invalidate() {
        applied = false;
    }

    void serialize(std::ostream &ss) const {
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
};
