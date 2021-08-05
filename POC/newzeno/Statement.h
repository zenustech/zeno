#pragma once

#include "common.h"
#include "Backend.h"

namespace zeno::v2::statement {

struct Statement {
    virtual void apply(backend::Scope *scope) const = 0;
    virtual std::string to_string() const = 0;

    virtual ~Statement() = default;
};

struct StmtCall : Statement {
    std::string node_name;
    std::vector<int> inputs;
    std::vector<int> outputs;

    virtual void apply(backend::Scope *scope) const override {
        auto const &node = scope->session->nodes.at(node_name);
        backend::Context ctx;
        ctx.inputs.resize(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            ctx.inputs[i] = scope->objects.at(inputs[i]);
        }
        ctx.outputs.resize(outputs.size());
        node(&ctx);
        for (int i = 0; i < outputs.size(); i++) {
            scope->objects[outputs[i]] = ctx.outputs[i];
        }
    }

    virtual std::string to_string() const override {
        std::stringstream os;
        os << "[";
        bool had = false;
        for (auto const &output: outputs) {
            if (had) os << ", ";
            else had = true;
            os << output;
        }
        os << "] = ";
        os << node_name;
        os << "(";
        had = false;
        for (auto const &input: inputs) {
            if (had) os << ", ";
            else had = true;
            os << input;
        }
        os << ");";
        return os.str();
    }
};


struct IRBlock {
    std::vector<std::unique_ptr<Statement>> stmts;
};

}
