#pragma once

#include "Container.h"
#include "Backend.h"

namespace zeno::v2::statement {

struct Statement {
    virtual void apply(backend::Scope *scope) const = 0;
    virtual std::string to_string() const = 0;

    virtual ~Statement() = default;
};

struct IRBlock {
    std::vector<std::unique_ptr<Statement>> stmts;
    IRBlock *parent = nullptr;

    void apply(backend::Scope *scope) const {
        for (auto const &stmt: stmts) {
            stmt->apply(scope);
        }
    }

    std::string to_string() const {
        std::stringstream os;
        for (auto const &stmt: stmts) {
            os << stmt->to_string() << "\n";
        }
        return os.str();
    }
};


struct StmtIfBlock : Statement {
    std::unique_ptr<IRBlock> block_true;
    int input_cond;
    int input_true;
    int input_false;

    virtual void apply(backend::Scope *scope) const override {
        block->apply(scope);
    }

    virtual std::string to_string() const override {
        std::stringstream os;
        os << "if (" << input_cond << ") {\n";
        os << block->to_string();
        os << "}";
        return os.str();
    }
};

struct StmtValue : Statement {
    container::any value;
    int output;

    virtual void apply(backend::Scope *scope) const override {
        scope->objects[output] = value;
    }

    virtual std::string to_string() const override {
        std::stringstream os;
        os << "[" << output << "] = ";
        os << "value(" << value.type().name() << ");";
        return os.str();
    }
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

}
