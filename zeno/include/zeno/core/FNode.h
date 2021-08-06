#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/any.h>
#include <functional>
#include <memory>
#include <string>
#include <set>
#include <map>

namespace zeno {

struct Context2 {
    std::map<std::string, any> inputs;
    std::map<std::string, any> outputs;
};

struct Descriptor2 {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::tuple<std::string, std::string, std::string>> params;
    std::string category;
};

struct Codebase {
    std::map<std::string, std::function<void(Context2 *)>> functions;
    std::map<std::string, Descriptor2> descriptors;
};

struct Scope {
    std::map<std::string, any> objects;
};

struct IOperation {
    virtual void apply(Scope *scope) const = 0;
    virtual ~IOperation() = default;
};

struct OpLoadValue : IOperation {
    std::string output_ref;
    any value;

    virtual void apply(Scope *scope) const override {
        scope->objects[output_ref] = value;
    }
};

struct OpCallNode : IOperation {
    std::function<void(Context2 *>)> functor;
    std::map<std::string, std::string> input_refs;
    std::map<std::string, std::string> output_refs;

    virtual void apply(Scope *scope) const override {
        for (auto const &ref: input_refs) {
            ctx.inputs[ref] = scope->objects.at(ref);
        }
        functor(ctx);
        for (auto const &ref: output_refs) {
            scope->objects[ref] = ctx.outputs.at(ref);
        }
    }
};

struct OpsBuilder {
    Codebase *codebase;
    std::vector<std::unique_ptr<IOperation>> ops;

    addCallNode
        (
        , std::string const &func_name
        , std::map<std::string, std::string> input_bounds
        ) {
            auto const &func = safe_at(
                    codebase.functions, func_name, "function");
            auto const &desc = safe_at(
                    codebase.descriptors, func_name, "descriptor");
            auto op = std::make_unique<OpCallNode>();
            op->functor = func;
            for (auto key: desc.inputs) {
                auto it = input_bounds.find(key);
                if (it != input_bounds.end())
                    ops.input_refs[key] = it->second;
                else
                    ops.input_refs[key] = "";
            }
            for (auto key: desc.outputs) {
                auto id = node
                ops.output_refs[key] = id;
            }
    }
};

}
