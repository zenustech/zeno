#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
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

struct Codebase {
    std::map<std::string, std::function<void(Context2 *)>> functions;
};

struct Scope {
    std::map<std::string, any> objects;
};

struct OpLoadValue : IOperation {
    std::string output_ref;
    any value;

    virtual void apply(Scope *scope) override {
        scope->objects[output_ref] = value;
    }
};

struct OpCallNode : IOperation {
    std::function<void(Context2 *>)> functor;
    std::map<std::string, std::string> input_refs;
    std::map<std::string, std::string> output_refs;

    virtual void apply(Scope *scope) override {
        for (auto const &ref: input_refs) {
            ctx.inputs[ref] = scope->objects.at(ref);
        }
        functor(ctx);
        for (auto const &ref: output_refs) {
            scope->objects[ref] = ctx.outputs.at(ref);
        }
    }
};

}
