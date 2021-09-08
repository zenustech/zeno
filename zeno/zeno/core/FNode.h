#pragma once

// wipdemo..

#include <zeno/utils/api.h>
#include <zeno/core/INode.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/Any.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <map>

namespace zeno {

struct Context2 {
    std::vector<zany> inputs;
    std::vector<zany> outputs;
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

    int defNodeFunction
            ( std::string const &name
            , std::function<void(Context2 *)> func
            , Descriptor2 const &desc
            ) {
        functions[name] = func;
        descriptors[name] = desc;
        return 1;
    }

    int defLegacyNodeClass
            ( std::string const &name
            , std::function<std::unique_ptr<INode>()> ctor
            , Descriptor2 const &desc
            ) {
        auto func = [ctor = std::move(ctor), desc] (Context2 *ctx) {
            auto node = ctor();
            for (int i = 0; i < desc.inputs.size(); i++) {
                auto key = desc.inputs[i];
                node->inputs[key] = ctx.inputs.at(i);
            }
            node->apply();
            for (int i = 0; i < desc.outputs.size(); i++) {
                auto key = desc.outputs[i];
                auto it = node->outputs.find(key);
                if (it != node->outputs.end())
                    ctx.outputs.at(i) = it->second;
                else
                    ctx.outputs.
            }
        };
        functions[name] = func;
        descriptors[name] = desc;
        return 1;
    }
};

extern Codebase &getCodebase();

#define ZENO_DEFNODE(func, ...) \
    static int zeno_defnode_##func = \
    getCodebase().defNodeFunction(#func, func, __VA_ARGS__)

#define ZENDEFNODE(cls, ...) \
    static int zeno_defnode_##cls = \
    getCodebase().defLegacyNodeClass(#cls, std::make_unique<cls>, __VA_ARGS__)


struct Scope {
    std::vector<zany> objects;
};

struct IOperation {
    virtual void apply(Scope *scope) const = 0;
    virtual ~IOperation() = default;
};

struct OpLoadValue : IOperation {
    int output_ref;
    zany value;

    virtual void apply(Scope *scope) const override {
        scope->objects[output_ref] = value;
    }
};

struct OpCallNode : IOperation {
    std::function<void(Context2 *>)> functor;
    std::vector<int> input_refs;
    std::vector<int> output_refs;

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
    Codebase const *codebase = nullptr;
    std::vector<std::unique_ptr<IOperation>> operations;

    setCodebase(Codebase const *codebase_) {
        codebase = codebase_;
    }

    std::map<std::pair<std::string, std::string>, int> lut;
    int lut_top_id = 0;

    int lut_at(std::string const &sn, std::string const &ss) {
        return safe_at(lut, {sn, ss}, "object");
    }

    int lut_put(std::string const &sn, std::string const &ss) {
        auto id = lut_top_id++;
        lut[std::make_pair(sn, ss)] = id;
        return id;
    }

    decltype(auto) getResult() {
        return std::move(operations);
    }

    int getObjectCount() const {
        return lut_top_id;
    }

    addLoadValue
        ( std::string const &node_ident
        , zany const &value
        ) {
        auto op = std::make_unique<OpLoadValue>();
        op.value = value;

        auto id = lut_put(node_ident, "value");
        op.output_ref = id;

        operations.push_back(std::move(op));
    }

    addCallNode
        ( std::string const &node_ident
        , std::string const &func_name
        , std::map<std::string, std::string> const &input_bounds
        , std::set<std::string> const &legacy_options = {}
        ) {
        auto const &func = safe_at(
                codebase.functions, func_name, "function");
        auto const &desc = safe_at(
                codebase.descriptors, func_name, "descriptor");

        auto op = std::make_unique<OpCallNode>();
        op->functor = func;

        op.input_refs.resize(desc.inputs.size());
        for (int i = 0; i < desc.inputs.size(); i++) {
            auto key = desc.inputs[i];
            auto it = input_bounds.find(key);
            if (it != input_bounds.end()) {
                auto [sn, ss] = it->second;
                ops.input_refs[i] = lut_at(sn, ss);
            } else {
                ops.input_refs[i] = "";
            }
        }

        op.output_refs.resize(desc.outputs.size());
        for (int i = 0; i < desc.outputs.size(); i++) {
            auto key = desc.outputs[i];
            auto id = lut_put(node_ident, key);
            op.output_refs[i] = id;
        }

        operations.push_back(std::move(op));
    }
};

}
