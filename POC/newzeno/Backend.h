#pragma once

#include <functional>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <set>
#include <any>



struct Context {
    std::vector<std::any> inputs;
    std::vector<std::any> outputs;
};


struct Session;
struct Scope {
    Session *session;
    std::map<int, std::any> objects;

    Scope(Session *session) : session(session) {
    }
};


struct Session {
    std::map<std::string, std::function<void(Context *)>> nodes;

    static Session &get() {
        static std::unique_ptr<Session> session;
        if (!session) {
            session = std::make_unique<Session>();
        }
        return *session;
    }

    template <class F>
    int defineNode(std::string const &name, F func) {
        nodes[name] = func;
        return 1;
    }

    std::unique_ptr<Scope> makeScope() const {
        return std::make_unique<Scope>(const_cast<Session *>(this));
    }
};



struct Invocation {
    std::string node_name;
    std::vector<int> inputs;
    std::vector<int> outputs;

    void invoke(Scope *scope) const {
        auto const &node = scope->session->nodes.at(node_name);
        Context ctx;
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
};


struct IRBlock {
    std::vector<Invocation> invos;
};
