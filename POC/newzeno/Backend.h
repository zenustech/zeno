#pragma once

#include "common.h"


namespace zeno::v2::backend {

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


struct NodeInfo {
    std::vector<std::string> outputs;
    std::vector<std::string> inputs;
};


struct Session {
    std::map<std::string, std::function<void(Context *)>> nodes;
    std::map<std::string, NodeInfo> nodeinfo;

    static Session &get() {
        static std::unique_ptr<Session> session;
        if (!session) {
            session = std::make_unique<Session>();
        }
        return *session;
    }

    template <class F>
    int defineNode
        ( std::string const &name
        , F func
        , NodeInfo const &info
        ) {
        nodes[name] = func;
        nodeinfo[name] = info;
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

}

namespace zeno::v2::helpers {

template <class Os>
void print_invocation(Os &&os, backend::Invocation const &invo) {
    os << "[";
    bool had = false;
    for (auto const &output: invo.outputs) {
        if (had) os << ", ";
        else had = true;
        os << output;
    }
    os << "] = ";
    os << invo.node_name;
    os << "(";
    had = false;
    for (auto const &input: invo.inputs) {
        if (had) os << ", ";
        else had = true;
        os << input;
    }
    os << ");\n";
}

}
