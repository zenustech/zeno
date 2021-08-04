#include <functional>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <any>
#include "Method.h"




struct Context;


struct Session {
    std::map<std::string, std::function<void(Context &)>> nodes;
    std::map<int, std::any> objects;
} session;


struct Context {
    std::vector<std::any> inputs;
    std::vector<std::any> outputs;
};


struct Invocation {
    std::string node_name;
    std::vector<int> inputs;
    std::vector<int> outputs;

    void operator()() {
        auto &node = session.nodes.at(node_name);
        Context ctx;
        ctx.inputs.resize(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            ctx.inputs[i] = session.objects.at(inputs[i]);
        }
        ctx.outputs.resize(outputs.size());
        node(ctx);
        for (int i = 0; i < outputs.size(); i++) {
            session.objects[outputs[i]] = ctx.outputs[i];
        }
    }
};


void myadd(Context &ctx) {
    auto x = std::any_cast<int>(ctx.inputs[0]);
    auto y = std::any_cast<int>(ctx.inputs[1]);
    auto z = x + y;
    ctx.outputs[0] = z;
}


int main() {
    session.nodes["myadd"] = myadd;
    session.objects[0] = 40;
    session.objects[1] = 2;
    Invocation{"myadd", {0, 1}, {2}}();
    std::cout << std::any_cast<int>(session.objects.at(2)) << std::endl;
}
