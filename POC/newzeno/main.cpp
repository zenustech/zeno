#include "Backend.h"
#include "Frontend.h"


void myadd(Context *ctx) {
    auto x = std::any_cast<int>(ctx->inputs[0]);
    auto y = std::any_cast<int>(ctx->inputs[1]);
    auto z = x + y;
    ctx->outputs[0] = z;
}
namespace { auto _ = Session::get().defineNode("myadd", myadd); }

void makeint(Context *ctx) {
    ctx->outputs[0] = 21;
}

void printint(Context *ctx) {
    auto x = std::any_cast<int>(ctx->inputs[0]);
    std::cout << "printint: " << x << std::endl;
}


int main() {
    Graph graph;
    graph.nodes.push_back({"makeint", {}, 1});
    graph.nodes.push_back({"myadd", {{0, 0}, {0, 0}}, 1});
    graph.nodes.push_back({"printint", {{1, 0}}, 0});

    ForwardSorter sorter(graph);
    sorter.touch(2);
    auto ir = sorter.linearize();
    for (auto const &invo: ir->invos) {
        print_invocation(invo);
    }

    auto scope = Session::get().makeScope();
    for (auto const &invo: ir->invos) {
        invo.invoke(scope.get());
    }

    return 0;
}
