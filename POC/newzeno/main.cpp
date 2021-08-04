#include "Backend.h"
#include "Frontend.h"
#include "Helper.h"
#include <iostream>


int myadd(int x, int y) {
    auto z = x + y;
    return z;
}
ZENO_DEFINE_NODE(myadd, {"ret"}, {"x", "y"});

int makeint() {
    return 21;
}
ZENO_DEFINE_NODE(makeint, {"value"}, {});

void printint(int x) {
    std::cout << "printint: " << x << std::endl;
}
ZENO_DEFINE_NODE(printint, {}, {});


int main() {
    zeno::v2::frontend::Graph graph;
    graph.nodes.push_back({"makeint", {}, 1});
    graph.nodes.push_back({"myadd", {{0, 0}, {0, 0}}, 1});
    graph.nodes.push_back({"printint", {{1, 0}}, 0});

    zeno::v2::frontend::ForwardSorter sorter(graph);
    sorter.require(2);
    auto ir = sorter.linearize();

    for (auto const &invo: ir->invos) {
        zeno::v2::helpers::print_invocation(std::cout, invo);
    }

    auto scope = zeno::v2::backend::Session::get().makeScope();
    for (auto const &invo: ir->invos) {
        invo.apply(scope.get());
    }

    return 0;
}
