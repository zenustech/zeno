#include "Backend.h"
#include "Frontend.h"
#include "Helpers.h"
#include <iostream>


namespace {

int myadd(int x, int y) {
    auto z = x + y;
    return z;
}
ZENO_DEFINE_NODE(myadd, "int myadd(int x, int y)");

void printint(int x) {
    std::cout << "printint: " << x << std::endl;
}
ZENO_DEFINE_NODE(printint, "void printint(int x)");

void printtype(zeno::v2::container::any x) {
    // see also https://en.cppreference.com/w/cpp/utility/any/type
    std::cout << "printtype: " << x.type().name() << std::endl;
}
ZENO_DEFINE_NODE(printtype, "void printtype(zeno::v2::container::any x)");

}


int main() {
    zeno::v2::frontend::Graph graph;
    // x ? x + x : x
    graph.nodes.push_back({"value", {}, 1, 21.34f});
    graph.nodes.push_back({"myadd", {{0, 0}, {0, 0}}, 1, nullptr});
    graph.nodes.push_back({"if", {{0, 0}, {1, 0}, {0, 0}}, 0, nullptr});

    zeno::v2::frontend::ForwardSorter sorter(graph);
    sorter.require(graph.nodes.size() - 1);
    auto ir = sorter.get_root();

    for (auto const &stmt: ir->stmts) {
        std::cout << stmt->to_string() << std::endl;
    }

    auto scope = zeno::v2::backend::Session::get().makeScope();
    ir->apply(scope.get());

    return 0;
}
