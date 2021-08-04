#include "Backend.h"
#include "Frontend.h"
#include "Helper.h"


int myadd(int x, int y) {
    auto z = x + y;
    return z;
}
ZENO_DEFINE_NODE(myadd);

int makeint() {
    return 21;
}
ZENO_DEFINE_NODE(makeint);

void printint(int x) {
    std::cout << "printint: " << x << std::endl;
}
ZENO_DEFINE_NODE(printint);


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
