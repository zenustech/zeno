#include <z2/dop/DopTable.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {


DopTable tab;


static int def_first = tab.define("first", {{
    "control", "return the first argument",
}, {
    {"lhs"},
    {"rhs"},
}, {
    {"lhs"},
}, [] (DopNode *node, DopContext *visited) {
    auto lhs = node->get_input(0, visited);
    auto rhs = node->get_input(1, visited);
    node->set_output(0, lhs);
}});


static int def_repeat = tab.define("repeat", {{
    "control", "repeat an operation for multiple times",
}, {
    {"value"},
    {"times"},
}, {
    {"value"},
}, [] (DopNode *node, DopContext *visited) {
    auto times = std::any_cast<int>(node->get_input(1, visited));
    for (int i = 0; i < times; i++) {
        auto saved_visited = *visited;
        node->get_input(0, &saved_visited);
    }
    node->set_output(0, 32);
}});


static int def_if = tab.define("if", {{
    "control", "if 'cond' is true, return 'then'; otherwise, return 'else'.",
}, {
    {"cond"},
    {"then"},
    {"else"},
}, {
    {"value"},
}, [] (DopNode *node, DopContext *visited) {
    auto cond = std::any_cast<int>(node->get_input(0, visited));
    if (cond) {
        node->set_output(0, node->get_input(1, visited));
    } else {
        node->set_output(0, node->get_input(2, visited));
    }
}});


}  // namespace z2::dop
