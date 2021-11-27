#include <z2/dop/DopTable.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {


DopTable tab;


static int def_route = tab.define("route", {{
    "control", "do nothing but return the argument",
}, {
    {"value"},
}, {
    {"value"},
}, [] (DopNode *node) {
    auto value = node->get_input(0);
    printf("route\n");
    node->set_output(0, value);
}});


static int def_first = tab.define("first", {{
    "control", "always return the first argument",
}, {
    {"lhs"},
    {"rhs"},
}, {
    {"lhs"},
}, [] (DopNode *node) {
    auto lhs = node->get_input(0);
    auto rhs = node->get_input(1);
    printf("first\n");
    node->set_output(0, lhs);
}});


static int def_repeat = tab.define("repeat", {{
    "control", "repeat an operation for multiple times",
}, {
    {"value"},
    {"times"},
}, {
    {"value"},
}, [] (DopNode *node) {
    auto times = std::any_cast<int>(node->get_input(1));
    for (int i = 0; i < times; i++) {
        node->get_input(0);
    }
    printf("repeat\n");
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
}, [] (DopNode *node) {
    auto cond = std::any_cast<int>(node->get_input(0));
    if (cond) {
        printf("if[1]\n");
        node->set_output(0, node->get_input(1));
    } else {
        printf("if[0]\n");
        node->set_output(0, node->get_input(2));
    }
}});


}  // namespace z2::dop
