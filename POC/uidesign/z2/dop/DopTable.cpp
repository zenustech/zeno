#include <z2/dop/DopTable.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {


DopTable tab;


static int def_readvdb = tab.define("readvdb", {{
    "openvdb", "load from .vdb file",
}, {
    {"path"},
    {"type"},
}, {
    {"grid"},
}, [] (DopNode *node, DopContext *visited) {
    printf("readvdb out[0]\n");
    node->set_output(0, 1024);
}});


static int def_vdbsmooth = tab.define("vdbsmooth", {{
    "openvdb", "gaussian smooth vdb grid",
}, {
    {"grid"},
    {"times"},
    {"width"},
}, {
    {"grid"},
}, [] (DopNode *node, DopContext *visited) {
    auto grid = std::any_cast<int>(node->get_input(0, visited));
    auto type = node->get_input(1, visited);
    grid += 1;
    printf("vdbsmooth out[0] %d\n", grid);
    node->set_output(0, grid);
}});


static int def_vdberode = tab.define("vdberode", {{
    "openvdb", "erode the levelset by levels",
}, {
    {"grid"},
    {"levels"},
}, {
    {"grid"},
}, [] (DopNode *node, DopContext *visited) {
    auto grid = std::any_cast<int>(node->get_input(0, visited));
    auto levels = node->get_input(1, visited);
    grid -= 3;
    printf("vdberode out[0] %d\n", grid);
    node->set_output(0, grid);
}});


static int def_repeat = tab.define("repeat", {{
    "control", "repeat an operation for multiple times",
}, {
    {"value"},
    {"times"},
}, {
    {"value"},
}, [] (DopNode *node, DopContext *visited) {
    auto times = std::any_cast<int>(node->get_input(1, visited)());
    for (int i = 0; i < times; i++) {
        auto saved_visited = *visited;
        node->get_input(0, &saved_visited)();
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
    auto cond = std::any_cast<int>(node->get_input(0, visited)());
    if (cond) {
        node->set_output(0, node->get_input(1, visited)());
    } else {
        node->set_output(0, node->get_input(2, visited)());
    }
}});


}  // namespace z2::dop
