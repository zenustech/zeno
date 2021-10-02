#include <z2/dop/DopTable.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {


DopTable tab;


static int def_readvdb = tab.define("readvdb", {
    {
        {"path"},
        {"type"},
    },
    {
        {"grid"},
    },
}, [] (auto *node, auto *visited) {
    printf("readvdb out[0]\n");
    node->set_output(0, 1024);
});


static int def_vdbsmooth = tab.define("vdbsmooth", {
}, [] (auto *node, auto *visited) {
    auto grid = std::any_cast<int>(node->get_input(0, visited));
    auto type = node->get_input(1, visited);
    grid += 1;
    printf("vdbsmooth out[0] %d\n", grid);
    node->set_output(0, grid);
});


static int def_vdberode = tab.define("vdberode", {
}, [] (auto *node, auto *visited) {
    auto grid = std::any_cast<int>(node->get_input(0, visited));
    auto type = node->get_input(1, visited);
    grid -= 3;
    printf("vdberode out[0] %d\n", grid);
    node->set_output(0, grid);
});


static int def_repeat = tab.define("repeat", {
}, [] (auto *node, auto *visited) {
    printf("repeat out[0]\n");
    for (int i = 0; i < 4; i++) {
        auto saved_visited = *visited;
        node->get_input(0, &saved_visited);
    }
    node->set_output(0, 32);
});


}  // namespace z2::dop
