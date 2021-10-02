#include "DopTable.h"


DopTable tab;


static int def_readvdb = tab.define("readvdb", [] (auto const &in, auto &out) {
    out[0] = [=] () -> std::any {
        printf("readvdb out[0]\n");
        return 1024;
    };
});


static int def_vdbsmooth = tab.define("vdbsmooth", [] (auto const &in, auto &out) {
    out[0] = [=] () -> std::any {
        auto grid = std::any_cast<int>(in[0]());
        auto type = in[1]();
        grid += 1;
        printf("vdbsmooth out[0] %d\n", grid);
        return grid;
    };
});


static int def_vdberode = tab.define("vdberode", [] (auto const &in, auto &out) {
    out[0] = [=] () -> std::any {
        auto grid = std::any_cast<int>(in[0]());
        grid -= 3;
        printf("vdberode out[0] %d\n", grid);
        return grid;
    };
});


static int def_repeat = tab.define("repeat", [] (auto const &in, auto &out) {
    out[0] = [=] () -> std::any {
        for (int i = 0; i < 4; i++) {
            in[0]();
        }
        return 32;
    };
});
