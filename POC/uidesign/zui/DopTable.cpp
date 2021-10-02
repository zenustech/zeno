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
        auto grid = in[0]();
        printf("vdbsmooth out[0]\n");
        return 1024;
    };
});

static int def_vdberode = tab.define("vdberode", [] (auto const &in, auto &out) {
    out[0] = [=] () -> std::any {
        printf("vdberode out[0]\n");
        return 1024;
    };
});
