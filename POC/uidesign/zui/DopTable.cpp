#include "DopTable.h"


DopTable tab;


static int def_readvdb = tab.define("readvdb", [] (DopContext *ctx) {
    //auto dx = std::any_cast<float>(ctx.in[0]);
    //printf("readvdb %f\n", dx);
    printf("readvdb\n");
});

static int def_vdbsmooth = tab.define("vdbsmooth", [] (DopContext *ctx) {
    printf("vdbsmooth\n");
});

static int def_vdberode = tab.define("vdberode", [] (DopContext *ctx) {
    printf("vdberode\n");
});
