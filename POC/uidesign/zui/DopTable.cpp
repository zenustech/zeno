#include "DopTable.h"


DopTable tab;


static int def_readvdb = tab.define("readvdb", [] (DopContext *ctx) {
    printf("readvdb\n");
});

static int def_vdbsmooth = tab.define("vdbsmooth", [] (DopContext *ctx) {
    printf("vdbsmooth\n");
});

static int def_vdberode = tab.define("vdberode", [] (DopContext *ctx) {
    printf("vdberode\n");
});
