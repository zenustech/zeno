#define ZFX_IMPLEMENTATION
#include <cstdio>
#include <zeno/zfx.h>

int main() {
    auto prog = zfx::compile_program("define f1 @x\n@x = 3 * @x");

    float *x = new float[233];

    for (int i = 0; i < 32; i++) {
        zfx::Context ctx;
        ctx.memtable[0] = x + i;
        prog->execute(&ctx);
    }
}
