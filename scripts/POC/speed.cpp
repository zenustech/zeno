#define ZFX_IMPLEMENTATION
#include <zeno/zfx.h>
#include <Hg/IOUtils.h>
#include <iostream>
#include <chrono>

#define N (128*1024*1024)

int main() {
    auto prog = zfx::compile_program(hg::file_get_content("code.zfx"));

    volatile float *x = new volatile float[N];
    memset((float *)x, 0, sizeof(float) * N);

    auto t0 = std::chrono::steady_clock::now();

#if 1
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        zfx::Context ctx;
        ctx.memtable[0] = (float *)x + i;
        for (int j = 0; j < 4; j++) {
            prog->execute(&ctx);
        }
    }
#else
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 4; j++) {
            x[i] = x[i] * x[i] + x[i] * 3.14f + 2.718f / x[i];
        }
    }
#endif

    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>
        (t1 - t0).count() << " ms" << std::endl;

    return 0;
}
