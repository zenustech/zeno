#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cmath>

static zfx::Compiler<zfx::x64::Program> compiler;

int main() {
#if 0
    std::string code("tmp = @pos + 0.5\n@pos = tmp + 3.14 * tmp + 2.718 / (@pos * tmp + 1)");
    auto func = [](float pos) -> float {
        auto tmp = pos + 0.5f;
        pos = tmp + 3.14f * tmp + 2.718f / (pos * tmp + 1);
        return pos;
    };
#else
    std::string code("@pos = 2.718");
    auto func = [](float pos) -> float {
        return 2.718f;
    };
#endif

    zfx::Options opts;
    opts.define_symbol("@pos", 1);
    opts.define_param("$dt", 1);
    auto prog = compiler.compile(code, opts);

    float arr[4] = {1, 2, 3, 4};

    printf("expected:");
    for (auto val: arr) {
        val = func(val);
        printf(" %f", val);
    }
    printf("\n");

    auto ctx = prog->make_context();

    prog->parameter(prog->param_id("$dt", 0)) = 3.14f;
    memcpy(&ctx.channel(prog->symbol_id("@pos", 0)), arr, sizeof(arr));
    ctx.execute();
    memcpy(arr, &ctx.channel(prog->symbol_id("@pos", 0)), sizeof(arr));

    printf("result:");
    for (auto val: arr) {
        printf(" %f", val);
    }
    printf("\n");

    return 0;
}
