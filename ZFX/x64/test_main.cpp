#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cmath>

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

int main() {
#if 0
    std::string code("tmp = @pos + 0.5\n@pos = tmp + 3.14 * tmp + 2.718 / (@pos * tmp + 1)");
    auto func = [](float pos) -> float {
        auto tmp = pos + 0.5f;
        pos = tmp + 3.14f * tmp + 2.718f / (pos * tmp + 1);
        return pos;
    };
#elif 0
    std::string code("@pos = $dt + 2.718");
    auto func = [](float pos) -> float {
        return 3.14f + 2.718f;
    };
#else
    std::string code("@pos = func($dt, 1 - 2)");
    auto func = [](float pos) -> float {
        return 3.14f + 2.718f;
    };
#endif

    zfx::Options opts(zfx::Options::for_x64);
    opts.define_symbol("@pos", 1);
    opts.define_param("$dt", 1);
    auto prog = compiler.compile(code, opts);
    auto exec = assembler.assemble(prog->assembly);

    /*float arr[4] = {1, 2, 3, 4};

    printf("expected:");
    for (auto val: arr) {
        val = func(val);
        printf(" %f", val);
    }
    printf("\n");

    auto ctx = exec->make_context();

    exec->parameter(prog->param_id("$dt", 0)) = 3.14f;
    memcpy(ctx.channel(prog->symbol_id("@pos", 0)), arr, sizeof(arr));
    ctx.execute();
    memcpy(arr, ctx.channel(prog->symbol_id("@pos", 0)), sizeof(arr));

    printf("result:");
    for (auto val: arr) {
        printf(" %f", val);
    }
    printf("\n");*/

    return 0;
}
