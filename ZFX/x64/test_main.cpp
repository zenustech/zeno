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
#else
    std::string code("@pos.x = sin(@clr)");
#endif

    zfx::Options opts(zfx::Options::for_x64);
    opts.define_symbol("@pos", 3);
    opts.define_symbol("@clr", 1);
    auto prog = compiler.compile(code, opts);
    auto exec = assembler.assemble(prog->assembly);

    for (auto const &[key, dim]: prog->symbols) {
        printf("%s.%d\n", key.c_str(), dim);
    }

    auto ctx = exec->make_context();
    ctx.channel(prog->symbol_id("@pos", 0))[0] = 1.f;
    ctx.channel(prog->symbol_id("@clr", 0))[0] = 3.14f;
    ctx.execute();
    printf("new_pos = %f\n", ctx.channel(prog->symbol_id("@pos", 0))[0]);

    return 0;
}
