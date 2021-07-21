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
    int n = 4;
    std::string code(R"(
@clr = 0.5 + @pos + @pos
)");
#endif

    zfx::Options opts(zfx::Options::for_x64);
    opts.define_symbol("@pos", n);
    opts.define_symbol("@clr", n);
    //opts.reassign_channels = false;
    auto prog = compiler.compile(code, opts);
    auto exec = assembler.assemble(prog->assembly);

    for (auto const &[key, dim]: prog->symbols) {
        printf("%s.%d\n", key.c_str(), dim);
    }

    auto ctx = exec->make_context();
    for (int i = 0; i < n; i++) {
        ctx.channel(prog->symbol_id("@pos", n))[0] = 1.0f;
    }
    ctx.execute();
    for (int i = 0; i < n; i++) {
        printf("%f\n", ctx.channel(prog->symbol_id("@clr", n))[0]);
    }

    return 0;
}
