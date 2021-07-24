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
    int n = 1;
    std::string code(R"(
@clr = @pos <= 2
)");
#endif

    zfx::Options opts(zfx::Options::for_x64);
    opts.detect_new_symbols = true;
    opts.reassign_channels = false;
    opts.reassign_parameters = false;
    opts.define_symbol("@pos", n);
    opts.define_symbol("@clr", n);
    auto prog = compiler.compile(code, opts);
    auto exec = assembler.assemble(prog->assembly);

    for (auto const &[key, dim]: prog->symbols) {
        printf("%s.%d\n", key.c_str(), dim);
    }
    for (auto const &[key, dim]: prog->newsyms) {
        printf("new symbol %s with dim %d\n", key.c_str(), dim);
    }

    auto ctx = exec->make_context();
    for (int i = 0; i < n; i++) {
        ctx.channel(prog->symbol_id("@pos", i))[0] = 1.414f;
    }
    ctx.execute();
    for (int i = 0; i < n; i++) {
        int sid = prog->symbol_id("@pos", i);
        printf("%d\n", sid);
        sid = prog->symbol_id("@clr", i);
        printf("%d\n", sid);
        printf("%f\n", ctx.channel(sid)[0]);
    }

    return 0;
}
