#include <zeno/zeno.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include "Particles.h"
#include <cassert>

static zfx::Compiler<zfx::x64::Program> compiler;

struct ParticlesWrangle : zeno::INode {
    virtual void apply() override {
        auto pars = get_input<Particles>("pars");
        auto code = get_param<std::string>("code");

        zfx::Options opts;
        pars->foreach_attr([&] (auto const &key, auto &arr) {
            opts.define_symbol("@" + key, arr.Dimension);
        });
        auto prog = compiler.compile(code, opts);

        decltype(auto) chsyms = prog->get_symbols();
        std::vector<float *> chbases(chsyms.size());
        for (int chid = 0; chid < chsyms.size(); chid++) {
            auto const &[name, compid] = chsyms[chid];
            assert(name[0] == '@');
            pars->visit_attr(name.substr(1), [&] (auto &arr) {
                chbases[chid] = arr.data(compid);
            });
        }

        float myptr[4];
        printf("chans %d\n", chbases.size());

        for (int i = 0; i < pars->size(); i += prog->SimdWidth) {
            auto ctx = prog->make_context();
            for (int chid = 0; chid < chbases.size(); chid++) {
                //ctx.channel_pointer(chid) = chbases[chid] + i;
                ctx.channel_pointer(chid) = myptr;
            }
            ctx.execute();
        }

        set_output("pars", std::move(pars));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"pars"},
    {"pars"},
    {{"multiline_string", "code", ""}},
    {"zenofx"},
});
