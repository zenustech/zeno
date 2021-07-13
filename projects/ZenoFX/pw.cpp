#include <zeno/zeno.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include "Particles.h"

static zfx::Compiler<zfx::x64::Program> compiler;

struct ParticlesWrangle : zeno::INode {
    virtual void apply() override {
        auto pars = get_input<Particles>("pars");
        auto code = get_param<std::string>("code");

        std::map<std::string, int> symdims;
        pars->foreach_attr([&symdims] (auto const &key, auto &arr) {
            symdims[key] = arr.Dimension;
        });
        auto prog = compiler.compile(code, symdims);

        pars->foreach_attr([&] (auto const &key, auto &arr) {
            std::vector<int> chids = prog->channel_ids("@" + key, arr.Dimension);
            for (int i = 0; i < arr.size(); i += prog->SimdWidth) {
                auto ctx = prog->make_context();
                for (int j = 0; j < arr.Dimension; j++) {
                    ctx.channel_pointer(chids[j]) = &arr.at(i, j);
                }
                ctx.execute();
            }
        });

        set_output("pars", std::move(pars));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"pars"},
    {"pars"},
    {{"multiline_string", "code", ""}},
    {"zenofx"},
});
