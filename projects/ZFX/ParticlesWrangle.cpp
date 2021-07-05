#include "Program.h"
#include "split_str.h"
#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <cassert>

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
};

static void vectors_wrangle(Program const *prog,
    std::vector<Buffer> const &chs) {
    if (chs.size() == 0)
        return;
    size_t size = chs[0].count;
    for (int i = 1; i < chs.size(); i++) {
        size = std::min(chs[i].count, size);
    }
    Context ctx;
    for (int i = 0; i < chs.size(); i++) {
        ctx.memtable[i] = chs[i].base;
    }
    for (int i = 0; i < size; i++) {
        prog->execute(&ctx);
        for (int i = 0; i < chs.size(); i++) {
            ctx.memtable[i] = chs[i].base;
        }
    }
}

static void particles_wrangle(Program const *prog,
    zeno::PrimitiveObject const *prim) {
    std::vector<Buffer> chs(prog->channels.size());
    for (int i = 0; i < chs.size(); i++) {
        auto chan = split_str(prog->channels[i], '.');
        assert(chan.size() == 2);
        int dimid = 0;
        std::stringstream(chan[1]) >> dimid;
        Buffer iob;
        auto const &attr = prim->attr(chan[0]);
        std::visit([&] (auto const &arr) {
            iob.base = (float *)arr.data() + dimid;
            iob.count = arr.size();
            iob.stride = sizeof(arr[0]);
        }, attr);
        chs[i] = iob;
    }
    vectors_wrangle(prog, chs);
}

struct ParticlesWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();
        auto prog = compile_program(code);
        particles_wrangle(prog, prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"prim", "zfxCode"},
    {"prim"},
    {},
    {"zenofx"},
});
