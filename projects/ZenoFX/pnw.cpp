#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include <tuple>

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    size_t stride2 = 0;
    size_t a;
};

struct ParticlesNeighborWrangle : zeno::INode {
    virtual void apply() override {
        set_output("prim", std::make_shared<zeno::PrimitiveObject>());
        std::vector<Buffer> chs(1);
        for (int i = 0; i < chs.size(); i++) {
            Buffer iob;
            chs[i] = iob;
        }
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {"prim", "primNei", "zfxCode", "params", "radius"},
    {"prim"},
    {},
    {"zenofx"},
});
