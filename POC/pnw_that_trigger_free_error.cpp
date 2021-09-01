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
    size_t x[4];
};

struct ParticlesNeighborWrangle : zeno::INode {
    virtual void apply() override {
        set_output("prim", std::make_shared<zeno::PrimitiveObject>());
        std::vector<Buffer> chs(1);
        chs[0] = Buffer();
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {},
    {"prim"},
    {},
    {"zenofx"},
});
