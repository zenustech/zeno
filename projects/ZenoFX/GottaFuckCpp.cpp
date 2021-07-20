#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>

struct Buffer {
    size_t x[4];
};

struct GottaFuckCpp : zeno::INode {
    virtual void apply() override {
        set_output("prim", std::make_shared<zeno::PrimitiveObject>());
        std::vector<Buffer> chs(1);
        chs[0] = Buffer();
    }
};

ZENDEFNODE(GottaFuckCpp, {
    {"prim"},
    {"prim"},
    {},
    {"debug"},
});
