#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>

struct Buffer {
    size_t x[4];
};

struct ThisWontFuckCpp : zeno::INode {
    virtual void apply() override {
        set_output("prim", std::make_shared<zeno::PrimitiveObject>());
        std::vector<Buffer> chs(1);
        chs[0] = Buffer();
    }
};

ZENDEFNODE(ThisWontFuckCpp, {
    {"prim"},
    {"prim"},
    {},
    {"debug"},
});
