#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

struct PrimitiveFlipPoly : zeno::INode {
    virtual void apply() override {
        auto surfIn = get_input<zeno::PrimitiveObject>("prim");
        for(size_t i = 0;i < surfIn->tris.size();++i){
            auto& tri = surfIn->tris[i];
            size_t tri_idx_tmp = tri[1];
            tri[1] = tri[2];
            tri[2] = tri_idx_tmp;
        }

        set_output("primOut",surfIn);
    }
};

ZENDEFNODE(PrimitiveFlipPoly,{
    {{"prim"}},
    {"primOut"},
    {},
    {"primitive"},
});

};