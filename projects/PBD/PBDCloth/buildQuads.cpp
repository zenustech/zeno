#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;

struct buildQuads : zeno::INode {
    void build(PrimitiveObject *prim, int numWidth)
    {
        int n = prim->verts.size();
        int numHeight = n/numWidth;

        //debug
        n= numWidth;
        
        for (int i = 0; i < numHeight; i++)
        {
            for (int j = 0; j < numWidth; j++)
            {
                int quad_id = i*(n-1) +j
                // # 1st triangle of the square
                indices[quad_id * 6 + 0] = i * n + j
                indices[quad_id * 6 + 1] = (i + 1) * n + j
                indices[quad_id * 6 + 2] = i * n + (j + 1)
                // # 2nd triangle of the square
                indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
                indices[quad_id * 6 + 4] = i * n + (j + 1)
                indices[quad_id * 6 + 5] = (i + 1) * n + j
            }
            
        }
        
    }

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto numWidth = get_input<NumericObject>("width")->get<int>();

        build(prim.get(), numWidth);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    buildQuads,{
        {gParamType_Primitive,{gParamType_Int,"numWidth","128"}},
        {"outPrim"},
        {},
        {"PBD"}
    }
)