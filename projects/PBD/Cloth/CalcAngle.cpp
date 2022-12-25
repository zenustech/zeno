#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;


struct PBDCalcAngle : zeno::INode {
    void calc(PrimitiveObject *prim)
    {
        auto &tris = prim->tris;
        auto &nrm = prim->tris.attr<float>("nrm");
        auto &adj = prim->tris.attr<vec3i>("adj");
        auto &ang = prim->tris.add_attr("ang");

        for (size_t i = 0; i < tris.size(); i++)
        {
            vec3f n1 = nrm[i];
            for (size_t j = 0; j < adj[i].size(); j++)
            {
                if(adj[i][j] == -1)
                    continue;
                vec3f n2 = nrm[adj[i][j]];
                ang[i][j] = dot(n1,n2);
            }
        }
    }

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        
        calc(prim.get());

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    PBDCalcAngle,{
        {"prim"},
        {"outPrim"},
        {},
        {"PBD"}
    }
)