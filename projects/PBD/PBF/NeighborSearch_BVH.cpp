#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/core/IObject.h>
#include "../ZenoFX/LinearBvh.h"
#include "NeighborListData.h"
#include <iostream>

using namespace zeno;

namespace zeno{
struct NeighborSearch_BVH: INode
{
    void buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, zeno::LBvh *lbvh, NeighborListData & list)
    {
        auto radius2 = searchRadius*searchRadius;
        list.value.resize(pos.size());
        // #pragma omp parallel for
        for (int i = 0; i < pos.size(); i++) 
        {
            std::cout<<"i: "<<i<<"\n";
            lbvh->iter_neighbors(
                pos[i], [&](int j) 
                {
                    if (lengthSquared(pos[i] - pos[j]) < radius2)
                    {
                        list.value[i].emplace_back(j);
                        std::cout<<"j: "<<j<<"\n";
                    }
                }
            );
        }
    
    }

    virtual void apply() override
    {
        std::cout<<"good\n";

        auto prim = get_input<PrimitiveObject>("prim");
        auto & pos = prim->verts;
        float searchRadius = get_input<NumericObject>("searchRadius")->get<float>();
        NeighborListData neighborList;
        auto lbvh = get_input<zeno::LBvh>("lbvh");

        buildNeighborList(pos, searchRadius, lbvh.get(), neighborList);

        //for debug only
        auto numNeighbors = prim->add_attr<int>("numNeighbors");
        for (size_t i = 0; i < pos.size(); i++)
        {
            numNeighbors[i]=neighborList.value[i].size();
            // std::cout<<"numNei["<<i<<"]: "<<neighborList.value[i].size()<<"\n";
        }
        
        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(NeighborSearch_BVH,
    {
        {
            {"prim"},
            {"lbvh"},
            {"float", "searchRadius",""},
        },
        {"outPrim"},
        {},
        {"PBD"},
    }
);


}//zeno
