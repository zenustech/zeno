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

    void buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list)
    {
        auto radius2 = searchRadius*searchRadius;
        // #pragma omp parallel for
        for (int i = 0; i < pos.size(); i++) 
        {
            // std::cout<<"good i: "<<i<<"\n";
            lbvh->iter_neighbors(pos[i], [&](int j) 
                {
                    // std::cout<<"i:"<<i<<"\nj:"<<j<<"\n";
                    if (lengthSquared(pos[i] - pos[j]) < radius2 && j!=i)
                    {
                        list[i].emplace_back(j);
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
        // NeighborListData neighborList;
        auto lbvh = get_input<zeno::LBvh>("lbvh");

        std::vector<std::vector<int>> neiList;
        neiList.resize(pos.size());

        std::vector<vec3f> pos1;
        buildNeighborList(pos1, searchRadius, lbvh.get(), neiList);
        
        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(NeighborSearch_BVH,
    {
        {
            {"prim"},
            {"LBvh","lbvh"},
            {"float", "searchRadius","1.155"},
        },
        {"outPrim"},
        {},
        {"PBD"},
    }
);


}//zeno
