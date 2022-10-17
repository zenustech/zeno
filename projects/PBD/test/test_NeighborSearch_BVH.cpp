#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../ZenoFX/LinearBvh.h" //BVH搜索
#include "../PBF/NeighborListData.h" // 数据类，用来节点间数据传递
#include "../Utils/myPrint.h"//test
#include "../Utils/readFile.h"//test


using namespace zeno;

namespace zeno{
struct test_NeighborSearch_BVH: INode
{

    void buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list)
    {
        auto radius2 = searchRadius*searchRadius;
        #pragma omp parallel for
        for (int i = 0; i < pos.size(); i++) 
        {
            lbvh->iter_neighbors(pos[i], [&](int j) 
                {
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
        // std::cout<<"good\n";

        auto prim = get_input<PrimitiveObject>("prim");
        float searchRadius = get_input<NumericObject>("searchRadius")->get<float>();
        auto lbvh = get_input<zeno::LBvh>("lbvh");
        auto & pos = prim->verts; 

        pos.clear();//test
        readVectorField("pos_input.csv",pos);//test
        auto neighborList = std::make_shared<NeighborListData>();
        std::vector<std::vector<int>> &neiList = neighborList->value;
        neiList.resize(pos.size());

        buildNeighborList(pos, searchRadius, lbvh.get(), neiList);
        printVectorField("neighborList_out4.csv",neiList,0);//test
        
        set_output("neighborList", std::move(neighborList));
    }
};

ZENDEFNODE(test_NeighborSearch_BVH,
    {
        {
            {"prim"},
            {"LBvh","lbvh"},
            {"float", "searchRadius","1.155"},
        },
        {"neighborList"},
        {},
        {"PBD"},
    }
);


}//zeno
