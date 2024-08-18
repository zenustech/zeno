#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../ZenoFX/LinearBvh.h" //BVH的构建和使用API
#include "./PBFWorld.h"
#include "../Utils/myPrint.h"
using namespace zeno;

namespace zeno{
struct PBFWorld_NeighborhoodSearch: INode
{

    void buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list)
    {
        auto radius2 = searchRadius*searchRadius;
        #pragma omp parallel for
        for (int i = 0; i < pos.size(); i++) 
        {
            //BVH的使用
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
        auto prim = get_input<PrimitiveObject>("prim");
        auto data = get_input<PBFWorld>("PBFWorld");
        auto &pos = prim->verts;

        //构建BVH
        auto lbvh = std::make_shared<zeno::LBvh>(prim,  data->neighborSearchRadius,zeno::LBvh::element_c<zeno::LBvh::element_e::point>);

        //清零
        data->neighborList.clear();
        data->neighborList.resize(pos.size());

        //邻域搜索
        buildNeighborList(pos, data->neighborSearchRadius, lbvh.get(), data->neighborList);

        // //debug
        // printVectorField("neighborList_out11.csv",data->neighborList,0);//test
        
        //输出数据
        set_output("outPrim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(PBFWorld_NeighborhoodSearch,
    {
        {gParamType_Primitive,"PBFWorld"},
        {"outPrim","PBFWorld"},
        {},
        {"PBD"},
    }
);


}//zeno
