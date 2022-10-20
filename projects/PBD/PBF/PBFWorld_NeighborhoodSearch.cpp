#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../ZenoFX/LinearBvh.h" //BVH的构建和使用API
#include "./PBFWorld.h"
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
        auto data = get_input<PBFWorld>("PBFWorld");
        auto & pos = data->prim->verts;

        //构建BVH
        auto lbvh = std::make_shared<zeno::LBvh>(data->prim,  data->neighborSearchRadius,zeno::LBvh::element_c<zeno::LBvh::element_e::point>);

        //清零
        data->neighborList.clear();
        data->neighborList.resize(pos.size());

        //邻域搜索
        buildNeighborList(pos, data->neighborSearchRadius, lbvh.get(), data->neighborList);
        
        //输出数据
        set_output("PBFWorld", std::move(data));//必须传递的是继承自IObject的类对象的shared_ptr
    }
};

ZENDEFNODE(PBFWorld_NeighborhoodSearch,
    {
        {
            {"PBFWorld"},
        },
        {"PBFWorld"},
        {},
        {"PBD"},
    }
);


}//zeno
