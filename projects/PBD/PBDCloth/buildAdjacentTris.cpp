#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/tuple_hash.h>
#include <iostream>
// #include  "../Utils/myPrint.h"

using namespace zeno;
/**
 * @brief 找到某个三角形的所有邻接三角形（有共享边的三角形）。至少有1个，至多有3个。
 * 需要先用节点lines2tris建立边到三角面的映射
 */
struct buildAdjacentTris : zeno::INode {

    //找到所有三角形的邻接三角形，然后建立一个邻接表
    void func(PrimitiveObject *prim)
    {
        auto &lines = prim->lines;
        auto &lines2tris = prim->lines.attr<vec3i>("lines2tris");
        

        //遍历所有边到三角面的映射表。
        //对每条边，假如恰好有个与其序号相反的边也在表内，那么这条边就是它的邻接边。
        //那么邻接边所属的三角面就是它的一个邻接三角面。把他们存到map3里面
        for(auto &[k,v]:map1)
        {
            edgeType inv{k[1],k[0]};
            if(map1.find(inv) != map1.end())
            {
                map3.emplace(v,map1.at(inv));
            }
        }

        //最后把邻接三角面存到tris的新建属性adjTri当中
        auto & adjTri = prim->tris.add_attr<vec3i>("adjTri");
        adjTri.clear(); 
        for(auto & t:tris)
        {
            if(map3.find(t) != map3.end())
            {
                adjTri.emplace_back(map3.find(t)->second);
            }
            else{
                adjTri.emplace_back(vec3i{-1,-1,-1});
            }
        }
        
    }



public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        func(prim.get());
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(buildAdjacentTris, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});