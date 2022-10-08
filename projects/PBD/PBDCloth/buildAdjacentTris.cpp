#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/tuple_hash.h>
#include <iostream>
// #include  "../Utils/myPrint.h"
using std::vector;

using namespace zeno;
/**
 * @brief 找到某个三角形的所有邻接三角形（有共享边的三角形）。至少有1个，至多有3个。
 * 
 */
struct buildAdjacentTris : zeno::INode {

    //找到所有三角形的邻接三角形，然后建立一个邻接表
    void func(PrimitiveObject *prim)
    {
        auto &lines = prim->lines;
        auto &tris = prim->tris;

        using edgeType = vec2i;
        using triType = vec3i;

        //把tris和lines都存成反向的序号map。也就是键是tris的三个int，值则是tris数组中的序号。
        std::unordered_map<triType, int, tuple_hash, tuple_equal> triMap;
        std::unordered_map<edgeType, int, tuple_hash, tuple_equal> edgeMap;
        for (int i = 0; i < tris.size(); i++)
            triMap[tris[i]] = i;
        for (int i = 0; i < lines.size(); i++)
            edgeMap[lines[i]] = i;


        // 1. 先建立边到面的映射表。存到map1里。
        std::unordered_map<edgeType, triType, tuple_hash, tuple_equal> map1;//边到三角面
        for(auto const & t : tris)
        {
            vec2i e1{t[0], t[1]};   //三角面的三条边
            vec2i e2{t[1], t[2]};   
            vec2i e3{t[2], t[0]};

            map1.emplace(e1,t);  
            map1.emplace(e2,t);  
            map1.emplace(e3,t);  
        } 


        //2. 找到三角面的邻接面。
        //遍历所有边到三角面的映射表。
        //对每条边，假如恰好有个与其序号相反的边也在表内，那么这条边就是它的邻接边。
        //那么邻接边所属的三角面就是它的一个邻接三角面。存到map2里。
        using multiTriType = std::vector<triType>;
        std::unordered_map<triType, multiTriType, tuple_hash, tuple_equal> map2;//面到邻接面(多个)
        for(auto &[k,v]:map1) //k是边，v是所属的面
        {
            edgeType inv{k[1],k[0]}; //相反序号的边
            if(map1.find(inv) != map1.end())
            {
                const triType & one= map1.at(inv); //其中一个邻接面
                map2[v].push_back(one);//放到邻接面表中
            }
        }

        //3. 最后把邻接三角面存到tris的属性 adjTriId 当中。用于可视化
        auto & adjTriId = prim->tris.add_attr<vec3i>("adjTriId");
        std::fill(adjTriId.begin(),adjTriId.end(),vec3i{-1,-1,-1});
        for (size_t i = 0; i < tris.size(); i++)
        {
            const multiTriType & adjs = map2.at(tris[i]); //这是一组邻接面，可能有多个，是个vector
            for(int j=0; j < adjs.size(); j++) //取出每个邻接面
            {
                int ind = triMap.at(adjs[j]); //找到在tris中对应的序号
                adjTriId[i][j] = ind; //j是0，1，2
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