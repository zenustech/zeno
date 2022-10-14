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
    

    void func(PrimitiveObject *prim)
    {
        auto &lines = prim->lines;
        auto &tris = prim->tris;
        lines.clear(); //清除边，后续会重建。
        lines.reserve(tris.size()*3);

        std::vector<vec3i> edges;
        edges.reserve(tris.size()*3);
        vec3i t;
        int id0,id1;
        for(int i=0; i<tris.size(); i++)
        {
            t = tris[i];
            //edges保存前两个值代表一条边的两个顶点，排序后存入。
            //第三个值是边的编号。
            id0 = t[0]; id1 = t[1];
            edges.emplace_back(std::min(id0,id1),std::max(id0,id1), i*3 + 0);
            id0 = t[1]; id1 = t[2];
            edges.emplace_back(std::min(id0,id1),std::max(id0,id1), i*3 + 1);
            id0 = t[0]; id1 = t[2]
            edges.emplace_back(std::min(id0,id1),std::max(id0,id1), i*3 + 2);
        } 
        
        // sort
        auto myLess = [](const vec3i &a, const vec3i &b){
            if(a[0]<b[0]) return true;
            else if(a[0]==b[0] && a[1]<b[1]) return true;
            else return false;
        }
        std::sort(edges.begin(),edges.end(),myLess);


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