#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/tuple_hash.h>

using namespace zeno;
/**
 * @brief 建立边到三角面的对应关系。给一个边，得到它所属的三角面
 * 
 */
struct lines2tris : zeno::INode {

    
    void func(PrimitiveObject *prim)
    {

        auto & tris = prim->tris;
        auto & lines = prim->lines;

        using edgeType = vec2i;
        using triType = vec3i;
        std::unordered_map<edgeType, triType, tuple_hash, tuple_equal> map1;//边到三角面

        // 边到面的映射
        for(auto const & t : tris)
        {
            vec2i e1{t[0], t[1]};   //三角面的三条边
            vec2i e2{t[1], t[2]};   
            vec2i e3{t[2], t[0]};

            map1.emplace(e1,t);  
            map1.emplace(e2,t);  
            map1.emplace(e3,t);  
        } 

        auto & l2t = lines.add_attr<vec3i>("lines2tris");
        l2t.clear();
        for(auto l:lines)
        {
            if(map1.find(l) != map1.end())
            {
                l2t.emplace_back(map1.at(l));
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

ZENDEFNODE(lines2tris, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});