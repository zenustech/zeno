#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <set>
#include <unordered_set>

using namespace zeno;
/**
 * @brief 找到某个三角形的所有邻接三角形（有共享边的三角形）。至少有1个，至多有3个。
 * 
 */
struct findAdjacent : zeno::INode {

    //找到所有三角形的邻接三角形，然后建立一个邻接表
    void buildAdjacentList(PrimitiveObject *prim)
    {
        auto & lines2Tris = prim->lines.add_attr<int>("lines2Tris");

        struct myLess {
        bool operator()(vec2i const &a, vec2i const &b) const {
                return std::make_pair(std::min(a[0], a[1]), std::max(a[0], a[1]))
                    < std::make_pair(std::min(b[0], b[1]), std::max(b[0], b[1]));
            }
        };
        std::set<vec2i, myLess> segments;
        auto append = [&] (int i, int j) {
            segments.emplace(i, j);
        };
        for (auto const &ind: prim->lines) {
            append(ind[0], ind[1]);
        }
        for (auto const &ind: prim->tris) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[0]);
        }
        
    }



    //找到某个邻接的三角形
    int find(PrimitiveObject *prim, int t, vec2i &adjacentEdge)
    {
        auto & tris = prim->tris
        //取出所有点
        auto p1 = tris[t][0];
        auto p2 = tris[t][1];
        auto p3 = tris[t][2];
        //取出所有边
        std::set<int> e1(pp1,pp2);
        std::set<int> e2(pp1,pp3);
        std::set<int> e3(pp3,pp2);

        std::vector<std::set<int>> edgesOfT{e1,e2,e3};
        // struct myLess {
        //     bool operator()(vec2i const &a, vec2i const &b) const {
        //         return std::make_pair(std::min(*a.cbegin(), *a.cend()), std::max(*a.cbegin(), *a.cend())) <
        //         std::make_pair(std::min(*b.cbegin(), *b.cend()), std::max(*b.cbegin(), *b.cend()));
        //     }
        // };
        // std::set<std::set<int>,myLess> edgesOft{e1,e2,e3};

        for (size_t i = 0; i < tris.size(); i++)
        {
            //取出所有点
            auto pp1 = tris[i][0];  
            auto pp2 = tris[i][1];
            auto pp3 = tris[i][2];

            //取出所有边
            std::set<int> ee1(pp1,pp2);
            std::set<int> ee2(pp1,pp3);
            std::set<int> ee3(pp3,pp2);
            
            std::vector<std::set<int>> edgesOfI{ee1,ee2,ee3};

            //比较边是否相同
            for(const auto & t1:edgesOfT)
            {
                for(const auto & t2:edgesOfI)
                {
                    if(*t1.cbegin() == *t2.cbegin && *t1.cend() == *t2.cend())
                    {
                        //t1和t2是邻接边
                        adjacent.push_back(i); //返回三角形编号
                    }
                }
            }
        }

        //如果没找到返回-1.
        if(i == (tris.size()+1) )
            return -1;
        
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        func(prim.get());
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(findAdjacent, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "dihedralCompliance", "0.0"},
                    {"float", "dt", "0.0016667"},
                    {"int", "triangle1", ""},
                    {"int", "triangle2", ""},
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});