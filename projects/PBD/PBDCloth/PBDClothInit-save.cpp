#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/funcs/PrimitiveUtils.h> //primCalcNormal and primTriangulateQuads
#include <limits>
#include <iostream>
#include <algorithm>
using std::cout;
namespace zeno {
/**
 * @brief 这个节点是用来为PBD的布料模拟初始化invMass, restLen和restAng的。
 * 注意不要和软体模拟的初始化混用，因为其中的invMass定义不同。
 * 
 */
struct PBDClothInit : zeno::INode {

    /**
     * @brief 计算两个面夹角的辅助函数。注意按照论文Muller2006中Fig.4的顺序。1-2是共享边。
     * 
     * @param p1 顶点1位置
     * @param p2 顶点2位置
     * @param p3 顶点3位置
     * @param p4 顶点4位置
     * @return float 两个面夹角
     */
    float computeAng(
        const vec3f &p1,
        const vec3f &p2, 
        const vec3f &p3, 
        const vec3f &p4)
    {
        auto n1 = cross((p2 - p1), (p3 - p1));
        n1 = n1 / length(n1);
        auto n2 = cross((p2 - p1), (p4 - p1));
        n2 = n2 / length(n2);
        auto res = abs(dot(n1, n2)); //只计算锐角。TODO:是否该如此存疑。
        if(res<-1.0) res = -1.0;  
        if(res>1.0)  res = 1.0;
        return acos(res);
    }

    //找到other中与self不同的那个点的位置（other的位置），可能是0,1,2
    int cmp33(vec3i self, vec3i other)
    {
        std::vector<bool> isSame{false,false,false};
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                if(self[j] == other[i])//注意是other[i]
                {
                    isSame[i] = true;
                    break;
                }
            }
        }

        for (size_t k = 0; k < 3; k++)
        {
            if(isSame[k] == false)
                return k;
        }
        return -1; //没找到
    }


    /**
     * @brief 计算所有原角度
     * 
     * @param pos 顶点
     * @param tris 三角面
     * @param adjTriId 邻接三角面在tris中的编号
     * @param restAng 原角度, 最多有三个，每个邻接三角面对应一个。没有则存-1
     */
    void initRestAng(
        const AttrVector<vec3f> &pos,
        const AttrVector<vec3i> &tris,
        const AttrVector<vec3i> &adj4th,
        std::vector<vec3f> &restAng
    ) 
    {
        for(int i = 0; i < tris.size(); i++)
        {
            const vec3i & self = tris[i];
            int pid1 = self[0]; //先取出本身的三个点的编号
            int pid2 = self[1];
            int pid3 = self[2];
            int pid4 = -1;      //第四个点要搜索另一个面的点
            for (int j = 0; j < 3; j++)
            {
                //因为可能有-1，所以默认值设置负的很大的数字来表示不存在该角度（因为不存在该邻接面）
                restAng[i][j] = std::numeric_limits<float>::lowest(); 
                if(adj4th[i][j] != -1) //如果是负一证明该边没有邻接面
                {
                    // vec3i other = tris[adjTriId[i][j]]; //取出一个邻接三角面
                    // //再找到非本身三个点的那个点，作为第四个点。
                    // pid4 = other[cmp33(self, other)];
                    pid4 = adj4th[i][j];
                    if(pid4==-1)
                        throw std::runtime_error("the adjacent tris failed");
                    restAng[i][j] = computeAng(pos[pid1],pos[pid2],pos[pid3],pos[pid4]);
                }
            }
        }
    }

    /**
     * @brief 计算所有原长
     * 
     * @param pos 顶点
     * @param edge 边连接关系
     * @param restLen 原长
     */
    void initRestLen(
        const AttrVector<vec3f> &pos,
        const AttrVector<vec2i> &edge,
        std::vector<float> &restLen
    ) 
    {
        for(int i = 0; i < edge.size(); i++)
            restLen[i] = length((pos[edge[i][0]] - pos[edge[i][1]]));
    }


    /**
     * @brief 计算所有质量倒数
     * 
     * @param pos 顶点
     * @param tris 三角面
     * @param areaDensity 面密度
     * @param invMass 质量倒数
     */
    void initInvMass(        
        const AttrVector<vec3f> &pos,
        const AttrVector<vec3i> &tris,
        const float areaDensity,
        std::vector<float> &invMass
        ) 
    {
        for(int i = 0; i < tris.size(); i++)
        {
            float area = abs(length(cross(pos[tris[i][1]] - pos[tris[i][0]],  pos[tris[i][1]] - pos[tris[i][2]]))/2.0);
            float pInvMass = 0.0;
            pInvMass = areaDensity * area / 3.0;
            for (int j = 0; j < 3; j++)
                invMass[tris[i][j]] += pInvMass;
        }
    }

    void findTriNeighbors(std::vector<vec3i> &tris, std::vector<int> &neighbors) 
    {
        neighbors.resize(3*tris.size());

        std::vector<vec3i> edges;
        
        for (auto i = 0; i < tris.size(); i++) {
            for (auto j = 0; j < 3; j++) {
                auto id0 = tris[i][j];
                auto id1 = tris[i][(j + 1) % 3];
                edges.push_back(vec3i{std::min(id0, id1), std::max(id0, id1), 3 * i + j});
            }
        }

        // sort so common edges are next to each other
        std::sort(edges.begin(), edges.end(),
                  [](vec3i &a, vec3i &b) { return ((a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1])) ? -1 : 1; });

        std::fill(neighbors.begin(), neighbors.end(), -1);

        auto nr = 0;
        while (nr < edges.size()) {
            auto e0 = edges[nr];
            nr++;
            if (nr < edges.size()) {
                auto e1 = edges[nr];
                if (e0[0] == e1[0] && e0[1] == e1[1]) {
                    neighbors[e0[2]] = e1[2];
                    neighbors[e1[2]] = e0[2];
                }
                nr++;
            }
        }
    }

public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &pos = prim->verts;
        auto &tris = prim->tris;
        auto &edge = prim->lines;

        //面密度，用来算invMass的参数
        auto areaDensity = get_input<zeno::NumericObject>("areaDensity")->get<int>();

        auto &invMass = prim->verts.add_attr<float>("invMass");
        auto &restLen = prim->lines.add_attr<float>("restLen");
        auto &restAng = prim->tris.add_attr<vec3f>("restAng");


        auto &triNeighbors = prim->verts.add_attr<int>("triNeighbors");
        // auto &adjTriId = prim->tris.attr<vec3i>("adjTriId");
        auto &adj4th = prim->tris.attr<vec3i>("adj4th");

        findTriNeighbors(tris, triNeighbors);

        initInvMass(pos,tris,areaDensity,invMass);
        initRestLen(pos,edge,restLen);
        // initRestAng(pos,tris,adjTriId,restAng);
        initRestAng(pos,tris,adj4th,restAng);

        //初始化速度和前一时刻位置变量
        auto &vel = prim->verts.add_attr<vec3f>("vel");
        auto &prevPos = prim->verts.add_attr<vec3f>("prevPos");

        set_output("outPrim", std::move(prim));

    };
};

ZENDEFNODE(PBDClothInit, {// inputs:
                          {
                              {gParamType_Primitive, "prim"},
                              {gParamType_Float, "areaDensity", "1.0"}
                          },
                          // outputs:
                          {
                              {"outPrim"}
                          },
                          // params:
                          {},
                          //category
                          {"PBD"}});

} // namespace zeno