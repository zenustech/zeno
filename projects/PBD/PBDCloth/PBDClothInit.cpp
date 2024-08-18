#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <limits>
#include <iostream>
#include <algorithm>
// #include <zeno/funcs/PrimitiveUtils.h> //primCalcNormal and primTriangulatequads

namespace zeno {
/**
 * @brief 这个节点是用来为PBD的布料模拟初始化invMass, restLen和restAng的。
 * 注意不要和软体模拟的初始化混用，因为其中的invMass定义不同。
 * 
 */
struct PBDClothInit : zeno::INode {

    /**
     * @brief 找到共享边。
     * 
     * @param tris 三角形三个顶点的连接关系
     * @param sharedEdges 用于查询共享边的。给定一个边的编号，则存储的就是它的共享边的编号。
     */
    void initSharedEdges(std::vector<vec3i> &tris, std::vector<int> &sharedEdges) 
    {
        sharedEdges.resize(3*tris.size());

        //这个edges是有重复的。
        // 它的前两个参数是顶点编号，组成一个边，第三个参数代表共享边的编号，无共享边则为-1。
        std::vector<vec3i> edges;
        
        for (auto i = 0; i < tris.size(); i++) {
            for (auto j = 0; j < 3; j++) {
                auto id0 = tris[i][j];
                auto id1 = tris[i][(j + 1) % 3];
                edges.push_back(vec3i{std::min(id0, id1), std::max(id0, id1), 3 * i + j});
            }
        }


        // sort so shared edges are next to each other
        std::sort(edges.begin(), edges.end(),
                  [](vec3i &a, vec3i &b) { return ((a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1])); });


        std::fill(sharedEdges.begin(), sharedEdges.end(), -1);

        auto nr = 0;
        while (nr < edges.size()) 
        {
            auto e0 = edges[nr];
            nr++;
            if (nr < edges.size()) 
            {
                auto e1 = edges[nr];
                if (e0[0] == e1[0] && e0[1] == e1[1]) 
                {
                    sharedEdges[e0[2]] = e1[2];
                    sharedEdges[e1[2]] = e0[2];
                }
                nr++;
            }
        }
    }

    /**
     * @brief 计算边和三角形对。
     * 
     * @param tris 三角形顶点连接关系。
     * @param sharedEdges 用于查询共享边。
     * @param edges 输出的边。无重复。
     * @param quads 输出的一对邻接三角形的四个顶点。
     */
    void initEdgesAndQuads(const std::vector<vec3i>& tris, const std::vector<int> &sharedEdges, std::vector<vec2i> &edges, std::vector<vec4i>& quads) 
    {
        edges.clear();
        quads.clear();

        for (auto i = 0; i < tris.size(); i++) 
        {
            for (auto j = 0; j < 3; j++) 
            {
                auto id0 = tris[i][j];
                auto id1 = tris[i][(j + 1) % 3];

                // 建立edges
                auto n = sharedEdges[3 * i + j];
                if (n < 0 || id0 < id1) 
                {
                    edges.push_back(vec2i{id0,id1});
                }

                // 建立三角形对
                if(n>=0)
                {
                    auto ni = std::floor(n/3); //global number
                    auto nj = n % 3; //local number
                    auto id2 = tris[i][(j+2)%3];
                    auto id3 = tris[ni][(nj+2)%3];
                    quads.push_back(vec4i{id0,id1,id2,id3});
                }
            }
        }
    }

    /**
     * @brief 计算初始集合数据。包括质量倒数、原长度、对角原长度
     * 
     * @param pos 输入：粒子位置
     * @param edges 输入：边
     * @param tris 输入：三角形
     * @param quads 输入：三角形对
     * @param invMass 输出：质量倒数
     * @param restLen 输出：原长度
     * @param bendingRestLen 输出：对角原长。用于对角方法的bending
     */
    void initGeometry(
        const std::vector<vec3f> & pos,
        const std::vector<vec2i> & edges,
        const std::vector<vec3i> & tris,
        const std::vector<vec4i> & quads,
        std::vector<float> & invMass,
        std::vector<float> & restLen,
        std::vector<float> & bendingRestLen
        )
    {
        std::fill(invMass.begin(),invMass.end(), 0.0);

        vec3f e0{0.0, 0.0, 0.0};
        vec3f e1{0.0, 0.0, 0.0};
        vec3f c{0.0, 0.0, 0.0};

        for (auto i = 0; i < tris.size(); i++) 
        {
            auto id0 = tris[i][0];
            auto id1 = tris[i][1];
            auto id2 = tris[i][2];

            e0 = pos[id1] - pos[id0];
            e1 = pos[id2] - pos[id0];
            c = cross(e0,e1);

            auto A = 0.5 * length(c);

            auto pInvMass = A > 0.0 ? (1.0 / A / 3.0) : 0.0;
            invMass[id0] += pInvMass;
            invMass[id1] += pInvMass;
            invMass[id2] += pInvMass;
        }

        for (auto i = 0; i < edges.size(); i++) 
        {
            auto id0 = edges[i][0];
            auto id1 = edges[i][1];
            restLen[i] = length(pos[id0] - pos[id1]);
        }

        for (auto i = 0; i < quads.size(); i++) 
        {
            auto id0 = quads[i][2];
            auto id1 = quads[i][3];
            bendingRestLen[i] = length(pos[id0] - pos[id1]);
        }


        // 设置挂载点，为左上角和右上角点。
        auto minX = std::numeric_limits<float>::max();
        auto maxX = -std::numeric_limits<float>::max();
        auto maxY = -std::numeric_limits<float>::max();
        for (auto i = 0; i < pos.size(); i++) 
        {
            minX = std::min(minX, pos[i][0]);
            maxX = std::max(maxX, pos[i][0]);
            maxY = std::max(maxY, pos[i][1]);
        }
        auto eps = 0.0001;

        for (auto i = 0; i < pos.size(); i++) 
        {
            auto x = pos[i][0];
            auto y = pos[i][1];
            if ((y > maxY - eps) && (x < minX + eps || x > maxX - eps))
                invMass[i] = 0.0;
        }
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &pos = prim->verts;
        auto &tris = prim->tris;

        // 共享边。 sharedEdges是用于寻找共享边的辅助数据结构。
        auto &sharedEdges = prim->add_attr<int>("sharedEdges");
        initSharedEdges(tris, sharedEdges);

        //建立边连接关系：edges。这次每个边只加入一次。
        auto &edges = prim->edges;
        auto &quads = prim->quads;
        initEdgesAndQuads(tris, sharedEdges, edges, quads);

        // 计算invMass和restLen和 bendingRestLen
        // bendingRestLen是使用对角距离法来计算bending的时候的restLen，与edges上的restLen不同，是专门针对quads第3和第4个元素的len。
        auto &invMass = prim->verts.add_attr<float>("invMass");
        auto &restLen = prim->edges.add_attr<float>("restLen");
        auto &bendingRestLen = prim->quads.add_attr<float>("bendingRestLen");
        initGeometry(pos,edges,tris,quads,invMass,restLen,bendingRestLen);

        //初始化速度和前一时刻位置变量
        auto &vel = prim->verts.add_attr<vec3f>("vel");
        auto &prevPos = prim->verts.add_attr<vec3f>("prevPos");

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDClothInit, {// inputs:
                          {
                              {gParamType_Primitive, "prim"}
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