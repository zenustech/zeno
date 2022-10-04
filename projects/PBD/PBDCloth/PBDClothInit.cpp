#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
namespace zeno {
struct PBDClothInit : zeno::INode {


    /**
     * @brief 计算两个面夹角的辅助函数。
     * 
     * @param p0 顶点0位置
     * @param p1 顶点1位置
     * @param p2 顶点2位置
     * @param p3 顶点3位置
     * @return float 两个面夹角
     */
    float computeAng(
        const vec3f &p0,
        const vec3f &p1, 
        const vec3f &p2, 
        const vec3f &p3)
    {
        auto n1 = cross((p1 - p0), (p2 - p0));
        n1 = n1 / length(n1);
        auto n2 = cross((p1 - p0), (p3 - p0));
        n2 = n2 / length(n2);
        auto res = dot(n1, n2);
        if(res<-1.0) res = -1.0;
        if(res>1.0)  res = 1.0;
        return acos(res);
    }

    /**
     * @brief 计算所有原角度
     * 
     * @param pos 顶点
     * @param quads 四个顶点连成一块布片
     * @param restAng 原角度
     */
    void initRestAng(
        const AttrVector<vec3f> &pos,
        const AttrVector<vec4i> &quads,
        std::vector<float> &restAng
    ) 
    {
        for(int i = 0; i < quads.size(); i++)
        {
            const vec3f &p1 = pos[quads[i][0]];
            const vec3f &p2 = pos[quads[i][1]];
            const vec3f &p3 = pos[quads[i][2]];
            const vec3f &p4 = pos[quads[i][3]];
            restAng[i] = computeAng(p1,p2,p3,p4);
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
        AttrVector<vec3f> &pos,
        AttrVector<vec2i> &edge,
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
     * @param quads 四个点连接关系
     * @param invMass 质量倒数
     * @param areaDensity 面密度
     */
    void initInvMass(        
        const AttrVector<vec3f> &pos,
        const AttrVector<vec4i> &quads,
        std::vector<float> &invMass,
        const float areaDensity) 
    {
        for(int i = 0; i < quads.size(); i++)
        {
            float quad_area = dot(pos[quads[i][1]] - pos[quads[i][0]],  pos[quads[i][3]] - pos[quads[i][2]]);
            float pInvMass = 0.0;
            pInvMass = areaDensity * quad_area;
            for (int j = 0; j < 4; j++)
                invMass[quads[i][j]] += pInvMass;
        }
    }

    void buildEdge(const AttrVector<vec4i> &quads, AttrVector<vec2i> &edge)
    {
        for(int i= 0; i<quads.size(); i++)
        {
            int q0 = quads[i][0];
            int q1 = quads[i][1];
            int q2 = quads[i][2];
            int q3 = quads[i][3];
            edge[i + 0] = vec2i(q0,q1);
            edge[i + 1] = vec2i(q1,q3);            
            edge[i + 2] = vec2i(q3,q0);
            edge[i + 3] = vec2i(q1,q2);           
            edge[i + 4] = vec2i(q3,q2);
        }
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &pos = prim->verts;
        auto &quads = prim->quads;
        auto &edge = prim->lines;

        auto areaDensity = get_input<zeno::NumericObject>("areaDensity")->get<int>();

        auto &invMass = prim->verts.add_attr<float>("invMass");
        auto &restLen = prim->lines.add_attr<float>("restLen");
        auto &restAng = prim->quads.add_attr<float>("restAng");
        invMass.resize(prim->verts.size());
        restLen.resize(prim->lines.size());
        invMass.resize(prim->quads.size());

        initInvMass(pos,quads,invMass,areaDensity);
        initRestLen(pos,edge,restLen);
        initRestAng(pos,quads,restAng);

        set_output("outPrim", std::move(prim));

    };
};

ZENDEFNODE(PBDClothInit, {// inputs:
                          {
                              {"PrimitiveObject", "prim"},
                              {"float", "areaDensity", "1.0"}
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