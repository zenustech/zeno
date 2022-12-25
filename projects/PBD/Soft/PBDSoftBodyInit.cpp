#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>

namespace zeno {

/**
 * @brief 计算PBD四面体的初始质量倒数、初始长度、初始体积。
 * 这个节点应该在导入四面体网格之后，并且只计算一次。
 * 节点输出时附加属性：初始质量倒数invMass、初始长度restLen、初始体积restVol。
 */
struct PBDSoftBodyInit : zeno::INode {

private:
    /**
     * @brief 计算当前体积的辅助函数
     * 
     * @param pos 点位置
     * @param tet 四面体顶点连接关系
     * @param i 四面体编号
     * @return float 四面体体积
     */
    float tetVolume(
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec4i> &tet,
        int i)
    {
        auto id = vec4i(-1, -1, -1, -1);
        for (int j = 0; j < 4; j++)
            id[j] = tet[i][j];
        auto temp = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));
        auto res = dot(temp, pos[id[3]] - pos[id[0]]);
        res *= 1.0 / 6.0;
        return res;
    }

    /**
     * @brief 计算四面体的初始质量倒数、初始长度、初始体积
     * 
     * @param prim 四面体网格。
     * 输入时应该包含顶点、边连接关系和四面体四个顶点连接关系。
     * 输出时附加属性：初始质量倒数、初始长度、初始体积。
     */
    void initGeo(
        AttrVector<vec3f> &pos,
        AttrVector<vec2i> &edge,
        AttrVector<vec4i> &tet,
        std::vector<float> & restLen,
        std::vector<float> & restVol,
        std::vector<float> & invMass
    )
    {
        //calculate restVol and restLen
        for(int i = 0; i < tet.size(); i++)
            restVol[i] = tetVolume(pos, tet, i);
        for(int i = 0; i < edge.size(); i++)
            restLen[i] = length((pos[edge[i][0]] - pos[edge[i][1]]));
        
        //calculate invMass
        for(int i = 0; i < tet.size(); i++)
        {
            float pInvMass = 0.0;
            if (restVol[i] > 0.0)
                pInvMass = 1.0 / (restVol[i] / 4.0);
            for (int j = 0; j < 4; j++)
                invMass[tet[i][j]] += pInvMass;
        }
    }

public:
        virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;

        auto &restLen = prim->lines.add_attr<float>("restLen");
        auto &restVol = prim->quads.add_attr<float>("restVol");
        auto &invMass = prim->verts.add_attr<float>("invMass");

        restLen.resize(edge.size());
        restVol.resize(tet.size());
        invMass.resize(pos.size());

        initGeo(pos, edge, tet, restLen, restVol, invMass);

        auto &vel = prim->verts.add_attr<vec3f>("vel");
        auto &prevPos = prim->verts.add_attr<vec3f>("prevPos");

        set_output("outPrim",std::move(prim));
    };
};

ZENDEFNODE(PBDSoftBodyInit, {// inputs:
                 {"prim"},
                 // outputs:
                {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});
} // namespace zeno


