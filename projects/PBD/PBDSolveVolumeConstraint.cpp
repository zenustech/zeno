#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>

namespace zeno {
struct PBDSolveVolumeConstraint : zeno::INode {
private:
    /**
     * @brief 求解PBD所有体积约束。目前采用Gauss-Seidel方式（难以并行）。
     * 
     * @param pos 点位置
     * @param tet 四面体的四个顶点连接关系
     * @param volumeCompliance 柔度（越小约束越强，最小为0）
     * @param dt 时间步长
     * @param restVol 原体积
     * @param invMass 点质量的倒数
     */
    void solveVolumeConstraint(
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec4i> &tet,
        const float volumeCompliance,
        const float dt,
        const std::vector<float> & restVol,
        const std::vector<float> & invMass
                    )
    {
        float alphaVol = volumeCompliance / dt / dt;
        vec3f grad[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};

        for (int i = 0; i < tet.size(); i++)
        {
            vec4i id{-1,-1,-1,-1};

            for (int j = 0; j < 4; j++)
                id[j] = tet[i][j];
            
            grad[0] = cross((pos[id[3]] - pos[id[1]]), (pos[id[2]] - pos[id[1]]));
            grad[1] = cross((pos[id[2]] - pos[id[0]]), (pos[id[3]] - pos[id[0]]));
            grad[2] = cross((pos[id[3]] - pos[id[0]]), (pos[id[1]] - pos[id[0]]));
            grad[3] = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));

            float w = 0.0;
            for (int j = 0; j < 4; j++)
                w += invMass[id[j]] * (length(grad[j])) * (length(grad[j])) ;

            float vol = tetVolume(pos, tet, i);
            float C = (vol - restVol[i]) * 6.0;
            float s = -C /(w + alphaVol);
            
            for (int j = 0; j < 4; j++)
                pos[id[j]] += grad[j] * s * invMass[id[j]];
        }
    }

    /**
     * @brief 计算当前体积的辅助函数
     * 
     * @param pos 点位置
     * @param tet 四面体顶点连接关系
     * @param i 四面体编号
     * @return float 四面体体积
     */
    float tetVolume(zeno::AttrVector<zeno::vec3f> &pos,
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


public:
    virtual void apply() override {
        //get data
        auto prim = get_input<PrimitiveObject>("prim");

        auto volumeCompliance = get_input<zeno::NumericObject>("volumeCompliance")->get<float>();
        float dt = prim->userData().getLiterial<float>("dt");

        auto &pos = prim->verts;
        auto &tet = prim->quads;
        auto &restVol = prim->quads.attr<float>("restVol");
        auto &invMass = prim->verts.attr<float>("invMass");

        // solve
        solveVolumeConstraint(pos, tet, volumeCompliance, dt, restVol, invMass);

        // output
        set_output("outPos", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveVolumeConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "volumeCompliance", "0.0"}
                },
                 // outputs:
                 {"outPos"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno
