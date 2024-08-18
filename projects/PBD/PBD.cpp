#include <iostream>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
namespace zeno {
struct PBD : zeno::INode {
private:
    //physical param
    zeno::vec3f g{0, -10.0, 0};
    int numSubsteps = 10;
    float dt = 1.0 / 60.0 / numSubsteps;
    float edgeCompliance = 100.0;
    float volumeCompliance = 0.0;

    std::vector<float> restLen;
    std::vector<float> restVol;
    std::vector<float> invMass;

    std::vector<zeno::vec3f> prevPos;
    std::vector<zeno::vec3f> vel;

    int numParticles;
    int numEdges;
    int numTets;
    int numSurfs;

    float tetVolume(zeno::AttrVector<zeno::vec3f> &pos,
                    const zeno::AttrVector<zeno::vec4i> &tet, int i)
    {
        auto id = vec4i(-1, -1, -1, -1);
        for (int j = 0; j < 4; j++)
            id[j] = tet[i][j];
        auto temp = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));
        auto res = dot(temp, pos[id[3]] - pos[id[0]]);
        res *= 1.0 / 6.0;
        return res;
    }

    void init_physics(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;
        for(int i = 0; i < tet.size(); i++)
            restVol[i] = tetVolume(pos, tet, i);
        for(int i = 0; i < edge.size(); i++)
            restLen[i] = length((pos[edge[i][0]] - pos[edge[i][1]]));
    }

    void init_invMass(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;

        for(int i = 0; i < tet.size(); i++)
        {
            float pInvMass = 0.0;
            if (restVol[i] > 0.0)
                pInvMass = 1.0 / (restVol[i] / 4.0);
            for (int j = 0; j < 4; j++)
                invMass[tet[i][j]] += pInvMass;
        }
    }


    void initGeo(PrimitiveObject *prim)
    {
        restLen.resize(prim->lines.size());
        restVol.resize(prim->quads.size());
        invMass.resize(prim->verts.size());

        init_physics(prim);
        init_invMass(prim);

        numParticles = prim->verts.size();
        numEdges = prim->lines.size();
        numTets = prim->quads.size();
        numSurfs = prim->tris.size();

        prevPos.resize(numParticles);
        vel.resize(numParticles);
    }

    void preSolve(  zeno::AttrVector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos,
                    std::vector<zeno::vec3f> &vel)
    {
        for (int i = 0; i < pos.size(); i++) 
        {
            prevPos[i] = pos[i];
            vel[i] += g * dt;
            pos[i] += vel[i] * dt;
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }
        }
    }

    void solveEdge( zeno::AttrVector<zeno::vec3f> &pos,
                    const zeno::AttrVector<zeno::vec2i> &edge)
    {
        float alpha = edgeCompliance / dt / dt;
        zeno::vec3f grads{0, 0, 0};
        for (int i = 0; i < numEdges; i++) 
        {
            int id0 = edge[i][0];
            int id1 = edge[i][1];

            grads = pos[id0] - pos[id1];
            float Len = length(grads);
            grads /= Len;
            float C = Len - restLen[i];
            float w = invMass[id0] + invMass[id1];
            float s = -C / (w + alpha);

            pos[id0] += grads *   s * invMass[id0];
            pos[id1] += grads * (-s * invMass[id1]);
        }
    }
    void solveVolume(zeno::AttrVector<zeno::vec3f> &pos,
                    const zeno::AttrVector<zeno::vec4i> &tet)
    {
        float alphaVol = volumeCompliance / dt / dt;
        vec3f gradsVol[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};

        for (int i = 0; i < numTets; i++)
        {
            vec4i id{-1,-1,-1,-1};

            for (int j = 0; j < 4; j++)
                id[j] = tet[i][j];
            
            gradsVol[0] = cross((pos[id[3]] - pos[id[1]]), (pos[id[2]] - pos[id[1]]));
            gradsVol[1] = cross((pos[id[2]] - pos[id[0]]), (pos[id[3]] - pos[id[0]]));
            gradsVol[2] = cross((pos[id[3]] - pos[id[0]]), (pos[id[1]] - pos[id[0]]));
            gradsVol[3] = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));

            float w = 0.0;
            for (int j = 0; j < 4; j++)
                w += invMass[id[j]] * (length(gradsVol[j])) * (length(gradsVol[j])) ;

            float vol = tetVolume(pos, tet, i);
            float C = (vol - restVol[i]) * 6.0;
            float s = -C /(w + alphaVol);
            
            for (int j = 0; j < 4; j++)
                pos[tet[i][j]] += gradsVol[j] * s * invMass[id[j]];
        }
    }

    void postSolve(const zeno::AttrVector<zeno::vec3f> &pos,
                   const std::vector<zeno::vec3f> &prevPos,
                   std::vector<zeno::vec3f> &vel)
    {
        for (int i = 0; i < pos.size(); i++) 
            vel[i] = (pos[i] - prevPos[i]) / dt;
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto external_force = get_input<zeno::NumericObject>("external_force")->get<zeno::vec3f>();

        g = external_force;

        numSubsteps = get_input<zeno::NumericObject>("numSubsteps")->get<int>();
        edgeCompliance = get_input<zeno::NumericObject>("edgeCompliance")->get<float>();
        volumeCompliance = get_input<zeno::NumericObject>("volumeCompliance")->get<float>();

        dt = 1.0/60.0/numSubsteps;
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;
        auto &surf = prim->tris;

        static bool firstTime = true;
        if(firstTime)
        {
            initGeo(prim.get());
            firstTime = false;
        }

        for (size_t i = 0; i < 1; i++)
        {
            for (int steps = 0; steps < numSubsteps; steps++) 
            {
                preSolve(pos, prevPos, vel);
                solveEdge(pos, edge);
                solveVolume(pos, tet);
                postSolve(pos, prevPos, vel);
            }
        }

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBD, {// inputs:
                 {
                    {gParamType_Primitive, "prim"},
                    {gParamType_Vec3f, "external_force", "0.0, -10.0, 0.0"},
                    {gParamType_Int, "numSubsteps", "10"},
                    {gParamType_Float, "edgeCompliance", "100.0"},
                    {gParamType_Float, "volumeCompliance", "0.0"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno