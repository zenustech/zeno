#include <iostream>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include "bunnyMesh.h"
namespace zeno {
struct PBD : zeno::INode {
private:
    //physical param
    zeno::vec3f g{0, -9.8, 0};
    int numSubsteps = 10;
    float dt = 1.0 / 60.0 / numSubsteps;
    float edgeCompliance = 100.0;
    float volumeCompliance = 0.0;

    std::vector<float> restLen;
    std::vector<float> restVol;
    std::vector<float> invMass;

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
    }

    void initPos(PrimitiveObject *prim)
    {
        auto &pos = prim->verts;

        for (int i = 0; i < pos.size(); i++) 
            pos[i] += vec3f(0.5,1,0);
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
            float Len = sqrt(grads[0] * grads[0] + grads[1] * grads[1] + grads[2] * grads[2]);
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
            vec4i id =  vec4i(-1,-1,-1,-1);

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

        static bool firstTime = true;
        if(firstTime)
        {
            initGeo(prim.get());
            initPos(prim.get());
            firstTime = false;
        }

        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;
        auto &surf = prim->tris;

        std::vector<zeno::vec3f> prevPos = pos;
        std::vector<zeno::vec3f> vel(numParticles);

        for (int steps = 0; steps < numSubsteps * 5; steps++) 
        {
            preSolve(pos, prevPos, vel);
            solveEdge(pos, edge);
            solveVolume(pos, tet);
            postSolve(pos, prevPos, vel);
        }

        set_output("prim", std::move(prim));
    };
};

ZENDEFNODE(PBD, {// inputs:
                 {"prim"},
                 // outputs:
                 {"prim"},
                 // params:
                 {{"vec3f", "external_force", "0, -9.8, 0"}},
                 //category
                 {"PBD"}});

} // namespace zeno