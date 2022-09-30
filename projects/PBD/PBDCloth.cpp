#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <Eigen/Eigen>
namespace zeno {
struct PBDCloth : zeno::INode {
private:
    //physical param
    zeno::vec3f externForce{0, -10.0, 0};
    int numSubsteps = 10;
    float dt = 1.0 / 60.0 / numSubsteps;
    float edgeCompliance = 100.0;
    float volumeCompliance = 0.0;

    std::vector<float> restLen;
    std::vector<float> restVol;
    std::vector<float> invMass;
    std::vector<float> restAngle;

    std::vector<zeno::vec3f> prevPos;
    std::vector<zeno::vec3f> vel;

    int numParticles;
    int numEdges;
    int numTets;
    int numSurfs;

    float computeAngle( zeno::AttrVector<zeno::vec3f> &pos,
                        const zeno::AttrVector<zeno::vec4i> &tet,
                        int i)
    {
        auto id = vec4i(-1, -1, -1, -1);
        for (int j = 0; j < 4; j++)
            id[j] = tet[i][j];

        auto first = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));
        first = first / length(first) ;
        auto second = cross((pos[id[1]] - pos[id[0]]), (pos[id[3]] - pos[id[0]]));
        second = second / length(second) ;
        auto res = dot(first, second);
        return res;
    }

    void initRestLenAndRestAngle(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;
        auto &tris = prim->tris;
        for(int i = 0; i < tris.size(); i++)
            restAngle[i] = computeAngle(pos, tet, i);
        for(int i = 0; i < edge.size(); i++)
            restLen[i] = length((pos[edge[i][0]] - pos[edge[i][1]]));
    }

    void initInvMass(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &tet = prim->quads;

        for(int i = 0; i < tet.size(); i++)
        {
            float pInvMass = 0.0;
            if (restVol[i] > 0.0)
                pInvMass = 1.0 / (restVol[i] / 3.0);
            for (int j = 0; j < 4; j++)
                invMass[tet[i][j]] += pInvMass;
        }
    }


    void initGeo(PrimitiveObject *prim)
    {
        restLen.resize(prim->lines.size());
        restVol.resize(prim->quads.size());
        invMass.resize(prim->verts.size());

        initRestLenAndRestAngle(prim);
        initInvMass(prim);

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
            vel[i] += (externForce) * dt;
            pos[i] += vel[i] * dt;
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }
        }
    }

    void solveDistanceConstraint( zeno::AttrVector<zeno::vec3f> &pos,
                    const zeno::AttrVector<zeno::vec2i> &edge)
    {
        float alpha = edgeCompliance / dt / dt;
        zeno::vec3f grad{0, 0, 0};
        for (int i = 0; i < numEdges; i++) 
        {
            int id0 = edge[i][0];
            int id1 = edge[i][1];

            grad = pos[id0] - pos[id1];
            float Len = length(grad);
            grad /= Len;
            float C = Len - restLen[i];
            float w = invMass[id0] + invMass[id1];
            float s = -C / (w + alpha);

            pos[id0] += grad *   s * invMass[id0];
            pos[id1] += grad * (-s * invMass[id1]);
        }
    }
    void solveBendingConstraint(
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec4i> &tet,
        const std::vector<Eigen::Matrix4r> &stiffMatrix
        )
    {
        float alpha = BendingCompliance / dt / dt;
        float lambda = 0.0;

        for (int i = 0; i < numTets; i++)
        {
            vec4i id{-1,-1,-1,-1};
            for (int j = 0; j < 4; j++)
                id[j] = tet[i][j];
            
            vec3f grad[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};
            Eigen::Matrix4r &Q = stiffMatrix[i];

            for (int k = 0; k < 4; k++)
		        for (int j = 0; j < 4; j++)
			        grad[j] += Q(j,k) * *pos[id[k]];

            float w = 0.0;
            for (int j = 0; j < 4; j++)
                w += invMass[id[j]] * (length(grad[j])) * (length(grad[j])) ;

            float energy = 0.0;
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 4; j++)
                    energy += Q(j, k) * (x[k]->dot(*x[j]));
            energy *= 0.5;


            // compute impulse-based scaling factor
            const float s = -(energy + alpha * lambda) / (w + alpha);
            lambda += s;

            vec4f dpos;
            //注意对应关系0对2, 1对3...
            dpos[0] = (s * invMass[2]) * grad[2];
            dpos[1] = (s * invMass[3]) * grad[3];
            dpos[2] = (s * invMass[0]) * grad[0];
            dpos[3] = (s * invMass[1]) * grad[1];
            
            for (int j = 0; j < 4; j++)
                pos[id[j]] += dpos[j];
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

        externForce = get_input<zeno::NumericObject>("externForce")->get<zeno::vec3f>();

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
                solveDistanceConstraint(pos, edge);
                solveBendingConstraint(pos, tet);
                postSolve(pos, prevPos, vel);
            }
        }

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDCloth, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"vec3f", "externForce", "0.0, -10.0, 0.0"},
                    {"int", "numSubsteps", "10"},
                    {"float", "edgeCompliance", "100.0"},
                    {"float", "volumeCompliance", "0.0"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno