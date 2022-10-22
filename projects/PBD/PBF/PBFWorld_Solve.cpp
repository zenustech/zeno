#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include "./PBFWorld.h"
#include "./SPHKernelFuncs.h"
#include "../Utils/myPrint.h"//debug
using namespace zeno;


struct PBFWorld_Solve : zeno::INode {

    //核心步骤
    void solve(PBFWorld* data)
    {
        //计算lambda
        computeLambda(data);

        //计算dpos
        computeDpos(data);

        //apply the dpos to the pos
        auto & pos = data->prim->verts;
        for (size_t i = 0; i < data->numParticles; i++)
            pos[i] += data->dpos[i];
    }
    

    void computeLambda(PBFWorld* data)
    {
        data->lambda.clear();
        data->lambda.resize(data->numParticles);
        const auto &pos = data->prim->verts;//这里只访问，不修改
        const auto &neighborList = data->neighborList;//这里只访问，不修改
        for (size_t i = 0; i < data->numParticles; i++)
        {
            vec3f gradI{0.0, 0.0, 0.0};
            float sumSqr = 0.0;
            float densityCons = 0.0;

            for (size_t j = 0; j < neighborList[i].size(); j++)
            {
                int pj = neighborList[i][j];//pj是第j个邻居的下标
                vec3f distVec = pos[i] - pos[pj];
                vec3f gradJ = kernelSpikyGradient(distVec, data->h);
                gradI += gradJ;
                sumSqr += dot(gradJ, gradJ);
                densityCons += kernelPoly6(length(distVec), data->h);
            }
            densityCons = (data->mass * densityCons / data->rho0) - 1.0;

            sumSqr += dot(gradI, gradI);
            //compute lambda
            data->lambda[i] = (-densityCons) / (sumSqr + data->lambdaEpsilon);
        }
    }

    void computeDpos(PBFWorld* data)
    {
        data->dpos.clear();
        data->dpos.resize(data->numParticles);
        const auto &pos = data->prim->verts; //这里只访问，不修改
        const auto &neighborList = data->neighborList;//这里只访问，不修改
        for (size_t i = 0; i < data->numParticles; i++)
        {
            vec3f dposI{0.0, 0.0, 0.0};
            for (size_t j = 0; j < neighborList[i].size(); j++)
            {
                int pj = neighborList[i][j];
                vec3f distVec = pos[i] - pos[pj];

                float sCorr = computeScorr(distVec, data->coeffDq, data->coeffK, data->h);
                dposI += (data->lambda[i] + data->lambda[pj] + sCorr) * kernelSpikyGradient(distVec, data->h);
            }
            dposI /= data->rho0;
            data->dpos[i] = dposI;
        }
    }

    //helper for computeDpos()
    inline float computeScorr(const vec3f& distVec, const float coeffDq, const float coeffK, const float h)
    {
        float x = kernelPoly6(length(distVec), h) / kernelPoly6(coeffDq * h, h);
        x = x * x * x * x;
        return (-coeffK) * x;
    }

     virtual void apply() override{
        auto data = get_input<PBFWorld>("PBFWorld");
        for(int i=0; i<data->numSubsteps; i++)
            solve(data.get());
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(
    PBFWorld_Solve,
    {
        {
            {"PBFWorld"}
        },
        {"PBFWorld"},
        {},
        {"PBD"}
    }
);