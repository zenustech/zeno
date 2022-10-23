#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include "./PBFWorld.h"
// #include "./SPHKernelFuncs.h"
#include "./SPHKernels.h"
#include "../Utils/myPrint.h"//debug
#include <cstdio>//debug
using namespace zeno;


struct PBFWorld_Solve : zeno::INode {

    //核心步骤
    void solve(PBFWorld* data, PrimitiveObject * prim)
    {
        //计算lambda
        computeLambda(data, prim);

        // //计算dpos
        computeDpos(data, prim);

        //debug
        // printScalarField("lambda.csv",data->lambda);//test
        // printVectorField("dpos.csv",data->dpos);//test

        //apply the dpos to the pos
        auto & pos = prim->verts;
        for (size_t i = 0; i < data->numParticles; i++)
            pos[i] += data->dpos[i];
    }
    

    void computeLambda(PBFWorld* data, PrimitiveObject * prim)
    {
        data->lambda.clear();
        data->lambda.resize(data->numParticles);
        const auto &pos = prim->verts;//这里只访问，不修改
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
                vec3f gradJ = CubicKernel::gradW(distVec);
                gradI += gradJ;
                sumSqr += dot(gradJ, gradJ);
                densityCons += CubicKernel::W(length(distVec));
            }
            densityCons = (data->mass * densityCons / data->rho0) - 1.0;

            sumSqr += dot(gradI, gradI);
            //compute lambda
            data->lambda[i] = (-densityCons) / (sumSqr + data->lambdaEpsilon);

        }
    }

    void computeDpos(PBFWorld* data, PrimitiveObject * prim)
    {
        data->dpos.clear();
        data->dpos.resize(data->numParticles);
        const auto &pos = prim->verts; //这里只访问，不修改
        const auto &neighborList = data->neighborList;//这里只访问，不修改

        for (size_t i = 0; i < data->numParticles; i++)
        {
            vec3f dposI{0.0, 0.0, 0.0};
            for (size_t j = 0; j < neighborList[i].size(); j++)
            {
                int pj = neighborList[i][j];
                vec3f distVec = pos[i] - pos[pj];

                float sCorr = 0.0;
                dposI += (data->lambda[i] + data->lambda[pj] + sCorr) * CubicKernel::W(length(distVec));
            }
            dposI /= data->rho0;
            data->dpos[i] = dposI;

            // printf("dpos[%d] = %.5e,%.5e,%.5e \n",i,data->dpos[i][0],data->dpos[i][1], data->dpos[i][2]);
        }
    }

    //helper for computeDpos()
    inline float computeScorr(const vec3f& distVec, const float coeffDq, const float coeffK, const float h)
    {
        float x = CubicKernel::W(length(distVec)) / CubicKernel::W(coeffDq * h);
        x = x * x * x * x;
        return (-coeffK) * x;
    }

     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto data = get_input<PBFWorld>("PBFWorld");

        for(int i=0; i<data->numSubsteps; i++)
            solve(data.get(), prim.get());

        set_output("outPrim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(
    PBFWorld_Solve,
    {
        {"prim","PBFWorld"},
        {"outPrim","PBFWorld"},
        {},
        {"PBD"}
    }
);