#include "../ZenoFX/LinearBvh.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include "./PBFWorld.h"
#include "./SPHKernels.h"
#include "../Utils/myPrint.h"//debug
#include <cstdio>//debug
using namespace zeno;


struct PBFWorld_Step : zeno::INode {
    void preSolve(PBFWorld* data, PrimitiveObject * prim)
    {
        auto &pos = prim->verts;
        for (int i = 0; i < pos.size(); i++)
            data->prevPos[i] = pos[i];

        //update the pos
        for (int i = 0; i < pos.size(); i++)
        {
            data->vel[i] += data->externForce * data->dt;
            pos[i] += data->vel[i] * data->dt;
            boundaryHandling(pos[i], data->bounds_min, data->bounds_max);
        }
    }

    void neighborhoodSearch(PBFWorld* data, std::shared_ptr<PrimitiveObject> prim)
    {
        auto &pos = prim->verts;

        //构建BVH
        auto lbvh = std::make_shared<zeno::LBvh>(prim,  data->neighborSearchRadius,zeno::LBvh::element_c<zeno::LBvh::element_e::point>);

        //清零
        data->neighborList.clear();
        data->neighborList.resize(pos.size());

        //邻域搜索
        buildNeighborList(pos, data->neighborSearchRadius, lbvh.get(), data->neighborList);
    }


    void buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list)
    {
        auto radius2 = searchRadius*searchRadius;
        #pragma omp parallel for
        for (int i = 0; i < pos.size(); i++) 
        {
            //BVH的使用
            lbvh->iter_neighbors(pos[i], [&](int j) 
                {
                    if (lengthSquared(pos[i] - pos[j]) < radius2 && j!=i)
                    {
                        list[i].emplace_back(j);
                    }
                }
            );
        }
    }

    void boundaryHandling(vec3f & p, const vec3f &bounds_min, const vec3f &bounds_max)
    {
        for (size_t dim = 0; dim < 3; dim++)
        {
            float r = ((float) rand() / (RAND_MAX));//0-1随机数
            if (p[dim] <= bounds_min[dim])
                p[dim] = bounds_min[dim] + 1e-5 * r;
            else if (p[dim]>= bounds_max[dim])
                p[dim] = bounds_max[dim] - 1e-5 * r;
        }
    }

    //核心步骤
    void solve(PBFWorld* data, PrimitiveObject * prim)
    {
        //计算lambda
        computeLambda(data, prim);
        printf("lambda[0] = %.5e \n",data->lambda[0]);

        // //计算dpos
        computeDpos(data, prim);
        printf("dpos[0] = %.5e,%.5e,%.5e \n",data->dpos[0][0],data->dpos[0][1], data->dpos[0][2]);

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
                vec3f gradJ = SpikyKernel::gradW(distVec);
                gradI += gradJ;
                sumSqr += dot(gradJ, gradJ);
                densityCons += Poly6Kernel::W(length(distVec));
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
                // float sCorr = computeScorr(distVec,data);
                dposI += (data->lambda[i] + data->lambda[pj] + sCorr) * Poly6Kernel::W(length(distVec));
            }
            dposI /= data->rho0;
            data->dpos[i] = dposI;


        }
    }

    //helper for computeDpos()
    inline float computeScorr(const vec3f& distVec, const PBFWorld* data)
    {
        float x = Poly6Kernel::W(length(distVec)) / Poly6Kernel::W(data->coeffDq * data->h);
        x = x * x * x * x;
        return (-data->coeffK) * x;
    }

    void postSolve(PBFWorld *data, PrimitiveObject* prim)
    {
        auto &pos = prim->verts;
        for (size_t i = 0; i < data->numParticles; i++)
            data->vel[i] = (pos[i] - data->prevPos[i]) / data->dt;
    }

     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto data = get_input<PBFWorld>("PBFWorld");

        auto & pos = prim->verts;
        preSolve(data.get(),prim.get());
        printf("pos[0] = %.5e, %.5e, %.5e \n",pos[0][0],pos[0][1], pos[0][2]);

        neighborhoodSearch(data.get(),prim);
        echoVec(data->neighborList[0]);

        for(int i=0; i<data->numSubsteps; i++)
            solve(data.get(), prim.get());
        postSolve(data.get(),prim.get());

        set_output("outPrim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(
    PBFWorld_Step,
    {
        {gParamType_Primitive,"PBFWorld"},
        {"outPrim","PBFWorld"},
        {},
        {"PBD"}
    }
);