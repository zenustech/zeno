#include "PBF_BVH.h"
#include "SPHKernels.h"
#include "../Utils/myPrint.h"
using namespace zeno;

void PBF_BVH::preSolve()
{
    auto &pos = prim->verts;
    for (int i = 0; i < numParticles; i++)
        oldPos[i] = pos[i];

    //update the pos
    for (int i = 0; i < numParticles; i++)
    {
        vec3f tempVel = vel[i];
        tempVel += externForce * dt;
        pos[i] += tempVel * dt;
        boundaryHandling(pos[i]);
    }
}


void PBF_BVH::boundaryHandling(vec3f & p)
{
    vec3f bmin = bounds_min + pRadius;
    vec3f bmax = bounds_max - pRadius;

    for (size_t dim = 0; dim < 3; dim++)
    {
        float r = ((float) rand() / (RAND_MAX));
        if (p[dim] <= bmin[dim])
            p[dim] = bmin[dim] + 1e-5 * r;
        else if (p[dim]>= bmax[dim])
            p[dim] = bmax[dim] - 1e-5 * r;
    }
}

void PBF_BVH::solve()
{
    auto &pos = prim->verts;

    computeLambda();

    computeDpos();

    // echo(lambda[100]);
    // echoVec(dpos[100]);

    //apply the dpos to the pos
    for (size_t i = 0; i < numParticles; i++)
        pos[i] += dpos[i];
}

// old way
// void PBF_BVH::computeLambda()
// {
//     lambda.clear();
//     lambda.resize(numParticles);
//     auto &pos = prim->verts;

//     for (size_t i = 0; i < numParticles; i++)
//     {
//         vec3f gradI{0.0, 0.0, 0.0};
//         float sumSqr = 0.0;
//         float densityCons = 0.0;

//         for (size_t j = 0; j < neighborList[i].size(); j++)
//         {
//             int pj = neighborList[i][j];
//             vec3f distVec = pos[i] - pos[pj];
//             vec3f gradJ = kernelSpikyGradient(distVec, h);
//             gradI += gradJ;
//             sumSqr += dot(gradJ, gradJ);
//             densityCons += kernelPoly6(length(distVec), h);
//         }
//         densityCons = (mass * densityCons / rho0) - 1.0;

//         //compute lambda
//         sumSqr += dot(gradI, gradI);
//         lambda[i] = (-densityCons) / (sumSqr + lambdaEpsilon);
//     }
// }




//new way
void PBF_BVH::computeLambda()
{
    lambda.clear();
    lambda.resize(numParticles);
    auto &pos = prim->verts;
    // std::vector<float> density;
    float density;
    float density_err;

    for (size_t i = 0; i < numParticles; i++)
    {
        // 计算粒子i的当前密度和密度误差
        density = mass* CubicKernel::W(0.0);
        for (unsigned int j = 0; j < neighborList[i].size(); j++)
        {
            const unsigned int pj = neighborList[i][j];
            float dist = zeno::length(pos[i] - pos[pj]);
            density += mass * CubicKernel::W(dist);
        }
        density_err = std::max(density, rho0) - rho0;

        //判断是否密度小于rho0，如果是，那么就不修正，用于防止particle deficiency造成的artifacts
        const float C = std::max(density / rho0 - 1.0, 0.0);			// clamp to prevent particle clumping at surface
        if (C == 0.0)
            continue;
        
        // Compute gradients dC/dx_j 
		float sum_grad_C2 = 0.0;
		vec3f gradC_i(0.0, 0.0, 0.0);

		for (unsigned int j = 0; j < neighborList[i].size(); j++)
		{
			const unsigned int pj = neighborList[i][j]; //neighborIndex
            const vec3f gradC_j = -mass / rho0 * CubicKernel::gradW(pos[i] - pos[pj]);
            sum_grad_C2 += lengthSquared(gradC_j);
            gradC_i -= gradC_j;
		}

		sum_grad_C2 += lengthSquared(gradC_i);

		// Compute lambda
		lambda[i] = -C / (sum_grad_C2 + lambdaEpsilon);
        echo(i);
        echo(lambda[i]);
    }
}

// void PBF_BVH::computeDpos()
// {
//     dpos.clear();
//     dpos.resize(numParticles);
//     auto &pos = prim->verts;

//     for (size_t i = 0; i < numParticles; i++)
//     {
//         vec3f dposI{0.0, 0.0, 0.0};
//         for (size_t j = 0; j < neighborList[i].size(); j++)
//         {
//             int pj = neighborList[i][j];
//             vec3f distVec = pos[i] - pos[pj];

//             float sCorr = computeScorr(distVec, coeffDq, coeffK, h);
//             dposI += (lambda[i] + lambda[pj] + sCorr) * kernelSpikyGradient(distVec, h);
//         }
//         dposI /= rho0;
//         dpos[i] = dposI;
//     }
// }


// new way
void PBF_BVH::computeDpos()
{
    dpos.clear();
    dpos.resize(numParticles);
    auto &pos = prim->verts;

    for (size_t i = 0; i < numParticles; i++)
    {
        vec3f corr(0,0,0);
        for (unsigned int j = 0; j < neighborList[i].size(); j++)
        {
            const unsigned int neighborIndex = neighborList[i][j];
            const vec3f gradC_j = -mass / rho0 * CubicKernel::gradW(pos[i] - pos[neighborIndex]);
            corr -= (lambda[i] + lambda[neighborIndex]) * gradC_j;
        }
        dpos[i] = corr;
    }
}

//helper for computeDpos()
inline float PBF_BVH::computeScorr(const vec3f& distVec, float coeffDq, float coeffK, float h)
{
    float x = kernelPoly6(length(distVec), h) / kernelPoly6(coeffDq * h, h);
    x = x * x;
    x = x * x;
    return (-coeffK) * x;
}


void PBF_BVH::postSolve()
{
    auto &pos = prim->verts;

    for (size_t i = 0; i < numParticles; i++)
        vel[i] = (pos[i] - oldPos[i]) / dt;
}


void PBF_BVH::neighborSearch(std::shared_ptr<PrimitiveObject> prim)
{
    auto &pos = prim->verts;

    //构建BVH
    // auto lbvh = std::make_shared<zeno::LBvh>(prim,  neighborSearchRadius,zeno::LBvh::element_c<zeno::LBvh::element_e::point>);

    //清零
    neighborList.clear();
    neighborList.resize(pos.size());

    lbvh->refit();
    
    //邻域搜索
    buildNeighborList(pos, neighborSearchRadius, lbvh.get(), neighborList);
}


void PBF_BVH::buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list)
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
