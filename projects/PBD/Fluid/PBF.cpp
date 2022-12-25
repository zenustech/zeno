#include "PBF.h"
#include "CellHelpers.h"
using namespace zeno;

void PBF::preSolve()
{
    auto &pos = prim->verts;
    for (int i = 0; i < numParticles; i++)
        oldPos[i] = pos[i];

    //update the pos
    for (int i = 0; i < numParticles; i++)
    {
        vec3f tempVel = vel[i];
        tempVel += gravity * dt;
        pos[i] += tempVel * dt;
        boundaryHandling(pos[i]);
    }
}


void PBF::boundaryHandling(vec3f & p)
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

void PBF::solve()
{
    auto &pos = prim->verts;

    computeLambda();

    computeDpos();

    //apply the dpos to the pos
    for (size_t i = 0; i < numParticles; i++)
        pos[i] += dpos[i];
}

void PBF::computeLambda()
{
    lambda.clear();
    lambda.resize(numParticles);
    auto &pos = prim->verts;

    for (size_t i = 0; i < numParticles; i++)
    {
        vec3f gradI{0.0, 0.0, 0.0};
        float sumSqr = 0.0;
        float densityCons = 0.0;

        for (size_t j = 0; j < neighborList[i].size(); j++)
        {
            int pj = neighborList[i][j];
            vec3f distVec = pos[i] - pos[pj];
            vec3f gradJ = kernelSpikyGradient(distVec, h);
            gradI += gradJ;
            sumSqr += dot(gradJ, gradJ);
            densityCons += kernelPoly6(length(distVec), h);
        }
        densityCons = (mass * densityCons / rho0) - 1.0;

        //compute lambda
        sumSqr += dot(gradI, gradI);
        lambda[i] = (-densityCons) / (sumSqr + lambdaEpsilon);
    }
}

void PBF::computeDpos()
{
    dpos.clear();
    dpos.resize(numParticles);
    auto &pos = prim->verts;

    for (size_t i = 0; i < numParticles; i++)
    {
        vec3f dposI{0.0, 0.0, 0.0};
        for (size_t j = 0; j < neighborList[i].size(); j++)
        {
            int pj = neighborList[i][j];
            vec3f distVec = pos[i] - pos[pj];

            float sCorr = computeScorr(distVec, coeffDq, coeffK, h);
            dposI += (lambda[i] + lambda[pj] + sCorr) * kernelSpikyGradient(distVec, h);
        }
        dposI /= rho0;
        dpos[i] = dposI;
    }
}

//helper for computeDpos()
inline float PBF::computeScorr(const vec3f& distVec, float coeffDq, float coeffK, float h)
{
    float x = kernelPoly6(length(distVec), h) / kernelPoly6(coeffDq * h, h);
    x = x * x;
    x = x * x;
    return (-coeffK) * x;
}


void PBF::postSolve()
{
    auto &pos = prim->verts;

    for (size_t i = 0; i < numParticles; i++)
        vel[i] = (pos[i] - oldPos[i]) / dt;
}


