#include "PBF.h"
#include "CellHelpers.h"
using namespace zeno;

void PBF::preSolve()
{
    for (int i = 0; i < numParticles; i++)
        oldPos[i] = pos[i];

    //update the pos
    for (int i = 0; i < numParticles; i++)
    {
        vec3f tempVel = vel[i];
        tempVel += g * dt;
        pos[i] += tempVel * dt;
        boundaryHandling(pos[i]);
    }

    neighborSearch();//grid-baed neighborSearch for now
}


void PBF::boundaryHandling(vec3f & p)
{
    float worldScale = 20.0; //scale from simulation space to real world space.
    // this is to prevent the kernel from being divergent.
    float bmin = pRadius/worldScale;
    vec3f bmax = bounds - pRadius/worldScale;

    for (size_t dim = 0; dim < 3; dim++)
    {
        float r = ((float) rand() / (RAND_MAX));
        if (p[dim] <= bmin)
            p[dim] = bmin + 1e-5 * r;
        else if (p[dim]>= bmax[dim])
            p[dim] = bmax[dim] - 1e-5 * r;
    }
}

void PBF::solve()
{
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
        float lambdaEpsilon = 100.0; // to prevent the singularity
        lambda[i] = (-densityCons) / (sumSqr + lambdaEpsilon);
    }
}

void PBF::computeDpos()
{
    dpos.clear();
    dpos.resize(numParticles);
    for (size_t i = 0; i < numParticles; i++)
    {
        vec3f dposI{0.0, 0.0, 0.0};
        for (size_t j = 0; j < neighborList[i].size(); j++)
        {
            int pj = neighborList[i][j];
            vec3f distVec = pos[i] - pos[pj];

            float sCorr = computeScorr(distVec);
            dposI += (lambda[i] + lambda[pj] + sCorr) * kernelSpikyGradient(distVec, h);
        }
        dposI /= rho0;
        dpos[i] = dposI;
    }
}

//helper for computeDpos()
inline float PBF::computeScorr(const vec3f& distVec)
{
    float coeffDq = 0.3;
    float coeffK = 0.001;

    float x = kernelPoly6(length(distVec), h) / kernelPoly6(coeffDq * h, h);
    x = x * x;
    x = x * x;
    return (-coeffK) * x;
}


void PBF::postSolve()
{
    for (size_t i = 0; i < numParticles; i++)
        boundaryHandling(pos[i]);
    for (size_t i = 0; i < numParticles; i++)
        vel[i] = (pos[i] - oldPos[i]) / dt;
}


inline void PBF::initData()
{     
    //prepare cell data
    // cell.resize(numCell);
    for (size_t i = 0; i < numCell; i++)
    {
        // //to calculate the x y z coord of cell
        vec3i xyz = cellID2XYZ(i);
        int hash = getCellHash(xyz[0],xyz[1],xyz[2]);

        cell[hash].x = xyz[0];
        cell[hash].y = xyz[1];
        cell[hash].z = xyz[2];
        cell[hash].parInCell.reserve(10); //pre-allocate memory to speed up
    }
    
    //prepare neighbor list 
    neighborList.resize(numParticles);
    for (size_t i = 0; i < numParticles; i++)
        neighborList[i].reserve(10);
}

inline void PBF::initCellData()
{
    // dx = 2.51; //default value for test
    // vec3i bound {40,40,40};
    dx = get_input<zeno::NumericObject>("dx")->get<float>();
    bound = get_input<zeno::NumericObject>("bound")->get<vec3i>();

    dxInv = 1.0/dx;
    int numX = int(bound[0] / dx) + 1;
    int numY = int(bound[1] / dx) + 1;
    int numZ = int(bound[2] / dx) + 1;
    numCellXYZ.resize(3);
    numCellXYZ[0] = numX;
    numCellXYZ[1] = numY;
    numCellXYZ[2] = numZ;
    numCell = numX * numY * numZ;
}