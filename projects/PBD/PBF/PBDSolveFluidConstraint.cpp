#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
namespace zeno {
struct PBDSolveFluidConstraint : zeno::INode {
private:

void solve(PrimitiveObject *prim)
{
    computeLambda();

    computeDpos();

    //apply the dpos to the pos
    for (size_t i = 0; i < numParticles; i++)
        pos[i] += dpos[i];
}

void computeLambda()
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

void computeDpos()
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
inline float computeScorr(const vec3f& distVec, float coeffDq = 0.3, float coeffK = 0.001)
{
    float x = kernelPoly6(length(distVec), h) / kernelPoly6(coeffDq * h, h);
    x = x * x;
    x = x * x;
    return (-coeffK) * x;
}


public:
    virtual void apply() override {
        //get data
        auto prim = get_input<PrimitiveObject>("prim");

        solve(prim.get());

        //output
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveFluidConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "rest_density", "1000.0"},
                    {"float", "dt", "0.0016667"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno