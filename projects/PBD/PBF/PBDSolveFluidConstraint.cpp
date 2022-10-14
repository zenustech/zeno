#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include "SPHKernelFuncs.h"
namespace zeno {
struct PBDSolveFluidConstraint : zeno::INode {
private:
    std::vector<vec3f> dpos;
    std::vector<vec3f> pos;
    std::vector<std::vector<int>> neighborList;

    float lambdaEpsilon;
    float coeffDq;
    float coeffK;
    float h;
    float rho0;
    float mass;

void solve(PrimitiveObject *prim)
{
    // float lambdaEpsilon = 100.0;
    // float coeffDq = 0.3;
    // float coeffK = 0.001;
    // float h = 0.001;
    // float rho0 = 1.0;
    // float mass = 1.0;

    std::vector<float> lambda;
    computeLambda(prim, neighborList, h, rho0, mass, lambdaEpsilon, lambda);

    computeDpos(prim, neighborList, lambda, h, rho0, coeffDq, coeffK, dpos);

    //apply the dpos to the pos
    for (size_t i = 0; i < prim->verts.size(); i++)
        pos[i] += dpos[i];
}

void computeLambda(
    const PrimitiveObject *prim,
    const std::vector<std::vector<int>>  &neighborList,
    const float h,
    const float rho0,
    const float mass,
    const float lambdaEpsilon , // to prevent the singularity
    std::vector<float> &lambda //输出
    )
{
    lambda.clear();
    lambda.resize(prim->verts.size());
    for (size_t i = 0; i < prim->verts.size(); i++)
    {
        vec3f gradI{0.0, 0.0, 0.0};
        float sumSqr = 0.0;
        float densityConstraint = 0.0;

        for (size_t j = 0; j < neighborList[i].size(); j++)
        {
            int pj = neighborList[i][j];
            vec3f distVec = pos[i] - pos[pj];
            vec3f gradJ = kernelSpikyGradient(distVec, h);
            gradI += gradJ;
            sumSqr += dot(gradJ, gradJ);
            densityConstraint += kernelPoly6(length(distVec), h);
        }
        densityConstraint = (mass * densityConstraint / rho0) - 1.0;

        //compute lambda
        sumSqr += dot(gradI, gradI);
        lambda[i] = (-densityConstraint) / (sumSqr + lambdaEpsilon);
    }
}

void computeDpos(
    const PrimitiveObject *prim,
    const std::vector<std::vector<int>>  &neighborList,
    const std::vector<float> &lambda,
    const float h,
    const float rho0,
    const float coeffDq,
    const float coeffK,
    std::vector<vec3f> &dpos //输出
    )
{
    dpos.clear();
    dpos.resize(prim->verts.size());
    for (size_t i = 0; i < prim->verts.size(); i++)
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
inline float computeScorr(const vec3f& distVec, float coeffDq, float coeffK, float h)
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

        //get fields
        if(!prim->verts.has_attr("dpos"))
            dpos = prim->verts.add_attr<vec3f>("dpos");
        pos = prim->verts.attr<vec3f>("pos");

        // if(!prim->verts.has_attr("neighborList"))
            // neighborList = prim->verts.add_attr<std::vector<int>>("neighborList");

        //get parameters
        float dt = get_input<NumericObject>("dt")->get<float>();
        rho0 = get_input<NumericObject>("rho0")->get<float>();
        h = get_input<NumericObject>("h")->get<float>();
        mass = get_input<NumericObject>("mass")->get<float>();
        coeffDq = get_input<NumericObject>("coeffDq")->get<float>();
        coeffK = get_input<NumericObject>("coeffK")->get<float>();
        lambdaEpsilon = get_input<NumericObject>("lambdaEpsilon")->get<float>();

        solve(prim.get());

        //output
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveFluidConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "dt", "0.005"},
                    {"float", "rho0", "1.0"},
                    {"float", "h", "0.001"},
                    {"float", "mass", "1.0"},
                    {"float", "coeffDq", "0.3"},
                    {"float", "coeffK", "0.001"},
                    {"float", "lambdaEpsilon", "100.0"}
                    // {"float", "dt", "0.005"},
                    // {"float", "rho0", "1000.0"},
                    // {"float", "h", "0.1"},
                    // {"float", "mass", "0.1"},
                    // {"float", "coeffDq", "0.3"},
                    // {"float", "coeffK", "0.1"},
                    // {"float", "lambdaEpsilon", "1e-6"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno