#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>

#include "../myPrint.h"

namespace zeno {
struct PBF : zeno::INode {
//physical params
public:    
    int numSubsteps;
    float dt;
    vec3f bounds{40.0, 40.0, 40.0};
    vec3f g{0.0, -9.8, -.0};
    float pRadius{3.0};
    vec3i numCell{16,16,16};

private:
    void preSolve();
    void solve();
    void postSolve();

//Data preparing
    void initData();
    std::shared_ptr<zeno::PrimitiveObject> prim;
    int numParticles;
    std::vector<vec3f> pos;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;

    std::array<std::vector<int>, 3> numParInCell;
    void PBF::beBounded(vec3f & p);

public:
    virtual void apply() override {
        //get input data
        prim = get_input<PrimitiveObject>("prim");
        numSubsteps = get_input<zeno::NumericObject>("numSubsteps")->get<int>();

        initData();        
        preSolve();
        for (size_t i = 0; i < numSubsteps; i++)
            solve();
        postSolve();            

        set_output("outPrim", std::move(prim));
    }
};


ZENDEFNODE(PBF, {   
                    {
                        {"PrimitiveObject", "prim"},
                        {"int", "numSubsteps", "10"}
                    },
                    {   {"PrimitiveObject", "outPrim"} },
                    {},
                    {"PBD"},
                });

void PBF::initData()
{
    static bool firstTime = true;
    if(firstTime)
    {
        firstTime = false;
         
        //copy the particle postions to local
        pos = prim->verts;
        numParticles = pos.size();
        oldPos.resize(numParticles);
        vel.resize(numParticles);
        log_info("numParticles: {}", numParticles);

        (numParInCell[0]).resize(numCell[0]);
        (numParInCell[0]).resize(numCell[1]);
        (numParInCell[0]).resize(numCell[2]);

    }
}

void PBF::preSolve()
{
    for (int i = 0; i < numParticles; i++)
    {
        oldPos[i] = pos[i];
    }

    for (int i = 0; i < numParticles; i++)
    {
        vel[i] += g * dt;
        pos[i] += vel[i] * dt;
        beBounded(pos[i]);
    }

    TODO:
    // numParInCell.clear();

}

void PBF::beBounded(vec3f & p)
{
    float bmin = pRadius/20.0;
    vec3f bmax = bounds - pRadius/20.0;

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
}

void PBF::postSolve()
{
}

} // namespace zeno