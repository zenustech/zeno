#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>

namespace zeno {
struct PBF : zeno::INode {
//physical params
public:    
    int numSubsteps;
    float dt;

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
    }
}

void PBF::preSolve()
{
    for (int i = 0; i < numParticles; i++)
    {
        oldPos[i] = pos[i];
    }
}

void PBF::solve()
{
}

void PBF::postSolve()
{
}

} // namespace zeno