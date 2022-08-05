#include <iostream>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>


namespace zeno {
struct PBD : zeno::INode
{
    virtual void apply() override 
    {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");
        std::vector<zeno::vec3f> prevPos = pos;
        std::vector<zeno::vec3f> vel;
        int numParticles = pos.size();
        vel.resize(numParticles);
        /* -------------------------------------------------------------------------- */
        /*                                  preSolve                                  */
        /* -------------------------------------------------------------------------- */
        // std::cout<<"presolve"<<std::endl;
        zeno::vec3f g{0, -9.8, 0};
        float dt = 1.0/30.0;
        for(int i = 0; i < pos.size(); i++)
        {
            prevPos[i] = pos[i];
            vel[i] += g * dt;
            pos[i] += vel[i] * dt;
            if(pos[i][1]<0.0)
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }
        }
        /* -------------------------------------------------------------------------- */
        /*                                    solve                                   */
        /* -------------------------------------------------------------------------- */


        /* -------------------------------------------------------------------------- */
        /*                                  postSolve                                 */
        /* -------------------------------------------------------------------------- */
        for(int i = 0; i < pos.size(); i++)
        {
            vel[i] = (pos[i] - prevPos[i]) / dt;
        }


        set_output("prim", std::move(prim));
        
    };
};

ZENDEFNODE(PBD, {
    // inputs:
    {"prim"},
    // outputs:
    {"prim"},
    // params:
    {{"vec3f","external_force","0, -9.8, 0"}},
    //category
    {"PBD"}
});

} // namespace zeno