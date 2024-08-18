#include <vector>
#include <array>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
using namespace zeno;

/**
 * @brief this node just for test, do not use!
 * 
 */
struct PBFWorld_testCube : INode{

    virtual void apply() override{
	    auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto cubeSize = get_input<NumericObject>("cubeSize")->get<float>();
        auto spacing = get_input<NumericObject>("spacing")->get<float>();
        auto initPos = get_input<NumericObject>("initPos")->get<vec3f>();
        auto numParticles = get_input<NumericObject>("numParticles")->get<int>();
        auto &pos = prim->verts;
        pos.resize(numParticles);

        int num_per_row = (int) (cubeSize / spacing) + 1; //21
        int num_per_floor = num_per_row * num_per_row; //21 * 21 =441
        for (size_t i = 0; i < numParticles; i++)
        {
            int floor = i / (num_per_floor);
            int row = (i % num_per_floor) / num_per_row ;
            int col = (i % num_per_floor) % num_per_row ;
            pos[i] = vec3f(col*spacing, floor*spacing, row*spacing) + initPos;
        }

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBFWorld_testCube, {
    {
        {gParamType_Float,"cubeSize","0.5"},
        {gParamType_Int,"numParticles","4500"},
        {gParamType_Vec3f,"initPos","0.0, 0.0, 0.0"},
        {gParamType_Float,"spacing","0.025"},
    },
    {"outPrim"},
    {},
    {"PBD"},
});