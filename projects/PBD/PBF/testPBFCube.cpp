#include <vector>
#include <array>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
using namespace zeno;

/**
 * @brief this node just for test, do not use!
 * 
 */
struct testPBFCube : INode{
    std::shared_ptr<zeno::PrimitiveObject> prim;
    void initCubeData()
    {
        // int numParticles = 10000;
        // vec3f initPos{10.0,10.0,10.0};
        // int cubeSize = 20;
        // float spacing = 1;
        auto numParticles = get_input<zeno::NumericObject>("numParticles")->get<int>();
        auto initPos = get_input<zeno::NumericObject>("initPos")->get<vec3f>();
        auto cubeSize = get_input<zeno::NumericObject>("cubeSize")->get<float>();
        auto spacing = get_input<zeno::NumericObject>("spacing")->get<float>();

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
    }
    virtual void apply() override{
	    prim = std::make_shared<zeno::PrimitiveObject>();
        initCubeData();
        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(testPBFCube, {
    {
        {gParamType_Int, "numParticles", "10000"},
        {gParamType_Vec3f, "initPos", "10,10,10"},
        {gParamType_Float, "cubeSize", "20"},
        {gParamType_Float, "spacing", "1"},
    },
    {"outPrim"},
    {},
    {"PBD"},
});