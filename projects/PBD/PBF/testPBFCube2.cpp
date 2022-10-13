#include <vector>
#include <array>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
using namespace zeno;

/**
 * @brief this node just for test, do not use!
 * 
 */
struct testPBFCube2 : INode{
    std::vector<vec3f> pos;
    std::shared_ptr<zeno::PrimitiveObject> prim;
    void initCubeData()
    {
        int numParticles = 15*20*15;
        pos.resize(numParticles);
        vec3f initPos{1.0,1.0,1.0};
        int cubeSize = 1.0;
        float spacing = 0.05;
        
        int num_per_row = (int) (cubeSize / spacing) + 1; //21
        int num_per_floor = num_per_row * num_per_row; //21 * 21 =441
        for (size_t i = 0; i < numParticles; i++)
        {
            int floor = i / (num_per_floor);
            int row = (i % num_per_floor) / num_per_row ;
            int col = (i % num_per_floor) % num_per_row ;
            pos[i] = vec3f(col*spacing, floor*spacing, row*spacing) + initPos;
        }

        prim->verts = std::move(pos);
    }
    virtual void apply() override{
	    prim = std::make_shared<zeno::PrimitiveObject>();
        initCubeData();
        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(testPBFCube2, {
    {},
    {"outPrim"},
    {},
    {"PBD"},
});