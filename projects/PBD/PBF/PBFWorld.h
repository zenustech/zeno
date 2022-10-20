#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
namespace zeno
{

/**
 * @brief 这是个数据类，用来在传递PBF的所有数据。如果要使用，请include此头文件。
 * 
 */
struct PBFWorld:IObject
{
//physical params
    float dt=  0.0025;
    float radius = 0.025;
    vec3f bounds;
    vec3f externForce{0, -10.0, 0};

    float rho0 = 1000.0;
    float mass; //0.8*diam*diam*diam*rho0
    float h; // 4*radius
    float neighborSearchRadius; //h * 1.05
    float lambdaEpsilon = 1e-6;
    float coeffDq = 0.3;
    float coeffK = 0.1;

    //data for physical fields
    int numParticles;
    // std::vector<vec3f> *pos;//transfered with prim
    std::vector<vec3f> prevPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;
    
    //neighborList
    std::vector<std::vector<int>> neighborList;
};

    
} // namespace zeno
