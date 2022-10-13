#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
namespace zeno
{

struct PBFWorld:IObject
{
//physical params
    float dt= 0.005;
    float pRadius = 0.025;
    vec3f bounds{15.0, 20.0, 15.0};
    vec3f g{0, -10.0, 0};

    float rho0 ;
    float mass;
    float h;
    float neighborSearchRadius = h * 1.05;
    float lambdaEpsilon;
    float coeffDq;
    float coeffK;

    //data for physical fields
    int numParticles;
    std::vector<vec3f> pos;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;
    
//data for cells
    std::vector<int> numCellXYZ;
    int numCell;
    float dx; //cell size
    float dxInv; 
    vec3i bound;
    void initCellData();
    struct Cell
    {
        int x,y,z;
        std::vector<int> parInCell; 
    };
    std::map<int, Cell>  cell;

    //neighborList
    std::vector<std::vector<int>> neighborList;
};

    
} // namespace zeno
