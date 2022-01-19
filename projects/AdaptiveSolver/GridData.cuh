
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>
#include <iostream>
namespace zeno{
    struct Parms
    {
        vec3f bmin;
        vec3f bmax;
        float dt;
        vec3i gridNum;
        float dx;
    };

    struct GridData:IObject
    {
        float3* pos;
        float3* vel;
        float*  vol;
        float*  temperature;
        int* key;

        Parms parm;
        void initData(vec3f bmin, vec3f bmax, float dx, float dt);
        void computeKey(float bmin[3], float bmax[3]);
    };

}