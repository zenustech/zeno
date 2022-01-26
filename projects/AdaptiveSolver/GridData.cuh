#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
namespace zeno{
    struct Parms
    {
        float3 bmin;
        float3 bmax;
        float dt;
        vec3i gridNum;
        float dx;

        dim3 threadsPerBlock, blockPerGrid, velBlockPerGrid;
    };
    struct PointData
    {
        float3  pos;
        float   vol;
        float   temperature;
        //int     level;
        //unsigned long   key;
    };
    struct TreeNode
    {
        int lChild, rChild;
        int lType, rType;//0:leaf  1:internal
        int delta_node;
        int parentNode;
    };
    struct Octree
    {
        TreeNode* nodes;
        int rootIndex;
    };

    struct gpuVDBGrid
    {
        float3* pos;
        std::map<std::string, float*> data;
        unsigned long* key;

        Octree tree;
        float3 drift;
        float dx;
        int size;
        void addProperty(std::string map_key);
        void initBox(int gNum[3], float3 bmin, float3 bmax, float3 drift, float dx, std::vector<std::string> properties);
    };
    struct GridData : IObject
    {
        gpuVDBGrid      data;
        //gpuVDBGrid      vel[3];

        PointData*   pData;
        float*      vel[3];
        unsigned long*  pKey, *velKey[3];
        Parms parm;
        Octree pTree, velTree[3];

        char* buffer;
        //iterator
        float*  b;
        float*  r;
        float*  press;

        __device__ Parms gpuParm;
        void initData(vec3f bmin, vec3f bmax, float dx, float dt);
        void step();
        // octree functions
        void constructOctree();
        
        // // adaptive grid functions
        void advection();
        void PossionSolver();
        void applyOtherForce();
        // virtual void subdivision();
        // virtual void coarsen();
    };
    void __global__ initPos(GridData data);
    void __global__ generateMortonCode(GridData data);
    void __global__ sortViaMortonCode(GridData data);
    void __global__ genOctree(GridData data);

    void __global__ findOctreeRoot(GridData data);

}