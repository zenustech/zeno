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
        Octree* deepCopy();
    };

    struct gpuVDBGrid
    {
        float3* pos;
        std::vector<float*>data;
        unsigned long* key;

        Octree tree;
        float3 bmin, bmax, drift;
        float dx;
        int size, gridNum[3];
        int prosNum;
        void addProperty(std::string map_key);
        void constructOctree();
        void initBox(int gNum[3], float3 bmin, float3 bmax, float3 drift, float dx, std::vector<std::string> properties);
        gpuVDBGrid* deepCopy();
        void assign(gpuVDBGrid* grid);
        void clear();
    };
    struct GridData : IObject
    {
        gpuVDBGrid      data;
        gpuVDBGrid      vel[3];

        Parms parm;
        gpuVDBGrid*          data_buf, vel_buf[3];
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
    static unsigned long __device__ morton3d(float3 const &rpos);
    int __device__ computePrefix(unsigned long*  keys, const int& index1, const int& index2, const int& maxIndex);
    int __device__ computePrefix2(unsigned long key1, unsigned long key2);
    void __device__ intepolateValue(gpuVDBGrid data, float3 pos, float* values);
}