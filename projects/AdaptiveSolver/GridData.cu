#include "GridData.cuh"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

namespace zeno
{
    static unsigned long __device__ morton3d(float3 const &rpos) 
    {
        auto tmp = (rpos * 1048576.0f);
        unsigned long v[3];
        v[0] = floorf(tmp.x);v[1] = floorf(tmp.y);v[2] = floorf(tmp.z);
        for(int i=0;i<3;++i)
        {
            v[i] = v[i] <0?0ul:(v[i] > 1048575ul? 1048575ul:v[i]);
            v[i] = (v[i] * 0x0000000100000001ul) & 0xFFFF00000000FFFFul;
            v[i] = (v[i] * 0x0000000000010001ul) & 0x00FF0000FF0000FFul;
            v[i] = (v[i] * 0x0000000000000101ul) & 0xF00F00F00F00F00Ful;
            v[i] = (v[i] * 0x0000000000000011ul) & 0x30C30C30C30C30C3ul;
            v[i] = (v[i] * 0x0000000000000005ul) & 0x4924924949249249ul;
        }

        return v[0] * 4 + v[1] * 2 + v[2];
    }   

    int __device__ computePrefix(unsigned long*  keys, const int& index1, const int& index2, const int& maxIndex)
    {
        if(index1 >= maxIndex || index2 >= maxIndex || index1 < 0 || index2 < 0)
            return -1;
        auto xx= keys[index1] ^ keys[index2];
        return __clzll(xx);
    };
    int __device__ computePrefix2(unsigned long key1, unsigned long key2)
    {
        auto xx= key1 ^ key2;
        return __clzll(xx);
    };
    PointData& __device__ findPoint(GridData data, float3 const& rpos)
    {
        auto key = morton3d(rpos);
        int node_type = 1, index = data.pTree.rootIndex;
        do
        {
            if(computePrefix2(key, data.pKey[index]) > data.pTree.nodes[index].delta_node)
            {
                node_type = data.pTree.nodes[index].rType;
                index = data.pTree.nodes[index].rChild;
            }
            else
            {
                node_type = data.pTree.nodes[index].lType;
                index = data.pTree.nodes[index].lChild;
            }
        }
        while (node_type == 1);
        return data.pData[index];
    };
    float __device__ findVel(GridData data, const int& dim, float3 const& rpos)
    {
        auto key = morton3d(rpos);
        int node_type = 1, index = data.velTree[dim].rootIndex;
        do
        {
            if(computePrefix2(key, data.velKey[dim][index]) > data.velTree[dim].nodes[index].delta_node)
            {
                node_type = data.velTree[dim].nodes[index].rType;
                index = data.velTree[dim].nodes[index].rChild;
            }
            else
            {
                node_type = data.velTree[dim].nodes[index].lType;
                index = data.velTree[dim].nodes[index].lChild;
            }
        }
        while (node_type == 1);
        return data.vel[dim][index];
    }
    float3 __device__ intepolateVel(GridData data, float3 index, int* gridNum)
    {
        float vel[3];
        for(int dim = 0;dim < 3;++dim)
        {
            float indx[3] = {(index.x), (index.y), (index.z)};
            indx[dim]+=0.5;
            float baseIndex[3] = {::floor(indx[0]), ::floor(indx[1]), ::floor(indx[2])};
            int gNum[3] = {gridNum[0], gridNum[1], gridNum[2]};
            gNum[dim]++;
            if(baseIndex[0] < 0 || baseIndex[1] < 0 || baseIndex[2] < 0 ||
                baseIndex[0]>=gNum[0] -1 || baseIndex[1]>=gNum[1] -1 || baseIndex[2]>=gNum[2] -1)
            {
                vel[dim] = 0;
                continue;
            }
            float w[3] = {indx[0] - baseIndex[0], indx[1] - baseIndex[1], indx[2] - baseIndex[2]};
            
            float accum = 0;
            for (size_t i = 0; i < 2; i++)
                for (size_t j = 0; j < 2; j++)
                    for (size_t k = 0; k < 2; k++)
                    {
                        float3 rpos = make_float3((baseIndex[0] + i)/gNum[0], (baseIndex[1] + j)/gNum[1], (baseIndex[2] + k)/gNum[2]);
                        accum += (i*w[0]+(1-i)*(1-w[0]))*
                            (j*w[1]+(1-j)*(1-w[1]))*
                            (k*w[2]+(1-k)*(1-w[2]))*findVel(data, dim, rpos);
                    }
            vel[dim] = accum;
        }
        return make_float3(vel[0], vel[1], vel[2]);
    }

    void __device__ intepolateData(GridData data, float3 index, int* gNum, PointData& pData)
    {
        float baseIndex[3] = {::floor(index.x), ::floor(index.y), ::floor(index.z)};
        float w[3] = {index.x - baseIndex[0], index.y - baseIndex[1],index.z - baseIndex[2]};
        float accum = 0;
        pData.vol = pData.temperature = 0;
        if(baseIndex[0] < 0 || baseIndex[1] < 0 || baseIndex[2] < 0 || 
            baseIndex[0] >= gNum[0] - 1 || baseIndex[1] >= gNum[1] - 1 || baseIndex[2] >= gNum[2] - 1)
            return;
        for (size_t i = 0; i < 2; i++)
            for (size_t j = 0; j < 2; j++)
                for (size_t k = 0; k < 2; k++)
                {
                    float3 rpos = make_float3((baseIndex[0] + i)/gNum[0], (baseIndex[1] + j)/gNum[1], (baseIndex[2] + k)/gNum[2]);
                    float weight = (i*w[0]+(1-i)*(1-w[0]))*
                        (j*w[1]+(1-j)*(1-w[1]))*
                        (k*w[2]+(1-k)*(1-w[2]));
                    auto point = findPoint(data, rpos);
                    pData.temperature += weight * point.temperature;
                    pData.vol += weight * point.vol;
                }
    }
    
    void __global__ initPos(GridData data)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        if(index.x >= data.gpuParm.gridNum[0] || index.y >= data.gpuParm.gridNum[1] || index.z >= data.gpuParm.gridNum[2])
            return;

        int i = index.x + index.y * data.gpuParm.gridNum[0] + index.z * data.gpuParm.gridNum[0] * data.gpuParm.gridNum[1];
        data.pData[i].pos = data.gpuParm.bmin +  data.gpuParm.dx * make_float3(index.x, index.y, index.z);
    }
    void __global__ generateMortonCode(GridData data)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        if(index.x >= data.gpuParm.gridNum[0] || index.y >= data.gpuParm.gridNum[1] || index.z >= data.gpuParm.gridNum[2])
            return;
        int i = index.x + index.y * data.gpuParm.gridNum[0] + index.z * data.gpuParm.gridNum[0] * data.gpuParm.gridNum[1];
        float3 rPos = (data.pData[i].pos - data.parm.bmin) / (data.parm.bmax - data.parm.bmin);
        data.pKey[i] = morton3d(rPos);
        
        if(i < data.gpuParm.gridNum[0] * data.gpuParm.gridNum[1] * data.gpuParm.gridNum[2] - 1)
            data.pTree.nodes[i].parentNode = -1;
    }
    void __global__ generateVelMortonCode(GridData data)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        for(int dim=0;dim<3;++dim)
        {
            if(*((float*)(&index)+dim) >= data.gpuParm.gridNum[dim] + 1 || 
                *((float*)(&index)+(dim + 1)%3) >= data.gpuParm.gridNum[(dim+1)%3] || 
                *((float*)(&index)+(dim + 2)%3) >= data.gpuParm.gridNum[(dim+2)%3])
                continue;
            float gNum[3] = {data.gpuParm.gridNum[0], data.gpuParm.gridNum[1],data.gpuParm.gridNum[2]};
            gNum[dim]++;
            float3 rPos = make_float3(index) / make_float3(gNum[0],gNum[1], gNum[2]);
            int i = index.x + index.y * gNum[0] + index.z * gNum[0] * gNum[1];
            data.velKey[dim][i] = morton3d(rPos);
        }
    }
    void __global__ genOctree(unsigned long* key, Octree& tree, int* gridNum)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = index.x + index.y * gridNum[0] + index.z * gridNum[0] * gridNum[1];
        int maxIndex = gridNum[0] * gridNum[1] * gridNum[2];
        if(index.x >= gridNum[0] || index.y >= gridNum[1] || index.z >= gridNum[2])
            return;
        if(i >= maxIndex - 1)
            return;
        //determine direction 
        int d = computePrefix(key, i,i+1, maxIndex) - computePrefix(key, i, i-1, maxIndex);
        d = d / ::abs(d);
        
        //compute upper bound for the length of the range
        int delta_min = computePrefix(key, i, i - d, maxIndex);
        int l_max = 1;
        do
        {
            l_max *= 2;
        } 
        while (computePrefix(key, i, i + l_max * d, maxIndex) > delta_min);
        
        //find the other end using binary search
        int l = 0;
        for(int t = l_max / 2;t>=1;t/=2)
        {
            if(computePrefix(key, i, i + (t + l) * d, maxIndex) > delta_min)
                l = l + d;
        }
        int j = i + l*d;

        // Find the split position using binary search
        int delta_node = computePrefix(key, i, j, maxIndex);
        int s = 0;
        for(int t = l / 2;t >= 1;t /= 2)
        {
            if(computePrefix(key, i, i + (s+t)*d, maxIndex) > delta_node)
                s = s+t;
        }
        int gamma = i + s * d + ::min(d,0);

        //Output child pointers
        tree.nodes[i].delta_node - delta_node;
        tree.nodes[i].lChild = gamma;
        tree.nodes[i].lType = (::min(i,j) == gamma)?0:1;
        
        tree.nodes[i].rChild = gamma + 1;
        tree.nodes[i].lType = (::max(i, j+1) == gamma + 1)?0:1;

        tree.nodes[gamma].parentNode = i;
        tree.nodes[gamma + 1].parentNode = i;
    }
    void __global__ genVelOctree(GridData data, Parms gpuParm)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        
        for(int dim = 0;dim<3;++dim)
        {
            int gNum[3] = {gpuParm.gridNum[0], gpuParm.gridNum[1], gpuParm.gridNum[2]};
            gNum[dim]++;
            if(index.x >= gNum[0] || index.y >= gNum[1] || index.z >= gNum[2])
                continue;
            int i = index.x + index.y * gNum[0] + index.z * gNum[0] * gNum[1];
            int maxIndex = gNum[0] * gNum[1] * gNum[2];
            if(i >= maxIndex - 1)
                continue;
            //determine direction 
            int d = computePrefix(data.velKey[dim], i,i+1, maxIndex) - computePrefix(data.velKey[dim], i, i-1, maxIndex);
            d = d / ::abs(d);
            //compute upper bound for the length of the range
            int delta_min = computePrefix(data.velKey[dim], i, i - d, maxIndex);
            int l_max = 1;
            do
            {
                l_max *= 2;
            } 
            while (computePrefix(data.velKey[dim], i, i + l_max * d, maxIndex) > delta_min);
            
            //find the other end using binary search
            int l = 0;
            for(int t = l_max / 2;t>=1;t/=2)
            {
                if(computePrefix(data.velKey[dim], i, i + (t + l) * d, maxIndex) > delta_min)
                    l = l + d;
            }
            int j = i + l*d;

            // Find the split position using binary search
            int delta_node = computePrefix(data.velKey[dim], i, j, maxIndex);
            int s = 0;
            for(int t = l / 2;t >= 1;t /= 2)
            {
                if(computePrefix(data.velKey[dim], i, i + (s+t)*d, maxIndex) > delta_node)
                    s = s+t;
            }
            int gamma = i + s * d + ::min(d,0);

            //Output child pointers
            data.velTree[dim].nodes[i].delta_node - delta_node;
            data.velTree[dim].nodes[i].lChild = gamma;
            data.velTree[dim].nodes[i].lType = (::min(i,j) == gamma)?0:1;
            
            data.velTree[dim].nodes[i].rChild = gamma + 1;
            data.velTree[dim].nodes[i].lType = (::max(i, j+1) == gamma + 1)?0:1;

            data.velTree[dim].nodes[gamma].parentNode = i;
            data.velTree[dim].nodes[gamma + 1].parentNode = i;
        }

    }
    
    void __global__ findOctreeRoot(unsigned long* key, Octree& tree, int3 gNum)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        int gridNum[3] = {gNum.x, gNum.y, gNum.z};
        int i = index.x + index.y * gridNum[0] + index.z * gridNum[0] * gridNum[1];
        int maxIndex = gridNum[0] * gridNum[1] * gridNum[2];
        if(index.x >= gridNum[0] || index.y >= gridNum[1] || index.z >= gridNum[2])
            return;
        if(i >= maxIndex - 1)
            return;

        if(tree.nodes[i].parentNode == -1)
        {
            tree.rootIndex = i;
            printf("findOctreeRoot:octree root index is %d\n", i);
        }
    }
    
    void __global__ semiLaAdvection(GridData data)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        int gridNum[3] = {data.gpuParm.gridNum[0], data.gpuParm.gridNum[1], data.gpuParm.gridNum[2]};
        int i = index.x + index.y * gridNum[0] + index.z * gridNum[0] * gridNum[1];
        int maxIndex = gridNum[0] * gridNum[1] * gridNum[2];
        if(index.x >= gridNum[0] || index.y >= gridNum[1] || index.z >= gridNum[2])
            return;
        PointData buffer;
        auto vel = intepolateVel(data, make_float3(index), gridNum);
        auto midpos = data.pData[i].pos - 0.5 * data.gpuParm.dt * vel;

        auto newIndex = (midpos - data.gpuParm.bmin)/data.gpuParm.dx;
        auto midvel = intepolateVel(data, newIndex, gridNum);

        auto newpos = midpos - 0.5 * data.gpuParm.dt * midvel;
        newIndex = (newpos - data.gpuParm.bmin)/data.gpuParm.dx;
        intepolateData(data, newIndex, gridNum, buffer);
        int bufIndex = sizeof(PointData) * i;
        PointData * pointer = (PointData*)(data.buffer+bufIndex);
        pointer->pos = data.pData[i].pos;
        pointer->temperature = data.pData[i].temperature;
        pointer->vol = data.pData[i].vol;
    }

    void __global__ semiLaVelAdvection(GridData data)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        int baseDrift = sizeof(PointData) * data.gpuParm.gridNum[0] * data.gpuParm.gridNum[1] * data.gpuParm.gridNum[2];
        for(int dim = 0;dim<3;++dim)
        {
            
            int gridNum[3] = {data.gpuParm.gridNum[0], data.gpuParm.gridNum[1], data.gpuParm.gridNum[2]};
            gridNum[dim]++;
            int i = index.x + index.y * gridNum[0] + index.z * gridNum[0] * gridNum[1];
            int maxIndex = gridNum[0] * gridNum[1] * gridNum[2];
            if(index.x >= gridNum[0] || index.y >= gridNum[1] || index.z >= gridNum[2])
                continue;
            float indx[3] = {index.x, index.y, index.z};
            indx[dim] -= 0.5;
            auto vel = intepolateVel(data, make_float3(indx[0], indx[1], indx[2]), gridNum);
            
            auto newIndex = make_float3(index) - 0.5 * vel * data.gpuParm.dt / data.gpuParm.dx;
            indx[0] = newIndex.x;indx[1] = newIndex.y;indx[2] = newIndex.z;
            indx[dim] -= 0.5;
            auto midvel = intepolateVel(data, make_float3(indx[0], indx[1], indx[2]), gridNum);

            newIndex = newIndex - 0.5 * data.gpuParm.dt * midvel / data.gpuParm.dx;
            indx[0] = newIndex.x;indx[1] = newIndex.y;indx[2] = newIndex.z;
            indx[dim] -= 0.5;

            vel = intepolateVel(data, make_float3(indx[0], indx[1], indx[2]), gridNum);
            indx[0] = vel.x;indx[1] = vel.y;indx[2] = vel.z;
            int bufDrift = baseDrift + sizeof(float) * i;
            *(float*)(data.buffer + bufDrift) = indx[dim];
            baseDrift += sizeof(float)*  gridNum[0] * gridNum[1] * gridNum[2];
        }
        
        
    }

    void __global__ computeRHS(GridData data)
    {
        int3 index;
        float rho = 1;

        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;

        int i = index.x + index.y * data.gpuParm.gridNum[0] + index.z * data.gpuParm.gridNum[0] * data.gpuParm.gridNum[1];
        if(index.x >= data.gpuParm.gridNum[0] || index.y >= data.gpuParm.gridNum[1] || index.z >= data.gpuParm.gridNum[2])
            return;
        float rhs = 0;
        for(int dim=0;dim<3;++dim)
        {
            int gNum[3] = {data.gpuParm.gridNum[0], data.gpuParm.gridNum[1], data.gpuParm.gridNum[2]};
            gNum[dim]++;
            float rpos[3] = {index.x, index.y, index.z};
            for(int drift = 0;drift < 2;++drift)
            {
                rpos[dim] += drift;
                auto vel = findVel(data, dim, make_float3(rpos[0] / gNum[0], rpos[1] / gNum[1], rpos[2] / gNum[2]));
                rhs += 2 * (drift - 0.5) * vel;
            }
        }
        data.b[i] = -rhs / data.gpuParm.dx * rho / data.gpuParm.dt;
    }
    void __global__ computeAp(GridData data, float*p, float*Ap)
    {
        int3 index;
        float rho = 1;

        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;

        int i = index.x + index.y * data.gpuParm.gridNum[0] + index.z * data.gpuParm.gridNum[0] * data.gpuParm.gridNum[1];
        if(index.x >= data.gpuParm.gridNum[0] || index.y >= data.gpuParm.gridNum[1] || index.z >= data.gpuParm.gridNum[2])
            return;
        
        for(int dim=0;dim<3;++dim)
        {
            for(int drift =-1;drift<=1;drift+=2)
            {
                int gNum[3] = {data.gpuParm.gridNum[0], data.gpuParm.gridNum[1], data.gpuParm.gridNum[2]};
                gNum[dim]++;
                float rpos[3] = {index.x, index.y, index.z};
                rpos[dim] += drift;
                auto point = findPoint(data, make_float3(rpos[0] / gNum[0], rpos[1] / gNum[1], rpos[2] / gNum[2]));

            }
        }
    }
    void GridData::constructOctree()
    {
        generateMortonCode<<<parm.blockPerGrid, parm.threadsPerBlock>>>(*this);
        generateVelMortonCode<<<parm.velBlockPerGrid, parm.threadsPerBlock>>>(*this);
        cudaThreadSynchronize();

        thrust::sort_by_key(thrust::device_ptr<unsigned long>(pKey),
                            thrust::device_ptr<unsigned long>(pKey + parm.gridNum[0] * parm.gridNum[1] * parm.gridNum[2]),
                            thrust::device_ptr<PointData>(pData));
        
        for(int i=0;i<3;++i)
        {
            thrust::sort_by_key(thrust::device_ptr<unsigned long>(velKey[i]),
                            thrust::device_ptr<unsigned long>(velKey[i] + 
                                (parm.gridNum[i]+1) * parm.gridNum[(i+1)%3] * parm.gridNum[(i+2)%3]),
                            thrust::device_ptr<float>(vel[i]));
        }

        genOctree<<<parm.blockPerGrid, parm.threadsPerBlock>>>(pKey, pTree, (int*)(&gpuParm.gridNum));
        genVelOctree<<<parm.velBlockPerGrid, parm.threadsPerBlock>>>(*this, gpuParm);
        cudaThreadSynchronize();

        findOctreeRoot<<<parm.blockPerGrid, parm.threadsPerBlock>>>
            (pKey, pTree, make_int3(parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]));
        for(int dim = 0;dim<3;++dim)
        {
            int gNum[3] = {parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]};
            gNum[dim]++;
            findOctreeRoot<<<parm.velBlockPerGrid, parm.threadsPerBlock>>>(pKey, pTree, make_int3(gNum[0], gNum[1], gNum[2]));
        }
        cudaThreadSynchronize();
    }
    void GridData::initData(vec3f bmin, vec3f bmax, float dx, float dt)
    {
        parm.bmin = make_float3(bmin[0], bmin[1], bmin[2]);
        parm.dx = dx;
        parm.dt = dt;
        parm.gridNum = ceil((bmax-bmin) / dx);
        for(int i=0;i<3;++i)
            if(parm.gridNum[i] <= 0)
                parm.gridNum[i] = 1;
        parm.bmax = parm.bmin + make_float3(parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]) * dx;
        printf("grid Num is (%d,%d,%d)\n", parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]);
        std::vector<std::string> pros;
        int gNum[3] = {parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]};
        data.initBox(gNum, parm.bmin, parm.bmax, make_float3(0), dx, pros);
        // cudaMalloc(&pData, sizeof(PointData) * parm.gridNum[0] * parm.gridNum[1] *parm.gridNum[2]);
        // cudaMalloc(&pKey, sizeof(unsigned long) * parm.gridNum[0] * parm.gridNum[1] *parm.gridNum[2]);
        // cudaMalloc(&pTree.nodes, sizeof(TreeNode) * (parm.gridNum[0] * parm.gridNum[1] *parm.gridNum[2] - 1));
        // for(int i=0;i<3;++i)
        // {
        //     int gNum[3] = {parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]};
        //     gNum[i]++;
        //     cudaMalloc(&vel[i], sizeof(float) * gNum[0] * gNum[1] * gNum[2]);
        //     cudaMalloc(&velKey[i], sizeof(unsigned long) * gNum[0] * gNum[1] * gNum[2]);
        //     cudaMalloc(&velTree[i].nodes, sizeof(TreeNode) * 
        //         (gNum[0] * gNum[1] * gNum[2] - 1));
        // }

        // dim3 threadsPerBlock(8,8,8);
        // parm.threadsPerBlock = threadsPerBlock;
        // parm.blockPerGrid = dim3(ceil(parm.gridNum[0] * 1.0 / threadsPerBlock.x), 
        //     ceil(parm.gridNum[1] * 1.0 / threadsPerBlock.y), 
        //     ceil(parm.gridNum[2] * 1.0 / threadsPerBlock.z));
        // parm.velBlockPerGrid = dim3(ceil((parm.gridNum[0] + 1) * 1.0 / threadsPerBlock.x), 
        //     ceil((parm.gridNum[1] + 1) * 1.0 / threadsPerBlock.y), 
        //     ceil((parm.gridNum[2] + 1) * 1.0 / threadsPerBlock.z));

        // cudaMemcpyToSymbol(&gpuParm, &parm, sizeof(Parms));
        // initPos<<<parm.blockPerGrid, parm.threadsPerBlock>>>(*this);
        // cudaThreadSynchronize();

        // int bufferBytesCount = sizeof(PointData) * parm.gridNum[0] * parm.gridNum[1] *parm.gridNum[2];
        // bufferBytesCount += 3 * sizeof(float) * (parm.gridNum[0] + 1) * (parm.gridNum[1] + 1) * (parm.gridNum[2]+1);
        // cudaMalloc(&buffer, bufferBytesCount);

        // //iterate buffers
        // int baseBytesCount = sizeof(float) * parm.gridNum[0] * parm.gridNum[1] *parm.gridNum[2];
        // cudaMalloc(&r, baseBytesCount);
        // cudaMalloc(&b, baseBytesCount);
        // cudaMalloc(&press, baseBytesCount);
    }

    void GridData::advection()
    {
        semiLaAdvection<<<parm.blockPerGrid, parm.threadsPerBlock>>>(*this);
        semiLaVelAdvection<<<parm.velBlockPerGrid, parm.threadsPerBlock>>>(*this);
        cudaThreadSynchronize();
        int basedrift = sizeof(PointData) * parm.gridNum[0] * parm.gridNum[1] * parm.gridNum[2];
        cudaMemcpy(pData,buffer, basedrift, cudaMemcpyDeviceToDevice);
        for(int dim=0;dim<3;++dim)
        {
            int gNum[3] = {parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]};
            gNum[dim]++;
            int bCounts = sizeof(float) * gNum[0] * gNum[1] * gNum[2];
            cudaMemcpy(vel[dim], buffer + basedrift, bCounts, cudaMemcpyDeviceToDevice);
            basedrift += bCounts;
        }
    }
    void GridData::applyOtherForce()
    {

    }
    void GridData::PossionSolver()
    {
        computeRHS<<<parm.blockPerGrid, parm.threadsPerBlock>>>(*this);
        cudaThreadSynchronize();


    }
    void GridData::step()
    {
        advection();
        constructOctree();
        PossionSolver();
    }












    void gpuVDBGrid::addProperty(std::string map_key)
    {
        float* buf;
        cudaMalloc(&buf, sizeof(float) * size);
        data[map_key] = buf;
        //data.insert(map_key, buf);
    }
    __global__ void genBoxPos(gpuVDBGrid grid, float3 bmin, float3 bmax, int gNum[3], float3 drift, float dx)
    {
        int3 index;
        index.x = blockIdx.x * blockDim.x + threadIdx.x;
        index.y = blockIdx.y * blockDim.y + threadIdx.y;
        index.z = blockIdx.z * blockDim.z + threadIdx.z;
        if(index.x >= gNum[0] || index.y >= gNum[1] || index.z >= gNum[2])
            return;

        int i = index.x + index.y * gNum[0] + index.z * gNum[0] * gNum[1];
        grid.pos[i] = bmin +  dx * make_float3(index.x, index.y, index.z) + drift;
        grid.key[i] = morton3d((grid.pos[i] - bmin) / (bmax - bmin));

    }

    void gpuVDBGrid::initBox(int gNum[3], float3 bmin, float3 bmax, float3 drift, float dx, std::vector<std::string> properties)
    {
        size = gNum[0] * gNum[1] * gNum[2];
        this->drift = drift;
        this->dx = dx;
        cudaMalloc(&pos, sizeof(float3) * size);
        cudaMalloc(&key, sizeof(unsigned long) * size);
        for(int i=0;i<properties.size();++i)
        {
            addProperty(properties[i]);
        }
        dim3 threadsPerBlock(8,8,8);
        auto blockPerGrid = dim3(ceil(gNum[0] * 1.0 / threadsPerBlock.x), 
            ceil(gNum[1] * 1.0 / threadsPerBlock.y), 
            ceil(gNum[2] * 1.0 / threadsPerBlock.z));
        genBoxPos<<<blockPerGrid, threadsPerBlock>>>(*(this), bmin, bmax, gNum, drift, dx);
        cudaThreadSynchronize();

        thrust::sort_by_key(thrust::device_ptr<unsigned long>(key),
                            thrust::device_ptr<unsigned long>(key + size),
                            thrust::device_ptr<float3>(pos));
        thrust::sort_by_key(thrust::device_ptr<unsigned long>(key),
                            thrust::device_ptr<unsigned long>(key + size),
                            thrust::device_ptr<float3>(key));
        genOctree<<<blockPerGrid, threadsPerBlock>>>(key, tree, gNum);
        cudaThreadSynchronize();

    }














    //node define
    struct generateAdaptiveGridGPU : zeno::INode{
        virtual void apply() override {
            auto bmin = get_input("bmin")->as<zeno::NumericObject>()->get<vec3f>();
            auto bmax = get_input("bmax")->as<zeno::NumericObject>()->get<vec3f>();
            auto dx = get_input("dx")->as<zeno::NumericObject>()->get<float>();
            float dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
            //int levelNum = get_input("levelNum")->as<zeno::NumericObject>()->get<int>();
                
            auto data = zeno::IObject::make<GridData>();
            data->initData(bmin, bmax, dx, dt);
            set_output("gridData", data);
        }
    };
    ZENDEFNODE(generateAdaptiveGridGPU, {
            {"bmin", "bmax", "dx", "dt"},
            {"gridData"},
            {},
            {"AdaptiveSolver"},
    });

    struct AdaptiveGridGPUSolver : zeno::INode{
        virtual void apply() override {
            auto data = get_input<GridData>("gridData");
            
            set_output("gridData", data);
        }
    };
    ZENDEFNODE(AdaptiveGridGPUSolver, {
            {"gridData"},
            {"gridData"},
            {},
            {"AdaptiveSolver"},
    });
}