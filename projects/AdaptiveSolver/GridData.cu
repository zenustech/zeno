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

    int __device__ findPointIndex(gpuVDBGrid data, float3 rpos)
    {
        auto key = morton3d(rpos);
        int node_type = 1, index = data.tree.rootIndex;
        do
        {
            if(computePrefix2(key, data.key[index]) > data.tree.nodes[index].delta_node)
            {
                node_type = data.tree.nodes[index].rType;
                index = data.tree.nodes[index].rChild;
            }
            else
            {
                node_type = data.tree.nodes[index].lType;
                index = data.tree.nodes[index].lChild;
            }
        }
        while (node_type == 1);
        return index;
    }
    void __device__ intepolateValue(gpuVDBGrid data, float3 pos, float* values)
    {
        float3 rpos = (pos - data.bmin - data.drift) / (data.bmax - data.bmin);
        if(rpos.x < 0 || rpos.y < 0 || rpos.z < 0 || rpos.x > 1 || rpos.y >1 || rpos.z > 1)
            return;
        
        float3 index = (pos - data.bmin - data.drift) / data.dx;
        int baseIndex[3] = {floor(index.x), floor(index.y), floor(index.z)};
        float w[3] = {index.x - baseIndex[0], index.y - baseIndex[1], index.z-baseIndex[2]};
        //float3 basePos = data.bmin + data.drift + data.dx * make_float3(baseIndex[0], baseIndex[1], baseIndex[2]);
        for(int i=0;i<2;++i)
        for(int j=0;j<2;++j)
        for(int k=0;k<2;++k)
        {
            int itIndex[3] = {baseIndex[0] + i, baseIndex[1] + j, baseIndex[2] + k};
            float3 itpos = make_float3(itIndex[0], itIndex[1], itIndex[2]) / 
                make_float3(data.gridNum[0],data.gridNum[1],data.gridNum[2]);
            int pindx = findPointIndex(data, itpos);
            for(int d=0;d<data.prosNum;++d)
                values[d] += (i*w[0]+(1-i)*(1-w[0]))*
                                (j*w[1]+(1-j)*(1-w[1]))*
                                (k*w[2]+(1-k)*(1-w[2]))*data.data[d][pindx];
        }
    }


    void __global__ genOctree(unsigned long* key, Octree& tree, int gridNum[3])
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
        float vel[3], itdata[10];
        for(int dim=0;dim<3;++dim)
            intepolateValue(data.vel[dim], data.data.pos[i], vel + dim);
        auto midpos = data.data.pos[i] - 0.5 * data.gpuParm.dt * make_float3(vel[0], vel[1], vel[2]);

        for(int dim=0;dim<3;++dim)
            intepolateValue(data.vel[dim], midpos, vel + dim);

        auto newpos = midpos - 0.5 * data.gpuParm.dt * make_float3(vel[0], vel[1], vel[2]);
        intepolateValue(data.data, newpos, itdata);

        for(int pindx = 0; pindx < data.data.prosNum; ++pindx)
        {
            data.data_buf->data[pindx][i] = itdata[pindx];
        }
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
            float vel[3];
            for(int l=0;l<3;++l)
                intepolateValue(data.vel[l], data.vel[l].pos[i], vel + l);
            auto midpos = data.vel[dim].pos[i] - 0.5 * data.gpuParm.dt * make_float3(vel[0], vel[1], vel[2]);
            for(int l=0;l<3;++l)
                intepolateValue(data.vel[l], midpos, vel + l);
            auto final_pos = midpos - 0.5 * data.gpuParm.dt * make_float3(vel[0], vel[1], vel[2]);

            intepolateValue(data.vel[dim], final_pos, vel + dim);

            for(int pro_indx = 0;pro_indx<data.vel[dim].prosNum;++pro_indx)
                data.data_buf.data
            //indx[0] = vel.x;indx[1] = vel.y;indx[2] = vel.z;
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
        pros.push_back("volume");
        pros.push_back("temperature");
        data.initBox(gNum, parm.bmin, parm.bmax, make_float3(0), dx, pros);
        pros.clear();
        pros.push_back("vel");
        for(int dim=0;dim<3;++dim)
        {
            int velGridNum[3] = {gNum[0], gNum[1], gNum[2]};
            velGridNum[dim] ++;
            float drift[3] = {0,0,0};
            drift[dim] -= 0.5 * dx;
            vel[dim].initBox(velGridNum, parm.bmin, parm.bmax, make_float3(drift[0], drift[1], drift[2]), dx, pros);
        }

        int bytesCount = 2 * sizeof(float) * gNum[0] * gNum[1] * gNum[2];

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
        data.push_back(buf);
        prosNum++;
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
        gridNum[0] = gNum[0];gridNum[1]=gNum[1];gridNum[2]=gNum[2];
        this->drift = drift;
        this->dx = dx;
        this->bmin = bmin;
        this->bmax = bmin + dx * make_float3(gNum[0], gNum[1],gNum[2]);
        prosNum = 0;
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
        genBoxPos<<<blockPerGrid, threadsPerBlock>>>(*(this), bmin, this->bmax, gNum, drift, dx);
        cudaThreadSynchronize();
        constructOctree();
    }

    void gpuVDBGrid::constructOctree()
    {
        thrust::sort_by_key(thrust::device_ptr<unsigned long>(key),
                            thrust::device_ptr<unsigned long>(key + size),
                            thrust::device_ptr<float3>(pos));
        dim3 threadsPerBlock(8,8,8);
        auto blockPerGrid = dim3(ceil(gridNum[0] * 1.0 / threadsPerBlock.x), 
            ceil(gridNum[1] * 1.0 / threadsPerBlock.y), 
            ceil(gridNum[2] * 1.0 / threadsPerBlock.z));
        genOctree<<<blockPerGrid, threadsPerBlock>>>(key, tree, gridNum);
        cudaThreadSynchronize();
    }

    gpuVDBGrid* gpuVDBGrid::deepCopy()
    {
        gpuVDBGrid * grid = new gpuVDBGrid();
        grid->bmax = this->bmax;
        grid->bmin = this->bmin;
        grid->drift = this->drift;
        grid->dx = this->dx;
        grid->gridNum[0] = this->gridNum[0];
        grid->gridNum[1] = this->gridNum[1];
        grid->gridNum[2] = this->gridNum[2];
        grid->size = this->size;
        grid->prosNum = this->prosNum;
        grid->tree.rootIndex = this->tree.rootIndex;

        cudaMalloc(&grid->tree.nodes, sizeof(TreeNode) * (grid->size - 1));
        cudaMemcpy(grid->tree.nodes, this->tree.nodes, sizeof(TreeNode) * (size-1), cudaMemcpyDeviceToDevice);

        cudaMalloc(&grid->pos, sizeof(float3) * grid->size);
        cudaMemcpy(grid->pos, this->pos, sizeof(float3) * size, cudaMemcpyDeviceToDevice);

        cudaMalloc(&grid->key, sizeof(unsigned long) * grid->size);
        cudaMemcpy(grid->key, this->key, sizeof(unsigned long) * size, cudaMemcpyDeviceToDevice);
        for(int i=0;i<prosNum;++i)
        {
            float* buf;
            cudaMalloc(&buf, sizeof(float) * grid->size);
            cudaMemcpy(buf, this->data[i], sizeof(float) * grid->size, cudaMemcpyDeviceToDevice);
            grid->data.push_back(buf);
        }

        return grid;
    }
    void gpuVDBGrid::assign(gpuVDBGrid* grid)
    {
        grid->bmax = this->bmax;
        grid->bmin = this->bmin;
        grid->drift = this->drift;
        grid->dx = this->dx;
        grid->gridNum[0] = this->gridNum[0];
        grid->gridNum[1] = this->gridNum[1];
        grid->gridNum[2] = this->gridNum[2];
        grid->size = this->size;
        grid->prosNum = this->prosNum;
        grid->tree.rootIndex = this->tree.rootIndex;

        cudaMemcpy(grid->tree.nodes, this->tree.nodes, sizeof(TreeNode) * (size-1), cudaMemcpyDeviceToDevice);
        cudaMemcpy(grid->pos, this->pos, sizeof(float3) * size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(grid->key, this->key, sizeof(unsigned long) * size, cudaMemcpyDeviceToDevice);
        for(int i=0;i<prosNum;++i)
        {
            float* buf;
            cudaMemcpy(buf, this->data[i], sizeof(float) * grid->size, cudaMemcpyDeviceToDevice);
            grid->data.push_back(buf);
        }
    }
    void gpuVDBGrid::clear()
    {
        cudaFree(tree.nodes);
        cudaFree(pos);
        cudaFree(key);
        for(int i=0;i<prosNum;++i)
            cudaFree(data[i]);
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