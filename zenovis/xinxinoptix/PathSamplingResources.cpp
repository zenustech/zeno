#include "PathSamplingResources.h"
#include "PathSamplingData.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace zeno_path_sampling {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("Path sampler CUDA: ")+cudaGetErrorString(e));}
struct Buffer {
    void* data=nullptr;
    ~Buffer(){if(data)cudaFree(data);}
    void upload(const unsigned char* source,size_t bytes) {
        if(data)return;
        void* allocation=nullptr;
        check(cudaMalloc(&allocation,bytes));
        const auto error=cudaMemcpy(allocation,source,bytes,cudaMemcpyHostToDevice);
        if(error!=cudaSuccess){cudaFree(allocation);check(error);}
        data=allocation;
    }
};
}
struct Resources::Impl {Buffer scalar,vector2;};
Resources::Resources():impl(std::make_unique<Impl>()){}
Resources::~Resources()=default;
Launch Resources::launch(unsigned mode,unsigned sampleIndex,unsigned seed) {
    Launch result{};result.mode=mode;result.sampleIndex=sampleIndex;result.seed=seed;
    result.independentTiles=1;
    if(mode==Legacy)return result;
    if(mode!=FAST)throw std::invalid_argument("Unknown path sampler mode");
    auto& p=*impl;
    p.scalar.upload(fastScalar,sizeof(fastScalar));
    p.vector2.upload(fastVector2,sizeof(fastVector2));
    result.scalar=static_cast<const unsigned char*>(p.scalar.data);
    result.vector2=static_cast<const unsigned char*>(p.vector2.data);
    return result;
}
}
