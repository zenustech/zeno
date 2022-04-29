// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include <cassert>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/cuda/Cuda.h"
namespace zeno {

__global__ void test(int *a) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  printf("[%d]: %d\n", id, a[id]);
}

struct ZSCULinkTest : INode {
  void apply() override {
    constexpr int n = 100;
    // cuInit(0);
    (void)zs::Cuda::instance();
    puts("1");
    int *a = nullptr;
    // cudaMalloc((void **)&a, n * sizeof(int));
    a = (int*)zs::allocate(zs::mem_um, n * sizeof(int), sizeof(int));
    puts("2");

#if 1
    std::vector<int> ha(n);
#else
    zs::Vector<int> ha{n, zs::memsrc_e::host, -1};
#endif
    for (int i = 0; i != n; ++i)
      ha[i] = i;
    puts("3");
    cudaMemcpy(a, ha.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    test<<<1, n>>>(a);
    cudaDeviceSynchronize();

    puts("4");
    // cudaFree(a);
    // zs::deallocate(zs::mem_um, a, );
    zs::raw_memory_resource<zs::um_mem_tag>::instance().deallocate(a, n * sizeof(int));
    puts("5");

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, "", "ahh", 0, NULL, NULL);

    printf("done!\n");
    getchar();
  }
};

ZENDEFNODE(ZSCULinkTest, {
                           {},
                           {},
                           {},
                           {"ZPCTest"},
                       });

} // namespace zeno
