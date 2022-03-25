// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include <cassert>
#include <cuda_runtime.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>

namespace zeno {

__global__ void test(int *a) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  printf("[%d]: %d\n", id, a[id]);
}

struct ZSCULinkTest : INode {
  void apply() override {
    constexpr int n = 100;

    int *a = nullptr;
    cudaMalloc((void **)&a, n * sizeof(int));

#if 1
    std::vector<int> ha(n);
#else
    zs::Vector<int> ha{n, zs::memsrc_e::host, -1};
#endif
    for (int i = 0; i != n; ++i)
      ha[i] = i;

    cudaMemcpy(a, ha.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    test<<<1, n>>>(a);
    cudaDeviceSynchronize();

    cudaFree(a);

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
