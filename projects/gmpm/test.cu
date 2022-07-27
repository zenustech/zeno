// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
namespace zeno {

__global__ void test(int *a) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto pi = 3.1415926535897932384626433832795028841972L;
  printf("[%d]: %d, pi: %f, sqrt_pi: %f\n", id, a[id], (float)zs::g_pi,
         (float)zs::sqrt(pi));
  auto testfunc = []() { return zs::vec<double, 3>::ones(); };
  auto t = testfunc();
  t = zs::vec<double, 3>::zeros();
  //(void)(testfunc() = zs::vec<double, 3>::zeros());
}

struct ZSCULinkTest : INode {
  void apply() override {
    constexpr int n = 100;
    // cuInit(0);
    (void)zs::Cuda::instance();
    puts("1");
    int *a = nullptr;
    // cudaMalloc((void **)&a, n * sizeof(int));
    a = (int *)zs::allocate(zs::mem_um, n * sizeof(int), sizeof(int));
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
    zs::raw_memory_resource<zs::um_mem_tag>::instance().deallocate(
        a, n * sizeof(int));
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
