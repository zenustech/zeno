// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/geometry/AdaptiveGrid.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/py_interop/BvhView.hpp"
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

struct A {
  constexpr A(int a) noexcept : a{a} {}
  int a;
};

struct B {
  constexpr B() {}
  constexpr B(B &&o) {}
  constexpr B(const B &o) {}
  constexpr B &operator=(B &&o) { return *this; }
  constexpr B &operator=(const B &o) { return *this; }
  ~B() noexcept = default;
  int a{1};
  float b{2};
};

__global__ void test(int *a) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) {
#if 0
        int a = -1;
        printf("a before : %d\n", a);
        auto pa = zs::construct_at<A>((A *)&a, 2);
        printf("a after : %d\n", a);
        zs::destroy_at((A *)&a);
#else
    B a;
    zs::ValueOrRef<B> b(a);
    printf("000");
    printf("self <a, b>: %d, %f\n", a.a, a.b);
    b.get().a = 3;
    printf("after proxy op\n");
    printf("self <a, b>: %d, %f\n", a.a, a.b);
    printf("proxy <a, b>: %d, %f\n", b.get().a, b.get().b);
    zs::ValueOrRef<B> c{};
    printf("new proxy <a, b>: %d, %f\n", c.get().a, c.get().b);
    printf("before c = b\n");
    c = b;
    printf("after c = b\n");
    printf("c copy-assigned from b: <a, b>: %d, %f, b validity: %d\n",
           c.get().a, c.get().b, b.isValid());
    printf("before c = move(b)\n");
    c = zs::move(b);
    printf("after c = move(b)\n");
    printf("c move-assigned from b: <a, b>: %d, %f, b validity: %d\n",
           c.get().a, c.get().b, b.isValid());
    printf("b later: <a, b>: %d, %f\n", b.get().a, b.get().b);
    using TTT = char[3][4][5];
    printf("arr[3][4][5]: [%d] %d, %d, %d\n", (int)zs::rank<TTT>::value,
           (int)zs::extent<TTT, 0>::value, (int)zs::extent<TTT, 1>::value,
           (int)zs::extent<TTT, 2>::value);
#endif
  }
  // printf("[%d]: %d\n", id, a[id]);
}

struct ZSCULinkTest : INode {
  void apply() override {

    {
      zs::VdbGrid<3, float, zs::index_sequence<3, 4, 5>> ag;
      using TT = RM_CVREF_T(ag);
      fmt::print("adaptive grid type: {}\n",
                 zs::get_var_type_str(ag).asChars());
      // fmt::print("tile bits: {}\n", zs::get_type_str<TT::tile_bits_type>());
      // fmt::print("hierarchy bits: {}\n",
      // zs::get_type_str<TT::hierarchy_bits_type>());

      fmt::print("num total blocks: {}\n", ag.numTotalBlocks());
      auto hag = ag.clone({zs::memsrc_e::host, -1});
    }

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

struct ZSCUShowActiveContext : INode {
  void apply() override {
    using namespace zs;
    CUcontext ctx = nullptr;
    auto ec = cuCtxGetCurrent(&ctx);
    bool valid = true;
    int did = Cuda::get_default_device();
    int devid = did;
    if (ec != CUDA_SUCCESS) {
      const char *errString = nullptr;
      cuGetErrorString(ec, &errString);
      checkCuApiError((u32)ec, errString);
      valid = false;
    } else {
      if (ctx != NULL) {
        auto ec = cuCtxGetDevice(&devid);
        if (ec != CUDA_SUCCESS) {
          const char *errString = nullptr;
          cuGetErrorString(ec, &errString);
          checkCuApiError((u32)ec, errString);
          valid = false;
        }
      } // otherwise, no context has been initialized yet.
    }
    fmt::print("valid: {}, cuda default did: {}, current active did: {}\n",
               valid, did, devid);
  }
};

ZENDEFNODE(ZSCUShowActiveContext, {
                                      {},
                                      {},
                                      {},
                                      {"ZPCTest"},
                                  });

struct ZSCUSetActiveDevice : INode {
  void apply() override {
    using namespace zs;
    auto did = get_input2<int>("device_no");
    bool success = Cuda::set_default_device(did);
    // Cuda::context(did).setContext();
    fmt::print("====================================\n");
    fmt::print("====================================\n");
    fmt::print("==========set dev {} active=========\n", did);
    fmt::print("==============={}=================\n", success);
    fmt::print("====================================\n");
    fmt::print("====================================\n");
  }
};

ZENDEFNODE(ZSCUSetActiveDevice, {
                                    {{"int", "device_no", "0"}},
                                    {},
                                    {},
                                    {"ZPCTest"},
                                });

} // namespace zeno
