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

#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

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


template <typename VectorView> struct SomeFunc {
    constexpr void operator()(int i) { vals[i] = i; }
    VectorView vals;
};
template <typename T> SomeFunc(T) -> SomeFunc<T>;  // CTAD

struct cathy : INode {
    void apply() override {
        using namespace zs;
        auto cudaPol = cuda_exec().device(0).profile(true);
        auto ompPol = omp_exec();
        Vector<int> vals{4, memsrc_e::device, 0};
        Vector<int> vals1{4, memsrc_e::host};
        // PrimitiveObject -> Vector<vec3>
        //range, execution
        ompPol(Collapse{vals1.size()},
               [vals1 = proxy<execspace_e::openmp>(vals1)](int i) mutable { vals1[i] = i; });
#if 0
#pragma omp parallel for
        for (int i = 0; i < 4; ++i)
            vals1[i] = i;
#endif
        int id = 0;
// for (auto &v : vals) v = id++;
#if 0
  cudaPol(Collapse{vals.size()}, [vals = proxy<execspace_e::cuda>(vals)] __device__(int i) mutable {
    vals[i] = i;
    if (i == 0)
#if 0
      for (auto [x, y, z] : ndrange<3>(2)) {
        printf("iterating (%d, %d, %d)\n", x, y, z);
      }
#else
    for (int x = 0; x != 2; ++x)
        for (int y = 0; y != 2; ++y)
            for (int z = 0; z != 2; ++z)
    printf("iterating (%d, %d, %d)\n", x, y, z);
#endif
  });
#elif 0
        cudaPol(enumerate(vals), [] __device__(auto id, auto &v) mutable { v = id; });
#else
        cudaPol(Collapse{vals.size()}, SomeFunc{proxy<execspace_e::cuda>(vals)});
#endif

        vals1 = vals1.clone({memsrc_e::device, 0});

        auto tmp = vals.clone({memsrc_e::host, -1});
        for (int i = 0; i != vals.size(); ++i) fmt::print("{} ", tmp[i]);
        fmt::print("\n");

        cudaPol(zip(vals, vals1),
                [] __device__(int &val, int &val1) mutable { printf("%d, %d\n", val, val1); });

        ///
        constexpr std::size_t numPars = 10;
        TileVector<float, 32> pars{{{"m", 1}, {"x", 3}}, numPars, memsrc_e::um, 0};
        pars.append_channels(cudaPol, {{"v", 3}});
        pars.resize(100);
        // init
        cudaPol(Collapse{pars.size()},
                [pars = proxy<execspace_e::cuda>({}, pars)] __device__(int pi) mutable {
                    pars("m", pi) = 1;
                    pars("x", 0, pi) = 0;
                    pars("x", 1, pi) = pi;

                    pars("v", 1, pi) = -pi;
                });

        auto tmp1 = proxy<execspace_e::host>({}, pars);
        for (int i = 0; i != numPars; ++i)
            fmt::print("par[{}]: mass {}, pos {}, {}, {}; vy: {}\n", i, tmp1("m", i), tmp1("x", 0, i),
                       tmp1("x", 1, i), tmp1("x", 2, i), tmp1("v", 1, i));
    }
};

ZENDEFNODE(cathy, {
                             {},
                             {},
                             {},
                             {"ZPCTest"},
                         });

} // namespace zeno
