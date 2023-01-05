// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include <cassert>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include "Structures.hpp"
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
#include "zensim/omp/execution/ExecutionPolicy.hpp"
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

struct ZSTestExtraction : INode {
  void apply() override {
    using namespace zs;
    constexpr auto space = execspace_e::openmp;
    auto zspars = get_input<ZenoParticles>("ZSParticles");
    auto ret = std::make_shared<PrimitiveObject>();
    const auto &verts = zspars->getParticles().clone({memsrc_e::host, -1});
    const auto &eles = (*zspars)[ZenoParticles::s_bendingEdgeTag].clone({memsrc_e::host, -1});
    auto &pos = ret->verts;
    auto &lines = ret->lines;
    pos.resize(verts.size());
    lines.resize(eles.size());
    auto ompExec = omp_exec();
    // verts
    ompExec(range(pos.size()), [&pos, verts = proxy<space>({}, verts)](int i) mutable {
      auto p = verts.pack(dim_c<3>, "x", i);
      pos[i] = vec3f{p[0], p[1], p[2]};
    });
    // eles
    ompExec(range(lines.size()), [&lines, eles = proxy<space>({}, eles)](int i) mutable {
      auto line = eles.pack(dim_c<4>, "inds", i).reinterpret_bits(int_c);
      lines[i] = vec2i{line[1], line[2]};
    });
    set_output("prim", ret);
  }
};

ZENDEFNODE(ZSTestExtraction, {
                           {"ZSParticles"},
                           {"prim"},
                           {},
                           {"ZPCTest"},
                       });


} // namespace zeno
