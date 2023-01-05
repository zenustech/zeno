// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include "Structures.hpp"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/math/DihedralAngle.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
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

struct ZSCUMathTest : INode {
    void apply() override {
        using namespace zs;
        constexpr int n = 100;
        using TV = zs::vec<float, 3>;
        //TV m_X[4] = {TV{0, 0, 0}, TV{0, 1, 0}, TV{0, 0, -1}, TV{0, 0, 1}};
        TV m_X[4] = {TV{0, 0, 0}, TV{0, 1, 0}, TV{0, 0, -1}, TV{-limits<float>::epsilon() * 5, 1, -1}};
        auto ra = zs::dihedral_angle(m_X[2], m_X[0], m_X[1], m_X[3], exec_seq);
        auto grad = zs::dihedral_angle_gradient(m_X[2], m_X[0], m_X[1], m_X[3], exec_seq);
        auto hess = zs::dihedral_angle_hessian(m_X[2], m_X[0], m_X[1], m_X[3], exec_seq);
        auto n1 = (m_X[0] - m_X[2]).cross(m_X[1] - m_X[2]);
        auto n2 = (m_X[1] - m_X[3]).cross(m_X[0] - m_X[3]);
        auto e = (m_X[1] - m_X[0]).norm(exec_seq);
        auto h = (n1.norm(exec_seq) + n2.norm(exec_seq)) / 2 / (e * 3);
        fmt::print("rest angle is: {}, e: {}, h: {}\n", ra, e, h);
        getchar();
    }
};

ZENDEFNODE(ZSCUMathTest, {
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
