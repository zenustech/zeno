// #include "zensim/container/Vector.hpp"
// #include "zensim/geometry/VdbLevelSet.h"
#include "Structures.hpp"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/graph/ConnectedComponents.hpp"
#include "zensim/math/DihedralAngle.hpp"
#include "zensim/math/matrix/SparseMatrix.hpp"
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
        TV m_X[4] = {TV{0, 0, 0}, TV{0, 1, 0}, TV{0, 0, -1}, TV{-detail::deduce_numeric_epsilon<float>() * 5, 1, -1}};
        auto ra = zs::dihedral_angle(m_X[2], m_X[0], m_X[1], m_X[3], exec_seq);
        auto grad = zs::dihedral_angle_gradient(m_X[2], m_X[0], m_X[1], m_X[3], exec_seq);
        auto hess = zs::dihedral_angle_hessian(m_X[2], m_X[0], m_X[1], m_X[3], exec_seq);
        auto n1 = (m_X[0] - m_X[2]).cross(m_X[1] - m_X[2]);
        auto n2 = (m_X[1] - m_X[3]).cross(m_X[0] - m_X[3]);
        auto e = (m_X[1] - m_X[0]).norm(exec_seq);
        auto h = (n1.norm(exec_seq) + n2.norm(exec_seq)) / 2 / (e * 3);
        fmt::print("rest angle is: {}, e: {}, h: {}\n", ra, e, h);

        using T = zs::vec<float, 3, 3>;
        const zs::SparseMatrix<T, true> spmat{100, 100};
        auto spv = proxy<execspace_e::host>(spmat);
        fmt::print("default spmat(0, 1): {}\n", spv(0, 1).extent);

        /// csr
        {
            zs::SparseMatrix<int, true> spmat{7, 7, memsrc_e::um, 0};
            zs::Vector<int> is{6}, js{6}, vs{6};
            is[0] = 0;
            js[0] = 1;
            is[1] = 1;
            js[1] = 2;
            is[2] = 1;
            js[2] = 5;
            is[3] = 3;
            js[3] = 4;
            is[4] = 3;
            js[4] = 6;
            is[5] = 4;
            js[5] = 6;
            for (auto &v : vs)
                v = 1;

            is = is.clone({memsrc_e::um, 0});
            js = js.clone({memsrc_e::um, 0});
            vs = vs.clone({memsrc_e::um, 0});
            spmat.build(zs::cuda_exec(), 7, 7, range(is), range(js), range(vs));
            fmt::print("row major csr:\n");
            for (int i = 0; i < 7; ++i) {
                auto bg = spmat._ptrs[i];
                fmt::print("row [{}] offset [{}]: ", i, bg);
                for (; bg != spmat._ptrs[i + 1]; ++bg)
                    fmt::print("<{}>  ", spmat._inds[bg] /*, spmat._vals[bg]*/);
                fmt::print("\n");
            }

            // alternative build
            spmat.build(zs::cuda_exec(), 7, 7, range(is), range(js), true_c);
            fmt::print("row major csr (sym, pure topo):\n");
            for (int i = 0; i < 7; ++i) {
                auto bg = spmat._ptrs[i];
                fmt::print("row [{}] offset [{}]: ", i, bg);
                for (; bg != spmat._ptrs[i + 1]; ++bg)
                    fmt::print("<{}>  ", spmat._inds[bg] /*, spmat._vals[bg]*/);
                fmt::print("\n");
            }

            zs::Vector<int> fas{7, memsrc_e::um, 0};
            union_find(zs::cuda_exec(), spmat, range(fas));

            for (int i = 0; i != 7; ++i)
                fmt::print("check first stat[{}]: {}\n", i, fas[i]);
        }

        constexpr int N = 7;
        zs::Vector<int> nidx{N + 1};
        nidx[0] = 0; //
        nidx[1] = 1; //
        nidx[2] = 4;
        nidx[3] = 5;
        nidx[4] = 7; //
        nidx[5] = 9; //
        nidx[6] = 10;
        nidx[7] = 12; //
        zs::Vector<int> nlist{12};
        nlist[0] = 1; //
        nlist[1] = 0; //
        nlist[2] = 2;
        nlist[3] = 5;
        nlist[4] = 1; //
        nlist[5] = 4; //
        nlist[6] = 6;
        nlist[7] = 3; //
        nlist[8] = 6;
        nlist[9] = 1;  //
        nlist[10] = 3; //
        nlist[11] = 4;
        zs::Vector<int> nstat{N};
        nstat.reset(0);

        auto pol = zs::omp_exec();
        auto representative = [&nstat](const int idx) -> int {
            int curr = nstat[idx];
            if (curr != idx) {
                int next, prev = idx;
                while (curr > (next = nstat[curr])) {
                    nstat[prev] = next;
                    prev = curr;
                    curr = next;
                }
            }
            return curr;
        };

        // init
        pol(range(N), [&](int v) {
            const int beg = nidx[v];
            const int end = nidx[v + 1];
            int m = v;
            int i = beg;
            while ((m == v) && (i < end)) {
                m = std::min(m, nlist[i]);
                i++;
            }
            nstat[v] = m;
        });
        // union find
        pol(range(N), [&](int v) {
            int vstat = representative(v);
            auto beg = nidx[v];
            auto end = nidx[v + 1];
            for (int i = beg; i < end; i++) {
                const int nli = nlist[i];
                if (v > nli) {
                    int ostat = representative(nli);
                    bool repeat;
                    do {
                        repeat = false;
                        if (vstat != ostat) {
                            int ret;
                            if (vstat < ostat) {
                                if ((ret = atomic_cas(exec_omp, &nstat[ostat], ostat, vstat)) != ostat) {
                                    ostat = ret;
                                    repeat = true;
                                }
                            } else {
                                if ((ret = atomic_cas(exec_omp, &nstat[vstat], vstat, ostat)) != vstat) {
                                    vstat = ret;
                                    repeat = true;
                                }
                            }
                        }
                    } while (repeat);
                }
            }
        });

        for (int i = 0; i != N; ++i)
            fmt::print("stat[{}]: {}\n", i, nstat[i]);

        getchar();
    }
};

ZENDEFNODE(ZSCUMathTest, {
                             {},
                             {},
                             {},
                             {"ZPCTest"},
                         });

struct ZSTestIterator : INode {
    struct CustomContainer {
        decltype(auto) operator[](int i) & {
            return arr[i];
        }
        decltype(auto) operator[](int i) && {
            return std::move(arr[i]);
        }

        int arr[100];
    };

    template <typename Container, typename Index>
    decltype(auto) authAndAccess(Container &&c, Index i) {
        return std::forward<Container>(c)[i];
    }
    template <typename Container>
    decltype(auto) authAndAccess(Container &&c) {
        return std::forward<Container>(c);
    }
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;

        auto &&bb = 1;
        fmt::print("check \"a\" size: {}\n", sizeof("a"));
        fmt::print("[{}]\n[{}]\n", get_type_str<decltype(bb)>(), detail::get_type_str_helper<decltype(bb)>());

        TileVector<float, 32> tv{{{"w", 1}, {"id", 1}, {"v", 3}, {"var", 1}}, 100};
        auto ompExec = omp_exec();
        // initialize
        ompExec(range(100), [&, tv = proxy<space>({}, tv)](int i) mutable {
            tv(0, i) = i;
            tv("id", i) = reinterpret_bits<zs::f32>(i + 1);
            tv.tuple(dim_c<3>, "v", i) = zs::vec<float, 3>{-i, i, i * i};
            tv("var", i, int_c) = ((i * i) << (sizeof(int) / 2 * 8)) | i;
        });
        puts("print by proxies");
        seq_exec()(range(10), [&, tv = proxy<space>({}, tv)](int i) {
            fmt::print("tv[{}]: w[{}], v[{}, {}, {}], id[{}]\n", i, tv("w", i), tv("v", 0, i), tv("v", 1, i),
                       tv("v", 2, i), tv.pack(dim_c<1>, "id", i).reinterpret_bits(int_c)[0]);
        });
        puts("print \"w\" by default iterators");
        // auto bg = tv.begin(0);
        // auto ed = tv.end(0);
        auto [bg, ed] = range(tv, 0);
        auto viter = std::begin(range(tv, "v", dim_c<3>, float_c));
        auto i0Iter = tv.begin("var", 0, dim_c<1>, wrapt<i16>{});
        auto i1Iter = tv.begin("var", 1, dim_c<1>, wrapt<i16>{});
        using TV = zs::vec<float, 3>;
#if 1
        auto idIter = tv.begin(1, dim_c<1>, int_c);
        ompExec(range(10), [bg = bg, idIter = idIter, viter = viter, i0Iter = i0Iter, i1Iter = i1Iter](int i) mutable {
            fmt::print("tv[{}]: {}, id {}, v [{}, {}, {}], split [{}, {}]\n", i, bg[i], idIter[i], viter[i][0],
                       viter[i][1], viter[i][2], i0Iter[i], i1Iter[i]);
        });
        {
            std::string str;
            char ch;
            int num;
            auto [a, b, c] = zs::make_tuple("sb", 'a', 2);
            zs::tie(str, ch, num) = zs::make_tuple("SB", 'A', 4);
            fmt::print("<a, b, c>: ({}, {}, {}) ({}, {}, {})\n", a, b, c, str, ch, num);
        }
#endif
        std::vector<int> a{-1, -2, -3};
        fmt::print("container types: [{}], [{}]\n", get_var_type_str(authAndAccess(a)),
                   get_var_type_str(authAndAccess(std::vector<int>{1, 2, 3})));
        fmt::print("container ele types: [{}], [{}]\n", get_var_type_str(authAndAccess(a, 0)),
                   get_var_type_str(authAndAccess(std::vector<int>{1, 2, 3}, 0)));
        /////
        CustomContainer b;
        fmt::print("container types: [{}], [{}]\n", get_var_type_str(authAndAccess(b)),
                   get_var_type_str(authAndAccess(CustomContainer{})));
        fmt::print("container ele types: [{}], [{}]\n", get_var_type_str(authAndAccess(b, 0)),
                   get_var_type_str(authAndAccess(CustomContainer{}, 0)));
    }
};

ZENDEFNODE(ZSTestIterator, {
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
