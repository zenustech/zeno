#include "../Utils.hpp"
#include "PBD.cuh"

namespace zeno {

void PBDSystem::PrimitiveHandle::initGeo() {
    // init rest volumes & edge lengths
    auto cudaPol = zs::cuda_exec();
    using namespace zs;
    constexpr auto space = zs::execspace_e::cuda;
    elesPtr->append_channels(cudaPol, {{"rv", 1}});
    cudaPol(zs::Collapse{elesPtr->size()},
            [eles = proxy<space>({}, *elesPtr), verts = proxy<space>({}, *vertsPtr)] ZS_LAMBDA(int ei) mutable {
                auto quad = eles.template pack<4>("inds", ei).template reinterpret_bits<int>();
                vec3 xs[4];
                for (int d = 0; d != 4; ++d)
                    xs[d] = verts.template pack<3>("x", quad[d]);
                vec3 ds[3] = {xs[1] - xs[0], xs[2] - xs[0], xs[3] - xs[0]};
#if 0
                mat3 D{};
                for (int d = 0; d != 3; ++d)
                    for (int i = 0; i != 3; ++i)
                        D(d, i) = ds[i][d];
                T vol = zs::abs(zs::determinant(D)) / 6;
#else
                T vol = zs::abs((ds[0]).cross(ds[1]).dot(ds[2])) / 6;
#endif
                eles("rv", ei) = vol;
            });
    edgesPtr->append_channels(cudaPol, {{"rl", 1}});
    cudaPol(zs::Collapse{edgesPtr->size()},
            [ses = proxy<space>({}, *edgesPtr), verts = proxy<space>({}, *vertsPtr)] ZS_LAMBDA(int sei) mutable {
                auto line = ses.template pack<2>("inds", sei).template reinterpret_bits<int>();
                vec3 xs[2];
                for (int d = 0; d != 2; ++d)
                    xs[d] = verts.template pack<3>("x", line[d]);
                ses("rl", sei) = (xs[1] - xs[0]).length();
            });
}

PBDSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<4>)
    : zsprimPtr{&zsprim, [](void *) {}}, models{zsprim.getModel()}, vertsPtr{&zsprim.getParticles(), [](void *) {}},
      elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}}, edgesPtr{&zsprim[ZenoParticles::s_edgeTag], [](void *) {}},
      surfTrisPtr{&zsprim[ZenoParticles::s_surfTriTag], [](void *) {}},
      surfEdgesPtr{&zsprim[ZenoParticles::s_surfEdgeTag], [](void *) {}},
      surfVertsPtr{&zsprim[ZenoParticles::s_surfVertTag], [](void *) {}}, vOffset{vOffset}, sfOffset{sfOffset},
      seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::tet)
        throw std::runtime_error("dimension of 4 but is not tetrahedra");
    initGeo();
    vOffset += getVerts().size();
    sfOffset += getSurfTris().size();
    seOffset += getSurfEdges().size();
    svOffset += getSurfVerts().size();
}

void PBDSystem::initialize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    stInds = tiles_t{vtemp.get_allocator(), {{"inds", 3}}, sfOffset};
    seInds = tiles_t{vtemp.get_allocator(), {{"inds", 2}}, seOffset};
    svInds = tiles_t{vtemp.get_allocator(), {{"inds", 1}}, svOffset};
    for (auto &primHandle : prims) {
        auto &verts = primHandle.getVerts();
        // record surface (tri) indices
        if (primHandle.category != ZenoParticles::category_e::curve) {
            auto &tris = primHandle.getSurfTris();
            pol(Collapse(tris.size()),
                [stInds = proxy<space>({}, stInds), tris = proxy<space>({}, tris), voffset = primHandle.vOffset,
                 sfoffset = primHandle.sfOffset] __device__(int i) mutable {
                    stInds.template tuple<3>("inds", sfoffset + i) =
                        (tris.template pack<3>("inds", i).template reinterpret_bits<int>() + (int)voffset)
                            .template reinterpret_bits<float>();
                });
        }
        auto &edges = primHandle.getSurfEdges();
        pol(Collapse(edges.size()),
            [seInds = proxy<space>({}, seInds), edges = proxy<space>({}, edges), voffset = primHandle.vOffset,
             seoffset = primHandle.seOffset] __device__(int i) mutable {
                seInds.template tuple<2>("inds", seoffset + i) =
                    (edges.template pack<2>("inds", i).template reinterpret_bits<int>() + (int)voffset)
                        .template reinterpret_bits<float>();
            });
        auto &points = primHandle.getSurfVerts();
        pol(Collapse(points.size()),
            [svInds = proxy<space>({}, svInds), points = proxy<space>({}, points), voffset = primHandle.vOffset,
             svoffset = primHandle.svOffset] __device__(int i) mutable {
                svInds("inds", svoffset + i) =
                    reinterpret_bits<float>(reinterpret_bits<int>(points("inds", i)) + (int)voffset);
            });
    }
    // init mass
    pol(Collapse(numDofs), [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable { vtemp("m", i) = 0; });
    for (auto &primHandle : prims) {
        auto density = primHandle.models.density;
        auto &eles = primHandle.getEles();
        pol(Collapse(eles.size()), [vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, eles),
                                    vOffset = primHandle.vOffset, density] __device__(int ei) mutable {
            auto m = eles("rv", ei) * density;
            auto inds = eles.template pack<4>("inds", ei).template reinterpret_bits<int>() + vOffset;
            for (int d = 0; d != 4; ++d)
                atomic_add(exec_cuda, &vtemp("m", inds[d]), m / 4);
        });
    }
    reinitialize(pol, dt);
}

void PBDSystem::reinitialize(zs::CudaExecutionPolicy &pol, T framedt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    this->framedt = framedt;
    this->dt = framedt / solveIterCap; // substep dt
    for (auto &primHandle : prims) {
        auto &verts = primHandle.getVerts();
        // initialize BC info
        // predict pos, initialize augmented lagrangian, constrain weights
        pol(Collapse(verts.size()), [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
                                     voffset = primHandle.vOffset, dt = dt] __device__(int i) mutable {
            auto x = verts.pack<3>("x", i);
            auto v = verts.pack<3>("v", i);
            vtemp.tuple<3>("x", voffset + i) = x;
            vtemp.tuple<3>("xpre", voffset + i) = x;
            vtemp.tuple<3>("v", voffset + i) = v;
        });
    }
}

PBDSystem::PBDSystem(std::vector<ZenoParticles *> zsprims, vec3 extForce, T dt, int numSolveIters, T ec, T vc)
    : extForce{extForce}, solveIterCap{numSolveIters}, edgeCompliance{ec}, volumeCompliance{vc}, prims{}, coOffset{0},
      numDofs{0}, sfOffset{0}, seOffset{0}, svOffset{0}, vtemp{}, temp{}, stInds{}, seInds{}, svInds{}, dt{dt} {
    for (auto primPtr : zsprims) {
        if (primPtr->category == ZenoParticles::category_e::tet) {
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<4>{});
        }
    }
    zeno::log_info("num total obj <verts, surfV, surfE, surfT>: {}, {}, {}, {}\n", coOffset, svOffset, seOffset,
                   sfOffset);
    numDofs = coOffset; // if there are boundaries, then updated
    vtemp = dtiles_t{zsprims[0]->getParticles().get_allocator(), {{"m", 1}, {"x", 3}, {"xpre", 3}, {"v", 3}}, numDofs};

    auto cudaPol = zs::cuda_exec();
    initialize(cudaPol);
}

struct ExtractTetEdges : INode {
    void apply() override {
        using namespace zs;

        auto zstets = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        for (auto tet : zstets) {
            using ivec2 = zs::vec<int, 2>;
            auto comp_v2 = [](const ivec2 &x, const ivec2 &y) {
                return x[0] < y[0] ? 1 : (x[0] == y[0] && x[1] < y[1] ? 1 : 0);
            };
            std::set<ivec2, RM_CVREF_T(comp_v2)> sedges(comp_v2);
            std::vector<ivec2> lines(0);
            auto ist2 = [&sedges, &lines](int i, int j) {
                if (sedges.find(ivec2{i, j}) == sedges.end() && sedges.find(ivec2{j, i}) == sedges.end()) {
                    sedges.insert(ivec2{i, j});
                    lines.push_back(ivec2{i, j});
                }
            };
            const auto &elements = tet->getQuadraturePoints().clone({memsrc_e::host, -1});
            constexpr auto space = execspace_e::host;
            auto eles = proxy<space>({}, elements);
            for (int ei = 0; ei != eles.size(); ++ei) {
                auto inds = eles.template pack<4>("inds", ei).template reinterpret_bits<int>();
                ist2(inds[0], inds[1]);
                ist2(inds[0], inds[2]);
                ist2(inds[0], inds[3]);
                ist2(inds[1], inds[2]);
                ist2(inds[1], inds[3]);
                ist2(inds[2], inds[3]);
            }
            auto &edges = (*tet)[ZenoParticles::s_edgeTag];
            edges = typename ZenoParticles::particles_t{{{"inds", 2}}, lines.size(), memsrc_e::host, -1};
            auto ev = proxy<space>({}, edges);
            for (int i = 0; i != lines.size(); ++i) {
                ev("inds", 0, i) = reinterpret_bits<float>((int)lines[i][0]);
                ev("inds", 1, i) = reinterpret_bits<float>((int)lines[i][1]);
            }
            edges = edges.clone(tet->getParticles().get_allocator());
        }

        set_output("ZSParticles", get_input("ZSParticles"));
    }
};

ZENDEFNODE(ExtractTetEdges, {{
                                 "ZSParticles",
                             },
                             {"ZSParticles"},
                             {},
                             {"PBD"}});

struct MakePBDSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto zstets = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        /// solver parameters
        auto input_cap = get_input2<int>("iter_cap");
        auto dt = get_input2<float>("dt");
        auto ec = get_input2<float>("edge_compliance");
        auto vc = get_input2<float>("edge_compliance");
        auto extForce = get_input<zeno::NumericObject>("ext_force")->get<zeno::vec3f>();

        auto A = std::make_shared<PBDSystem>(zstets, zs::vec<float, 3>{extForce[0], extForce[1], extForce[2]}, dt,
                                             input_cap, ec, vc);

        set_output("ZSPBDSystem", A);
    }
};

ZENDEFNODE(MakePBDSystem, {{
                               "ZSParticles",
                               {"float", "dt", "0.01"},
                               {"vec3f", "ext_force", "0,-9,0"},
                               {"int", "iter_cap", "100"},
                               {"float", "edge_compliance", "0.001"},
                               {"float", "volume_compliance", "0.001"},
                           },
                           {"ZSPBDSystem"},
                           {},
                           {"PBD"}});

} // namespace zeno