#include "UnifiedSolver.cuh"
#include "Utils.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/math/DihedralAngle.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/types/SmallVector.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector>
#include <zeno/utils/log.h>

namespace zeno {

struct UnifiedIPCSystemClothBinding : INode { // usually called once before stepping
    using tiles_t = typename ZenoParticles::particles_t;
#if 1
    // unordered version
    using bvh_t = zs::LBvh<3, int, zs::f32>;
    using bv_t = typename bvh_t::Box;
#else
    using bvh_t = typename UnifiedIPCSystem::bvh_t;
    using bv_t = typename UnifiedIPCSystem::bv_t;
#endif
    template <typename VecT>
    static constexpr float distance(const bv_t &bv, const zs::VecInterface<VecT> &x) {
        using namespace zs;
        const auto &mi = bv._min;
        const auto &ma = bv._max;
        // const auto &[mi, ma] = bv;
        auto center = (mi + ma) / 2;
        auto point = (x - center).abs() - (ma - mi) / 2;
        float max = limits<float>::lowest();
        for (int d = 0; d != 3; ++d) {
            if (point[d] > max)
                max = point[d];
            if (point[d] < 0)
                point[d] = 0;
        }
        return (max < 0.f ? max : 0.f) + point.length();
    }
    void markBoundaryVerts(zs::CudaExecutionPolicy &pol, UnifiedIPCSystem *ipcsys) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto &vtemp = ipcsys->vtemp;
        vtemp.append_channels(pol, std::vector<zs::PropertyTag>{{"on_boundary", 1}});
        auto markIter = vtemp.begin("on_boundary", dim_c<1>, zs::wrapt<i64>{});
        auto markIterEnd = vtemp.end("on_boundary", dim_c<1>, zs::wrapt<i64>{});
        // pol(range(vtemp, "on_boundary", dim_c<1>, zs::wrapt<i64>{}), [] ZS_LAMBDA(auto &mark) mutable { mark = 0; });
        pol(detail::iter_range(markIter, markIterEnd), [] ZS_LAMBDA(auto &mark) mutable { mark = 0; });
        for (auto &primHandle : ipcsys->prims) {
            if (primHandle.isAuxiliary())
                continue;
            auto vOffset = primHandle.vOffset;
            if (primHandle.category == ZenoParticles::curve) {
                auto &eles = primHandle.getEles();
                mark_surface_boundary_verts(pol, eles, wrapv<2>{}, markIter, (size_t)vOffset);
            } else if (primHandle.category == ZenoParticles::surface) {
                auto &eles = primHandle.getEles();
                mark_surface_boundary_verts(pol, eles, wrapv<3>{}, markIter, (size_t)vOffset);
            } else if (primHandle.category == ZenoParticles::tet) {
                auto &surf = primHandle.getSurfTris();
                mark_surface_boundary_verts(pol, surf, wrapv<3>{}, markIter, (size_t)vOffset);
            }
        }
    }
    template <typename VTilesT, typename LsView, typename Bvh>
    std::shared_ptr<tiles_t> bindStrings(zs::CudaExecutionPolicy &cudaPol, VTilesT &vtemp, std::size_t numVerts,
                                         LsView lsv, const Bvh &bvh, float k, float distCap, float rl,
                                         bool boundaryWise) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // assume all verts
        Vector<int> nStrings{vtemp.get_allocator(), 1};
        nStrings.setVal(0);
        tiles_t strings{vtemp.get_allocator(), {{"inds", 2}, {"vol", 1}, {"k", 1}, {"rl", 1}}, numVerts};
        cudaPol(range(numVerts), [vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, strings), lsv, distCap,
                                  bvh = proxy<space>(bvh), cnt = proxy<space>(nStrings), coOffset = numVerts, k, rl,
                                  boundaryWise] ZS_LAMBDA(int i) mutable {
            if (boundaryWise && vtemp.hasProperty("on_boundary"))
                if (vtemp("on_boundary", i, zs::wrapt<i64>{}) == 0) // only operate on verts on boundary
                    return;
            auto x = vtemp.pack(dim_c<3>, "xn", i);
            if (lsv.getSignedDistance(x) < 0) {
                float dist = distCap;
                int j = -1;
                int numNodes = bvh.numNodes();
#if 0
                auto nt = bvh.numLeaves() - 1;
                int node = bvh._root;
                while (node != -1) {
                    for (; node < nt; node = bvh._trunkTopo("lc", node))
                        if (auto d = distance(bvh.getNodeBV(node), x); d > dist)
                            break;
                    // leaf node check
                    if (node >= nt) {
                        auto bouId = bvh._leafTopo("inds", node - nt) + coOffset;
                        auto d = (vtemp.pack(dim_c<3>, "xn", bouId) - x).length();
                        if (d < dist) {
                            dist = d;
                            j = bouId;
                        }
                        node = bvh._leafTopo("esc", node - nt);
                    } else // separate at internal nodes
                        node = bvh._trunkTopo("esc", node);
                }
#else
                int node = 0;
                while (node != -1 && node != numNodes) {
                    int level = bvh._levels[node];
                    for (; level; --level, ++node)
                        if (auto d = distance(bvh.getNodeBV(node), x); d > dist)
                            break;
                    // leaf node check
                    if (level == 0) {
                        auto bouId = bvh._auxIndices[node] + coOffset;
                        auto d = (vtemp.pack(dim_c<3>, "xn", bouId) - x).length();
                        if (d < dist) {
                            dist = d;
                            j = bouId;
                        }
                        node++;
                    } else // separate at internal nodes
                        node = bvh._auxIndices[node];
                }
#endif
                if (j != -1) {
                    auto no = atomic_add(exec_cuda, &cnt[0], 1);
                    eles.tuple(dim_c<2>, "inds", no, int_c) = zs::vec<int, 2>{i, j};
                    eles("vol", no) = 1;
                    eles("k", no) = k;
                    eles("rl", no) = zs::min(dist / 4, rl);
                }
            }
        });
        auto cnt = nStrings.getVal();
        strings.resize(cnt);
        return std::make_shared<tiles_t>(std::move(strings));
    }
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto A = get_input<UnifiedIPCSystem>("ZSUnifiedIPCSystem");
        if (!A->hasBoundary()) {
            set_output("ZSUnifiedIPCSystem", A);
            return;
        }
        const auto &bouVerts = *A->coVerts;
        const auto numBouVerts = bouVerts.size();
        if (numBouVerts == 0) {
            set_output("ZSUnifiedIPCSystem", A);
            return;
        }
        auto &vtemp = A->vtemp;
        const auto numVerts = A->coOffset;

        auto cudaPol = zs::cuda_exec().sync(true);

        auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");
        bool ifHardCons = get_input2<bool>("hard_constraint");
        bool boundaryWise = get_input2<bool>("boundary_wise");

        bvh_t bouBvh;
        Vector<bv_t> bouVertBvs{vtemp.get_allocator(), numBouVerts};
        cudaPol(enumerate(bouVertBvs),
                [vtemp = proxy<space>({}, vtemp), coOffset = numVerts] ZS_LAMBDA(int i, bv_t &bv) {
                    auto p = vtemp.pack(dim_c<3>, "xn", i + coOffset);
                    bv = bv_t{p - limits<float>::epsilon() * 8, p + limits<float>::epsilon() * 8};
                });
        bouBvh.build(cudaPol, bouVertBvs);

        // stiffness
        float k = get_input2<float>("strength"); // pulling stiffness
        if (k == 0)                              // auto setup
            k = A->largestMu() * 100;
        // dist cap
        float dist_cap = get_input2<float>("dist_cap"); // only proximity pairs within this range considered
        if (dist_cap == 0)
            dist_cap = limits<float>::max();
        float rl = get_input2<float>("rest_length"); // rest length cap
        match([&](const auto &ls) {
            using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
            using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
            using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;
            if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
                match([&](const auto &lsPtr) {
                    auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
                    A->pushBoundarySprings(
                        bindStrings(cudaPol, vtemp, numVerts, lsv, bouBvh, k, dist_cap, rl, boundaryWise),
                        ifHardCons ? ZenoParticles::category_e::tracker : ZenoParticles::category_e::curve);
                })(ls._ls);
            } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
                match([&](auto lsv) {
                    A->pushBoundarySprings(bindStrings(cudaPol, vtemp, numVerts, SdfVelFieldView{lsv}, bouBvh, k,
                                                       dist_cap, rl, boundaryWise),
                                           ifHardCons ? ZenoParticles::category_e::tracker
                                                      : ZenoParticles::category_e::curve);
                })(ls.template getView<execspace_e::cuda>());
            } else if constexpr (is_same_v<RM_CVREF_T(ls), const_transition_ls_t>) {
                match([&](auto fieldPair) {
                    auto &fvSrc = zs::get<0>(fieldPair);
                    auto &fvDst = zs::get<1>(fieldPair);
                    A->pushBoundarySprings(
                        bindStrings(cudaPol, vtemp, numVerts,
                                    TransitionLevelSetView{SdfVelFieldView{fvSrc}, SdfVelFieldView{fvDst}, ls._stepDt,
                                                           ls._alpha},
                                    bouBvh, k, dist_cap, rl, boundaryWise),
                        ifHardCons ? ZenoParticles::category_e::tracker : ZenoParticles::category_e::curve);
                })(ls.template getView<zs::execspace_e::cuda>());
            }
        })(zsls->getLevelSet());

        set_output("ZSUnifiedIPCSystem", A);
    }
};

ZENDEFNODE(UnifiedIPCSystemClothBinding, {{
                                              "ZSUnifiedIPCSystem",
                                              "ZSLevelSet",
                                              {"bool", "boundary_wise", "1"},
                                              {"bool", "hard_constraint", "1"},
                                              {"float", "dist_cap", "0"},
                                              {"float", "rest_length", "0.1"},
                                              {"float", "strength", "0"},
                                          },
                                          {"ZSUnifiedIPCSystem"},
                                          {},
                                          {"FEM"}});

struct UnifiedIPCSystemForceField : INode {
    template <typename VelSplsViewT>
    void computeForce(zs::CudaExecutionPolicy &cudaPol, float windDragCoeff, float windDensity, int vOffset,
                      VelSplsViewT velLs, typename UnifiedIPCSystem::dtiles_t &vtemp,
                      const typename UnifiedIPCSystem::tiles_t &eles) {
        using namespace zs;
        cudaPol(range(eles.size()), [windDragCoeff, windDensity, velLs, vtemp = proxy<execspace_e::cuda>({}, vtemp),
                                     eles = proxy<execspace_e::cuda>({}, eles), vOffset] ZS_LAMBDA(size_t ei) mutable {
            auto inds = eles.pack<3>("inds", ei, int_c) + vOffset;
            auto p0 = vtemp.pack(dim_c<3>, "xn", inds[0]);
            auto p1 = vtemp.pack(dim_c<3>, "xn", inds[1]);
            auto p2 = vtemp.pack(dim_c<3>, "xn", inds[2]);
            auto cp = (p1 - p0).cross(p2 - p0);
            auto area = cp.length();
            auto n = cp / area;
            area *= 0.5;

            auto pos = (p0 + p1 + p2) / 3; // get center to sample velocity
            auto windVel = velLs.getMaterialVelocity(pos);

            auto vel = (vtemp.pack(dim_c<3>, "vn", inds[0]) + vtemp.pack(dim_c<3>, "vn", inds[1]) +
                        vtemp.pack(dim_c<3>, "vn", inds[2])) /
                       3;
            auto vrel = windVel - vel;
            auto vnSignedLength = n.dot(vrel);
            auto vn = n * vnSignedLength;
            auto vt = vrel - vn; // tangent
            auto windForce = windDensity * area * zs::abs(vnSignedLength) * vn + windDragCoeff * area * vt;
            auto f = windForce;
            for (int i = 0; i != 3; ++i)
                for (int d = 0; d != 3; ++d) {
                    atomic_add(exec_cuda, &vtemp("extf", d, inds[i]), f[d] / 3);
                }
        });
    }
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto A = get_input<UnifiedIPCSystem>("ZSUnifiedIPCSystem");
        auto &vtemp = A->vtemp;
        const auto numVerts = A->coOffset;
        auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");

        auto cudaPol = zs::cuda_exec();
        vtemp.append_channels(cudaPol, {{"extf", 3}});
        cudaPol(range(numVerts), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.template tuple<3>("extf", i) = zs::vec<double, 3>::zeros();
        });

        auto windDrag = get_input2<float>("wind_drag");
        auto windDensity = get_input2<float>("wind_density");

        for (auto &primHandle : A->prims) {
            if (primHandle.category != ZenoParticles::category_e::surface)
                continue;
            const auto &eles = primHandle.getEles();
            match([&](const auto &ls) {
                using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
                using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
                using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;
                if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
                    match([&](const auto &lsPtr) {
                        auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
                        computeForce(cudaPol, windDrag, windDensity, primHandle.vOffset, lsv, vtemp, eles);
                    })(ls._ls);
                } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
                    match([&](auto lsv) {
                        computeForce(cudaPol, windDrag, windDensity, primHandle.vOffset, SdfVelFieldView{lsv}, vtemp,
                                     eles);
                    })(ls.template getView<execspace_e::cuda>());
                } else if constexpr (is_same_v<RM_CVREF_T(ls), const_transition_ls_t>) {
                    match([&](auto fieldPair) {
                        auto &fvSrc = zs::get<0>(fieldPair);
                        auto &fvDst = zs::get<1>(fieldPair);
                        computeForce(cudaPol, windDrag, windDensity, primHandle.vOffset,
                                     TransitionLevelSetView{SdfVelFieldView{fvSrc}, SdfVelFieldView{fvDst}, ls._stepDt,
                                                            ls._alpha},
                                     vtemp, eles);
                    })(ls.template getView<zs::execspace_e::cuda>());
                }
            })(zsls->getLevelSet());
        }

        set_output("ZSUnifiedIPCSystem", A);
    }
};

ZENDEFNODE(UnifiedIPCSystemForceField,
           {
               {"ZSUnifiedIPCSystem", "ZSLevelSet", {"float", "wind_drag", "0"}, {"float", "wind_density", "1"}},
               {"ZSUnifiedIPCSystem"},
               {},
               {"FEM"},
           });

struct UnifiedIPCSystemMarkExclusion : INode {
    template <typename SdfViewT>
    void markExclusion(zs::CudaExecutionPolicy &cudaPol, bool includeObject, bool includeBoundary, int coOffset,
                       int numDofs, SdfViewT sdfv, typename UnifiedIPCSystem::dtiles_t &vtemp,
                       zs::Vector<zs::u8> &marks) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        cudaPol(range(numDofs), [sdfv, vtemp = proxy<space>({}, vtemp), marks = proxy<space>(marks), includeObject,
                                 includeBoundary, coOffset] ZS_LAMBDA(int vi) mutable {
            auto pos = vtemp.pack(dim_c<3>, "xn", vi);
            auto sdf = sdfv.getSignedDistance(pos);
            if (sdf < 0) {
                if (includeObject && vi < coOffset)
                    marks[vi] = 1;
                if (includeBoundary && vi >= coOffset)
                    marks[vi] = 1;
            }
        });
    }
    void apply() override {
        using namespace zs;

        auto A = get_input<UnifiedIPCSystem>("ZSUnifiedIPCSystem");
        auto &vtemp = A->vtemp;
        auto &exclDofs = A->exclDofs;
        if (get_input2<bool>("clear_mark"))
            exclDofs.reset(0);
        const auto numVerts = A->coOffset;
        const auto numDofs = A->numDofs;
        auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");

        auto cudaPol = zs::cuda_exec();

        auto includeObject = get_input2<bool>("include_object");
        auto includeBoundary = get_input2<bool>("include_boundary");

        match([&](const auto &ls) {
            using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
            using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
            using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;
            if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
                match([&](const auto &lsPtr) {
                    auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
                    markExclusion(cudaPol, includeObject, includeBoundary, numVerts, numDofs, lsv, vtemp, exclDofs);
                })(ls._ls);
            } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
                match([&](auto lsv) {
                    markExclusion(cudaPol, includeObject, includeBoundary, numVerts, numDofs, SdfVelFieldView{lsv},
                                  vtemp, exclDofs);
                })(ls.template getView<execspace_e::cuda>());
            } else if constexpr (is_same_v<RM_CVREF_T(ls), const_transition_ls_t>) {
                match([&](auto fieldPair) {
                    auto &fvSrc = zs::get<0>(fieldPair);
                    auto &fvDst = zs::get<1>(fieldPair);
                    markExclusion(
                        cudaPol, includeObject, includeBoundary, numVerts, numDofs,
                        TransitionLevelSetView{SdfVelFieldView{fvSrc}, SdfVelFieldView{fvDst}, ls._stepDt, ls._alpha},
                        vtemp, exclDofs);
                })(ls.template getView<zs::execspace_e::cuda>());
            }
        })(zsls->getLevelSet());

        set_output("ZSUnifiedIPCSystem", A);
    }
};

ZENDEFNODE(UnifiedIPCSystemMarkExclusion, {
                                              {"ZSUnifiedIPCSystem",
                                               "ZSLevelSet",
                                               {"bool", "clear_mark", "false"},
                                               {"bool", "include_object", "false"},
                                               {"bool", "include_boundary", "true"}},
                                              {"ZSUnifiedIPCSystem"},
                                              {},
                                              {"FEM"},
                                          });

} // namespace zeno