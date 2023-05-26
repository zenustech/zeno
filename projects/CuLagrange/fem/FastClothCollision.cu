#include "FastCloth.cuh"
#include "TopoUtils.hpp"
#include "collision_energy/vertex_face_sqrt_collision.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

void FastClothSystem::initialStepping(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    /// @brief Xinit
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), D = D] ZS_LAMBDA(int i) mutable {
        auto xk = vtemp.pack(dim_c<3>, "xn", i);
        auto ykp1 = vtemp.pack(dim_c<3>, "yn", i);
        auto diff = ykp1 - xk;
        T coeff = 1;
        if (auto len2 = diff.l2NormSqr(); len2 > limits<T>::epsilon() * 10)
            coeff = zs::min(D / zs::sqrt(len2), (T)1);
        vtemp.tuple(dim_c<3>, "xinit", i) = xk + coeff * diff;
    });
}

void FastClothSystem::findConstraints(zs::CudaExecutionPolicy &pol, T dHat, const zs::SmallString &tag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    zs::CppTimer timer;
    if (enableContact) {
        nPP.setVal(0);
        if (enableContactSelf) {
            bvs.resize(svInds.size());
            retrieve_bounding_volumes(pol, vtemp, tag, svInds, zs::wrapv<1>{}, 0, bvs);
            // auto pBvs = retrieve_bounding_volumes(pol, vtemp, tag, svInds, zs::wrapv<1>{}, 0);

            /// bvh
            if constexpr (s_enableProfile)
                timer.tick();

            svBvh.refit(pol, bvs);

            if constexpr (s_enableProfile) {
                timer.tock();
                auxCnt[0]++;
                auxTime[0] += timer.elapsed();
            }

            /// sh
            if constexpr (s_testSh) {
                if constexpr (s_enableProfile)
                    timer.tick();

                svSh.build(pol, L, bvs);

                if constexpr (s_enableProfile) {
                    timer.tock();
                    auxCnt[2]++;
                    auxTime[2] += timer.elapsed();
                }
            }

            /// @note all cloth edge lower-bound constraints inheritly included
            findCollisionConstraints(pol, dHat, false);
        }
        if (hasBoundary()) {
            bvs.resize(coPoints->size());
            retrieve_bounding_volumes(pol, vtemp, tag, *coPoints, zs::wrapv<1>{}, coOffset, bvs);
            // auto pBvs = retrieve_bounding_volumes(pol, vtemp, tag, *coPoints, zs::wrapv<1>{}, coOffset);

            /// bvh
            if constexpr (s_enableProfile)
                timer.tick();

            bouSvBvh.refit(pol, bvs);

            if constexpr (s_enableProfile) {
                timer.tock();
                auxCnt[0]++;
                auxTime[0] += timer.elapsed();
            }

            /// sh
            if constexpr (s_testSh) {
                if constexpr (s_enableProfile)
                    timer.tick();

                bouSvSh.build(pol, L, bvs);

                if constexpr (s_enableProfile) {
                    timer.tock();
                    auxCnt[2]++;
                    auxTime[2] += timer.elapsed();
                }
            }
            findCollisionConstraints(pol, dHat, true);
        }
    }
    /// @note check upper-bound constraints for cloth edges
    nE.setVal(0);
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        auto &ses = primHandle.getSurfEdges();
        pol(Collapse{ses.size()},
            [ses = view<space>({}, ses), vtemp = view<space>({}, vtemp), E = view<space>(E), nE = view<space>(nE),
             threshold = L * L - epsSlack, vOffset = primHandle.vOffset, tag] __device__(int sei) mutable {
                const auto vij = ses.pack(dim_c<2>, "inds", sei, int_c) + vOffset;
                const auto &vi = vij[0];
                const auto &vj = vij[1];
                auto pi = vtemp.pack(dim_c<3>, tag, vi);
                auto pj = vtemp.pack(dim_c<3>, tag, vj);
                if (auto d2 = dist2_pp(pi, pj); d2 >= threshold) {
                    auto no = atomic_add(exec_cuda, &nE[0], 1);
                    E[no] = vij;
                }
            });
    }
    std::tie(npp, ne) = getConstraintCnt();
}

void FastClothSystem::lightCD(zs::CudaExecutionPolicy &pol, T dHat, const zs::SmallString &tag) {
    using namespace zs;

    zs::CppTimer timer;
    if (enableContact) {
        ncPP.setVal(0);
        if (enableContactSelf) {
            bvs.resize(svInds.size());
            retrieve_bounding_volumes(pol, vtemp, tag, svInds, zs::wrapv<1>{}, 0, bvs);
            // auto pBvs = retrieve_bounding_volumes(pol, vtemp, tag, svInds, zs::wrapv<1>{}, 0);

            /// bvh
            if constexpr (s_enableProfile)
                timer.tick();

            svBvh.refit(pol, bvs);

            if constexpr (s_enableProfile) {
                timer.tock();
                auxCnt[0]++;
                auxTime[0] += timer.elapsed();
            }

            /// sh
            if constexpr (s_testSh) {
                if constexpr (s_enableProfile)
                    timer.tick();

                svSh.build(pol, L, bvs);

                if constexpr (s_enableProfile) {
                    timer.tock();
                    auxCnt[2]++;
                    auxTime[2] += timer.elapsed();
                }
            }

            /// @note all cloth edge lower-bound constraints inheritly included
            lightFindCollisionConstraints(pol, dHat, false);
        }
        if (hasBoundary()) {
            bvs.resize(coPoints->size());
            retrieve_bounding_volumes(pol, vtemp, tag, *coPoints, zs::wrapv<1>{}, coOffset, bvs);
            // auto pBvs = retrieve_bounding_volumes(pol, vtemp, tag, *coPoints, zs::wrapv<1>{}, coOffset);

            /// bvh
            if constexpr (s_enableProfile)
                timer.tick();

            bouSvBvh.refit(pol, bvs);

            if constexpr (s_enableProfile) {
                timer.tock();
                auxCnt[0]++;
                auxTime[0] += timer.elapsed();
            }

            /// sh
            if constexpr (s_testSh) {
                if constexpr (s_enableProfile)
                    timer.tick();

                bouSvSh.build(pol, L, bvs);

                if constexpr (s_enableProfile) {
                    timer.tock();
                    auxCnt[2]++;
                    auxTime[2] += timer.elapsed();
                }
            }
            lightFindCollisionConstraints(pol, dHat, true);
        }
    }
    ncpp = ncPP.getVal();
}

void FastClothSystem::lightFilterConstraints(zs::CudaExecutionPolicy &pol, T dHat, const zs::SmallString &tag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    nPP.setVal(0);
    pol(range(ncpp), [vtemp = view<space>({}, vtemp), nPP = view<space>(nPP), cPP = view<space>(cPP),
                      PP = view<space>(PP), dHat2 = dHat * dHat, tag] __device__(int i) mutable {
        auto pp = cPP[i];
        auto v0 = vtemp.pack(dim_c<3>, tag, pp[0]);
        auto v1 = vtemp.pack(dim_c<3>, tag, pp[1]);
        auto dist = (v0 - v1).l2NormSqr();
        if (dist < dHat2) {
            int no = atomic_add(exec_cuda, &nPP[0], 1);
            PP[no] = pp;
        }
    });
    npp = nPP.getVal();
    /// @note check upper-bound constraints for cloth edges
    nE.setVal(0);
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        auto &ses = primHandle.getSurfEdges();
        pol(Collapse{ses.size()},
            [ses = view<space>({}, ses), vtemp = view<space>({}, vtemp), E = view<space>(E), nE = view<space>(nE),
             threshold = L * L - epsSlack, vOffset = primHandle.vOffset, tag] __device__(int sei) mutable {
                const auto vij = ses.pack(dim_c<2>, "inds", sei, int_c) + vOffset;
                const auto &vi = vij[0];
                const auto &vj = vij[1];
                auto pi = vtemp.pack(dim_c<3>, tag, vi);
                auto pj = vtemp.pack(dim_c<3>, tag, vj);
                if (auto d2 = dist2_pp(pi, pj); d2 >= threshold) {
                    auto no = atomic_add(exec_cuda, &nE[0], 1);
                    E[no] = vij;
                }
            });
    }
}

#define PROFILE_CD 0

void FastClothSystem::findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    zs::CppTimer timer;
    pol.profile(PROFILE_CD);

#if !s_testSh
    /// pt
    if constexpr (s_enableProfile)
        timer.tick();

    const auto &svbvh = withBoundary ? bouSvBvh : svBvh;
    pol(Collapse{svInds.size()},
        [svInds = view<space>({}, svInds), eles = view<space>({}, withBoundary ? *coPoints : svInds),
#if !s_debugRemoveHashTable
         eTab = view<space>(eTab),
#endif
         vtemp = view<space>({}, vtemp), bvh = view<space>(svbvh), PP = view<space>(PP), nPP = view<space>(nPP),
         dHat2 = dHat * dHat, thickness = dHat, voffset = withBoundary ? coOffset : 0,
         withBoundary] __device__(int i) mutable {
            auto vi = svInds("inds", i, int_c);
            auto pi = vtemp.pack(dim_c<3>, "xn", vi);
            auto bv = bv_t{get_bounding_box(pi - thickness, pi + thickness)};
#if 1
            auto f = [&](int svI) {
                auto vj = eles("inds", svI, int_c) + voffset;
                if ((!withBoundary) && (vi >= vj))
                    return;
                auto pj = vtemp.pack(dim_c<3>, "xn", vj);
            // skip edges for point-point lower-bound constraints
#if !s_debugRemoveHashTable
                if (!withBoundary && (eTab.query(ivec2{vi, vj}) >= 0 || eTab.query(ivec2{vj, vi}) >= 0))
                    return;
#endif
                if (auto d2 = dist2_pp(pi, pj); d2 <= dHat2) {
                    auto no = atomic_add(exec_cuda, &nPP[0], 1);
                    PP[no] = pair_t{vi, vj};
                }
            };
            bvh.iter_neighbors(bv, f);
#else
            {
                const auto &lbvh = bvh;
                using bvh_t = RM_CVREF_T(lbvh);
                using index_t = typename bvh_t::index_t;
                index_t node = 0;
                while (node != -1 && node != lbvh._numNodes) {
                    index_t level = lbvh._levels[node];
                    // level and node are always in sync
                    for (; level; --level, ++node)
                        if (!overlaps(lbvh.getNodeBV(node), bv))
                            break;
                    // leaf node check
                    if (level == 0) {
                        if (overlaps(lbvh.getNodeBV(node), bv)) {
                            int svI = lbvh._auxIndices[node];
                            {
                                auto vj = eles("inds", svI, int_c) + voffset;
                                if ((!withBoundary) && (vi >= vj))
                                    goto NEXT;
                                auto pj = vtemp.pack(dim_c<3>, "xn", vj);
                            // skip edges for point-point lower-bound constraints
#if !s_debugRemoveHashTable
                                if (!withBoundary && (eTab.query(ivec2{vi, vj}) >= 0 || eTab.query(ivec2{vj, vi}) >= 0))
                                    goto NEXT;
#endif
                                if (auto d2 = dist2_pp(pi, pj); d2 <= dHat2) {
                                    auto no = atomic_add(exec_cuda, &nPP[0], 1);
                                    PP[no] = pair_t{vi, vj};
                                }
                            }
                        }
                        NEXT:
                        node++;
                    } else // separate at internal nodes
                        node = lbvh._auxIndices[node];
                }
            }
#endif
        });

    if constexpr (s_enableProfile) {
        timer.tock();
        auxCnt[1]++;
        auxTime[1] += timer.elapsed();
    }
#endif

#if s_testSh
    /// sh
    if constexpr (s_enableProfile)
        timer.tick();
    const auto &sh = withBoundary ? bouSvSh : svSh;
    pol(Collapse{svInds.size()},
        [svInds = view<space>({}, svInds), eles = view<space>({}, withBoundary ? *coPoints : svInds),
         eTab = view<space>(eTab), vtemp = view<space>({}, vtemp), sh = view<space>(sh), PP = view<space>(PP),
         nPP = view<space>(nPP), dHat2 = dHat * dHat, thickness = dHat, voffset = withBoundary ? coOffset : 0,
         withBoundary] __device__(int i) mutable {
            auto vi = svInds("inds", i, int_c);
            auto pi = vtemp.pack(dim_c<3>, "xn", vi);
            auto bv = bv_t{get_bounding_box(pi - thickness, pi + thickness)};
            auto f = [&](int svI) {
                auto vj = eles("inds", svI, int_c) + voffset;
                if ((!withBoundary) && (vi >= vj))
                    return;
                auto pj = vtemp.pack(dim_c<3>, "xn", vj);
                // skip edges for point-point lower-bound constraints
                if (!withBoundary && (eTab.query(ivec2{vi, vj}) >= 0 || eTab.query(ivec2{vj, vi}) >= 0))
                    return;
                if (auto d2 = dist2_pp(pi, pj); d2 <= dHat2) {
                    auto no = atomic_add(exec_cuda, &nPP[0], 1);
                    PP[no] = pair_t{vi, vj};
                }
            };
            sh.iter_neighbors(bv, f);
        });
    if constexpr (s_enableProfile) {
        timer.tock();
        auxTime[3] += timer.elapsed();
    }
#endif
    pol.profile(false);
}

void FastClothSystem::lightFindCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    zs::CppTimer timer;
    pol.profile(PROFILE_CD);

#if !s_testSh
    /// pt
    if constexpr (s_enableProfile)
        timer.tick();

    const auto &svbvh = withBoundary ? bouSvBvh : svBvh;
    pol(Collapse{svInds.size()},
        [svInds = view<space>({}, svInds), eles = view<space>({}, withBoundary ? *coPoints : svInds),
#if !s_debugRemoveHashTable
         eTab = view<space>(eTab),
#endif
         vtemp = view<space>({}, vtemp), bvh = view<space>(svbvh), PP = view<space>(cPP), nPP = view<space>(ncPP),
         dHat2 = dHat * dHat, thickness = dHat, voffset = withBoundary ? coOffset : 0,
         withBoundary] __device__(int i) mutable {
            auto vi = svInds("inds", i, int_c);
            auto pi = vtemp.pack(dim_c<3>, "xn", vi);
            auto bv = bv_t{get_bounding_box(pi - thickness, pi + thickness)};
            auto f = [&](int svI) {
                // if (exclTris[stI]) return;
                auto vj = eles("inds", svI, int_c) + voffset;
                if (!withBoundary && vi >= vj)
                    return;
                auto pj = vtemp.pack(dim_c<3>, "xn", vj);
            // edge or not
            // TODO: use query
#if !s_debugRemoveHashTable
                if (eTab.query(ivec2{vi, vj}) >= 0 || eTab.query(ivec2{vj, vi}) >= 0)
                    return;
#endif
                if (auto d2 = dist2_pp(pi, pj); d2 <= dHat2) {
                    auto no = atomic_add(exec_cuda, &nPP[0], 1);
                    PP[no] = pair_t{vi, vj};
                }
            };
            bvh.iter_neighbors(bv, f);
        });

    if constexpr (s_enableProfile) {
        timer.tock();
        auxCnt[1]++;
        auxTime[1] += timer.elapsed();
    }
#endif
    pol.profile(false);
}

bool FastClothSystem::collisionStep(zs::CudaExecutionPolicy &pol, bool enableHardPhase) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    zs::CppTimer timer;

    std::tie(npp, ne) = getConstraintCnt();
#if !s_silentMode
    fmt::print("collision stepping [pp, edge constraints]: {}, {}\n", npp, ne);
#endif
    if (!enableHardPhase) {
        ///
        /// @brief soft phase for constraints
        ///
        pol(range(numDofs), [vtemp = view<space>({}, vtemp)] __device__(int i) mutable {
            auto xinit = vtemp.pack(dim_c<3>, "xinit", i);
#pragma unroll 3
            for (int d = 0; d < 3; ++d) {
                vtemp("xn", d, i) = xinit(d); // soft phase optimization starts from xinit
            }
        });
        pol.sync(false);
        for (int l = 0; l != ISoft; ++l) {
            timer.tick();
            softPhase(pol);
            timer.tock();
            collisionCnt[5]++;
            collisionTime[5] += timer.elapsed();
        }
        pol.sync(true);
        ///
        /// @brief check whether constraints satisfied
        ///
        if (constraintSatisfied(pol)) {
#if !s_silentMode
            fmt::print(fg(fmt::color::yellow), "\tsoft phase finished successfully!\n");
#endif
            return true;
        }
#if !s_silentMode
        fmt::print(fg(fmt::color::red), "\tsoft phase failed!\n");
#endif
        return false;
    }

    ///
    /// @brief hard phase for constraints
    ///
#if !s_silentMode
    fmt::print(fg(fmt::color::light_golden_rod_yellow), "entering hard phase.\n");
#endif
    /// @note start from collision-free state x^k
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xk", i);
    });
    auto E0 = constraintEnergy(pol);
    for (int l = 0; l != IHard; ++l) {
        /// @note "xk" will be used for backtracking in hardphase
        E0 = hardPhase(pol, E0);
    }

    return constraintSatisfied(pol, false);
}
void FastClothSystem::softPhase(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    T descentStepsize = 0.2f;
    /// @note shape matching
    pol(range(coOffset), [vtemp = view<space>({}, vtemp)] __device__(int i) mutable {
        auto xinit = vtemp.pack(dim_c<3>, "xinit", i);
        auto xn = vtemp.pack(dim_c<3>, "xn", i);
#pragma unroll 3
        for (int d = 0; d < 3; ++d) {
            vtemp("dir", d, i) = 2.0f * (xinit(d) - xn(d)); // minus grad of ||x-xinit||^2
        }
    });
    /// @note constraints
    pol(range(npp), [vtemp = view<space>({}, vtemp), PP = view<space>(PP), rho = rho,
                     dHat2 = dHat * dHat] __device__(int i) mutable {
        auto pp = PP[i];
        auto x0 = vtemp.pack(dim_c<3>, "xn", pp[0]);
        auto x1 = vtemp.pack(dim_c<3>, "xn", pp[1]);
        // ||v0 - v1||^2 >= (B + Bt)^2 + epsSlack
        // c(x) = ||v0 - v1||^2 - (B + Bt)^2
        if ((x0 - x1).l2NormSqr() >= dHat2)
            return;
        auto grad0 = -rho * (T)2.0 * (x0 - x1);
#pragma unroll 3
        for (int d = 0; d < 3; d++) {
            atomic_add(exec_cuda, &vtemp("dir", d, pp[0]), -grad0(d));
            atomic_add(exec_cuda, &vtemp("dir", d, pp[1]), grad0(d));
        }
    });

    pol(range(ne), [vtemp = view<space>({}, vtemp), E = view<space>(E), rho = rho,
                    maxLen2 = L * L - epsSlack] __device__(int i) mutable {
        auto e = E[i];
        auto x0 = vtemp.pack(dim_c<3>, "xn", e[0]);
        auto x1 = vtemp.pack(dim_c<3>, "xn", e[1]);
        // ||v0 - v1||^2 <= L^2 - epsSlack
        // i.e. L^2 - ||v0 - v1||^2 >= epsSlack
        // c(x) = L^2 - ||v0 - v1||^2
        if ((x0 - x1).l2NormSqr() <= maxLen2)
            return;
        auto grad0 = rho * (T)2.0 * (x0 - x1);
#pragma unroll 3
        for (int d = 0; d < 3; d++) {
            atomic_add(exec_cuda, &vtemp("dir", d, e[0]), -grad0(d));
            atomic_add(exec_cuda, &vtemp("dir", d, e[1]), grad0(d));
        }
    });
    pol(range(coOffset), [vtemp = view<space>({}, vtemp), descentStepsize] __device__(int i) mutable {
        auto dir = vtemp.pack(dim_c<3>, "dir", i);
#pragma unroll 3
        for (int d = 0; d < 3; ++d) {
            atomic_add(exec_cuda, &vtemp("xn", d, i), descentStepsize * dir(d));
        }
    });
}
typename FastClothSystem::T FastClothSystem::hardPhase(zs::CudaExecutionPolicy &pol, typename FastClothSystem::T E0) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    /// @note shape matching (reset included)
    pol(range(numDofs), [vtemp = view<space>({}, vtemp), coOffset = coOffset] __device__(int i) mutable {
        auto xinit = vtemp.pack(dim_c<3>, "xinit", i);
        auto xn = vtemp.pack(dim_c<3>, "xn", i);
#pragma unroll 3
        for (int d = 0; d < 3; ++d)
            vtemp("dir", d, i) = 2.0f * (xinit(d) - xn(d));
    });
    /// @note constraints
    pol(range(npp),
        [vtemp = view<space>({}, vtemp), PP = view<space>(PP), mu = mu, Btot2 = (B + Btight) * (B + Btight),
         eps = epsSlack, dHat2 = dHat * dHat, a2 = a2, a3 = a3, coOffset = coOffset] __device__(int i) mutable {
            auto pp = PP[i];
            auto x0 = vtemp.pack(dim_c<3>, "xn", pp[0]);
            auto x1 = vtemp.pack(dim_c<3>, "xn", pp[1]);
            if ((x0 - x1).l2NormSqr() >= dHat2)
                return;
#if 0
        zs::vec<T, 3> vs[2] = {x0, x1};
        const auto &a = vs[0];
        const auto &b = vs[1];
        const auto t2 = a[0] * 2;
        const auto t3 = a[1] * 2;
        const auto t4 = a[2] * 2;
        const auto t5 = b[0] * 2;
        const auto t6 = b[1] * 2;
        const auto t7 = b[2] * 2;

        auto t8 = -Btot2;
        auto t9 = -b[0];
        auto t11 = -b[1];
        auto t13 = -b[2];
        auto t15 = 1 / eps;
        auto t10 = -t5;
        auto t12 = -t6;
        auto t14 = -t7;
        auto t16 = t15 * t15;
        auto t17 = a[0] + t9;
        auto t18 = a[1] + t11;
        auto t19 = a[2] + t13;
        auto t20 = t2 + t10;
        auto t21 = t3 + t12;
        auto t22 = t4 + t14;
        auto t23 = t17 * t17;
        auto t24 = t18 * t18;
        auto t25 = t19 * t10;
        auto t26 = t8 + t23 + t24 + t25;
        auto t27 = t26 * t26;
        auto t28 = t26 * t26 * t26;
        auto t32 = t15 * t20 * t26 * 2;
        auto t33 = t15 * t21 * t26 * 2;
        auto t34 = t15 * t22 * t26 * 2;
        auto t29 = t15 * t27;
        auto t30 = t16 * t28;
        auto t35 = t16 * t20 * t27 * 3;
        auto t36 = t16 * t21 * t27 * 3;
        auto t37 = t16 * t22 * t27 * 3;
        auto t31 = -t30;
        auto t38 = -t35;
        auto t39 = -t36;
        auto t40 = -t37;
        auto t41 = t20 + t32 + t38;
        auto t42 = t21 + t33 + t39;
        auto t43 = t22 + t34 + t40;
        auto t44 = t26 + t29 + t31;
        auto t45 = 1 / t44;
        auto t46 = mu * t41 * t45;
        auto t47 = mu * t42 * t45;
        auto t48 = mu * t43 * t45;
        auto grad = zs::vec<T, 6>{-t46, -t47, -t48, t46, t47, t48};
#pragma unroll 3
        for (int d = 0; d < 3; ++d) {
            atomic_add(exec_cuda, &vtemp("dir", d, pp[0]), grad(d));
            atomic_add(exec_cuda, &vtemp("dir", d, pp[1]), grad(3 + d));
        }
#else 
        auto cons = (x0 - x1).l2NormSqr() - Btot2; 
        if (cons >= eps)
            return; 
        auto f = cons * (cons * (cons * a3 + a2) + 1.0f); 
        auto grad = mu * (cons * (cons * 3.0f * a3 + 2.0f * a2) + 1.0f) / f * 2.0f * (x0 - x1); // negative grad for x0
#pragma unroll 3
        for (int d = 0; d < 3; ++d) {
            if (pp[0] < coOffset)
                atomic_add(exec_cuda, &vtemp("dir", d, pp[0]), grad(d));
            if (pp[1] < coOffset)
                atomic_add(exec_cuda, &vtemp("dir", d, pp[1]), -grad(d));
        }
#endif
        });
    pol(range(ne), [vtemp = view<space>({}, vtemp), E = view<space>(E), mu = mu, L2 = L * L, eps = epsSlack,
                    maxLen2 = L * L - epsSlack, a2 = a2, a3 = a3, coOffset = coOffset] __device__(int i) mutable {
        auto e = E[i];
        auto x0 = vtemp.pack(dim_c<3>, "xn", e[0]);
        auto x1 = vtemp.pack(dim_c<3>, "xn", e[1]);
        if ((x0 - x1).l2NormSqr() <= maxLen2)
            return;
#if 0
        zs::vec<T, 3> vs[2] = {x0, x1};
        const auto &a = vs[0];
        const auto &b = vs[1];
        const auto t2 = a[0] * 2;
        const auto t3 = a[1] * 2;
        const auto t4 = a[2] * 2;
        const auto t5 = b[0] * 2;
        const auto t6 = b[1] * 2;
        const auto t7 = b[2] * 2;
        auto t8 = -L2;
        auto t12 = -b[0];
        auto t14 = -b[1];
        auto t16 = -b[2];
        auto t18 = 1 / eps;
        auto t9 = -t2;
        auto t10 = -t3;
        auto t11 = -t4;
        auto t13 = -t5;
        auto t15 = -t6;
        auto t17 = -t7;
        auto t19 = t18 * t18;
        auto t20 = a[0] + t12;
        auto t21 = a[1] + t14;
        auto t22 = a[2] + t16;
        auto t23 = t2 + t13;
        auto t24 = t3 + t15;
        auto t25 = t4 + t17;
        auto t26 = t20 * t20;
        auto t27 = t21 * t21;
        auto t28 = t22 * t22;
        auto t29 = -t26;
        auto t30 = -t27;
        auto t31 = -t28;
        auto t32 = t8 + t26 + t27 + t28;
        auto t33 = t32 * t32;
        auto t34 = t32 * t32 * t32;
        auto t37 = t18 * t23 * t32 * 2;
        auto t38 = t18 * t24 * t32 * 2;
        auto t39 = t18 * t25 * t32 * 2;
        auto t35 = t18 * t33;
        auto t36 = t19 * t34;
        auto t40 = t19 * t23 * t33 * 3;
        auto t41 = t19 * t24 * t33 * 3;
        auto t42 = t19 * t25 * t33 * 3;
        auto t43 = t5 + t9 + t37 + t40;
        auto t44 = t6 + t10 + t38 + t41;
        auto t45 = t7 + t11 + t39 + t42;
        auto t46 = L2 + t29 + t30 + t31 + t35 + t36;
        auto t47 = 1 / t46;
        auto t48 = mu * t43 * t47;
        auto t49 = mu * t44 * t47;
        auto t50 = mu * t45 * t47;
        auto grad = zs::vec<T, 6>{-t48, -t49, -t50, t48, t49, t50};
#pragma unroll 3
        for (int d = 0; d < 3; ++d) {
            atomic_add(exec_cuda, &vtemp("dir", d, e[0]), -grad(d));
            atomic_add(exec_cuda, &vtemp("dir", d, e[1]), -grad(3 + d));
        }
#else 
        auto cons = L2 - (x0 - x1).l2NormSqr(); 
        if (cons >= eps)
            return; 
        auto f = cons * (cons * (cons * a3 + a2) + 1.0f); 
        auto grad = - mu * (cons * (cons * 3.0f * a3 + 2.0f * a2) + 1.0f) / f * 2.0f * (x0 - x1); // negative grad for x0
#pragma unroll 3
        for (int d = 0; d < 3; ++d) {
            if (e[0] < coOffset)
                atomic_add(exec_cuda, &vtemp("dir", d, e[0]), grad(d));
            if (e[1] < coOffset)
                atomic_add(exec_cuda, &vtemp("dir", d, e[1]), -grad(d));
        }
#endif
    });
    /// @brief compute appropriate step size that does not violates constraints
    auto alpha = (T)0.1;

    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, "xn0", i) = vtemp.pack(dim_c<3>, "xn", i);
    });
#if !s_hardPhaseSilent
    auto c1m = armijoParam * dot(pol, "dir", "dir");
    fmt::print(fg(fmt::color::white), "c1m : {}\n", c1m);
#endif
    do {
        pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), alpha] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xn0", i) + alpha * vtemp.pack(dim_c<3>, "dir", i);
        });

        ///
        /// @note check c_ij(x^{l+1}). ref 4.2.2, item 1
        ///
        temp.setVal(0);
        auto B2 = B * B;
        pol(range(npp),
            [vtemp = view<space>({}, vtemp), PP = view<space>(PP), mark = view<space>(temp),
             threshold = (B + Btight) * (B + Btight) + epsCond,
             B2 = B2 + epsCond] __device__(int i) mutable { // no constraints margin here according to paper 4.2.2
                auto pp = PP[i];
                auto x0 = vtemp.pack(dim_c<3>, "xn", pp[0]);
                auto x1 = vtemp.pack(dim_c<3>, "xn", pp[1]);
                auto x0k = vtemp.pack(dim_c<3>, "xn0", pp[0]);
                auto x1k = vtemp.pack(dim_c<3>, "xn0", pp[1]);
                auto ek = x1k - x0k, ek1 = x1 - x0;
                auto dir = ek1 - ek;
                auto de2 = dir.l2NormSqr();
                if (de2 > 10 * limits<T>::epsilon()) // check continuous constraints 4.2.1 & 4.1
                {
                    auto numerator = -ek.dot(dir);
                    auto t = numerator / de2;
                    if (t > 0 && t < 1) {
                        auto et = t * dir + ek;
                        if (et.l2NormSqr() < B2) {
#if !s_hardPhaseSilent
                            printf("linesearch t: %f, et.l2NormSqr: %f, threshold: %f, pp: %d, %d, last: %f, de2: %f\n",
                                   (float)t, (float)(et.l2NormSqr()), (float)threshold, pp[0], pp[1],
                                   (float)ek.l2NormSqr(), (float)de2);
#endif
                            mark[0] = 1;
                            return;
                        }
                    }
                }
                if (auto d2 = dist2_pp(x0, x1); d2 < threshold) {
                    auto dir0 = vtemp.pack(dim_c<3>, "dir", pp[0]).l2NormSqr();
                    auto dir1 = vtemp.pack(dim_c<3>, "dir", pp[1]).l2NormSqr();
#if !s_hardPhaseSilent
                    printf("linesearch discrete pp, d2: %f, pp: %d, %d, dir: %f, %f \n", (float)d2, pp[0], pp[1],
                           (float)dir0, (float)dir1);
#endif
                    mark[0] = 1;
                }
            });
#if 1
        if (temp.getVal() == 0) {
            pol(range(ne),
                [vtemp = view<space>({}, vtemp), E = view<space>(E), mark = view<space>(temp),
                 threshold =
                     L * L - epsCond] __device__(int i) mutable { // no constraints margin here according to paper 4.2.2
                    auto e = E[i];
                    auto x0 = vtemp.pack(dim_c<3>, "xn", e[0]);
                    auto x1 = vtemp.pack(dim_c<3>, "xn", e[1]);
                    auto xk0 = vtemp.pack(dim_c<3>, "xn0", e[0]);
                    auto xk1 = vtemp.pack(dim_c<3>, "xn0", e[1]);
                    if (auto d2 = dist2_pp(x0, x1); d2 > threshold) {
                        auto dir0 = vtemp.pack(dim_c<3>, "dir", e[0]).l2NormSqr();
                        auto dir1 = vtemp.pack(dim_c<3>, "dir", e[1]).l2NormSqr();
                        auto dk2 = dist2_pp(xk0, xk1);
#if !s_hardPhaseSilent
                        printf("linesearch ee, ee: %d, %d, d2: %f, dir: %f, %f, dk2: %f, threshold: %f\n", e[0], e[1],
                               (float)d2, (float)dir0, (float)dir1, (float)dk2, (float)threshold);
#endif
                        mark[0] = 1;
                    }
                });
        }
#endif

        /// @brief backtracking if discrete constraints violated
        if (temp.getVal() == 1) {
            if (alpha < 1e-15) {
                throw std::runtime_error("stepsize too tiny in hard phase collision solve");
            }
            alpha /= 2.0f;
#if !s_hardPhaseSilent
            fmt::print("\t[back-tracing] alpha: {} constraint not satisfied\n", alpha);
#endif
            continue;
        }
#if !s_hardPhaseSilent
        fmt::print("[back-tracing] acceptable alpha: {}\n", alpha);
#endif
        ///
        /// @note objective decreases adequately. ref 4.2.2, item 2
        ///
#if s_hardPhaseLinesearch
        auto E = constraintEnergy(pol);
        if (E <= E0 + limits<T>::epsilon() * 10.0f)
            return E;
        alpha /= 2;
        if (alpha < 1e-3) {
            fmt::print("hard-phase linesearch early exit\n");
            return E;
        }
#else
        return 0;
#endif
    } while (true);
#if !s_hardPhaseSilent
    fmt::print(fg(fmt::color::antique_white), "alpha_l^hard: {}\n", alpha);
#endif
}

bool FastClothSystem::constraintSatisfied(zs::CudaExecutionPolicy &pol, bool hasEps) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    zs::CppTimer timer;
    if constexpr (s_enableProfile)
        timer.tick();

    temp.setVal(0);
    auto threshold = (B + Btight) * (B + Btight);
    auto B2 = B * B;
    if (hasEps) {
        B2 += epsCond;
        threshold += epsCond;
    }
    pol(range(npp), [vtemp = view<space>({}, vtemp), PP = view<space>(PP), mark = view<space>(temp), threshold, B2,
                     hasEps] __device__(int i) mutable { // epsCond: paper 4.2.2
        auto pp = PP[i];
        auto x0 = vtemp.pack(dim_c<3>, "xn", pp[0]);
        auto x1 = vtemp.pack(dim_c<3>, "xn", pp[1]);
        auto x0k = vtemp.pack(dim_c<3>, "xk", pp[0]);
        auto x1k = vtemp.pack(dim_c<3>, "xk", pp[1]);
        auto ek = x1k - x0k, ek1 = x1 - x0;
        auto dir = ek1 - ek;
        auto de2 = dir.l2NormSqr();
        if (hasEps && de2 > limits<T>::epsilon()) // check continuous constraints 4.2.1 & 4.1
        {
            auto numerator = -ek.dot(dir);
            auto t = numerator / de2;
            if (t > 0 && t < 1) {
                auto et = t * dir + ek;
                if (et.l2NormSqr() < B2) {
#if 0
                    printf("t: %f, et.l2NormSqr: %f, threshold: %f\n", 
                        (float)t, (float)(et.l2NormSqr()), (float)threshold);
#endif
                    mark[0] = 1;
                    return;
                }
            }
        }
        if (auto d2 = dist2_pp(x0, x1); d2 < threshold)
            mark[0] = 1;
    });
    threshold = L * L;
    if (hasEps)
        threshold -= epsCond;
    if (temp.getVal() == 0) {
        pol(range(ne), [vtemp = view<space>({}, vtemp), E = view<space>(E), mark = view<space>(temp),
                        threshold] __device__(int i) mutable { // epsCond: paper 4.2.2
            auto e = E[i];
            auto x0 = vtemp.pack(dim_c<3>, "xn", e[0]);
            auto x1 = vtemp.pack(dim_c<3>, "xn", e[1]);
            if (auto d2 = dist2_pp(x0, x1); d2 > threshold)
                mark[0] = 1;
        });
    }

    if constexpr (s_enableProfile) {
        timer.tock();
        collisionCnt[4]++;
        collisionTime[4] += timer.elapsed();
    }
    // all constraints satisfied if temp.getVal() == 0
    return temp.getVal() == 0;
}

typename FastClothSystem::T FastClothSystem::constraintEnergy(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    temp.setVal(0);
    pol(range(numDofs),
        [vtemp = view<space>({}, vtemp), energy = view<space>(temp), n = numDofs] __device__(int i) mutable {
            auto xinit = vtemp.pack(dim_c<3>, "xinit", i);
            auto xn = vtemp.pack(dim_c<3>, "xn", i);
            reduce_to(i, n, (xinit - xn).l2NormSqr(), energy[0]);
        });
    pol(range(npp), [vtemp = view<space>({}, vtemp), PP = view<space>(PP), energy = view<space>(temp), n = npp, mu = mu,
                     Btot2 = (B + Btight) * (B + Btight), eps = epsSlack, a3 = a3, a2 = a2] __device__(int i) mutable {
        auto pp = PP[i];
        zs::vec<T, 3> vs[2] = {vtemp.pack(dim_c<3>, "xn", pp[0]), vtemp.pack(dim_c<3>, "xn", pp[1])};
        T cij = (vs[1] - vs[0]).l2NormSqr() - Btot2;
        T f = eps;
        if (cij <= 0)
            printf("\n\n\nthis should not happen! pp constraint <%d, %d> cij: %f\n", (int)pp[0], (int)pp[1], cij);
        if (cij <= eps) {
            auto x2 = cij * cij;
            f = a3 * x2 * cij + a2 * x2 + cij;
        }
        T E = -mu * zs::log(f);
        reduce_to(i, n, E, energy[0]);
    });
    pol(range(ne), [vtemp = view<space>({}, vtemp), E = view<space>(E), energy = view<space>(temp), n = ne, mu = mu,
                    L2 = L * L, eps = epsSlack, a3 = a3, a2 = a2] __device__(int i) mutable {
        auto e = E[i];
        zs::vec<T, 3> vs[2] = {vtemp.pack(dim_c<3>, "xn", e[0]), vtemp.pack(dim_c<3>, "xn", e[1])};
        T cij = L2 - (vs[1] - vs[0]).l2NormSqr();
        T f = eps;
        if (cij <= 0)
            printf("\n\n\nthis should not happen! edge constraint <%d, %d> cij: %f\n", (int)e[0], (int)e[1], cij);
        if (cij <= eps) {
            auto x2 = cij * cij;
            f = a3 * x2 * cij + a2 * x2 + cij;
        }
        T E = -mu * zs::log(f);
        reduce_to(i, n, E, energy[0]);
    });
    return temp.getVal();
}

#if 0
void FastClothSystem::computeConstraintGradients(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto [npp, ne] = getConstraintCnt();
    fmt::print("dcd broad phase [pp, edge constraints]: {}, {}", npp, ne);
    pol(range(npp),
        [vtemp = view<space>({}, vtemp), tempPP = view<space>({}, tempPP), PP = view<space>(PP), rho = rho, mu = mu,
         Btot2 = (B + Btight) * (B + Btight), eps = epsSlack] __device__(int i) mutable {
            auto pp = PP[i];
            zs::vec<T, 3> vs[2] = {vtemp.pack(dim_c<3>, "xn", pp[0]), vtemp.pack(dim_c<3>, "xn", pp[1])};
            const auto &a = vs[0];
            const auto &b = vs[1];
            const auto t2 = a[0] * 2;
            const auto t3 = a[1] * 2;
            const auto t4 = a[2] * 2;
            const auto t5 = b[0] * 2;
            const auto t6 = b[1] * 2;
            const auto t7 = b[2] * 2;
            {

                const auto t8 = -t5;
                const auto t9 = -t6;
                const auto t10 = -t7;
                const auto t11 = t2 + t8;
                const auto t12 = t3 + t9;
                const auto t13 = t4 + t10;
                const auto t14 = rho * t11;
                const auto t15 = rho * t12;
                const auto t16 = rho * t13;
                auto grad = zs::vec<T, 6>{-t14, -t15, -t16, t14, t15, t16};
                tempPP.tuple(dim_c<6>, "softG", i) = grad;
            }
            {
                auto t8 = -Btot2;
                auto t9 = -b[0];
                auto t11 = -b[1];
                auto t13 = -b[2];
                auto t15 = 1 / eps;
                auto t10 = -t5;
                auto t12 = -t6;
                auto t14 = -t7;
                auto t16 = t15 * t15;
                auto t17 = a[0] + t9;
                auto t18 = a[1] + t11;
                auto t19 = a[2] + t13;
                auto t20 = t2 + t10;
                auto t21 = t3 + t12;
                auto t22 = t4 + t14;
                auto t23 = t17 * t17;
                auto t24 = t18 * t18;
                auto t25 = t19 * t10;
                auto t26 = t8 + t23 + t24 + t25;
                auto t27 = t26 * t26;
                auto t28 = t26 * t26 * t26;
                auto t32 = t15 * t20 * t26 * 2;
                auto t33 = t15 * t21 * t26 * 2;
                auto t34 = t15 * t22 * t26 * 2;
                auto t29 = t15 * t27;
                auto t30 = t16 * t28;
                auto t35 = t16 * t20 * t27 * 3;
                auto t36 = t16 * t21 * t27 * 3;
                auto t37 = t16 * t22 * t27 * 3;
                auto t31 = -t30;
                auto t38 = -t35;
                auto t39 = -t36;
                auto t40 = -t37;
                auto t41 = t20 + t32 + t38;
                auto t42 = t21 + t33 + t39;
                auto t43 = t22 + t34 + t40;
                auto t44 = t26 + t29 + t31;
                auto t45 = 1 / t44;
                auto t46 = mu * t41 * t45;
                auto t47 = mu * t42 * t45;
                auto t48 = mu * t43 * t45;
                auto grad = zs::vec<T, 6>{-t46, -t47, -t48, t46, t47, t48};
                tempPP.tuple(dim_c<6>, "hardG", i) = grad;
            }
        });

    pol(range(ne), [vtemp = view<space>({}, vtemp), tempE = view<space>({}, tempE), E = view<space>(E), rho = rho,
                    mu = mu, L2 = L * L, eps = epsSlack] __device__(int i) mutable {
        auto e = E[i];
        zs::vec<T, 3> vs[2] = {vtemp.pack(dim_c<3>, "xn", e[0]), vtemp.pack(dim_c<3>, "xn", e[1])};
        const auto &a = vs[0];
        const auto &b = vs[1];
        const auto t2 = a[0] * 2;
        const auto t3 = a[1] * 2;
        const auto t4 = a[2] * 2;
        const auto t5 = b[0] * 2;
        const auto t6 = b[1] * 2;
        const auto t7 = b[2] * 2;
        {
            const auto t8 = -t5;
            const auto t9 = -t6;
            const auto t10 = -t7;
            const auto t11 = t2 + t8;
            const auto t12 = t3 + t9;
            const auto t13 = t4 + t10;
            const auto t14 = rho * t11;
            const auto t15 = rho * t12;
            const auto t16 = rho * t13;
            auto grad = zs::vec<T, 6>{t14, t15, t16, -t14, -t15, -t16};
            tempE.tuple(dim_c<6>, "softG", i) = grad;
        }
        {
            auto t8 = -L2;
            auto t12 = -b[0];
            auto t14 = -b[1];
            auto t16 = -b[2];
            auto t18 = 1 / eps;
            auto t9 = -t2;
            auto t10 = -t3;
            auto t11 = -t4;
            auto t13 = -t5;
            auto t15 = -t6;
            auto t17 = -t7;
            auto t19 = t18 * t18;
            auto t20 = a[0] + t12;
            auto t21 = a[1] + t14;
            auto t22 = a[2] + t16;
            auto t23 = t2 + t13;
            auto t24 = t3 + t15;
            auto t25 = t4 + t17;
            auto t26 = t20 * t20;
            auto t27 = t21 * t21;
            auto t28 = t22 * t22;
            auto t29 = -t26;
            auto t30 = -t27;
            auto t31 = -t28;
            auto t32 = t8 + t26 + t27 + t28;
            auto t33 = t32 * t32;
            auto t34 = t32 * t32 * t32;
            auto t37 = t18 * t23 * t32 * 2;
            auto t38 = t18 * t24 * t32 * 2;
            auto t39 = t18 * t25 * t32 * 2;
            auto t35 = t18 * t33;
            auto t36 = t19 * t34;
            auto t40 = t19 * t23 * t33 * 3;
            auto t41 = t19 * t24 * t33 * 3;
            auto t42 = t19 * t25 * t33 * 3;
            auto t43 = t5 + t9 + t37 + t40;
            auto t44 = t6 + t10 + t38 + t41;
            auto t45 = t7 + t11 + t39 + t42;
            auto t46 = L2 + t29 + t30 + t31 + t35 + t36;
            auto t47 = 1 / t46;
            auto t48 = mu * t43 * t47;
            auto t49 = mu * t44 * t47;
            auto t50 = mu * t45 * t47;
            auto grad = zs::vec<T, 6>{-t48, -t49, -t50, t48, t49, t50};
            tempE.tuple(dim_c<6>, "hardG", i) = grad;
        }
    });
}
#endif

} // namespace zeno