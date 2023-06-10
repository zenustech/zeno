#include "../Ccds.hpp"
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
#include <zeno/utils/log.h>

#define PROFILE_IPC 0

namespace zeno {

inline typename UnifiedIPCSystem::T infNorm(zs::CudaExecutionPolicy &cudaPol,
                                            typename UnifiedIPCSystem::dtiles_t &vertData,
                                            const zs::SmallString tag = "dir") {
    using namespace zs;
    using T = typename UnifiedIPCSystem::T;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), count_warps(vertData.size())};
    zs::memset(zs::mem_device, res.data(), 0, sizeof(T) * count_warps(vertData.size()));
    cudaPol(range(vertData.size()), [data = proxy<space>({}, vertData), res = proxy<space>(res), n = vertData.size(),
                                     offset = vertData.getPropertyOffset(tag)] __device__(int pi) mutable {
        auto v = data.pack<3>(offset, pi);
        auto val = v.abs().max();

#if __CUDA_ARCH__ >= 800
        auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
        auto ret = zs::cg::reduce(tile, val, zs::cg::greater<T>());
        if (tile.thread_rank() == 0)
            res[pi / 32] = ret;
#else
        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = zs::max(val, tmp);
        }
        if (locid == 0)
            res[pi / 32] = val;
#endif
    });
    return reduce(cudaPol, res, thrust::maximum<T>{});
}

void UnifiedIPCSystem::computeConstraints(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(Collapse{numDofs}, [vtemp = proxy<space>({}, vtemp)] __device__(int vi) mutable {
        auto BCtarget = vtemp.pack(dim_c<3>, "BCtarget", vi);
        int BCorder = vtemp("BCorder", vi);
        // auto x = BCbasis.transpose() * vtemp.pack<3>("xn", vi);
        auto x = vtemp.pack(dim_c<3>, "xn", vi);
        int d = 0;
        for (; d != BCorder; ++d)
            vtemp("cons", d, vi) = x[d] - BCtarget[d];
        for (; d != 3; ++d)
            vtemp("cons", d, vi) = 0;
    });
}
bool UnifiedIPCSystem::areConstraintsSatisfied(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    computeConstraints(pol);
    auto res = constraintResidual(pol);
    return res < s_constraint_residual;
}
typename UnifiedIPCSystem::T UnifiedIPCSystem::constraintResidual(zs::CudaExecutionPolicy &pol, bool maintainFixed) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (projectDBC)
        return 0;
    Vector<T> num{vtemp.get_allocator(), numDofs}, den{vtemp.get_allocator(), numDofs};
    pol(Collapse{numDofs}, [vtemp = proxy<space>({}, vtemp), den = proxy<space>(den), num = proxy<space>(num),
                            maintainFixed] __device__(int vi) mutable {
        auto BCtarget = vtemp.pack(dim_c<3>, "BCtarget", vi);
        int BCorder = vtemp("BCorder", vi);
        auto cons = vtemp.pack(dim_c<3>, "cons", vi);
        auto xt = vtemp.pack(dim_c<3>, "xhat", vi);
        T n = 0, d_ = 0;
        // https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf Eq5
        for (int d = 0; d != BCorder; ++d) {
            n += zs::sqr(cons[d]);
            d_ += zs::sqr(xt[d] - BCtarget[d]);
        }
        num[vi] = n;
        den[vi] = d_;
        if (maintainFixed && BCorder > 0) {
            if (d_ != 0) {
                if (zs::sqrt(n / d_) < 1e-6)
                    vtemp("BCfixed", vi) = 1;
            } else {
                if (zs::sqrt(n) < 1e-6)
                    vtemp("BCfixed", vi) = 1;
            }
        }
    });
    auto nsqr = reduce(pol, num);
    auto dsqr = reduce(pol, den);
    T ret = 0;
    if (dsqr == 0)
        ret = std::sqrt(nsqr);
    else
        ret = std::sqrt(nsqr / dsqr);
    return ret < 1e-3 ? 0 : ret;
}

// https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
namespace cg = cooperative_groups;
__forceinline__ __device__ int atomicAggInc(int *ctr) noexcept {
    auto g = cg::coalesced_threads();
    int warp_res;
    if (g.thread_rank() == 0)
        warp_res = atomicAdd(ctr, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank();
}
#define USE_COALESCED 1

///
/// DCD
///
void UnifiedIPCSystem::findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    PP.reset();
    PE.reset();
    PT.reset();
    if (enableContactEE) {
        EE.reset();
        if (enableMollification) {
            PPM.reset();
            PEM.reset();
            EEM.reset();
        }
    }
    csPT.reset();
    csEE.reset();

#if PROFILE_IPC
    zs::CppTimer timer;
    timer.tick();
#endif
    if (enableContactSelf) {
        bvs.resize(stInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0, bvs);
        stBvh.refit(pol, bvs);
#if ENABLE_STQ
        stBvs.refit(pol, bvs);
#endif
        bvs.resize(seInds.size());
        /// @note embed thickness in bvs for faster self ee iteration
        retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, 0, dHat / 2, bvs);
        seBvh.refit(pol, bvs);
#if ENABLE_STQ
        seBvs.refit(pol, bvs);
#endif
        findCollisionConstraintsImpl(pol, dHat, xi, false);
    }

    if (hasBoundary()) {
        bvs.resize(coEles->size());
        retrieve_bounding_volumes(pol, vtemp, "xn", *coEles, zs::wrapv<3>{}, coOffset, bvs);
        bouStBvh.refit(pol, bvs);
#if ENABLE_STQ
        bouStBvs.refit(pol, bvs);
#endif
        bvs.resize(coEdges->size());
        retrieve_bounding_volumes(pol, vtemp, "xn", *coEdges, zs::wrapv<2>{}, coOffset, bvs);
        bouSeBvh.refit(pol, bvs);
#if ENABLE_STQ
        bouSeBvs.refit(pol, bvs);
#endif
        findCollisionConstraintsImpl(pol, dHat, xi, true);

        /// @note assume stBvh is already updated
        if (!enableContactSelf) {
            bvs.resize(stInds.size());
            retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0, bvs);
            stBvh.refit(pol, bvs);
#if ENABLE_STQ
            stBvs.refit(pol, bvs);
#endif
        }
        findBoundaryCollisionConstraintsImpl(pol, dHat, xi);
    }
    auto [npt, nee] = getCollisionCnts();
#if PROFILE_IPC
    timer.tock(fmt::format("dcd broad phase [pt, ee]({}, {})", npt, nee));
#else
    fmt::print(fg(fmt::color::light_golden_rod_yellow), "dcd broad phase [pt, ee]({}, {})\n", npt, nee);
#endif
}
void UnifiedIPCSystem::findBoundaryCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol.profile(PROFILE_IPC);
    /// pt
    snapshot(PP, PE, PT, csPT);
    do {
        pol(Collapse{numBouDofs},
            [eles = proxy<space>({}, stInds), exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp),
             bvh = proxy<space>(stBvh), PP = PP.port(), PE = PE.port(), PT = PT.port(), csPT = csPT.port(),
             dHat2 = zs::sqr(dHat + xi), thickness = xi + dHat, coOffset = coOffset] __device__(int i) mutable {
                auto vi = coOffset + i;
                if (exclDofs[vi])
                    return;
                auto p = vtemp.pack(dim_c<3>, "xn", vi);
                auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
                auto f = [&](int stI) {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c);
                    // all affected by sticky boundary conditions
                    if (vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 && vtemp("BCorder", tri[2]) == 3)
                        return;
                    if (exclDofs[tri[0]] || exclDofs[tri[1]] || exclDofs[tri[2]])
                        return;

                    auto t0 = vtemp.pack(dim_c<3>, "xn", tri[0]);
                    auto t1 = vtemp.pack(dim_c<3>, "xn", tri[1]);
                    auto t2 = vtemp.pack(dim_c<3>, "xn", tri[2]);

                    switch (pt_distance_type(p, t0, t1, t2)) {
                    case 0: {
                        if (auto d2 = dist2_pp(p, t0); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 1: {
                        if (auto d2 = dist2_pp(p, t1); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 2: {
                        if (auto d2 = dist2_pp(p, t2); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 3: {
                        if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[0], tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 4: {
                        if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 5: {
                        if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[2], tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 6: {
                        if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                            PT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    default: break;
                    }
                };
                bvh.iter_neighbors(bv, f);
            });
        if (allFit(PP, PE, PT, csPT))
            break;
        resizeAndRewind(PP, PE, PT, csPT);
    } while (true);
    pol.profile(false);
}
void UnifiedIPCSystem::findCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol.profile(PROFILE_IPC);
    /// pt
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    snapshot(PP, PE, PT, csPT);
    do {
        pol(range(svInds, "inds", dim_c<1>, int_c),
            [eles = proxy<space>({}, withBoundary ? *coEles : stInds), exclDofs = proxy<space>(exclDofs),
             vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stbvh), PP = PP.port(), PE = PE.port(), PT = PT.port(),
             csPT = csPT.port(), dHat, xi, thickness = xi + dHat,
             voffset = withBoundary ? coOffset : 0] __device__(int vi) mutable {
                if (exclDofs[vi])
                    return;
                // auto vi = front.prim(i);
                // vi = svInds("inds", vi, int_c);
                const auto dHat2 = zs::sqr(dHat + xi);
                int BCorder0 = vtemp("BCorder", vi);
                auto p = vtemp.pack(dim_c<3>, "xn", vi);
                auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
                auto f = [&](int stI) {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c) + voffset;
                    if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                        return;
                    // all affected by sticky boundary conditions
                    if (BCorder0 == 3 && vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 &&
                        vtemp("BCorder", tri[2]) == 3)
                        return;
                    if (exclDofs[tri[0]] || exclDofs[tri[1]] || exclDofs[tri[2]])
                        return;

                    auto t0 = vtemp.pack(dim_c<3>, "xn", tri[0]);
                    auto t1 = vtemp.pack(dim_c<3>, "xn", tri[1]);
                    auto t2 = vtemp.pack(dim_c<3>, "xn", tri[2]);

                    auto [cate, d2] = pt_category_and_dist2(p, t0, t1, t2);

                    switch (cate) {
                    case 0: {
                        if (d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 1: {
                        if (d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 2: {
                        if (d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 3: {
                        if (d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[0], tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 4: {
                        if (d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 5: {
                        if (d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[2], tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 6: {
                        if (d2 < dHat2) {
                            PT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    default: break;
                    }

#if 0
                    if (trueCate != chkCate) {
                        printf("$pt$\t\t$$$$\tref: %d, d2: %f; fact: %d, d2: %f\n", trueCate, (float)d2, chkCate,
                               (float)dd2);
                    }
#endif
                };
                bvh.iter_neighbors(bv, f);
            });
        if (allFit(PP, PE, PT, csPT))
            break;
        resizeAndRewind(PP, PE, PT, csPT);
    } while (true);
    /// ee
    if (enableContactEE) {
        const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
        snapshot(PP, PE, EE, PPM, PEM, EEM, csEE);
        do {
            if (withBoundary)
                pol(Collapse{seInds.size()},
                    [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, *coEdges),
                     exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh),
                     PP = PP.port(), PE = PE.port(), EE = EE.port(),
                     // mollifier
                     PPM = PPM.port(), PEM = PEM.port(), EEM = EEM.port(), enableMollification = enableMollification,
                     //
                     csEE = csEE.port(), dHat2 = zs::sqr(dHat + xi), xi, thickness = xi + dHat,
                     voffset = coOffset] __device__(int sei) mutable {
                        auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                        if (exclDofs[eiInds[0]] || exclDofs[eiInds[1]])
                            return;

                        bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;
                        auto v0 = vtemp.pack(dim_c<3>, "xn", eiInds[0]);
                        auto v1 = vtemp.pack(dim_c<3>, "xn", eiInds[1]);
                        auto rv0 = vtemp.pack(dim_c<3>, "x0", eiInds[0]);
                        auto rv1 = vtemp.pack(dim_c<3>, "x0", eiInds[1]);
                        auto [mi, ma] = get_bounding_box(v0, v1);
                        auto bv = bv_t{mi - thickness, ma + thickness};
                        auto f = [&](int sej) {
                            auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c) + voffset;
                            if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                                eiInds[1] == ejInds[1])
                                return;
                            // all affected by sticky boundary conditions
                            if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                                return;
                            if (exclDofs[ejInds[0]] || exclDofs[ejInds[1]])
                                return;

                            auto v2 = vtemp.pack(dim_c<3>, "xn", ejInds[0]);
                            auto v3 = vtemp.pack(dim_c<3>, "xn", ejInds[1]);
                            auto rv2 = vtemp.pack(dim_c<3>, "x0", ejInds[0]);
                            auto rv3 = vtemp.pack(dim_c<3>, "x0", ejInds[1]);

                            bool mollify = false;
                            if (enableMollification) {
                                // IPC (24)
                                T c = cn2_ee(v0, v1, v2, v3);
                                T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                                mollify = c < epsX;
                            }

                            auto [cate, d2] = ee_category_and_dist2(v0, v1, v2, v3);

                            switch (cate) {
                            case 0: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[0], ejInds[0]});
                                }
                                break;
                            }
                            case 1: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[1], ejInds[0]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[0], ejInds[1]});
                                }
                                break;
                            }
                            case 2: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{eiInds[0], ejInds[0], ejInds[1]});
                                }
                                break;
                            }
                            case 3: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[1], ejInds[0]});
                                }
                                break;
                            }
                            case 4: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[1], ejInds[0]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[1], ejInds[1]});
                                }
                                break;
                            }
                            case 5: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{eiInds[1], ejInds[0], ejInds[1]});
                                }
                                break;
                            }
                            case 6: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{ejInds[0], ejInds[1], eiInds[0], eiInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{ejInds[0], eiInds[0], eiInds[1]});
                                }
                                break;
                            }
                            case 7: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{ejInds[1], ejInds[0], eiInds[0], eiInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{ejInds[1], eiInds[0], eiInds[1]});
                                }
                                break;
                            }
                            case 8: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        EEM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    EE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                }
                                break;
                            }
                            default: break;
                            }
                        };
                        bvh.iter_neighbors(bv, f);
                    });
            else
                pol(Collapse{seInds.size()},
                    [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
                     exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh),
                     PP = PP.port(), PE = PE.port(), EE = EE.port(),
                     // mollifier
                     PPM = PPM.port(), PEM = PEM.port(), EEM = EEM.port(), enableMollification = enableMollification,
                     //
                     csEE = csEE.port(), dHat2 = zs::sqr(dHat + xi), xi,
                     thickness = xi + dHat] __device__(int ii) mutable {
                        int node = bvh._leafInds[ii];
                        int sei = bvh._auxIndices[node];
                        auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                        if (exclDofs[eiInds[0]] || exclDofs[eiInds[1]])
                            return;

                        bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;
                        auto v0 = vtemp.pack(dim_c<3>, "xn", eiInds[0]);
                        auto v1 = vtemp.pack(dim_c<3>, "xn", eiInds[1]);
                        auto rv0 = vtemp.pack(dim_c<3>, "x0", eiInds[0]);
                        auto rv1 = vtemp.pack(dim_c<3>, "x0", eiInds[1]);
                        auto f = [&](int sej) {
                            auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c);
                            if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                                eiInds[1] == ejInds[1])
                                return;
                            // all affected by sticky boundary conditions
                            if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                                return;
                            if (exclDofs[ejInds[0]] || exclDofs[ejInds[1]])
                                return;

                            auto v2 = vtemp.pack(dim_c<3>, "xn", ejInds[0]);
                            auto v3 = vtemp.pack(dim_c<3>, "xn", ejInds[1]);
                            auto rv2 = vtemp.pack(dim_c<3>, "x0", ejInds[0]);
                            auto rv3 = vtemp.pack(dim_c<3>, "x0", ejInds[1]);

                            bool mollify = false;
                            if (enableMollification) {
                                // IPC (24)
                                T c = cn2_ee(v0, v1, v2, v3);
                                T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                                mollify = c < epsX;
                            }

                            auto [cate, d2] = ee_category_and_dist2(v0, v1, v2, v3);

                            switch (cate) {
                            case 0: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[0], ejInds[0]});
                                }
                                break;
                            }
                            case 1: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[1], ejInds[0]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[0], ejInds[1]});
                                }
                                break;
                            }
                            case 2: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{eiInds[0], ejInds[0], ejInds[1]});
                                }
                                break;
                            }
                            case 3: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[1], ejInds[0]});
                                }
                                break;
                            }
                            case 4: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PPM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[1], ejInds[0]});
                                        break;
                                    }
                                    PP.try_push(pair_t{eiInds[1], ejInds[1]});
                                }
                                break;
                            }
                            case 5: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{eiInds[1], ejInds[0], ejInds[1]});
                                }
                                break;
                            }
                            case 6: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{ejInds[0], ejInds[1], eiInds[0], eiInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{ejInds[0], eiInds[0], eiInds[1]});
                                }
                                break;
                            }
                            case 7: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        PEM.try_push(pair4_t{ejInds[1], ejInds[0], eiInds[0], eiInds[1]});
                                        break;
                                    }
                                    PE.try_push(pair3_t{ejInds[1], eiInds[0], eiInds[1]});
                                }
                                break;
                            }
                            case 8: {
                                if (d2 < dHat2) {
                                    csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    if (mollify) {
                                        EEM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                        break;
                                    }
                                    EE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                }
                                break;
                            }
                            default: break;
                            }

#if 0
                        if (trueCate != chkCate) {
                            printf("#ee#\t\t####\tref: %d, d2: %f; fact: %d, d2: %f\n", trueCate, (float)d2, chkCate,
                                   (float)dd2);
                        }
#endif
                        };
                        bvh.self_iter_neighbors(ii, f);
                    });
            if (allFit(PP, PE, EE, PPM, PEM, EEM, csEE))
                break;
            resizeAndRewind(PP, PE, EE, PPM, PEM, EEM, csEE);
        } while (true);
    }
    pol.profile(false);
}

///
/// DCD (exp)
///
void UnifiedIPCSystem::findCollisionConstraints2(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    PP.reset();
    PE.reset();
    PT.reset();
    if (enableContactEE) {
        EE.reset();
        if (enableMollification) {
            PPM.reset();
            PEM.reset();
            EEM.reset();
        }
    }
    csPT.reset();
    csEE.reset();

    zs::CppTimer timer;
    timer.tick();
    if (enableContactSelf) {
        bvs.resize(stInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0, bvs);
        stBvh.refit(pol, bvs);
#if ENABLE_STQ
        stBvs.refit(pol, bvs);
#endif
        bvs.resize(seInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, 0, bvs);
        seBvh.refit(pol, bvs);
#if ENABLE_STQ
        seBvs.refit(pol, bvs);
#endif
        findCollisionConstraintsImpl2(pol, dHat, xi, false);
    }

    if (hasBoundary()) {
        bvs.resize(coEles->size());
        retrieve_bounding_volumes(pol, vtemp, "xn", *coEles, zs::wrapv<3>{}, coOffset, bvs);
        bouStBvh.refit(pol, bvs);
#if ENABLE_STQ
        bouStBvs.refit(pol, bvs);
#endif
        bvs.resize(coEdges->size());
        retrieve_bounding_volumes(pol, vtemp, "xn", *coEdges, zs::wrapv<2>{}, coOffset, bvs);
        bouSeBvh.refit(pol, bvs);
#if ENABLE_STQ
        bouSeBvs.refit(pol, bvs);
#endif
        findCollisionConstraintsImpl2(pol, dHat, xi, true);

        /// @note assume stBvh is already updated
        if (!enableContactSelf) {
            bvs.resize(stInds.size());
            retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0, bvs);
            stBvh.refit(pol, bvs);
#if ENABLE_STQ
            stBvs.refit(pol, bvs);
#endif
        }
        findBoundaryCollisionConstraintsImpl2(pol, dHat, xi);
    }
    auto [npt, nee] = getCollisionCnts();
    timer.tock(fmt::format("latest dcd broad phase [pt, ee]({}, {})", npt, nee));
}
void UnifiedIPCSystem::findBoundaryCollisionConstraintsImpl2(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol.profile(PROFILE_IPC);
    /// pt
    snapshot(PP, PE, PT, csPT);
    do {
        pol(Collapse{numBouDofs},
            [eles = proxy<space>({}, stInds), exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp),
             bvh = proxy<space>(stBvh), PP = PP.port(), PE = PE.port(), PT = PT.port(), csPT = csPT.port(),
             dHat2 = zs::sqr(dHat + xi), thickness = xi + dHat, coOffset = coOffset] __device__(int i) mutable {
                auto vi = coOffset + i;
                if (exclDofs[vi])
                    return;
                auto p = vtemp.pack(dim_c<3>, "xn", vi);
                auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
                auto f = [&](int stI) {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c);
                    // all affected by sticky boundary conditions
                    if (vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 && vtemp("BCorder", tri[2]) == 3)
                        return;
                    if (exclDofs[tri[0]] || exclDofs[tri[1]] || exclDofs[tri[2]])
                        return;

                    auto t0 = vtemp.pack(dim_c<3>, "xn", tri[0]);
                    auto t1 = vtemp.pack(dim_c<3>, "xn", tri[1]);
                    auto t2 = vtemp.pack(dim_c<3>, "xn", tri[2]);

                    switch (pt_distance_type(p, t0, t1, t2)) {
                    case 0: {
                        if (auto d2 = dist2_pp(p, t0); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 1: {
                        if (auto d2 = dist2_pp(p, t1); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 2: {
                        if (auto d2 = dist2_pp(p, t2); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 3: {
                        if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[0], tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 4: {
                        if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 5: {
                        if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[2], tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 6: {
                        if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                            PT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    default: break;
                    }
                };
                bvh.iter_neighbors(bv, f);
            });
        if (allFit(PP, PE, PT, csPT))
            break;
        resizeAndRewind(PP, PE, PT, csPT);
    } while (true);
    pol.profile(false);
}
void UnifiedIPCSystem::findCollisionConstraintsImpl2(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol.profile(PROFILE_IPC);
    /// pt
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    snapshot(PP, PE, PT, csPT);
    do {
        pol(range(svInds, "inds", dim_c<1>, int_c),
            [eles = proxy<space>({}, withBoundary ? *coEles : stInds), exclDofs = proxy<space>(exclDofs),
             vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stbvh), PP = PP.port(), PE = PE.port(), PT = PT.port(),
             csPT = csPT.port(), dHat, xi, thickness = xi + dHat,
             voffset = withBoundary ? coOffset : 0] __device__(int vi) mutable {
                if (exclDofs[vi])
                    return;
                // auto vi = front.prim(i);
                // vi = svInds("inds", vi, int_c);
                const auto dHat2 = zs::sqr(dHat + xi);
                int BCorder0 = vtemp("BCorder", vi);
                auto p = vtemp.pack(dim_c<3>, "xn", vi);
                auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
                auto f = [&](int stI) {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c) + voffset;
                    if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                        return;
                    // all affected by sticky boundary conditions
                    if (BCorder0 == 3 && vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 &&
                        vtemp("BCorder", tri[2]) == 3)
                        return;
                    if (exclDofs[tri[0]] || exclDofs[tri[1]] || exclDofs[tri[2]])
                        return;

                    auto t0 = vtemp.pack(dim_c<3>, "xn", tri[0]);
                    auto t1 = vtemp.pack(dim_c<3>, "xn", tri[1]);
                    auto t2 = vtemp.pack(dim_c<3>, "xn", tri[2]);

                    switch (pt_distance_type(p, t0, t1, t2)) {
                    case 0: {
                        if (auto d2 = dist2_pp(p, t0); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 1: {
                        if (auto d2 = dist2_pp(p, t1); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 2: {
                        if (auto d2 = dist2_pp(p, t2); d2 < dHat2) {
                            PP.try_push(pair_t{vi, tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 3: {
                        if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[0], tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 4: {
                        if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 5: {
                        if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2) {
                            PE.try_push(pair3_t{vi, tri[2], tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 6: {
                        if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                            PT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    default: break;
                    }
                };
                bvh.iter_neighbors(bv, f);
            });
        if (allFit(PP, PE, PT, csPT))
            break;
        resizeAndRewind(PP, PE, PT, csPT);
    } while (true);
    /// ee
    if (enableContactEE) {
        const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
        snapshot(PP, PE, EE, PPM, PEM, EEM, csEE);
        do {
            pol(Collapse{seInds.size()},
                [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
                 exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh),
                 PP = PP.port(), PE = PE.port(), EE = EE.port(),
                 // mollifier
                 PPM = PPM.port(), PEM = PEM.port(), EEM = EEM.port(), enableMollification = enableMollification,
                 //
                 csEE = csEE.port(), dHat2 = zs::sqr(dHat + xi), xi, thickness = xi + dHat,
                 voffset = withBoundary ? coOffset : 0] __device__(int sei) mutable {
                    auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                    if (exclDofs[eiInds[0]] || exclDofs[eiInds[1]])
                        return;

                    bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;
                    auto v0 = vtemp.pack(dim_c<3>, "xn", eiInds[0]);
                    auto v1 = vtemp.pack(dim_c<3>, "xn", eiInds[1]);
                    auto rv0 = vtemp.pack(dim_c<3>, "x0", eiInds[0]);
                    auto rv1 = vtemp.pack(dim_c<3>, "x0", eiInds[1]);
                    auto [mi, ma] = get_bounding_box(v0, v1);
                    auto bv = bv_t{mi - thickness, ma + thickness};
                    auto f = [&](int sej) {
                        if (voffset == 0 && sei < sej)
                            return;
                        auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c) + voffset;
                        if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                            eiInds[1] == ejInds[1])
                            return;
                        // all affected by sticky boundary conditions
                        if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                            return;
                        if (exclDofs[ejInds[0]] || exclDofs[ejInds[1]])
                            return;

                        auto v2 = vtemp.pack(dim_c<3>, "xn", ejInds[0]);
                        auto v3 = vtemp.pack(dim_c<3>, "xn", ejInds[1]);
                        auto rv2 = vtemp.pack(dim_c<3>, "x0", ejInds[0]);
                        auto rv3 = vtemp.pack(dim_c<3>, "x0", ejInds[1]);

                        bool mollify = false;
                        if (enableMollification) {
                            // IPC (24)
                            T c = cn2_ee(v0, v1, v2, v3);
                            T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                            mollify = c < epsX;
                        }

                        switch (ee_distance_type(v0, v1, v2, v3)) {
                        case 0: {
                            if (auto d2 = dist2_pp(v0, v2); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PPM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    break;
                                }
                                PP.try_push(pair_t{eiInds[0], ejInds[0]});
                            }
                            break;
                        }
                        case 1: {
                            if (auto d2 = dist2_pp(v0, v3); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PPM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[1], ejInds[0]});
                                    break;
                                }
                                PP.try_push(pair_t{eiInds[0], ejInds[1]});
                            }
                            break;
                        }
                        case 2: {
                            if (auto d2 = dist2_pe(v0, v2, v3); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PEM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    break;
                                }
                                PE.try_push(pair3_t{eiInds[0], ejInds[0], ejInds[1]});
                            }
                            break;
                        }
                        case 3: {
                            if (auto d2 = dist2_pp(v1, v2); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PPM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]});
                                    break;
                                }
                                PP.try_push(pair_t{eiInds[1], ejInds[0]});
                            }
                            break;
                        }
                        case 4: {
                            if (auto d2 = dist2_pp(v1, v3); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PPM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[1], ejInds[0]});
                                    break;
                                }
                                PP.try_push(pair_t{eiInds[1], ejInds[1]});
                            }
                            break;
                        }
                        case 5: {
                            if (auto d2 = dist2_pe(v1, v2, v3); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PEM.try_push(pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]});
                                    break;
                                }
                                PE.try_push(pair3_t{eiInds[1], ejInds[0], ejInds[1]});
                            }
                            break;
                        }
                        case 6: {
                            if (auto d2 = dist2_pe(v2, v0, v1); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PEM.try_push(pair4_t{ejInds[0], ejInds[1], eiInds[0], eiInds[1]});
                                    break;
                                }
                                PE.try_push(pair3_t{ejInds[0], eiInds[0], eiInds[1]});
                            }
                            break;
                        }
                        case 7: {
                            if (auto d2 = dist2_pe(v3, v0, v1); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    PEM.try_push(pair4_t{ejInds[1], ejInds[0], eiInds[0], eiInds[1]});
                                    break;
                                }
                                PE.try_push(pair3_t{ejInds[1], eiInds[0], eiInds[1]});
                            }
                            break;
                        }
                        case 8: {
                            if (auto d2 = dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                if (mollify) {
                                    EEM.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                                    break;
                                }
                                EE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                            }
                            break;
                        }
                        default: break;
                        }
                    };
                    bvh.iter_neighbors(bv, f);
                });
            if (allFit(PP, PE, EE, PPM, PEM, EEM, csEE))
                break;
            resizeAndRewind(PP, PE, EE, PPM, PEM, EEM, csEE);
        } while (true);
    }
    pol.profile(false);
}

///
/// CCD
///
void UnifiedIPCSystem::findCCDConstraints(zs::CudaExecutionPolicy &pol, T alpha, T xi) {
    csPT.reset();
    csEE.reset();

    if (enableContactSelf) {
        bvs.resize(stInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, vtemp, "dir", alpha, 0, bvs);
        stBvh.refit(pol, bvs);
#if ENABLE_STQ
        stBvs.refit(pol, bvs);
#endif
        bvs.resize(seInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, vtemp, "dir", alpha, 0, bvs);

        zs::CppTimer timer;
        // timer.tick();
        seBvh.refit(pol, bvs);
        // timer.tock("sebvh refit");

#if ENABLE_STQ
        // timer.tick();
        seBvs.refit(pol, bvs);
        // timer.tock("sebvs refit");
#endif

        findCCDConstraintsImpl(pol, alpha, xi, false);
    }

#if PROFILE_IPC
    zs::CppTimer timer;
    timer.tick();
#endif

    if (hasBoundary()) {
        bvs.resize(coEles->size());
        retrieve_bounding_volumes(pol, vtemp, "xn", *coEles, zs::wrapv<3>{}, vtemp, "dir", alpha, coOffset, bvs);
        bouStBvh.refit(pol, bvs);
#if ENABLE_STQ
        bouStBvs.refit(pol, bvs);
#endif

        bvs.resize(coEdges->size());
        retrieve_bounding_volumes(pol, vtemp, "xn", *coEdges, zs::wrapv<2>{}, vtemp, "dir", alpha, coOffset, bvs);
        bouSeBvh.refit(pol, bvs);
#if ENABLE_STQ
        bouSeBvs.refit(pol, bvs);
#endif

        findCCDConstraintsImpl(pol, alpha, xi, true);

        /// @note assume stBvh is already updated
        if (!enableContactSelf) {
            bvs.resize(stInds.size());
            retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, vtemp, "dir", alpha, 0, bvs);
            stBvh.refit(pol, bvs);
#if ENABLE_STQ
            stBvs.refit(pol, bvs);
#endif
        }
        findBoundaryCCDConstraintsImpl(pol, alpha, xi);
    }
    auto [npt, nee] = getCollisionCnts();
#if PROFILE_IPC
    timer.tock(fmt::format("ccd broad phase [pt, ee]({}, {})", npt, nee));
#else
    fmt::print(fg(fmt::color::light_golden_rod_yellow), "ccd broad phase [pt, ee]({}, {})\n", npt, nee);
#endif
}
void UnifiedIPCSystem::findBoundaryCCDConstraintsImpl(zs::CudaExecutionPolicy &pol, T alpha, T xi) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    const auto dHat2 = dHat * dHat;

    pol.profile(PROFILE_IPC);
    /// pt
    snapshot(csPT);
    do {
        pol(Collapse{numBouDofs},
            [eles = proxy<space>({}, stInds), exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp),
             bvh = proxy<space>(stBvh), csPT = csPT.port(), xi, alpha, coOffset = coOffset] __device__(int i) mutable {
                auto vi = coOffset + i;
                if (exclDofs[vi])
                    return;
                auto p = vtemp.pack(dim_c<3>, "xn", vi);
                auto dir = vtemp.pack(dim_c<3>, "dir", vi);
                auto bv = bv_t{get_bounding_box(p, p + alpha * dir)};
                bv._min -= xi;
                bv._max += xi;
                bvh.iter_neighbors(bv, [&](int stI) {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c);
                    if (vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 && vtemp("BCorder", tri[2]) == 3)
                        return;
                    if (exclDofs[tri[0]] || exclDofs[tri[1]] || exclDofs[tri[2]])
                        return;
                    csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                });
            });
        if (allFit(csPT))
            break;
        resizeAndRewind(csPT);
    } while (true);
    pol.profile(false);
}
void UnifiedIPCSystem::findCCDConstraintsImpl(zs::CudaExecutionPolicy &pol, T alpha, T xi, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    const auto dHat2 = dHat * dHat;

    pol.profile(PROFILE_IPC);
    /// pt
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    snapshot(csPT);
    do {
        pol(range(svInds, "inds", dim_c<1>, int_c),
            [svInds = proxy<space>({}, svInds), eles = proxy<space>({}, withBoundary ? *coEles : stInds),
             exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stbvh),
             csPT = csPT.port(), xi, alpha, voffset = withBoundary ? coOffset : 0] __device__(int vi) mutable {
                if (exclDofs[vi])
                    return;
                auto p = vtemp.pack(dim_c<3>, "xn", vi);
                auto dir = vtemp.pack(dim_c<3>, "dir", vi);
                auto bv = bv_t{get_bounding_box(p, p + alpha * dir)};
                bv._min -= xi;
                bv._max += xi;
                bvh.iter_neighbors(bv, [&](int stI) {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c) + voffset;
                    if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                        return;
                    // all affected by sticky boundary conditions
                    if (vtemp("BCorder", vi) == 3 && vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 &&
                        vtemp("BCorder", tri[2]) == 3)
                        return;
                    if (exclDofs[tri[0]] || exclDofs[tri[1]] || exclDofs[tri[2]])
                        return;

                    csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                });
            });
        if (allFit(csPT))
            break;
        resizeAndRewind(csPT);
    } while (true);
    /// ee
    if (enableContactEE) {
        if (!ENABLE_STQ) { //withBoundary
            const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
            snapshot(csEE);

            zs::CppTimer timer;
            if (!withBoundary)
                timer.tick();

            do {
                pol(Collapse{seInds.size()},
                    [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
                     exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh),
                     csEE = csEE.port(), xi, alpha, voffset = withBoundary ? coOffset : 0] __device__(int sei) mutable {
                        auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                        if (exclDofs[eiInds[0]] || exclDofs[eiInds[1]])
                            return;

                        bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;
                        auto v0 = vtemp.pack(dim_c<3>, "xn", eiInds[0]);
                        auto v1 = vtemp.pack(dim_c<3>, "xn", eiInds[1]);
                        auto dir0 = vtemp.pack(dim_c<3>, "dir", eiInds[0]);
                        auto dir1 = vtemp.pack(dim_c<3>, "dir", eiInds[1]);
                        auto bv = bv_t{get_bounding_box(v0, v0 + alpha * dir0)};
                        merge(bv, v1);
                        merge(bv, v1 + alpha * dir1);
                        bv._min -= xi;
                        bv._max += xi;
                        bvh.iter_neighbors(bv, [&](int sej) {
                            if (voffset == 0 && sei < sej)
                                return;
                            auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c) + voffset;
                            if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                                eiInds[1] == ejInds[1])
                                return;
                            // all affected by sticky boundary conditions
                            if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                                return;
                            if (exclDofs[ejInds[0]] || exclDofs[ejInds[1]])
                                return;

                            csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                        });
                    });
                if (allFit(csEE))
                    break;
                resizeAndRewind(csEE);
            } while (true);

            if (!withBoundary)
                timer.tock("bvh ee self ccd (bvh)");
        } else {
#if ENABLE_STQ
            findCCDConstraintsImplEE(pol, alpha, xi);
#endif
        }
    }
    pol.profile(false);
}
#if ENABLE_STQ
void UnifiedIPCSystem::findCCDConstraintsImplEE(zs::CudaExecutionPolicy &pol, T alpha, T xi) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    const auto dHat2 = dHat * dHat;

    pol.profile(PROFILE_IPC);
    /// ee
    if (enableContactEE) {
        const auto &sebvs = seBvs;
        snapshot(csEE);

        zs::CppTimer timer;
        timer.tick();
        auto offset = csEE.getCount();
        do {
#if 0
            // reference transitional impl
            pol(Collapse{seInds.size()},
                [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, seInds),
                 exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp), bvs = proxy<space>(sebvs),
                 csEE = csEE.port(), xi, alpha, axis = principalAxis, voffset = 0] __device__(int sei) mutable {
                    auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                    if (exclDofs[eiInds[0]] || exclDofs[eiInds[1]])
                        return;

                    bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;

#if 1
                    auto v0 = vtemp.pack(dim_c<3>, "xn", eiInds[0]);
                    auto v1 = vtemp.pack(dim_c<3>, "xn", eiInds[1]);
                    auto dir0 = vtemp.pack(dim_c<3>, "dir", eiInds[0]);
                    auto dir1 = vtemp.pack(dim_c<3>, "dir", eiInds[1]);
                    auto bv = bv_t{get_bounding_box(v0, v0 + alpha * dir0)};
                    merge(bv, v1);
                    merge(bv, v1 + alpha * dir1);
                    bv._min -= xi;
                    bv._max += xi;

                    if constexpr (false) {
                        int target;
                        for (int target = 0; target != bvs.numNodes(); ++target)
                            if (bvs._auxIndices[target] == sei)
                                break;
                        // auto bv = bv_t{bvs._bvs.pack(dim_c<3>, 0, target) , bvs._bvs.pack(dim_c<3>, 3, target)};

                        //
                        auto v = bv._min[axis];
                        auto loc = bvs.locate(v - 1e-4);
                        if (loc != target - 1)
                            printf("searching [%f] wrong ! [back] found: %d, ref: %d\n", v - 1e-4, loc, target - 1);
                        loc = bvs.locate(v);
                        if (loc != target)
                            printf("searching [%f] wrong ! [forward] found: %d, ref: %d\n", v, loc, target);
                    }
#endif
                    bvs.iter_neighbors(bv, [&](int sej) {
                        if (sei < sej)
                            return;
                        auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c);
                        if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                            eiInds[1] == ejInds[1])
                            return;
                        // all affected by sticky boundary conditions
                        if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                            return;
                        if (exclDofs[ejInds[0]] || exclDofs[ejInds[1]])
                            return;

                        csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                    });
                });
#else
            // expected impl
            pol(Collapse{sebvs.getNumNodes()},
                [seInds = proxy<space>({}, seInds), exclDofs = proxy<space>(exclDofs), vtemp = proxy<space>({}, vtemp),
                 bvs = proxy<space>(sebvs), csEE = csEE.port(), xi, alpha, axis = principalAxis,
                 voffset = 0] __device__(int ii) mutable {
                    int sei = bvs._auxIndices[ii];
                    auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                    if (exclDofs[eiInds[0]] || exclDofs[eiInds[1]])
                        return;

                    bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;

                    bvs.self_iter_neighbors(ii, [&](int sej) {
                        auto ejInds = seInds.pack(dim_c<2>, "inds", sej, int_c);
                        if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                            eiInds[1] == ejInds[1])
                            return;
                        // all affected by sticky boundary conditions
                        if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                            return;
                        if (exclDofs[ejInds[0]] || exclDofs[ejInds[1]])
                            return;

                        csEE.try_push(pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                    });
                });
#endif
            if (allFit(csEE))
                break;
            resizeAndRewind(csEE);
        } while (true);
        timer.tock("bvh ee self ccd (bvs)");
    }
    pol.profile(false);
}
#endif

///
/// Frictions
///
void UnifiedIPCSystem::precomputeFrictions(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    using namespace zs;

    if (!needFricPrecompute)
        return;
    needFricPrecompute = false;

    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    FPP.reset();
    FPE.reset();
    FPT.reset();
    FEE.reset();
    if (enableContact) {
        if (s_enableSelfFriction) {
            FPP.assignCounterFrom(PP);
            FPE.assignCounterFrom(PE);
            FPT.assignCounterFrom(PT);
            FEE.assignCounterFrom(EE);

            auto numFPP = FPP.getCount();
            fricPP.resize(numFPP);
            pol(range(numFPP),
                [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP), PP = PP.port(), FPP = FPP.port(),
                 xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fppi) mutable {
                    auto fpp = PP[fppi];
                    FPP[fppi] = fpp;
                    auto x0 = vtemp.pack<3>("xn", fpp[0]);
                    auto x1 = vtemp.pack<3>("xn", fpp[1]);
                    auto dist2 = dist2_pp(x0, x1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPP("fn", fppi) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPP.tuple<6>("basis", fppi) = point_point_tangent_basis(x0, x1);
                });
            auto numFPE = FPE.getCount();
            fricPE.resize(numFPE);
            pol(range(numFPE),
                [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE), PE = PE.port(), FPE = FPE.port(),
                 xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fpei) mutable {
                    auto fpe = PE[fpei];
                    FPE[fpei] = fpe;
                    auto p = vtemp.pack<3>("xn", fpe[0]);
                    auto e0 = vtemp.pack<3>("xn", fpe[1]);
                    auto e1 = vtemp.pack<3>("xn", fpe[2]);
                    auto dist2 = dist2_pe(p, e0, e1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPE("fn", fpei) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPE("yita", fpei) = point_edge_closest_point(p, e0, e1);
                    fricPE.tuple<6>("basis", fpei) = point_edge_tangent_basis(p, e0, e1);
                });
            auto numFPT = FPT.getCount();
            fricPT.resize(numFPT);
            pol(range(numFPT),
                [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT), PT = PT.port(), FPT = FPT.port(),
                 xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fpti) mutable {
                    auto fpt = PT[fpti];
                    FPT[fpti] = fpt;
                    auto p = vtemp.pack<3>("xn", fpt[0]);
                    auto t0 = vtemp.pack<3>("xn", fpt[1]);
                    auto t1 = vtemp.pack<3>("xn", fpt[2]);
                    auto t2 = vtemp.pack<3>("xn", fpt[3]);
                    auto dist2 = dist2_pt(p, t0, t1, t2);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPT("fn", fpti) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPT.tuple<2>("beta", fpti) = point_triangle_closest_point(p, t0, t1, t2);
                    fricPT.tuple<6>("basis", fpti) = point_triangle_tangent_basis(p, t0, t1, t2);
                });
            auto numFEE = FEE.getCount();
            fricEE.resize(numFEE);
            pol(range(numFEE),
                [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE), EE = EE.port(), FEE = FEE.port(),
                 xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int feei) mutable {
                    auto fee = EE[feei];
                    FEE[feei] = fee;
                    auto ea0 = vtemp.pack<3>("xn", fee[0]);
                    auto ea1 = vtemp.pack<3>("xn", fee[1]);
                    auto eb0 = vtemp.pack<3>("xn", fee[2]);
                    auto eb1 = vtemp.pack<3>("xn", fee[3]);
                    auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricEE("fn", feei) = -bGrad * 2 * zs::sqrt(dist2);
                    fricEE.tuple<2>("gamma", feei) = edge_edge_closest_point(ea0, ea1, eb0, eb1);
                    fricEE.tuple<6>("basis", feei) = edge_edge_tangent_basis(ea0, ea1, eb0, eb1);
                });
        }
    }
    if (enableGround) {
        for (auto &primHandle : prims) {
            if (primHandle.isBoundary()) // skip soft boundary
                continue;
            const auto &svs = primHandle.getSurfVerts();
            pol(range(svs.size()),
                [vtemp = proxy<space>({}, vtemp), svs = proxy<space>({}, svs),
                 svtemp = proxy<space>({}, primHandle.svtemp), kappa = kappa, xi2 = xi * xi, activeGap2,
                 gn = s_groundNormal, vOffset = primHandle.vOffset] ZS_LAMBDA(int svi) mutable {
                    const auto vi = svs("inds", svi, int_c) + vOffset;
                    auto x = vtemp.pack<3>("xn", vi);
                    auto dist = gn.dot(x);
                    auto dist2 = dist * dist;
                    if (dist2 < activeGap2) {
                        auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                        svtemp("fn", svi) = -bGrad * 2 * dist;
                    } else
                        svtemp("fn", svi) = 0;
                });
        }
    }
}

void UnifiedIPCSystem::project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    // projection
    if (projectDBC)
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), tagOffset = vtemp.getPropertyOffset(tag),
                                 orderOffset = vtemp.getPropertyOffset("BCorder")] ZS_LAMBDA(int vi) mutable {
            int BCorder = vtemp(orderOffset, vi);
            if (BCorder != 0)
                for (int d = 0; d != 3; ++d)
                    vtemp(tagOffset + d, vi) = 0;
        });
    else
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), tagOffset = vtemp.getPropertyOffset(tag),
                                 fixedOffset = vtemp.getPropertyOffset("BCfixed"),
                                 orderOffset = vtemp.getPropertyOffset("BCorder")] ZS_LAMBDA(int vi) mutable {
            int BCfixed = vtemp(fixedOffset, vi);
            if (BCfixed) {
                int BCorder = vtemp(orderOffset, vi);
                if (BCorder != 0)
                    for (int d = 0; d != 3; ++d)
                        vtemp(tagOffset + d, vi) = 0;
            }
        });
}

void UnifiedIPCSystem::precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag,
                                    const zs::SmallString dstTag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
// precondition
#if 0
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), srcTag, dstTag] ZS_LAMBDA(int vi) mutable {
        vtemp.tuple(dim_c<3>, dstTag, vi) = vtemp.pack(dim_c<3, 3>, "P", vi) * vtemp.pack(dim_c<3>, srcTag, vi);
    });
#endif
    pol(zs::range(numDofs * 3), [vtemp = proxy<space>({}, vtemp), srcOffset = vtemp.getPropertyOffset(srcTag),
                                 dstOffset = vtemp.getPropertyOffset(dstTag),
                                 POffset = vtemp.getPropertyOffset("P")] ZS_LAMBDA(int vi) mutable {
        int d = vi % 3;
        vi /= 3;
        float sum = 0;
        POffset += d * 3;
        for (int j = 0; j != 3; ++j)
            sum += vtemp(POffset + j, vi) * vtemp(srcOffset + j, vi);
        vtemp(dstOffset + d, vi) = sum;
    });
}

void UnifiedIPCSystem::systemMultiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag,
                                      const zs::SmallString bTag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    constexpr auto execTag = wrapv<space>{};
    using T = typename RM_CVREF_T(linsys)::T;
    using vec3 = zs::vec<T, 3>;
    // dx -> b
    pol(range(numDofs), [execTag, vtemp = proxy<space>({}, vtemp), bTag] ZS_LAMBDA(int vi) mutable {
        vtemp.tuple(dim_c<3>, bTag, vi) = vec3::zeros();
    });
    // CppTimer timer;
    // timer.tick();
    {
        const auto &spmat = linsys.spmat;
#if 0
        /// upper part (with diagonal)
        pol(range(spmat.outerSize() * 8),
            [vtemp = view<space>(vtemp), dxOffset = vtemp.getPropertyOffset(dxTag),
             bOffset = vtemp.getPropertyOffset(bTag), spmat = proxy<space>(spmat)] ZS_LAMBDA(int tid) mutable {
                auto tile = cg::tiled_partition<8>(cg::this_thread_block());
                auto laneId = tile.thread_rank();
                auto laneR = laneId / 3;
                auto laneC = laneId % 3;
                auto row = tid / tile.num_threads();
                auto bg = spmat._ptrs[row];
                auto ed = spmat._ptrs[row + 1];
                typename RM_CVREF_T(spmat)::value_type::value_type sum = 0;
                for (auto i = bg; i < ed; ++i) {
                    sum += spmat._vals[i](laneR, laneC) * vtemp(dxOffset + laneC, spmat._inds[i]);
                    if (laneId == 7)
                        sum += spmat._vals[i].val(8) * vtemp(dxOffset + 2, spmat._inds[i]);
                }
                auto v1 = tile.shfl_down(sum, 1);
                if (laneC == 0)
                    sum += v1;
                auto v2 = tile.shfl_down(sum, 2);
                if (laneC == 0 && laneR != 2)
                    sum += v2;
                if (laneC == 0)
                    vtemp(bOffset + laneR, row) += sum;
            });

        /// lower part (without diagonal)
        pol(range(spmat.outerSize() * 8),
            [vtemp = view<space>(vtemp), dxOffset = vtemp.getPropertyOffset(dxTag),
             bOffset = vtemp.getPropertyOffset(bTag), spmat = proxy<space>(spmat)] ZS_LAMBDA(int tid) mutable {
                auto tile = zs::cg::tiled_partition<8>(zs::cg::this_thread_block());
                auto laneId = tile.thread_rank();
                auto laneR = laneId / 3;
                auto laneC = laneId % 3;
                auto col = tid / tile.num_threads();
                auto bg = spmat._ptrs[col] + 1; // skip the diagonal part
                auto ed = spmat._ptrs[col + 1];

                auto dx = vtemp(dxOffset + laneR, col);
                for (auto k = bg; k < ed; ++k) {
                    auto row = spmat._inds[k];
                    auto inc = spmat._vals[k](laneR, laneC) * dx;
                    atomic_add(exec_cuda, &vtemp(bOffset + laneC, row), inc);
                    if (laneId == 7) {
                        inc = spmat._vals[k].val(8) * dx;
                        atomic_add(exec_cuda, &vtemp(bOffset + 2, row), inc);
                    }
                }
            });
#else
        pol(range(spmat.outerSize() * 8),
            [vtemp = view<space>(vtemp), dxOffset = vtemp.getPropertyOffset(dxTag),
             bOffset = vtemp.getPropertyOffset(bTag), spmat = proxy<space>(spmat)] ZS_LAMBDA(int tid) mutable {
                auto tile = cg::tiled_partition<8>(cg::this_thread_block());
                auto laneId = tile.thread_rank();
                auto laneR = laneId / 3;
                auto laneC = laneId % 3;
                auto row = tid / tile.num_threads();
                auto bg = spmat._ptrs[row];
                auto ed = spmat._ptrs[row + 1];
                // for upper part
                typename RM_CVREF_T(spmat)::value_type::value_type rowSum = 0;
                // for lower part
                auto dx = vtemp(dxOffset + laneR, row);
                for (auto i = bg; i < ed; ++i) {
                    auto col = spmat._inds[i];
                    auto He = spmat._vals[i](laneR, laneC);
                    // upper part
                    rowSum += He * vtemp(dxOffset + laneC, col);
                    // lower part
                    if (i != bg)
                        atomic_add(exec_cuda, &vtemp(bOffset + laneC, col), He * dx);
                    if (laneId == 7) {
                        He = spmat._vals[i].val(8);
                        // upper part
                        rowSum += He * vtemp(dxOffset + 2, col);
                        // lower part
                        if (i != bg)
                            atomic_add(exec_cuda, &vtemp(bOffset + 2, col), He * dx);
                    }
                }
                auto v1 = tile.shfl_down(rowSum, 1);
                if (laneC == 0)
                    rowSum += v1;
                auto v2 = tile.shfl_down(rowSum, 2);
                if (laneC == 0 && laneR != 2)
                    rowSum += v2;
                /// @note must use atomic version now that it's all mixed up
                if (laneC == 0)
                    atomic_add(exec_cuda, &vtemp(bOffset + laneR, row), rowSum);
            });
#endif
    }
    {
        auto &dynHess = linsys.dynHess;
        auto n = dynHess.getCount();
        pol(range(n * 8),
            [vtemp = view<space>(vtemp), dxOffset = vtemp.getPropertyOffset(dxTag),
             bOffset = vtemp.getPropertyOffset(bTag), dynHess = dynHess.port()] __device__(int i_) mutable {
                auto tile = zs::cg::tiled_partition<8>(zs::cg::this_thread_block());
                auto laneId = tile.thread_rank();
                auto laneR = laneId / 3;
                auto laneC = laneId % 3;
                auto [inds, H] = dynHess[i_ / tile.num_threads()];
                auto He = H.val(laneId);
                // upper
                atomic_add(exec_cuda, &vtemp(bOffset + laneR, inds[0]), He * vtemp(dxOffset + laneC, inds[1]));
                // lower
                atomic_add(exec_cuda, &vtemp(bOffset + laneC, inds[1]), He * vtemp(dxOffset + laneR, inds[0]));
                if (laneId == 7) {
                    He = H.val(8);
                    // upper
                    atomic_add(exec_cuda, &vtemp(bOffset + 2, inds[0]), He * vtemp(dxOffset + 2, inds[1]));
                    // lower
                    atomic_add(exec_cuda, &vtemp(bOffset + 2, inds[1]), He * vtemp(dxOffset + 2, inds[0]));
                }
            });
    }
#if 0
    // hess1
    pol(zs::range(numDofs), [execTag, hess1 = proxy<space>(hess1), cgtemp = proxy<space>({}, cgtemp),
                             dxOffset = cgtemp.getPropertyOffset(dxTag),
                             bOffset = cgtemp.getPropertyOffset(bTag)] __device__(int i) mutable {
        auto H = hess1.hess[i];
        zs::vec<float, 3> dx{cgtemp(dxOffset, i), cgtemp(dxOffset + 1, i), cgtemp(dxOffset + 2, i)};
        // auto dx = cgtemp.pack(dim_c<3>, dxTag, i);
        dx = H * dx;
        for (int d = 0; d != 3; ++d)
            atomic_add(execTag, &cgtemp(bOffset + d, i), dx(d));
    });
#endif
    // timer.tock("multiply takes");
}

template <typename Model>
typename UnifiedIPCSystem::T elasticityEnergy(zs::CudaExecutionPolicy &pol, typename UnifiedIPCSystem::dtiles_t &vtemp,
                                              typename UnifiedIPCSystem::PrimitiveHandle &primHandle,
                                              const Model &model, typename UnifiedIPCSystem::T dt,
                                              zs::Vector<typename UnifiedIPCSystem::T> &es) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename UnifiedIPCSystem::mat3;
    using vec3 = typename UnifiedIPCSystem::vec3;
    using T = typename UnifiedIPCSystem::T;

    auto &eles = primHandle.getEles();
    es.resize(count_warps(eles.size()));
    es.reset(0);
    const zs::SmallString tag = "xn";
    if (primHandle.category == ZenoParticles::curve) {
        if (primHandle.isBoundary() && !primHandle.isAuxiliary())
            return 0;
        // elasticity
        pol(range(eles.size()),
            [eles = proxy<space>({}, eles), vtemp = proxy<space>({}, vtemp), es = proxy<space>(es), tag, model = model,
             vOffset = primHandle.vOffset, n = eles.size()] __device__(int ei) mutable {
                auto inds = eles.pack(dim_c<2>, "inds", ei, int_c) + vOffset;

                int BCorder[2];
                for (int i = 0; i != 2; ++i)
                    BCorder[i] = vtemp("BCorder", inds[i]);
                T E;
                if (BCorder[0] == 3 && BCorder[1] == 3)
                    E = 0;
                else {
                    auto vole = eles("vol", ei);
                    auto k = eles("k", ei);
                    // auto k = model.mu;
                    auto rl = eles("rl", ei);
                    vec3 xs[2] = {vtemp.pack(dim_c<3>, tag, inds[0]), vtemp.pack(dim_c<3>, tag, inds[1])};
                    auto xij = xs[1] - xs[0];
                    auto lij = xij.norm();

                    E = (T)0.5 * k * zs::sqr(lij - rl) * vole;
                }
                reduce_to(ei, n, E, es[ei / 32]);
            });
        return reduce(pol, es) * dt * dt;
    } else if (primHandle.category == ZenoParticles::surface) {
        if (primHandle.isBoundary())
            return 0;
        // elasticity
        pol(range(eles.size()), [eles = proxy<space>({}, eles), vtemp = proxy<space>({}, vtemp), es = proxy<space>(es),
                                 tag, model = model, vOffset = primHandle.vOffset,
                                 n = eles.size()] __device__(int ei) mutable {
            auto IB = eles.template pack<2, 2>("IB", ei);
            auto inds = eles.pack(dim_c<3>, "inds", ei, int_c) + vOffset;

            int BCorder[3];
            for (int i = 0; i != 3; ++i)
                BCorder[i] = vtemp("BCorder", inds[i]);
            T E;
            if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 || determinant(IB) <= limits<float>::epsilon())
                E = 0;
            else {
                auto vole = eles("vol", ei);
                vec3 xs[3] = {vtemp.pack(dim_c<3>, tag, inds[0]), vtemp.pack(dim_c<3>, tag, inds[1]),
                              vtemp.pack(dim_c<3>, tag, inds[2])};
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];

                zs::vec<T, 3, 2> Ds{x1x0[0], x2x0[0], x1x0[1], x2x0[1], x1x0[2], x2x0[2]};
                auto F = Ds * IB;
                auto f0 = col(F, 0);
                auto f1 = col(F, 1);
                auto f0Norm = zs::sqrt(f0.l2NormSqr());
                auto f1Norm = zs::sqrt(f1.l2NormSqr());
                auto Estretch = model.mu * vole * (zs::sqr(f0Norm - 1) + zs::sqr(f1Norm - 1));
                auto Eshear = (model.mu * 0.3) * vole * zs::sqr(f0.dot(f1));
                E = Estretch + Eshear;
            }
            reduce_to(ei, n, E, es[ei / 32]);
        });
        return (reduce(pol, es) * dt * dt);
    } else if (primHandle.category == ZenoParticles::tet) {
        pol(zs::range(eles.size()),
            [vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, eles), es = proxy<space>(es), model, tag,
             vOffset = primHandle.vOffset, n = eles.size()] __device__(int ei) mutable {
                auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                auto inds = eles.pack(dim_c<4>, "inds", ei, int_c) + vOffset;
                auto vole = eles("vol", ei);
                vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]), vtemp.pack<3>(tag, inds[2]),
                              vtemp.pack<3>(tag, inds[3])};

                int BCorder[4];
                for (int i = 0; i != 4; ++i)
                    BCorder[i] = vtemp("BCorder", inds[i]);
                T E;
                if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 && BCorder[3] == 3)
                    E = 0;
                else {
                    mat3 F{};
                    auto x1x0 = xs[1] - xs[0];
                    auto x2x0 = xs[2] - xs[0];
                    auto x3x0 = xs[3] - xs[0];
                    auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1], x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                    F = Ds * IB;
                    E = model.psi(F) * vole;
                }
                reduce_to(ei, n, E, es[ei / 32]);
            });
        return (reduce(pol, es) * dt * dt);
    }
    return 0;
}

typename UnifiedIPCSystem::T UnifiedIPCSystem::energy(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> &es = temp;

    std::vector<T> Es(0);

    // inertial
    es.resize(count_warps(coOffset));
    es.reset(0);
    pol(range(coOffset), [vtemp = proxy<space>({}, vtemp), es = proxy<space>(es), tag, dt = this->dt,
                          n = coOffset] __device__(int vi) mutable {
        auto m = vtemp("ws", vi);
        auto x = vtemp.pack(dim_c<3>, tag, vi);
        auto xt = vtemp.pack(dim_c<3>, "xhat", vi);
        int BCorder = vtemp("BCorder", vi);
        T E = 0;
        if (BCorder == 0) {
            // inertia
            E = (T)0.5 * m * (x - vtemp.pack(dim_c<3>, "xtilde", vi)).l2NormSqr();
        }
        reduce_to(vi, n, E, es[vi / 32]);
    });
    Es.push_back(reduce(pol, es));

    if (vtemp.hasProperty("extf")) {
        es.resize(count_warps(coOffset));
        es.reset(0);
        pol(range(coOffset), [vtemp = proxy<space>({}, vtemp), es = proxy<space>(es), tag, dt = this->dt,
                              n = coOffset] ZS_LAMBDA(int vi) mutable {
            auto x = vtemp.pack<3>(tag, vi);
            auto xt = vtemp.pack<3>("xhat", vi);
            int BCorder = vtemp("BCorder", vi);
            T E = 0;
            {
                // external force
                // if (vtemp("BCsoft", vi) == 0 && vtemp("BCorder", vi) != 3)
                if (BCorder == 0) {
                    auto extf = vtemp.pack(dim_c<3>, "extf", vi);
                    E += -extf.dot(x - xt) * dt * dt;
                }
            }
            reduce_to(vi, n, E, es[vi / 32]);
        });
        Es.push_back(reduce(pol, es));
    }

    for (auto &primHandle : prims) {
        /// @note elasticity
        match([&](auto &elasticModel) {
            Es.push_back(elasticityEnergy(pol, vtemp, primHandle, elasticModel, dt, es));
        })(primHandle.getModels().getElasticModel());

        if (primHandle.hasBendingConstraints()) {
            /// @note bending energy (if exist)
            auto &bedges = *primHandle.bendingEdgesPtr;
            es.resize(count_warps(bedges.size()));
            es.reset(0);
            pol(range(bedges.size()),
                [vtemp = proxy<space>({}, vtemp), es = proxy<space>(es), bedges = proxy<space>({}, bedges), dt = dt,
                 vOffset = primHandle.vOffset, n = bedges.size()] __device__(int i) mutable {
                    auto stcl = bedges.pack(dim_c<4>, "inds", i, int_c) + vOffset;
                    int BCorder[4];
#pragma unroll
                    for (int i = 0; i != 4; ++i)
                        BCorder[i] = vtemp("BCorder", stcl[i]);
                    auto x0 = vtemp.pack(dim_c<3>, "xn", stcl[0]);
                    auto x1 = vtemp.pack(dim_c<3>, "xn", stcl[1]);
                    auto x2 = vtemp.pack(dim_c<3>, "xn", stcl[2]);
                    auto x3 = vtemp.pack(dim_c<3>, "xn", stcl[3]);
                    auto e = bedges("e", i);
                    auto h = bedges("h", i);
                    auto k = bedges("k", i); /// @note k here also determines the validity of this bending pair
                    auto ra = bedges("ra", i);
                    auto theta = dihedral_angle(x0, x1, x2, x3);
                    T E;
                    if (!(BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 && BCorder[3] == 3) &&
                        k > limits<float>::epsilon())
                        E = k * zs::sqr(theta - ra) * e / h * dt * dt;
                    else
                        E = 0;
                    reduce_to(i, n, E, es[i / 32]);
                });
            Es.push_back(reduce(pol, es));
        }
    }
    for (auto &primHandle : auxPrims) {
        using ModelT = RM_CVREF_T(primHandle.getModels().getElasticModel());
        const ModelT &model = primHandle.modelsPtr ? primHandle.getModels().getElasticModel() : ModelT{};
        match([&](auto &elasticModel) {
            Es.push_back(elasticityEnergy(pol, vtemp, primHandle, elasticModel, dt, es));
        })(model);
    }
    // contacts
    {
        if (enableContact)
            Es.push_back(collisionEnergy(pol, tag));

        if (enableGround) {
            for (auto &primHandle : prims) {
                if (primHandle.isBoundary()) // skip soft boundary
                    continue;
                const auto &svs = primHandle.getSurfVerts();
                // boundary
                es.resize(count_warps(svs.size()));
                es.reset(0);
                pol(range(svs.size()), [vtemp = proxy<space>({}, vtemp), svs = proxy<space>({}, svs),
                                        es = proxy<space>(es), gn = s_groundNormal, dHat2 = dHat * dHat, n = svs.size(),
                                        vOffset = primHandle.vOffset] ZS_LAMBDA(int svi) mutable {
                    const auto vi = svs("inds", svi, int_c) + vOffset;
                    auto x = vtemp.pack<3>("xn", vi);
                    auto dist = gn.dot(x);
                    auto dist2 = dist * dist;
                    T E;
                    if (dist2 < dHat2)
                        E = -zs::sqr(dist2 - dHat2) * zs::log(dist2 / dHat2);
                    else
                        E = 0;
                    reduce_to(svi, n, E, es[svi / 32]);
                });
                Es.push_back(reduce(pol, es) * kappa);

                if (s_enableFriction)
                    if (fricMu != 0) {
                        es.resize(count_warps(svs.size()));
                        es.reset(0);
                        pol(range(svs.size()),
                            [vtemp = proxy<space>({}, vtemp), svtemp = proxy<space>({}, primHandle.svtemp),
                             svs = proxy<space>({}, svs), es = proxy<space>(es), gn = s_groundNormal, dHat = dHat,
                             epsvh = epsv * dt, fricMu = fricMu, n = svs.size(),
                             vOffset = primHandle.vOffset] ZS_LAMBDA(int svi) mutable {
                                const auto vi = svs("inds", svi, int_c) + vOffset;
                                auto fn = svtemp("fn", svi);
                                T E = 0;
                                if (fn != 0) {
                                    auto x = vtemp.pack<3>("xn", vi);
                                    auto dx = x - vtemp.pack<3>("xhat", vi);
                                    auto relDX = dx - gn.dot(dx) * gn;
                                    auto relDXNorm2 = relDX.l2NormSqr();
                                    auto relDXNorm = zs::sqrt(relDXNorm2);
                                    if (relDXNorm > epsvh) {
                                        E = fn * (relDXNorm - epsvh / 2);
                                    } else {
                                        E = fn * relDXNorm2 / epsvh / 2;
                                    }
                                }
                                reduce_to(svi, n, E, es[svi / 32]);
                            });
                        Es.push_back(reduce(pol, es) * fricMu);
                    }
            }
        }
    }
    // constraints
    if (!BCsatisfied) {
        computeConstraints(pol);
        es.resize(count_warps(numDofs));
        es.reset(0);
        pol(range(numDofs), [vtemp = proxy<space>({}, vtemp), es = proxy<space>(es), n = numDofs,
                             boundaryKappa = boundaryKappa] __device__(int vi) mutable {
            // already updated during "xn" update
            auto cons = vtemp.pack(dim_c<3>, "cons", vi);
            auto w = vtemp("ws", vi);
            // auto lambda = vtemp.pack<3>("lambda", vi);
            int BCfixed = vtemp("BCfixed", vi);
            T E = 0;
            if (!BCfixed)
                // E = (T)(-lambda.dot(cons) * w + 0.5 * w * boundaryKappa * cons.l2NormSqr());
                E = (T)(0.5 * w * boundaryKappa * cons.l2NormSqr());
            reduce_to(vi, n, E, es[vi / 32]);
        });
        Es.push_back(reduce(pol, es));
    }

    std::sort(Es.begin(), Es.end());
    T E = 0;
    for (auto e : Es)
        E += e;
    return E;
}

void UnifiedIPCSystem::systemSolve(zs::CudaExecutionPolicy &cudaPol) {
    // input "grad", multiply, constraints
    // output "dir"
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// @note assume right-hand side is already projected
    /// copy diagonal block preconditioners
    cudaPol.sync(false);
    /// solve for A dir = grad;
    // initial guess for hard boundary constraints
    cudaPol(zs::range(numDofs),
            [vtemp = proxy<space>({}, vtemp), coOffset = coOffset, dt = dt, dirOffset = vtemp.getPropertyOffset("dir"),
             xtildeOffset = vtemp.getPropertyOffset("xtilde"),
             xnOffset = vtemp.getPropertyOffset("xn")] ZS_LAMBDA(int i) mutable {
                vtemp.tuple<3>(dirOffset, i) = vec3::zeros();
            });
    // temp = A * dir
    systemMultiply(cudaPol, "dir", "temp");
    project(cudaPol, "temp"); // project production
    // r = grad - temp
    cudaPol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), rOffset = vtemp.getPropertyOffset("r"),
                                 gradOffset = vtemp.getPropertyOffset("grad"),
                                 tempOffset = vtemp.getPropertyOffset("temp")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple<3>(rOffset, i) = vtemp.pack<3>(gradOffset, i) - vtemp.pack<3>(tempOffset, i);
    });
    // project(cudaPol, "r"); // project right hand side
#if USE_MAS
    linsys.precondition(cudaPol, vtemp, "r", "q");
#else
    precondition(cudaPol, "r", "q");
#endif
    cudaPol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), pOffset = vtemp.getPropertyOffset("p"),
                                 qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple<3>(pOffset, i) = vtemp.pack<3>(qOffset, i);
    });
    double zTrk = dot(cudaPol, "r", "q");
    double residualPreconditionedNorm2 = zTrk;
    double localTol2 = cgRel * cgRel * residualPreconditionedNorm2;
    int iter = 0;

    CppTimer timer;
    timer.tick();
    for (; iter != CGCap; ++iter) {
        if (iter % 50 == 0)
            fmt::print("cg iter: {}, norm2: {} (zTrk: {})\n", iter, residualPreconditionedNorm2, zTrk);

        if (residualPreconditionedNorm2 <= localTol2)
            break;
        systemMultiply(cudaPol, "p", "temp");
        project(cudaPol, "temp"); // project production

        double alpha = zTrk / dot(cudaPol, "temp", "p");
        cudaPol(range(numDofs), [vtemp = proxy<space>({}, vtemp), dirOffset = vtemp.getPropertyOffset("dir"),
                                 pOffset = vtemp.getPropertyOffset("p"), rOffset = vtemp.getPropertyOffset("r"),
                                 tempOffset = vtemp.getPropertyOffset("temp"), alpha] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, dirOffset, vi) =
                vtemp.pack(dim_c<3>, dirOffset, vi) + alpha * vtemp.pack<3>(pOffset, vi);
            vtemp.tuple(dim_c<3>, rOffset, vi) =
                vtemp.pack(dim_c<3>, rOffset, vi) - alpha * vtemp.pack<3>(tempOffset, vi);
        });

#if USE_MAS
        linsys.precondition(cudaPol, vtemp, "r", "q");
#else
        precondition(cudaPol, "r", "q");
#endif
        double zTrkLast = zTrk;
        zTrk = dot(cudaPol, "q", "r");
        if (zs::isnan(zTrk, zs::exec_seq)) {
            iter = CGCap;
            residualPreconditionedNorm2 = (localTol2 / (cgRel * cgRel)) + std::max((localTol2 / (cgRel * cgRel)), (T)1);
            continue;
        }
        double beta = zTrk / zTrkLast;
        cudaPol(range(numDofs), [vtemp = proxy<space>(vtemp), beta, pOffset = vtemp.getPropertyOffset("p"),
                                 qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple<3>(pOffset, vi) = vtemp.pack<3>(qOffset, vi) + beta * vtemp.pack<3>(pOffset, vi);
        });

        residualPreconditionedNorm2 = zTrk;
    } // end cg step
    /// copy back results
    if (iter == CGCap && residualPreconditionedNorm2 > (localTol2 / (cgRel * cgRel))) {
        // r = grad - temp
        cudaPol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), tempOffset = vtemp.getPropertyOffset("temp"),
                                     gradOffset = vtemp.getPropertyOffset("grad")] ZS_LAMBDA(int i) mutable {
            vtemp.tuple<3>(tempOffset, i) = vtemp.pack<3>(gradOffset, i);
        });
#if USE_MAS
        linsys.precondition(cudaPol, vtemp, "grad", "dir");
#else
        precondition(cudaPol, "grad", "dir");
#endif
        zeno::log_warn("falling back to gradient descent.");
    }
    cudaPol.sync(true);
    timer.tock(fmt::format("{} cgiters", iter));
}

void UnifiedIPCSystem::groundIntersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T &stepSize) {
    using namespace zs;
    // constexpr T slackness = 0.8;
    constexpr auto space = execspace_e::cuda;

    // zs::Vector<T> finalAlpha{vtemp.get_allocator(), 1};
    auto &finalAlpha = temp;
    finalAlpha.setVal(stepSize);
    pol(Collapse{coOffset},
        [vtemp = proxy<space>({}, vtemp),
         // boundary
         gn = s_groundNormal, finalAlpha = proxy<space>(finalAlpha), stepSize] ZS_LAMBDA(int vi) mutable {
            // this vert affected by sticky boundary conditions
            if (vtemp("BCorder", vi) == 3)
                return;
            auto dir = vtemp.pack<3>("dir", vi);
            auto coef = gn.dot(dir);
            if (coef < 0) { // impacting direction
                auto x = vtemp.pack<3>("xn", vi);
                auto dist = gn.dot(x);
                auto maxAlpha = (dist * 0.8) / (-coef);
                if (maxAlpha < stepSize)
                    atomic_min(exec_cuda, &finalAlpha[0], maxAlpha);
            }
        });
    stepSize = finalAlpha.getVal();
    fmt::print(fg(fmt::color::dark_cyan), "ground alpha: {}\n", stepSize);
}
void UnifiedIPCSystem::intersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T xi, T &stepSize) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    // Vector<T> alpha{vtemp.get_allocator(), 1};
    auto &alpha = temp;
    alpha.setVal(stepSize);
    auto npt = csPT.getCount();
#if PROFILE_IPC
    CppTimer timer;
    timer.tick();
#endif
    pol.profile(PROFILE_IPC);
    pol(range(npt), [csPT = csPT.port(), vtemp = proxy<space>({}, vtemp), alpha = proxy<space>(alpha), stepSize, xi,
                     coOffset = (int)coOffset] __device__(int pti) {
        auto ids = csPT[pti];
        auto p = vtemp.pack(dim_c<3>, "xn", ids[0]);
        auto t0 = vtemp.pack(dim_c<3>, "xn", ids[1]);
        auto t1 = vtemp.pack(dim_c<3>, "xn", ids[2]);
        auto t2 = vtemp.pack(dim_c<3>, "xn", ids[3]);
        auto dp = vtemp.pack(dim_c<3>, "dir", ids[0]);
        auto dt0 = vtemp.pack(dim_c<3>, "dir", ids[1]);
        auto dt1 = vtemp.pack(dim_c<3>, "dir", ids[2]);
        auto dt2 = vtemp.pack(dim_c<3>, "dir", ids[3]);
        T tmp = alpha[0];
#if 1
        if (accd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, xi, tmp))
#elif 1
            if (ticcd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, xi, tmp))
#else
            if (pt_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, xi, tmp))
#endif
            atomic_min(exec_cuda, &alpha[0], tmp);
    });
    auto nee = csEE.getCount();
    pol(range(nee), [csEE = csEE.port(), vtemp = proxy<space>({}, vtemp), alpha = proxy<space>(alpha), stepSize, xi,
                     coOffset = (int)coOffset] __device__(int eei) {
        auto ids = csEE[eei];
        auto ea0 = vtemp.pack(dim_c<3>, "xn", ids[0]);
        auto ea1 = vtemp.pack(dim_c<3>, "xn", ids[1]);
        auto eb0 = vtemp.pack(dim_c<3>, "xn", ids[2]);
        auto eb1 = vtemp.pack(dim_c<3>, "xn", ids[3]);
        auto dea0 = vtemp.pack(dim_c<3>, "dir", ids[0]);
        auto dea1 = vtemp.pack(dim_c<3>, "dir", ids[1]);
        auto deb0 = vtemp.pack(dim_c<3>, "dir", ids[2]);
        auto deb1 = vtemp.pack(dim_c<3>, "dir", ids[3]);
        auto tmp = alpha[0];
#if 1
        if (accd::eeccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, (T)0.2, xi, tmp))
#elif 1
            if (ticcd::eeccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, (T)0.2, xi, tmp))
#else
            if (ee_ccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, xi, tmp))
#endif
            atomic_min(exec_cuda, &alpha[0], tmp);
    });
    pol.profile(false);
#if PROFILE_IPC
    timer.tock("ccd time");
#endif
    stepSize = alpha.getVal();
}
void UnifiedIPCSystem::lineSearch(zs::CudaExecutionPolicy &cudaPol, T &alpha) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // initial energy
    T E0 = energy(cudaPol, "xn"); // must be "xn", cuz elasticity is hardcoded

    T E{E0};
    T c1m = 0;
    int lsIter = 0;
    // c1m = -armijoParam * dot(cudaPol, "dir", "grad");
    fmt::print(fg(fmt::color::white), "c1m : {}\n", c1m);
    do {
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp), alpha] __device__(int i) mutable {
            vtemp.tuple<3>("xn", i) = vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });

        if (enableContact)
            findCollisionConstraints(cudaPol, dHat, xi);

        E = energy(cudaPol, "xn"); // must be "xn", cuz elasticity is hardcoded

        fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
        if (E <= E0 + alpha * c1m)
            break;

        if (alpha < 1e-3) { // adhoc
            fmt::print(fg(fmt::color::light_yellow), "linesearch early exit with alpha {}\n", alpha);
            break;
        }

        alpha /= 2;
        if (++lsIter > 30) {
            auto cr = constraintResidual(cudaPol);
            fmt::print("too small stepsize at iteration [{}]! alpha: {}, cons "
                       "res: {}\n",
                       lsIter, alpha, cr);
        }
    } while (true);
}

bool UnifiedIPCSystem::newtonKrylov(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    if (!linsys.initialized) {
        initializeSystemHessian(pol);
        linsys.initialized = true;
    }

    /// optimizer
    int newtonIter = 0;
    T res = limits<T>::max();
    for (; newtonIter != PNCap; ++newtonIter) {
        // check constraints
        if (!BCsatisfied) {
            computeConstraints(pol);
            auto cr = constraintResidual(pol, true);
            if (cr < s_constraint_residual) {
                // zeno::log_info("satisfied cons res [{}] at newton iter [{}]\n", cr, newtonIter);
                projectDBC = true;
                BCsatisfied = true;
            }
            fmt::print(fg(fmt::color::alice_blue), "newton iter {} cons residual: {}\n", newtonIter, cr);
        }
        // PRECOMPUTE
        if (enableContact) {
            findCollisionConstraints(pol, dHat, xi);
        }
        if (s_enableFriction)
            if (fricMu != 0) {
                precomputeFrictions(pol, dHat, xi);
            }
        // GRAD, HESS, P
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::zeros();
            vtemp.tuple(dim_c<3>, "grad", i) = vec3::zeros();
        });

        /// prepare linsys.spmat
        updateInherentGradientAndHessian(pol, "grad");
        /// prepare linsys.hessx
        updateDynamicGradientAndHessian(pol, "grad");

#if USE_MAS
        // prepare MAS preconditioner
        linsys.buildPreconditioner(pol, *this); // 1
#else
        // prepare diagonal block preconditioner
        prepareDiagonalPreconditioner(pol);
        // P (INVERSION)
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            auto mat = vtemp.pack(dim_c<3, 3>, "P", i);
            if (zs::abs(zs::determinant(mat)) > limits<T>::epsilon() * 10)
                vtemp.tuple(dim_c<3, 3>, "P", i) = inverse(mat);
            else
                vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::identity();
        });
#endif

        project(pol, "grad");
        systemSolve(pol);

        // CHECK PN CONDITION
        res = infNorm(pol, "dir");
        T cons_res = constraintResidual(pol);
        /// @note do not exit in the beginning
        // if (res < targetGRes * dt && cons_res == 0 && newtonIter != 0)
        if (res < targetGRes * dt && newtonIter != 0) {
            break;
        }
        fmt::print(fg(fmt::color::aquamarine), "substep {} newton iter {}: direction residual {}\n", state.getSubstep(),
                   newtonIter, res);
        // LINESEARCH
        pol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "xn0", i) = vtemp.pack(dim_c<3>, "xn", i);
        });
        T alpha = 1.;
        if (enableContact) {
            /// @note dcd filter to reduce potential ccd stepsize
            intersectionFreeStepsize(pol, xi, alpha);
        }
        if (enableGround) {
            groundIntersectionFreeStepsize(pol, alpha);
        }
        if (enableContact) {
            // A.intersectionFreeStepsize(cudaPol, xi, alpha);
            // fmt::print("\tstepsize after intersection-free: {}\n", alpha);
            findCCDConstraints(pol, alpha, xi);
            intersectionFreeStepsize(pol, xi, alpha);
        }
        lineSearch(pol, alpha);
        pol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp), alpha] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xn0", i) + alpha * vtemp.pack(dim_c<3>, "dir", i);
        });
        // UPDATE RULE
        cons_res = constraintResidual(pol);
#if 0
        if (res * dt < updateZoneTol && cons_res > consTol) {
            if (boundaryKappa < kappaMax) {
                boundaryKappa *= 2;
                fmt::print(fg(fmt::color::ivory),
                           "increasing boundarykappa to {} due to constraint "
                           "difficulty.\n",
                           boundaryKappa);
                // getchar();
            } else {
#if 0
                pol(Collapse{numDofs},
                    [vtemp = proxy<space>({}, vtemp), boundaryKappa = boundaryKappa] ZS_LAMBDA(int vi) mutable {
                        if (int BCorder = vtemp("BCorder", vi); BCorder > 0) {
                            vtemp.tuple<3>("lambda", vi) = vtemp.pack<3>("lambda", vi) -
                                                           boundaryKappa * vtemp("ws", vi) * vtemp.pack<3>("cons", vi);
                        }
                    });
                fmt::print(fg(fmt::color::ivory), "updating constraint lambda due to constraint difficulty.\n");
                // getchar();
#endif
            }
        }
#endif
    }
    zeno::log_warn("\t# substep {} newton optimizer ends in {} iters with residual {}\n", state.getSubstep(),
                   newtonIter, res);
    return newtonIter != PNCap;
}

struct StepUnifiedIPCSystem : INode {
    void apply() override {
        using namespace zs;
        auto A = get_input<UnifiedIPCSystem>("ZSUnifiedIPCSystem");

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);
        A->suggestKappa(cudaPol);

        for (int subi = 0; subi != nSubsteps; ++subi) {
            A->advanceSubstep(cudaPol, (typename UnifiedIPCSystem::T)1 / nSubsteps);

            int numFricSolve = A->s_enableFriction && A->fricMu != 0 ? A->fricIterCap : 1;
        for_fric:
            A->needFricPrecompute = true;

            bool success = A->newtonKrylov(cudaPol);

            if (--numFricSolve > 0)
                goto for_fric;

            /// @note only update substep velocity when converged
            if (success)
                A->updateVelocities(cudaPol);
        }
        // update velocity and positions
        A->writebackPositionsAndVelocities(cudaPol);

        set_output("ZSUnifiedIPCSystem", A);
    }
};

ZENDEFNODE(StepUnifiedIPCSystem, {{
                                      "ZSUnifiedIPCSystem",
                                      {"int", "num_substeps", "1"},
                                      {"float", "dt", "0.01"},
                                  },
                                  {"ZSUnifiedIPCSystem"},
                                  {},
                                  {"FEM"}});

} // namespace zeno