#include "../Utils.hpp"
#include "Solver.cuh"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"

namespace zeno {

void IPCSystem::computeConstraints(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(Collapse{numDofs}, [vtemp = proxy<space>({}, vtemp)] __device__(int vi) mutable {
        auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
        auto BCtarget = vtemp.pack<3>("BCtarget", vi);
        int BCorder = vtemp("BCorder", vi);
        auto x = BCbasis.transpose() * vtemp.pack<3>("xn", vi);
        int d = 0;
        for (; d != BCorder; ++d)
            vtemp("cons", d, vi) = x[d] - BCtarget[d];
        for (; d != 3; ++d)
            vtemp("cons", d, vi) = 0;
    });
}
bool IPCSystem::areConstraintsSatisfied(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    computeConstraints(pol);
    auto res = constraintResidual(pol);
    return res < s_constraint_residual;
}
typename IPCSystem::T IPCSystem::constraintResidual(zs::CudaExecutionPolicy &pol, bool maintainFixed) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (projectDBC)
        return 0;
    Vector<T> num{vtemp.get_allocator(), numDofs}, den{vtemp.get_allocator(), numDofs};
    pol(Collapse{numDofs}, [vtemp = proxy<space>({}, vtemp), den = proxy<space>(den), num = proxy<space>(num),
                            maintainFixed] __device__(int vi) mutable {
        auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
        auto BCtarget = vtemp.pack<3>("BCtarget", vi);
        int BCorder = vtemp("BCorder", vi);
        auto cons = vtemp.pack<3>("cons", vi);
        auto xt = vtemp.pack<3>("xhat", vi);
        T n = 0, d_ = 0;
        // https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf Eq5
        for (int d = 0; d != BCorder; ++d) {
            n += zs::sqr(cons[d]);
            d_ += zs::sqr(col(BCbasis, d).dot(xt) - BCtarget[d]);
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
    return ret < 1e-6 ? 0 : ret;
}

void IPCSystem::findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    nPP.setVal(0);
    nPE.setVal(0);
    nPT.setVal(0);
    nEE.setVal(0);
    nPPM.setVal(0);
    nPEM.setVal(0);
    nEEM.setVal(0);

    ncsPT.setVal(0);
    ncsEE.setVal(0);
    {
        auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0);
        stBvh.refit(pol, triBvs);
        auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, 0);
        seBvh.refit(pol, edgeBvs);
        findCollisionConstraintsImpl(pol, dHat, xi, false);
    }

    if (coVerts)
        if (coVerts->size()) {
            auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", *coEles, zs::wrapv<3>{}, coOffset);
            bouStBvh.refit(pol, triBvs);
            auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", *coEdges, zs::wrapv<2>{}, coOffset);
            bouSeBvh.refit(pol, edgeBvs);
            findCollisionConstraintsImpl(pol, dHat, xi, true);
        }
}
void IPCSystem::findCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// pt
    pol(Collapse{svInds.size()},
        [svInds = proxy<space>({}, svInds), eles = proxy<space>({}, withBoundary ? *coEles : stInds),
         vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(withBoundary ? bouStBvh : stBvh), PP = proxy<space>(PP),
         nPP = proxy<space>(nPP), PE = proxy<space>(PE), nPE = proxy<space>(nPE), PT = proxy<space>(PT),
         nPT = proxy<space>(nPT), csPT = proxy<space>(csPT), ncsPT = proxy<space>(ncsPT), dHat, xi,
         thickness = xi + dHat, voffset = withBoundary ? coOffset : 0] __device__(int vi) mutable {
            vi = reinterpret_bits<int>(svInds("inds", vi));
            const auto dHat2 = zs::sqr(dHat + xi);
            int BCorder0 = vtemp("BCorder", vi);
            auto p = vtemp.template pack<3>("xn", vi);
            auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
            bvh.iter_neighbors(bv, [&](int stI) {
                auto tri = eles.template pack<3>("inds", stI).template reinterpret_bits<int>() + voffset;
                if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                    return;
                // all affected by sticky boundary conditions
                if (BCorder0 == 3 && vtemp("BCorder", tri[0]) == 3 && vtemp("BCorder", tri[1]) == 3 &&
                    vtemp("BCorder", tri[2]) == 3)
                    return;
                // ccd
                auto t0 = vtemp.template pack<3>("xn", tri[0]);
                auto t1 = vtemp.template pack<3>("xn", tri[1]);
                auto t2 = vtemp.template pack<3>("xn", tri[2]);

                switch (pt_distance_type(p, t0, t1, t2)) {
                case 0: {
                    if (auto d2 = dist2_pp(p, t0); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        PP[no] = pair_t{vi, tri[0]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 1: {
                    if (auto d2 = dist2_pp(p, t1); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        PP[no] = pair_t{vi, tri[1]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 2: {
                    if (auto d2 = dist2_pp(p, t2); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        PP[no] = pair_t{vi, tri[2]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 3: {
                    if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        PE[no] = pair3_t{vi, tri[0], tri[1]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 4: {
                    if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        PE[no] = pair3_t{vi, tri[1], tri[2]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 5: {
                    if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        PE[no] = pair3_t{vi, tri[2], tri[0]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 6: {
                    if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPT[0], 1);
                        PT[no] = pair4_t{vi, tri[0], tri[1], tri[2]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                default: break;
                }
            });
        });
    /// ee
    pol(Collapse{seInds.size()},
        [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
         vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(withBoundary ? bouSeBvh : seBvh), PP = proxy<space>(PP),
         nPP = proxy<space>(nPP), PE = proxy<space>(PE), nPE = proxy<space>(nPE), EE = proxy<space>(EE),
         nEE = proxy<space>(nEE),
#if s_enableMollification
         // mollifier
         PPM = proxy<space>(PPM), nPPM = proxy<space>(nPPM), PEM = proxy<space>(PEM), nPEM = proxy<space>(nPEM),
         EEM = proxy<space>(EEM), nEEM = proxy<space>(nEEM),
#endif
         //
         csEE = proxy<space>(csEE), ncsEE = proxy<space>(ncsEE), dHat, xi, thickness = xi + dHat,
         voffset = withBoundary ? coOffset : 0] __device__(int sei) mutable {
            const auto dHat2 = zs::sqr(dHat + xi);
            auto eiInds = seInds.template pack<2>("inds", sei).template reinterpret_bits<int>();
            bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 && vtemp("BCorder", eiInds[1]) == 3;
            auto v0 = vtemp.template pack<3>("xn", eiInds[0]);
            auto v1 = vtemp.template pack<3>("xn", eiInds[1]);
            auto rv0 = vtemp.template pack<3>("x0", eiInds[0]);
            auto rv1 = vtemp.template pack<3>("x0", eiInds[1]);
            auto [mi, ma] = get_bounding_box(v0, v1);
            auto bv = bv_t{mi - thickness, ma + thickness};
            bvh.iter_neighbors(bv, [&](int sej) {
                if (voffset == 0 && sei < sej)
                    return;
                auto ejInds = sedges.template pack<2>("inds", sej).template reinterpret_bits<int>() + voffset;
                if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                    eiInds[1] == ejInds[1])
                    return;
                // all affected by sticky boundary conditions
                if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 && vtemp("BCorder", ejInds[1]) == 3)
                    return;
                // ccd
                auto v2 = vtemp.template pack<3>("xn", ejInds[0]);
                auto v3 = vtemp.template pack<3>("xn", ejInds[1]);
                auto rv2 = vtemp.template pack<3>("x0", ejInds[0]);
                auto rv3 = vtemp.template pack<3>("x0", ejInds[1]);

#if s_enableMollification
                // IPC (24)
                T c = cn2_ee(v0, v1, v2, v3);
                T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                bool mollify = c < epsX;
#endif

                switch (ee_distance_type(v0, v1, v2, v3)) {
                case 0: {
                    if (auto d2 = dist2_pp(v0, v2); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                            PPM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPP[0], 1);
#if 0
                printf("ee category 0: %d-th <%d, %d, %d, %d>, dist: %f (%f) < "
                       "%f\n",
                       (int)no, (int)eiInds[0], (int)eiInds[1], (int)ejInds[0],
                       (int)ejInds[1], (float)zs::sqrt(d2),
                       (float)(v0 - v2).norm(), (float)dHat);
#endif
                            PP[no] = pair_t{eiInds[0], ejInds[0]};
                        }
                    }
                    break;
                }
                case 1: {
                    if (auto d2 = dist2_pp(v0, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                            PPM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[1], ejInds[0]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPP[0], 1);
                            PP[no] = pair_t{eiInds[0], ejInds[1]};
                        }
                    }
                    break;
                }
                case 2: {
                    if (auto d2 = dist2_pe(v0, v2, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                            PEM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPE[0], 1);
                            PE[no] = pair3_t{eiInds[0], ejInds[0], ejInds[1]};
                        }
                    }
                    break;
                }
                case 3: {
                    if (auto d2 = dist2_pp(v1, v2); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                            PPM[no] = pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPP[0], 1);
                            PP[no] = pair_t{eiInds[1], ejInds[0]};
                        }
                    }
                    break;
                }
                case 4: {
                    if (auto d2 = dist2_pp(v1, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                            PPM[no] = pair4_t{eiInds[1], eiInds[0], ejInds[1], ejInds[0]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPP[0], 1);
                            PP[no] = pair_t{eiInds[1], ejInds[1]};
                        }
                    }
                    break;
                }
                case 5: {
                    if (auto d2 = dist2_pe(v1, v2, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                            PEM[no] = pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPE[0], 1);
                            PE[no] = pair3_t{eiInds[1], ejInds[0], ejInds[1]};
                        }
                    }
                    break;
                }
                case 6: {
                    if (auto d2 = dist2_pe(v2, v0, v1); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                            PEM[no] = pair4_t{ejInds[0], ejInds[1], eiInds[0], eiInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPE[0], 1);
                            PE[no] = pair3_t{ejInds[0], eiInds[0], eiInds[1]};
                        }
                    }
                    break;
                }
                case 7: {
                    if (auto d2 = dist2_pe(v3, v0, v1); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                            PEM[no] = pair4_t{ejInds[1], ejInds[0], eiInds[0], eiInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nPE[0], 1);
                            PE[no] = pair3_t{ejInds[1], eiInds[0], eiInds[1]};
                        }
                    }
                    break;
                }
                case 8: {
                    if (auto d2 = dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
#if s_enableMollification
                        if (mollify) {
                            auto no = atomic_add(exec_cuda, &nEEM[0], 1);
                            EEM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                            break;
                        }
#endif
                        {
                            auto no = atomic_add(exec_cuda, &nEE[0], 1);
                            EE[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        }
                    }
                    break;
                }
                default: break;
                }
            });
        });
}
void IPCSystem::precomputeFrictions(zs::CudaExecutionPolicy &pol, T dHat, T xi) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    nFPP.setVal(0);
    nFPE.setVal(0);
    nFPT.setVal(0);
    nFEE.setVal(0);
    if (s_enableContact) {
        if (s_enableSelfFriction) {
            nFPP = nPP;
            nFPE = nPE;
            nFPT = nPT;
            nFEE = nEE;

            auto numFPP = nFPP.getVal();
            pol(range(numFPP),
                [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP), PP = proxy<space>(PP),
                 FPP = proxy<space>(FPP), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fppi) mutable {
                    auto fpp = PP[fppi];
                    FPP[fppi] = fpp;
                    auto x0 = vtemp.pack<3>("xn", fpp[0]);
                    auto x1 = vtemp.pack<3>("xn", fpp[1]);
                    auto dist2 = dist2_pp(x0, x1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPP("fn", fppi) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPP.tuple<6>("basis", fppi) = point_point_tangent_basis(x0, x1);
                });
            auto numFPE = nFPE.getVal();
            pol(range(numFPE),
                [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE), PE = proxy<space>(PE),
                 FPE = proxy<space>(FPE), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fpei) mutable {
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
            auto numFPT = nFPT.getVal();
            pol(range(numFPT),
                [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT), PT = proxy<space>(PT),
                 FPT = proxy<space>(FPT), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fpti) mutable {
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
            auto numFEE = nFEE.getVal();
            pol(range(numFEE),
                [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE), EE = proxy<space>(EE),
                 FEE = proxy<space>(FEE), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int feei) mutable {
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
    if (s_enableGround) {
        for (auto &primHandle : prims) {
            if (primHandle.isBoundary()) // skip soft boundary
                continue;
            const auto &svs = primHandle.getSurfVerts();
            pol(range(svs.size()),
                [vtemp = proxy<space>({}, vtemp), svs = proxy<space>({}, svs),
                 svtemp = proxy<space>({}, primHandle.svtemp), kappa = kappa, xi2 = xi * xi, activeGap2,
                 gn = s_groundNormal, svOffset = primHandle.svOffset] ZS_LAMBDA(int svi) mutable {
                    const auto vi = reinterpret_bits<int>(svs("inds", svi)) + svOffset;
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

void IPCSystem::project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    // projection
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), projectDBC = projectDBC, tag] ZS_LAMBDA(int vi) mutable {
        int BCfixed = vtemp("BCfixed", vi);
        if (projectDBC || (!projectDBC && BCfixed)) {
            int BCorder = vtemp("BCorder", vi);
            for (int d = 0; d != BCorder; ++d)
                vtemp(tag, d, vi) = 0;
        }
    });
}

void IPCSystem::newtonKrylov(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// optimizer
    for (int newtonIter = 0; newtonIter != PNCap; ++newtonIter) {
        // check constraints
        if (!BCsatisfied) {
            computeConstraints(pol);
            auto cr = constraintResidual(pol, true);
            if (cr < s_constraint_residual) {
                fmt::print("satisfied cons res [{}] at newton iter [{}]\n", cr, newtonIter);
                projectDBC = true;
                BCsatisfied = true;
            }
            fmt::print(fg(fmt::color::alice_blue), "newton iter {} cons residual: {}\n", newtonIter, cr);
        }
        // PRECOMPUTE
        if (s_enableContact) {
            findCollisionConstraints(pol, dHat, xi);
        }
        if (s_enableFriction)
            if (fricMu != 0) {
                precomputeFrictions(pol, dHat, xi);
            }
        // GRAD, HESS, P
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple<9>("P", i) = mat3::zeros();
            vtemp.tuple<3>("grad", i) = vec3::zeros();
        });
        computeInertialAndGravityPotentialGradient(pol);
        computeElasticGradientAndHessian(pol);
        if (s_enableGround)
            computeBoundaryBarrierGradientAndHessian(pol);
        if (s_enableContact) {
            computeBarrierGradientAndHessian(pol);
            if (s_enableFriction)
                if (fricMu != 0) {
                    computeFrictionBarrierGradientAndHessian(pol);
                }
        }
        // ROTATE GRAD, APPLY CONSTRAINTS, PROJ GRADIENT
        pol(zs::range(coOffset), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            auto grad = vtemp.pack<3, 3>("BCbasis", i).transpose() * vtemp.pack<3>("grad", i);
            vtemp.tuple<3>("grad", i) = grad;
        });
        if (!BCsatisfied) {
            // grad
            pol(zs::range(numDofs),
                [vtemp = proxy<space>({}, vtemp), boundaryKappa = boundaryKappa] ZS_LAMBDA(int i) mutable {
                    // computed during the previous constraint residual check
                    auto cons = vtemp.pack<3>("cons", i);
                    auto w = vtemp("ws", i);
                    vtemp.tuple<3>("grad", i) =
                        vtemp.pack<3>("grad", i) + w * vtemp.pack<3>("lambda", i) - boundaryKappa * w * cons;
                    int BCfixed = vtemp("BCfixed", i);
                    if (!BCfixed) {
                        int BCorder = vtemp("BCorder", i);
                        for (int d = 0; d != BCorder; ++d)
                            vtemp("P", 4 * d, i) += boundaryKappa * w;
                    }
                });
            // hess (embedded in multiply)
        }
        project(pol, "grad");
        // PREPARE P
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            auto mat = vtemp.pack<3, 3>("P", i);
            if (zs::abs(zs::determinant(mat)) > limits<T>::epsilon() * 10)
                vtemp.tuple<9>("P", i) = inverse(mat);
            else
                vtemp.tuple<9>("P", i) = mat3::identity();
        });
        // CG SOLVE
        // ROTATE BACK
        // CHECK PN CONDITION
        // LINESEARCH
        // UPDATE RULE
    }
}

struct AdvanceIPCSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto A = get_input<IPCSystem>("ZSIPCSystem");

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);
        for (int subi = 0; subi != nSubsteps; ++subi) {
            A->advanceSubstep(cudaPol, (typename IPCSystem::T)1 / nSubsteps);

            int numFricSolve = A->s_enableFriction ? 2 : 1;
        for_fric:
            A->newtonKrylov(cudaPol);
            if (--numFricSolve > 0)
                goto for_fric;

            A->updateVelocities(cudaPol);
        }
        // update velocity and positions
        A->writebackPositionsAndVelocities(cudaPol);

        set_output("ZSIPCSystem", A);
    }
};

ZENDEFNODE(AdvanceIPCSystem, {{
                                  "ZSIPCSystem",
                                  {"int", "num_substeps", "1"},
                                  {"float", "dt", "0.01"},
                              },
                              {"ZSIPCSystem"},
                              {},
                              {"FEM"}});
} // namespace zeno