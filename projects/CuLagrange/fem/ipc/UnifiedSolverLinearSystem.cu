#include "UnifiedSolver.cuh"
#include "Utils.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/math/DihedralAngle.hpp"
#include "zensim/types/SmallVector.hpp"

namespace zeno {

/// inertia
void UnifiedIPCSystem::computeInertialAndGravityPotentialGradient(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // inertial
    cudaPol(zs::range(coOffset), [tempI = proxy<space>({}, tempI),
                                  vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        auto m = vtemp("ws", i);
        vtemp.tuple(dim_c<3>, "grad", i) =
            vtemp.pack(dim_c<3>, "grad", i) - m * (vtemp.pack(dim_c<3>, "xn", i) - vtemp.pack(dim_c<3>, "xtilde", i));

        int BCorder[1] = {(int)vtemp("BCorder", i)};
        auto M = mat3::identity() * m;
        for (int d = 0; d != BCorder[0]; ++d)
            M.val(d * 4) = 0;
        tempI.tuple(dim_c<9>, "Hi", i) = M;
        // prepare preconditioner
        for (int d = 0; d != 3; ++d)
            vtemp("P", d * 3 + d, i) += M(d, d);
    });
    if (vtemp.hasProperty("extf")) {
        cudaPol(zs::range(coOffset), [vtemp = proxy<space>({}, vtemp), dt = dt] ZS_LAMBDA(int vi) mutable {
            int BCorder = vtemp("BCorder", vi);
            if (BCorder == 0) // BCsoft == 0 &&
                vtemp.tuple(dim_c<3>, "grad", vi) =
                    vtemp.pack(dim_c<3>, "grad", vi) + vtemp.pack(dim_c<3>, "extf", vi) * dt * dt;
        });
    }
}
void UnifiedIPCSystem::computeInertialPotentialGradient(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // inertial
    cudaPol(zs::range(coOffset), [vtemp = proxy<space>({}, vtemp), gTag, dt = dt] ZS_LAMBDA(int i) mutable {
        auto m = vtemp("ws", i);
        vtemp.tuple(dim_c<3>, gTag, i) =
            vtemp.pack(dim_c<3>, gTag, i) - m * (vtemp.pack(dim_c<3>, "xn", i) - vtemp.pack(dim_c<3>, "xtilde", i));
    });
}

/// @note writes to sparse matrix with fixed topo
template <typename Model, typename SpmatH>
void computeElasticGradientAndHessianImpl(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag,
                                          typename UnifiedIPCSystem::dtiles_t &vtemp,
                                          typename UnifiedIPCSystem::PrimitiveHandle &primHandle, const Model &model,
                                          typename UnifiedIPCSystem::T dt, SpmatH &spmat) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename UnifiedIPCSystem::mat3;
    using vec3 = typename UnifiedIPCSystem::vec3;
    using T = typename UnifiedIPCSystem::T;
    if (primHandle.category == ZenoParticles::curve) {
        if (primHandle.isBoundary() && !primHandle.isAuxiliary())
            return;
        /// ref: Fast Simulation of Mass-Spring Systems
        /// credits: Tiantian Liu
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = proxy<space>({}, vtemp), spmat = view<space>(spmat),
                 eles = proxy<space>({}, primHandle.getEles()), model, gTag, dt = dt,
                 vOffset = primHandle.vOffset] __device__(int ei) mutable {
                    auto inds = eles.pack(dim_c<2>, "inds", ei, int_c) + vOffset;
                    int BCorder[2];
                    for (int i = 0; i != 2; ++i) {
                        BCorder[i] = vtemp("BCorder", inds[i]);
                    }

                    if (BCorder[0] == 3 && BCorder[1] == 3)
                        return;

                    auto vole = eles("vol", ei);
                    auto k = eles("k", ei);
                    auto rl = eles("rl", ei);

                    vec3 xs[2] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1])};
                    auto xij = xs[1] - xs[0];
                    auto lij = xij.norm();
                    auto dij = xij / lij;
                    auto gij = k * (lij - rl) * dij;

#if 0
                    // gradient
                    auto vfdt2 = gij * (dt * dt) * vole;
                    for (int d = 0; d != 3; ++d) {
                        atomic_add(exec_cuda, &vtemp(gTag, d, inds[0]), (T)vfdt2(d));
                        atomic_add(exec_cuda, &vtemp(gTag, d, inds[1]), (T)-vfdt2(d));
                    }
#endif

                    auto H = zs::vec<T, 6, 6>::zeros();
                    auto K = k * (mat3::identity() - rl / lij * (mat3::identity() - dyadic_prod(dij, dij)));
                    // make_pd(K);  // symmetric semi-definite positive, not
                    // necessary

                    for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j) {
                            H(i, j) = K(i, j);
                            H(i, 3 + j) = -K(i, j);
                            H(3 + i, j) = -K(i, j);
                            H(3 + i, 3 + j) = K(i, j);
                        }
                    H *= dt * dt * vole;

                    for (int vi = 0; vi != 2; ++vi) {
                        auto i = inds[vi];
                        for (int vj = 0; vj != 2; ++vj) {
                            auto j = inds[vj];
                            if (i > j)
                                continue;
                            auto loc = spmat.locate(i, j, true_c);
                            auto &mat = spmat._vals[loc];
                            for (int r = 0; r != 3; ++r)
                                for (int c = 0; c != 3; ++c) {
                                    atomic_add(exec_cuda, &mat(r, c), H(vi * 3 + r, vj * 3 + c));
                                }
                        }
                    }
                });
    } else if (primHandle.category == ZenoParticles::surface) {
        if (primHandle.isBoundary())
            return;
        cudaPol(zs::range(primHandle.getEles().size()), [vtemp = proxy<space>({}, vtemp), spmat = view<space>(spmat),
                                                         eles = proxy<space>({}, primHandle.getEles()), model, gTag,
                                                         dt = dt,
                                                         vOffset = primHandle.vOffset] __device__(int ei) mutable {
            auto IB = eles.pack(dim_c<2, 2>, "IB", ei);
            auto inds = eles.pack(dim_c<3>, "inds", ei, int_c) + vOffset;
            auto vole = eles("vol", ei);
            vec3 xs[3] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1]),
                          vtemp.pack(dim_c<3>, "xn", inds[2])};
            auto x1x0 = xs[1] - xs[0];
            auto x2x0 = xs[2] - xs[0];

            int BCorder[3];
            for (int i = 0; i != 3; ++i) {
                BCorder[i] = vtemp("BCorder", inds[i]);
            }
            zs::vec<T, 9, 9> H;
            if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3) {
                return;
            }

            zs::vec<T, 3, 2> Ds{x1x0[0], x2x0[0], x1x0[1], x2x0[1], x1x0[2], x2x0[2]};
            auto F = Ds * IB;

            auto dFdX = dFdXMatrix(IB, wrapv<3>{});
            auto dFdXT = dFdX.transpose();
            auto f0 = col(F, 0);
            auto f1 = col(F, 1);
#if 0
                    auto f0Norm = zs::sqrt(f0.l2NormSqr());
                    auto f1Norm = zs::sqrt(f1.l2NormSqr());
                    auto f0Tf1 = f0.dot(f1);
                    zs::vec<T, 3, 2> Pstretch, Pshear;
                    for (int d = 0; d != 3; ++d) {
                        Pstretch(d, 0) = 2 * (1 - 1 / f0Norm) * F(d, 0);
                        Pstretch(d, 1) = 2 * (1 - 1 / f1Norm) * F(d, 1);
                        Pshear(d, 0) = 2 * f0Tf1 * f1(d);
                        Pshear(d, 1) = 2 * f0Tf1 * f0(d);
                    }
                    auto vecP = flatten(model.mu * Pstretch + (model.mu * 0.3) * Pshear);
                    auto vfdt2 = -vole * (dFdXT * vecP) * (dt * dt);

                    for (int i = 0; i != 3; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp(gTag, d, vi), (T)vfdt2(i * 3 + d));
                    }
#endif
            /// ref: A Finite Element Formulation of Baraff-Witkin Cloth
            // suggested by huang kemeng
            auto stretchHessian = [&F, &f0, &f1, &model]() {
                auto H = zs::vec<T, 6, 6>::zeros();
                // const zs::vec<T, 2> u{1, 0};
                // const zs::vec<T, 2> v{0, 1};
                const T I5u = f0.l2NormSqr(); // Fu
                const T I5v = f1.l2NormSqr(); // Fv
                const T invSqrtI5u = (T)1 / zs::sqrt(I5u);
                const T invSqrtI5v = (T)1 / zs::sqrt(I5v);

                H(0, 0) = H(1, 1) = H(2, 2) = zs::max(1 - invSqrtI5u, (T)0);
                H(3, 3) = H(4, 4) = H(5, 5) = zs::max(1 - invSqrtI5v, (T)0);

                const auto fu = f0.normalized();
                const T uCoeff = (1 - invSqrtI5u >= 0) ? invSqrtI5u : (T)1;
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j)
                        H(i, j) += uCoeff * fu(i) * fu(j);

                const auto fv = f1.normalized();
                const T vCoeff = (1 - invSqrtI5v >= 0) ? invSqrtI5v : (T)1;
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j)
                        H(3 + i, 3 + j) += vCoeff * fv(i) * fv(j);

                H *= model.mu;
                return H;
            };
            auto shearHessian = [&F, &f0, &f1, &model]() {
                using mat6 = zs::vec<T, 6, 6>;
                auto H = mat6::zeros();
                // const zs::vec<T, 2> u{1, 0};
                // const zs::vec<T, 2> v{0, 1};
                const T I6 = f0.dot(f1);
                const T signI6 = I6 >= 0 ? 1 : -1;

                H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) = (T)1;

                // F * | 0  1 |
                //     | 1  0 |
                // =
                // | F01 F00 |
                // | F11 F10 |
                // | F21 F20 |
                // const auto g_ = F * (dyadic_prod(u, v) + dyadic_prod(v, u));
                zs::vec<T, 6> g{F(0, 1), F(1, 1), F(2, 1), F(0, 0), F(1, 0), F(2, 0)};
#if 0
                        for (int j = 0, offset = 0; j != 2; ++j) {
                            for (int i = 0; i != 3; ++i)
                                g(offset++) = g_(i, j);
                        }
#endif

                const T I2 = F.l2NormSqr();
                const T lambda0 = (T)0.5 * (I2 + zs::sqrt(I2 * I2 + (T)12 * I6 * I6));

                const zs::vec<T, 6> q0 = (I6 * H * g + lambda0 * g).normalized();

                auto t = 0.5 * (mat6::identity() + signI6 * H);

                const zs::vec<T, 6> Tq = t * q0;
                const auto normTq = Tq.l2NormSqr();

                mat6 dPdF = zs::abs(I6) * (t - (dyadic_prod(Tq, Tq) / normTq)) + lambda0 * (dyadic_prod(q0, q0));
                dPdF *= (model.mu * 0.3);
                return dPdF;
            };
            auto He = stretchHessian() + shearHessian();
            H = dFdXT * He * dFdX;
            H *= dt * dt * vole;

            for (int vi = 0; vi != 3; ++vi) {
                auto i = inds[vi];
                for (int vj = 0; vj != 3; ++vj) {
                    auto j = inds[vj];
                    if (i > j)
                        continue;
                    auto loc = spmat.locate(i, j, true_c);
                    auto &mat = spmat._vals[loc];
                    for (int r = 0; r != 3; ++r)
                        for (int c = 0; c != 3; ++c) {
                            atomic_add(exec_cuda, &mat(r, c), H(vi * 3 + r, vj * 3 + c));
                        }
                }
            }
        });
    } else if (primHandle.category == ZenoParticles::tet)
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = proxy<space>({}, vtemp), spmat = view<space>(spmat),
                 eles = proxy<space>({}, primHandle.getEles()), model, gTag, dt = dt,
                 vOffset = primHandle.vOffset] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                    auto inds = eles.pack(dim_c<4>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[4] = {vtemp.pack<3>("xn", inds[0]), vtemp.pack<3>("xn", inds[1]),
                                  vtemp.pack<3>("xn", inds[2]), vtemp.pack<3>("xn", inds[3])};

                    int BCorder[4];
                    for (int i = 0; i != 4; ++i) {
                        BCorder[i] = vtemp("BCorder", inds[i]);
                    }
                    zs::vec<T, 12, 12> H;
                    if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 && BCorder[3] == 3) {
                        return;
                    }
                    mat3 F{};
                    {
                        auto x1x0 = xs[1] - xs[0];
                        auto x2x0 = xs[2] - xs[0];
                        auto x3x0 = xs[3] - xs[0];
                        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1], x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                        F = Ds * IB;
                    }
                    auto dFdX = dFdXMatrix(IB);
                    auto dFdXT = dFdX.transpose();
#if 0
                    auto P = model.first_piola(F);
                    auto vecP = flatten(P);
                    auto vfdt2 = -vole * (dFdXT * vecP) * dt * dt;

                    for (int i = 0; i != 4; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp(gTag, d, vi), (T)vfdt2(i * 3 + d));
                    }
#endif

                    auto Hq = model.first_piola_derivative(F, true_c);
                    H = dFdXT * Hq * dFdX * vole * dt * dt;

                    for (int vi = 0; vi != 4; ++vi) {
                        auto i = inds[vi];
                        for (int vj = 0; vj != 4; ++vj) {
                            auto j = inds[vj];
                            if (i > j)
                                continue;
                            auto loc = spmat.locate(i, j, true_c);
                            auto &mat = spmat._vals[loc];
                            for (int r = 0; r != 3; ++r)
                                for (int c = 0; c != 3; ++c) {
                                    atomic_add(exec_cuda, &mat(r, c), H(vi * 3 + r, vj * 3 + c));
                                }
                        }
                    }
                });
}
/// @brief inertial, kinetic, external force, elasticity, bending, boundary motion, ground collision
void UnifiedIPCSystem::updateInherentHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    auto &spmat = linsys.spmat;
    /// clear entry values
    spmat._vals.reset(0);
    /// @note inertial, gravity, external force, boundary motion
    cudaPol(zs::range(coOffset),
            [spmat = view<space>(spmat), tempI = proxy<space>({}, tempI), vtemp = proxy<space>({}, vtemp),
             boundaryKappa = boundaryKappa] ZS_LAMBDA(int i) mutable {
                using mat3 = RM_CVREF_T(spmat)::value_type;
                auto m = vtemp("ws", i);
                int BCorder[1] = {(int)vtemp("BCorder", i)};
                auto Hi = mat3::identity() * m;
                for (int d = 0; d != BCorder[0]; ++d)
                    Hi.val(d * 4) = 0;
                auto loc = spmat.locate(i, i, true_c);
                auto &mat = spmat._vals[loc];
                for (int r = 0; r != 3; ++r)
                    for (int c = 0; c != 3; ++c)
                        mat(r, c) = Hi(r, c);
            });
    if (!BCsatisfied) {
        cudaPol(zs::range(numDofs), [spmat = view<space>(spmat), vtemp = proxy<space>({}, vtemp),
                                     boundaryKappa = boundaryKappa] ZS_LAMBDA(int vi) mutable {
            auto w = vtemp("ws", vi);
            int BCfixed = vtemp("BCfixed", vi);
            if (!BCfixed) {
                auto loc = spmat.locate(vi, vi, true_c);
                auto &mat = spmat._vals[loc];
                int BCorder = vtemp("BCorder", vi);
                for (int d = 0; d != BCorder; ++d)
                    mat.val(d * 4) += boundaryKappa * w;
            }
        });
    }
#if 0
    /// deprecated
    cudaPol(zs::range(numDofs * 3), [spmat = view<space>(spmat), vtemp = proxy<space>({}, vtemp),
                                     boundaryKappa = boundaryKappa] ZS_LAMBDA(int i) mutable {
        auto dofi = i / 3;
        int d = i % 3;
        int BCfixed = vtemp("BCfixed", dofi);
        if (!BCfixed) {
            auto m = vtemp("ws", dofi);
            int BCorder = vtemp("BCorder", dofi);
            auto loc = spmat.locate(dofi, dofi, true_c);
            auto &mat = spmat._vals[loc];

            auto val = d < BCorder ? boundaryKappa * m : m;
            mat(d, d) = val;
        }
    });
#endif

    /// @note ground collision
    if (enableGround) {
        for (auto &primHandle : prims) {
            if (primHandle.isBoundary()) // skip soft boundary
                continue;
            const auto &svs = primHandle.getSurfVerts();
            cudaPol(range(svs.size()),
                    [vtemp = proxy<space>({}, vtemp), svtemp = proxy<space>({}, primHandle.svtemp),
                     spmat = view<space>(linsys.spmat), svs = proxy<space>({}, svs), gn = s_groundNormal,
                     dHat2 = dHat * dHat, kappa = kappa, svOffset = primHandle.svOffset] ZS_LAMBDA(int svi) mutable {
                        const auto vi = svs("inds", svi, int_c) + svOffset;
                        auto x = vtemp.pack<3>("xn", vi);
                        auto dist = gn.dot(x);
                        auto dist2 = dist * dist;
                        auto t = dist2 - dHat2;
                        auto g_b = t * zs::log(dist2 / dHat2) * -2 - (t * t) / dist2;
                        auto H_b = (zs::log(dist2 / dHat2) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
#if 0
                if (dist2 < dHat2) {
                    auto grad = -gn * (kappa * g_b * 2 * dist);
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_cuda, &vtemp("grad", d, vi), grad(d));
                }
#endif

                        auto param = 4 * H_b * dist2 + 2 * g_b;
                        auto hess = mat3::zeros();
                        if (dist2 < dHat2 && param > 0) {
                            auto nn = dyadic_prod(gn, gn);
                            hess = (kappa * param) * nn;
                        }

                        // make_pd(hess);
                        auto loc = spmat.locate(vi, vi, true_c);
                        auto &mat = spmat._vals[loc];
                        for (int r = 0; r != 3; ++r) {
                            for (int c = 0; c != 3; ++c) {
                                mat(r, c) += hess(r, c);
                            }
                        }
                    });

            if (s_enableFriction)
                if (fricMu != 0) {
                    cudaPol(range(svs.size()),
                            [vtemp = proxy<space>({}, vtemp), svtemp = proxy<space>({}, primHandle.svtemp),
                             spmat = view<space>(linsys.spmat), svs = proxy<space>({}, svs), epsvh = epsv * dt,
                             gn = s_groundNormal, fricMu = fricMu,
                             svOffset = primHandle.svOffset] ZS_LAMBDA(int svi) mutable {
                                const auto vi = svs("inds", svi, int_c) + svOffset;
                                auto dx = vtemp.pack<3>("xn", vi) - vtemp.pack<3>("xhat", vi);
                                auto fn = svtemp("fn", svi);
                                if (fn == 0) {
                                    return;
                                }
                                auto coeff = fn * fricMu;
                                auto relDX = dx - gn.dot(dx) * gn;
                                auto relDXNorm2 = relDX.l2NormSqr();
                                auto relDXNorm = zs::sqrt(relDXNorm2);
#if 0
                    vec3 grad{};
                    if (relDXNorm2 > epsvh * epsvh)
                        grad = -relDX * (coeff / relDXNorm);
                    else
                        grad = -relDX * (coeff / epsvh);
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_cuda, &vtemp("grad", d, vi), grad(d));
#endif

                                auto hess = mat3::zeros();
                                if (relDXNorm2 > epsvh * epsvh) {
                                    zs::vec<T, 2, 2> mat{
                                        relDX[0] * relDX[0] * -coeff / relDXNorm2 / relDXNorm + coeff / relDXNorm,
                                        relDX[0] * relDX[2] * -coeff / relDXNorm2 / relDXNorm,
                                        relDX[0] * relDX[2] * -coeff / relDXNorm2 / relDXNorm,
                                        relDX[2] * relDX[2] * -coeff / relDXNorm2 / relDXNorm + coeff / relDXNorm};
                                    make_pd(mat);
                                    hess(0, 0) = mat(0, 0);
                                    hess(0, 2) = mat(0, 1);
                                    hess(2, 0) = mat(1, 0);
                                    hess(2, 2) = mat(1, 1);
                                } else {
                                    hess(0, 0) = coeff / epsvh;
                                    hess(2, 2) = coeff / epsvh;
                                }
                                auto loc = spmat.locate(vi, vi, true_c);
                                auto &mat = spmat._vals[loc];
                                for (int r = 0; r != 3; ++r) {
                                    for (int c = 0; c != 3; ++c) {
                                        mat(r, c) += hess(r, c);
                                    }
                                }
                            });
                }
        }
    }

    /// @note bending
    for (auto &primHandle : prims) {
        if (primHandle.hasBendingConstraints()) {
            auto &bedges = *primHandle.bendingEdgesPtr;
            cudaPol(range(bedges.size()), [bedges = view<space>({}, bedges), spmat = view<space>(linsys.spmat),
                                           vtemp = view<space>({}, vtemp), dt2 = dt * dt,
                                           vOffset = primHandle.vOffset] __device__(int i) mutable {
                auto stcl = bedges.pack(dim_c<4>, "inds", i, int_c) + vOffset;

                int BCorder[4];
#pragma unroll
                for (int i = 0; i != 4; ++i)
                    BCorder[i] = vtemp("BCorder", stcl[i]);
                if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 && BCorder[3] == 3)
                    return;
                auto e = bedges("e", i);
                auto h = bedges("h", i);
                auto k = bedges("k", i);
                auto ra = bedges("ra", i);
                auto x0 = vtemp.pack(dim_c<3>, "xn", stcl[0]);
                auto x1 = vtemp.pack(dim_c<3>, "xn", stcl[1]);
                auto x2 = vtemp.pack(dim_c<3>, "xn", stcl[2]);
                auto x3 = vtemp.pack(dim_c<3>, "xn", stcl[3]);
                auto theta = dihedral_angle(x0, x1, x2, x3);

                auto localGrad = dihedral_angle_gradient(x0, x1, x2, x3);
#if 0
                auto grad = -localGrad * dt2 * k * 2 * (theta - ra) * e / h;
                for (int j = 0; j != 4; ++j)
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_cuda, &vtemp("grad", d, stcl[j]), grad(j * 3 + d));
#endif
                // rotate and project
                auto H = (dihedral_angle_hessian(x0, x1, x2, x3) * (theta - ra) + dyadic_prod(localGrad, localGrad)) *
                         k * 2 * e / h;
                make_pd(H);
                H *= dt2;

                // 12 * 12 = 16 * 9
                for (int vi = 0; vi < 4; ++vi) {
                    auto i = stcl[vi];
                    for (int vj = 0; vj < 4; ++vj) {
                        auto j = stcl[vj];
                        if (i > j)
                            continue;
                        auto loc = spmat.locate(i, j, true_c);
                        auto &mat = spmat._vals[loc];
                        for (int r = 0; r != 3; ++r) {
                            for (int c = 0; c != 3; ++c) {
                                atomic_add(exec_cuda, &mat(r, c), H(vi * 3 + r, vj * 3 + c));
                            }
                        }
                    }
                }
            });
        }
    }
    /// @note elasticity
    for (auto &primHandle : prims) {
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, gTag, vtemp, primHandle, elasticModel, dt, spmat);
        })(primHandle.getModels().getElasticModel());
    }
    for (auto &primHandle : auxPrims) {
        using ModelT = RM_CVREF_T(primHandle.getModels().getElasticModel());
        const ModelT &model = primHandle.modelsPtr ? primHandle.getModels().getElasticModel() : ModelT{};
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, gTag, vtemp, primHandle, elasticModel, dt, spmat);
        })(model);
    }
}

/// @note currently directly borrow results from computeBarrierGradientAndHessian and computeFrictionBarrierGradientAndHessian
void UnifiedIPCSystem::updateDynamicHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto &hess2 = linsys.hess2;
    auto &hess3 = linsys.hess3;
    auto &hess4 = linsys.hess4;
    hess2.reset(false, 0); // overwrite style
    hess3.reset(false, 0);
    hess4.reset(false, 0);
    if (enableContact) {
        updateBarrierGradientAndHessian(pol, gTag);

        if (s_enableFriction) {
            if (s_enableSelfFriction) {
                if (fricMu != 0) {
                    updateFrictionBarrierGradientAndHessian(pol, gTag);
                } //
            }     // enable self friction, fricmu
        }         // enable friction
    }             // enable contact
}

/// elasticity
template <typename Model>
void computeElasticGradientAndHessianImpl(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag,
                                          typename UnifiedIPCSystem::dtiles_t &vtemp,
                                          typename UnifiedIPCSystem::PrimitiveHandle &primHandle, const Model &model,
                                          typename UnifiedIPCSystem::T dt, bool projectDBC, bool includeHessian) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename UnifiedIPCSystem::mat3;
    using vec3 = typename UnifiedIPCSystem::vec3;
    using T = typename UnifiedIPCSystem::T;
    if (primHandle.category == ZenoParticles::curve) {
        if (primHandle.isBoundary() && !primHandle.isAuxiliary())
            return;
        /// ref: Fast Simulation of Mass-Spring Systems
        /// credits: Tiantian Liu
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, primHandle.etemp),
                 eles = proxy<space>({}, primHandle.getEles()), model, gTag, dt = dt, projectDBC = projectDBC,
                 vOffset = primHandle.vOffset, includeHessian,
                 n = primHandle.getEles().size()] __device__(int ei) mutable {
                    auto inds = eles.pack(dim_c<2>, "inds", ei, int_c) + vOffset;
                    int BCorder[2];
                    for (int i = 0; i != 2; ++i) {
                        BCorder[i] = vtemp("BCorder", inds[i]);
                    }

                    if (BCorder[0] == 3 && BCorder[1] == 3) {
                        etemp.tuple<6 * 6>("He", ei) = zs::vec<T, 6, 6>::zeros();
                        return;
                    }

                    auto vole = eles("vol", ei);
                    auto k = eles("k", ei);
                    auto rl = eles("rl", ei);

                    vec3 xs[2] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1])};
                    auto xij = xs[1] - xs[0];
                    auto lij = xij.norm();
                    auto dij = xij / lij;
                    auto gij = k * (lij - rl) * dij;

                    // gradient
                    auto vfdt2 = gij * (dt * dt) * vole;
                    for (int d = 0; d != 3; ++d) {
                        atomic_add(exec_cuda, &vtemp(gTag, d, inds[0]), (T)vfdt2(d));
                        atomic_add(exec_cuda, &vtemp(gTag, d, inds[1]), (T)-vfdt2(d));
                    }

                    if (!includeHessian)
                        return;
                    auto H = zs::vec<T, 6, 6>::zeros();
                    auto K = k * (mat3::identity() - rl / lij * (mat3::identity() - dyadic_prod(dij, dij)));
                    // make_pd(K);  // symmetric semi-definite positive, not
                    // necessary

                    for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j) {
                            H(i, j) = K(i, j);
                            H(i, 3 + j) = -K(i, j);
                            H(3 + i, j) = -K(i, j);
                            H(3 + i, 3 + j) = K(i, j);
                        }
                    H *= dt * dt * vole;

                    // rotate and project
                    etemp.tuple(dim_c<6, 6>, "He", ei) = H;
                    for (int vi = 0; vi != 2; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
                });
    } else if (primHandle.category == ZenoParticles::surface) {
        if (primHandle.isBoundary())
            return;
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, primHandle.etemp),
                 eles = proxy<space>({}, primHandle.getEles()), model, gTag, dt = dt, projectDBC = projectDBC,
                 vOffset = primHandle.vOffset, includeHessian] __device__(int ei) mutable {
                    auto IB = eles.template pack<2, 2>("IB", ei);
                    auto inds = eles.pack(dim_c<3>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[3] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1]),
                                  vtemp.pack(dim_c<3>, "xn", inds[2])};
                    auto x1x0 = xs[1] - xs[0];
                    auto x2x0 = xs[2] - xs[0];

                    int BCorder[3];
                    for (int i = 0; i != 3; ++i) {
                        BCorder[i] = vtemp("BCorder", inds[i]);
                    }
                    zs::vec<T, 9, 9> H;
                    if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3) {
                        etemp.tuple(dim_c<9, 9>, "He", ei) = H.zeros();
                        return;
                    }

                    zs::vec<T, 3, 2> Ds{x1x0[0], x2x0[0], x1x0[1], x2x0[1], x1x0[2], x2x0[2]};
                    auto F = Ds * IB;

                    auto dFdX = dFdXMatrix(IB, wrapv<3>{});
                    auto dFdXT = dFdX.transpose();
                    auto f0 = col(F, 0);
                    auto f1 = col(F, 1);
                    auto f0Norm = zs::sqrt(f0.l2NormSqr());
                    auto f1Norm = zs::sqrt(f1.l2NormSqr());
                    auto f0Tf1 = f0.dot(f1);
                    zs::vec<T, 3, 2> Pstretch, Pshear;
                    for (int d = 0; d != 3; ++d) {
                        Pstretch(d, 0) = 2 * (1 - 1 / f0Norm) * F(d, 0);
                        Pstretch(d, 1) = 2 * (1 - 1 / f1Norm) * F(d, 1);
                        Pshear(d, 0) = 2 * f0Tf1 * f1(d);
                        Pshear(d, 1) = 2 * f0Tf1 * f0(d);
                    }
                    auto vecP = flatten(model.mu * Pstretch + (model.mu * 0.3) * Pshear);
                    auto vfdt2 = -vole * (dFdXT * vecP) * (dt * dt);

                    for (int i = 0; i != 3; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp(gTag, d, vi), (T)vfdt2(i * 3 + d));
                    }

                    if (!includeHessian)
                        return;
                    /// ref: A Finite Element Formulation of Baraff-Witkin Cloth
                    // suggested by huang kemeng
                    auto stretchHessian = [&F, &model]() {
                        auto H = zs::vec<T, 6, 6>::zeros();
                        const zs::vec<T, 2> u{1, 0};
                        const zs::vec<T, 2> v{0, 1};
                        const T I5u = (F * u).l2NormSqr();
                        const T I5v = (F * v).l2NormSqr();
                        const T invSqrtI5u = (T)1 / zs::sqrt(I5u);
                        const T invSqrtI5v = (T)1 / zs::sqrt(I5v);

                        H(0, 0) = H(1, 1) = H(2, 2) = zs::max(1 - invSqrtI5u, (T)0);
                        H(3, 3) = H(4, 4) = H(5, 5) = zs::max(1 - invSqrtI5v, (T)0);

                        const auto fu = col(F, 0).normalized();
                        const T uCoeff = (1 - invSqrtI5u >= 0) ? invSqrtI5u : (T)1;
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j)
                                H(i, j) += uCoeff * fu(i) * fu(j);

                        const auto fv = col(F, 1).normalized();
                        const T vCoeff = (1 - invSqrtI5v >= 0) ? invSqrtI5v : (T)1;
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j)
                                H(3 + i, 3 + j) += vCoeff * fv(i) * fv(j);

                        H *= model.mu;
                        return H;
                    };
                    auto shearHessian = [&F, &model]() {
                        using mat6 = zs::vec<T, 6, 6>;
                        auto H = mat6::zeros();
                        const zs::vec<T, 2> u{1, 0};
                        const zs::vec<T, 2> v{0, 1};
                        const T I6 = (F * u).dot(F * v);
                        const T signI6 = I6 >= 0 ? 1 : -1;

                        H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) = (T)1;

                        const auto g_ = F * (dyadic_prod(u, v) + dyadic_prod(v, u));
                        zs::vec<T, 6> g{};
                        for (int j = 0, offset = 0; j != 2; ++j) {
                            for (int i = 0; i != 3; ++i)
                                g(offset++) = g_(i, j);
                        }

                        const T I2 = F.l2NormSqr();
                        const T lambda0 = (T)0.5 * (I2 + zs::sqrt(I2 * I2 + (T)12 * I6 * I6));

                        const zs::vec<T, 6> q0 = (I6 * H * g + lambda0 * g).normalized();

                        auto t = mat6::identity();
                        t = 0.5 * (t + signI6 * H);

                        const zs::vec<T, 6> Tq = t * q0;
                        const auto normTq = Tq.l2NormSqr();

                        mat6 dPdF =
                            zs::abs(I6) * (t - (dyadic_prod(Tq, Tq) / normTq)) + lambda0 * (dyadic_prod(q0, q0));
                        dPdF *= (model.mu * 0.3);
                        return dPdF;
                    };
                    auto He = stretchHessian() + shearHessian();
                    H = dFdX.transpose() * He * dFdX;
                    H *= dt * dt * vole;

                    // rotate and project
                    etemp.tuple(dim_c<9, 9>, "He", ei) = H;
                    for (int vi = 0; vi != 3; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
                });
    } else if (primHandle.category == ZenoParticles::tet)
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, primHandle.etemp),
                 eles = proxy<space>({}, primHandle.getEles()), model, gTag, dt = dt, projectDBC = projectDBC,
                 vOffset = primHandle.vOffset, includeHessian] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                    auto inds = eles.pack(dim_c<4>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[4] = {vtemp.pack<3>("xn", inds[0]), vtemp.pack<3>("xn", inds[1]),
                                  vtemp.pack<3>("xn", inds[2]), vtemp.pack<3>("xn", inds[3])};

                    int BCorder[4];
                    for (int i = 0; i != 4; ++i) {
                        BCorder[i] = vtemp("BCorder", inds[i]);
                    }
                    zs::vec<T, 12, 12> H;
                    if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 && BCorder[3] == 3) {
                        etemp.tuple<12 * 12>("He", ei) = H.zeros();
                        return;
                    }
                    mat3 F{};
                    {
                        auto x1x0 = xs[1] - xs[0];
                        auto x2x0 = xs[2] - xs[0];
                        auto x3x0 = xs[3] - xs[0];
                        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1], x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                        F = Ds * IB;
                    }
                    auto P = model.first_piola(F);
                    auto vecP = flatten(P);
                    auto dFdX = dFdXMatrix(IB);
                    auto dFdXT = dFdX.transpose();
                    auto vfdt2 = -vole * (dFdXT * vecP) * dt * dt;

                    for (int i = 0; i != 4; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp(gTag, d, vi), (T)vfdt2(i * 3 + d));
                    }

                    if (!includeHessian)
                        return;
                    auto Hq = model.first_piola_derivative(F, true_c);
                    H = dFdXT * Hq * dFdX * vole * dt * dt;

                    // rotate and project
                    etemp.tuple<12 * 12>("He", ei) = H;
                    for (int vi = 0; vi != 4; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
                });
}

void UnifiedIPCSystem::computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag,
                                                        bool includeHessian) {
    using namespace zs;
    for (auto &primHandle : prims) {
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, gTag, vtemp, primHandle, elasticModel, dt, projectDBC,
                                                 includeHessian);
        })(primHandle.getModels().getElasticModel());
    }
    for (auto &primHandle : auxPrims) {
        using ModelT = RM_CVREF_T(primHandle.getModels().getElasticModel());
        const ModelT &model = primHandle.modelsPtr ? primHandle.getModels().getElasticModel() : ModelT{};
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, gTag, vtemp, primHandle, elasticModel, dt, projectDBC,
                                                 includeHessian);
        })(model);
    }
}

void UnifiedIPCSystem::computeBendingGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag,
                                                        bool includeHessian) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    for (auto &primHandle : prims) {
        if (!primHandle.hasBendingConstraints())
            continue;
        auto &btemp = primHandle.btemp;
        auto &bedges = *primHandle.bendingEdgesPtr;
        cudaPol(range(btemp.size()), [bedges = proxy<space>({}, bedges), btemp = proxy<space>(btemp),
                                      vtemp = proxy<space>({}, vtemp), dt2 = dt * dt, projectDBC = projectDBC,
                                      vOffset = primHandle.vOffset, includeHessian] __device__(int i) mutable {
            auto stcl = bedges.pack(dim_c<4>, "inds", i, int_c) + vOffset;

            int BCorder[4];
            for (int i = 0; i != 4; ++i) {
                BCorder[i] = vtemp("BCorder", stcl[i]);
            }
            if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3 && BCorder[3] == 3) {
                btemp.tuple(dim_c<12 * 12>, 0, i) = zs::vec<T, 12, 12>::zeros();
                return;
            }

            auto e = bedges("e", i);
            auto h = bedges("h", i);
            auto k = bedges("k", i);
            auto ra = bedges("ra", i);
            auto x0 = vtemp.pack(dim_c<3>, "xn", stcl[0]);
            auto x1 = vtemp.pack(dim_c<3>, "xn", stcl[1]);
            auto x2 = vtemp.pack(dim_c<3>, "xn", stcl[2]);
            auto x3 = vtemp.pack(dim_c<3>, "xn", stcl[3]);
            auto theta = dihedral_angle(x0, x1, x2, x3);

            auto localGrad = dihedral_angle_gradient(x0, x1, x2, x3);
            auto grad = -localGrad * dt2 * k * 2 * (theta - ra) * e / h;
            for (int j = 0; j != 4; ++j)
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_cuda, &vtemp("grad", d, stcl[j]), grad(j * 3 + d));

            if (!includeHessian)
                return;

            // rotate and project
            auto H = (dihedral_angle_hessian(x0, x1, x2, x3) * (theta - ra) + dyadic_prod(localGrad, localGrad)) * k *
                     2 * e / h;
            make_pd(H);
            H *= dt2;

            btemp.tuple(dim_c<12 * 12>, 0, i) = H;
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, stcl[vi]), H(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    }
}

void UnifiedIPCSystem::computeBoundaryBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, bool includeHessian) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary()) // skip soft boundary
            continue;
        const auto &svs = primHandle.getSurfVerts();
        pol(range(svs.size()),
            [vtemp = proxy<space>({}, vtemp), svtemp = proxy<space>({}, primHandle.svtemp), svs = proxy<space>({}, svs),
             gn = s_groundNormal, dHat2 = dHat * dHat, kappa = kappa, projectDBC = projectDBC, includeHessian,
             svOffset = primHandle.svOffset] ZS_LAMBDA(int svi) mutable {
                const auto vi = svs("inds", svi, int_c) + svOffset;
                auto x = vtemp.pack<3>("xn", vi);
                auto dist = gn.dot(x);
                auto dist2 = dist * dist;
                auto t = dist2 - dHat2;
                auto g_b = t * zs::log(dist2 / dHat2) * -2 - (t * t) / dist2;
                auto H_b = (zs::log(dist2 / dHat2) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
                if (dist2 < dHat2) {
                    auto grad = -gn * (kappa * g_b * 2 * dist);
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_cuda, &vtemp("grad", d, vi), grad(d));
                }

                if (!includeHessian)
                    return;
                auto param = 4 * H_b * dist2 + 2 * g_b;
                auto hess = mat3::zeros();
                if (dist2 < dHat2 && param > 0) {
                    auto nn = dyadic_prod(gn, gn);
                    hess = (kappa * param) * nn;
                }

                // make_pd(hess);
                int BCorder[1] = {(int)vtemp("BCorder", vi)};
                svtemp.tuple<9>("H", svi) = hess;
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, vi), hess(i, j));
                    }
            });

        if (s_enableFriction)
            if (fricMu != 0) {
                pol(range(svs.size()), [vtemp = proxy<space>({}, vtemp), svtemp = proxy<space>({}, primHandle.svtemp),
                                        svs = proxy<space>({}, svs), epsvh = epsv * dt, gn = s_groundNormal,
                                        fricMu = fricMu, projectDBC = projectDBC, includeHessian,
                                        svOffset = primHandle.svOffset] ZS_LAMBDA(int svi) mutable {
                    const auto vi = svs("inds", svi, int_c) + svOffset;
                    auto dx = vtemp.pack<3>("xn", vi) - vtemp.pack<3>("xhat", vi);
                    auto fn = svtemp("fn", svi);
                    if (fn == 0) {
                        return;
                    }
                    auto coeff = fn * fricMu;
                    auto relDX = dx - gn.dot(dx) * gn;
                    auto relDXNorm2 = relDX.l2NormSqr();
                    auto relDXNorm = zs::sqrt(relDXNorm2);

                    vec3 grad{};
                    if (relDXNorm2 > epsvh * epsvh)
                        grad = -relDX * (coeff / relDXNorm);
                    else
                        grad = -relDX * (coeff / epsvh);
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_cuda, &vtemp("grad", d, vi), grad(d));

                    if (!includeHessian)
                        return;

                    auto hess = mat3::zeros();
                    if (relDXNorm2 > epsvh * epsvh) {
                        zs::vec<T, 2, 2> mat{relDX[0] * relDX[0] * -coeff / relDXNorm2 / relDXNorm + coeff / relDXNorm,
                                             relDX[0] * relDX[2] * -coeff / relDXNorm2 / relDXNorm,
                                             relDX[0] * relDX[2] * -coeff / relDXNorm2 / relDXNorm,
                                             relDX[2] * relDX[2] * -coeff / relDXNorm2 / relDXNorm + coeff / relDXNorm};
                        make_pd(mat);
                        hess(0, 0) = mat(0, 0);
                        hess(0, 2) = mat(0, 1);
                        hess(2, 0) = mat(1, 0);
                        hess(2, 2) = mat(1, 1);
                    } else {
                        hess(0, 0) = coeff / epsvh;
                        hess(2, 2) = coeff / epsvh;
                    }

                    int BCorder[1] = {(int)vtemp("BCorder", vi)};
                    svtemp.tuple(dim_c<9>, "H", svi) = svtemp.pack(dim_c<3, 3>, "H", svi) + hess;
                    for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j) {
                            atomic_add(exec_cuda, &vtemp("P", i * 3 + j, vi), hess(i, j));
                        }
                });
            }
    }
    return;
}

UnifiedIPCSystem::T UnifiedIPCSystem::infNorm(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag) {
    using namespace zs;
    using T = typename UnifiedIPCSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto &vertData = vtemp;
    auto &res = temp;
    res.resize(count_warps(vertData.size()));
    res.reset(0);
    cudaPol(range((vertData.size() + 31) / 32 * 32),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), n = vertData.size(),
             offset = vertData.getPropertyOffset(tag)] __device__(int pi) mutable {
                T val = 0;
                if (pi < n) {
                    auto v = data.pack<3>(offset, pi);
                    val = v.abs().max();
                }

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
UnifiedIPCSystem::T UnifiedIPCSystem::dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0,
                                          const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto &vertData = vtemp;
    auto &res = temp;
    res.resize(count_warps(vertData.size()));
    cudaPol(range((vertData.size() + 31) / 32 * 32),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), n = vertData.size(),
             offset0 = vertData.getPropertyOffset(tag0),
             offset1 = vertData.getPropertyOffset(tag1)] __device__(int pi) mutable {
                T val = 0;
                if (pi < n) {
                    auto v0 = data.pack(dim_c<3>, offset0, pi);
                    auto v1 = data.pack(dim_c<3>, offset1, pi);
                    val = v0.dot(v1);
                }
        // reduce_to(pi, n, v, res[pi / 32]);

#if __CUDA_ARCH__ >= 800
                auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
                auto ret = zs::cg::reduce(tile, val, zs::cg::plus<T>());
                if (tile.thread_rank() == 0)
                    res[pi / 32] = ret;
#else
        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = val + tmp;
        }
        if (locid == 0)
            res[pi / 32] = val;
#endif
            });
    return reduce(cudaPol, res, thrust::plus<double>{});
}

} // namespace zeno