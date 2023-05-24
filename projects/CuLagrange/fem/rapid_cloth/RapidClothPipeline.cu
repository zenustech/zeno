#include "RapidCloth.cuh"
#include "zensim/Logger.hpp"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {
void RapidClothSystem::computeInertialAndForceGradient(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString& tag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    /// @brief inertial
    cudaPol(zs::range(coOffset),
            [vtemp = view<space>({}, vtemp), dt = dt, sigma = sigma * dt * dt, 
            BCStiffness = BCStiffness, tag] ZS_LAMBDA(int i) mutable {
                auto m = vtemp("ws", i);
                auto yk = vtemp.pack<3>(tag, i);
                auto grad = vtemp.pack<3>("grad", i) - m * (yk - vtemp.pack<3>("x_tilde", i)); 
                bool isBC = vtemp("isBC", i) > 0.5f;
                auto BCtarget = vtemp.pack(dim_c<3>, "BCtarget", i);  
                if (isBC)
                    grad -= m * BCStiffness * (yk - BCtarget); 
                vtemp.tuple<3>("grad", i) = grad; 
                // prepare preconditioner
                for (int d = 0; d != 3; ++d)
                    vtemp("P", d * 3 + d, i) += isBC ? m * (BCStiffness + 1.0f): m;
            });
    /// @brief extforce (only grad modified)
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary()) // skip soft boundary
            continue;
        cudaPol(zs::range(primHandle.getVerts().size()), [vtemp = view<space>({}, vtemp), gravAccel = gravAccel, dt = dt,
                                                          vOffset = primHandle.vOffset] ZS_LAMBDA(int vi) mutable {
            vi += vOffset;
            auto m = vtemp("ws", vi);
            vtemp.tuple(dim_c<3>, "grad", vi) = vtemp.pack(dim_c<3>, "grad", vi) + m * gravAccel * dt * dt;
        });
    }
    if (vtemp.hasProperty("extf")) {
        cudaPol(zs::range(coOffset), [vtemp = view<space>({}, vtemp), dt = dt] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, "grad", vi) =
                vtemp.pack(dim_c<3>, "grad", vi) + vtemp.pack(dim_c<3>, "extf", vi) * dt * dt;
        });
    }
}

/// elasticity
template <typename Model>
void computeElasticGradientAndHessianImpl(zs::CudaExecutionPolicy &cudaPol, typename RapidClothSystem::tiles_t &vtemp,
                                          const zs::SmallString& tag, 
                                          typename RapidClothSystem::tiles_t &seInds,
                                          typename RapidClothSystem::PrimitiveHandle &primHandle, const Model &model,
                                          typename RapidClothSystem::T dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename RapidClothSystem::mat3;
    using vec3 = typename RapidClothSystem::vec3;
    using T = typename RapidClothSystem::T;
    if (primHandle.category == ZenoParticles::curve) {
        if (primHandle.isBoundary() && !primHandle.isAuxiliary())
            return;
        /// ref: Fast Simulation of Mass-Spring Systems
        /// credits: Tiantian Liu
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = view<space>({}, vtemp), etemp = view<space>({}, primHandle.etemp),
                 eles = view<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset, n = primHandle.getEles().size(), tag] __device__(int ei) mutable {
                    auto inds = eles.pack(dim_c<2>, "inds", ei, int_c) + vOffset;

                    auto vole = eles("vol", ei);
                    auto k = eles("k", ei);
                    auto rl = eles("rl", ei);

                    vec3 xs[2] = {vtemp.pack(dim_c<3>, tag, inds[0]), vtemp.pack(dim_c<3>, tag, inds[1])};
                    auto xij = xs[1] - xs[0];
                    auto lij = xij.norm();
                    auto dij = xij / lij;
                    auto gij = k * (lij - rl) * dij;

                    // gradient
                    auto vfdt2 = gij * (dt * dt) * vole;
                    for (int d = 0; d != 3; ++d) {
                        atomic_add(exec_cuda, &vtemp(gradOffset + d, inds[0]), (T)vfdt2(d));
                        atomic_add(exec_cuda, &vtemp(gradOffset + d, inds[1]), (T)-vfdt2(d));
                    }

                    auto H = zs::vec<T, 6, 6>::zeros();
                    auto K = k * (mat3::identity() - rl / lij * (mat3::identity() - dyadic_prod(dij, dij)));
                    // make_pd(K);  // symmetric semi-definite positive, not necessary

                    for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j) {
                            H(i, j) = K(i, j);
                            H(i, 3 + j) = -K(i, j);
                            H(3 + i, j) = -K(i, j);
                            H(3 + i, 3 + j) = K(i, j);
                        }
                    H *= dt * dt * vole;

                    etemp.tuple<6 * 6>("He", ei) = H;
#pragma unroll
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
                [vtemp = view<space>({}, vtemp), etemp = view<space>({}, primHandle.etemp),
                 eles = view<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset, tag] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<2, 2>, "IB", ei);
                    auto inds = eles.pack(dim_c<3>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[3] = {vtemp.pack(dim_c<3>, tag, inds[0]), vtemp.pack(dim_c<3>, tag, inds[1]),
                                  vtemp.pack(dim_c<3>, tag, inds[2])};
                    auto x1x0 = xs[1] - xs[0];
                    auto x2x0 = xs[2] - xs[0];

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
                    auto vecP = flatten(model.mu * Pstretch + (model.mu * 0.3f) * Pshear);
                    auto vfdt2 = -vole * (dFdXT * vecP) * (dt * dt);

                    for (int i = 0; i != 3; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp(gradOffset + d, vi), (T)vfdt2(i * 3 + d));
                    }

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
                        dPdF *= (model.mu * 0.3f);
                        return dPdF;
                    };
                    auto He = stretchHessian() + shearHessian();
                    auto H = dFdX.transpose() * He * dFdX * (dt * dt * vole);

                    etemp.tuple(dim_c<9, 9>, "He", ei) = H;
#pragma unroll
                    for (int vi = 0; vi != 3; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
            }); 
    } else if (primHandle.category == ZenoParticles::tet)
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = view<space>({}, vtemp), etemp = view<space>({}, primHandle.etemp),
                 eles = view<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset, tag] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                    auto inds = eles.pack(dim_c<4>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                                  vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};

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
                            atomic_add(exec_cuda, &vtemp(gradOffset + d, vi), (T)vfdt2(i * 3 + d));
                    }

                    auto Hq = model.first_piola_derivative(F, true_c);
                    auto H = dFdXT * Hq * dFdX * vole * dt * dt;

                    etemp.tuple(dim_c<12, 12>, "He", ei) = H;
#pragma unroll
                    for (int vi = 0; vi != 4; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
                });
}

void RapidClothSystem::computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &tag) {
    using namespace zs;
    for (auto &primHandle : prims) {
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, vtemp, tag, seInds, primHandle, elasticModel, dt);
        })(primHandle.getModels().getElasticModel());
    }
}

void RapidClothSystem::computeRepulsionGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag) {
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    pol(range(npt), [vtemp = proxy<space>({}, vtemp), 
                    tempPT = proxy<space>({}, tempPT), 
                    repulsionCoef = repulsionCoef, 
                    repulsionRange = repulsionRange, 
                    coOffset = coOffset, 
                    delta = delta, 
                    tag] __device__ (int i) mutable {
        // calculate grad 
        auto inds = tempPT.pack(dim_c<4>, "inds", i, int_c); 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto t0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto t1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto t2 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        T avgMass = 0.f; 
        int vertCnt = 0; 
        for (int k = 0; k < 4; k++)
            if (inds[k] < coOffset)
            {
                avgMass += vtemp("ws", inds[k]); 
                vertCnt++; 
            }
        avgMass /= (T)vertCnt; 
        auto dist = tempPT("dist", i); 
        if (dist > repulsionRange)
            return; 
        // auto grad = repulsionCoef * avgMass * (repulsionRange - dist) * dist_grad_pt(p, t0, t1, t2); 
        auto grad = repulsionCoef * avgMass * dist * dist_grad_pt(p, t0, t1, t2); 
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp("grad", d, inds[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp("grad", d, inds[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp("grad", d, inds[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp("grad", d, inds[3]), grad(3, d));
        }
        auto ptHess = repulsionCoef * dist_hess_pt(p, t0, t1, t2);
        // make pd
        make_pd(ptHess);        
        tempPT.tuple(dim_c<144>, "hess", i) = ptHess; 
        for (int vi = 0; vi != 4; ++vi) 
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) 
                    atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), ptHess(vi * 3 + i, vi * 3 + j));
    }); 

    pol(range(nee), [vtemp = proxy<space>({}, vtemp), 
                    tempEE = proxy<space>({}, tempEE), 
                    repulsionCoef = repulsionCoef, 
                    repulsionRange = repulsionRange, 
                    coOffset = coOffset, 
                    delta = delta, 
                    tag] __device__ (int i) mutable {
        // calculate grad 
        auto inds = tempEE.pack(dim_c<4>, "inds", i, int_c); 
        auto ei0 = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto ei1 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto ej0 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto ej1 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        T avgMass = 0.f; 
        int vertCnt = 0; 
        for (int k = 0; k < 4; k++)
            if (inds[k] < coOffset)
            {
                avgMass += vtemp("ws", inds[k]); 
                vertCnt++; 
            }
        avgMass /= (T)vertCnt; 
        auto dist = tempEE("dist", i); 
        if (dist > repulsionRange)
            return; 
        // auto grad = repulsionCoef * avgMass * (repulsionRange - dist) * dist_grad_ee(ei0, ei1, ej0, ej1); 
        auto grad = repulsionCoef * avgMass * dist * dist_grad_ee(ei0, ei1, ej0, ej1); 
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp("grad", d, inds[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp("grad", d, inds[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp("grad", d, inds[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp("grad", d, inds[3]), grad(3, d));
        }
        auto eeHess = repulsionCoef * dist_hess_ee(ei0, ei1, ej0, ej1); 
        make_pd(eeHess);
        tempEE.tuple(dim_c<144>, "hess", i) = eeHess;
        for (int vi = 0; vi != 4; ++vi) 
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) 
                    atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), eeHess(vi * 3 + i, vi * 3 + j));
    }); 

    if (enableDegeneratedDist)
    {
        pol(range(npp), [vtemp = proxy<space>({}, vtemp), 
                        tempPP = proxy<space>({}, tempPP), 
                        repulsionCoef = repulsionCoef, 
                        repulsionRange = repulsionRange, 
                        coOffset = coOffset, 
                        delta = delta, 
                        tag] __device__ (int i) mutable {
            // calculate grad 
            auto inds = tempPP.pack(dim_c<2>, "inds", i, int_c); 
            auto x0 = vtemp.pack(dim_c<3>, tag, inds[0]); 
            auto x1 = vtemp.pack(dim_c<3>, tag, inds[1]); 
            T avgMass = 0.f; 
            int vertCnt = 0; 
            for (int k = 0; k < 2; k++)
                if (inds[k] < coOffset)
                {
                    avgMass += vtemp("ws", inds[k]); 
                    vertCnt++; 
                }
            avgMass /= (T)vertCnt; 
            auto dist = tempPP("dist", i); 
            if (dist > repulsionRange)
                return; 
            // auto grad = repulsionCoef * avgMass * (repulsionRange - dist) * dist_grad_pp(x0, x1); 
            auto grad = repulsionCoef * avgMass * dist * dist_grad_pp(x0, x1); 
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp("grad", d, inds[0]), grad(0, d));
                atomic_add(exec_cuda, &vtemp("grad", d, inds[1]), grad(1, d));
            }
            auto ppHess = repulsionCoef * dist_hess_pp(x0, x1);
            // make pd
            make_pd(ppHess);        
            tempPP.tuple(dim_c<36>, "hess", i) = ppHess;
            for (int vi = 0; vi != 2; ++vi) 
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) 
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), ppHess(vi * 3 + i, vi * 3 + j));
        }); 

        pol(range(npe), [vtemp = proxy<space>({}, vtemp), 
                        tempPE = proxy<space>({}, tempPE), 
                        repulsionCoef = repulsionCoef, 
                        repulsionRange = repulsionRange, 
                        coOffset = coOffset, 
                        delta = delta, 
                        tag] __device__ (int i) mutable {
            // calculate grad 
            auto inds = tempPE.pack(dim_c<3>, "inds", i, int_c); 
            auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
            auto e0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
            auto e1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
            T avgMass = 0.f; 
            int vertCnt = 0; 
            for (int k = 0; k < 3; k++)
                if (inds[k] < coOffset)
                {
                    avgMass += vtemp("ws", inds[k]); 
                    vertCnt++; 
                }
            avgMass /= (T)vertCnt; 
            auto dist = tempPE("dist", i);
            if (dist > repulsionRange)
                return; 
            // auto grad = repulsionCoef * avgMass * (repulsionRange - dist) * dist_grad_pe(p, e0, e1); 
            auto grad = repulsionCoef * avgMass * dist * dist_grad_pe(p, e0, e1); 
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp("grad", d, inds[0]), grad(0, d));
                atomic_add(exec_cuda, &vtemp("grad", d, inds[1]), grad(1, d));
                atomic_add(exec_cuda, &vtemp("grad", d, inds[2]), grad(2, d));
            }
            auto peHess = repulsionCoef *  dist_hess_pe(p, e0, e1);
            // make pd
            make_pd(peHess);        
            tempPE.tuple(dim_c<81>, "hess", i) = peHess;
            for (int vi = 0; vi != 3; ++vi) 
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) 
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), peHess(vi * 3 + i, vi * 3 + j));
        }); 
    }
}

void RapidClothSystem::subStepping(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
            vtemp.tuple(dim_c<3>, "x(l)", vi) = vtemp.pack(dim_c<3>, "x[k]", vi); 
        }); 
    for (int iters = 0; iters < PNCap; iters++)
    {
        if (iters != 0)
            pol(range(vtemp.size()), 
                [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
                    vtemp.tuple(dim_c<3>, "x[k]", vi) = vtemp.pack(dim_c<3>, "y[k+1]", vi); 
                }); 
        newtonDynamicsStep(pol);  
    }
    // y(l) = y[k+1]
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
            vtemp.tuple(dim_c<3>, "y(l)", vi) = vtemp.pack(dim_c<3>, "y[k+1]", vi);  
            vtemp.tuple(dim_c<3>, "x[k]", vi) = vtemp.pack(dim_c<3>, "x(l)", vi); 
            vtemp("r(l)", vi) = 1.f; 
        }); 
    for (int iters = 0; iters < L; iters++)
    {
        if (D < D_min)
        {
            if (!silentMode_c)
                fmt::print("[proximity] iters: {}, tiny D: {} < D_min: {} < D_max: {} doing proximity search...\n", 
                    iters, D, D_min, D_max); 
            findConstraints(pol, D_max); 
            D = D_max; 
        }
        backwardStep(pol); 
        forwardStep(pol); 
        auto disp = infNorm(pol, "disp", numDofs, wrapv<1>{}); 
        D -= 2 * disp; 
        auto res = infNorm(pol, "r(l)", numDofs, wrapv<1>{}); 
        if (!silentMode_c)
            fmt::print("disp: {}, D: {}, res * 1e4: {}\n", 
                disp, D, res * 1e4f); 
        if (res < eps)
        {
            fmt::print(fg(fmt::color::orange_red), "converged in {} iterations with res: {}\n", 
                iters, res); 
            break; 
        }
        if (iters == L - 1)
            fmt::print(fg(fmt::color::red), "failed to converged within {} iters, exit with res: {}\n", 
                L - 1, res); 
    }
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
            vtemp.tuple(dim_c<3>, "x[k]", vi) = vtemp.pack(dim_c<3>, "x(l)", vi); 
        }); 
    // DEBUG 
    if (debugVis_c)
    {
        auto hv = vtemp.clone({memsrc_e::host, -1}); 
        auto hv_view = proxy<execspace_e::host>({}, hv); 
        int n = vtemp.size(); 
        for (int vi = 0; vi < n; vi++)
        {
            visPrim->verts.values[vi] = zeno::vec3f {
                hv_view("x(l)", 0, vi), 
                hv_view("x(l)", 1, vi), 
                hv_view("x(l)", 2, vi)
            }; 
            visPrim->verts.values[vi + n] = zeno::vec3f {
                hv_view("y[k+1]", 0, vi), 
                hv_view("y[k+1]", 1, vi), 
                hv_view("y[k+1]", 2, vi)
            }; 
            visPrim->lines.values[vi] = zeno::vec2i {vi, vi + n}; 
        }

        auto ht = stInds.clone({memsrc_e::host, -1}); 
        auto ht_view = proxy<execspace_e::host>({}, ht); 
        int tn = stInds.size(); 
        visPrim->tris.resize(tn); 
        for (int ti = 0; ti < tn; ti++)
        {
            visPrim->tris.values[ti] = zeno::vec3i {n + ht_view("inds", 0, ti, int_c), 
                n + ht_view("inds", 1, ti, int_c), n + ht_view("inds", 2, ti, int_c)}; 
        }
    }
}

struct StepRapidClothSystem : INode {
    using T = typename RapidClothSystem::T; 

    void apply() override {
        using namespace zs;
        auto A = get_input<RapidClothSystem>("ZSRapidClothSystem");
        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);

        for (int subi = 0; subi != nSubsteps; ++subi) {
            A->advanceSubstep(cudaPol, (T)1 / nSubsteps);
            A->subStepping(cudaPol);
            A->updateVelocities(cudaPol);
        }
        // update velocity and positions
        A->writebackPositionsAndVelocities(cudaPol);

        set_output("ZSRapidClothSystem", A);
        set_output("visPrim", A->visPrim); 
    }
};

ZENDEFNODE(StepRapidClothSystem, {{
                                 "ZSRapidClothSystem",
                                 {"int", "num_substeps", "1"},
                                 {"float", "dt", "0.01"},
                             },
                             {"ZSRapidClothSystem", "visPrim"},
                             {},
                             {"FEM"}});

#if 0 // TODO: add later 
struct RapidClothSystemForceField : INode {
    using T = typename RapidClothSystem::T; 

    template <typename VelSplsViewT>
    void computeForce(zs::CudaExecutionPolicy &cudaPol, float windDragCoeff, float windDensity, int vOffset,
                      VelSplsViewT velLs, typename RapidClothSystem::tiles_t &vtemp,
                      const typename RapidClothSystem::tiles_t &eles) {
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

        auto A = get_input<RapidClothSystem>("ZSRapidClothSystem");
        auto &vtemp = A->vtemp;
        const auto numVerts = A->coOffset;
        auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");

        auto cudaPol = zs::cuda_exec();
        vtemp.append_channels(cudaPol, {{"extf", 3}});
        cudaPol(range(numVerts), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.template tuple<3>("extf", i) = zs::vec<T, 3>::zeros();
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

        set_output("ZSRapidClothSystem", A);
    }
};

ZENDEFNODE(RapidClothSystemForceField,
           {
               {"ZSRapidClothSystem", "ZSLevelSet", {"float", "wind_drag", "0"}, {"float", "wind_density", "1"}},
               {"ZSRapidClothSystem"},
               {},
               {"FEM"},
           });
#endif 
} // namespace zeno