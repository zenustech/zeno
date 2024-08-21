#include "FastCloth.cuh"
#include "zensim/Logger.hpp"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

typename FastClothSystem::T FastClothSystem::infNorm(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    using T = typename FastClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(numDofs);
    temp.resize(nwarps);
    cudaPol(range(numDofs), [data = view<space>({}, vtemp), res = view<space>(temp), n = numDofs,
                             offset = vtemp.getPropertyOffset("dir")] __device__(int pi) mutable {
        auto v = data.pack(dim_c<3>, offset, pi);
        auto val = v.abs().max();

        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = zs::max(val, tmp);
        }
        if (locid == 0)
            res[pi / 32] = val;
    });
    return reduce(cudaPol, temp, thrust::maximum<T>{});
}

typename FastClothSystem::T FastClothSystem::l2Norm(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    return zs::sqrt(dot(pol, tag, tag));
}

typename FastClothSystem::T FastClothSystem::dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0,
                                                 const zs::SmallString tag1) {
    using namespace zs;
    using T = typename FastClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(numDofs);
    temp.resize(nwarps);
    temp.reset(0);
    cudaPol(range(numDofs), [data = view<space>({}, vtemp), res = view<space>(temp), n = numDofs,
                             offset0 = vtemp.getPropertyOffset(tag0),
                             offset1 = vtemp.getPropertyOffset(tag1)] __device__(int pi) mutable {
        auto v0 = data.pack(dim_c<3>, offset0, pi);
        auto v1 = data.pack(dim_c<3>, offset1, pi);
        reduce_to(pi, n, v0.dot(v1), res[pi / 32]);
    });
    return reduce(cudaPol, temp, thrust::plus<T>{});
}

void FastClothSystem::computeBoundaryConstraints(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(Collapse{numBouDofs}, [vtemp = view<space>({}, vtemp), coOffset = coOffset] __device__(int vi) mutable {
        vi += coOffset;
        auto xtarget = vtemp.pack<3>("ytilde", vi);
        auto x = vtemp.pack<3>("yn", vi);
        vtemp.tuple(dim_c<3>, "cons", vi) = x - xtarget;
    });
}
bool FastClothSystem::areBoundaryConstraintsSatisfied(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    computeBoundaryConstraints(pol);
    auto res = boundaryConstraintResidual(pol);
    return res < s_constraint_residual;
}
typename FastClothSystem::T FastClothSystem::boundaryConstraintResidual(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (projectDBC)
        return 0;
    temp.resize(numBouDofs * 2);
    pol(Collapse{numBouDofs}, [vtemp = view<space>({}, vtemp), den = temp.data(), num = temp.data() + numBouDofs,
                               coOffset = coOffset] __device__(int vi) mutable {
        vi += coOffset;
        auto cons = vtemp.pack<3>("cons", vi);
        auto xt = vtemp.pack<3>("yhat", vi);
        auto xtarget = vtemp.pack<3>("ytilde", vi);
        T n = 0, d_ = 0;
        // https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf Eq5
        for (int d = 0; d != 3; ++d) {
            n += zs::sqr(cons[d]);
            d_ += zs::sqr(xt[d] - xtarget[d]);
        }
        num[vi] = n;
        den[vi] = d_;
    });
    // denominator ... numerator ...
    auto tot = reduce(pol, temp);
    temp.resize(numBouDofs);
    auto dsqr = reduce(pol, temp);
    auto nsqr = tot - dsqr;
    T ret = 0;
    if (dsqr == 0)
        ret = std::sqrt(nsqr);
    else
        ret = std::sqrt(nsqr / dsqr);
    return ret < 1e-6 ? 0 : ret;
}

void FastClothSystem::computeInertialAndCouplingAndForceGradient(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    /// @brief inertial and coupling
    cudaPol(zs::range(coOffset),
            [vtemp = view<space>({}, vtemp), dt = dt, sigma = sigma * dt * dt, 
            BCStiffness = BCStiffness] ZS_LAMBDA(int i) mutable {
                auto m = vtemp("ws", i);
                auto yn = vtemp.pack<3>("yn", i);
                auto grad = vtemp.pack<3>("grad", i) - m * (yn - vtemp.pack<3>("ytilde", i)) -
                                            m * sigma * (yn - vtemp.pack<3>("xn", i));
                bool isBC = vtemp("isBC", i) > 0.5f;
                auto BCtarget = vtemp.pack(dim_c<3>, "BCtarget", i);  
                if (isBC)
                    grad -= m * BCStiffness * (yn - BCtarget); 
                vtemp.tuple<3>("grad", i) = grad; 
                // prepare preconditioner
                for (int d = 0; d != 3; ++d)
#if !s_useGDDiagHess
                    vtemp("P", d * 3 + d, i) += isBC ? (m * (sigma + BCStiffness + 1.0f)): (m * (sigma + 1.0f));
#else 
                    vtemp("P", d, i) += isBC ? (m * (sigma + BCStiffness + 1.0f)): (m * (sigma + 1.0f));
#endif
            });
    /// @brief extforce (only grad modified)
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary()) // skip soft boundary
            continue;
        cudaPol(zs::range(primHandle.getVerts().size()), [vtemp = view<space>({}, vtemp), extAccel = extAccel, dt = dt,
                                                          vOffset = primHandle.vOffset] ZS_LAMBDA(int vi) mutable {
            vi += vOffset;
            auto m = vtemp("ws", vi);
            vtemp.tuple(dim_c<3>, "grad", vi) = vtemp.pack(dim_c<3>, "grad", vi) + m * extAccel * dt * dt;
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
void computeElasticGradientAndHessianImpl(zs::CudaExecutionPolicy &cudaPol, typename FastClothSystem::tiles_t &vtemp,
                                          typename FastClothSystem::tiles_t &seInds,
                                          typename FastClothSystem::PrimitiveHandle &primHandle, const Model &model,
                                          typename FastClothSystem::T dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename FastClothSystem::mat3;
    using vec3 = typename FastClothSystem::vec3;
    using T = typename FastClothSystem::T;
    if (primHandle.category == ZenoParticles::curve) {
        if (primHandle.isBoundary() && !primHandle.isAuxiliary())
            return;
        /// ref: Fast Simulation of Mass-Spring Systems
        /// credits: Tiantian Liu
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = view<space>({}, vtemp), etemp = view<space>({}, primHandle.etemp),
                 eles = view<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset, n = primHandle.getEles().size()] __device__(int ei) mutable {
                    auto inds = eles.pack(dim_c<2>, "inds", ei, int_c) + vOffset;

                    auto vole = eles("vol", ei);
                    auto k = eles("k", ei);
                    auto rl = eles("rl", ei);

                    vec3 xs[2] = {vtemp.pack(dim_c<3>, "yn", inds[0]), vtemp.pack(dim_c<3>, "yn", inds[1])};
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

#if !s_useGDDiagHess
                    etemp.tuple<6 * 6>("He", ei) = H;
#pragma unroll
                    for (int vi = 0; vi != 2; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
#else
#pragma unroll
                    for (int vi = 0; vi != 2; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            atomic_add(exec_cuda, &vtemp("P", i, inds[vi]), H(vi * 3 + i, vi * 3 + i));
                    }
#endif
                });
    } else if (primHandle.category == ZenoParticles::surface) {
        if (primHandle.isBoundary())
            return;
#if s_useMassSpring
        auto &edges = primHandle.getSurfEdges();
        cudaPol(range(edges.size()), [seInds = view<space>({}, seInds), vtemp = view<space>({}, vtemp), model = model,
                                      vOffset = primHandle.vOffset, seoffset = primHandle.seOffset, dt = dt,
                                      limit = limits<T>::epsilon() * 10.0f] __device__(int ei) mutable {
            int sei = ei + seoffset;
            auto inds = seInds.pack(dim_c<2>, "inds", sei, int_c);
            T E;
            auto m = 0.5f * (vtemp("ws", inds[0]) + vtemp("ws", inds[1]));
            auto v0 = vtemp.pack(dim_c<3>, "yn", inds[0]);
            auto v1 = vtemp.pack(dim_c<3>, "yn", inds[1]);
            auto restL = seInds("restLen", sei);
            // auto restL = 4.2f;
            auto dist2 = zs::max((v0 - v1).l2NormSqr(), limit);
            auto dist = zs::sqrt(dist2);
            auto grad = m * model.mu * (1.0f - restL / dist) * (v0 - v1) * dt * dt; // minus grad for v1, -grad for v0
            for (int d = 0; d != 3; d++) {
                atomic_add(exec_cuda, &vtemp("grad", d, inds[0]), (T)-grad(d));
                atomic_add(exec_cuda, &vtemp("grad", d, inds[1]), (T)grad(d));
            }
#if s_useGDDiagHess
            auto restLDivDist3 = restL / (dist2 * dist);
            auto diag1 = (v0 - v1) * (v0 - v1) * restLDivDist3;
            auto diag2 = (dist - restL) / dist;
            for (int d = 0; d != 3; d++) {
                T val = (diag1(d) + diag2) * m * model.mu * dt * dt;
                if (val <= 0)
                    continue;
                atomic_add(exec_cuda, &vtemp("P", d, inds[0]), val);
                atomic_add(exec_cuda, &vtemp("P", d, inds[1]), val);
            }
#else 
            // TODO: 9x9 etemp hessian? not implemented currently; as a result do not support newton solver
            zs::vec<T, 6> y; 
            for (int d = 0; d != 3; d++)
            {
                auto yd = v0(d) - v1(d); 
                y(d) = yd; 
                y(d + 3) = -yd; 
            }
            auto hess = restL / (dist2 * dist) * dyadic_prod(y, y); 
            auto val = 1.0f - restL / dist; 
            for (int d = 0; d != 3; d++)
            {
                hess(d, d) += val; 
                hess(d + 3, d + 3) -= val; 
            }
            for (int vi = 0; vi != 2; vi++)
                for (int di = 0; di != 3; di++)
                    for (int dj = 0; dj != 3; dj++)
                        atomic_add(exec_cuda, &vtemp("P", di * 3 + dj, inds[vi]), 
                            hess(vi * 3 + di, vi * 3 + dj));
#endif
        });
#else
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = view<space>({}, vtemp), etemp = view<space>({}, primHandle.etemp),
                 eles = view<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<2, 2>, "IB", ei);
                    auto inds = eles.pack(dim_c<3>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[3] = {vtemp.pack(dim_c<3>, "yn", inds[0]), vtemp.pack(dim_c<3>, "yn", inds[1]),
                                  vtemp.pack(dim_c<3>, "yn", inds[2])};
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
                    auto vecP = flatten(model.mu * Pstretch + (model.mu * s_clothShearingCoeff) * Pshear);
                    auto vfdt2 = -vole * (dFdXT * vecP) * (dt * dt);

                    for (int i = 0; i != 3; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp(gradOffset + d, vi), (T)vfdt2(i * 3 + d));
                    }

            /// ref: A Finite Element Formulation of Baraff-Witkin Cloth
            // suggested by huang kemeng
#if !s_useGDDiagHess
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
                        dPdF *= (model.mu * s_clothShearingCoeff);
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
#else 
                    // vole * model.mu, s_clothShearingCoeff(0.1) for shear energy 
                    mat3 H0, H1, H01, H10; 
                    zs::vec<T, 6, 6> He;  
                    auto f0f0T = dyadic_prod(f0, f0); 
                    auto f1f1T = dyadic_prod(f1, f1); 
                    auto f0f1T = dyadic_prod(f0, f1); 
                    auto f1f0T = f0f1T.transpose(); 
                    auto id3 = mat3::identity(); 
                    H0 = 2.0f / (f0Norm * f0Norm * f0Norm) * f0f0T
                         + 2.0f * (1 - 1.0f / (f0Norm)) * id3 + 2.0f * f1f1T * s_clothShearingCoeff;
                    H1 = 2.0f / (f1Norm * f1Norm * f1Norm) * f1f1T
                         + 2.0f * (1 - 1.0f / (f1Norm)) * id3 + 2.0f * f0f0T * s_clothShearingCoeff;
                    H01 = 2.0f * s_clothShearingCoeff * (f1f0T + f0Tf1 * id3); 
                    H10 = H01.transpose(); 
                    for (int di = 0; di < 3; di++)
                        for (int dj = 0; dj < 3; dj++)
                        {
                            He(di, dj) = H0(di, dj); 
                            He(di + 3, dj) = H10(di, dj); 
                            He(di, dj + 3) = H01(di, dj); 
                            He(di + 3, dj + 3) = H1(di, dj); 
                        }
                    auto H = model.mu * dFdXT * He * dFdX * (dt * dt * vole);
#pragma unroll
                    for (int vi = 0; vi != 3; ++vi) {
                        for (int i = 0; i != 3; ++i)
                        {
                            auto v = H(vi * 3 + i, vi * 3 + i); 
                            if (v > 0)
                                atomic_add(exec_cuda, &vtemp("P", i, inds[vi]), v);
                        }
                    }
#endif
                });
#endif
    } else if (primHandle.category == ZenoParticles::tet)
        cudaPol(zs::range(primHandle.getEles().size()),
                [vtemp = view<space>({}, vtemp), etemp = view<space>({}, primHandle.etemp),
                 eles = view<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                    auto inds = eles.pack(dim_c<4>, "inds", ei, int_c) + vOffset;
                    auto vole = eles("vol", ei);
                    vec3 xs[4] = {vtemp.pack<3>("yn", inds[0]), vtemp.pack<3>("yn", inds[1]),
                                  vtemp.pack<3>("yn", inds[2]), vtemp.pack<3>("yn", inds[3])};

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

#if !s_useGDDiagHess
                    etemp.tuple(dim_c<12, 12>, "He", ei) = H;
#pragma unroll
                    for (int vi = 0; vi != 4; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            for (int j = 0; j != 3; ++j) {
                                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]), H(vi * 3 + i, vi * 3 + j));
                            }
                    }
#else
#pragma unroll
                    for (int vi = 0; vi != 4; ++vi) {
                        for (int i = 0; i != 3; ++i)
                            atomic_add(exec_cuda, &vtemp("P", i, inds[vi]), H(vi * 3 + i, vi * 3 + i));
                    }
#endif
                });
}

void FastClothSystem::computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    for (auto &primHandle : prims) {
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, vtemp, seInds, primHandle, elasticModel, dt);
        })(primHandle.getModels().getElasticModel());
    }
    for (auto &primHandle : auxPrims) {
        using ModelT = RM_CVREF_T(primHandle.getModels().getElasticModel());
        const ModelT &model = primHandle.modelsPtr ? primHandle.getModels().getElasticModel() : ModelT{};
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, vtemp, seInds, primHandle, elasticModel, dt);
        })(model);
    }
}

void FastClothSystem::subStepping(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    /// y0, x0
    // y0, x0
    timer.tick();
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), dt = dt, coOffset = coOffset] ZS_LAMBDA(int i) mutable {
        auto yt = vtemp.pack(dim_c<3>, "yn", i);
        auto vt = vtemp.pack(dim_c<3>, "vn", i);
        vtemp.tuple(dim_c<3>, "yn", i) = i < coOffset ? (yt + dt * vt) : (vtemp.pack(dim_c<3>, "xt", i) + dt * vt);
        vtemp.tuple(dim_c<3>, "xn", i) = yt;
        if (i >= coOffset)
            vtemp.tuple(dim_c<3>, "xt", i) = vtemp.pack(dim_c<3>, "yn", i);
    });
    // xinit
    {
        /// @brief spatially smooth initial displacement
        auto spg = typename ZenoSparseGrid::spg_t{vtemp.get_allocator(), {{"w", 1}, {"dir", 3}}, (std::size_t)numDofs};
        spg._transform.preScale(zs::vec<T, 3>::uniform(L)); // using L as voxel size
        spg.resizePartition(pol, numDofs * 8);
        pol(range(numDofs), [spgv = proxy<space>(spg), vtemp = view<space>({}, vtemp)] __device__(int i) mutable {
            using spg_t = RM_CVREF_T(spgv);
            auto xt = vtemp.pack(dim_c<3>, "xn", i);
            auto arena = spgv.wArena(xt);
            for (auto loc : arena.range()) {
                auto coord = arena.coord(loc);
                auto bcoord = coord - (coord & (spg_t::side_length - 1));
                spgv._table.insert(bcoord);
            }
        });
        const auto nbs = spg.numBlocks();
        spg.resizeGrid(nbs);
        spg._grid.reset(0);

        pol(range(numDofs),
            [spgv = proxy<space>(spg), vtemp = view<space>({}, vtemp), coOffset = coOffset] __device__(int pi) mutable {
                auto m = vtemp("ws", pi);
                auto xt = vtemp.pack(dim_c<3>, "xn", pi);
                auto dir = vtemp.pack(dim_c<3>, "yn", pi) - xt;
                auto arena = spgv.wArena(xt);
                for (auto loc : arena.range()) {
                    auto coord = arena.coord(loc);
                    auto [bno, cno] = spgv.decomposeCoord(coord);
                    auto W = arena.weight(loc) * m;
#pragma unroll
                    for (int d = 0; d < 3; ++d)
                        atomic_add(exec_cuda, &spgv("dir", d, bno, cno), W * dir(d));

                    atomic_add(exec_cuda, &spgv("w", bno, cno), W);
                }
                if (pi >= coOffset)
                    vtemp.tuple(dim_c<3>, "gridDir", pi) = dir;
            });
        pol(range((std::size_t)nbs * spg.block_size),
            [grid = view<space>({}, spg._grid)] __device__(std::size_t cellno) mutable {
                using T = typename RM_CVREF_T(grid)::value_type;
                if (grid("w", cellno) > limits<T>::epsilon())
                    grid.tuple(dim_c<3>, "dir", cellno) = grid.pack(dim_c<3>, "dir", cellno) / grid("w", cellno);
            });
        // test: fixing the boundary issues; numDofs -> coOffset
        pol(range(coOffset), [spgv = proxy<space>(spg), vtemp = view<space>({}, vtemp)] __device__(int pi) mutable {
            using T = typename RM_CVREF_T(spgv)::value_type;
            using vec3 = zs::vec<T, 3>;
            auto xt = vtemp.pack(dim_c<3>, "xn", pi);
            auto dir = vec3::zeros();
            auto arena = spgv.wArena(xt);
            for (auto loc : arena.range()) {
                auto coord = arena.coord(loc);
                auto [bno, cno] = spgv.decomposeCoord(coord);
                auto W = arena.weight(loc);
                dir[0] += W * spgv("dir", 0, bno, cno);
                dir[1] += W * spgv("dir", 1, bno, cno);
                dir[2] += W * spgv("dir", 2, bno, cno);
                // dir += W * spgv._grid.pack(dim_c<3>, "dir", bno, cno);
            }
            vtemp.tuple(dim_c<3>, "gridDir", pi) = dir;
        });
    }
    timer.tock();
    initInterpolationTime = timer.elapsed();
    {
        const auto ratio = (T)1 / (T)IInit;
        pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), ratio] ZS_LAMBDA(int i) mutable {
            auto dir = vtemp.pack(dim_c<3>, "gridDir", i);
            vtemp.tuple(dim_c<3>, "yk", i) = vtemp.pack(dim_c<3>, "yn", i); // record current yn in yk
            vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "xn", i) + ratio * dir;
        });

        for (int i = 0; true;) {
            /// start collision solver
            ///
            /// @brief Xinit
            initialStepping(pol);

            fmt::print(fg(fmt::color::alice_blue), "init iter [{}]\n", i);
            findConstraints(pol, dHat);

            /// @brief backup xn for potential hard phase
            pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                vtemp.tuple(dim_c<3>, "xk", i) = vtemp.pack(dim_c<3>, "xn", i);
            });
            /// @brief collision handling
            bool success = false;
            /// @note ref: sec 4.3.4
#if 0 
            int r = 0;
            for (; r != R; ++r) {
                if (success = collisionStep(pol, false); success)
                    break;
            }
#else
            // TODO: check paper
            success = collisionStep(pol, false);
#endif
#if 0
            if (success) {
                fmt::print(fg(fmt::color::alice_blue), "done pure soft collision iters {} out of {}\n", r, R);
                pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), n = numDofs] ZS_LAMBDA(int i) mutable {
                    auto xk = vtemp.pack(dim_c<3>, "xk", i);
                    auto xn = vtemp.pack(dim_c<3>, "xn", i);
                    auto xinit = vtemp.pack(dim_c<3>, "xinit", i);
                    if (i < 8 || i > n - 2) {
                        printf("par [%d]: xn <%f, %f, %f>, xk <%f, %f, %f> -> xinit <%f, %f, %f>\n", i, xn[0], xn[1],
                               xn[2], xk[0], xk[1], xk[2], xinit[0], xinit[1], xinit[2]);
                    }
                });
            }
#endif
#if s_useHardPhase
            if (!success) {
                /// @brief restore xn with xk for hard phase
                pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                    vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xk", i);
                });
                success = collisionStep(pol, true);
            }
            if (!success)
                throw std::runtime_error("collision step in initialization fails!\n");
#endif
            /// done collision solver

            if (++i == IInit)
                break;

            /// @brief update xinit for the next initialization iteration
            pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), ratio] ZS_LAMBDA(int i) mutable {
                auto dir = vtemp.pack(dim_c<3>, "gridDir", i);
                vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "yn", i) + ratio * dir;
            });
        }
        pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "yk", i); // restore yn from yk
        });
    }

    /// @brief collect pairs only once depending on x^init
    /// @note ref: Sec 4.3.2
#if s_useChebyshevAcc
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, "yn-2", i) = vtemp.pack(dim_c<3>, "yn", i);
        vtemp.tuple(dim_c<3>, "yn-1", i) = vtemp.pack(dim_c<3>, "yn", i);
    });
#endif
#if s_useLineSearch
    T lastEnergy = dynamicsEnergy(pol);
    if (alpha < alphaMin)
        alpha = alphaMin;
    alpha /= alphaDecrease;
#endif

    int maxIters = K * IDyn;
    int k = 0;
#if s_testLightCache
    lightCD(pol, dHat * 3.5f);
    // lightFilterConstraints(pol, dHat, "xinit");
#else
    findConstraints(pol, dHat);
#endif
    /// optimizer
    zs::CppTimer timer;
    for (int iters = 0; iters < maxIters; ++iters, ++k) {
        pol.sync(true);
        bool converged = false;

        if constexpr (s_enableProfile)
            timer.tick();

        // check constraints
        // TODO: boundaries are dealt in init step,
        // it seemes that projectDBC can be directly set to true in the entire process
        if (!projectDBC) {
            computeBoundaryConstraints(pol);
            auto cr = boundaryConstraintResidual(pol);
            if (cr < s_constraint_residual) {
                projectDBC = true;
            }
#if !s_silentMode
            fmt::print(fg(fmt::color::alice_blue), "iteration {} cons residual: {}\n", iters, cr);
#endif
        }
        pol.sync(false);
#if s_useNewtonSolver
        newtonDynamicsStep(pol);
#else
        gdDynamicsStep(pol);
#endif
        // CHECK PN CONDITION
        // T res = infNorm(pol) / dt;
        // T cons_res = boundaryConstraintResidual(pol);
#if 0
        if (res < targetGRes && cons_res == 0) {
            fmt::print("\t\t\tdynamics ended: substep {} iteration {}: direction residual(/dt) {}, "
                                "grad residual {}\n",
                                substep, iters, res, res * dt); 
            converged = true; 
            break;
        }
        ZS_INFO(fmt::format("substep {} iteration {}: direction residual(/dt) {}, "
                            "grad residual {}\n",
                            substep, iters, res, res * dt));
#endif
#if s_useLineSearch
        if (alpha < 1e-2f)
            converged = true;
        if (k % 30 == 0)
            fmt::print("\t\tdynamics iteration: {}, alpha: {}\n", k, alpha);
#if s_debugOutput
        pol.sync(true);
        T gradNorm = l2Norm(pol, "grad");
        T dirNorm = l2Norm(pol, "dir");
        writeFile(fmt::format("../zeno_out/optimization_data/{}.csv", frameCnt),
                  fmt::format("{},\t{},\t{}\n", E0, gradNorm, dirNorm));
#endif
        if (!converged && iters % 8 == 0) {
            pol.sync(true);
            T E = dynamicsEnergy(pol);
            if (firstStepping || (!std::isnan(E) && E <= lastEnergy + limits<T>::epsilon() * 10.0f)) {
                pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), alpha = alpha] ZS_LAMBDA(int i) mutable {
                    vtemp.tuple(dim_c<3>, "ytmp", i) = vtemp.pack(dim_c<3>, "yn", i);
                    vtemp.tuple(dim_c<3>, "xtmp", i) = vtemp.pack(dim_c<3>, "xn", i);
                });
                lastEnergy = E;
                firstStepping = false;
            } else {
                pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), alpha = alpha] ZS_LAMBDA(int i) mutable {
                    vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "ytmp", i);
                    vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xtmp", i);
                    // TODO: when to calculate E: before or after calculating "xn"?
                });
                alpha *= alphaDecrease;
                maxIters -= (iters - 8);
                iters = -1;
                k -= 9;
                firstStepping = true;
                if constexpr (s_enableProfile) {
                    timer.tock();
                    dynamicsCnt[3]++; // total time including line-search
                    dynamicsTime[3] += timer.elapsed();
                }
                continue;
            }
        }
#else
        alpha = 0.1f;
#endif
        if (!converged)
            pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), alpha = alpha, projectDBC = projectDBC,
                                     coOffset = coOffset] ZS_LAMBDA(int i) mutable {
#if s_useChebyshevAcc
                vtemp.tuple(dim_c<3>, "yn-1", i) = vtemp.pack(dim_c<3>, "yn", i);
#endif
                if (!projectDBC || (i < coOffset))
                    vtemp.tuple(dim_c<3>, "yn", i) =
                        vtemp.pack(dim_c<3>, "yn", i) + alpha * vtemp.pack(dim_c<3>, "dir", i);
            });
#if s_useChebyshevAcc
        if (iters == 0) {
            chebyOmega = 1.0f; // omega_{k}
        } else if (iters == 1) {
            chebyOmega = 2.0f / (2.0f - chebyRho * chebyRho);
        } else {
            chebyOmega = 4.0f / (4.0f - chebyRho * chebyRho * chebyOmega);
        }
        if (!converged && iters > 0) {
            // fmt::print("\t\tChebyshev acceleration: chebyOmega: {}\n", (float)chebyOmega);
            pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), omega = chebyOmega] __device__(int vi) mutable {
                vtemp.tuple(dim_c<3>, "yn", vi) =
                    omega * vtemp.pack(dim_c<3>, "yn", vi) + (1.0f - omega) * vtemp.pack(dim_c<3>, "yn-2", vi);
            });
        }
        if (!converged)
            pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), alpha = alpha] ZS_LAMBDA(int i) mutable {
                vtemp.tuple(dim_c<3>, "yn-2", i) = vtemp.pack(dim_c<3>, "yn-1", i);
            });
#endif
        if constexpr (s_enableProfile) {
            timer.tock();
            dynamicsCnt[3]++; // total time including line-search
            dynamicsTime[3] += timer.elapsed();
        }

        if (!converged && (k % IDyn != 0))
            continue;

        /// start collision solver
        ///
        initialStepping(pol);

        if constexpr (s_enableProfile) {
            timer.tick();
            collisionCnt[3]++;
        }

        // x^{k+1}
        pol.sync(true);
#if s_testLightCache
        lightFilterConstraints(pol, dHat, "xinit");
#else
        findConstraints(pol, dHat); // DWX: paper 4.3.2; do only once in the future
#endif

        if constexpr (s_enableProfile) {
            timer.tock();
            collisionTime[0] += timer.elapsed();
            collisionTime[3] += timer.elapsed();
        }

        /// @brief backup xn for potential hard phase
        pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "xk", i) = vtemp.pack(dim_c<3>, "xn", i);
        });

        bool success = false;
        /// @note ref: sec 4.3.4

        if constexpr (s_enableProfile) {
            timer.tick();
        }

        // TODO: add reduction parameter (check whether constraints are statisfied per R iterations)
        success = collisionStep(pol, false);
        collisionCnt[0]++;
        collisionCnt[1]++;

        if constexpr (s_enableProfile) {
            timer.tock();
            collisionTime[0] += timer.elapsed();
            collisionTime[1] += timer.elapsed();
        }
#if s_useHardPhase
        if (!success) {
            /// @brief restore xn with xk for hard phase
            pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xk", i);
            });

            if constexpr (s_enableProfile) {
                timer.tick();
                collisionCnt[3]++;
            }

            if constexpr (s_enableProfile) {
                timer.tock();
                collisionTime[0] += timer.elapsed();
                collisionTime[3] += timer.elapsed();
            }

            if constexpr (s_enableProfile) {
                timer.tick();
                collisionCnt[2]++;
            }

            success = collisionStep(pol, true);

            if constexpr (s_enableProfile) {
                timer.tock();
                collisionTime[0] += timer.elapsed();
                collisionTime[2] += timer.elapsed();
            }
        }
        if (!success) {
            throw std::runtime_error("collision step failure!\n");
        }
#endif
        /// done collision solver
        if (converged)
            break;
    }
    pol.sync(true);

    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        auto xk = vtemp.pack(dim_c<3>, "xn", i);
        vtemp.tuple(dim_c<3>, "yn", i) = xk;
    });
}

struct StepClothSystem : INode {
    void apply() override {
        using namespace zs;
        auto A = get_input<FastClothSystem>("ZSClothSystem");
        zs::CppTimer timer;
        float profileTime[10] = {};
        timer.tick();

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);
        timer.tock();
        profileTime[0] += timer.elapsed();
        profileTime[1] += timer.elapsed();

        for (int subi = 0; subi != nSubsteps; ++subi) {
            timer.tick();
            A->advanceSubstep(cudaPol, (typename FastClothSystem::T)1 / nSubsteps);
            timer.tock();
            profileTime[0] += timer.elapsed();
            profileTime[2] += timer.elapsed();

            timer.tick();
            A->subStepping(cudaPol);
            timer.tock();
            profileTime[0] += timer.elapsed();
            profileTime[3] += timer.elapsed();

            timer.tick();
            A->updateVelocities(cudaPol);
            timer.tock();
            profileTime[0] += timer.elapsed();
            profileTime[4] += timer.elapsed();
        }
        // update velocity and positions
        timer.tick();
        A->writebackPositionsAndVelocities(cudaPol);
        timer.tock();
        profileTime[0] += timer.elapsed();
        profileTime[5] += timer.elapsed();
#if s_debugOutput
        A->frameCnt++;
#endif

        if constexpr (FastClothSystem::s_enableProfile) {
            auto str = fmt::format(
                "total time: {} ({}, {}, {}, {}, {}); dynamics time [{}]: ({}: [grad_hess] {}, [cgsolve] {});"
                " dynamics-including-linesearch[{}]: {}; collision time [{}, {}, {}; "
                "{}; {}; {}]: ({}(total): "
                "{}(soft-phase-total), {}(hard), {}(findConstraints); {}(constraintSatisfied); {}(soft-phase-iters). "
                "aux time: (bvh build/cd) [{}, {}({})], (sh build/cd) [{}, {}], initInterpolationTime: {}\n",
                profileTime[0], profileTime[1], profileTime[2], profileTime[3], profileTime[4], profileTime[5],
                A->dynamicsCnt[0], A->dynamicsTime[0], A->dynamicsTime[1], A->dynamicsTime[2], A->dynamicsCnt[3],
                A->dynamicsTime[3], A->collisionCnt[0], A->collisionCnt[1], A->collisionCnt[2], A->collisionCnt[3],
                A->collisionCnt[4], A->collisionCnt[5], A->collisionTime[0], A->collisionTime[1], A->collisionTime[2],
                A->collisionTime[3], A->collisionTime[4], A->collisionTime[5], A->auxTime[0], A->auxTime[1],
                A->auxCnt[1], A->auxTime[2], A->auxTime[3], A->initInterpolationTime);
            zeno::log_warn(str);
            ZS_WARN(str);
        }

        set_output("ZSClothSystem", A);
    }
};

ZENDEFNODE(StepClothSystem, {{
                                 "ZSClothSystem",
                                 {gParamType_Int, "num_substeps", "1"},
                                 {gParamType_Float, "dt", "0.01"},
                             },
                             {"ZSClothSystem"},
                             {},
                             {"FEM"}});

struct FastClothSystemForceField : INode {
    using T = typename FastClothSystem::T; 

    template <typename VelSplsViewT>
    void computeForce(zs::CudaExecutionPolicy &cudaPol, float windDragCoeff, float windDensity, int vOffset,
                      VelSplsViewT velLs, typename FastClothSystem::tiles_t &vtemp,
                      const typename FastClothSystem::tiles_t &eles) {
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

        auto A = get_input<FastClothSystem>("ZSFastClothSystem");
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

        set_output("ZSFastClothSystem", A);
    }
};

ZENDEFNODE(FastClothSystemForceField,
           {
               {"ZSFastClothSystem", "ZSLevelSet", {gParamType_Float, "wind_drag", "0"}, {gParamType_Float, "wind_density", "1"}},
               {"ZSFastClothSystem"},
               {},
               {"FEM"},
           });

} // namespace zeno