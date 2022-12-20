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
    cudaPol(range(numDofs), [data = proxy<space>({}, vtemp), res = proxy<space>(temp), n = numDofs,
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
typename FastClothSystem::T FastClothSystem::dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0,
                                                 const zs::SmallString tag1) {
    using namespace zs;
    using T = typename FastClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(numDofs);
    temp.resize(nwarps);
    temp.reset(0);
    cudaPol(range(numDofs), [data = proxy<space>({}, vtemp), res = proxy<space>(temp), n = numDofs,
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
    pol(Collapse{numBouDofs}, [vtemp = proxy<space>({}, vtemp), coOffset = coOffset] __device__(int vi) mutable {
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
    pol(Collapse{numBouDofs}, [vtemp = proxy<space>({}, vtemp), den = temp.data(), num = temp.data() + numBouDofs,
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
    cudaPol(zs::range(coOffset), [vtemp = proxy<space>({}, vtemp), dt = dt, sigma = sigma] ZS_LAMBDA(int i) mutable {
        auto m = vtemp("ws", i);
        auto yn = vtemp.pack<3>("yn", i);
        vtemp.tuple<3>("grad", i) = vtemp.pack<3>("grad", i) - m * (yn - vtemp.pack<3>("ytilde", i)) -
                                    m * sigma * (yn - vtemp.pack<3>("xn", i));

        // prepare preconditioner
        for (int d = 0; d != 3; ++d)
            vtemp("P", d * 3 + d, i) += (m * (sigma + 1.0f));
    });
    /// @brief extforce (only grad modified)
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary()) // skip soft boundary
            continue;
        cudaPol(zs::range(primHandle.getVerts().size()), [vtemp = proxy<space>({}, vtemp), extAccel = extAccel, dt = dt,
                                                          vOffset = primHandle.vOffset] ZS_LAMBDA(int vi) mutable {
            vi += vOffset;
            auto m = vtemp("ws", vi);
            vtemp.tuple(dim_c<3>, "grad", vi) = vtemp.pack(dim_c<3>, "grad", vi) + m * extAccel * dt * dt;
        });
    }
    if (vtemp.hasProperty("extf")) {
        cudaPol(zs::range(coOffset), [vtemp = proxy<space>({}, vtemp), dt = dt] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, "grad", vi) =
                vtemp.pack(dim_c<3>, "grad", vi) + vtemp.pack(dim_c<3>, "extf", vi) * dt * dt;
        });
    }
}

/// elasticity
template <typename Model>
void computeElasticGradientAndHessianImpl(zs::CudaExecutionPolicy &cudaPol, typename FastClothSystem::tiles_t &vtemp,
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
                [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, primHandle.etemp),
                 eles = proxy<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset, n = primHandle.getEles().size()] __device__(int ei) mutable {
                    auto inds = eles.pack(dim_c<2>, "inds", ei).template reinterpret_bits<int>() + vOffset;

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

                    // rotate and project
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
                [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, primHandle.etemp),
                 eles = proxy<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset] __device__(int ei) mutable {
                    auto IB = eles.template pack<2, 2>("IB", ei);
                    auto inds = eles.pack(dim_c<3>, "inds", ei).reinterpret_bits(int_c) + vOffset;
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
                    auto vecP = flatten(model.mu * Pstretch + (model.mu * 0.3) * Pshear);
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
                        dPdF *= (model.mu * 0.3);
                        return dPdF;
                    };
                    auto He = stretchHessian() + shearHessian();
                    auto H = dFdX.transpose() * He * dFdX * (dt * dt * vole);

                    // rotate and project
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
                [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, primHandle.etemp),
                 eles = proxy<space>({}, primHandle.getEles()), model, gradOffset = vtemp.getPropertyOffset("grad"),
                 dt = dt, vOffset = primHandle.vOffset] __device__(int ei) mutable {
                    auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                    auto inds = eles.pack(dim_c<4>, "inds", ei).template reinterpret_bits<int>() + vOffset;
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

                    // rotate and project
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

void FastClothSystem::computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    for (auto &primHandle : prims) {
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, vtemp, primHandle, elasticModel, dt);
        })(primHandle.getModels().getElasticModel());
    }
    for (auto &primHandle : auxPrims) {
        using ModelT = RM_CVREF_T(primHandle.getModels().getElasticModel());
        const ModelT &model = primHandle.modelsPtr ? primHandle.getModels().getElasticModel() : ModelT{};
        match([&](auto &elasticModel) {
            computeElasticGradientAndHessianImpl(cudaPol, vtemp, primHandle, elasticModel, dt);
        })(model);
    }
}

void FastClothSystem::subStepping(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    /// y0, x0
    // y0, x0
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), dt = dt, coOffset = coOffset] ZS_LAMBDA(int i) mutable {
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
        pol(range(numDofs), [spgv = proxy<space>(spg), vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
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

        pol(range(numDofs), [spgv = proxy<space>(spg), vtemp = proxy<space>({}, vtemp)] __device__(int pi) mutable {
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
        });
        pol(range((std::size_t)nbs * spg.block_size),
            [grid = proxy<space>({}, spg._grid)] __device__(std::size_t cellno) mutable {
                using T = typename RM_CVREF_T(grid)::value_type;
                if (grid("w", cellno) > limits<T>::epsilon())
                    grid.tuple(dim_c<3>, "dir", cellno) = grid.pack(dim_c<3>, "dir", cellno) / grid("w", cellno);
            });
        pol(range(numDofs), [spgv = proxy<space>(spg), vtemp = proxy<space>({}, vtemp)] __device__(int pi) mutable {
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
            vtemp.tuple(dim_c<3>, "dir", pi) = dir;
        });
    }
    {
        const auto ratio = (T)1 / (T)IInit;
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), ratio] ZS_LAMBDA(int i) mutable {
            auto dir = vtemp.pack(dim_c<3>, "dir", i);
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
            pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                vtemp.tuple(dim_c<3>, "xk", i) = vtemp.pack(dim_c<3>, "xn", i);
            });
            /// @brief collision handling
            bool success = false;
            /// @note ref: sec 4.3.4
            int r = 0;
            for (; r != R; ++r) {
                if (success = collisionStep(pol, false); success)
                    break;
            }
#if 0
            if (success) {
                fmt::print(fg(fmt::color::alice_blue), "done pure soft collision iters {} out of {}\n", r, R);
                pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), n = numDofs] ZS_LAMBDA(int i) mutable {
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

            if (!success) {
                /// @brief restore xn with xk for hard phase
                pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                    vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xk", i);
                });
                findConstraints(pol, dHat, "xn"); 
                success = collisionStep(pol, true);
            }
            if (!success)
                throw std::runtime_error("collision step in initialization fails!\n");
            /// done collision solver

            if (++i == IInit)
                break;

            /// @brief update xinit for the next initialization iteration
            pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), ratio] ZS_LAMBDA(int i) mutable {
                auto dir = vtemp.pack(dim_c<3>, "dir", i);
                vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "yn", i) + ratio * dir;
            });
        }
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA (int i) mutable {
            vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "yk", i); // restore yn from yk 
        }); 
    }

    /// @brief collect pairs only once depending on x^init
    /// @note ref: Sec 4.3.2

    /// optimizer
    for (int k = 0; k != K; ++k) {
        // check constraints
        if (!projectDBC) {
            computeBoundaryConstraints(pol);
            auto cr = boundaryConstraintResidual(pol);
            if (cr < s_constraint_residual) {
                projectDBC = true;
            }
            fmt::print(fg(fmt::color::alice_blue), "iteration {} cons residual: {}\n", k, cr);
        }
        // GRAD, HESS, P
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::zeros();
            vtemp.tuple(dim_c<3>, "grad", i) = vec3::zeros();
        });
        computeInertialAndCouplingAndForceGradient(pol);
        computeElasticGradientAndHessian(pol);

        // APPLY BOUNDARY CONSTRAINTS, PROJ GRADIENT
        if (!projectDBC) {
            // grad
            pol(zs::range(numBouDofs), [vtemp = proxy<space>({}, vtemp), boundaryKappa = boundaryKappa,
                                        coOffset = coOffset] ZS_LAMBDA(int i) mutable {
                i += coOffset;
                // computed during the previous constraint residual check
                auto cons = vtemp.pack(dim_c<3>, "cons", i);
                auto w = vtemp("ws", i);
                vtemp.tuple(dim_c<3>, "grad", i) = vtemp.pack(dim_c<3>, "grad", i) - boundaryKappa * w * cons;
                {
                    for (int d = 0; d != 3; ++d)
                        vtemp("P", 4 * d, i) += boundaryKappa * w;
                }
            });
            // hess (embedded in multiply)
        }
        project(pol, "grad");
        // PREPARE P
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            auto mat = vtemp.pack<3, 3>("P", i);
#if 0
            if (zs::abs(zs::determinant(mat)) > limits<T>::epsilon() * 10)
                vtemp.tuple<9>("P", i) = inverse(mat);
            else
                vtemp.tuple<9>("P", i) = mat3::identity();
#else 
            // diag preconditioner for tiny node mass 
            mat3 tempP; 
            for (int d = 0; d < 3; d++)
            {
                auto elem = vtemp("P", d * 3 + d, i);
                if (elem > limits<T>::epsilon() * 10)
                    tempP(d, d) = 1.0f / elem;
                else 
                    tempP(d, d) = 1.0f; 
            }
            vtemp.tuple(dim_c<9>, "P", i) = tempP; 
#endif 
        });
        // prepare float edition
        // convertHessian(pol);
        // CG SOLVE
        cgsolve(pol);
        // CHECK PN CONDITION
        T res = infNorm(pol) / dt;
        T cons_res = boundaryConstraintResidual(pol);
        if (res < targetGRes && cons_res == 0) {
            break;
        }
        ZS_INFO(fmt::format("substep {} iteration {}: direction residual(/dt) {}, "
                            "grad residual {}\n",
                            substep, k, res, res * dt));
        // OMIT LINESEARCH
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "yn", i) = vtemp.pack(dim_c<3>, "yn", i) + vtemp.pack(dim_c<3>, "dir", i);
        });

        /// start collision solver
        ///
        initialStepping(pol); 
        // x^{k+1}
        findConstraints(pol, dHat); // DWX: paper 4.3.2; do only once in the future

        /// @brief backup xn for potential hard phase
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "xk", i) = vtemp.pack(dim_c<3>, "xn", i);
        });
        bool success = false;
        /// @note ref: sec 4.3.4
        for (int r = 0; r != R; ++r) {
            if (success = collisionStep(pol, false); success)
                break;
        }
        if (!success) {
            /// @brief restore xn with xk for hard phase
            pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xk", i);
            });
            findConstraints(pol, dHat, "xn");
            success = collisionStep(pol, true);
        }
        if (!success) {
            throw std::runtime_error("collision step failure!\n");
        }
        /// done collision solver
    }

    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        auto xk = vtemp.pack(dim_c<3>, "xn", i);
        vtemp.tuple(dim_c<3>, "yn", i) = xk;
    });
}

struct StepClothSystem : INode {
    void apply() override {
        using namespace zs;
        auto A = get_input<FastClothSystem>("ZSClothSystem");

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);

        for (int subi = 0; subi != nSubsteps; ++subi) {
            A->advanceSubstep(cudaPol, (typename FastClothSystem::T)1 / nSubsteps);

            A->subStepping(cudaPol);

            A->updateVelocities(cudaPol);
        }
        // update velocity and positions
        A->writebackPositionsAndVelocities(cudaPol);

        set_output("ZSClothSystem", A);
    }
};

ZENDEFNODE(StepClothSystem, {{
                                 "ZSClothSystem",
                                 {"int", "num_substeps", "1"},
                                 {"float", "dt", "0.01"},
                             },
                             {"ZSClothSystem"},
                             {},
                             {"FEM"}});

} // namespace zeno