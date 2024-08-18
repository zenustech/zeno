#include "Cloth.cuh"
#include "zensim/Logger.hpp"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

typename ClothSystem::T ClothSystem::infNorm(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    using T = typename ClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(vtemp.size());
    temp.resize(nwarps);
    cudaPol(range(vtemp.size()), [data = proxy<space>({}, vtemp), res = proxy<space>(temp), n = vtemp.size(),
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
typename ClothSystem::T ClothSystem::dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0,
                                         const zs::SmallString tag1) {
    using namespace zs;
    using T = typename ClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(vtemp.size());
    temp.resize(nwarps);
    temp.reset(0);
    cudaPol(range(vtemp.size()), [data = proxy<space>({}, vtemp), res = proxy<space>(temp), n = vtemp.size(),
                                  offset0 = vtemp.getPropertyOffset(tag0),
                                  offset1 = vtemp.getPropertyOffset(tag1)] __device__(int pi) mutable {
        auto v0 = data.pack(dim_c<3>, offset0, pi);
        auto v1 = data.pack(dim_c<3>, offset1, pi);
        reduce_to(pi, n, v0.dot(v1), res[pi / 32]);
    });
    return reduce(cudaPol, temp, thrust::plus<T>{});
}

void ClothSystem::computeConstraints(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(Collapse{numBouDofs}, [vtemp = proxy<space>({}, vtemp), coOffset = coOffset] __device__(int vi) mutable {
        vi += coOffset;
        auto xtarget = vtemp.pack<3>("xtilde", vi);
        auto x = vtemp.pack<3>("xn", vi);
        vtemp.tuple(dim_c<3>, "cons", vi) = x - xtarget;
    });
}
bool ClothSystem::areConstraintsSatisfied(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    computeConstraints(pol);
    auto res = constraintResidual(pol);
    return res < s_constraint_residual;
}
typename ClothSystem::T ClothSystem::constraintResidual(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (projectDBC)
        return 0;
    temp.resize(numBouDofs * 2);
    pol(Collapse{numBouDofs}, [vtemp = proxy<space>({}, vtemp), den = temp.data(), num = temp.data() + numBouDofs,
                               coOffset = coOffset] __device__(int vi) mutable {
        vi += coOffset;
        auto cons = vtemp.pack<3>("cons", vi);
        auto xt = vtemp.pack<3>("xhat", vi);
        auto xtarget = vtemp.pack<3>("xtilde", vi);
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

void ClothSystem::computeInertialAndGravityGradientAndHessian(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // inertial
    cudaPol(zs::range(coOffset),
            [vtemp = proxy<space>({}, vtemp), dt = dt, projectDBC = projectDBC] ZS_LAMBDA(int i) mutable {
                auto m = vtemp("ws", i);
                vtemp.tuple<3>("grad", i) =
                    vtemp.pack<3>("grad", i) - m * (vtemp.pack<3>("xn", i) - vtemp.pack<3>("xtilde", i));

                // prepare preconditioner
                for (int d = 0; d != 3; ++d)
                    vtemp("P", d * 3 + d, i) += m;
            });
    // extforce (only grad modified)
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
void computeElasticGradientAndHessianImpl(zs::CudaExecutionPolicy &cudaPol, typename ClothSystem::tiles_t &vtemp,
                                          typename ClothSystem::PrimitiveHandle &primHandle, const Model &model,
                                          typename ClothSystem::T dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename ClothSystem::mat3;
    using vec3 = typename ClothSystem::vec3;
    using T = typename ClothSystem::T;
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

                    vec3 xs[2] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1])};
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
                    vec3 xs[3] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1]),
                                  vtemp.pack(dim_c<3>, "xn", inds[2])};
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
                    vec3 xs[4] = {vtemp.pack<3>("xn", inds[0]), vtemp.pack<3>("xn", inds[1]),
                                  vtemp.pack<3>("xn", inds[2]), vtemp.pack<3>("xn", inds[3])};

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

void ClothSystem::computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol) {
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

void ClothSystem::newtonKrylov(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// optimizer
    for (int newtonIter = 0; newtonIter != PNCap; ++newtonIter) {
        // check constraints
        if (!projectDBC) {
            computeConstraints(pol);
            auto cr = constraintResidual(pol);
            if (cr < s_constraint_residual) {
                // zeno::log_info("satisfied cons res [{}] at newton iter [{}]\n", cr, newtonIter);
                projectDBC = true;
            }
            fmt::print(fg(fmt::color::alice_blue), "newton iter {} cons residual: {}\n", newtonIter, cr);
        }
        // PRECOMPUTE
        if (enableContact) {
            findCollisionConstraints(pol, dHat);
        }
        // GRAD, HESS, P
        pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::zeros();
            vtemp.tuple(dim_c<3>, "grad", i) = vec3::zeros();
        });
        computeInertialAndGravityGradientAndHessian(pol);
        computeElasticGradientAndHessian(pol);
        if (enableContact) {
            computeCollisionGradientAndHessian(pol);
        }
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
            if (zs::abs(zs::determinant(mat)) > limits<T>::epsilon() * 10)
                vtemp.tuple<9>("P", i) = inverse(mat);
            else
                vtemp.tuple<9>("P", i) = mat3::identity();
        });
        // prepare float edition
        // convertHessian(pol);
        // CG SOLVE
        cgsolve(pol);
        // CHECK PN CONDITION
        T res = infNorm(pol) / dt;
        T cons_res = constraintResidual(pol);
        if (res < targetGRes && cons_res == 0) {
            break;
        }
        ZS_INFO(fmt::format("substep {} newton iter {}: direction residual(/dt) {}, "
                            "grad residual {}\n",
                            substep, newtonIter, res, res * dt));
        // OMIT LINESEARCH
        pol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, "xn", i) = vtemp.pack(dim_c<3>, "xn", i) + vtemp.pack(dim_c<3>, "dir", i);
        });
    }
}

struct StepClothSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto A = get_input<ClothSystem>("ZSClothSystem");

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);

        for (int subi = 0; subi != nSubsteps; ++subi) {
            A->advanceSubstep(cudaPol, (typename ClothSystem::T)1 / nSubsteps);

            A->newtonKrylov(cudaPol);

            A->updateVelocities(cudaPol);
        }
        // update velocity and positions
        A->writebackPositionsAndVelocities(cudaPol);

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

} // namespace zeno