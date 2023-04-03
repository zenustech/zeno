#include "FastCloth.cuh"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

void FastClothSystem::project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    // only project boundary (character)
    if (projectDBC)
        pol(zs::range(numBouDofs), [vtemp = view<space>({}, vtemp), tagOffset = vtemp.getPropertyOffset(tag),
                                    coOffset = coOffset] ZS_LAMBDA(int vi) mutable {
            vi += coOffset;
#pragma unroll
            for (int d = 0; d != 3; ++d)
                vtemp(tagOffset + d, vi) = 0;
        });
}

void FastClothSystem::precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag,
                                   const zs::SmallString dstTag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    // precondition
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), srcOffset = vtemp.getPropertyOffset(srcTag),
                             dstOffset = vtemp.getPropertyOffset(dstTag)] ZS_LAMBDA(int vi) mutable {
        vtemp.tuple(dim_c<3>, dstOffset, vi) = vtemp.pack(dim_c<3, 3>, "P", vi) * vtemp.pack(dim_c<3>, srcOffset, vi);
    });
}

void FastClothSystem::multiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    constexpr auto execTag = wrapv<space>{};
    // dx -> b
    auto wsOffset = vtemp.getPropertyOffset("ws");
    auto dxOffset = vtemp.getPropertyOffset(dxTag);
    auto bOffset = vtemp.getPropertyOffset(bTag);

    /// @brief initialize
    pol(range(numDofs), [execTag, vtemp = view<space>({}, vtemp), bOffset] ZS_LAMBDA(int vi) mutable {
        vtemp.tuple(dim_c<3>, bOffset, vi) = vec3::zeros();
    });
    /// @brief inertial and coupling
    pol(zs::range(coOffset), [execTag, vtemp = view<space>(vtemp), sigma = sigma * dt * dt, wsOffset, dxOffset,
                              bOffset] __device__(int i) mutable {
        auto m = vtemp(wsOffset, i);
        auto dx = vtemp.pack(dim_c<3>, dxOffset, i) * m * (sigma + 1.0f);
        for (int d = 0; d != 3; ++d)
            atomic_add(execTag, &vtemp(bOffset + d, i), dx(d));
    });
    /// @brief elasticity
    for (auto &primHandle : prims) {
        auto &eles = primHandle.getEles();
        // elasticity
        if (primHandle.category == ZenoParticles::curve) {
            if (primHandle.isBoundary() && !primHandle.isAuxiliary())
                continue;
            pol(Collapse{eles.size(), 32},
                [execTag, etemp = view<space>(primHandle.etemp), vtemp = view<space>(vtemp),
                 eles = view<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
                 hOffset = primHandle.etemp.getPropertyOffset("He"), dxOffset, bOffset,
                 vOffset = primHandle.vOffset] ZS_LAMBDA(int ei, int tid) mutable {
                    int rowid = tid / 5;
                    int colid = tid % 5;
                    auto inds = eles.pack(dim_c<2>, indsOffset, ei, int_c) + vOffset;
                    T entryH = 0, entryDx = 0, entryG = 0;
                    if (tid < 30) {
                        entryH = etemp(hOffset + rowid * 6 + colid, ei);
                        entryDx = vtemp(dxOffset + colid % 3, inds[colid / 3]);
                        entryG = entryH * entryDx;
                        if (colid == 0) {
                            entryG += etemp(hOffset + rowid * 6 + 5, ei) * vtemp(dxOffset + 2, inds[1]);
                        }
                    }
                    for (int iter = 1; iter <= 4; iter <<= 1) {
                        T tmp = __shfl_down_sync(0xFFFFFFFF, entryG, iter);
                        if (colid + iter < 5 && tid < 30)
                            entryG += tmp;
                    }
                    if (colid == 0 && rowid < 6)
                        atomic_add(execTag, &vtemp(bOffset + rowid % 3, inds[rowid / 3]), entryG);
                });
        } else if (primHandle.category == ZenoParticles::surface) {
            if (primHandle.isBoundary())
                continue;
            pol(range(eles.size() * 81),
                [execTag, etemp = view<space>(primHandle.etemp), vtemp = view<space>(vtemp),
                 eles = view<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
                 hOffset = primHandle.etemp.getPropertyOffset("He"), dxOffset, bOffset, vOffset = primHandle.vOffset,
                 n = eles.size() * 81] ZS_LAMBDA(int idx) mutable {
                    constexpr int dim = 3;
                    __shared__ int offset;
                    // directly use PCG_Solve_AX9_b2 from kemeng huang
                    int ei = idx / 81;
                    int entryId = idx % 81;
                    int MRid = entryId / 9;
                    int MCid = entryId % 9;
                    int vId = MCid / dim;
                    int axisId = MCid % dim;
                    int GRtid = idx % 9;

                    auto inds = eles.pack(dim_c<3>, indsOffset, ei, int_c) + vOffset;
                    T rdata = etemp(hOffset + entryId, ei) * vtemp(dxOffset + axisId, inds[vId]);

                    if (threadIdx.x == 0)
                        offset = 9 - GRtid;
                    __syncthreads();

                    int BRid = (threadIdx.x - offset + 9) / 9;
                    int landidx = (threadIdx.x - offset) % 9;
                    if (BRid == 0) {
                        landidx = threadIdx.x;
                    }

                    auto [mask, numValid] = warp_mask(idx, n);
                    int laneId = threadIdx.x & 0x1f;
                    bool bBoundary = (landidx == 0) || (laneId == 0);

                    unsigned int mark = __ballot_sync(mask, bBoundary); // a bit-mask
                    mark = __brev(mark);
                    unsigned int interval = zs::math::min(__clz(mark << (laneId + 1)), 31 - laneId);

                    for (int iter = 1; iter < 9; iter <<= 1) {
                        T tmp = __shfl_down_sync(mask, rdata, iter);
                        if (interval >= iter && laneId + iter < numValid)
                            rdata += tmp;
                    }

                    if (bBoundary)
                        atomic_add(exec_cuda, &vtemp(bOffset + MRid % 3, inds[MRid / 3]), rdata);
                });
        } else if (primHandle.category == ZenoParticles::tet)
            pol(range(eles.size() * 144),
                [execTag, etemp = view<space>(primHandle.etemp), vtemp = view<space>(vtemp),
                 eles = view<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
                 hOffset = primHandle.etemp.getPropertyOffset("He"), dxOffset, bOffset, vOffset = primHandle.vOffset,
                 n = eles.size() * 144] ZS_LAMBDA(int idx) mutable {
                    constexpr int dim = 3;
                    __shared__ int offset;
                    // directly use PCG_Solve_AX9_b2 from kemeng huang
                    int Hid = idx / 144;
                    int entryId = idx % 144;
                    int MRid = entryId / 12;
                    int MCid = entryId % 12;
                    int vId = MCid / dim;
                    int axisId = MCid % dim;
                    int GRtid = idx % 12;

                    auto inds = eles.pack(dim_c<4>, indsOffset, Hid, int_c) + vOffset;
                    T rdata = etemp(hOffset + entryId, Hid) * vtemp(dxOffset + axisId, inds[vId]);

                    if (threadIdx.x == 0)
                        offset = 12 - GRtid;
                    __syncthreads();

                    int BRid = (threadIdx.x - offset + 12) / 12;
                    int landidx = (threadIdx.x - offset) % 12;
                    if (BRid == 0) {
                        landidx = threadIdx.x;
                    }

                    auto [mask, numValid] = warp_mask(idx, n);
                    int laneId = threadIdx.x & 0x1f;
                    bool bBoundary = (landidx == 0) || (laneId == 0);

                    unsigned int mark = __ballot_sync(mask, bBoundary); // a bit-mask
                    mark = __brev(mark);
                    unsigned int interval = zs::math::min(__clz(mark << (laneId + 1)), 31 - laneId);

                    for (int iter = 1; iter < 12; iter <<= 1) {
                        T tmp = __shfl_down_sync(mask, rdata, iter);
                        if (interval >= iter && laneId + iter < numValid)
                            rdata += tmp;
                    }

                    if (bBoundary)
                        atomic_add(exec_cuda, &vtemp(bOffset + MRid % 3, inds[MRid / 3]), rdata);
                });
    }
    /// @brief hard binding constraint
    for (auto &primHandle : auxPrims) {
        auto &eles = primHandle.getEles();
        // soft bindings
        if (primHandle.category == ZenoParticles::curve) {
            pol(Collapse{eles.size(), 32},
                [execTag, etemp = view<space>(primHandle.etemp), vtemp = view<space>(vtemp),
                 eles = view<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
                 hOffset = primHandle.etemp.getPropertyOffset("He"), dxOffset, bOffset,
                 vOffset = primHandle.vOffset] ZS_LAMBDA(int ei, int tid) mutable {
                    int rowid = tid / 5;
                    int colid = tid % 5;
                    auto inds = eles.pack(dim_c<2>, indsOffset, ei, int_c) + vOffset;
                    T entryH = 0, entryDx = 0, entryG = 0;
                    if (tid < 30) {
                        entryH = etemp(hOffset + rowid * 6 + colid, ei);
                        entryDx = vtemp(dxOffset + colid % 3, inds[colid / 3]);
                        entryG = entryH * entryDx;
                        if (colid == 0) {
                            entryG += etemp(hOffset + rowid * 6 + 5, ei) * vtemp(dxOffset + 2, inds[1]);
                        }
                    }
                    for (int iter = 1; iter <= 4; iter <<= 1) {
                        T tmp = __shfl_down_sync(0xFFFFFFFF, entryG, iter);
                        if (colid + iter < 5 && tid < 30)
                            entryG += tmp;
                    }
                    if (colid == 0 && rowid < 6)
                        atomic_add(execTag, &vtemp(bOffset + rowid % 3, inds[rowid / 3]), entryG);
                });
        }
    }
    /// @brief boundary constraint
    if (!projectDBC) {
        pol(range(numBouDofs), [execTag, vtemp = view<space>(vtemp), dxOffset, bOffset, wsOffset, coOffset = coOffset,
                                boundaryKappa = boundaryKappa] ZS_LAMBDA(int vi) mutable {
            vi += coOffset;
            auto dx = vtemp.pack(dim_c<3>, dxOffset, vi);
            auto w = vtemp(wsOffset, vi);
            for (int d = 0; d != 3; ++d)
                atomic_add(execTag, &vtemp(bOffset + d, vi), boundaryKappa * w * dx(d));
        });
    }
}

void FastClothSystem::cgsolve(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// copy diagonal block preconditioners
    pol.sync(false);
    /// solve for A dir = grad;
    // initial guess for hard boundary constraints
    pol(zs::range(numDofs),
        [vtemp = view<space>({}, vtemp), coOffset = coOffset, dt = dt, dirOffset = vtemp.getPropertyOffset("dir"),
         xtildeOffset = vtemp.getPropertyOffset("ytilde"),
         ynOffset = vtemp.getPropertyOffset("yn")] ZS_LAMBDA(int i) mutable {
            if (i < coOffset) {
                vtemp.tuple(dim_c<3>, dirOffset, i) = vec3::zeros();
            } else {
                vtemp.tuple(dim_c<3>, dirOffset, i) =
                    (vtemp.pack(dim_c<3>, xtildeOffset, i) - vtemp.pack(dim_c<3>, ynOffset, i)) * dt;
            }
        });
    // temp = A * dir
    multiply(pol, "dir", "temp");
    // r = grad - temp
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), rOffset = vtemp.getPropertyOffset("r"),
                             gradOffset = vtemp.getPropertyOffset("grad"),
                             tempOffset = vtemp.getPropertyOffset("temp")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, rOffset, i) = vtemp.pack(dim_c<3>, gradOffset, i) - vtemp.pack(dim_c<3>, tempOffset, i);
    });
    project(pol, "r");
    precondition(pol, "r", "q");
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), pOffset = vtemp.getPropertyOffset("p"),
                             qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, pOffset, i) = vtemp.pack(dim_c<3>, qOffset, i);
    });
    T zTrk = dot(pol, "r", "q");
    T residualPreconditionedNorm2 = zTrk;
    T localTol2 = cgRel * cgRel * residualPreconditionedNorm2;
    int iter = 0;

    CppTimer timer;
    timer.tick();
    for (; iter != CGCap; ++iter) {
        if (iter % 50 == 0)
            fmt::print("cg iter: {}, norm2: {} (zTrk: {})\n", iter, residualPreconditionedNorm2, zTrk);

        if (residualPreconditionedNorm2 <= localTol2)
            break;
        multiply(pol, "p", "temp");
        project(pol, "temp"); // need further checking hessian!

        T alpha = zTrk / dot(pol, "temp", "p");
        pol(range(numDofs), [vtemp = view<space>({}, vtemp), dirOffset = vtemp.getPropertyOffset("dir"),
                             pOffset = vtemp.getPropertyOffset("p"), rOffset = vtemp.getPropertyOffset("r"),
                             tempOffset = vtemp.getPropertyOffset("temp"), alpha] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, dirOffset, vi) =
                vtemp.pack(dim_c<3>, dirOffset, vi) + alpha * vtemp.pack(dim_c<3>, pOffset, vi);
            vtemp.tuple(dim_c<3>, rOffset, vi) =
                vtemp.pack(dim_c<3>, rOffset, vi) - alpha * vtemp.pack(dim_c<3>, tempOffset, vi);
        });

        precondition(pol, "r", "q");
        T zTrkLast = zTrk;
        zTrk = dot(pol, "q", "r");
        T beta = zTrk / zTrkLast;
        pol(range(numDofs), [vtemp = view<space>({}, vtemp), beta, pOffset = vtemp.getPropertyOffset("p"),
                             qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, pOffset, vi) =
                vtemp.pack(dim_c<3>, qOffset, vi) + beta * vtemp.pack(dim_c<3>, pOffset, vi);
        });

        residualPreconditionedNorm2 = zTrk;
    } // end cg step
    pol.sync(true);
    timer.tock(fmt::format("{} cgiters", iter));
}

void FastClothSystem::newtonDynamicsStep(zs::CudaExecutionPolicy &pol) {
    // GRAD, HESS, P
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::zeros();
        vtemp.tuple(dim_c<3>, "grad", i) = vec3::zeros();
    });
    computeInertialAndCouplingAndForceGradient(pol);
    computeElasticGradientAndHessian(pol);

    // APPLY BOUNDARY CONSTRAINTS, PROJ GRADIENT
    if (!projectDBC) {
        // grad
        pol(zs::range(numBouDofs), [vtemp = view<space>({}, vtemp), boundaryKappa = boundaryKappa,
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

    if constexpr (s_enableProfile) {
        timer.tock();
        dynamicsTime[0] += timer.elapsed();
        dynamicsTime[1] += timer.elapsed();
    }

    // PREPARE P
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
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

    if constexpr (s_enableProfile)
        timer.tick();

    // CG SOLVE
    cgsolve(pol);

    if constexpr (s_enableProfile) {
        timer.tock();
        dynamicsCnt[0]++;
        dynamicsTime[0] += timer.elapsed();
        dynamicsTime[2] += timer.elapsed();
    }
}

void FastClothSystem::gdDynamicsStep(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // GRAD, HESS, P
    if constexpr (s_enableProfile)
        timer.tick();
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
#if 1
        vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::zeros();
#else 
        vtemp.tuple(dim_c<3>, "P", i) = vec3::zeros();
#endif
        vtemp.tuple(dim_c<3>, "grad", i) = vec3::zeros();
    });
    computeInertialAndCouplingAndForceGradient(pol);
    computeElasticGradientAndHessian(pol);

    // APPLY BOUNDARY CONSTRAINTS, PROJ GRADIENT
    if (!projectDBC) {
        // grad
        pol(zs::range(numBouDofs), [vtemp = view<space>({}, vtemp), boundaryKappa = boundaryKappa,
                                    coOffset = coOffset] ZS_LAMBDA(int i) mutable {
            i += coOffset;
            // computed during the previous constraint residual check
            auto cons = vtemp.pack(dim_c<3>, "cons", i);
            auto w = vtemp("ws", i);
            vtemp.tuple(dim_c<3>, "grad", i) = vtemp.pack(dim_c<3>, "grad", i) - boundaryKappa * w * cons;
            {
                for (int d = 0; d != 3; ++d)
#if !s_useGDDiagHess
                    vtemp("P", 4 * d, i) += boundaryKappa * w;
#else 
                    vtemp("P", d, i) += boundaryKappa * w;
#endif
            }
        });
        // hess (embedded in multiply)
    }
#if !s_useGDDiagHess
    pol(zs::range(vtemp.size()),
        [vtemp = view<space>({}, vtemp), coOffset = coOffset, projectDBC = projectDBC] __device__(int vi) mutable {
            if (projectDBC && (vi >= coOffset)) {
                vtemp.tuple(dim_c<3>, "dir", vi) = vec3::zeros();
                return;
            }
            auto grad = vtemp.pack(dim_c<3>, "grad", vi);
            auto pre = vtemp.pack(dim_c<3, 3>, "P", vi);
            vtemp.tuple(dim_c<3>, "dir", vi) = inverse(pre) * grad;
        });
#else
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp), n = numDofs] __device__(int i) mutable {
        for (int d = 0; d != 3; d++) {
            auto grad = vtemp("grad", d, i);
            auto pre = vtemp("P", d, i);
            if (pre < limits<T>::epsilon() * 10.0f)
                pre = 1.0f;
            vtemp("dir", d, i) = grad / pre;
        }
    });
#endif
    if constexpr (s_enableProfile) {
        timer.tock();
        dynamicsTime[0] += timer.elapsed();
    }
}

template <typename Model>
typename FastClothSystem::T
elasticityEnergy(zs::CudaExecutionPolicy &pol, typename FastClothSystem::tiles_t &vtemp,
                 typename FastClothSystem::tiles_t &seInds, typename FastClothSystem::PrimitiveHandle &primHandle,
                 const Model &model, typename FastClothSystem::T dt, zs::Vector<typename FastClothSystem::T> &energy) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename FastClothSystem::mat3;
    using vec3 = typename FastClothSystem::vec3;
    using T = typename FastClothSystem::T;

    auto &eles = primHandle.getEles();
    auto &edges = primHandle.getSurfEdges();
    const zs::SmallString tag = "yn";
    energy.setVal(0);
    if (primHandle.category == ZenoParticles::curve) {
        if (primHandle.isBoundary() && !primHandle.isAuxiliary())
            return 0;
        // elasticity
        pol(range(eles.size()),
            [eles = view<space>({}, eles), vtemp = view<space>({}, vtemp), energy = view<space>(energy), tag,
             model = model, vOffset = primHandle.vOffset, n = eles.size()] __device__(int ei) mutable {
                auto inds = eles.pack(dim_c<2>, "inds", ei, int_c) + vOffset;
                T E;
                auto vole = eles("vol", ei);
                auto k = eles("k", ei);
                // auto k = model.mu;
                auto rl = eles("rl", ei);
                vec3 xs[2] = {vtemp.template pack<3>(tag, inds[0]), vtemp.template pack<3>(tag, inds[1])};
                auto xij = xs[1] - xs[0];
                auto lij = xij.norm();

                E = (T)0.5 * k * zs::sqr(lij - rl) * vole;

                reduce_to(ei, n, E, energy[0]);
            });
        return energy.getVal() * dt * dt;
    } else if (primHandle.category == ZenoParticles::surface) {
        if (primHandle.isBoundary())
            return 0;
            // elasticity
#if s_useMassSpring
#if 1
        pol(range(edges.size()),
            [seInds = view<space>({}, seInds), vtemp = view<space>({}, vtemp), energy = view<space>(energy),
             model = model, n = edges.size(), vOffset = primHandle.vOffset,
             seoffset = primHandle.seOffset] __device__(int ei) mutable {
                int sei = ei + seoffset;
                auto inds = seInds.pack(dim_c<2>, "inds", sei, int_c);
                T E;
                auto m = 0.5f * (vtemp("ws", inds[0]) + vtemp("ws", inds[1]));
                auto v0 = vtemp.pack(dim_c<3>, "yn", inds[0]);
                auto v1 = vtemp.pack(dim_c<3>, "yn", inds[1]);
                auto restL = seInds("restLen", sei);
                // auto restL = 4.2f;
                E = 0.5f * m * model.mu * zs::sqr((v0 - v1).norm() - restL);
                reduce_to(ei, n, E, energy[0]);
            });
        return energy.getVal() * dt * dt;
#endif
#else
        pol(range(eles.size()),
            [eles = view<space>({}, eles), vtemp = view<space>({}, vtemp), energy = view<space>(energy), tag,
             model = model, vOffset = primHandle.vOffset, n = eles.size()] __device__(int ei) mutable {
                auto IB = eles.pack(dim_c<2, 2>, "IB", ei);
                auto inds = eles.pack(dim_c<3>, "inds", ei, int_c) + vOffset;
                T E;
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
                auto Eshear = (model.mu * s_clothShearingCoeff) * vole * zs::sqr(f0.dot(f1));
                E = Estretch + Eshear;
                reduce_to(ei, n, E, energy[0]);
            });
        return energy.getVal() * dt * dt;
#endif
    } else if (primHandle.category == ZenoParticles::tet) {
        pol(zs::range(eles.size()),
            [vtemp = view<space>({}, vtemp), eles = view<space>({}, eles), energy = view<space>(energy), model, tag,
             vOffset = primHandle.vOffset, n = eles.size()] __device__(int ei) mutable {
                auto IB = eles.pack(dim_c<3, 3>, "IB", ei);
                auto inds = eles.pack(dim_c<4>, "inds", ei, int_c) + vOffset;
                auto vole = eles("vol", ei);
                vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]), vtemp.pack<3>(tag, inds[2]),
                              vtemp.pack<3>(tag, inds[3])};

                T E;
                mat3 F{};
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];
                auto x3x0 = xs[3] - xs[0];
                auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1], x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                F = Ds * IB;
                E = model.psi(F) * vole;
                reduce_to(ei, n, E, energy[0]);
            });
        return energy.getVal() * dt * dt;
    }
    return 0;
}

typename FastClothSystem::T FastClothSystem::dynamicsEnergy(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // elasticity energy
    T elasticE = 0;
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        match([&](auto &elasticModel) {
            elasticE += elasticityEnergy(pol, vtemp, seInds, primHandle, elasticModel, dt, temp);
        })(primHandle.getModels().getElasticModel());
    }

    temp.setVal(0);
    // inertial energy and coupling
    pol(range(coOffset),
        [vtemp = view<space>({}, vtemp), energy = view<space>(temp), n = coOffset, sigma = sigma * dt * dt, dt = dt,
         hasExtf = vtemp.hasProperty("extf"), extAccel = extAccel, BCStiffness = BCStiffness] __device__(int vi) mutable {
            auto m = vtemp("ws", vi);
            auto yn = vtemp.pack(dim_c<3>, "yn", vi);
            auto ytilde = vtemp.pack(dim_c<3>, "ytilde", vi);
            auto xn = vtemp.pack(dim_c<3>, "xn", vi);
            auto E =
                0.5f * m * ((yn - ytilde).l2NormSqr() + sigma * (yn - xn).l2NormSqr()) - m * yn.dot(extAccel) * dt * dt;
            if (hasExtf)
                E -= yn.dot(vtemp.pack(dim_c<3>, "extf", vi)) * dt * dt;
            bool isBC  = vtemp("isBC", vi) > 0.5f; 
            if (isBC)
            {
                auto BCtarget = vtemp.pack(dim_c<3>, "BCtarget", vi); 
                E += m * BCStiffness * (yn - BCtarget).l2NormSqr(); 
            }
            reduce_to(vi, n, E, energy[0]);
        });

    // constraint energy
    if (!projectDBC)
        pol(range(numBouDofs), [vtemp = view<space>({}, vtemp), off = coOffset, kappa = boundaryKappa,
                                energy = view<space>(temp), n = numBouDofs] __device__(int i) mutable {
            auto vi = i + off;
            auto w = vtemp("ws", vi);
            auto cons = vtemp.pack(dim_c<3>, "cons", vi);
            reduce_to(i, n, 0.5f * kappa * w * cons.l2NormSqr(), energy[0]);
        });
    return temp.getVal() + elasticE;
}
} // namespace zeno