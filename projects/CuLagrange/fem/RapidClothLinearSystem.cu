#include "RapidCloth.cuh"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

// dynamics: 
// "yn" -> "x[k]"
// "yhat" -> "x_hat"
// "ytilde" -> "x_tilde"
namespace zeno {

void RapidClothSystem::project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    // only project boundary (character)
    pol(zs::range(numBouDofs), [vtemp = view<space>({}, vtemp), tagOffset = vtemp.getPropertyOffset(tag),
                                coOffset = coOffset] ZS_LAMBDA(int vi) mutable {
        vi += coOffset;
#pragma unroll
        for (int d = 0; d != 3; ++d)
            vtemp(tagOffset + d, vi) = 0;
    });
}

void RapidClothSystem::precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag,
                                   const zs::SmallString dstTag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    // precondition
    pol(zs::range(coOffset), [vtemp = view<space>({}, vtemp), srcOffset = vtemp.getPropertyOffset(srcTag),
                             dstOffset = vtemp.getPropertyOffset(dstTag)] ZS_LAMBDA(int vi) mutable {
        vtemp.tuple(dim_c<3>, dstOffset, vi) = vtemp.pack(dim_c<3, 3>, "P", vi) * vtemp.pack(dim_c<3>, srcOffset, vi);
    });
}

void RapidClothSystem::multiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag) {
    using namespace zs;
    constexpr execspace_e space = execspace_e::cuda;
    constexpr auto execTag = wrapv<space>{};
    // dx -> b
    auto wsOffset = vtemp.getPropertyOffset("ws");
    auto dxOffset = vtemp.getPropertyOffset(dxTag);
    auto bOffset = vtemp.getPropertyOffset(bTag);

    /// @brief initialize
    pol(range(coOffset), [execTag, vtemp = view<space>({}, vtemp), bOffset] ZS_LAMBDA(int vi) mutable {
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
    // repulsion 
    // TODO: use cooperative group to optimize in the future 
    if (enableRepulsion)
    {
        if (enableDegeneratedDist)
        {
            pol(range(npp), 
                [tempPP = proxy<space>({}, tempPP), 
                vtemp = proxy<space>({}, vtemp), 
                dxOffset, bOffset] __device__ (int i) mutable {
                    auto inds = tempPP.pack(dim_c<2>, "inds", i, int_c); 
                    auto hess = tempPP.pack(dim_c<6, 6>, "hess", i); 
                    for (int vi = 0; vi < 2; vi++)
                        for (int vj = 0; vj < 2; vj++)
                            for (int di = 0; di < 3; di++)
                                for (int dj = 0; dj < 3; dj++)
                                    atomic_add(exec_cuda, &vtemp(bOffset + di, inds[vi]), 
                                        hess(vi * 3 + di, vj * 3 + dj) * vtemp(dxOffset + dj, inds[vj])); 
                });     
            pol(range(npe), 
                [tempPE = proxy<space>({}, tempPE), 
                vtemp = proxy<space>({}, vtemp), 
                dxOffset, bOffset] __device__ (int i) mutable {
                    auto inds = tempPE.pack(dim_c<3>, "inds", i, int_c); 
                    auto hess = tempPE.pack(dim_c<9, 9>, "hess", i); 
                    for (int vi = 0; vi < 3; vi++)
                        for (int vj = 0; vj < 3; vj++)
                            for (int di = 0; di < 3; di++)
                                for (int dj = 0; dj < 3; dj++)
                                    atomic_add(exec_cuda, &vtemp(bOffset + di, inds[vi]), 
                                        hess(vi * 3 + di, vj * 3 + dj) * vtemp(dxOffset + dj, inds[vj])); 
                });         
        }
        pol(range(npt), 
            [tempPT = proxy<space>({}, tempPT), 
            vtemp = proxy<space>({}, vtemp), 
            dxOffset, bOffset] __device__ (int i) mutable {
                auto inds = tempPT.pack(dim_c<4>, "inds", i, int_c); 
                auto hess = tempPT.pack(dim_c<12, 12>, "hess", i); 
                for (int vi = 0; vi < 4; vi++)
                    for (int vj = 0; vj < 4; vj++)
                        for (int di = 0; di < 3; di++)
                            for (int dj = 0; dj < 3; dj++)
                                atomic_add(exec_cuda, &vtemp(bOffset + di, inds[vi]), 
                                    hess(vi * 3 + di, vj * 3 + dj) * vtemp(dxOffset + dj, inds[vj])); 
            });  
        pol(range(nee), 
            [tempEE = proxy<space>({}, tempEE), 
            vtemp = proxy<space>({}, vtemp), 
            dxOffset, bOffset] __device__ (int i) mutable {
                auto inds = tempEE.pack(dim_c<4>, "inds", i, int_c); 
                auto hess = tempEE.pack(dim_c<12, 12>, "hess", i); 
                for (int vi = 0; vi < 4; vi++)
                    for (int vj = 0; vj < 4; vj++)
                        for (int di = 0; di < 3; di++)
                            for (int dj = 0; dj < 3; dj++)
                                atomic_add(exec_cuda, &vtemp(bOffset + di, inds[vi]), 
                                    hess(vi * 3 + di, vj * 3 + dj) * vtemp(dxOffset + dj, inds[vj])); 
            });         
    }
}

void RapidClothSystem::cgsolve(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// copy diagonal block preconditioners
    pol.sync(false);
    /// solve for A dir = grad;
    // initial guess for hard boundary constraints
    pol(zs::range(coOffset),
        [vtemp = view<space>({}, vtemp), coOffset = coOffset, dt = dt, dirOffset = vtemp.getPropertyOffset("dir"),
         xtildeOffset = vtemp.getPropertyOffset("x_tilde"),
         xkOffset = vtemp.getPropertyOffset("x[k]")] ZS_LAMBDA(int i) mutable {
            vtemp.tuple(dim_c<3>, dirOffset, i) = vec3::zeros(); 
            if (i < coOffset) {
                vtemp.tuple(dim_c<3>, dirOffset, i) = vec3::zeros();
            } else {
                vtemp.tuple(dim_c<3>, dirOffset, i) =
                    vtemp.pack(dim_c<3>, xtildeOffset, i) - vtemp.pack(dim_c<3>, xkOffset, i);
            }
        });
    // temp = A * dir
    multiply(pol, "dir", "temp");
    // r = grad - temp
    pol(zs::range(coOffset), [vtemp = view<space>({}, vtemp), rOffset = vtemp.getPropertyOffset("r"),
                             gradOffset = vtemp.getPropertyOffset("grad"),
                             tempOffset = vtemp.getPropertyOffset("temp")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, rOffset, i) = vtemp.pack(dim_c<3>, gradOffset, i) - vtemp.pack(dim_c<3>, tempOffset, i);
    });
    project(pol, "r");
    precondition(pol, "r", "q");
    pol(zs::range(coOffset), [vtemp = view<space>({}, vtemp), pOffset = vtemp.getPropertyOffset("p"),
                             qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, pOffset, i) = vtemp.pack(dim_c<3>, qOffset, i);
    });
    T zTrk = dot(pol, "r", "q", coOffset);
    T residualPreconditionedNorm2 = zTrk;
    T localTol2 = cgRel * cgRel * residualPreconditionedNorm2;
    int iter = 0;

    CppTimer timer;
    if (enableProfile_c)
        timer.tick();
    for (; iter != CGCap; ++iter) {
        if (!silentMode_c && iter % 50 == 0)
            fmt::print("cg iter: {}, norm2: {} (zTrk: {})\n", iter, residualPreconditionedNorm2, zTrk);

        if (residualPreconditionedNorm2 <= localTol2)
            break;
        multiply(pol, "p", "temp");
        project(pol, "temp"); // need further checking hessian!

        T alpha = zTrk / dot(pol, "temp", "p", coOffset);
        pol(range(coOffset), [vtemp = view<space>({}, vtemp), dirOffset = vtemp.getPropertyOffset("dir"),
                             pOffset = vtemp.getPropertyOffset("p"), rOffset = vtemp.getPropertyOffset("r"),
                             tempOffset = vtemp.getPropertyOffset("temp"), alpha] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, dirOffset, vi) =
                vtemp.pack(dim_c<3>, dirOffset, vi) + alpha * vtemp.pack(dim_c<3>, pOffset, vi);
            vtemp.tuple(dim_c<3>, rOffset, vi) =
                vtemp.pack(dim_c<3>, rOffset, vi) - alpha * vtemp.pack(dim_c<3>, tempOffset, vi);
        });

        precondition(pol, "r", "q");
        T zTrkLast = zTrk;
        zTrk = dot(pol, "q", "r", coOffset);
        T beta = zTrk / zTrkLast;
        pol(range(coOffset), [vtemp = view<space>({}, vtemp), beta, pOffset = vtemp.getPropertyOffset("p"),
                             qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, pOffset, vi) =
                vtemp.pack(dim_c<3>, qOffset, vi) + beta * vtemp.pack(dim_c<3>, pOffset, vi);
        });

        residualPreconditionedNorm2 = zTrk;
    } // end cg step
    pol.sync(true);
    if (enableProfile_c)
        timer.tock(fmt::format("{} cgiters", iter));
}

void RapidClothSystem::newtonDynamicsStep(zs::CudaExecutionPolicy &pol) {
    // GRAD, HESS, P
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3, 3>, "P", i) = mat3::zeros();
        vtemp.tuple(dim_c<3>, "grad", i) = vec3::zeros();
    });
    // TODO: line-search 
    // TODO: calculate energy 
    computeInertialAndForceGradient(pol, "x[k]");
    computeElasticGradientAndHessian(pol, "x[k]");
    if (enableRepulsion)
    {
        findConstraints(pol, repulsionRange, "x[k]"); 
        D = 0;    
        computeRepulsionGradientAndHessian(pol, "x[k]"); 
    }
    // APPLY BOUNDARY CONSTRAINTS, PROJ GRADIENT
    // TODO: revise codes for BC 
    project(pol, "grad");
    // PREPARE P
    pol(zs::range(numDofs), [vtemp = view<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
        vtemp.tuple<9>("P", i) = inverse(vtemp.template pack<3, 3>("P", i));
    });
    // CG SOLVE
    // TODO: use sparse matrix 
    cgsolve(pol);
    // fix the boundary 
    pol(range(coOffset), 
        [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
            vtemp.tuple(dim_c<3>, "y[k+1]", vi) = 
                vtemp.pack(dim_c<3>, "x[k]", vi) + vtemp.pack(dim_c<3>, "dir", vi);  
        }); 
    if (enableProfile_c)
        timer.tock("Newton step"); 
}

void RapidClothSystem::gdDynamicsStep(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // TODO
}

template <typename Model>
typename RapidClothSystem::T
elasticityEnergy(zs::CudaExecutionPolicy &pol, typename RapidClothSystem::tiles_t &vtemp, 
                 const zs::SmallString& tag, typename RapidClothSystem::tiles_t &seInds, 
                 typename RapidClothSystem::PrimitiveHandle &primHandle, const Model &model, 
                 typename RapidClothSystem::T dt, zs::Vector<typename RapidClothSystem::T> &energy) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using mat3 = typename RapidClothSystem::mat3;
    using vec3 = typename RapidClothSystem::vec3;
    using T = typename RapidClothSystem::T;

    auto &eles = primHandle.getEles();
    auto &edges = primHandle.getSurfEdges();
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
                auto Eshear = (model.mu * 0.3f) * vole * zs::sqr(f0.dot(f1));
                E = Estretch + Eshear;
                reduce_to(ei, n, E, energy[0]);
            });
        return energy.getVal() * dt * dt;
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

typename RapidClothSystem::T RapidClothSystem::dynamicsEnergy(zs::CudaExecutionPolicy &pol, 
    const zs::SmallString &tag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // elasticity energy
    T elasticE = 0;
    for (auto &primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        match([&](auto &elasticModel) {
            elasticE += elasticityEnergy(pol, vtemp, tag, seInds, primHandle, elasticModel, dt, temp);
        })(primHandle.getModels().getElasticModel());
    }

    temp.setVal(0);
    // inertial energy 
    // TODO: remove every "ytilde"
    pol(range(coOffset),
        [vtemp = view<space>({}, vtemp), energy = view<space>(temp), n = coOffset, sigma = sigma * dt * dt, dt = dt,
         hasExtf = vtemp.hasProperty("extf"), gravAccel = gravAccel, BCStiffness = BCStiffness, tag] __device__(int vi) mutable {
            auto m = vtemp("ws", vi);
            auto yk = vtemp.pack(dim_c<3>, tag, vi);
            auto x_tilde = vtemp.pack(dim_c<3>, "x_tilde", vi);
            auto E =
                0.5f * m * (yk - x_tilde).l2NormSqr() - m * yk.dot(gravAccel) * dt * dt;
            if (hasExtf)
                E -= yk.dot(vtemp.pack(dim_c<3>, "extf", vi)) * dt * dt;
            bool isBC  = vtemp("isBC", vi) > 0.5f; 
            if (isBC)
            {
                auto BCtarget = vtemp.pack(dim_c<3>, "BCtarget", vi); 
                E += m * BCStiffness * (yk - BCtarget).l2NormSqr(); 
            }
            reduce_to(vi, n, E, energy[0]);
        });

    return temp.getVal() + elasticE;
}
} // namespace zeno