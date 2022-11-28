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
        pol(zs::range(numBouDofs), [vtemp = proxy<space>({}, vtemp), tagOffset = vtemp.getPropertyOffset(tag),
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
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), srcOffset = vtemp.getPropertyOffset(srcTag),
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
    pol(range(numDofs), [execTag, vtemp = proxy<space>({}, vtemp), bOffset] ZS_LAMBDA(int vi) mutable {
        vtemp.tuple(dim_c<3>, bOffset, vi) = vec3::zeros();
    });
    /// @brief inertial and coupling
    pol(zs::range(coOffset),
        [execTag, vtemp = proxy<space>(vtemp), sigma = sigma, wsOffset, dxOffset, bOffset] __device__(int i) mutable {
            auto m = vtemp(wsOffset, i);
            auto dx = vtemp.pack(dim_c<3>, dxOffset, i) * (m + 1) * sigma;
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
                [execTag, etemp = proxy<space>(primHandle.etemp), vtemp = proxy<space>(vtemp),
                 eles = proxy<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
                 hOffset = primHandle.etemp.getPropertyOffset("He"), dxOffset, bOffset,
                 vOffset = primHandle.vOffset] ZS_LAMBDA(int ei, int tid) mutable {
                    int rowid = tid / 5;
                    int colid = tid % 5;
                    auto inds = eles.pack(dim_c<2>, indsOffset, ei).reinterpret_bits(int_c) + vOffset;
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
                [execTag, etemp = proxy<space>(primHandle.etemp), vtemp = proxy<space>(vtemp),
                 eles = proxy<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
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

                    auto inds = eles.pack(dim_c<3>, indsOffset, ei).reinterpret_bits(int_c) + vOffset;
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
                [execTag, etemp = proxy<space>(primHandle.etemp), vtemp = proxy<space>(vtemp),
                 eles = proxy<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
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

                    auto inds = eles.pack(dim_c<4>, indsOffset, Hid).template reinterpret_bits<int>() + vOffset;
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
                [execTag, etemp = proxy<space>(primHandle.etemp), vtemp = proxy<space>(vtemp),
                 eles = proxy<space>(eles), indsOffset = eles.getPropertyOffset("inds"),
                 hOffset = primHandle.etemp.getPropertyOffset("He"), dxOffset, bOffset,
                 vOffset = primHandle.vOffset] ZS_LAMBDA(int ei, int tid) mutable {
                    int rowid = tid / 5;
                    int colid = tid % 5;
                    auto inds = eles.pack(dim_c<2>, indsOffset, ei).reinterpret_bits(int_c) + vOffset;
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
        pol(range(numBouDofs), [execTag, vtemp = proxy<space>(vtemp), dxOffset, bOffset, wsOffset, coOffset = coOffset,
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
        [vtemp = proxy<space>({}, vtemp), coOffset = coOffset, dt = dt, dirOffset = vtemp.getPropertyOffset("dir"),
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
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), rOffset = vtemp.getPropertyOffset("r"),
                             gradOffset = vtemp.getPropertyOffset("grad"),
                             tempOffset = vtemp.getPropertyOffset("temp")] ZS_LAMBDA(int i) mutable {
        vtemp.tuple(dim_c<3>, rOffset, i) = vtemp.pack(dim_c<3>, gradOffset, i) - vtemp.pack(dim_c<3>, tempOffset, i);
    });
    project(pol, "r");
    precondition(pol, "r", "q");
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), pOffset = vtemp.getPropertyOffset("p"),
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
        pol(range(numDofs), [vtemp = proxy<space>({}, vtemp), dirOffset = vtemp.getPropertyOffset("dir"),
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
        pol(range(numDofs), [vtemp = proxy<space>({}, vtemp), beta, pOffset = vtemp.getPropertyOffset("p"),
                             qOffset = vtemp.getPropertyOffset("q")] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple(dim_c<3>, pOffset, vi) =
                vtemp.pack(dim_c<3>, qOffset, vi) + beta * vtemp.pack(dim_c<3>, pOffset, vi);
        });

        residualPreconditionedNorm2 = zTrk;
    } // end cg step
    pol.sync(true);
    timer.tock(fmt::format("{} cgiters", iter));
}

} // namespace zeno