#include "Cloth.cuh"
#include "collision_energy/vertex_face_sqrt_collision.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

void ClothSystem::findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat) {
    // nPP.setVal(0);
    // nPE.setVal(0);
    // nPT.setVal(0);
    // nEE.setVal(0);

    ncsPT.setVal(0);
    ncsEE.setVal(0);

    zs::CppTimer timer;
    timer.tick();
    if (enableContactSelf) {
        auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0);
        stBvh.refit(pol, triBvs);
        auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, 0);
        seBvh.refit(pol, edgeBvs);
        findCollisionConstraintsImpl(pol, dHat, false);
    }

    if (coVerts)
        if (coVerts->size()) {
            auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", *coEles, zs::wrapv<3>{}, coOffset);
            bouStBvh.refit(pol, triBvs);
            auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", *coEdges, zs::wrapv<2>{}, coOffset);
            bouSeBvh.refit(pol, edgeBvs);
            findCollisionConstraintsImpl(pol, dHat, true);
        }
    auto [npt, nee] = getCollisionCnts();
    timer.tock(fmt::format("dcd broad phase [pt, ee]({}, {})", npt, nee));

    frontManageRequired = false;
}

#define PROFILE_CD 0

void ClothSystem::findCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol.profile(PROFILE_CD);
    /// pt
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    auto &stfront = withBoundary ? boundaryStFront : selfStFront;
    pol(Collapse{stfront.size()},
        [svInds = proxy<space>({}, svInds), eles = proxy<space>({}, withBoundary ? *coEles : stInds),
         exclTris = withBoundary ? proxy<space>(exclBouSts) : proxy<space>(exclSts), vtemp = proxy<space>({}, vtemp),
         bvh = proxy<space>(stbvh), front = proxy<space>(stfront),
         // PP = proxy<space>(PP), nPP = proxy<space>(nPP), PE = proxy<space>(PE),
         // nPE = proxy<space>(nPE), PT = proxy<space>(PT), nPT = proxy<space>(nPT),
         csPT = proxy<space>(csPT), ncsPT = proxy<space>(ncsPT), dHat2 = dHat * dHat, thickness = dHat,
         voffset = withBoundary ? coOffset : 0, frontManageRequired = frontManageRequired] __device__(int i) mutable {
            auto vi = front.prim(i);
            vi = reinterpret_bits<int>(svInds("inds", vi));
            auto p = vtemp.pack(dim_c<3>, "xn", vi);
            auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
            auto f = [&](int stI) {
                if (exclTris[stI])
                    return;
                auto tri = eles.pack(dim_c<3>, "inds", stI).reinterpret_bits(int_c) + voffset;
                if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                    return;
                // ccd
                auto t0 = vtemp.pack(dim_c<3>, "xn", tri[0]);
                auto t1 = vtemp.pack(dim_c<3>, "xn", tri[1]);
                auto t2 = vtemp.pack(dim_c<3>, "xn", tri[2]);

                switch (pt_distance_type(p, t0, t1, t2)) {
                case 0: {
                    if (auto d2 = dist2_pp(p, t0); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //PP[no] = pair_t{vi, tri[0]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 1: {
                    if (auto d2 = dist2_pp(p, t1); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //PP[no] = pair_t{vi, tri[1]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 2: {
                    if (auto d2 = dist2_pp(p, t2); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //PP[no] = pair_t{vi, tri[2]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 3: {
                    if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        //PE[no] = pair3_t{vi, tri[0], tri[1]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 4: {
                    if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        //PE[no] = pair3_t{vi, tri[1], tri[2]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 5: {
                    if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        //PE[no] = pair3_t{vi, tri[2], tri[0]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                case 6: {
                    if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                        //auto no = atomic_add(exec_cuda, &nPT[0], 1);
                        //PT[no] = pair4_t{vi, tri[0], tri[1], tri[2]};
                        csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair4_t{vi, tri[0], tri[1], tri[2]};
                    }
                    break;
                }
                default: break;
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        stfront.reorder(pol);
    /// ee
    if (enableContactEE) {
        const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
        auto &sefront = withBoundary ? boundarySeFront : selfSeFront;
        pol(Collapse{sefront.size()}, [seInds = proxy<space>({}, seInds),
                                       sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
                                       exclSes = proxy<space>(exclSes), vtemp = proxy<space>({}, vtemp),
                                       bvh = proxy<space>(sebvh), front = proxy<space>(sefront),
                                       // PP = proxy<space>(PP), nPP = proxy<space>(nPP), PE = proxy<space>(PE),
                                       // nPE = proxy<space>(nPE), EE = proxy<space>(EE), nEE = proxy<space>(nEE),
                                       //
                                       csEE = proxy<space>(csEE), ncsEE = proxy<space>(ncsEE), dHat2 = dHat * dHat,
                                       thickness = dHat, voffset = withBoundary ? coOffset : 0,
                                       frontManageRequired = frontManageRequired] __device__(int i) mutable {
            auto sei = front.prim(i);
            if (exclSes[sei])
                return;
            auto eiInds = seInds.pack(dim_c<2>, "inds", sei).reinterpret_bits(int_c);
            auto v0 = vtemp.pack(dim_c<3>, "xn", eiInds[0]);
            auto v1 = vtemp.pack(dim_c<3>, "xn", eiInds[1]);
            auto [mi, ma] = get_bounding_box(v0, v1);
            auto bv = bv_t{mi - thickness, ma + thickness};
            auto f = [&](int sej) {
                if (voffset == 0 && sei < sej) // only check this for self intersection
                    return;
                auto ejInds = sedges.pack(dim_c<2>, "inds", sej).reinterpret_bits(int_c) + voffset;
                if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                    eiInds[1] == ejInds[1])
                    return;
                // ccd
                auto v2 = vtemp.pack(dim_c<3>, "xn", ejInds[0]);
                auto v3 = vtemp.pack(dim_c<3>, "xn", ejInds[1]);

                switch (ee_distance_type(v0, v1, v2, v3)) {
                case 0: {
                    if (auto d2 = dist2_pp(v0, v2); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //   PP[no] = pair_t{eiInds[0], ejInds[0]};
                    }
                    break;
                }
                case 1: {
                    if (auto d2 = dist2_pp(v0, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //   PP[no] = pair_t{eiInds[0], ejInds[1]};
                    }
                    break;
                }
                case 2: {
                    if (auto d2 = dist2_pe(v0, v2, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        //   PE[no] = pair3_t{eiInds[0], ejInds[0], ejInds[1]};
                    }
                    break;
                }
                case 3: {
                    if (auto d2 = dist2_pp(v1, v2); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //   PP[no] = pair_t{eiInds[1], ejInds[0]};
                    }
                    break;
                }
                case 4: {
                    if (auto d2 = dist2_pp(v1, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPP[0], 1);
                        //   PP[no] = pair_t{eiInds[1], ejInds[1]};
                    }
                    break;
                }
                case 5: {
                    if (auto d2 = dist2_pe(v1, v2, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        // PE[no] = pair3_t{eiInds[1], ejInds[0], ejInds[1]};
                    }
                    break;
                }
                case 6: {
                    if (auto d2 = dist2_pe(v2, v0, v1); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        // PE[no] = pair3_t{ejInds[0], eiInds[0], eiInds[1]};
                    }
                    break;
                }
                case 7: {
                    if (auto d2 = dist2_pe(v3, v0, v1); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nPE[0], 1);
                        // PE[no] = pair3_t{ejInds[1], eiInds[0], eiInds[1]};
                    }
                    break;
                }
                case 8: {
                    if (auto d2 = dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                        csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        // auto no = atomic_add(exec_cuda, &nEE[0], 1);
                        // EE[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                    }
                    break;
                }
                default: break;
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
        if (frontManageRequired)
            sefront.reorder(pol);
    }
    pol.profile(false);
}

void ClothSystem::computeCollisionGradientAndHessian(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto numPT = ncsPT.getVal();
    // group of size-4 tiles?
    pol(range(numPT), [vtemp = proxy<space>({}, vtemp), tempPT = proxy<space>({}, tempPT), csPT = proxy<space>(csPT),
                       gradOffset = vtemp.getPropertyOffset("grad"), thickness = dHat, maxMu = maxMu,
                       maxLam = maxLam] __device__(int i) mutable {
        auto pt = csPT[i];
        zs::vec<T, 3> vs[4] = {vtemp.pack(dim_c<3>, "xn", pt[0]), vtemp.pack(dim_c<3>, "xn", pt[1]),
                               vtemp.pack(dim_c<3>, "xn", pt[2]), vtemp.pack(dim_c<3>, "xn", pt[3])};
        auto grad = VERTEX_FACE_SQRT_COLLISION::gradient(vs, maxMu, maxLam, thickness);
        auto hess = VERTEX_FACE_SQRT_COLLISION::hessian(vs, maxMu, maxLam, thickness);
        // gradient
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gradOffset + d, pt[0]), -grad(0 + d));
            atomic_add(exec_cuda, &vtemp(gradOffset + d, pt[1]), -grad(3 + d));
            atomic_add(exec_cuda, &vtemp(gradOffset + d, pt[2]), -grad(6 + d));
            atomic_add(exec_cuda, &vtemp(gradOffset + d, pt[3]), -grad(9 + d));
        }
        // hessian
        tempPT.tuple(dim_c<12, 12>, "H", i) = hess;
    });
    if (enableContactEE) {
        ;
    }
}

} // namespace zeno