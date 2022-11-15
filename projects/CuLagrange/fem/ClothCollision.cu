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
#if 0
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
#endif
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

void compute_surface_neighbors(typename ZenoParticles::particles_t &sfs, typename ZenoParticles::particles_t &ses,
                               typename ZenoParticles::particles_t &svs) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using vec2i = zs::vec<int, 2>;
    using vec3i = zs::vec<int, 3>;
    auto pol = cuda_exec();
    sfs.append_channels(pol, {{"ff_inds", 3}, {"fe_inds", 3}, {"fp_inds", 3}});
    ses.append_channels(pol, {{"fe_inds", 2}});

    bcht<vec2i, int, true, universal_hash<vec2i>, 32> etab{sfs.get_allocator(), sfs.size() * 3};
    Vector<int> sfi{etab.get_allocator(), sfs.size() * 3}; // surftri indices corresponding to edges in the table
    /// @brief compute ff neighbors
    {
        pol(range(sfs.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs),
                                sfi = proxy<space>(sfi)] __device__(int ti) mutable {
            auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
            for (int i = 0; i != 3; ++i)
                if (auto no = etab.insert(vec2i{tri[i], tri[(i + 1) % 3]}); no >= 0) {
                    sfi[no] = ti;
                } else
                    printf("the same directed edge has been inserted twice!\n");
        });
        pol(range(sfs.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs),
                                sfi = proxy<space>(sfi)] __device__(int ti) mutable {
            auto neighborIds = vec3i::uniform(-1);
            auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
            for (int i = 0; i != 3; ++i)
                if (auto no = etab.query(vec2i{tri[(i + 1) % 3], tri[i]}); no >= 0) {
                    neighborIds[i] = sfi[no];
                }
            sfs.tuple(dim_c<3>, "ff_inds", ti) = neighborIds.reinterpret_bits(float_c);
        });
    }
    /// @brief compute fe neighbors
    {
        pol(range(ses.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs), ses = proxy<space>({}, ses),
                                sfi = proxy<space>(sfi)] __device__(int li) mutable {
            auto findLineIdInTri = [](const auto &tri, int v0, int v1) -> int {
#pragma unroll
                for (int loc = 0; loc < 3; ++loc)
                    if (tri[loc] == v0 && tri[(loc + 1) % 3] == v1)
                        return loc;
                return -1;
            };
            auto neighborTris = vec2i::uniform(-1);
            auto line = ses.pack(dim_c<2>, "inds", li).reinterpret_bits(int_c);

            auto extract = [&](int evI) {
                if (auto no = etab.query(line); no >= 0) {
                    // tri
                    auto triNo = sfi[no];
                    auto tri = sfs.pack(dim_c<3>, "inds", triNo).reinterpret_bits(int_c);
                    auto loc = findLineIdInTri(tri, line[0], line[1]);
                    if (loc == -1)
                        printf("ridiculous, this edge <%d, %d> does not belong to tri <%d, %d, %d>\n", line[0], line[1],
                               tri[0], tri[1], tri[2]);
                    sfs("fe_inds", loc, triNo) = li;
                    // edge
                    neighborTris[evI] = triNo;
                }
            };
            extract(0);
            auto tmp = line[0];
            line[0] = line[1];
            line[1] = tmp;
            extract(1);
            ses.tuple(dim_c<2>, "fe_inds", li) = neighborTris;
        });
    }
    /// @brief compute fp neighbors
    /// @note  surface vertex index is not necessarily consecutive, thus hashing
    {
        bcht<int, int, true, universal_hash<int>, 32> vtab{svs.get_allocator(), svs.size()};
        Vector<int> svi{etab.get_allocator(), svs.size()}; // surftri indices corresponding to edges in the table
        // svs
        pol(range(svs.size()), [vtab = proxy<space>(vtab), svs = proxy<space>({}, svs),
                                svi = proxy<space>(svi)] __device__(int vi) mutable {
            int vert = reinterpret_bits<int>(svs("inds", vi));
            if (auto no = vtab.insert(vert); no >= 0)
                svi[no] = vi;
            else
                printf("the same directed edge has been inserted twice!\n");
        });
        //
        pol(range(sfs.size()), [vtab = proxy<space>(vtab), sfs = proxy<space>({}, sfs),
                                svi = proxy<space>(svi)] __device__(int ti) mutable {
            auto neighborIds = vec3i::uniform(-1);
            auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
            for (int i = 0; i != 3; ++i)
                if (auto no = vtab.query(tri[i]); no >= 0) {
                    neighborIds[i] = svi[no];
                }
            sfs.tuple(dim_c<3>, "fp_inds", ti) = neighborIds.reinterpret_bits(float_c);
        });
    }
}

} // namespace zeno