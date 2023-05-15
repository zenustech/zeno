#include "RapidCloth.cuh"
#include "RapidClothUtils.hpp"
#include "Structures.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/graph/Coloring.hpp"
#include "zensim/math/matrix/SparseMatrixOperations.hpp"
#include "RapidClothGradHess.inl"
#include <fstream> 
#include <Eigen/SparseCore>

namespace zeno {
void RapidClothSystem::findConstraintsImpl(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::T radius, bool withBoundary, const zs::SmallString &tag)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    // p -> t
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    auto &stfront = withBoundary ? boundaryStFront : selfStFront;
    pol(Collapse{stfront.size()},
        [spInds = view<space>({}, spInds, false_c, "spInds"), svOffset = svOffset, coOffset = coOffset, 
         eles = view<space>({}, withBoundary ? *coEles : stInds, false_c, "eles"),
         vtemp = view<space>({}, vtemp, false_c, "vtemp"), bvh = view<space>(stbvh, false_c), 
         front = proxy<space>(stfront), tempPT = view<space>({}, tempPT, false_c, "tempPT"),
         tempPP = proxy<space>({}, tempPP), tempPE = proxy<space>({}, tempPE), 
         nPT = view<space>(nPT, false_c, "nPT"), radius, voffset = withBoundary ? coOffset : 0,
         nPP = view<space>(nPP, false_c, "nPP"), nPE = view<space>(nPE, false_c, "nPE"), 
         exclTab = proxy<space>(exclTab), 
         enableExclEdges = enableExclEdges, 
         enableDegeneratedDist = enableDegeneratedDist, 
         frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
            auto vi = front.prim(i);
            vi = spInds("inds", vi, int_c); 
            const auto dHat2 = zs::sqr(radius);
            auto p = vtemp.pack(dim_c<3>, tag, vi);
            auto bv = bv_t{get_bounding_box(p - radius, p + radius)};
            if (enableDegeneratedDist)
            {
                auto f = [&](int stI) mutable {
                    auto tri = eles.pack(dim_c<3>, "inds", stI, int_c) + voffset;
                    if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                        return;
                    bool onlyPT = false; 
                    if (enableExclEdges)
                        for (int k = 0; k < 3; k++)
                            if (exclTab.query({vi, tri[k]}) >= 0)
                            {
                                onlyPT = true; 
                                break; 
                            }
                    // ccd
                    auto t0 = vtemp.pack(dim_c<3>, tag, tri[0]);
                    auto t1 = vtemp.pack(dim_c<3>, tag, tri[1]);
                    auto t2 = vtemp.pack(dim_c<3>, tag, tri[2]);

                    switch (pt_distance_type(p, t0, t1, t2)) {
                        case 0: 
                        {
                            if (onlyPT)
                                break; 
                            if (auto d2 = dist2_pp(p, t0); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{vi, tri[0]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            }

                            break; 
                        }
                        case 1: 
                        {
                            if (onlyPT)
                                break; 
                            if (auto d2 = dist2_pp(p, t1); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{vi, tri[1]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 2: 
                        {
                            if (onlyPT)
                                break; 
                            if (auto d2 = dist2_pp(p, t2); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{vi, tri[2]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            } 
                            break; 
                        }
                        case 3: 
                        {
                            if (onlyPT)
                                break; 
                            if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{vi, tri[0], tri[1]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 4: 
                        {
                            if (onlyPT)
                                break; 
                            if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{vi, tri[1], tri[2]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 5: 
                        {                        
                            if (onlyPT)
                                break; 
                            if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{vi, tri[2], tri[0]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 6: 
                        {
                            if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPT[0], 1); 
                                auto inds = pair4_t{vi, tri[0], tri[1], tri[2]}; 
                                tempPT.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                                tempPT("dist", no) = (float)zs::sqrt(d2); 
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
            } else {
                    auto f = [&](int stI) mutable {
                        auto tri = eles.pack(dim_c<3>, "inds", stI, int_c) + voffset;
                        if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                            return;
                        // ccd
                        auto t0 = vtemp.pack(dim_c<3>, tag, tri[0]);
                        auto t1 = vtemp.pack(dim_c<3>, tag, tri[1]);
                        auto t2 = vtemp.pack(dim_c<3>, tag, tri[2]);

                        if (auto d2 = dist2_pt_unclassified(p, t0, t1, t2); d2 < dHat2)
                        {
                            auto no = atomic_add(exec_cuda, &nPT[0], 1); 
                            auto inds = pair4_t{vi, tri[0], tri[1], tri[2]}; 
                            tempPT.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                            tempPT("dist", no) = (float)zs::sqrt(d2); 
                        }   
                    }; 

                    if (frontManageRequired)
                        bvh.iter_neighbors(bv, i, front, f);
                    else
                        bvh.iter_neighbors(bv, front.node(i), f);
            }

        });
    if (frontManageRequired)
        stfront.reorder(pol); 

#define check_potential_pe(p, e0, e1, pi, e0i, e1i)                                                                                                     \
{                                                                                                                                                       \
        switch (pe_distance_type(p, e0, e1)) {                                                                                                          \
            case 0: {                                                                                                                                   \
                if (auto d2 = dist2_pp(p, e0); d2 < dHat2) {                                                                                            \
                    auto no = atomic_add(exec_cuda, &nPP[0], 1);                                                                                        \
                    tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{pi, e0i};                                                                        \
                }                                                                                                                                       \
                break;                                                                                                                                  \
            }                                                                                                                                           \
            case 1: {                                                                                                                                   \
                if (auto d2 = dist2_pp(p, e1); d2 < dHat2) {                                                                                            \
                    auto no = atomic_add(exec_cuda, &nPP[0], 1);                                                                                        \
                    tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{pi, e1i};                                                                        \
                }                                                                                                                                       \
                break;                                                                                                                                  \
            }                                                                                                                                           \
            case 2: {                                                                                                                                   \
                if (auto d2 = dist2_pe(p, e0, e1); d2 < dHat2) {                                                                                        \
                    auto no = atomic_add(exec_cuda, &nPE[0], 1);                                                                                        \
                    tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{pi, e0i, e1i};                                                                  \
                }                                                                                                                                       \
                break;                                                                                                                                  \
            }                                                                                                                                           \
            default: break;                                                                                                                             \
        }                                                                                                                                               \
}
    // e -> e
    const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
    auto &seefront = withBoundary ? boundarySeeFront : selfSeeFront;
    pol(Collapse{seefront.size()},
        [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
            vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(seefront),
            tempEE = proxy<space>({}, tempEE), nEE = proxy<space>(nEE), dHat2 = zs::sqr(radius),
            tempPP = proxy<space>({}, tempPP), nPP = proxy<space>(nPP), 
            tempPE = proxy<space>({}, tempPE), nPE = proxy<space>(nPE), 
            radius, voffset = withBoundary ? coOffset : 0,
            exclTab = proxy<space>(exclTab), 
            enableDegeneratedDist = enableDegeneratedDist, 
            frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
            auto sei = front.prim(i);
            auto eiInds = seInds.pack(dim_c<2>, "inds", sei, int_c);
            auto v0 = vtemp.pack(dim_c<3>, tag, eiInds[0]);
            auto v1 = vtemp.pack(dim_c<3>, tag, eiInds[1]);
            auto [mi, ma] = get_bounding_box(v0, v1);
            auto bv = bv_t{mi - radius, ma + radius};
            auto f = [&](int sej) {
                if (voffset == 0 && sei <= sej)
                    return;
                auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c) + voffset;
                if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] || eiInds[1] == ejInds[0] ||
                    eiInds[1] == ejInds[1])
                    return;
                auto v2 = vtemp.pack(dim_c<3>, tag, ejInds[0]);
                auto v3 = vtemp.pack(dim_c<3>, tag, ejInds[1]);

                if (enableDegeneratedDist)
                {
                    switch(ee_distance_type(v0, v1, v2, v3)) {
                        case 0: {
                            if (auto d2 = dist2_pp(v0, v2); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{eiInds[0], ejInds[0]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 1: {
                            if (auto d2 = dist2_pp(v0, v3); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{eiInds[0], ejInds[1]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 2: {
                            if (auto d2 = dist2_pe(v0, v2, v3); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{eiInds[0], ejInds[0], ejInds[1]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 3: {
                            if (auto d2 = dist2_pp(v1, v2); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{eiInds[1], ejInds[0]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 4: {
                            if (auto d2 = dist2_pp(v1, v3); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                                tempPP.tuple(dim_c<2>, "inds", no, int_c) = pair_t{eiInds[1], ejInds[1]}; 
                                tempPP("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 5: {
                            if (auto d2 = dist2_pe(v1, v2, v3); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{eiInds[1], ejInds[0], ejInds[1]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 6: {
                            if (auto d2 = dist2_pe(v2, v0, v1); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{ejInds[0], eiInds[0], eiInds[1]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 7: {
                            if (auto d2 = dist2_pe(v3, v0, v1); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                                tempPE.tuple(dim_c<3>, "inds", no, int_c) = pair3_t{ejInds[1], eiInds[0], eiInds[1]}; 
                                tempPE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        case 8: {
                            if ((v1 - v0).cross(v3 - v2).l2NormSqr() / ((v1 - v0).l2NormSqr() * (v3 - v2).l2NormSqr() + eps_c) < eps_c)
                            {                                                                                                                                                \
                                check_potential_pe(v0, v2, v3, eiInds[0], ejInds[0], ejInds[1]); 
                                check_potential_pe(v1, v2, v3, eiInds[1], ejInds[0], ejInds[1]); 
                                check_potential_pe(v2, v0, v1, ejInds[0], eiInds[0], eiInds[1]); 
                                check_potential_pe(v3, v0, v1, ejInds[1], eiInds[0], eiInds[1]); 
                                break; 
                            }
                            if (auto d2 = safe_dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                                auto no = atomic_add(exec_cuda, &nEE[0], 1); 
                                auto inds = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                                tempEE.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                                tempEE("dist", no) = (float)zs::sqrt(d2); 
                            }
                            break; 
                        }
                        default: break; 
                    }
                } else {
                    if (auto d2 = safe_dist2_ee_unclassified(v0, v1, v2, v3); d2 < dHat2)
                    {
                        auto no = atomic_add(exec_cuda, &nEE[0], 1); 
                        auto inds = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                        tempEE.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                        tempEE("dist", no) = (float)zs::sqrt(d2); 
                    }
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);       
        });
    if (frontManageRequired)
        seefront.reorder(pol);

    // e -> p 
    if (enablePE_c)
    {
        auto &sevfront = withBoundary ? boundarySevFront : selfSevFront;
        pol(Collapse{sevfront.size()},
            [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
                sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
                vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(sevfront),
                tempPE = proxy<space>({}, tempPE), nPE = proxy<space>(nPE), dHat2 = zs::sqr(radius),
                radius, voffset = withBoundary ? coOffset : 0,
                frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
                auto vi = front.prim(i);
                vi = spInds("inds", vi, int_c); 
                const auto dHat2 = zs::sqr(radius);
                auto p = vtemp.pack(dim_c<3>, tag, vi);
                auto bv = bv_t{get_bounding_box(p - radius, p + radius)};
                auto f = [&](int sej) {
                    auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c) + voffset;
                    if (vi == ejInds[0] || vi == ejInds[1])
                        return; 
                    auto v2 = vtemp.pack(dim_c<3>, tag, ejInds[0]);
                    auto v3 = vtemp.pack(dim_c<3>, tag, ejInds[1]);

                    if (pe_distance_type(p, v2, v3) != 2)
                        return; 
                    if (auto d2 = dist2_pe(p, v2, v3); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                        auto inds = pair3_t{vi, ejInds[0], ejInds[1]};
                        tempPE.tuple(dim_c<3>, "inds", no, int_c) = inds; 
                        tempPE("dist", no) = (float)zs::sqrt(d2); 
                    }
                };
                if (frontManageRequired)
                    bvh.iter_neighbors(bv, i, front, f);
                else
                    bvh.iter_neighbors(bv, front.node(i), f);
            });
        if (frontManageRequired)
            sevfront.reorder(pol);
    }
    // v-> v
    if (enablePP_c)
    {
        if (!withBoundary)
        {
            const auto &svbvh = svBvh;
            auto &svfront = selfSvFront;
            pol(Collapse{svfront.size()},
                [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
                bvh = proxy<space>(svbvh), front = proxy<space>(svfront), tempPP = proxy<space>({}, tempPP),
                eles = proxy<space>({}, svInds), 
                vtemp = proxy<space>({}, vtemp), 
                nPP = proxy<space>(nPP), radius, voffset = withBoundary ? coOffset : 0,
                frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
                    auto svI = front.prim(i);
                    auto vi = spInds("inds", svI, int_c); 
                    const auto dHat2 = zs::sqr(radius);
                    auto pi = vtemp.pack(dim_c<3>, tag, vi);
                    auto bv = bv_t{get_bounding_box(pi - radius, pi + radius)};
                    auto f = [&](int svJ) {
                        if (voffset == 0 && svI <= svJ)
                            return; 
                        auto vj = eles("inds", svJ, int_c) + voffset; 
                        auto pj = vtemp.pack(dim_c<3>, tag, vj); 
                        if (auto d2 = dist2_pp(pi, pj); d2 < dHat2) {
                            auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                            auto inds = pair_t{vi, vj};
                            tempPP.tuple(dim_c<2>, "inds", no, int_c) = inds; 
                            tempPP("dist", no) = (float)zs::sqrt(d2); 
                        }
                    }; 

                    if (frontManageRequired)
                        bvh.iter_neighbors(bv, i, front, f);
                    else
                        bvh.iter_neighbors(bv, front.node(i), f);
                });     
            if (frontManageRequired)
                svfront.reorder(pol);   
        }        
    }

}

void RapidClothSystem::findConstraints(zs::CudaExecutionPolicy &pol, T dist, const zs::SmallString &tag)
{
    // TODO: compute oE in initialize
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 

    nPP.setVal(0);
    nPE.setVal(0);  
    nPT.setVal(0); 
    nEE.setVal(0); 
    
    // nE.setVal(0); TODO: put into findEdgeConstraints(bool init = false) and calls it in every iteration 

    // collect PP, PE, PT, EE, E constraints from bvh 
    if (enablePP_c)
    {
        bvs.resize(svInds.size()); 
        retrieve_bounding_volumes(pol, vtemp, tag, svInds, zs::wrapv<1>{}, 0, bvs);
        svBvh.refit(pol, bvs);         
    }
    bvs.resize(stInds.size());
    retrieve_bounding_volumes(pol, vtemp, tag, stInds, zs::wrapv<3>{}, 0, bvs);
    stBvh.refit(pol, bvs);
    bvs.resize(seInds.size());
    retrieve_bounding_volumes(pol, vtemp, tag, seInds, zs::wrapv<2>{}, 0, bvs);
    seBvh.refit(pol, bvs);

    findConstraintsImpl(pol, dist, false, tag); 

    if (hasBoundary()) {
        bvs.resize(coEles->size());
        retrieve_bounding_volumes(pol, vtemp, tag, *coEles, zs::wrapv<3>{}, coOffset, bvs); 
        bouStBvh.refit(pol, bvs); 
        bvs.resize(coEdges->size()); 
        retrieve_bounding_volumes(pol, vtemp, tag, *coEdges, zs::wrapv<2>{}, coOffset, bvs);
        bouSeBvh.refit(pol, bvs);

        findConstraintsImpl(pol, dist, true, tag); 
    }

    updateConstraintCnt(); 
    D = D_max; 
    if (!silentMode_c)
        fmt::print("[CD] ne: {}, npp: {}, npe: {}, npt: {}, nee: {}\n", 
            ne, npp, npe, npt, nee); 
    if (enableProfile_c)
        timer.tock("proximity search"); 
}

template<class T>
static constexpr T simple_hash(T a)
{
    // https://burtleburtle.net/bob/hash/integer.html
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a; 
}

// reference: https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
template<class T>
static constexpr T combine_hash(T lhs, T rhs)
{
    if constexpr (sizeof(T) >= 8) {
        lhs ^= rhs + 0x517cc1b727220a95 + (lhs << 6) + (lhs >> 2);
    } else {
        lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    }
    return lhs; 
}

bool RapidClothSystem::checkConsColoring(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    zs::Vector<int> correct {vtemp.get_allocator(), 1}; 
    correct.setVal(1); 
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         colors = proxy<space>(colors), 
         lcpMat = proxy<space>(lcpMat), 
         correct = proxy<space>(correct)] __device__ (int i) mutable {
            int color = colors[i]; 
            auto &ap = lcpMat._ptrs; 
            auto &aj = lcpMat._inds; 
            for (int k = ap[i]; k < ap[i + 1]; k++)
            {
                int j = aj[k]; 
                if (j == i)
                    continue; 
                if (colors[j] == color)
                {
                    printf("\t\t\t\t[debug] cons-coloring error at i = %d, j = %d, color_i = %d, color_j = %d at %d-th neighbor\n", 
                        i, j, color, colors[j], k); 
                    correct[0] = 0;  
                    return; 
                }
            }
        }); 
    return correct.getVal() == 1; 
}

void RapidClothSystem::consColoring(zs::CudaExecutionPolicy &pol)
{
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    
    // init weights 
    colorMinWeights.resize(nCons); 
    colorWeights.resize(nCons); 
    {
        bht<int, 1, int> tab{lcpTopMat.get_allocator(), (std::size_t)(nCons * 16)}; 
        tab.reset(pol, true); 
        u32 seed = 114514u; 
        pol(enumerate(colorWeights), 
            [tab_view = proxy<space>(tab), 
            seed = seed, nCons = nCons] __device__ (int idx, u32& w) mutable {
                using tab_t = RM_CVREF_T(tab_view); 
                u32 uidx = idx; 
                u32 v = combine_hash(simple_hash(seed++), simple_hash(uidx)) % (u32)4294967291u;
                tab_view.insert(v);
                while (tab_view.insert(v) != tab_t::sentinel_v) {
                    u32 tmp = combine_hash(simple_hash(seed++), simple_hash(uidx));
                    printf("[%d]-th (%d) random number : %u -> %u\n", idx, (int)nCons, v, tmp);
                    v = tmp;
                }
                w = v; 
        }); 
        if (tab.size() != nCons) {
            fmt::print("{} expected, {} inserted.\n", nCons, tab.size());
            throw std::runtime_error("weight hash failed");
        }
    }
    colorMaskOut.resize(nCons); 
    colorMaskOut.reset(0);
    colors.resize(nCons); 
    colors.reset(-1); 

#if 1
    auto iter = maximum_independent_sets(pol, lcpTopMat, colorWeights, colors);
    nConsColor = iter; 
#else 
    int iter = 0; 
    zs::Vector<int> done{1, memsrc_e::device, 0};
    auto update = 
        [&] (int iter) -> bool {
        done.setVal(1);  
        pol(zip(colorWeights, colorMinWeights, colorMaskOut, colors), 
            [done = proxy<space>(done), 
             iter] __device__ (u32 &w, u32 &mw, int &mask, int &color) {
            //if (w < mw && mask == 0)
            if (w == mw && mw != limits<u32>::max()) 
            {
                done[0] = 0;
                mask = 1;
                color = iter;
                w = limits<u32>::max();
            }
        });
        return done.getVal() == 1;
    };
    for (iter++;;++iter)
    {
        spmv_mask(pol, lcpTopMat, colorWeights, colorMaskOut, colorMinWeights, wrapv<semiring_e::min_times>{});
        if (update(iter))
            break;
    }
    nConsColor = iter - 1; 
#endif 
    if (enableProfile_c)
        timer.tock("constraint coloring"); 
    if (!silentMode_c)
        fmt::print("\t\t[graph coloring] Ended with {} colors\n", nConsColor); 
    if (!checkConsColoring(pol))
        fmt::print("\t\t[graph coloring] Wrong results!\n"); 
}


// xl, cons -> c(xl), J(xl)   
void RapidClothSystem::computeConstraints(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag, 
    T shrinking)
{    
    using namespace zs; 
    constexpr auto space = execspace_e::cuda;
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 

    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp), maxDi = D] __device__ (int vi) mutable {
            vtemp("Di", vi) = maxDi; 
        }); 
    oE.setVal(0); 
    pol(range(ne), [vtemp = proxy<space>({}, vtemp), 
                    tempE = proxy<space>({}, tempE), 
                    tempCons = proxy<space>({}, tempCons), 
                    oE = proxy<space>(oE), 
                    sigma = sigma, 
                    yTag = enableSL ? zs::SmallString{"x0"} : zs::SmallString{"y[k+1]"}, 
                    coOffset = coOffset] __device__ (int i) mutable {
        // if val > 0: tempPP[i] -> tempCons[consInd]
        auto inds = tempE.pack(dim_c<2>, "inds", i, int_c); 
        auto fixi = vtemp("BCfixed", inds[0]) > 0.5f; 
        auto fixj = vtemp("BCfixed", inds[1]) > 0.5f;         
        if (fixi && fixj)
            return; 
        auto xli = vtemp.pack(dim_c<3>, "x(l)", inds[0]); 
        auto xlj = vtemp.pack(dim_c<3>, "x(l)", inds[1]); 
        auto xi = vtemp.pack(dim_c<3>, "y(l)", inds[0]); 
        auto xj = vtemp.pack(dim_c<3>, "y(l)", inds[1]);
        auto yi = vtemp.pack(dim_c<3>, yTag, inds[0]); 
        auto yj = vtemp.pack(dim_c<3>, yTag, inds[1]);
        auto xij_norm = (xi - xj).norm() + eps_c; 
        auto yij_norm_inv = 1.0f / ((yi - yj).norm() + eps_c); 
#if 0
        if ((xli - xlj).norm() * yij_norm_inv <= sigma)
            return; 
        auto grad = - (xi - xj) / xij_norm * yij_norm_inv; 
        auto val = sigma - xij_norm * yij_norm_inv; 
#else 
        auto grad = - 2.f * (xi - xj) * zs::sqr(yij_norm_inv); 
        auto val = sigma * sigma - zs::sqr(xij_norm) * zs::sqr(yij_norm_inv); 
#endif
        // DEBUG  
        // if (val >= 0)
        //     return; 
        auto consInd = atomic_add(exec_cuda, &oE[0], 1); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d + 3, consInd, T_c) = -grad(d); 
        tempCons("val", consInd, T_c) = val; 
        tempCons("vN", consInd) = 2; 
        tempCons("type", consInd) = 4; 
        for (int k = 0; k < 2; k++)
            tempCons("vi", k, consInd) = inds[k]; 
    }); 
    opp = oE.getVal(); 

    oPP.setVal(opp); 
    pol(range(npp), [vtemp = proxy<space>({}, vtemp), 
                    tempPP = proxy<space>({}, tempPP), 
                    tempCons = proxy<space>({}, tempCons), 
                    oPP = proxy<space>(oPP), 
                    delta = delta, tag, 
                    coOffset = coOffset] __device__ (int i) mutable {
        // calculate grad 
        auto inds = tempPP.pack(dim_c<2>, "inds", i, int_c); 
        auto fixi = vtemp("BCfixed", inds[0]) > 0.5f; 
        auto fixj = vtemp("BCfixed", inds[1]) > 0.5f; 
        if (fixi && fixj)
            return; 
        auto xi = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto xj = vtemp.pack(dim_c<3>, tag, inds[1]);
        auto dist = (xi - xj).norm(); 
        auto xij_norm = dist + eps_c; 
        auto delta_inv = 1.0f / delta; 
        auto val = xij_norm * delta_inv - 1.0f; 
        for (int k = 0; k < 2; k++)
            atomic_min(exec_cuda, &vtemp("Di", inds[k]), dist); 
        if (val >= 0)
            return; 
        auto grad = (xi - xj) / xij_norm * delta_inv;             
        auto consInd = atomic_add(exec_cuda, &oPP[0], 1); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d + 3, consInd, T_c) = -grad(d); 
        tempCons("val", consInd, T_c) = val; 
        tempCons("vN", consInd) = 2; 
        tempCons("dist", consInd, T_c) = dist; 
        tempCons("type", consInd) = 2; 
        for (int k = 0; k < 2; k++)
            tempCons("vi", k, consInd) = inds[k]; 
    }); 
    ope = oPP.getVal(); 

    oPE.setVal(ope); 
    pol(range(npe), [vtemp = proxy<space>({}, vtemp), 
                    tempPE = proxy<space>({}, tempPE), 
                    tempCons = proxy<space>({}, tempCons), 
                    oPE = proxy<space>(oPE), 
                    enableDistConstraint = enableDistConstraint, 
                    delta = delta, tag, 
                    coOffset = coOffset] __device__ (int i) mutable {
        // calculate grad 
        auto inds = tempPE.pack(dim_c<3>, "inds", i, int_c); 
        auto fixed = bvec3{
            vtemp("BCfixed", inds[0]) > 0.5f, 
            vtemp("BCfixed", inds[1]) > 0.5f, 
            vtemp("BCfixed", inds[2]) > 0.5f
        };         
        if (fixed[0] && fixed[1] && fixed[2])
            return; 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto e0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto e1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto area = (e0 - p).cross(e1 - p).norm(); 
        auto dist = area / ((e1 - e0).norm() + eps_c); 
        for (int k = 0; k < 3; k++)
            atomic_min(exec_cuda, &vtemp("Di", inds[k]), dist); 
        if (dist >= delta)
            return; 
        T val; 
        zs::vec<T, 9> grad; 
        if (enableDistConstraint) {
            val = dist - delta; 
            grad = dist_grad_pe(p, e0, e1); 
        } else {
            T coef = (e1 - e0).norm() * delta; 
            val = area / coef - 1.0f;           
            PE_area2_grad(p.data(), e0.data(), e1.data(), grad.data());             
            grad /= (2.0f * area * coef + eps_c);        
        }
        auto consInd = atomic_add(exec_cuda, &oPE[0], 1); 
        for (int k = 0; k < 3; k++)
            for (int d = 0; d < 3; d++)
                tempCons("grad", k * 3 + d, consInd, T_c) = grad(k * 3 + d); 
        tempCons("val", consInd, T_c) = val; 
        tempCons("vN", consInd) = 3; 
        tempCons("dist", consInd, T_c) = dist; 
        tempCons("type", consInd) = 3; 
        for (int k = 0; k < 3; k++)
            tempCons("vi", k, consInd) = inds[k]; 
    }); 
    opt = oPE.getVal(); 

    oPT.setVal(opt); 
    pol(range(npt), [vtemp = proxy<space>({}, vtemp), 
                    tempPT = proxy<space>({}, tempPT), 
                    tempCons = proxy<space>({}, tempCons), 
                    oPT = proxy<space>(oPT), 
                    delta = delta, tag, 
                    enableDistConstraint = enableDistConstraint, 
                    enableDegeneratedDist = enableDegeneratedDist, 
                    coOffset = coOffset] __device__ (int i) mutable {
        // calculate grad 
        auto inds = tempPT.pack(dim_c<4>, "inds", i, int_c); 
        auto fixed = bvec4{
            vtemp("BCfixed", inds[0]) > 0.5f, 
            vtemp("BCfixed", inds[1]) > 0.5f, 
            vtemp("BCfixed", inds[2]) > 0.5f, 
            vtemp("BCfixed", inds[3]) > 0.5f
        };         
        if (fixed[0] && fixed[1] && fixed[2] && fixed[3])
            return; 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto t0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto t1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto t2 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        auto pt_dist2 = enableDegeneratedDist ? dist2_pt(p, t0, t1, t2) : 
            dist2_pt_unclassified(p, t0, t1, t2); 
        auto pt_dist = zs::sqrt(pt_dist2); 
        for (int k = 0; k < 4; k++)
            atomic_min(exec_cuda, &vtemp("Di", inds[k]), pt_dist); 
        if (pt_dist2 >= delta * delta)
            return; 
        zs::vec<T, 4, 3> grad; 
        T val; 
        if (enableDistConstraint)
        {
            // TODO: dist constraint 
            val = zs::sqrt(dist2_pt(p, t0, t1, t2)) - delta; 
            grad = dist_grad_pt(p, t0, t1, t2); 
        } else {
            zs::vec<T, 3, 3> mat;
            for (int d = 0; d < 3; d++)
            {
                mat(d, 0) = t0(d) - p(d); 
                mat(d, 1) = t1(d) - p(d); 
                mat(d, 2) = t2(d) - p(d); 
            }
            auto vol = determinant(mat); 
            auto sgn = vol > 0 ? 1.0f : -1.0f; 
            auto area = (t1 - t0).cross(t2 - t0).norm(); 
            auto coef = sgn * area * delta; 
            if (zs::abs(coef) < eps_c)
                coef = sgn * eps_c; 
            val = vol / coef - 1.0f; 
            if (val >= 0)
                return;             
            mat = adjoint(mat).transpose();
        
            for (int d = 0; d < 3; d++)
                grad(0, d) = 0; 
            for (int k = 1; k < 4; k++)
                for (int d = 0; d < 3; d++)
                {
                    grad(k, d) = mat(d, k - 1); 
                    grad(0, d) -= mat(d, k - 1); 
                }
            grad /= coef; 
        }

        auto consInd = atomic_add(exec_cuda, &oPT[0], 1); 
        for (int k = 0; k < 4; k++)
            for (int d = 0; d < 3; d++)
                tempCons("grad", k * 3 + d, consInd, T_c) = grad(k, d); 
        tempCons("val", consInd, T_c) = val; 
        tempCons("vN", consInd) = 4; 
        tempCons("dist", consInd, T_c) = pt_dist; 
        tempCons("type", consInd) = 0; 
        for (int k = 0; k < 4; k++)
            tempCons("vi", k, consInd) = inds[k]; 
    }); 
    oee = oPT.getVal(); 

    oEE.setVal(oee); 
    pol(range(nee), [vtemp = proxy<space>({}, vtemp), 
                    tempEE = proxy<space>({}, tempEE), 
                    tempCons = proxy<space>({}, tempCons), 
                    oEE = proxy<space>(oEE), 
                    delta = delta, tag, 
                    enableDistConstraint = enableDistConstraint, 
                    enableDegeneratedDist = enableDegeneratedDist, 
                    coOffset = coOffset] __device__ (int i) mutable {
        // calculate grad 
        auto inds = tempEE.pack(dim_c<4>, "inds", i, int_c); 
        auto fixed = bvec4{
            vtemp("BCfixed", inds[0]) > 0.5f, 
            vtemp("BCfixed", inds[1]) > 0.5f, 
            vtemp("BCfixed", inds[2]) > 0.5f, 
            vtemp("BCfixed", inds[3]) > 0.5f
        };         
        if (fixed[0] && fixed[1] && fixed[2] && fixed[3])
            return; 
        auto ei0 = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto ei1 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto ej0 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto ej1 = vtemp.pack(dim_c<3>, tag, inds[3]);  
        auto ee_dist2 = enableDegeneratedDist ? safe_dist2_ee(ei0, ei1, ej0, ej1) : 
            safe_dist2_ee_unclassified(ei0, ei1, ej0, ej1); 
        auto ee_dist = zs::sqrt(ee_dist2); 
        for (int k = 0; k < 4; k++)
            atomic_min(exec_cuda, &vtemp("Di", inds[k]), ee_dist); 
        if (ee_dist2 >= delta * delta)
            return; 
        T val, dist; 
        zs::vec<T, 4, 3> grad; 
        if (enableDistConstraint) 
        {
            // TODO: dist constraints
#if 1 
            auto gammas = safe_edge_edge_closest_point(ei0, ei1, ej0, ej1); 
#else 
            auto gammas = edge_edge_closest_point(ei0, ei1, ej0, ej1);
            if ((ei1 - ei0).cross(ej1 - ej0).l2NormSqr() / ((ei1 - ei0).l2NormSqr() * (ej1 - ej0).l2NormSqr() + eps_c) < eps_c)
            {
                // gammas = zs::vec<T, 2>{0.5f, 0.5f};
                auto ejProj = (ej1 - ej0).dot(ei1 - ei0); 
                auto ej0Proj = (ej0 - ei0).dot(ei1 - ei0); 
                auto ej1Proj = (ej1 - ei0).dot(ei1 - ei0); 
                auto ei2 = (ei1 - ei0).l2NormSqr(); 
                if (ej0Proj < 0 && ej1Proj < 0)
                {
                    gammas[0] = 0; 
                    if (ejProj > 0)
                        gammas[1] = 1; 
                    else
                        gammas[1] = 0; 
                } else if (ej0Proj > ei2 && ej1Proj > ei2)
                {
                    gammas[0] = 1; 
                    if (ejProj > 0)
                        gammas[1] = 0;
                    else 
                        gammas[1] = 1; 
                } else {
                    auto minProj = zs::max(zs::min(ej0Proj, ej1Proj) / (ei2 + eps_c), 0.f); 
                    auto maxProj = zs::min(zs::max(ej0Proj, ej1Proj) / (ei2 + eps_c), 1.f); 
                    gammas[0] = (minProj + maxProj) * 0.5f; 
                    gammas[1] = zs::min(zs::max((gammas[0] * ei2 - ej0Proj) / (ejProj + eps_c), 0.f), 1.f); 
                }
            }
#endif 
            auto pi =  gammas[0] * (ei1 - ei0) + ei0; 
            auto pj = gammas[1] * (ej1 - ej0) + ej0;
            auto pji = pi - pj;
            dist = pji.norm();
            val = dist - delta; 
            if (val >= 0)
                return; 
            dist += eps_c; 
            auto piGrad = pji / dist; 
            for (int d = 0; d < 3; d++)
            {
                grad(0, d) = piGrad(d) * (1.f - gammas[0]); 
                grad(1, d) = piGrad(d) * gammas[0]; 
                grad(2, d) = -piGrad(d) * (1.f - gammas[1]); 
                grad(3, d) = -piGrad(d) * gammas[1]; 
            }
        } else {
            zs::vec<T, 3, 3> mat, rMat;
            for (int d = 0; d < 3; d++)
            {
                mat(d, 0) = ei1(d) - ei0(d); 
                mat(d, 1) = ej0(d) - ei0(d); 
                mat(d, 2) = ej1(d) - ei0(d); 
            }
            auto vol = determinant(mat); 
            auto gammas = edge_edge_closest_point(ei0, ei1, ej0, ej1);
            auto pi =  gammas[0] * (ei1 - ei0) + ei0; 
            auto pj = gammas[1] * (ej1 - ej0) + ej0; 
            auto dij = pj - pi; 
            dist = dij.norm(); 
            auto dijNrm = dij / (dist + eps_c);  
            auto ri0 = ei0 + (dist - delta) * 0.5f * dijNrm;  
            auto ri1 = ei1 + (dist - delta) * 0.5f * dijNrm; 
            auto rj0 = ej0 - (dist - delta) * 0.5f * dijNrm; 
            auto rj1 = ej1 - (dist - delta) * 0.5f * dijNrm; 
            for (int d = 0; d < 3; d++)
            {
                rMat(d, 0) = ri1(d) - ri0(d); 
                rMat(d, 1) = rj0(d) - ri0(d); 
                rMat(d, 2) = rj1(d) - ri0(d); 
            }
            auto coef = determinant(rMat);
            val = vol / coef - 1.0f;
            if (val >= 0)
                return;             
            mat = adjoint(mat).transpose();
            for (int d = 0; d < 3; d++)
                grad(0, d) = 0; 
            for (int k = 1; k < 4; k++)
                for (int d = 0; d < 3; d++)
                {
                    grad(k, d) = mat(d, k - 1); 
                    grad(0, d) -= mat(d, k - 1); 
                }
            grad /= coef; 
        }

        auto consInd = atomic_add(exec_cuda, &oEE[0], 1); 
        for (int k = 0; k < 4; k++)
            for (int d = 0; d < 3; d++)
                tempCons("grad", k * 3 + d, consInd, T_c) = grad(k, d); 
        tempCons("val", consInd, T_c) = val; 
        tempCons("vN", consInd) = 4; 
        tempCons("dist", consInd, T_c) = dist; 
        tempCons("type", consInd) = 1; 
        for (int k = 0; k < 4; k++)
            tempCons("vi", k, consInd) = inds[k]; 
    }); 
    nCons = oEE.getVal(); 
    nae = opp, napp = ope - opp, nape = opt - ope, napt = oee - opt, naee = nCons - opt; 
    if (!silentMode_c)
        fmt::print("[CD] Active Constraints: e: {}, pp: {}, pe: {}, pt: {}, ee: {}, nCons: {}\n", 
            nae, napp, nape, napt, naee, nCons); 

    // TODO: construct lcp matrix & coloring initialization 
    // init vert cons list 
    // clear vertex -> cons list size 
    pol(range(vCons.size()), 
        [vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
            vCons("n", i) = 0; 
        }); 
    pol(range(nCons), 
        [tempCons = view<space>({}, tempCons, false_c, "tempCons"), 
         vtemp = proxy<space>({}, vtemp), 
         vCons = view<space>({}, vCons, false_c, "vCons"), 
         coOffset = coOffset] __device__ (int ci) mutable {
            int vN = tempCons("vN", ci); 
            for (int k = 0; k < vN; k++)
            {
                int vi = tempCons("vi", k, ci); 
                if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
                    continue; 
                int n = atomic_add(exec_cuda, &vCons("n", vi), 1); 
                vCons("cons", n, vi) = ci; 
                vCons("ind", n, vi) = k; 
            }
        }); 
    if (enableProfile_c)
        timer.tock("compute constraints gradients"); 
    
    if (showStatistics_c)
    {
        auto maxN = tvMax(pol, vCons, "n", coOffset, int_c); 
        fmt::print(fg(fmt::color::sea_green), "\t[statistics]\tmax_vCons_cache: {}\n", 
            maxN); 
    }

    if (enableProfile_c)
        timer.tick(); 
    lcpMatSize.setVal(0); 
    pol(range(nCons), 
        [vCons = proxy<space>({}, vCons), 
         vtemp = proxy<space>({}, vtemp), 
         tempCons = proxy<space>({}, tempCons), 
         lcpMatIs = view<space>(lcpMatIs, false_c, "lcpMatIs"), 
         lcpMatJs = view<space>(lcpMatJs, false_c, "lcpMatJs"), 
         lcpMatSize = view<space>(lcpMatSize, false_c, "lcpMatSize"), 
         shrinking, coOffset = coOffset] __device__ (int ci) mutable {
            int vN = tempCons("vN", ci); 
            for (int k = 0; k < vN; k++)
            {
                int vi = tempCons("vi", k, ci); 
                if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
                    continue; 
                int n = vCons("n", vi); 
                for (int j = 0; j < n; j++)
                {
                    int aj = vCons("cons", j, vi); 
                    auto no = atomic_add(exec_cuda, &lcpMatSize[0], 1); 
                    lcpMatIs[no] = ci; 
                    lcpMatJs[no] = aj; 
                }
            }
         }); 
    auto lcpSize = lcpMatSize.getVal();
    lcpMatIs.resize(lcpSize); 
    lcpMatJs.resize(lcpSize); 
    lcpMat.build(pol, nCons, nCons, lcpMatIs, lcpMatJs, false_c);
    lcpMat.localOrdering(pol, false_c);  
    lcpMat._vals.resize(lcpMat.nnz());

    lcpTopMat._nrows = lcpMat.rows();
    lcpTopMat._ncols = lcpMat.cols();
    lcpTopMat._ptrs = lcpMat._ptrs;
    lcpTopMat._inds = lcpMat._inds;
    lcpTopMat._vals = zs::Vector<zs::u32>{lcpTopMat._inds.get_allocator(),lcpTopMat._inds.size()};
    lcpMatIs.resize(spmatCps); 
    lcpMatJs.resize(spmatCps);  
    // compute lcpMat = J * M^{-1} * J.T
    pol(range(lcpMat.nnz()), 
        [lcpMat = proxy<space>(lcpMat), 
         lcpTopMat = proxy<space>(lcpTopMat)] __device__ (int i) mutable {
            lcpMat._vals[i] = 0.f;
            lcpTopMat._vals[i] = 1u; 
        });

    // A = J * M^{-1} * J.T
    pol(range(nCons), 
        [tempCons = view<space>({}, tempCons, false_c, "tempCons"), 
        vtemp = proxy<space>({}, vtemp), 
        vCons = view<space>({}, vCons, false_c, "vCons"), 
        lcpMat = proxy<space>(lcpMat), 
        coOffset = coOffset] __device__ (int i) mutable {
            auto &ax = lcpMat._vals; 
            int vN = tempCons("vN", i); 
            for (int j = 0; j < vN; j++)                        // this V
            {
                int vi = tempCons("vi", j, i); 
                if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
                    continue; 
                int n = vCons("n", vi);
                for (int k = 0; k < n; k++)
                {
                    int neCons = vCons("cons", k, vi); 
                    int neV = vCons("ind", k, vi); 
                    auto m = vtemp("ws", vi); // TODO: no ofb information when typing vtemp("ws", k, vi)? 
                    if (m <= 0.f)
                        m = eps_c; 
                    auto mInv = 1.0f / m;  
                    // cons.grad(j) * m_inv * neCons.grad(neV)
                    T val = 0.f; 
                    for (int d = 0; d < 3; d++)
                        val += tempCons("grad", j * 3 + d, i, T_c) * mInv * 
                            tempCons("grad", neV * 3 + d, neCons, T_c); 
                    auto spInd = lcpMat.locate(i, neCons, true_c); 
                    atomic_add(exec_cuda, &ax[spInd], val); 
                }
            }
        }); 
    if (enableProfile_c)
        timer.tock("construct lcp matrix"); 
}

// yl, y[k], (c, J), xl -> lambda_{l+1}, y_{l+1} 
void RapidClothSystem::solveLCP(zs::CudaExecutionPolicy &pol)
{
    // PGS solver 
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 
    // b = c(x(l)) + J(x(l)) * (y(l) - x(l))
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         vtemp = proxy<space>({}, vtemp), coOffset = coOffset] __device__ (int ci) mutable {   
            auto val = tempCons("val", ci, T_c); 
            if (tempCons("type", ci) != 4) // edge constraints are linearized at y(l) instead of x(l)
                for (int i = 0; i < tempCons("vN", ci); i++)
                {
                    int vi = tempCons("vi", i, ci); 
                    // if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
                    //     continue; 
                    for (int d = 0; d < 3; d++)
                        val += tempCons("grad", i * 3 + d, ci, T_c) * 
                            (vtemp("y(l)", d, vi) - vtemp("x(l)", d, vi)); 
                }
            tempCons("b", ci, T_c) = -val; 
            tempCons("lambda", ci, T_c) = 0.f; 
         }); 
    
    for (int iter = 0; iter < lcpCap; iter++)
        for (int color = 0; color < nConsColor; color++)
            pol(range(nCons), 
                [tempCons = proxy<space>({}, tempCons), 
                colors = proxy<space>(colors), 
                lcpMat = proxy<space>(lcpMat), 
                lcpTol = lcpTol, color] __device__ (int i) mutable {
                    if (colors[i] != color + 1)
                        return; 
                    auto &ap = lcpMat._ptrs; 
                    auto &aj = lcpMat._inds; 
                    auto &ax = lcpMat._vals;
                    auto oldLam = tempCons("lambda", i, T_c); 
                    T maj = 0; 
                    T rhs = tempCons("b", i, T_c); 
                    for (int k = ap[i]; k < ap[i + 1]; k++)
                    {
                        auto j = aj[k]; 
                        if (j == i)
                        {
                            maj += ax[k]; 
                            continue; 
                        }
                        rhs -= ax[k] * tempCons("lambda", j, T_c); 
                    } 
                    auto newLam = zs::max(rhs / maj, 0.f); 
                    tempCons("lambda", i, T_c) = newLam; 
                }); 
    if (enableProfile_c)
        timer.tock("solve LCP"); 
}      

// call cons + solveLCP 
void RapidClothSystem::backwardStep(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    // dynamicsStep should be done previously 
    // x(l), y[k+1] -> LCP -> updated lambda -> updated y(l) 
    computeConstraints(pol, "x(l)"); 
    consColoring(pol); 
    solveLCP(pol); 
    // y(l+1) = M^{-1} * (J(l)).T * lambda + y(l)
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         vtemp = proxy<space>({}, vtemp), 
         coOffset = coOffset] __device__ (int ci) mutable {
            if (tempCons("val", ci, T_c) == 0)
                return; 
            int n = tempCons("vN", ci); 
            auto lambda = tempCons("lambda", ci, T_c); 
            for (int k = 0; k < n; k++)
            {
                int vi = tempCons("vi", k, ci); 
                if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
                    continue; 
                auto m = vtemp("ws", vi); 
                if (m <= 0.f)
                    m = eps_c; 
                auto mInv = 1.0f / m; 
                for (int d = 0; d < 3; d++)
                {
                    atomic_add(exec_cuda, &vtemp("y(l)", d, vi), 
                        mInv * lambda * tempCons("grad", k * 3 + d, ci, T_c)); 
                }
            }
        }); 
    
    if (enableFriction)
    {
        // NOTE: for now, only support PT&&EE & distance-based constraints 
        for (int iter = 0; iter < lcpCap; iter++)
        {
            for (int color = 0; color < nConsColor; color++)
            {
                pol(range(nCons), 
                    [tempCons = proxy<space>({}, tempCons), 
                    vtemp = proxy<space>({}, vtemp), 
                    colors = proxy<space>(colors), 
                    coOffset = coOffset, 
                    clothFricMu = clothFricMu, 
                    boundaryFricMu = boundaryFricMu, 
                    color] __device__ (int ci) mutable {
                        if (tempCons("val", ci, T_c) == 0)
                            return; 
                        if (colors[ci] != color + 1)
                            return; 
                        auto consType = tempCons("type", ci); 
                        auto lambda = tempCons("lambda", ci, T_c); 
                        if (consType == 0)
                        {
                            // pt 
                            T intensity = 0.f; 
                            auto inds = tempCons.pack(dim_c<4>, "vi", ci);
                            auto xp =  vtemp.pack(dim_c<3>, "x[k]", inds[0]); 
                            auto xt0 =  vtemp.pack(dim_c<3>, "x[k]", inds[1]); 
                            auto xt1 =  vtemp.pack(dim_c<3>, "x[k]", inds[2]); 
                            auto xt2 =  vtemp.pack(dim_c<3>, "x[k]", inds[3]); 
                            auto yp =  vtemp.pack(dim_c<3>, "y(l)", inds[0]); 
                            auto yt0 =  vtemp.pack(dim_c<3>, "y(l)", inds[1]); 
                            auto yt1 =  vtemp.pack(dim_c<3>, "y(l)", inds[2]); 
                            auto yt2 =  vtemp.pack(dim_c<3>, "y(l)", inds[3]); 
                            auto dxp = yp - xp; 
                            auto dxt0 = yt0 - xt0; 
                            auto dxt1 = yt1 - xt1; 
                            auto dxt2 = yt2 - xt2; 
                            auto xlp =  vtemp.pack(dim_c<3>, "x(l)", inds[0]); 
                            auto xlt0 =  vtemp.pack(dim_c<3>, "x(l)", inds[1]); 
                            auto xlt1 =  vtemp.pack(dim_c<3>, "x(l)", inds[2]); 
                            auto xlt2 =  vtemp.pack(dim_c<3>, "x(l)", inds[3]); 
                            auto nrm = (xlt1 - xlt0).cross(xlt2 - xlt0);
                            nrm = nrm / zs::max(nrm.norm(), eps_c); 
                            dxp = dxp - dxp.dot(nrm) * nrm; 
                            dxt0 = dxt0 - dxt0.dot(nrm) * nrm; 
                            dxt1 = dxt1 - dxt1.dot(nrm) * nrm; 
                            dxt2 = dxt2 - dxt2.dot(nrm) * nrm; 
                            auto betas = point_triangle_closest_point(xlp, xlt0, xlt1, xlt2); 
                            auto weights = zs::vec<T, 4> {1.f, -(1.f - betas[0] - betas[1]), 
                                -betas[0], -betas[1]}; 
                            auto dxRel = dxp + weights[1] * dxt0 + weights[2] * dxt1 + weights[3] * dxt2; 
                            auto dxRelNorm = dxRel.norm(); 
                            bool boundaryInvolved = false; 
                            for (int k = 0; k < 4; k++)
                            {
                                if (weights[k] < 0 || weights[k] > 1)
                                    return; 
                                if (inds[k] < coOffset)
                                {
                                    auto m = vtemp("ws", inds[k]); 
                                    intensity += 1.f / zs::max(m, eps_c) * zs::sqr(weights[k]); 
                                } else {
                                    boundaryInvolved = true; 
                                }
                            }
                            auto fricMu = boundaryInvolved ? boundaryFricMu: clothFricMu; 
                            // TODO: clamp intensity
                            intensity = 1.f / zs::max(intensity, eps_c);  
                            intensity = zs::max(zs::min(intensity, fricMu * lambda), -fricMu * lambda); 
                            for (int k = 0; k < 4; k++)
                            {
                                if (inds[k] < coOffset)
                                {
                                    auto m = vtemp("ws", inds[k]); 
                                    for (int d = 0; d < 3; d++)
                                        atomic_add(exec_cuda, &vtemp("y(l)", d, inds[k]), 
                                            -1.f * intensity / zs::max(m, eps_c) * dxRel(d) * weights[k]); 
                                }
                            }
                        } else if (consType == 1) {
                            // ee 
                            T intensity = 0.f; 
                            auto inds = tempCons.pack(dim_c<4>, "vi", ci);
                            auto xei0 =  vtemp.pack(dim_c<3>, "x[k]", inds[0]); 
                            auto xei1 =  vtemp.pack(dim_c<3>, "x[k]", inds[1]); 
                            auto xej0 =  vtemp.pack(dim_c<3>, "x[k]", inds[2]); 
                            auto xej1 =  vtemp.pack(dim_c<3>, "x[k]", inds[3]); 
                            auto yei0 =  vtemp.pack(dim_c<3>, "y(l)", inds[0]); 
                            auto yei1 =  vtemp.pack(dim_c<3>, "y(l)", inds[1]); 
                            auto yej0 =  vtemp.pack(dim_c<3>, "y(l)", inds[2]); 
                            auto yej1 =  vtemp.pack(dim_c<3>, "y(l)", inds[3]); 
                            auto dxei0 = yei0 - xei0; 
                            auto dxei1 = yei1 - xei1; 
                            auto dxej0 = yej0 - xej0; 
                            auto dxej1 = yej1 - xej1; 
                            auto xlei0 =  vtemp.pack(dim_c<3>, "x(l)", inds[0]); 
                            auto xlei1 =  vtemp.pack(dim_c<3>, "x(l)", inds[1]); 
                            auto xlej0 =  vtemp.pack(dim_c<3>, "x(l)", inds[2]); 
                            auto xlej1 =  vtemp.pack(dim_c<3>, "x(l)", inds[3]); 

                            auto gammas = safe_edge_edge_closest_point(xlei0, xlei1, xlej0, xlej1);
                            auto xlpi =  gammas[0] * (xlei1 - xlei0) + xlei0;
                            auto xlpj = gammas[1] * (xlej1 - xlej0) + xlej0;
                            auto xlpji = xlpi - xlpj;
                            auto nrm = xlpji / zs::max(xlpji.norm(), eps_c); 

                            dxei0 = dxei0 - dxei0.dot(nrm) * nrm; 
                            dxei1 = dxei1 - dxei1.dot(nrm) * nrm; 
                            dxej0 = dxej0 - dxej0.dot(nrm) * nrm; 
                            dxej1 = dxej1 - dxej1.dot(nrm) * nrm; 

                            auto weights = zs::vec<T, 4> {1.f - gammas[0], gammas[0], 
                                1.f - gammas[1], gammas[1]}; 
                            auto dxRel = weights[0] * dxei0 + weights[1] * dxei1 + 
                                weights[2] * dxej0 + weights[3] * dxej1; 
                            auto dxRelNorm = dxRel.norm(); 
                            bool boundaryInvolved = false; 
                            for (int k = 0; k < 4; k++)
                            {
                                if (inds[k] < coOffset)
                                {
                                    auto m = vtemp("ws", inds[k]); 
                                    intensity += 1.f / zs::max(m, eps_c) * zs::sqr(weights[k]); 
                                } else {
                                    boundaryInvolved = true; 
                                }
                            }
                            auto fricMu = boundaryInvolved ? boundaryFricMu: clothFricMu; 
                            // TODO: clamp intensity
                            intensity = 1.f / zs::max(intensity, eps_c);  
                            intensity = zs::max(zs::min(intensity, fricMu * lambda), -fricMu * lambda);  
                            for (int k = 0; k < 4; k++)
                            {
                                if (inds[k] < coOffset)
                                {
                                    auto m = vtemp("ws", inds[k]); 
                                    for (int d = 0; d < 3; d++)
                                        atomic_add(exec_cuda, &vtemp("y(l)", d, inds[k]), 
                                            -1.f * intensity / zs::max(m, eps_c) * dxRel(d) * weights[k]); 
                                }
                            }
                        }
                    }); 
            }
        }
    }
    if (enableProfile_c)
        timer.tock("update y(l)"); 
}   

// async stepping  
void RapidClothSystem::forwardStep(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    zs::CppTimer timer; 
    if (enableProfile_c)
        timer.tick(); 
    // debug 
    {
        auto iterPrim = std::make_shared<PrimitiveObject>(); 
        auto iterBouPrim = std::make_shared<PrimitiveObject>(); 
        int n = coOffset;
        int nBou = numBouDofs; 
        auto hv = vtemp.clone({memsrc_e::host, -1}); 
        auto hv_view = proxy<execspace_e::host>({}, hv);  
        
        iterPrim->verts.resize(n * 2); 
        iterPrim->lines.resize(n); 
        for (int vi = 0; vi < n; vi++)
        {
            iterPrim->verts.values[vi] = zeno::vec3f {
                hv_view("x(l)", 0, vi), 
                hv_view("x(l)", 1, vi), 
                hv_view("x(l)", 2, vi)
            }; 
            iterPrim->verts.values[vi + n] = zeno::vec3f {
                hv_view("y(l)", 0, vi), 
                hv_view("y(l)", 1, vi), 
                hv_view("y(l)", 2, vi)
            }; 
            iterPrim->lines.values[vi] = zeno::vec2i {vi, vi + n}; 
        }

        iterBouPrim->verts.resize(nBou * 2); 
        iterBouPrim->lines.resize(nBou); 
        for (int ofs = 0; ofs < nBou; ofs++)
        {
            auto vi = ofs + n; 
            iterBouPrim->verts.values[ofs] = zeno::vec3f {
                hv_view("x(l)", 0, vi), 
                hv_view("x(l)", 1, vi), 
                hv_view("x(l)", 2, vi)
            }; 
            iterBouPrim->verts.values[ofs + nBou] = zeno::vec3f {
                hv_view("y(l)", 0, vi), 
                hv_view("y(l)", 1, vi), 
                hv_view("y(l)", 2, vi)
            }; 
            iterBouPrim->lines.values[ofs] = zeno::vec2i {ofs, ofs + nBou}; 
        }

        auto ht = stInds.clone({memsrc_e::host, -1}); 
        auto ht_view = proxy<execspace_e::host>({}, ht); 
        int tn = stInds.size(); 
        iterPrim->tris.resize(tn); 
        for (int ti = 0; ti < tn; ti++)
        {
            iterPrim->tris.values[ti] = zeno::vec3i {ht_view("inds", 0, ti, int_c), 
                ht_view("inds", 1, ti, int_c), ht_view("inds", 2, ti, int_c)}; 
        }
        iterPrims->arr.push_back(std::move(iterPrim)); 
        iterBouPrims->arr.push_back(std::move(iterBouPrim)); 
    }

    // updated y(l) -> updated x(l)
#if 0 
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp),
         coOffset = coOffset, 
         gamma = gamma] __device__ (int vi) mutable {
            auto y = vtemp.pack(dim_c<3>, "y(l)", vi); 
            auto x = vtemp.pack(dim_c<3>, "x(l)", vi); 
            if ((y - x).l2NormSqr() < eps_c)
            {
                vtemp("disp", vi) = 0.f;
                vtemp("r(l)", vi) = 0.f; 
                return; 
            }
            auto alpha = 0.5f * vtemp("Di", vi) * gamma / 
                ((y - x).norm() + eps_c);
            if (alpha > 1.0f)
                alpha = 1.0f; 
            vtemp.tuple(dim_c<3>, "x(l)", vi) = vtemp.pack(dim_c<3>, "x(l)", vi) + 
                alpha * (y - x); 
            vtemp("r(l)", vi) *= 1.0f - alpha; 
            vtemp("disp", vi) = alpha * (y - x).norm(); 
        }); 
#else 
    // async + sync
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp),
         coOffset = coOffset, 
         gamma = gamma] __device__ (int vi) mutable {
            auto y = vtemp.pack(dim_c<3>, "y(l)", vi); 
            auto x = vtemp.pack(dim_c<3>, "x(l)", vi); 
            vtemp("sync", vi) = 0.f; 
            if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
            {
                vtemp("temp", 0,  vi) = 0.f;                 
                if ((y - x).l2NormSqr() < eps_c)
                {
                    vtemp("disp", vi) = 0.f;
                    vtemp("r(l)", vi) = 0.f; 
                    vtemp("alpha", vi) = 1.f; 
                    return;
                }  
                // gamma = 0.6f;               
            }
            auto alpha = 0.5f * vtemp("Di", vi) * gamma / 
                ((y - x).norm() + eps_c);
            if (alpha > 1.0f)
                alpha = 1.0f; 
            vtemp("alpha", vi) = alpha; 
        }); 
#if 1
    temp.resize(1); 
    temp.setVal(1.f); 
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         vtemp = proxy<space>({}, vtemp), 
         temp = proxy<space>(temp), 
         coOffset = coOffset, 
         bouTinyDist = bouTinyDist, 
         tinyDist = tinyDist] __device__ (int ci) mutable {
            if (tempCons("dist", ci, T_c) < bouTinyDist) // slow_boundary_dist
            {
                int vN = tempCons("vN", ci); 
                for (int k = 0; k < vN; k++)
                {
                    int vi = tempCons("vi", k, ci); 
                    if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
                        vtemp("temp", 0, vi) = 1.f; // isBou and included in tiny pairs 
                }
            }
            auto syncAlpha = limits<T>::max(); 
            if (tempCons("dist", ci, T_c) >= tinyDist)
                return; 
            // sync alpha 
            int vN = tempCons("vN", ci); 
            for (int k = 0; k < vN; k++)
            {
                int vi = tempCons("vi", k, ci); 
                vtemp("sync", vi) = 1.f; 
                if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f) // do not use bou step to sync  
                    continue; 
                auto alpha = vtemp("alpha", vi); 
                syncAlpha = zs::min(syncAlpha, alpha); 
            }
            atomic_min(exec_cuda, &temp[0], syncAlpha); 
        }); 
    auto alpha = temp.getVal(); 
#else 
    auto alpha = tvMin(pol, vtemp, "alpha", coOffset, T_c); 
    fmt::print("uniform alpha: {}\n", alpha); 
#endif 
// #if 1
//     temp.resize(1); 
//     temp.setVal(1.f); 
//     pol(range(nCons), 
//         [tempCons = proxy<space>({}, tempCons), 
//          vtemp = proxy<space>({}, vtemp), 
//          temp = proxy<space>(temp), 
//          coOffset = coOffset, 
//          bouTinyDist = bouTinyDist, 
//          tinyDist = tinyDist] __device__ (int ci) mutable {
//             // if (tempCons("dist", ci, T_c) < bouTinyDist) // slow_boundary_dist
//             // {
//             //     int vN = tempCons("vN", ci); 
//             //     for (int k = 0; k < vN; k++)
//             //     {
//             //         int vi = tempCons("vi", k, ci); 
//             //         if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f)
//             //             vtemp("temp", 0, vi) = 1.f; // isBou and included in tiny pairs 
//             //     }
//             // }
//             auto syncAlpha = limits<T>::max(); 
//             if (tempCons("dist", ci, T_c) >= tinyDist)
//                 return; 
//             // sync alpha 
//             int vN = tempCons("vN", ci); 
//             for (int k = 0; k < vN; k++)
//             {
//                 int vi = tempCons("vi", k, ci); 
//                 vtemp("sync", vi) = 1.f; 
//                 // if (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f) // do not use bou step to sync  
//                 //     continue; 
//                 auto alpha = vtemp("alpha", vi); 
//                 syncAlpha = zs::min(syncAlpha, alpha); 
//             }
//             atomic_min(exec_cuda, &temp[0], syncAlpha); 
//         }); 
//     auto alpha = temp.getVal(); 
// #else 
//     auto alpha = tvMin(pol, vtemp, "alpha", coOffset, T_c); 
//     fmt::print("uniform alpha: {}\n", alpha); 
// #endif 
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp),
         coOffset = coOffset, 
         gamma = gamma
          , syncAlpha = alpha
         ] __device__ (int vi) mutable {
            auto y = vtemp.pack(dim_c<3>, "y(l)", vi); 
            auto x = vtemp.pack(dim_c<3>, "x(l)", vi); 
            auto isBou = (vi >= coOffset || vtemp("BCfixed", vi) > 0.5f); 
            if ((y - x).l2NormSqr() < eps_c && isBou)
                return; 
            auto alpha = vtemp("sync", vi) > 0.5f ? syncAlpha : vtemp("alpha", vi); 
            {
                if (isBou && vtemp("sync", vi) > 0.5f) // stop tiny bou 
                {
                    vtemp("disp", vi) = 0.f; 
                    return; 
                }
                if (isBou && vtemp("temp", 0, vi) > 0.5f)
                {
                    vtemp("disp", vi) = 0.f; 
                    return; 
                }
            }
            vtemp.tuple(dim_c<3>, "x(l)", vi) = vtemp.pack(dim_c<3>, "x(l)", vi) + 
                alpha * (y - x); 
            vtemp("r(l)", vi) *= 1.0f - alpha; 
            vtemp("disp", vi) = alpha * (y - x).norm(); 
        }); 

#endif 
    if (enableProfile_c)
        timer.tock("forward step"); 
}
}