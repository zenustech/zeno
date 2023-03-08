#include "RapidCloth.cuh"
#include "Structures.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/Distance.hpp"

void findConstraintsImpl(zs::CudaExecutionPolicy &pol, T radius, bool withBoundary, const zs::SmallString &tag)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    
    // p -> t
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    auto &stfront = withBoundary ? boundaryStFront : selfStFront;
    opt = ne; 
    pol(Collapse{stfront.size()},
        [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, eles = proxy<space>({}, withBoundary ? *coEles : stInds),
         vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stbvh), front = proxy<space>(stfront), tempPT = proxy<space>({}, tempPT),
         vCons = proxy<space>({}, vCons), 
         nPT = proxy<space>(nPT), radius, xi, voffset = withBoundary ? coOffset : 0,
         frontManageRequired = frontManageRequired, tag, opt] __device__(int i) mutable {
            auto vi = front.prim(i);
            vi = spInds("inds", vi, int_c); 
            const auto dHat2 = zs::sqr(radius);
            auto p = vtemp.pack(dim_c<3>, tag, vi);
            auto bv = bv_t{get_bounding_box(p - radius, p + radius)};
            auto f = [&](int stI) {
                auto tri = eles.pack(dim_c<3>, "inds", stI, int_c) + voffset;
                if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                    return;
                // ccd
                auto t0 = vtemp.pack(dim_c<3>, tag, tri[0]);
                auto t1 = vtemp.pack(dim_c<3>, tag, tri[1]);
                auto t2 = vtemp.pack(dim_c<3>, tag, tri[2]);

                if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
                    auto no = atomic_add(exec_cuda, &nPT[0], 1); 
                    auto inds = pair4_t{vi, tri[0], tri[1], tri[2]}; 
                    tempPT.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                    auto consInd = no + opt; 
                    for (int k = 0; k < 4; k++)
                    {
                        auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                        vCons("cons", no, inds[k]) = consInd; 
                    }
                }

            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        stfront.reorder(pol);   
    npt = nPT.getVal(); 
    oee = opt + npt; 

    // e -> e
    const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
    auto &seefront = withBoundary ? boundarySeeFront : selfSeeFront;
    pol(Collapse{seefront.size()},
        [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
            vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(seefront),
            vCons = proxy<space>({}, vCons), 
            tempEE = proxy<space>(tempEE), nEE = proxy<space>(nEE), dHat2 = zs::sqr(radius), xi,
            radius, voffset = withBoundary ? coOffset : 0,
            frontManageRequired = frontManageRequired, tag, oee] __device__(int i) mutable {
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

                if (auto d2 = dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                    auto inds = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                    tempEE.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                    auto consInd = no + oee; 
                    for (int k = 0; k < 4; k++)
                    {
                        auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                        vCons("cons", no, inds[k]) = consInd; 
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
    nee = nEE.getVal(); 
    ope = oee + nee; 

    // e -> p 
    auto &sevfront = withBoundary ? boundarySevFront : selfSevFront;
    pol(Collapse{sevfront.size()},
        [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
            sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
            vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(sevfront),
            tempPE = proxy<space>(tempPE), nPE = proxy<space>(nPE), dHat2 = zs::sqr(radius), xi,
            radius, voffset = withBoundary ? coOffset : 0,
            frontManageRequired = frontManageRequired, tag, ope] __device__(int i) mutable {
            auto vi = front.prim(i);
            vi = spInds("inds", vi, int_c); 
            const auto dHat2 = zs::sqr(radius);
            auto p = vtemp.pack(dim_c<3>, tag, vi);
            auto bv = bv_t{get_bounding_box(p - radius, p + radius)};
            auto f = [&](int sej) {
                if (voffset == 0 && sei <= sej)
                    return;
                auto ejInds = sedges.pack(dim_c<2>, "inds", sej, int_c) + voffset;
                if (vi == ejInds[0] || vi == ejInds[1])
                    return; 
                auto v2 = vtemp.pack(dim_c<3>, tag, ejInds[0]);
                auto v3 = vtemp.pack(dim_c<3>, tag, ejInds[1]);

                if (auto d2 = dist2_pe(p, v2, v3); d2 < dHat2) {
                    auto no = atomic_add(exec_cuda, &nPE[0], 1); 
                    auto inds = pair3_t{vi, ejInds[0], ejInds[1]};
                    tempPE.tuple(dim_c<3>, "inds", no, int_c) = inds; 
                    auto consInd = no + ope; 
                    for (int k = 0; k < 3; k++)
                    {
                        auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                        vCons("cons", no, inds[k]) = consInd; 
                    }
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        sevfront.reorder(pol);
    npe = nPE.getVal(); 
    opp = npe + ope; 

    // v-> v
    if (!withBoundary)
    {
        const auto &svbvh = svBvh;
        auto &svfront = selfSvFront;
        pol(Collapse{svfront.size()},
            [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
            bvh = proxy<space>(stbvh), front = proxy<space>(svfront), tempPP = proxy<space>(tempPP),
            vCons = proxy<space>({}, vCons), 
            nPP = proxy<space>(nPP), radius, xi, voffset = withBoundary ? coOffset : 0,
            frontManageRequired = frontManageRequired, tag, opp] __device__(int i) mutable {
                auto vi = front.prim(i);
                vi = spInds("inds", vi, int_c); 
                const auto dHat2 = zs::sqr(radius);
                auto pi = vtemp.pack(dim_c<3>, tag, vi);
                auto bv = bv_t{get_bounding_box(p - radius, p + radius)};
                auto f = [&](int svI) {
                    if (voffset == 0 && vi <= vj)
                        return; 
                    auto vj = eles("inds", svI, int_c) + voffset; 
                    auto pj = vtemp.pack(dim_c<3>, tag, vj); 
                    if (auto d2 = dist2_pp(pi, pj); d2 < dHat2) {
                        auto no = atomic_add(exec_cuda, &nPP[0], 1); 
                        auto inds = pair2_t{vi, vj};
                        tempPP.tuple(dim_c<2>, "inds", no, int_c) = inds; 
                        auto consInd = no + opp; 
                        for (int k = 0; k < 2; k++)
                        {
                            auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                            vCons("cons", no, inds[k]) = consInd; 
                        }
                    }

                if (frontManageRequired)
                    bvh.iter_neighbors(bv, i, front, f);
                else
                    bvh.iter_neighbors(bv, front.node(i), f);
            });     
        if (frontManageRequired)
            svfront.reorder(pol);   
        npp = nPP.getVal(); 
    }
}

void findConstraints(zs::CudaExecutionPolicy &pol, T dist, const zs::SmallString &tag)
{
    // TODO: compute oE in initialize
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    nPP.setVal(0);
    nPE.setVal(0);  
    nPT.setVal(0); 
    nEE.setVal(0); 
    pol(range(vCons.size()), [vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
        vCons("n", i) = vCons("nE", i); 
    }); 
    
    // nE.setVal(0); TODO: put into findEdgeConstraints(bool init = false) and calls it in every iteration 

    // collect PP, PE, PT, EE, E constraints from bvh 
    bvs.resize(svInds.size()); 
    retrieve_bounding_volumes(pol, vtemp, tag, svInds, zs::wrapv<1>{}, 0, bvs);
    svBvh.refit(pol, bvs); 
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
    D = D_max; 
    // TODO: coloring for multi-color PGS 
}

// xl, cons -> c(xl), J(xl)   
void computeConstraints(zs::CudaExecutionPolicy &pol); 

// yl, y[k], (c, J), xl -> lambda_{l+1}, y_{l+1} 
void solveLCP(zs::CudaExecutionPolicy &pol);        

// call cons + solveLCP 
void backwardStep(zs::CudaExecutionPolicy &pol);    

// async stepping  
void forwardStep(zs::CudaExecutionPolicy &pol);     