#include "RapidCloth.cuh"
#include "Structures.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/Distance.hpp"
#include "RapidClothGradHess.inl"

static void findConstraintsImpl(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::T radius, bool withBoundary, const zs::SmallString &tag)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    
    // p -> t
    const auto &stbvh = withBoundary ? bouStBvh : stBvh;
    auto &stfront = withBoundary ? boundaryStFront : selfStFront;
    opt = ne; 
    pol(Collapse{stfront.size()},
        [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
         eles = proxy<space>({}, withBoundary ? *coEles : stInds),
         vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stbvh), 
         front = proxy<space>(stfront), tempPT = proxy<space>({}, tempPT),
         vCons = proxy<space>({}, vCons), 
         nPT = proxy<space>(nPT), radius, xi, voffset = withBoundary ? coOffset : 0,
         frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
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
                    // auto consInd = no + opt; 
                    // for (int k = 0; k < 4; k++)
                    // {
                    //     auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                    //     // vCons("cons", no, inds[k]) = consInd; 
                    // }
                }

            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        stfront.reorder(pol);   
    // npt = nPT.getVal(); 
    // oee = opt + npt; 

    // e -> e
    const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
    auto &seefront = withBoundary ? boundarySeeFront : selfSeeFront;
    pol(Collapse{seefront.size()},
        [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
            vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(seefront),
            vCons = proxy<space>({}, vCons), 
            tempEE = proxy<space>(tempEE), nEE = proxy<space>(nEE), dHat2 = zs::sqr(radius), xi,
            radius, voffset = withBoundary ? coOffset : 0,
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

                if (auto d2 = dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                    auto inds = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                    tempEE.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                    auto consInd = no + oee; 
                    // for (int k = 0; k < 4; k++)
                    // {
                    //     auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                    //     // vCons("cons", no, inds[k]) = consInd; 
                    // }
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        seefront.reorder(pol);
    // nee = nEE.getVal(); 
    // ope = oee + nee; 

    // e -> p 
    auto &sevfront = withBoundary ? boundarySevFront : selfSevFront;
    pol(Collapse{sevfront.size()},
        [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
            sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
            vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(sevfront),
            tempPE = proxy<space>(tempPE), nPE = proxy<space>(nPE), dHat2 = zs::sqr(radius), xi,
            radius, voffset = withBoundary ? coOffset : 0,
            frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
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
                    // auto consInd = no + ope; 
                    // for (int k = 0; k < 3; k++)
                    // {
                    //     auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                    //     // vCons("cons", no, inds[k]) = consInd; 
                    // }
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        sevfront.reorder(pol);
    // npe = nPE.getVal(); 
    // opp = npe + ope; 

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
            frontManageRequired = frontManageRequired, tag] __device__(int i) mutable {
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
                        // auto consInd = no + opp; 
                        // for (int k = 0; k < 2; k++)
                        // {
                        //     auto no = atomic_add(exec_cuda, &vCons("n", inds[k]), 1); 
                        //     // vCons("cons", no, inds[k]) = consInd; 
                        // }
                    }

                if (frontManageRequired)
                    bvh.iter_neighbors(bv, i, front, f);
                else
                    bvh.iter_neighbors(bv, front.node(i), f);
            });     
        if (frontManageRequired)
            svfront.reorder(pol);   
        // npp = nPP.getVal(); 
    }
}

void RapidClothSystem::findConstraints(zs::CudaExecutionPolicy &pol, T dist, const zs::SmallString &tag)
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

    updateConstraintCnt(); 
    D = D_max; 
    // TODO: coloring for multi-color PGS 
    consColoring(pol); 
}


static void constructVertexConsList(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::tiles_t& tempPair, 
    typename RapidClothSystem::itiles_t& vCons, 
    int pairNum, 
    int pairSize, 
    std::size_t offset)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    pol(range(pairNum), 
        [tempPair = proxy<space>({}, tempPair), 
         vCons = proxy<space>({}, vCons), 
         offset, pairSize] __device__ (int i) mutable {
            for (int k = 0; k < pairSize; k++)
            {
                auto vi = tempPair("inds", k, i, int_c); 
                auto n = atomic_add(exec_cuda, &vCons("n", vi), 1); 
                auto nE = vCons("nE", vi); 
                vCons("cons", n + nE, vi) = i + offset; 
                vCons("cons", n + nE, vi) = k; 
            }
        })
}

static void constructEEVertexConsList(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::tiles_t& tempE, 
    typename RapidClothSystem::itiles_t& vCons, 
    int pairNum)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    pol(range(pairNum), 
        [tempE = proxy<space>({}, tempE), 
         vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
            for (int k = 0; k < 2; k++)
            {
                auto vi = tempE("inds", k, i, int_c); 
                auto nE = atomic_add(exec_cuda, &vCons("nE", vi), 1); 
                vCons("cons", nE, vi) = i; 
                vCons("ind", nE, vi) = k; 
            }
        }); 
}

static void initPalettes(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::tiles_t &tempPair, 
    typename RapidClothSystem::itiles_t &vCons, 
    typename RapidClothSystem::tiles_t &tempCons, 
    int pairNum, 
    int pairSize, 
    std::size_t offset, 
    T shrinking)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    pol(range(pairNum), 
        [tempPair = proxy<space>({}, tempPair), 
         vCons = proxy<space>({}, vCons), 
         tempCons = proxy<space>({}, tempCons), 
         pairSize, offset] __device__ (int i) mutable {
            int degree = 0; 
            for (int k = 0; k < pairSize; k++)
            {
                auto vi = tempPair("inds", k, i, int_c); 
                auto nE = vCons("nE", vi); 
                auto n = vCons("n", vi); 
                degree += nE + n; 
                tempCons("vi", k, i + offset) = vi; 
            }
            int max_color = (int)zs::ceil(((T)degree) / shrinking); 
            if (max_color < 2)
                max_color = 2;
            tempCons("fixed", i + offset) = 0; 
            tempCons("max_color", i + offset) = max_color; 
            tempCons("num_color", i + offset) = max_color; 
            constexpr int len = sizeof(int) * 8; 
            tempCons("colors", i + offset) = (1 << (len - 1)) - 1; 
            tempCons("vi_len", i + offset) = pairSize; 
         }); 
}

static constexpr int simple_hash(int a)
{
    // https://burtleburtle.net/bob/hash/integer.html
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a; 
}

static bool checkConsColoring(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    zs::Vector<int> correct; 
    correct.setVal(1); 
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         vCons = proxy<space>({}, vCons), 
         correct = proxy<space>(correct)] __device__ (int i) mutable {
            int color = tempCons("color", i); 
            for (int k = 0; k < tempCons("vi_len", i); k++)
            {
                int vi = tempCons("vi", k, i); 
                int n = vCons("n", vi); 
                int nE = vCons("nE", vi); 
                for (int j = 0; j < n + nE; j++)
                {
                    int neighCons = vCons("cons", j);
                    if (tempCons("color", neighCons) == color)
                    {
                        correct[0] = 0; 
                        return; 
                    }
                }
            }
        }); 
    return correct.getVal() == 1; 
}

void RapidClothSystem::consColoring(zs::CudaExecutionPolicy &pol, T shrinking = 3.0f)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    // clear vertex -> cons list size 
    pol(range(vCons.size()), 
        [vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
            vCons("n", i) = 0; 
            vCons("nE", i) = 0; 
        }); 
    // construct vertex -> cons list 
    constructEEVertexConsList(pol, tempE, vCons, ne); 
    constructVertexConsList(pol, tempPP, vCons, npp, 2, opp); 
    constructVertexConsList(pol, tempPE, vCons, npe, 3, ope); 
    constructVertexConsList(pol, tempPT, vCons, npt, 4, opt); 
    constructVertexConsList(pol, tempEE, vCons, nee, 4, oee); 
    // construct cons adj list 
    initPalettes(pol, tempE, vCons, tempCons, ne, 2, 0, shrinking); 
    initPalettes(pol, tempPP, vCons, tempCons, npp, 2, opp, shrinking); 
    initPalettes(pol, tempPE, vCons, tempCons, npe, 3, ope, shrinking); 
    initPalettes(pol, tempPT, vCons, tempCons, npt, 4, opt, shrinking); 
    initPalettes(pol, tempEE, vCons, tempCons, nee, 4, oee, shrinking); 
    // cons graph coloring 
    zs::Vector<int> finished; 
    finished.setVal(1); 
    int seed = 0; 
    while (!finished.getVal())
    { 
        // pick random color for unfixed constraints
        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), seed = seed++] __device__ (int i) mutable {
                tempCons("tmp", i) = 0; 
                if (tempCons("fixed", i))
                    return; 
                int ind = simple_hash(simple_hash(seed) + simple_hash(i)) % temp("num_color", i);
                int colors = tempCons("colors", i); 
                int maxColor = tempCons("max_color", i); 
                int curInd = -1; 
                int pos = -1;  
                while(++pos < maxColor && colors)
                {
                    int digit = colors % 2; 
                    if (digit && (++curInd == ind))
                        break; 
                    colors >>= 1; 
                }
                if (curInd < ind)
                {
                    printf("err in coloring: palette exhausted in the random-picking phase!\n"); 
                    return; 
                }
                tempCons("color", i) = pos; 
            }); 

        // conflict resolution: fix the colors of 'good' constraints, remove them from their neighbors' palettes
        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), 
             vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
                if (tempCons("fixed", i))
                    return; 
                int color = tempCons("color", i); 
                bool flagConflict = false; 
                bool flagHigherInd = true; 
                for (int k = 0; k < tempCons("vi_len", i); k++)
                {
                    int vi = tempCons("vi", k, i); 
                    int n = vCons("n", vi); 
                    int nE = vCons("nE", vi); 
                    for (int j = 0; j < n + nE; j++)
                    {
                        int neighCons = vCons("cons", j);
                        if (neighCons > i)
                            flagHigherInd = false;  
                        int neighColor = tempCons("color", neighCons); 
                        if (neighColor == color)
                            flagConflict = true;
                        if (flagConflict && !flagHigherInd)
                            break; 
                    }
                    if (flagConflict && !flagHigherInd)
                        break; 
                }
                if (!flagConflict || flagHigherInd)
                {
                    tempCons("fixed", i) = 1; 
                    tempCons("tmp", i) = 1; // 1 means need to remove current color from neighbors' palettes
                }
             }); 

        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), 
             vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
                if (tempCons("fixed", i))
                    return; 
                int maxColor = tempCons("max_color", i); 
                for (int l = 0; k < tempCons("vi_len", i); k++)
                {
                    int vi = tempCons("vi", k, i); 
                    int n = vCons("n", vi); 
                    int nE = vCons("nE", vi); 
                    for (int j = 0; j < n + nE; j++)
                    {
                        int neighCons = vCons("cons", j); 
                        if (tempCons("tmp", neighCons))
                        {
                            int neighColor = tempCons("color", neighCons); 
                            if (neighColor >= maxColor)
                                continue; 
                            int colors = tempCons("colors", i); 
                            if ((colors >> neighColor) % 2)
                            {
                                tempCons("num_color", i) -= 1; 
                                tempCons("colors", i) -= (1 << neighColor); 
                            }
                        }
                    }
                }
             }); 

        // feed the hungry & check if finished 
        finished.setVal(1); 
        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), 
            finished = proxy<space>(finished)] __device__ (int i) mutable {
                if (tempCons("fixed", i))
                    return; 
                finished[0] = 1; 
                if (tempCons("num_color", i) == 0)
                    tempCons("max_color", i) += 1; 
            }); 
    }

    if (checkConsColoring(pol))
        printf("\t\t[graph coloring] The result is correct.\n");
    else 
        printf("\t\t[graph coloring] Wrong results!"); 
}



// xl, cons -> c(xl), J(xl)   
void RapidClothSystem::computeConstraints(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    pol(range(ne), [vtemp = proxy<space>({}, vtemp), 
                    tempE = proxy<space>({}, tempE), 
                    tempCons = proxy<space>({}, tempCons), 
                    oe, sigma] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + oe; 
        auto inds = tempE.pack(dim_c<2>, "inds", int_c); 
        auto xi = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto xj = vtemp.pack(dim_c<3>, tag, inds[1]);
        auto yi = vtemp.pack(dim_c<3>, "y[k+1]", inds[0]); 
        auto yj = vtemp.pack(dim_c<3>, "y[k+1]", inds[1]);
        auto xij_norm = (xi - xj).norm() + limits<T>::epsilon(); 
        auto yij_norm_inv = 1.0f / ((yi - yj).norm() + limits<T>::epsilon(); 
        auto grad = - (xi - xj) / xij_norm * yij_norm_inv; 
        auto val = sigma - xij_norm * yij_norm_inv; 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d + 3, consInd, T_c) = -grad(d); 
        tempCons("val", consInd, T_c) = val; 
    }); 

    pol(range(npp), [vtemp = proxy<space>({}, vtemp), 
                    tempPP = proxy<space>({}, tempPP), 
                    tempCons = proxy<space>({}, tempCons), 
                    opp, delta] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + opp; 
        auto inds = tempPP.pack(dim_c<2>, "inds", int_c); 
        auto xi = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto xj = vtemp.pack(dim_c<3>, tag, inds[1]);
        auto xij_norm = (xi - xj).norm() + limits<T>::epsilon(); 
        auto delta_inv = 1.0f / delta; 
        auto grad = (xi - xj) / xij_norm * delta_inv; 
        auto val = xij_norm * delta_inv - 1.0f; 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d + 3, consInd, T_c) = -grad(d); 
        tempCons("val", consInd) = val; 
    }); 

    pol(range(npe), [vtemp = proxy<space>({}, vtemp), 
                    tempPE = proxy<space>({}, tempPE), 
                    tempCons = proxy<space>({}, tempCons), 
                    ope, delta] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + ope; 
        auto inds = tempPE.pack(dim_c<3>, "inds", int_c); 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto e0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto e1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        zs::vec<9> grad; 
        PE_area2_grad(p.data(), e0.data(), e1.data(), grad.data()); 
        auto area = (e0 - p).cross(e1 - p).norm(); 
        T coef = (e1 - e0).norm() * delta; 
        grad /= (2.0f * area * coef + limits<T>::epsilon()); 
        tempCons.tuple(dim_c<9>, "grad", consInd, T_c) = grad; 
        tempCons("val", consInd, T_c) = area / coef - 1.0f; 
    }); 

    pol(range(npt), [vtemp = proxy<space>({}, vtemp), 
                    tempPT = proxy<space>({}, tempPT), 
                    tempCons = proxy<space>({}, tempCons), 
                    opt, delta] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + opt; 
        auto inds = tempPT.pack(dim_c<4>, "inds", int_c); 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto t0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto t1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto t2 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        zs::vec<3, 3> mat;
        for (int d = 0; d < 3; d++)
        {
            mat(d, 0) = t0(d) - p(d); 
            mat(d, 1) = t1(d) - p(d); 
            mat(d, 2) = t2(d) - p(d); 
        }
        auto vol = determinant(mat); 
        auto sgn = vol > 0 ? 1.0f : -1.0f; 
        auto coef = sgn * (t1 - t0).cross(t2 - t0).norm() * delta + limits<T>::epsilon(); 
        mat = adjoint(mat).transpose();

        zs::vec<3, 4> grad; 
        for (int d = 0; d < 3; d++)
            grad(d, 3) = 0; 
        for (k = 1; k < 4; k++)
            for (int d = 0; d < 3; d++)
            {
                grad(d, k) = mat(d, k); 
                grad(d, 0) -= mat(d, k); 
            }
        grad /= coef; 
        tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = grad; 
        tempCons("val", consInd, T_c) = vol / coef - 1.0f; 
    }); 

    pol(range(nee), [vtemp = proxy<space>({}, vtemp), 
                    tempEE = proxy<space>({}, tempEE), 
                    tempCons = proxy<space>({}, tempCons), 
                    opt, delta] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + opt; 
        auto inds = tempPT.pack(dim_c<4>, "inds", int_c); 
        auto ei0 = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto ei1 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto ej0 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto ej1 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        zs::vec<3, 3> mat, rMat;
        for (int d = 0; d < 3; d++)
        {
            mat(d, 0) = ei1(d) - ei0(d); 
            mat(d, 1) = ei1(d) - ei0(d); 
            mat(d, 2) = ej1(d) - ei0(d); 
        }
        auto vol = determinant(mat); 
        auto gammas = edge_edge_closest_point(ei0, ei1, ej0, ej1);
        auto pi =  gammas[0] * (ei1 - ei0) + ei0; 
        auto pj = gammas[1] * (ej1 - ej0) + ej0; 
        auto dij = pj - pi; 
        auto dist = dij.norm(); 
        auto ri0 = ei0 + (dist - delta) * 0.5f * dij; 
        auto ri1 = ei1 + (dist - delta) * 0.5f * dij; 
        auto rj0 = ej0 - (dist - delta) * 0.5f * dij;
        auto rj1 = ej1 - (dist - delta) * 0.5f * dij; 
        for (int d = 0; d < 3; d++)
        {
            rMat(d, 0) = ri1(d) - ri0(d); 
            rMat(d, 1) = ri1(d) - ri0(d); 
            rMat(d, 2) = rj1(d) - ri0(d); 
        }
        auto coef = determinant(rMat);  
        mat = adjoint(mat).transpose();

        zs::vec<3, 4> grad; 
        for (int d = 0; d < 3; d++)
            grad(d, 3) = 0; 
        for (k = 1; k < 4; k++)
            for (int d = 0; d < 3; d++)
            {
                grad(d, k) = mat(d, k); 
                grad(d, 0) -= mat(d, k); 
            }
        grad /= coef; 
        tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = grad; 
        tempCons("val", consInd, T_c) = vol / coef - 1.0f; 
    }); 
}

// yl, y[k], (c, J), xl -> lambda_{l+1}, y_{l+1} 
void RapidClothSystem::solveLCP(zs::CudaExecutionPolicy &pol)
{
    // PGS solver 
}      

// call cons + solveLCP 
void RapidClothSystem::backwardStep(zs::CudaExecutionPolicy &pol);    

// async stepping  
void RapidClothSystem::forwardStep(zs::CudaExecutionPolicy &pol);     