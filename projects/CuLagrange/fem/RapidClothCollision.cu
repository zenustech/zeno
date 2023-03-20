#include "RapidCloth.cuh"
#include "RapidClothUtils.hpp"
#include "Structures.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/Distance.hpp"
#include "RapidClothGradHess.inl"

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
         vCons = view<space>({}, vCons, false_c, "vCons"), 
         nPT = view<space>(nPT, false_c, "nPT"), radius, voffset = withBoundary ? coOffset : 0,
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

                if (pt_distance_type(p, t0, t1, t2) != 6)
                    return; 
                if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2) {
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
        });
    if (frontManageRequired)
        stfront.reorder(pol);   
    // e -> e
    const auto &sebvh = withBoundary ? bouSeBvh : seBvh;
    auto &seefront = withBoundary ? boundarySeeFront : selfSeeFront;
    pol(Collapse{seefront.size()},
        [seInds = proxy<space>({}, seInds), sedges = proxy<space>({}, withBoundary ? *coEdges : seInds),
            vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(sebvh), front = proxy<space>(seefront),
            vCons = proxy<space>({}, vCons), 
            tempEE = proxy<space>({}, tempEE), nEE = proxy<space>(nEE), dHat2 = zs::sqr(radius),
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

                if (ee_distance_type(v0, v1, v2, v3) != 8)
                    return; 
                if ((v0 - v1).cross(v2 - v3).l2NormSqr() < limits<T>::epsilon())
                    return; // PE
                if (auto d2 = safe_dist2_ee(v0, v1, v2, v3); d2 < dHat2) {
                    auto no = atomic_add(exec_cuda, &nEE[0], 1); 
                    auto inds = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                    tempEE.tuple(dim_c<4>, "inds", no, int_c) = inds; 
                    tempEE("dist", no) = (float)zs::sqrt(d2); 
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
    if constexpr (enablePE_c)
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
    if constexpr (enablePP_c)
    {
        if (!withBoundary)
        {
            const auto &svbvh = svBvh;
            auto &svfront = selfSvFront;
            pol(Collapse{svfront.size()},
                [spInds = proxy<space>({}, spInds), svOffset = svOffset, coOffset = coOffset, 
                bvh = proxy<space>(svbvh), front = proxy<space>(svfront), tempPP = proxy<space>({}, tempPP),
                eles = proxy<space>({}, svInds), 
                vCons = proxy<space>({}, vCons), 
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

    nPP.setVal(0);
    nPE.setVal(0);  
    nPT.setVal(0); 
    nEE.setVal(0); 
    
    // nE.setVal(0); TODO: put into findEdgeConstraints(bool init = false) and calls it in every iteration 

    // collect PP, PE, PT, EE, E constraints from bvh 
    if constexpr (enablePP_c)
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
    fmt::print("ne: {}, npp: {}, npe: {}, npt: {}, nee: {}, nCons: {}\n", 
        ne, npp, npe, npt, nee, nCons); 
    // TODO: coloring for multi-color PGS 
    fmt::print("consColoring started\n"); 
    consColoring(pol, consShrinking); 
}


static void constructVertexConsList(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::tiles_t& tempPair, 
    typename RapidClothSystem::itiles_t& vCons, 
    int pairNum, 
    int pairSize, 
    int offset, 
    int coOffset, 
    const zs::SmallString& tag = "tempPair")
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    pol(range(pairNum), 
        [tempPair = view<space>({}, tempPair, false_c, tag), 
         vCons = view<space>({}, vCons, false_c, "vCons"), 
         offset, pairSize, coOffset] __device__ (int i) mutable {
            for (int k = 0; k < pairSize; k++)
            {
                auto vi = tempPair("inds", k, i, int_c); 
                if (vi >= coOffset)
                    continue; 
                auto n = atomic_add(exec_cuda, &vCons("n", vi), 1); 
                auto nE = vCons("nE", vi); 
                vCons("cons", n + nE, vi) = i + offset; 
                vCons("ind", n + nE, vi) = k; 
            }
        }); 
}

static void constructEEVertexConsList(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::tiles_t& tempE, 
    typename RapidClothSystem::itiles_t& vCons, 
    int pairNum, 
    int coOffset)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    pol(range(pairNum), 
        [tempE = proxy<space>({}, tempE), 
         vCons = proxy<space>({}, vCons), coOffset] __device__ (int i) mutable {
            for (int k = 0; k < 2; k++)
            {
                auto vi = tempE("inds", k, i, int_c); 
                if (vi >= coOffset)
                    continue; 
                auto nE = atomic_add(exec_cuda, &vCons("nE", vi), 1); 
                vCons("cons", nE, vi) = i; 
                vCons("ind", nE, vi) = k; 
            }
        }); 
}

void RapidClothSystem::initPalettes(zs::CudaExecutionPolicy &pol, 
    typename RapidClothSystem::tiles_t &tempPair, 
    typename RapidClothSystem::itiles_t &vCons, 
    typename RapidClothSystem::itiles_t &tempCons, 
    int pairNum, 
    int pairSize, 
    int offset, 
    typename RapidClothSystem::T shrinking)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    pol(range(pairNum), 
        [tempPair = proxy<space>({}, tempPair), 
         vCons = proxy<space>({}, vCons), 
         tempCons = proxy<space>({}, tempCons), 
         tempColors = proxy<space>(tempColors), 
         lcpMatIs = view<space>(lcpMatIs, false_c, "lcpMatIs"), 
         lcpMatJs = view<space>(lcpMatJs, false_c, "lcpMatJs"), 
         lcpMatSize = view<space>(lcpMatSize, false_c, "lcpMatSize"), 
         pairSize, offset, shrinking, coOffset = coOffset] __device__ (int i) mutable {
            int degree = 0; 
            for (int k = 0; k < pairSize; k++)
            {
                auto vi = tempPair("inds", k, i, int_c); 
                tempCons("vi", k, i + offset) = vi; 
                if (vi >= coOffset)
                    continue; 
                auto nE = vCons("nE", vi); 
                auto n = vCons("n", vi); 
                degree += nE + n; 
                for (int j = 0; j < nE + n; j++)
                {
                    int aj = vCons("cons", j, vi); 
                    auto no = atomic_add(exec_cuda, &lcpMatSize[0], 1); 
                    lcpMatIs[no] = i + offset; 
                    lcpMatJs[no] = aj; 
                }
            }
            int max_color = (int)zs::ceil(((T)degree) / shrinking); 
            if (max_color < 10)
                max_color = 10;
            if (max_color > 63)
            {
                printf("init max_color %d exceeded 63 at i = %d, clamped to 63\n", max_color, i); 
                max_color = 63; 
            }
            tempCons("fixed", i + offset) = 0; 
            tempCons("max_color", i + offset) = max_color; 
            tempCons("num_color", i + offset) = max_color; 
            constexpr int len = sizeof(zs::i64) * 8; 
            // tempCons("colors", i + offset) = (1 << (len - 2)) - 1 + (1 << (len - 2)); 
            tempColors[i + offset] = (((zs::i64)1) << (len - 2)) - ((zs::i64)1) + (((zs::i64)1) << (len - 2)); 
            tempCons("vN", i + offset) = pairSize; 
            tempCons("dist", i + offset, T_c) = tempPair("dist", i); 
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

bool RapidClothSystem::checkConsColoring(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    zs::Vector<int> correct {vtemp.get_allocator(), 1}; 
    correct.setVal(1); 
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         vCons = proxy<space>({}, vCons), 
         lcpMat = proxy<space>(lcpMat), 
         correct = proxy<space>(correct)] __device__ (int i) mutable {
            int color = tempCons("color", i); 
            auto &ap = lcpMat._ptrs; 
            auto &aj = lcpMat._inds; 
            for (int k = ap[i]; k < ap[i + 1]; k++)
            {
                int j = aj[k]; 
                if (j == i)
                    continue; 
                if (tempCons("color", j) == color)
                {
                    correct[0] = 0;  
                    return; 
                }
            }
        }); 
    return correct.getVal() == 1; 
}

void RapidClothSystem::consColoring(zs::CudaExecutionPolicy &pol, T shrinking)
{
    zs::CppTimer timer; 
    timer.tick(); 
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    // clear vertex -> cons list size 
    pol(range(vCons.size()), 
        [vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
            vCons("n", i) = 0; 
            vCons("nE", i) = 0; 
        }); 
    // construct vertex -> cons list 
    constructEEVertexConsList(pol, tempE, vCons, ne, coOffset); 
    constructVertexConsList(pol, tempPP, vCons, npp, 2, opp, coOffset, "tempPP"); 
    constructVertexConsList(pol, tempPE, vCons, npe, 3, ope, coOffset, "tempPE"); 
    constructVertexConsList(pol, tempPT, vCons, npt, 4, opt, coOffset, "tempPT"); 
    constructVertexConsList(pol, tempEE, vCons, nee, 4, oee, coOffset, "tempEE"); 
    // construct cons adj list 
    lcpMatSize.setVal(0); 
    initPalettes(pol, tempE, vCons, tempCons, ne, 2, 0, shrinking); 
    initPalettes(pol, tempPP, vCons, tempCons, npp, 2, opp, shrinking); 
    initPalettes(pol, tempPE, vCons, tempCons, npe, 3, ope, shrinking); 
    initPalettes(pol, tempPT, vCons, tempCons, npt, 4, opt, shrinking); 
    initPalettes(pol, tempEE, vCons, tempCons, nee, 4, oee, shrinking); 

    auto lcpSize = lcpMatSize.getVal();
    lcpMatIs.resize(lcpSize); 
    lcpMatJs.resize(lcpSize); 
    lcpMat.build(pol, nCons, nCons, lcpMatIs, lcpMatJs, wrapv<false>{});
    lcpMat.localOrdering(pol, false_c);  
    lcpMat._vals.resize(lcpMat.nnz());
    lcpMatIs.resize(estNumCps); 
    lcpMatJs.resize(estNumCps); 
    
    // cons graph coloring 
    zs::Vector<int> finished {vtemp.get_allocator(), 1}; 
    finished.setVal(0); 
    int seed = 0; 
    while (!finished.getVal())
    { 
        // pick random color for unfixed constraints
        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), 
             tempColors = proxy<space>(tempColors), 
             seed = seed++] __device__ (int i) mutable {
                tempCons("tmp", i) = 0; 
                if (tempCons("fixed", i))
                    return; 
                int ind = simple_hash(simple_hash(seed) + simple_hash(i)) % tempCons("num_color", i);
                // int colors = tempCons("colors", i); 
                zs::i64 colors = tempColors[i]; 
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
                    colors = tempColors[i]; 
                    int colors_high = colors >> 32;
                    int colors_low = (int)(colors - (((zs::i64)colors_high) << 32)); 
                    // printf("[graph coloring] err in coloring: palette exhausted in the random-picking phase!, num_color: %d, colors: %d, max_color: %d\n", 
                    //     tempCons("num_color", i), tempCons("colors", i), tempCons("max_color", i)); 
                    printf("[graph coloring] err in coloring: palette exhausted in the random-picking phase!, num_color: %d, max_color: %d, colors-high: %d, colors-low: %d\n", 
                        tempCons("num_color", i), tempCons("max_color", i), colors_high, colors_low); 
                    return; 
                }
                tempCons("color", i) = pos; 
            }); 

        // conflict resolution: fix the colors of 'good' constraints, remove them from their neighbors' palettes
        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), 
             lcpMat = proxy<space>(lcpMat)] __device__ (int i) mutable {
                if (tempCons("fixed", i))
                    return; 
                int color = tempCons("color", i); 
                bool flagConflict = false; 
                bool flagHigherInd = true; 
                auto &ap = lcpMat._ptrs; 
                auto &aj = lcpMat._inds; 
                for (int k = ap[i]; k < ap[i + 1]; k++)
                {
                    int neCons = aj[k]; // neighbor constraint 
                    if (neCons == i)
                        continue; 
                    if (neCons > i)
                        flagHigherInd = false; 
                    int neColor = tempCons("color", neCons); 
                    if (neColor == color)
                        flagConflict = true; 
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
             tempColors = proxy<space>(tempColors), 
             lcpMat = proxy<space>(lcpMat), 
             vCons = proxy<space>({}, vCons)] __device__ (int i) mutable {
                if (tempCons("fixed", i))
                    return; 
                int maxColor = tempCons("max_color", i); 
                int numColor = tempCons("num_color", i); 
                // int colors = tempCons("colors", i); 
                zs::i64 colors = tempColors[i]; 
                auto &ap = lcpMat._ptrs; 
                auto &aj = lcpMat._inds; 
                for (int k = ap[i]; k < ap[i + 1]; k++)
                {
                    int neCons = aj[k]; // neighbor constraint 
                    if (neCons == i)
                        continue; 
                    if (tempCons("tmp", neCons))
                    {
                        int neColor = tempCons("color", neCons); 
                        if ((colors >> neColor) % 2)
                        {
                            if (neColor < maxColor)
                                numColor--; 
                            // colors -= (1 << neColor); 
                            colors -= (((zs::i64)1) << neColor); 
                        }
                    }
                }
                // tempCons("colors", i) = colors; 
                tempColors[i] = colors; 
                tempCons("num_color", i) = numColor; 
             }); 

        // feed the hungry & check if finished 
        finished.setVal(1); 
        pol(range(nCons), 
            [tempCons = proxy<space>({}, tempCons), 
            tempColors = proxy<space>(tempColors), 
            finished = proxy<space>(finished)] __device__ (int i) mutable {
                if (tempCons("fixed", i))
                    return; 
                finished[0] = 0; 
                while (tempCons("num_color", i) == 0)
                {
                    // if ((tempCons("colors", i) >> tempCons("max_color", i)) % 2)
                    if ((tempColors[i] >> tempCons("max_color", i)) % 2)
                        tempCons("num_color", i) += 1; 
                    tempCons("max_color", i) += 1; 
                    if (tempCons("max_color", i) == 64)
                    {
                        printf("max_color exceeded threshold 32!\n"); 
                        return; 
                    }
                }
            }); 
    }
    consColorBits.reset(0); 
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         consColorBits = proxy<space>(consColorBits)] __device__ (int i) mutable {
            consColorBits[tempCons("color", i)] = 1; 
         }); 
    nConsColor = 0; 
    for (int i = consColorBits.size() - 1; i >= 0; i--)
        if (consColorBits[i] == 1)
        {
            nConsColor = i; 
            break; 
        }
    timer.tock("constraint coloring"); 
    fmt::print("\t\t[graph coloring] Ended with {} colors\n", nConsColor + 1); 

    if (checkConsColoring(pol))
        fmt::print("\t\t[graph coloring] The result is correct.\n");
    else 
        fmt::print("\t\t[graph coloring] Wrong results!\n"); 
}


// xl, cons -> c(xl), J(xl)   
void RapidClothSystem::computeConstraints(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag)
{
    // TODO: use SparseMatrix to store J * M^{-1} * J.T
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    pol(range(ne), [vtemp = proxy<space>({}, vtemp), 
                    tempE = proxy<space>({}, tempE), 
                    tempCons = proxy<space>({}, tempCons), 
                    oe = oe, sigma = sigma, tag, delta = delta] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + oe; 
        auto inds = tempE.pack(dim_c<2>, "inds", i, int_c); 
        auto xi = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto xj = vtemp.pack(dim_c<3>, tag, inds[1]);
        auto yi = vtemp.pack(dim_c<3>, "y[k+1]", inds[0]); 
        auto yj = vtemp.pack(dim_c<3>, "y[k+1]", inds[1]);
        auto xij_norm = (xi - xj).norm() + limits<T>::epsilon(); 
        auto yij_norm_inv = 1.0f / ((yi - yj).norm() + limits<T>::epsilon()); 
        auto grad = - (xi - xj) / xij_norm * yij_norm_inv; 
        auto val = sigma - xij_norm * yij_norm_inv; 
        if (val >= 0)
        {
            tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = zs::vec<T, 12>::zeros(); 
            tempCons("val", consInd, T_c) = 0; 
            return; 
        } 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d + 3, consInd, T_c) = -grad(d); 
        tempCons("val", consInd, T_c) = val; 
    }); 

    pol(range(npp), [vtemp = proxy<space>({}, vtemp), 
                    tempPP = proxy<space>({}, tempPP), 
                    tempCons = proxy<space>({}, tempCons), 
                    opp = opp, delta = delta, tag] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + opp; 
        auto inds = tempPP.pack(dim_c<2>, "inds", i, int_c); 
        auto xi = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto xj = vtemp.pack(dim_c<3>, tag, inds[1]);
        auto xij_norm = (xi - xj).norm() + limits<T>::epsilon(); 
        auto delta_inv = 1.0f / delta; 
        auto grad = (xi - xj) / xij_norm * delta_inv; 
        auto val = xij_norm * delta_inv - 1.0f; 
        if (val >= 0)
        {
            tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = zs::vec<T, 12>::zeros(); 
            tempCons("val", consInd, T_c) = 0; 
            return; 
        } 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        for (int d = 0; d < 3; d++)
            tempCons("grad", d + 3, consInd, T_c) = -grad(d); 
        tempCons("val", consInd) = val; 
    }); 

    pol(range(npe), [vtemp = proxy<space>({}, vtemp), 
                    tempPE = proxy<space>({}, tempPE), 
                    tempCons = proxy<space>({}, tempCons), 
                    ope = ope, delta = delta, tag] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + ope; 
        auto inds = tempPE.pack(dim_c<3>, "inds", i, int_c); 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto e0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto e1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        zs::vec<T, 9> grad; 
        PE_area2_grad(p.data(), e0.data(), e1.data(), grad.data()); 
        auto area = (e0 - p).cross(e1 - p).norm(); 
        T coef = (e1 - e0).norm() * delta; 
        auto val = area / coef - 1.0f; 
        if (val >= 0)
        {
            tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = zs::vec<T, 12>::zeros(); 
            tempCons("val", consInd, T_c) = 0; 
            return; 
        } 
        grad /= (2.0f * area * coef + limits<T>::epsilon()); 
        // tempCons.tuple(dim_c<9>, "grad", consInd, T_c) = grad; 
        for (int d = 0; d < 9; d++)
            tempCons("grad", d, consInd, T_c) = grad(d); 
        tempCons("val", consInd, T_c) = val; 
    }); 

    pol(range(npt), [vtemp = proxy<space>({}, vtemp), 
                    tempPT = proxy<space>({}, tempPT), 
                    tempCons = proxy<space>({}, tempCons), 
                    opt = opt, delta = delta, tag] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + opt; 
        auto inds = tempPT.pack(dim_c<4>, "inds", i, int_c); 
        auto p = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto t0 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto t1 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto t2 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        zs::vec<T, 3, 3> mat;
        for (int d = 0; d < 3; d++)
        {
            mat(d, 0) = t0(d) - p(d); 
            mat(d, 1) = t1(d) - p(d); 
            mat(d, 2) = t2(d) - p(d); 
        }
        auto vol = determinant(mat); 
        auto sgn = vol > 0 ? 1.0f : -1.0f; 
        auto coef = sgn * (t1 - t0).cross(t2 - t0).norm() * delta; 
        if (zs::abs(coef) < limits<T>::epsilon())
            coef = sgn * limits<T>::epsilon(); 
        mat = adjoint(mat).transpose();
        auto val = vol / coef - 1.0f; 
        if (val >= 0)
        {
            tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = zs::vec<T, 12>::zeros(); 
            tempCons("val", consInd, T_c) = 0; 
            return; 
        } 
        zs::vec<T, 4, 3> grad; 
        for (int d = 0; d < 3; d++)
            grad(0, d) = 0; 
        for (int k = 1; k < 4; k++)
            for (int d = 0; d < 3; d++)
            {
                grad(k, d) = mat(d, k - 1); 
                grad(0, d) -= mat(d, k - 1); 
            }
        grad /= coef; 
        tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = grad; 
        tempCons("val", consInd, T_c) = val; 
    }); 

    pol(range(nee), [vtemp = proxy<space>({}, vtemp), 
                    tempEE = proxy<space>({}, tempEE), 
                    tempCons = proxy<space>({}, tempCons), 
                    opt = opt, delta = delta, tag] __device__ (int i) mutable {
        // calculate grad 
        int consInd = i + opt; 
        auto inds = tempEE.pack(dim_c<4>, "inds", i, int_c); 
        auto ei0 = vtemp.pack(dim_c<3>, tag, inds[0]); 
        auto ei1 = vtemp.pack(dim_c<3>, tag, inds[1]); 
        auto ej0 = vtemp.pack(dim_c<3>, tag, inds[2]); 
        auto ej1 = vtemp.pack(dim_c<3>, tag, inds[3]); 
        zs::vec<T, 3, 3> mat, rMat;
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
            rMat(d, 1) = rj0(d) - ri0(d); 
            rMat(d, 2) = rj1(d) - ri0(d); 
        }
        auto coef = determinant(rMat);
        // DEBUG OUTPUT
        {  
            if (zs::abs(coef) < limits<T>::epsilon())
            {
                printf("tiny coef in ee: coef = %f, ei = %d, inds: %d, %d, %d, %d, ei0: %f, %f, %f, ei1: %f, %f, %f, ej0: %f, %f, %f, ej1: %f, %f, %f\n", 
                    coef, i, inds[0], inds[1], inds[2], inds[3], 
                    ei0(0), ei0(1), ei0(2), ei1(0), ei1(1), ei1(2), 
                    ej0(0), ej0(1), ej0(2), ej1(0), ej1(1), ej1(2)); 
            }
        }
        mat = adjoint(mat).transpose();
        auto val = vol / coef - 1.0f; 
        if (val >= 0)
        {
            tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = zs::vec<T, 12>::zeros(); 
            tempCons("val", consInd, T_c) = 0; 
            return; 
        } 

        zs::vec<T, 4, 3> grad; 
        for (int d = 0; d < 3; d++)
            grad(0, d) = 0; 
        for (int k = 1; k < 4; k++)
            for (int d = 0; d < 3; d++)
            {
                grad(k, d) = mat(d, k - 1); 
                grad(0, d) -= mat(d, k - 1); 
            }
        grad /= coef; 
        tempCons.tuple(dim_c<12>, "grad", consInd, T_c) = grad; 
        tempCons("val", consInd, T_c) = val; 
    }); 

    // compute lcpMat = J * M^{-1} * J.T
    pol(range(lcpMat.nnz()), 
        [lcpMat = proxy<space>(lcpMat)] __device__ (int i) mutable {
            auto &ax = lcpMat._vals; 
            ax[i] = 0.f;
        });

    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
        vCons = proxy<space>({}, vCons), 
        vtemp = proxy<space>({}, vtemp), 
        lcpMat = proxy<space>(lcpMat), 
        coOffset = coOffset] __device__ (int i) mutable {
            auto &ax = lcpMat._vals; 
            auto vN = tempCons("vN", i); 
            for (int j = 0; j < vN; j++)                        // this V
            {
                int vi = tempCons("vi", j, i); 
                if (vi >= coOffset)
                    continue; 
                int n = vCons("n", vi) + vCons("nE", vi); 
                for (int k = 0; k < n; k++)
                {
                    int neCons = vCons("cons", k, vi); 
                    int neV = vCons("ind", k, vi); 
                    auto m = vtemp("ws", vi); // TODO: no ofb information when typing vtemp("ws", k, vi)? 
                    if (m <= 0.f)
                        m = limits<T>::epsilon(); 
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
}

// yl, y[k], (c, J), xl -> lambda_{l+1}, y_{l+1} 
void RapidClothSystem::solveLCP(zs::CudaExecutionPolicy &pol)
{
    // PGS solver 
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 

    // b = c(x(l)) - J(x(l)) * (y[k+1] - x(l))
    pol(range(nCons), 
        [tempCons = proxy<space>({}, tempCons), 
         vtemp = proxy<space>({}, vtemp), coOffset = coOffset] __device__ (int ci) mutable {   
            int val = tempCons("val", ci, T_c); 
            for (int i = 0; i < tempCons("vN", ci); i++)
            {
                int vi = tempCons("vi", i, ci); 
                if (vi >= coOffset)
                    continue; 
                for (int d = 0; d < 3; d++)
                    val -= tempCons("grad", i * 3 + d, ci, T_c) * 
                        (vtemp("y[k+1]", d, vi) - vtemp("x(l)", d, vi)); 
            }
            tempCons("b", ci, T_c) = val; 
            tempCons("lambda", ci, T_c) = 0.f; 
         }); 
    
    for (int iter = 0; iter < lcpCap; iter++)
    {
        lcpConverged.setVal(1); 
        for (int color = 0; color < nConsColor; color++)
        {
            pol(range(nCons), 
                [tempCons = proxy<space>({}, tempCons), 
                lcpMat = proxy<space>(lcpMat), 
                lcpConverged = proxy<space>(lcpConverged), 
                lcpTol = lcpTol, color] __device__ (int i) mutable {
                    if (tempCons("val", i, T_c) == 0)
                        return; 
                    if (tempCons("color", i) != color)
                        return; 
                    auto &ap = lcpMat._ptrs; 
                    auto &aj = lcpMat._inds; 
                    auto &ax = lcpMat._vals;
                    auto oldLam = tempCons("lambda", i, T_c); 
                    T maj = 0.f; 
                    T rhs = tempCons("b", i, T_c); 
                    for (int k = ap[i]; k < ap[i + 1]; k++)
                    {
                        auto j = aj[k]; 
                        if (j == i)
                        {
                            // DEBUG OUTPUT
                            // {
                            //     printf("A(%d, %d) += %f\n", 
                            //         i, j, (float)ax[k]); 
                            // }
                            maj += ax[k]; 
                            continue; 
                        }
                        rhs -= ax[k] * tempCons("lambda", j, T_c); 
                    } 
                    // check maj 
                    if (zs::abs(maj) < limits<T>::epsilon())
                    {
                        // DEBUG OUTPUT
                        // {
                        //     printf("\t\ttiny maj!: maj at ci = %d: %f\n", i, (float)maj); 
                        // }
                        return; 
                    }
                    auto newLam = rhs / maj; 
                    tempCons("lambda", i, T_c) = newLam > 0.f ? newLam: 0.f;
                    if (zs::abs(newLam - oldLam) > lcpTol)
                        lcpConverged[0] = 0; 
                }); 
        }
        if (lcpConverged.getVal())
            break;         
    }
}      

// call cons + solveLCP 
void RapidClothSystem::backwardStep(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    // dynamicsStep should be done previously 
    // x(l), y[k+1] -> LCP -> updated lambda -> updated y(l) 
    computeConstraints(pol, "x(l)"); 
    // TODO: project dof on boundaries? 
    solveLCP(pol); 
    // y(l+1) = M^{-1} * (J(l)).T * lambda + y[k+1]
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
            vtemp.tuple(dim_c<3>, "y(l)", vi) = vtemp.pack(dim_c<3>, "y[k+1]", vi); 
        }); 
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
                if (vi >= coOffset)
                    continue; 
                auto m = vtemp("ws", vi); 
                if (m <= 0.f)
                    m = limits<T>::epsilon(); 
                auto mInv = 1.0f / m; 
                for (int d = 0; d < 3; d++)
                {
                    atomic_add(exec_cuda, &vtemp("y(l)", d, vi), 
                        mInv * lambda * tempCons("grad", k * 3 + d, ci, T_c)); 
                }
            }
        }); 
}   

// async stepping  
void RapidClothSystem::forwardStep(zs::CudaExecutionPolicy &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    // updated y(l) -> updated x(l)
    // update Di: atomic_min? 
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp), maxDi = D_max] __device__ (int vi) mutable {
            vtemp("Di", vi) = maxDi; 
        }); 
    pol(range(nCons - ne), 
        [tempCons = proxy<space>({}, tempCons), 
         vtemp = proxy<space>({}, vtemp), ne = ne] __device__ (int i) mutable {
            int ci = i + ne; 
            auto dist = tempCons("dist", ci, T_c); 
            auto vN = tempCons("vN", ci); 
            for (int k = 0; k < vN; k++)
            {
                auto vi = tempCons("vi", k, ci); 
                atomic_min(exec_cuda, &vtemp("Di", vi), dist); 
            }
        }); 
    // calculate alpha, update x(l), r(l)
    pol(range(vtemp.size()), 
        [vtemp = proxy<space>({}, vtemp)] __device__ (int vi) mutable {
            auto y = vtemp.pack(dim_c<3>, "y(l)", vi); 
            auto x = vtemp.pack(dim_c<3>, "x(l)", vi); 
            auto alpha = 0.5f * vtemp("Di", vi) / 
                ((y - x).norm() + limits<T>::epsilon());
            if (alpha > 1.0f)
                alpha = 1.0f; 
            vtemp.tuple(dim_c<3>, "x(l)", vi) = vtemp.pack(dim_c<3>, "x(l)", vi) + 
                alpha * (y - x); 
            vtemp("r(l)", vi) *= 1.0f - alpha; 
            vtemp("disp", vi) = alpha * (y - x).norm(); 
        }); 
    // check infnorm of r(l) after calling forwardStep
}
}