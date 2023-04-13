#include "RapidCloth.cuh"
#include "Structures.hpp"
#include "TopoUtils.hpp"
#include <zeno/types/ListObject.h>

#define RETRIEVE_OBJECT_PTRS(T, STR)                                                  \
    ([this](const std::string_view str) {                                             \
        std::vector<T *> objPtrs{};                                                   \
        if (has_input<T>(str.data()))                                                 \
            objPtrs.push_back(get_input<T>(str.data()).get());                        \
        else if (has_input<zeno::ListObject>(str.data())) {                           \
            auto &objSharedPtrLists = *get_input<zeno::ListObject>(str.data());       \
            for (auto &&objSharedPtr : objSharedPtrLists.get())                       \
                if (auto ptr = dynamic_cast<T *>(objSharedPtr.get()); ptr != nullptr) \
                    objPtrs.push_back(ptr);                                           \
        }                                                                             \
        return objPtrs;                                                               \
    })(STR);

namespace zeno {

RapidClothSystem::PrimitiveHandle::PrimitiveHandle(std::shared_ptr<tiles_t> elesPtr_, ZenoParticles::category_e category)
    : zsprimPtr{}, modelsPtr{}, vertsPtr{}, elesPtr{elesPtr_},
      etemp{elesPtr_->get_allocator(), {{"He", 6 * 6}}, elesPtr_->size()}, surfTrisPtr{}, surfEdgesPtr{},
      surfVertsPtr{}, svtemp{}, vOffset{0}, sfOffset{0}, seOffset{0}, svOffset{0}, category{category} {
    ;
}
RapidClothSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset,
                                                  Ti &svOffset, zs::wrapv<2>)
    : zsprimPtr{&zsprim, [](void *) {}}, modelsPtr{&zsprim.getModel(), [](void *) {}}, vertsPtr{&zsprim.getParticles(),
                                                                                                [](void *) {}},
      elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}}, etemp{zsprim.getQuadraturePoints().get_allocator(),
                                                                   {{"He", 6 * 6}},
                                                                   zsprim.numElements()},
      surfTrisPtr{&zsprim.getQuadraturePoints(), [](void *) {}},  // this is fake!
      surfEdgesPtr{&zsprim.getQuadraturePoints(), [](void *) {}}, // all elements are surface edges
      surfVertsPtr{&zsprim[ZenoParticles::s_surfVertTag], [](void *) {}}, vOffset{vOffset},
      svtemp{zsprim.getQuadraturePoints().get_allocator(),
             {{"H", 3 * 3}, {"fn", 1}},
             zsprim[ZenoParticles::s_surfVertTag].size()},
      sfOffset{sfOffset}, seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::curve)
        throw std::runtime_error("dimension of 2 but is not curve");
    vOffset += getVerts().size();
    // sfOffset += 0; // no surface triangles
    seOffset += getSurfEdges().size();
    svOffset += getSurfVerts().size();
}
RapidClothSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset,
                                                  Ti &svOffset, zs::wrapv<3>)
    : zsprimPtr{&zsprim, [](void *) {}}, modelsPtr{&zsprim.getModel(), [](void *) {}}, vertsPtr{&zsprim.getParticles(),
                                                                                                [](void *) {}},
      elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}}, etemp{zsprim.getQuadraturePoints().get_allocator(),
                                                                   {{"He", 9 * 9}},
                                                                   zsprim.numElements()},
      surfTrisPtr{&zsprim.getQuadraturePoints(), [](void *) {}}, surfEdgesPtr{&zsprim[ZenoParticles::s_surfEdgeTag],
                                                                              [](void *) {}},
      surfVertsPtr{&zsprim[ZenoParticles::s_surfVertTag], [](void *) {}}, vOffset{vOffset},
      svtemp{zsprim.getQuadraturePoints().get_allocator(),
             {{"H", 3 * 3}, {"fn", 1}},
             zsprim[ZenoParticles::s_surfVertTag].size()},
      sfOffset{sfOffset}, seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::surface)
        throw std::runtime_error("dimension of 3 but is not surface");
    vOffset += getVerts().size();
    sfOffset += getSurfTris().size();
    seOffset += getSurfEdges().size();
    svOffset += getSurfVerts().size();
}
RapidClothSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset,
                                                  Ti &svOffset, zs::wrapv<4>)
    : zsprimPtr{&zsprim, [](void *) {}}, modelsPtr{&zsprim.getModel(), [](void *) {}}, vertsPtr{&zsprim.getParticles(),
                                                                                                [](void *) {}},
      elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}}, etemp{zsprim.getQuadraturePoints().get_allocator(),
                                                                   {{"He", 12 * 12}},
                                                                   zsprim.numElements()},
      surfTrisPtr{&zsprim[ZenoParticles::s_surfTriTag], [](void *) {}},
      surfEdgesPtr{&zsprim[ZenoParticles::s_surfEdgeTag], [](void *) {}},
      surfVertsPtr{&zsprim[ZenoParticles::s_surfVertTag], [](void *) {}}, vOffset{vOffset},
      svtemp{zsprim.getQuadraturePoints().get_allocator(),
             {{"H", 3 * 3}, {"fn", 1}},
             zsprim[ZenoParticles::s_surfVertTag].size()},
      sfOffset{sfOffset}, seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::tet)
        throw std::runtime_error("dimension of 4 but is not tetrahedra");
    vOffset += getVerts().size();
    sfOffset += getSurfTris().size();
    seOffset += getSurfEdges().size();
    svOffset += getSurfVerts().size();
}
typename RapidClothSystem::T RapidClothSystem::PrimitiveHandle::maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol,
                                                                                    zs::Vector<T> &temp) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto &verts = getVerts();
    auto &edges = getSurfEdges();
    temp.resize(edges.size());
    auto &edgeLengths = temp;
    pol(Collapse{edges.size()}, [edges = view<space>({}, edges), verts = view<space>({}, verts),
                                 edgeLengths = view<space>(edgeLengths)] ZS_LAMBDA(int ei) mutable {
        auto inds = edges.pack(dim_c<2>, "inds", ei, int_c);
        edgeLengths[ei] = (verts.pack<3>("x0", inds[0]) - verts.pack<3>("x0", inds[1])).norm();
    });
    auto tmp = reduce(pol, edgeLengths, thrust::maximum<T>());
    return tmp;
}
typename RapidClothSystem::T RapidClothSystem::PrimitiveHandle::averageNodalMass(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprimPtr->hasMeta(s_meanMassTag))
        return zsprimPtr->readMeta(s_meanMassTag, zs::wrapt<T>{});
    auto &verts = getVerts();
    Vector<T> masses{verts.get_allocator(), verts.size()};
    pol(Collapse{verts.size()}, [verts = view<space>({}, verts), masses = view<space>(masses)] ZS_LAMBDA(
                                    int vi) mutable { masses[vi] = verts("m", vi); });
    auto tmp = reduce(pol, masses) / masses.size();
    zsprimPtr->setMeta(s_meanMassTag, tmp);
    return tmp;
}
typename RapidClothSystem::T RapidClothSystem::PrimitiveHandle::totalVolume(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprimPtr->hasMeta(s_totalVolumeTag))
        return zsprimPtr->readMeta(s_totalVolumeTag, zs::wrapt<T>{});
    auto &eles = getEles();
    Vector<T> vols{eles.get_allocator(), eles.size()};
    pol(Collapse{eles.size()}, [eles = view<space>({}, eles), vols = view<space>(vols)] ZS_LAMBDA(int ei) mutable {
        vols[ei] = eles("vol", ei);
    });
    auto tmp = reduce(pol, vols);
    zsprimPtr->setMeta(s_totalVolumeTag, tmp);
    return tmp;
}

/// RapidClothSystem
typename RapidClothSystem::T RapidClothSystem::maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol, bool includeBoundary) {
    using T = typename RapidClothSystem::T;
    T maxEdgeLength = 0;
    for (auto &&primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        if (auto tmp = primHandle.maximumSurfEdgeLength(pol, temp); tmp > maxEdgeLength)
            maxEdgeLength = tmp;
    }
    if (coVerts && includeBoundary) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // auto &verts = vtemp;
        auto &edges = *coEdges;
        temp.resize(edges.size());
        auto &edgeLengths = temp;
        pol(Collapse{edges.size()},
            [edges = view<space>({}, edges), verts = view<space>({}, vtemp), edgeLengths = view<space>(edgeLengths),
             coOffset = coOffset] ZS_LAMBDA(int ei) mutable {
                auto inds = edges.pack(dim_c<2>, "inds", ei, int_c) + coOffset;
                edgeLengths[ei] = (verts.pack<3>("x[k]", inds[0]) - verts.pack<3>("x[k]", inds[1])).norm();
            });
        if (auto tmp = reduce(pol, edgeLengths, thrust::maximum<T>()); tmp > maxEdgeLength)
            maxEdgeLength = tmp;
    }
    return maxEdgeLength;
}
typename RapidClothSystem::T RapidClothSystem::averageNodalMass(zs::CudaExecutionPolicy &pol) {
    using T = typename RapidClothSystem::T;
    T sumNodalMass = 0;
    int sumNodes = 0;
    for (auto &&primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        auto numNodes = primHandle.getVerts().size();
        sumNodes += numNodes;
        sumNodalMass += primHandle.averageNodalMass(pol) * numNodes;
    }
    if (sumNodes)
        return sumNodalMass / sumNodes;
    else
        return 0;
}
typename RapidClothSystem::T RapidClothSystem::totalVolume(zs::CudaExecutionPolicy &pol) {
    using T = typename RapidClothSystem::T;
    T sumVolume = 0;
    for (auto &&primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        sumVolume += primHandle.totalVolume(pol);
    }
    return sumVolume;
}

// TODO first
void RapidClothSystem::initialize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// @brief cloth system surface topo construction
    stInds = tiles_t{vtemp.get_allocator(), {{"inds", 3}}, (std::size_t)sfOffset};
    seInds = tiles_t{vtemp.get_allocator(), {{"inds", 2}}, (std::size_t)seOffset};
    svInds = tiles_t{vtemp.get_allocator(), {{"inds", 1}}, (std::size_t)svOffset};
    ne = seInds.size(); 
    tempE.resize(ne); 
    nE.setVal(ne); 

    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        const auto &verts = primHandle.getVerts();
        pol(Collapse(verts.size()), [vtemp = view<space>({}, vtemp), verts = view<space>({}, verts),
                                     voffset = primHandle.vOffset, dt = dt] __device__(int i) mutable {
            auto x0 = verts.pack(dim_c<3>, "x0", i);
            auto vi = voffset + i; 
            vtemp.tuple(dim_c<3>, "x0", vi) = x0;
        });
        // record surface (tri) indices
        if (primHandle.category != ZenoParticles::category_e::curve) {
            auto &tris = primHandle.getSurfTris();
            pol(Collapse(tris.size()),
                [stInds = view<space>({}, stInds), tris = view<space>({}, tris), voffset = primHandle.vOffset,
                 sfoffset = primHandle.sfOffset] __device__(int i) mutable {
                    stInds.tuple(dim_c<3>, "inds", sfoffset + i, int_c) =
                        (tris.pack(dim_c<3>, "inds", i, int_c) + (int)voffset);
                });
        }
        exclTab.reset(pol, true); 
        const auto &edges = primHandle.getSurfEdges();
        pol(Collapse(edges.size()), [seInds = view<space>({}, seInds), edges = view<space>({}, edges),
                                     voffset = primHandle.vOffset, seoffset = primHandle.seOffset,
                                     exclTab = proxy<space>(exclTab), 
                                     tempE = proxy<space>({}, tempE), 
                                     verts = view<space>({}, verts)] __device__(int i) mutable {
            auto inds = edges.pack(dim_c<2>, "inds", i, int_c);
            auto edge = inds + (int)voffset;
            seInds.tuple(dim_c<2>, "inds", seoffset + i, int_c) = edge;
            tempE.tuple(dim_c<2>, "inds", seoffset + i, int_c) = edge; 
            exclTab.insert(edge); 
            exclTab.insert({edge[1], edge[0]}); 
        });
        const auto &points = primHandle.getSurfVerts();
        pol(Collapse(points.size()),
            [svInds = view<space>({}, svInds), points = view<space>({}, points), voffset = primHandle.vOffset,
             svoffset = primHandle.svOffset] __device__(int i) mutable {
                svInds("inds", svoffset + i, int_c) = points("inds", i, int_c) + (int)voffset;
            });
    }
    spInds = svInds; 
    spInds.resize((std::size_t)(svInds.size() + coPoints->size())); 
    pol(range(coPoints->size()), 
        [spInds = proxy<space>({}, spInds), 
         svoffset = svInds.size(), 
         coOffset = coOffset] __device__ (int i) mutable {
            spInds("inds", i + svoffset, int_c) = coOffset + i; 
        }); 

    auto deduce_node_cnt = [](std::size_t numLeaves) {
        if (numLeaves <= 2)
            return numLeaves;
        return numLeaves * 2 - 1;
    };
    selfStFront = bvfront_t{(int)deduce_node_cnt(stInds.size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
    selfSeeFront = bvfront_t{(int)deduce_node_cnt(seInds.size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
    if (enablePE_c)
        selfSevFront = bvfront_t{(int)deduce_node_cnt(seInds.size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
    if (enablePP_c)
        selfSvFront = bvfront_t{(int)deduce_node_cnt(svInds.size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
    if (hasBoundary()) {
        boundaryStFront = bvfront_t{(int)deduce_node_cnt(coEles->size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
        boundarySeeFront = bvfront_t{(int)deduce_node_cnt(coEdges->size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
        if (enablePE_c)
            boundarySevFront = bvfront_t{(int)deduce_node_cnt(coEdges->size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
        // boundarySvFront = bvfront_t{(int)deduce_node_cnt(coPoints->size()), (int)bvhFrontCps, zs::memsrc_e::um, vtemp.devid()};
    }

    /// WARN: ignore BC verts initialization here
    D = 0;  
    reinitialize(pol, dt); 
}

void RapidClothSystem::reinitialize(zs::CudaExecutionPolicy &pol, T framedt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    dt = framedt;
    this->framedt = framedt;
    curRatio = 0;
    substep = -1;

    /// cloth dynamics status
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto &verts = primHandle.getVerts();
        pol(Collapse(verts.size()), [vtemp = view<space>({}, vtemp), verts = view<space>({}, verts),
                                     voffset = primHandle.vOffset, dt = dt] __device__(int i) mutable {
            auto x = verts.pack<3>("x", i);
            auto v = verts.pack<3>("v", i);
            auto vi = voffset + i; 

            vtemp("ws", vi) = verts("m", i);
            vtemp("BCfixed", vi) = (T)(verts("BCorder", i) == 3);
            vtemp.tuple(dim_c<3>, "x[0]", vi) = x;
            vtemp.tuple(dim_c<3>, "x[k]", vi) = x;
            vtemp.tuple(dim_c<3>, "x(l)", vi) = x;
            vtemp.tuple(dim_c<3>, "v[0]", vi) = v;
        });
    }
    if (hasBoundary())
        if (auto coSize = coVerts->size(); coSize) {
            pol(Collapse(coSize),
                [vtemp = view<space>({}, vtemp), coverts = view<space>({}, *coVerts), coOffset = coOffset, dt = dt,
                 augLagCoeff = augLagCoeff, avgNodeMass = avgNodeMass] __device__(int i) mutable {
                    auto x = coverts.pack<3>("x", i);
                    auto v = coverts.pack<3>("v", i);
                    int vi = coOffset + i; 

                    vtemp("ws", coOffset + i) = avgNodeMass * augLagCoeff;
                    vtemp.tuple(dim_c<3>, "x[0]", vi) = x;
                    vtemp.tuple(dim_c<3>, "x[k]", vi) = x;
                    vtemp.tuple(dim_c<3>, "x(l)", vi) = x; 
                    vtemp.tuple(dim_c<3>, "y[k+1]", vi) = x + v * dt; 
                    vtemp.tuple(dim_c<3>, "y(l)", vi) = x + v * dt; 
                    vtemp.tuple(dim_c<3>, "v[0]", vi) = v;
                });
        }

    frontManageRequired = true; 
#define init_front(sInds, front)                                                                           \
    {                                                                                                      \
        auto numNodes = front.numNodes();                                                                  \
        if (numNodes <= 2) {                                                                               \
            front.reserve(sInds.size() * numNodes);                                                        \
            front.setCounter(sInds.size() * numNodes);                                                     \
            pol(Collapse{sInds.size()}, [front = proxy<space>(front), numNodes] ZS_LAMBDA(int i) mutable { \
                for (int j = 0; j != numNodes; ++j)                                                        \
                    front.assign(i *numNodes + j, i, j);                                                   \
            });                                                                                            \
        } else {                                                                                           \
            front.reserve(sInds.size());                                                                   \
            front.setCounter(sInds.size());                                                                \
            pol(Collapse{sInds.size()},                                                                    \
                [front = proxy<space>(front)] ZS_LAMBDA(int i) mutable { front.assign(i, i, 0); });        \
        }                                                                                                  \
    }
    {
        if (enablePP_c)
        {
            bvs.resize(svInds.size()); 
            retrieve_bounding_volumes(pol, vtemp, "x[0]", svInds, zs::wrapv<1>{}, 0, bvs);
            svBvh.build(pol, bvs); 
            init_front(spInds, selfSvFront);             
        }

        bvs.resize(stInds.size());
        retrieve_bounding_volumes(pol, vtemp, "x[0]", stInds, zs::wrapv<3>{}, 0, bvs);
        stBvh.build(pol, bvs);
        init_front(spInds, selfStFront); 

        bvs.resize(seInds.size());
        retrieve_bounding_volumes(pol, vtemp, "x[0]", seInds, zs::wrapv<2>{}, 0, bvs);
        seBvh.build(pol, bvs);
        init_front(seInds, selfSeeFront);
        if (enablePE_c)
            init_front(spInds, selfSevFront); 
    }
    if (hasBoundary()) {
        bvs.resize(coEles->size());
        retrieve_bounding_volumes(pol, vtemp, "x[0]", *coEles, zs::wrapv<3>{}, coOffset, bvs);
        bouStBvh.build(pol, bvs);
        init_front(svInds, boundaryStFront);

        bvs.resize(coEdges->size());
        retrieve_bounding_volumes(pol, vtemp, "x[0]", *coEdges, zs::wrapv<2>{}, coOffset, bvs);
        bouSeBvh.build(pol, bvs);
        init_front(seInds, boundarySeeFront);
        if (enablePE_c)
            init_front(svInds, boundarySevFront);
    }
}

RapidClothSystem::RapidClothSystem(std::vector<ZenoParticles *> zsprims, tiles_t *coVerts, tiles_t *coPoints, tiles_t *coEdges,
                    tiles_t *coEles, T dt, std::size_t spmatCps, std::size_t ncps, std::size_t bvhFrontCps, bool withContact, 
                    T augLagCoeff, T cgRel, T lcpTol, int PNCap, int CGCap, int lcpCap, T gravity, int L, T delta, T sigma, bool enableSL, 
                    T gamma, T eps, int maxVertCons, T BCStiffness, bool enableExclEdges, T repulsionCoef, bool enableDegeneratedDist, 
                    bool enableDistConstraint, T repulsionRange, T tinyDist, bool enableFric, float clothFricMu, float boundaryFricMu)
    : coVerts{coVerts}, coPoints{coPoints}, coEdges{coEdges}, coEles{coEles}, spmatCps{spmatCps}, estNumCps{ncps}, bvhFrontCps{bvhFrontCps}, 
        nPP{zsprims[0]->getParticles().get_allocator(), 1}, nPE{zsprims[0]->getParticles().get_allocator(), 1},
        nPT{zsprims[0]->getParticles().get_allocator(), 1}, nEE{zsprims[0]->getParticles().get_allocator(), 1},
        nE{zsprims[0]->getParticles().get_allocator(), 1}, temp{estNumCps, zs::memsrc_e::um, 0}, itemp{estNumCps, zs::memsrc_e::um, 0},
        oPP{zsprims[0]->getParticles().get_allocator(), 1}, oPE{zsprims[0]->getParticles().get_allocator(), 1},
        oPT{zsprims[0]->getParticles().get_allocator(), 1}, oEE{zsprims[0]->getParticles().get_allocator(), 1},
        oE{zsprims[0]->getParticles().get_allocator(), 1}, 
        dt{dt}, framedt{dt}, curRatio{0}, enableContact{withContact}, augLagCoeff{augLagCoeff},
        cgRel{cgRel}, lcpTol{lcpTol}, PNCap{PNCap}, CGCap{CGCap}, lcpCap{lcpCap}, gravAccel{0, gravity, 0}, L{L}, delta{delta}, 
        D_min{delta}, D_max{delta * 4}, sigma{sigma}, enableSL{enableSL}, gamma{gamma}, eps{eps}, maxVertCons{maxVertCons}, 
        consDegree{maxVertCons * 4}, BCStiffness{BCStiffness}, enableExclEdges{enableExclEdges}, 
        repulsionCoef{repulsionCoef}, enableDegeneratedDist{enableDegeneratedDist}, enableDistConstraint{enableDistConstraint}, 
        enableRepulsion{repulsionCoef != 0.f}, repulsionRange{repulsionRange}, tinyDist{tinyDist}, enableFriction{enableFric}, 
        clothFricMu{clothFricMu}, boundaryFricMu{boundaryFricMu} { 
    auto cudaPol = zs::cuda_exec();
    coOffset = sfOffset = seOffset = svOffset = 0;
    for (auto primPtr : zsprims) {
        auto bv = primPtr->computeBoundingVolume(cudaPol, "x");
        primPtr->orderByMortonCode(cudaPol, bv);

        if (primPtr->category == ZenoParticles::category_e::curve) {
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<2>{});
        } else if (primPtr->category == ZenoParticles::category_e::surface)
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<3>{});
        else if (primPtr->category == ZenoParticles::category_e::tet)
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<4>{});
    }
    numDofs = coOffset;
    if (hasBoundary())
        numDofs += coVerts->size();
    numBouDofs = numDofs - coOffset;

    fmt::print("num total obj <verts, bouVerts, surfV, surfE, surfT>: {}, {}, {}, {}, {}\n", coOffset, numBouDofs,
               svOffset, seOffset, sfOffset);

    tempCons = itiles_t{
        zsprims[0]->getParticles().get_allocator(), 
        {
            // topology
            {"vi", 4}, 
            {"vN", 1}, 
            {"adj", maxVertCons * 4}, 
            // LCP
            {"type", 1}, 
            {"dist", 1}, 
            {"lambda", 1},  // float, lambda in LCP  
            {"grad", 12},   // float, constraint gradient 
            {"val", 1},     // float, constraint value
            {"b", 1},       // float, b in A*lambda+b, for LCP  
            {"diag", 1}     // float, diag element in the current row 
        }, 
        (std::size_t)estNumCps
    }; 
    tempPP = tiles_t{
        zsprims[0]->getParticles().get_allocator(), 
        {
            {"inds", 2}, 
            {"dist", 1}, 
            {"hess", 36}
        }, 
        (std::size_t)estNumCps
    }; 
    tempPE = tiles_t{
        zsprims[0]->getParticles().get_allocator(), 
        {
            {"inds", 3}, 
            {"dist", 1}, 
            {"hess", 81}
        }, 
        (std::size_t)estNumCps
    }; 
    tempPT = tiles_t{
        zsprims[0]->getParticles().get_allocator(), 
        {
            {"inds", 4}, 
            {"dist", 1}, 
            {"hess", 144}
        }, 
        (std::size_t)estNumCps
    }; 
    tempEE = tiles_t{
        zsprims[0]->getParticles().get_allocator(), 
        {
            {"inds", 4}, 
            {"dist", 1}, 
            {"hess", 144}
        }, 
        (std::size_t)estNumCps
    }; 
    tempE = tiles_t{
        zsprims[0]->getParticles().get_allocator(), 
        {
            {"inds", 2},  
            {"dist", 1}
        }, 
        (std::size_t)estNumCps
    }; 
    vCons = itiles_t{zsprims[0]->getParticles().get_allocator(), 
                    {
                        {"n", 1}, 
                        {"cons", maxVertCons}, 
                        {"ind", maxVertCons}    // its index in each constraint 
                    }, 
                    (std::size_t)numDofs    // stiff BC
    }; 
    opp = ope = opt = oee = oe = 0;             // offsets
    npp = npe = npt = nee = ne = 0;
    vtemp = tiles_t{zsprims[0]->getParticles().get_allocator(),
                    {
                        // boundary
                        {"ws", 1},
                        {"cons", 3},
                        {"isBC", 1},            // 0 or 1
                        {"BCtarget", 3},  
                        {"BCfixed", 1}, 
                        // cloth dynamics
                        {"x0", 3},              // rest state 
                        {"x[0]", 3},
                        {"x[k]", 3},  
                        {"y[k+1]", 3}, 
                        {"v[0]", 3}, 
                        {"x(l)", 3}, 
                        {"r(l)", 1}, 
                        {"y(l)", 3}, 
                        {"disp", 1}, 
                        {"x_tilde", 3},
                        {"x_hat", 3}, 
                        {"alpha", 1}, 
                        // linear solver
                        {"dir", 3},
                        {"grad", 3},
                        {"P", 9},               // implement Newton solver first 
                        {"r", 3},
                        {"p", 3},
                        {"q", 3},
                        // forward step
                        {"Di", 1}, 
                        {"sync", 1}, 
                        // intermediate
                        {"temp", 3},
                    },
                    (std::size_t)numDofs};
    bvs = zs::Vector<bv_t>{vtemp.get_allocator(), vtemp.size()}; // this size is the upper bound
    colorMinWeights = colorWeights = zs::Vector<zs::u32>{vtemp.get_allocator(), 0}; 
    colors = colorMaskOut = zs::Vector<int>{vtemp.get_allocator(), 0}; 
    exclTab = i2tab_t{vtemp.get_allocator(), estNumCps}; 
    lcpMat = spmat_t{zs::memsrc_e::device}; 
    lcpTopMat = ispmat_t{zs::memsrc_e::device}; 
    lcpMatIs = lcpMatJs = {vtemp.get_allocator(), spmatCps}; 
    lcpConverged = lcpMatSize = {vtemp.get_allocator(), 1}; 
    initialize(cudaPol); 

    // debug 
    visPrim = std::make_shared<PrimitiveObject>(); 
    visPrim->verts.resize(vtemp.size() * 2); 
    visPrim->lines.resize(vtemp.size()); 
}

void RapidClothSystem::advanceSubstep(zs::CudaExecutionPolicy &pol, T ratio) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    // setup substep dt
    ++substep;
    dt = framedt * ratio;
    curRatio += ratio;

    pol(Collapse(coOffset), [vtemp = view<space>({}, vtemp), coOffset = coOffset, dt = dt] __device__(int vi) mutable {
        auto xk = vtemp.pack(dim_c<3>, "x[k]", vi);
        vtemp.tuple(dim_c<3>, "x_hat", vi) = xk;
        auto newX = xk + vtemp.pack(dim_c<3>, "v[0]", vi) * dt;
        vtemp.tuple(dim_c<3>, "x_tilde", vi) = newX;
    });
    if (hasBoundary())
        if (auto coSize = coVerts->size(); coSize)
            pol(Collapse(coSize), [vtemp = view<space>({}, vtemp), coverts = view<space>({}, *coVerts),
                                   coOffset = coOffset, dt = dt] __device__(int i) mutable {
                auto xk = vtemp.pack(dim_c<3>, "x[k]", coOffset + i);
                vtemp.tuple(dim_c<3>, "x_hat", coOffset + i) = xk;
                auto newX = xk + coverts.pack(dim_c<3>, "v", i) * dt;
                vtemp.tuple(dim_c<3>, "x_tilde", coOffset + i) = newX;
            });
}

void RapidClothSystem::updateVelocities(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(zs::range(coOffset), [vtemp = view<space>({}, vtemp), dt = dt] __device__(int vi) mutable {
        auto newX = vtemp.pack<3>("x[k]", vi);
        auto dv = (newX - vtemp.pack<3>("x_tilde", vi)) / dt;
        auto vn = vtemp.pack<3>("v[0]", vi);
        vn += dv;
        vtemp.tuple<3>("v[0]", vi) = vn;
    });
}

void RapidClothSystem::writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        if (primHandle.isBoundary())
            continue;
        auto &verts = primHandle.getVerts();
        // update velocity and positions
        pol(zs::range(verts.size()),
            [vtemp = view<space>({}, vtemp), verts = view<space>({}, verts), dt = dt, vOffset = primHandle.vOffset,
             asBoundary = primHandle.isBoundary()] __device__(int vi) mutable {
                verts.tuple<3>("x", vi) = vtemp.pack<3>("x[k]", vOffset + vi);
                if (!asBoundary)
                    verts.tuple<3>("v", vi) = vtemp.pack<3>("v[0]", vOffset + vi);
            });
    }
    if (hasBoundary())
        pol(Collapse(coVerts->size()),
            [vtemp = view<space>({}, vtemp), verts = view<space>({}, *const_cast<tiles_t *>(coVerts)),
             coOffset = coOffset] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>, "x", vi) = vtemp.pack(dim_c<3>, "x[k]", coOffset + vi);
                // no need to update v here. positions are moved accordingly
                // also, boundary velocies are set elsewhere
            });
}

struct MakeRapidClothSystem : INode {
    using tiles_t = typename RapidClothSystem::tiles_t; 

    void apply() override {
        using namespace zs;
        auto zsprims = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        std::shared_ptr<ZenoParticles> zsboundary;
        if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
            zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");

        tiles_t *coVerts = zsboundary ? &zsboundary->getParticles() : nullptr;
        tiles_t *coPoints =
            zsboundary ? &(*zsboundary)[ZenoParticles::s_surfVertTag] : nullptr;
        tiles_t *coEdges =
            zsboundary ? &(*zsboundary)[ZenoParticles::s_surfEdgeTag] : nullptr;
        tiles_t *coEles = zsboundary ? &zsboundary->getQuadraturePoints() : nullptr;

        if (zsboundary) {
            auto pol = cuda_exec();
            compute_surface_neighbors(pol, *coEles, *coEdges, *coPoints);
            coEles->append_channels(pol, {{"nrm", 3}});
            coEdges->append_channels(pol, {{"nrm", 3}});
        }

        /// solver parameters
        auto input_spmat_cps = get_input2<int>("spmat_cps"); 
        auto input_est_num_cps = get_input2<int>("est_num_cps");
        auto input_bvh_front_cps = get_input2<int>("bvh_front_cps"); 
        auto input_withContact = get_input2<bool>("with_contact");
        auto input_contactEE = get_input2<bool>("contact_with_ee");
        auto input_contactSelf = get_input2<bool>("contact_with_self");
        auto input_aug_coeff = get_input2<float>("aug_coeff");
        auto input_cg_rel = get_input2<float>("cg_rel");
        auto input_pn_cap = get_input2<int>("pn_iter_cap");
        auto input_cg_cap = get_input2<int>("cg_iter_cap");
        auto input_gravity = get_input2<float>("gravity");
        auto input_BC_stiffness = get_input2<float>("BC_stiffness"); 
        auto input_dt = get_input2<float>("dt");
        auto input_L = get_input2<int>("collision_iters");
        auto input_delta = get_input2<float>("delta"); 
        auto input_sigma = get_input2<float>("edge_violation_ratio"); 
        auto input_enable_SL = get_input2<bool>("enable_SL"); 
        auto input_gamma = get_input2<float>("stepping_limit"); 
        auto input_eps = get_input2<float>("term_thresh");
        auto input_max_vert_cons = get_input2<int>("max_vert_cons");  
        auto input_lcp_tol = get_input2<int>("lcp_tol"); 
        auto input_lcp_cap = get_input2<int>("lcp_cap"); 
        auto input_enable_excl_edges = get_input2<bool>("enable_excl_edges"); 
        auto input_repulsion_coef = get_input2<float>("repulsion_coef");
        auto input_enable_degenerated_dist = get_input2<bool>("enable_degenerated_dist"); 
        auto input_enable_dist_constraint = get_input2<bool>("enable_dist_constraint"); 
        auto input_repulsion_range = get_input2<float>("repulsion_range"); 
        auto input_sync_dist_thresh = get_input2<float>("sync_dist_thresh"); 
        auto input_enable_friction = get_input2<bool>("enable_friction"); 
        auto input_cloth_fric_coef = get_input2<float>("cloth_fric_coef"); 
        auto input_boundary_fric_coef = get_input2<float>("boundary_fric_coef"); 

        // T delta, T sigma, T gamma, T eps
        auto A = std::make_shared<RapidClothSystem>(zsprims, coVerts, coPoints, coEdges, coEles, input_dt, 
                                                   (std::size_t)(input_spmat_cps ? input_spmat_cps : 1000000), 
                                                   (std::size_t)(input_est_num_cps ? input_est_num_cps : 100000),
                                                   (std::size_t)(input_bvh_front_cps ? input_bvh_front_cps : 10000000), 
                                                   input_withContact, input_aug_coeff, input_cg_rel, input_lcp_tol,  
                                                   input_pn_cap, input_cg_cap, input_lcp_cap, input_gravity, input_L, 
                                                   input_delta, input_sigma, input_enable_SL, input_gamma, input_eps, 
                                                   input_max_vert_cons, input_BC_stiffness, input_enable_excl_edges, 
                                                   input_repulsion_coef, input_enable_degenerated_dist, input_enable_dist_constraint, 
                                                   input_repulsion_range, input_sync_dist_thresh, input_enable_friction, 
                                                   input_cloth_fric_coef, input_boundary_fric_coef);
        A->enableContactSelf = input_contactSelf;

        set_output("ZSClothSystem", A);
    }
};

ZENDEFNODE(MakeRapidClothSystem, {{"ZSParticles",
                              "ZSBoundaryPrimitives",
                              {"int", "spmat_cps", "1000000"}, 
                              {"int", "est_num_cps", "100000"},
                              {"int", "bvh_front_cps", "10000000"}, 
                              {"int", "max_vert_cons", "32"}, 
                              {"bool", "with_contact", "1"},
                              {"bool", "contact_with_ee", "1"},
                              {"bool", "contact_with_self", "1"},
                              {"float", "dt", "0.01"},
                              {"float", "aug_coeff", "1e2"},
                              {"float", "cg_rel", "0.001"},
                              {"float", "lcp_tol", "0.001"}, 
                              {"int", "pn_iter_cap", "3"},
                              {"int", "cg_iter_cap", "200"},
                              {"int", "lcp_cap", "256"}, 
                              {"float", "gravity", "-9.8"},
                              {"int", "collision_iters", "512"}, 
                              {"float", "delta", "1"}, 
                              {"float", "edge_violation_ratio", "1.1"}, 
                              {"bool", "enable_SL", "1"}, 
                              {"float", "stepping_limit", "0.9"},  
                              {"float", "term_thresh", "1e-4"}, 
                              {"float", "BC_stiffness", "1000"}, 
                              {"bool", "enable_excl_edges", "0"}, 
                              {"float", "repulsion_coef", "0"}, 
                              {"float", "repulsion_range", "2"}, 
                              {"float", "sync_dist_thresh", "1e-2"}, 
                              {"bool", "enable_degenerated_dist" , "1"} , 
                              {"bool", "enable_dist_constraint", "1"}, 
                              {"bool", "enable_friction", "0"}, 
                              {"float", "cloth_fric_coef", "0.1"}, 
                              {"float", "boundary_fric_coef", "10.0"}},
                             {"ZSClothSystem"},
                             {},
                             {"FEM"}});

} // namespace zeno