#include "FastCloth.cuh"
#include "Structures.hpp"
#include "TopoUtils.hpp"
#include "zensim/geometry/Distance.hpp"
#include <fstream>
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

FastClothSystem::PrimitiveHandle::PrimitiveHandle(std::shared_ptr<tiles_t> elesPtr_, ZenoParticles::category_e category)
    : zsprimPtr{}, modelsPtr{}, vertsPtr{}, elesPtr{elesPtr_},
      etemp{elesPtr_->get_allocator(), {{"He", 6 * 6}}, elesPtr_->size()}, surfTrisPtr{}, surfEdgesPtr{},
      surfVertsPtr{}, svtemp{}, vOffset{0}, sfOffset{0}, seOffset{0}, svOffset{0}, category{category} {
    ;
}
FastClothSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset,
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
FastClothSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset,
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
FastClothSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset,
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
typename FastClothSystem::T FastClothSystem::PrimitiveHandle::maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol,
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
        edgeLengths[ei] = (verts.pack<3>("x0", inds[0]) - verts.pack<3>("xt", inds[1])).norm();
    });
    auto tmp = reduce(pol, edgeLengths, thrust::maximum<T>());
    return tmp;
}
typename FastClothSystem::T FastClothSystem::PrimitiveHandle::averageNodalMass(zs::CudaExecutionPolicy &pol) const {
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
typename FastClothSystem::T FastClothSystem::PrimitiveHandle::totalVolume(zs::CudaExecutionPolicy &pol) const {
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

/// FastClothSystem
typename FastClothSystem::T FastClothSystem::maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol, bool includeBoundary) {
    using T = typename FastClothSystem::T;
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
        auto &verts = vtemp;
        auto &edges = *coEdges;
        temp.resize(edges.size());
        auto &edgeLengths = temp;
        pol(Collapse{edges.size()},
            [edges = view<space>({}, edges), verts = view<space>({}, vtemp), edgeLengths = view<space>(edgeLengths),
             coOffset = coOffset] ZS_LAMBDA(int ei) mutable {
                auto inds = edges.pack(dim_c<2>, "inds", ei, int_c) + coOffset;
                edgeLengths[ei] = (verts.pack<3>("yn", inds[0]) - verts.pack<3>("yn", inds[1])).norm();
            });
        if (auto tmp = reduce(pol, edgeLengths, thrust::maximum<T>()); tmp > maxEdgeLength)
            maxEdgeLength = tmp;
    }
    return maxEdgeLength;
}
typename FastClothSystem::T FastClothSystem::averageNodalMass(zs::CudaExecutionPolicy &pol) {
    using T = typename FastClothSystem::T;
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
typename FastClothSystem::T FastClothSystem::totalVolume(zs::CudaExecutionPolicy &pol) {
    using T = typename FastClothSystem::T;
    T sumVolume = 0;
    for (auto &&primHandle : prims) {
        if (primHandle.isBoundary())
            continue;
        sumVolume += primHandle.totalVolume(pol);
    }
    return sumVolume;
}

void FastClothSystem::setupCollisionParams(zs::CudaExecutionPolicy &pol) {
#if 0 
    dHat = proximityRadius();
    // soft phase coeff
    // auto [mu_, lam_] = largestLameParams();
    mu = 0.2f;

    avgNodeMass = averageNodalMass(pol);
    // hard phase coeff
    rho = 0.1;
#else 
    // assume L_ref is get from input 
    B = LRef / 4.2f * 6.f;  
    L = B * 1.4142f; 
    Btight = B / 12.f; 
    LAda = B + Btight; 
    D = B / 6.f * 0.25f;  
    epsSlack = 30.f * B * B / 36.f; 
    rho = 0.1f; // move into input 
    epsCond = B * B / 36.f * 0.01f; 
    dHat = proximityRadius();
#endif 
    zeno::log_warn("automatically computed params: Btot[{}], L[{}]; D[{}], dHat[{}]; rho[{}], mu[{}]\n", B + Btight, L,
                   D, dHat, rho, mu);
}
void FastClothSystem::updateWholeBoundingBoxSize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    bv_t bv = svBvh.getTotalBox(pol);
    if (hasBoundary()) {
        auto bouBv = bouSvBvh.getTotalBox(pol);
        merge(bv, bouBv._min);
        merge(bv, bouBv._max);
    }
    boxDiagSize2 = (bv._max - bv._min).l2NormSqr();
}

void FastClothSystem::initialize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // exclSes = Vector<u8>{vtemp.get_allocator(), (std::size_t)seOffset};
    // exclSts = Vector<u8>{vtemp.get_allocator(), (std::size_t)sfOffset};
    // std::size_t nBouSes = 0, nBouSts = 0;
    // if (coPoints) {
    //     nBouSes = coEdges->size();
    //     nBouSts = coEles->size();
    // }
    // exclBouSes = Vector<u8>{vtemp.get_allocator(), nBouSes};
    // exclBouSts = Vector<u8>{vtemp.get_allocator(), nBouSts};

    /// @brief cloth system surface topo construction
    stInds = tiles_t{vtemp.get_allocator(), {{"inds", 3}}, (std::size_t)sfOffset};
    seInds = tiles_t{vtemp.get_allocator(), {{"inds", 2}, {"restLen", 1}}, (std::size_t)seOffset};
#if !s_debugRemoveHashTable
    eTab = etab_t{seInds.get_allocator(), (std::size_t)seInds.size() * 25};
#endif
    svInds = tiles_t{vtemp.get_allocator(), {{"inds", 1}}, (std::size_t)svOffset};
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        const auto &verts = primHandle.getVerts();
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
        const auto &edges = primHandle.getSurfEdges();
#if !s_debugRemoveHashTable
        Vector<int> tmpAdjVerts{vtemp.get_allocator(), estNumCps};
        auto adjVerts = tmpAdjVerts;
        Vector<int> adjVertsOff{vtemp.get_allocator(), vtemp.size()};
        Vector<int> adjVertsDeg{vtemp.get_allocator(), vtemp.size()};
        Vector<int> adjLen{vtemp.get_allocator(), 1};
        adjLen.setVal(0);
        adjVertsDeg.reset(0);
#endif
        constexpr int intHalfLen = sizeof(int) * 4;
        pol(Collapse(edges.size()), [seInds = view<space>({}, seInds), edges = view<space>({}, edges),
                                     voffset = primHandle.vOffset, seoffset = primHandle.seOffset,
#if !s_debugRemoveHashTable
                                     eTab = view<space>(eTab), tmpAdjVerts = view<space>(tmpAdjVerts),
                                     adjLen = view<space>(adjLen), adjVertsDeg = view<space>(adjVertsDeg),
#endif
                                     verts = view<space>({}, verts), intHalfLen] __device__(int i) mutable {
            auto inds = edges.pack(dim_c<2>, "inds", i, int_c);
            auto edge = inds + (int)voffset;
            seInds.tuple(dim_c<2>, "inds", seoffset + i, int_c) = edge;
            auto v0 = verts.pack(dim_c<3>, "x", inds[0]);
            auto v1 = verts.pack(dim_c<3>, "x", inds[1]);
#if s_useMassSpring
            seInds("restLen", seoffset + i) = (v0 - v1).norm();
            printf("restL: %f\n", (v0 - v1).norm());
#endif
#if !s_debugRemoveHashTable
            auto adjNo = atomic_add(exec_cuda, &adjLen[0], 1);
            tmpAdjVerts[adjNo] = (edge[0] << intHalfLen) + edge[1];
            atomic_add(exec_cuda, &adjVertsDeg[edge[0]], 1);
            adjNo = atomic_add(exec_cuda, &adjLen[0], 1);
            tmpAdjVerts[adjNo] = (edge[1] << intHalfLen) + edge[0];
            atomic_add(exec_cuda, &adjVertsDeg[edge[1]], 1);
            if (auto no = eTab.insert(edge); no < 0) {
                printf("the same directed edge <%d, %d> has been inserted twice!\n", edge[0], edge[1]);
            }
#endif
        });
#if !s_debugRemoveHashTable
        auto aN = adjLen.getVal();
        tmpAdjVerts.resize(aN);
        adjVerts.resize(aN);
        radix_sort(pol, tmpAdjVerts.begin(), tmpAdjVerts.end(), adjVerts.begin());
        exclusive_scan(pol, adjVertsDeg.begin(), adjVertsDeg.end(), adjVertsOff.begin());
        pol(range(vtemp.size()),
            [adjVerts = view<space>(adjVerts), adjVertsOff = view<space>(adjVertsOff), eTab = view<space>(eTab),
             vN = vtemp.size(), aN, intHalfLen] __device__(int vi) mutable {
                int idxSt = adjVertsOff[vi];
                int idxEnd = vi < vN - 1 ? adjVertsOff[vi + 1] : aN;
                for (int j = idxSt; j < idxEnd; j++) {
                    for (int k = j + 1; k < idxEnd; k++) {
                        int vj = adjVerts[j] - (vi << intHalfLen);
                        int vk = adjVerts[k] - (vi << intHalfLen);
                        auto edge = ivec2{vj, vk};
                        if (auto no = eTab.query(edge); no < 0)
                            eTab.insert(edge);
                    }
                }
            });
#endif
        const auto &points = primHandle.getSurfVerts();
        pol(Collapse(points.size()),
            [svInds = view<space>({}, svInds), points = view<space>({}, points), voffset = primHandle.vOffset,
             svoffset = primHandle.svOffset] __device__(int i) mutable {
                svInds("inds", svoffset + i, int_c) = points("inds", i, int_c) + (int)voffset;
            });
    }

    /// @brief initialize xt for moving boundaries
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto &verts = primHandle.getVerts();
        // initialize BC info
        // predict pos, initialize augmented lagrangian, constrain weights
        pol(Collapse(verts.size()), [vtemp = view<space>({}, vtemp), verts = view<space>({}, verts),
                                     voffset = primHandle.vOffset, dt = dt] __device__(int i) mutable {
            auto x = verts.pack<3>("x", i);
            vtemp.tuple<3>("xt", voffset + i) = x;
        });
    }
    reinitialize(pol, dt);

    /// @brief setup collision solver parameters
    setupCollisionParams(pol);
}

void FastClothSystem::reinitialize(zs::CudaExecutionPolicy &pol, T framedt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    dt = framedt;
    this->framedt = framedt;
    curRatio = 0;

    substep = -1;
    projectDBC = false;

    /// @note profile
    if constexpr (s_enableProfile)
        for (int i = 0; i != 10; ++i) {
            auxTime[i] = 0;
            dynamicsTime[i] = 0;
            collisionTime[i] = 0;
            auxCnt[i] = 0;
            dynamicsCnt[i] = 0;
            collisionCnt[i] = 0;
        }

    /// cloth dynamics status
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto &verts = primHandle.getVerts();
        primHandle.hasBC = verts.hasProperty("isBC") && verts.hasProperty("BCtarget"); 
        // initialize BC info
        // predict pos, initialize augmented lagrangian, constrain weights
        pol(Collapse(verts.size()), [vtemp = view<space>({}, vtemp), verts = view<space>({}, verts),
                                     voffset = primHandle.vOffset, dt = dt, 
                                     hasBC = primHandle.hasBC] __device__(int i) mutable {
            auto x = verts.pack<3>("x", i);
            auto v = verts.pack<3>("v", i);
            auto vi = voffset + i; 

            vtemp("ws", vi) = verts("m", i);
            vtemp.tuple<3>("yn", vi) = x;
            vtemp.tuple<3>("vn", vi) = v;

            if (hasBC)
            {
                vtemp.tuple(dim_c<3>, "BCtarget", vi) = verts.pack(dim_c<3>, "BCtarget", i); 
                vtemp("isBC", vi) = verts("isBC", vi); 
            } else {
                vtemp("isBC", vi) = 0; // otherwise the vertex would not be a BC vertex 
            }
        });
    }
    if (hasBoundary())
        if (auto coSize = coVerts->size(); coSize) {
            pol(Collapse(coSize),
                [vtemp = view<space>({}, vtemp), coverts = view<space>({}, *coVerts), coOffset = coOffset, dt = dt,
                 augLagCoeff = augLagCoeff, avgNodeMass = avgNodeMass] __device__(int i) mutable {
                    auto x = coverts.pack<3>("x", i);
                    auto v = coverts.pack<3>("v", i);
                    auto newX = x + v * dt;

                    vtemp("ws", coOffset + i) = avgNodeMass * augLagCoeff;
                    vtemp.tuple<3>("yn", coOffset + i) = x;
                    vtemp.tuple<3>("xt", coOffset + i) = x;
                    vtemp.tuple<3>("vn", coOffset + i) = v;
                });
        }

    {
        bvs.resize(svInds.size());
        retrieve_bounding_volumes(pol, vtemp, "yn", svInds, zs::wrapv<1>{}, 0, bvs);
        // auto ptBvs = retrieve_bounding_volumes(pol, vtemp, "xn", svInds, zs::wrapv<1>{}, 0);
        /// bvh
        if constexpr (s_enableProfile)
            timer.tick();

        svBvh.build(pol, bvs);
        if constexpr (s_enableProfile) {
            timer.tock();
            auxCnt[0]++;
            auxTime[0] += timer.elapsed();
        }
        if constexpr (s_testSh) {
            /// sh
            puts("begin self sh ctor");
            if constexpr (s_enableProfile)
                timer.tick();

            svSh.build(pol, L, bvs);

            if constexpr (s_enableProfile) {
                timer.tock();
                auxCnt[2]++;
                auxTime[2] += timer.elapsed();
            }
        }
    }
    if (hasBoundary()) {
        bvs.resize(coPoints->size());
        retrieve_bounding_volumes(pol, vtemp, "yn", *coPoints, zs::wrapv<1>{}, coOffset, bvs);
        // auto ptBvs = retrieve_bounding_volumes(pol, vtemp, "xn", *coPoints, zs::wrapv<1>{}, coOffset);
        /// bvh
        if constexpr (s_enableProfile)
            timer.tick();

        bouSvBvh.build(pol, bvs);

        if constexpr (s_enableProfile) {
            timer.tock();
            auxTime[0] += timer.elapsed();
        }

        if constexpr (s_testSh) {
            /// sh
            puts("begin boundary sh ctor");
            if constexpr (s_enableProfile)
                timer.tick();

            bouSvSh.build(pol, L, bvs);

            if constexpr (s_enableProfile) {
                timer.tock();
                auxTime[2] += timer.elapsed();
            }
        }
    }
    updateWholeBoundingBoxSize(pol);
    /// update grad pn residual tolerance
    targetGRes = pnRel * std::sqrt(boxDiagSize2);
    zeno::log_info("box diag size: {}, targetGRes: {}\n", std::sqrt(boxDiagSize2), targetGRes);
}

FastClothSystem::FastClothSystem(std::vector<ZenoParticles *> zsprims, tiles_t *coVerts, tiles_t *coPoints,
                                 tiles_t *coEdges, tiles_t *coEles, T dt, std::size_t estNumCps, bool withContact,
                                 T augLagCoeff, T pnRel, T cgRel, int PNCap, int CGCap, T dHat_, T gravity, int K,
                                 int IDyn, T BCStiffness, T mu, T LRef, T rho)
    : coVerts{coVerts}, coPoints{coPoints}, coEdges{coEdges}, coEles{coEles}, PP{estNumCps, zs::memsrc_e::um, 0},
      cPP{estNumCps * 20, zs::memsrc_e::um, 0}, nPP{zsprims[0]->getParticles().get_allocator(), 1},
      ncPP{zsprims[0]->getParticles().get_allocator(), 1},
      tempPP{{{"softG", 6}, {"hardG", 6}}, estNumCps, zs::memsrc_e::um, 0}, E{estNumCps, zs::memsrc_e::um, 0},
      nE{zsprims[0]->getParticles().get_allocator(), 1}, tempE{{{"softG", 6}, {"hardG", 6}},
                                                               estNumCps,
                                                               zs::memsrc_e::um,
                                                               0},
      //
      temp{estNumCps, zs::memsrc_e::um, 0},
      //
      dt{dt}, framedt{dt}, curRatio{0}, estNumCps{estNumCps}, enableContact{withContact}, augLagCoeff{augLagCoeff},
      pnRel{pnRel}, cgRel{cgRel}, PNCap{PNCap}, CGCap{CGCap}, dHat{dHat_}, extAccel{0, gravity, 0}, K{K}, IDyn{IDyn}, 
      BCStiffness{BCStiffness}, mu{mu}, LRef{LRef}, rho{rho} {
    coOffset = sfOffset = seOffset = svOffset = 0;
    for (auto primPtr : zsprims) {
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

    vtemp = tiles_t{zsprims[0]->getParticles().get_allocator(),
                    {
                        // boundary
                        {"ws", 1},
                        {"cons", 3},
                        // cloth dynamics
                        {"yn", 3},
                        {"xtmp", 3},
                        {"yn0", 3},
                        {"yk", 3},
                        {"ytmp", 3},
#if s_useChebyshevAcc
                        {"yn-1", 3},
                        {"yn-2", 3},
#endif
                        {"vn", 3},
                        {"ytilde", 3},
                        {"yhat", 3}, // initial pos at the current substep (constraint, extAccel)
                        {"isBC", 1}, // 0, 1
                        {"BCtarget", 3},  
                        // linear solver
                        {"dir", 3},
                        {"gridDir", 3},
                        {"grad", 3},
#if !s_useGDDiagHess
                        {"P", 9},
#else
                        {"P", 3},
#endif
                        {"r", 3},
                        {"p", 3},
                        {"q", 3},
                        // intermediate
                        {"temp", 3},
                        // collision dynamics
                        {"xn", 3},
                        {"xt", 3},   // for boundary objects
                        {"xn0", 3},  // for backtracking during hardphase
                        {"xk", 3},   // backup before collision step
                        {"xinit", 3} // initial trial collision-free step
                    },
                    (std::size_t)numDofs};
    bvs = zs::Vector<bv_t>{vtemp.get_allocator(), vtemp.size()}; // this size is the upper bound

    auto cudaPol = zs::cuda_exec();
    // average edge length (for CCD filtering)
    initialize(cudaPol); // update vtemp, bvh, boxsize, targetGRes
                         // adaptive dhat, targetGRes, kappa

    // check initial self intersections including proximity pairs, do once
    // markSelfIntersectionPrimitives(cudaPol);
}

void FastClothSystem::advanceSubstep(zs::CudaExecutionPolicy &pol, T ratio) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    // setup substep dt
    ++substep;
    dt = framedt * ratio;
    curRatio += ratio;

    projectDBC = false;
    pol(Collapse(coOffset), [vtemp = view<space>({}, vtemp), coOffset = coOffset, dt = dt] __device__(int vi) mutable {
        auto yn = vtemp.pack(dim_c<3>, "yn", vi);
        vtemp.tuple(dim_c<3>, "yhat", vi) = yn;
        auto newX = yn + vtemp.pack(dim_c<3>, "vn", vi) * dt;
        vtemp.tuple(dim_c<3>, "ytilde", vi) = newX;
    });
    if (hasBoundary())
        if (auto coSize = coVerts->size(); coSize)
            pol(Collapse(coSize), [vtemp = view<space>({}, vtemp), coverts = view<space>({}, *coVerts),
                                   coOffset = coOffset, dt = dt] __device__(int i) mutable {
                auto yn = vtemp.pack(dim_c<3>, "yn", coOffset + i);
                vtemp.tuple(dim_c<3>, "yhat", coOffset + i) = yn;
                auto newX = yn + coverts.pack(dim_c<3>, "v", i) * dt;
                vtemp.tuple(dim_c<3>, "ytilde", coOffset + i) = newX;
            });

    for (auto &primHandle : auxPrims) {
        /// @note hard binding
        if (primHandle.category == ZenoParticles::category_e::tracker) {
            const auto &eles = primHandle.getEles();
            pol(Collapse(eles.size()),
                [vtemp = view<space>({}, vtemp), eles = view<space>({}, eles)] __device__(int ei) mutable {
                    auto inds = eles.pack(dim_c<2>, "inds", ei, int_c);
                    // retrieve motion from the associated boundary vert
                    auto deltaX = vtemp.pack(dim_c<3>, "ytilde", inds[1]) - vtemp.pack(dim_c<3>, "yhat", inds[1]);
                    auto yn = vtemp.pack(dim_c<3>, "yn", inds[0]);
                    vtemp.tuple(dim_c<3>, "ytilde", inds[0]) = yn + deltaX;
                });
        }
    }
}

void FastClothSystem::updateVelocities(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(zs::range(coOffset), [vtemp = view<space>({}, vtemp), dt = dt] __device__(int vi) mutable {
        auto newX = vtemp.pack<3>("yn", vi);
        auto dv = (newX - vtemp.pack<3>("ytilde", vi)) / dt;
        auto vn = vtemp.pack<3>("vn", vi);
        vn += dv;
        vtemp.tuple<3>("vn", vi) = vn;
    });
}

void FastClothSystem::writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol) {
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
                verts.tuple<3>("x", vi) = vtemp.pack<3>("yn", vOffset + vi);
                if (!asBoundary)
                    verts.tuple<3>("v", vi) = vtemp.pack<3>("vn", vOffset + vi);
            });
    }
    if (hasBoundary())
        pol(Collapse(coVerts->size()),
            [vtemp = view<space>({}, vtemp), verts = view<space>({}, *const_cast<tiles_t *>(coVerts)),
             coOffset = coOffset] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>, "x", vi) = vtemp.pack(dim_c<3>, "yn", coOffset + vi);
                // no need to update v here. positions are moved accordingly
                // also, boundary velocies are set elsewhere
            });
}

void FastClothSystem::writeFile(std::string filename, std::string info) {
    std::ofstream outFile;
    outFile.open(filename, std::ios::app);
    outFile << info;
    outFile.close();
}

struct MakeClothSystem : INode {
    void apply() override {
        using namespace zs;
        auto zsprims = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        std::shared_ptr<ZenoParticles> zsboundary;
        if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
            zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");

        typename FastClothSystem::tiles_t *coVerts = zsboundary ? &zsboundary->getParticles() : nullptr;
        typename FastClothSystem::tiles_t *coPoints =
            zsboundary ? &(*zsboundary)[ZenoParticles::s_surfVertTag] : nullptr;
        typename FastClothSystem::tiles_t *coEdges =
            zsboundary ? &(*zsboundary)[ZenoParticles::s_surfEdgeTag] : nullptr;
        typename FastClothSystem::tiles_t *coEles = zsboundary ? &zsboundary->getQuadraturePoints() : nullptr;
#if 0
        const typename FastClothSystem::tiles_t *coSvs =
            zsboundary ? &(*zsboundary)[ZenoParticles::s_surfVertTag] : nullptr;
#endif

        if (zsboundary) {
            auto pol = cuda_exec();
            compute_surface_neighbors(pol, *coEles, *coEdges, *coPoints);
            coEles->append_channels(pol, {{"nrm", 3}});
            coEdges->append_channels(pol, {{"nrm", 3}});
        }

        /// solver parameters
        auto input_est_num_cps = get_input2<int>("est_num_cps");
        auto input_withContact = get_input2<bool>("with_contact");
        auto input_contactEE = get_input2<bool>("contact_with_ee");
        auto input_contactSelf = get_input2<bool>("contact_with_self");
        auto input_dHat = get_input2<float>("dHat");
        auto input_aug_coeff = get_input2<float>("aug_coeff");
        auto input_pn_rel = get_input2<float>("pn_rel");
        auto input_cg_rel = get_input2<float>("cg_rel");
        auto input_pn_cap = get_input2<int>("pn_iter_cap");
        auto input_cg_cap = get_input2<int>("cg_iter_cap");
        auto input_gravity = get_input2<float>("gravity");
        auto input_BC_stiffness = get_input2<float>("BC_stiffness"); 
        auto input_gd_step = get_input2<float>("gd_step"); 
        auto input_avg_edge_len = get_input2<float>("avg_edge_len"); 
        auto input_collision_weight = get_input2<float>("collision_weight"); 
        auto dt = get_input2<float>("dt");
        auto K = get_input2<int>("K");
        auto IDyn = get_input2<int>("IDyn");

        auto A = std::make_shared<FastClothSystem>(zsprims, coVerts, coPoints, coEdges, coEles, dt,
                                                   (std::size_t)(input_est_num_cps ? input_est_num_cps : 1000000),
                                                   input_withContact, input_aug_coeff, input_pn_rel, input_cg_rel,
                                                   input_pn_cap, input_cg_cap, input_dHat, input_gravity, K, IDyn, input_BC_stiffness, 
                                                   input_gd_step, input_avg_edge_len, input_collision_weight);
        A->enableContactSelf = input_contactSelf;

        set_output("ZSClothSystem", A);
    }
};

ZENDEFNODE(MakeClothSystem, {{"ZSParticles",
                              "ZSBoundaryPrimitives",
                              {gParamType_Int, "est_num_cps", "1000000"},
                              {gParamType_Bool, "with_contact", "1"},
                              {gParamType_Bool, "contact_with_ee", "1"},
                              {gParamType_Bool, "contact_with_self", "1"},
                              {gParamType_Float, "dt", "0.01"},
                              {gParamType_Float, "dHat", "0.001"},
                              {gParamType_Float, "aug_coeff", "1e2"},
                              {gParamType_Float, "pn_rel", "0.01"},
                              {gParamType_Float, "cg_rel", "0.001"},
                              {gParamType_Float, "gd_step", "0.2"},
                              {gParamType_Float, "collision_weight", "0.1"}, 
                              {gParamType_Float, "avg_edge_len", "4.2"},  
                              {gParamType_Int, "pn_iter_cap", "1000"},
                              {gParamType_Int, "cg_iter_cap", "1000"},
                              {gParamType_Float, "gravity", "-9.8"},
                              {gParamType_Int, "K", "72"},
                              {gParamType_Int, "IDyn", "2"}, 
                              {gParamType_Float, "BC_stiffness", "1000"}},
                             {"ZSClothSystem"},
                             {},
                             {"FEM"}});

} // namespace zeno