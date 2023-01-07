#include "Solver.cuh"
#include "Utils.hpp"
#include <zeno/types/NumericObject.h>

namespace zeno {

IPCSystem::PrimitiveHandle::PrimitiveHandle(std::shared_ptr<tiles_t> elesPtr_, ZenoParticles::category_e category)
    : zsprimPtr{}, modelsPtr{}, vertsPtr{}, elesPtr{elesPtr_},
      etemp{elesPtr_->get_allocator(), {{"He", 6 * 6}}, elesPtr_->size()}, surfTrisPtr{}, surfEdgesPtr{},
      surfVertsPtr{}, svtemp{}, vOffset{0}, sfOffset{0}, seOffset{0}, svOffset{0}, category{category} {
    ;
}
IPCSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<2>)
    : zsprimPtr{&zsprim, [](void *) {}}, modelsPtr{&zsprim.getModel(), [](void *) {}},
      vertsPtr{&zsprim.getParticles<true>(), [](void *) {}}, elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}},
      etemp{zsprim.getQuadraturePoints().get_allocator(), {{"He", 6 * 6}}, zsprim.numElements()},
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
IPCSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<3>)
    : zsprimPtr{&zsprim, [](void *) {}}, modelsPtr{&zsprim.getModel(), [](void *) {}},
      vertsPtr{&zsprim.getParticles<true>(), [](void *) {}}, elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}},
      etemp{zsprim.getQuadraturePoints().get_allocator(), {{"He", 9 * 9}}, zsprim.numElements()},
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
    /// @note check if bending constraints exist
    if (zsprim.hasAuxData(ZenoParticles::s_bendingEdgeTag)) {
        bendingEdgesPtr =
            std::shared_ptr<ZenoParticles::particles_t>(&zsprim[ZenoParticles::s_bendingEdgeTag], [](void *) {});
        btemp = typename ZenoParticles::dtiles_t{etemp.get_allocator(), {{"Hb", 12 * 12}}, bendingEdgesPtr->size()};
    }
}
IPCSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<4>)
    : zsprimPtr{&zsprim, [](void *) {}}, modelsPtr{&zsprim.getModel(), [](void *) {}},
      vertsPtr{&zsprim.getParticles<true>(), [](void *) {}}, elesPtr{&zsprim.getQuadraturePoints(), [](void *) {}},
      etemp{zsprim.getQuadraturePoints().get_allocator(), {{"He", 12 * 12}}, zsprim.numElements()},
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
typename IPCSystem::T IPCSystem::PrimitiveHandle::averageNodalMass(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprimPtr->hasMeta(s_meanMassTag))
        return zsprimPtr->readMeta(s_meanMassTag, zs::wrapt<T>{});
    auto &verts = getVerts();
    Vector<T> masses{verts.get_allocator(), verts.size()};
    pol(Collapse{verts.size()}, [verts = proxy<space>({}, verts), masses = proxy<space>(masses)] ZS_LAMBDA(
                                    int vi) mutable { masses[vi] = verts("m", vi); });
    auto tmp = reduce(pol, masses) / masses.size();
    zsprimPtr->setMeta(s_meanMassTag, tmp);
    return tmp;
}
typename IPCSystem::T IPCSystem::PrimitiveHandle::averageSurfEdgeLength(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprimPtr->hasMeta(s_meanSurfEdgeLengthTag))
        return zsprimPtr->readMeta(s_meanSurfEdgeLengthTag, zs::wrapt<T>{});
    auto &verts = getVerts();
    auto &edges = getSurfEdges();
    Vector<T> edgeLengths{edges.get_allocator(), edges.size()};
    pol(Collapse{edges.size()}, [edges = proxy<space>({}, edges), verts = proxy<space>({}, verts),
                                 edgeLengths = proxy<space>(edgeLengths)] ZS_LAMBDA(int ei) mutable {
        auto inds = edges.pack(dim_c<2>, "inds", ei).template reinterpret_bits<int>();
        edgeLengths[ei] = (verts.pack<3>("x0", inds[0]) - verts.pack<3>("x0", inds[1])).norm();
    });
    auto tmp = reduce(pol, edgeLengths) / edges.size();
    zsprimPtr->setMeta(s_meanSurfEdgeLengthTag, tmp);
    return tmp;
}
typename IPCSystem::T IPCSystem::PrimitiveHandle::averageSurfArea(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprimPtr->category == ZenoParticles::curve)
        return (T)0;
    if (zsprimPtr->hasMeta(s_meanSurfAreaTag))
        return zsprimPtr->readMeta(s_meanSurfAreaTag, zs::wrapt<T>{});
    auto &verts = getVerts();
    auto &tris = getSurfTris();
    Vector<T> surfAreas{tris.get_allocator(), tris.size()};
    pol(Collapse{surfAreas.size()}, [tris = proxy<space>({}, tris), verts = proxy<space>({}, verts),
                                     surfAreas = proxy<space>(surfAreas)] ZS_LAMBDA(int ei) mutable {
        auto inds = tris.pack(dim_c<3>, "inds", ei).template reinterpret_bits<int>();
        surfAreas[ei] = (verts.pack<3>("x0", inds[1]) - verts.pack<3>("x0", inds[0]))
                            .cross(verts.pack<3>("x0", inds[2]) - verts.pack<3>("x0", inds[0]))
                            .norm() /
                        2;
    });
    auto tmp = reduce(pol, surfAreas) / tris.size();
    zsprimPtr->setMeta(s_meanSurfAreaTag, tmp);
    return tmp;
}

/// IPCSystem
typename IPCSystem::T IPCSystem::averageNodalMass(zs::CudaExecutionPolicy &pol) {
    using T = typename IPCSystem::T;
    T sumNodalMass = 0;
    std::size_t sumNodes = 0;
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
typename IPCSystem::T IPCSystem::averageSurfEdgeLength(zs::CudaExecutionPolicy &pol) {
    using T = typename IPCSystem::T;
    T sumSurfEdgeLengths = 0;
    std::size_t sumSE = 0;
    for (auto &&primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto numSE = primHandle.getSurfEdges().size();
        sumSE += numSE;
        sumSurfEdgeLengths += primHandle.averageSurfEdgeLength(pol) * numSE;
    }
    if (sumSE)
        return sumSurfEdgeLengths / sumSE;
    else
        return 0;
}
typename IPCSystem::T IPCSystem::averageSurfArea(zs::CudaExecutionPolicy &pol) {
    using T = typename IPCSystem::T;
    T sumSurfArea = 0;
    std::size_t sumSF = 0;
    for (auto &&primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        if (primHandle.category == ZenoParticles::curve)
            continue;
        auto numSF = primHandle.getSurfTris().size();
        sumSF += numSF;
        sumSurfArea += primHandle.averageSurfArea(pol) * numSF;
    }
    if (sumSF)
        return sumSurfArea / sumSF;
    else
        return 0;
}
void IPCSystem::updateWholeBoundingBoxSize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    bv_t bv = seBvh.getTotalBox(pol);
    if (coVerts)
        if (coVerts->size()) {
            auto bouBv = bouSeBvh.getTotalBox(pol);
            merge(bv, bouBv._min);
            merge(bv, bouBv._max);
        }
    boxDiagSize2 = (bv._max - bv._min).l2NormSqr();
}
void IPCSystem::initKappa(zs::CudaExecutionPolicy &pol) {
    // should be called after dHat set
    if (!enableContact)
        return;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
        vtemp.tuple<3>("p", i) = vec3::zeros();
        vtemp.tuple<3>("q", i) = vec3::zeros();
    });
    // inertial + elasticity
    computeInertialPotentialGradient(pol, "p");
    computeElasticGradientAndHessian(pol, "p", false);
    // contacts
    findCollisionConstraints(pol, dHat, xi);
    auto prevKappa = kappa;
    kappa = 1;
    computeBarrierGradientAndHessian(pol, "q", false);
    // computeBoundaryBarrierGradientAndHessian(pol, "q", false);
    kappa = prevKappa;
    auto gsum = dot(pol, vtemp, "p", "q");
    auto gsnorm = dot(pol, vtemp, "q", "q");
    if (gsnorm < limits<T>::epsilon() * 10)
        kappaMin = 0;
    else
        kappaMin = -gsum / gsnorm;
    // zeno::log_info("kappaMin: {}, gsum: {}, gsnorm: {}\n", kappaMin, gsum, gsnorm);
}

void IPCSystem::initialize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    stInds = tiles_t{vtemp.get_allocator(), {{"inds", 3}}, sfOffset};
    seInds = tiles_t{vtemp.get_allocator(), {{"inds", 2}}, seOffset};
    svInds = tiles_t{vtemp.get_allocator(), {{"inds", 1}}, svOffset};
    exclSes = Vector<u8>{vtemp.get_allocator(), seOffset};
    exclSts = Vector<u8>{vtemp.get_allocator(), sfOffset};
    std::size_t nBouSes = 0, nBouSts = 0;
    if (coEdges) {
        nBouSes = coEdges->size();
        nBouSts = coEles->size();
    }
    exclBouSes = Vector<u8>{vtemp.get_allocator(), nBouSes};
    exclBouSts = Vector<u8>{vtemp.get_allocator(), nBouSts};

    auto deduce_node_cnt = [](std::size_t numLeaves) {
        if (numLeaves <= 2)
            return numLeaves;
        return numLeaves * 2 - 1;
    };
    selfStFront = bvfront_t{(int)deduce_node_cnt(stInds.size()), (int)estNumCps, zs::memsrc_e::um, vtemp.devid()};
    selfSeFront = bvfront_t{(int)deduce_node_cnt(seInds.size()), (int)estNumCps, zs::memsrc_e::um, vtemp.devid()};
    if (coVerts) {
        boundaryStFront =
            bvfront_t{(int)deduce_node_cnt(coEles->size()), (int)estNumCps, zs::memsrc_e::um, vtemp.devid()};
        boundarySeFront =
            bvfront_t{(int)deduce_node_cnt(coEdges->size()), (int)estNumCps, zs::memsrc_e::um, vtemp.devid()};
    }

    meanEdgeLength = averageSurfEdgeLength(pol);
    meanSurfaceArea = averageSurfArea(pol);
    avgNodeMass = averageNodalMass(pol);
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto &verts = primHandle.getVerts();
        // record surface (tri) indices
        if (primHandle.category != ZenoParticles::category_e::curve) {
            auto &tris = primHandle.getSurfTris();
            pol(Collapse(tris.size()),
                [stInds = proxy<space>({}, stInds), tris = proxy<space>({}, tris), voffset = primHandle.vOffset,
                 sfoffset = primHandle.sfOffset] __device__(int i) mutable {
                    stInds.template tuple<3>("inds", sfoffset + i) =
                        (tris.pack(dim_c<3>, "inds", i).template reinterpret_bits<int>() + (int)voffset)
                            .template reinterpret_bits<float>();
                });
        }
        auto &edges = primHandle.getSurfEdges();
        pol(Collapse(edges.size()),
            [seInds = proxy<space>({}, seInds), edges = proxy<space>({}, edges), voffset = primHandle.vOffset,
             seoffset = primHandle.seOffset] __device__(int i) mutable {
                seInds.template tuple<2>("inds", seoffset + i) =
                    (edges.pack(dim_c<2>, "inds", i).template reinterpret_bits<int>() + (int)voffset)
                        .template reinterpret_bits<float>();
            });
        auto &points = primHandle.getSurfVerts();
        pol(Collapse(points.size()),
            [svInds = proxy<space>({}, svInds), points = proxy<space>({}, points), voffset = primHandle.vOffset,
             svoffset = primHandle.svOffset] __device__(int i) mutable {
                svInds("inds", svoffset + i) =
                    reinterpret_bits<float>(reinterpret_bits<int>(points("inds", i)) + (int)voffset);
            });
    }
    // initialize vtemp & spatial accel
    reinitialize(pol, dt);
}

IPCSystem::IPCSystem(std::vector<ZenoParticles *> zsprims, const typename IPCSystem::dtiles_t *coVerts,
                     const typename IPCSystem::tiles_t *coLowResVerts, const typename IPCSystem::tiles_t *coEdges,
                     const tiles_t *coEles, T dt, std::size_t estNumCps, bool withGround, bool withContact,
                     bool withMollification, T augLagCoeff, T pnRel, T cgRel, int PNCap, int CGCap, int CCDCap,
                     T kappa0, T fricMu, T dHat_, T epsv_, zeno::vec3f gn, T gravity)
    : coVerts{coVerts}, coLowResVerts{coLowResVerts}, coEdges{coEdges}, coEles{coEles},
      PP{estNumCps, zs::memsrc_e::um, 0}, nPP{zsprims[0]->getParticles<true>().get_allocator(), 1},
      tempPP{{{"H", 36}}, estNumCps, zs::memsrc_e::um, 0}, PE{estNumCps, zs::memsrc_e::um, 0},
      nPE{zsprims[0]->getParticles<true>().get_allocator(), 1}, tempPE{{{"H", 81}}, estNumCps, zs::memsrc_e::um, 0},
      PT{estNumCps, zs::memsrc_e::um, 0}, nPT{zsprims[0]->getParticles<true>().get_allocator(), 1},
      tempPT{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0}, EE{estNumCps, zs::memsrc_e::um, 0},
      nEE{zsprims[0]->getParticles<true>().get_allocator(), 1}, tempEE{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0},
      // mollify
      PPM{estNumCps, zs::memsrc_e::um, 0}, nPPM{zsprims[0]->getParticles<true>().get_allocator(), 1},
      tempPPM{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0}, PEM{estNumCps, zs::memsrc_e::um, 0},
      nPEM{zsprims[0]->getParticles<true>().get_allocator(), 1}, tempPEM{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0},
      EEM{estNumCps, zs::memsrc_e::um, 0}, nEEM{zsprims[0]->getParticles<true>().get_allocator(), 1},
      tempEEM{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0},
      // friction
      FPP{estNumCps, zs::memsrc_e::um, 0}, nFPP{zsprims[0]->getParticles<true>().get_allocator(), 1},
      fricPP{{{"H", 36}, {"basis", 6}, {"fn", 1}}, estNumCps, zs::memsrc_e::um, 0}, FPE{estNumCps, zs::memsrc_e::um, 0},
      nFPE{zsprims[0]->getParticles<true>().get_allocator(), 1},
      fricPE{{{"H", 81}, {"basis", 6}, {"fn", 1}, {"yita", 1}}, estNumCps, zs::memsrc_e::um, 0},
      FPT{estNumCps, zs::memsrc_e::um, 0}, nFPT{zsprims[0]->getParticles<true>().get_allocator(), 1},
      fricPT{{{"H", 144}, {"basis", 6}, {"fn", 1}, {"beta", 2}}, estNumCps, zs::memsrc_e::um, 0},
      FEE{estNumCps, zs::memsrc_e::um, 0}, nFEE{zsprims[0]->getParticles<true>().get_allocator(), 1},
      fricEE{{{"H", 144}, {"basis", 6}, {"fn", 1}, {"gamma", 2}}, estNumCps, zs::memsrc_e::um, 0},
      //
      temp{estNumCps, zs::memsrc_e::um, zsprims[0]->getParticles<true>().devid()}, csPT{estNumCps, zs::memsrc_e::um, 0},
      csEE{estNumCps, zs::memsrc_e::um, 0}, ncsPT{zsprims[0]->getParticles<true>().get_allocator(), 1},
      ncsEE{zsprims[0]->getParticles<true>().get_allocator(), 1},
      //
      dt{dt}, framedt{dt}, curRatio{0}, estNumCps{estNumCps}, enableGround{withGround}, enableContact{withContact},
      enableMollification{withMollification}, s_groundNormal{gn[0], gn[1], gn[2]},
      augLagCoeff{augLagCoeff}, pnRel{pnRel}, cgRel{cgRel}, PNCap{PNCap}, CGCap{CGCap}, CCDCap{CCDCap}, kappa{kappa0},
      kappa0{kappa0}, kappaMin{0}, kappaMax{kappa0}, fricMu{fricMu}, dHat{dHat_}, epsv{epsv_}, extForce{0, gravity, 0} {
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
    if (coVerts)
        numDofs += coVerts->size();
    numBouDofs = numDofs - coOffset;

    fmt::print("num total obj <verts, bouVerts, surfV, surfE, surfT>: {}, {}, {}, {}, {}\n", coOffset, numBouDofs,
               svOffset, seOffset, sfOffset);

    vtemp = dtiles_t{zsprims[0]->getParticles<true>().get_allocator(),
                     {{"grad", 3},
                      {"P", 9},
                      // dirichlet boundary condition type; 0: NOT, 1: ZERO, 2: NONZERO
                      {"BCorder", 1}, // 0: unbounded; 1: soft; 2: hard
                      {"BCbasis", 9},
                      {"BCtarget", 3},
                      {"BCfixed", 1},
                      {"BCsoft", 1}, // mark if this dof is a soft boundary vert or not
                      {"ws", 1},     // also as constraint jacobian
                      {"cons", 3},
                      {"lambda", 3},

                      {"dir", 3},
                      {"xn", 3},
                      {"vn", 3},
                      {"x0", 3},  // initial positions
                      {"xn0", 3}, // for line search
                      {"xtilde", 3},
                      {"xhat", 3}, // initial positions at the current substep (constraint,
                                   // extforce)
                      {"temp", 3},
                      {"r", 3},
                      {"p", 3},
                      {"q", 3}},
                     numDofs};
    bvs = zs::Vector<bv_t>{vtemp.get_allocator(), vtemp.size()}; // this size is the upper bound

    // inertial hessian
    tempI = dtiles_t{vtemp.get_allocator(), {{"Hi", 9}}, coOffset};

    // connect vtemp with "dir", "grad"
    cgtemp = tiles_t{vtemp.get_allocator(),
                     {{"P", 9},

                      {"dir", 3},

                      {"temp", 3},
                      {"r", 3},
                      {"p", 3},
                      {"q", 3}},
                     numDofs};

    auto cudaPol = zs::cuda_exec();
    // average edge length (for CCD filtering)
    initialize(cudaPol); // update vtemp, bvh, boxsize, targetGRes

    // adaptive dhat, targetGRes, kappa
    {
        // dHat (static)
        this->dHat = dHat_ * std::sqrt(boxDiagSize2);
        // adaptive epsv (static)
        if (epsv_ == 0) {
            this->epsv = this->dHat;
        } else {
            this->epsv = epsv_ * this->dHat;
        }
        // kappa (dynamic)
        suggestKappa(cudaPol);
        if (kappa0 != 0) {
            // zeno::log_info("manual kappa: {}\n", this->kappa);
        }
    }

    {
        // check initial self intersections
        // including proximity pairs
        // do once
        markSelfIntersectionPrimitives(cudaPol);
    }

    // output adaptive setups
    // zeno::log_info("auto dHat: {}, epsv (friction): {}\n", this->dHat, this->epsv);
}

void IPCSystem::reinitialize(zs::CudaExecutionPolicy &pol, typename IPCSystem::T framedt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    dt = framedt;
    this->framedt = framedt;
    curRatio = 0;

    substep = -1;
    projectDBC = false;
    BCsatisfied = false;

    if (enableContact) {
        nPP.setVal(0);
        nPE.setVal(0);
        nPT.setVal(0);
        nEE.setVal(0);

        nPPM.setVal(0);
        nPEM.setVal(0);
        nEEM.setVal(0);

        nFPP.setVal(0);
        nFPE.setVal(0);
        nFPT.setVal(0);
        nFEE.setVal(0);

        ncsPT.setVal(0);
        ncsEE.setVal(0);
    }

    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto &verts = primHandle.getVerts();
        // initialize BC info
        // predict pos, initialize augmented lagrangian, constrain weights
        pol(Collapse(verts.size()), [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
                                     voffset = primHandle.vOffset, dt = dt, asBoundary = primHandle.isBoundary(),
                                     avgNodeMass = avgNodeMass, augLagCoeff = augLagCoeff] __device__(int i) mutable {
            auto x = verts.pack<3>("x", i);
            auto v = verts.pack<3>("v", i);
            int BCorder = 0;
            auto BCtarget = x + v * dt;
            auto BCbasis = mat3::identity();
            int BCfixed = 0;
            if (!asBoundary) {
                BCorder = verts("BCorder", i);
                BCtarget = verts.pack(dim_c<3>, "BCtarget", i);
                BCbasis = verts.pack(dim_c<3, 3>, "BCbasis", i);
                BCfixed = verts("BCfixed", i);
            }
            vtemp("BCorder", voffset + i) = BCorder;
            vtemp.template tuple<3>("BCtarget", voffset + i) = BCtarget;
            vtemp.tuple(dim_c<9>, "BCbasis", voffset + i) = BCbasis;
            vtemp("BCfixed", voffset + i) = BCfixed;
            vtemp("BCsoft", voffset + i) = (int)asBoundary;

            vtemp("ws", voffset + i) = asBoundary || BCorder == 3 ? avgNodeMass * augLagCoeff : zs::sqrt(verts("m", i));
            vtemp.tuple<3>("xtilde", voffset + i) = x + v * dt;
            vtemp.tuple<3>("lambda", voffset + i) = vec3::zeros();
            vtemp.tuple<3>("xn", voffset + i) = x;
            vtemp.tuple<3>("xhat", voffset + i) = x;
            if (BCorder > 0) {
                // recover original BCtarget
                BCtarget = BCbasis * BCtarget;
                vtemp.tuple<3>("vn", voffset + i) = (BCtarget - x) / dt;
            } else {
                vtemp.tuple<3>("vn", voffset + i) = v;
            }
            // vtemp.tuple<3>("xt", voffset + i) = x;
            vtemp.tuple<3>("x0", voffset + i) = verts.pack<3>("x0", i);
        });
    }
    if (coVerts)
        if (auto coSize = coVerts->size(); coSize) {
            pol(Collapse(coSize),
                [vtemp = proxy<space>({}, vtemp), coverts = proxy<space>({}, *coVerts), coOffset = coOffset, dt = dt,
                 augLagCoeff = augLagCoeff, avgNodeMass = avgNodeMass] __device__(int i) mutable {
                    auto x = coverts.pack<3>("x", i);
                    vec3 newX{};
                    if (coverts.hasProperty("BCtarget"))
                        newX = coverts.pack<3>("BCtarget", i);
                    else {
                        auto v = coverts.pack<3>("v", i);
                        newX = x + v * dt;
                    }
                    vtemp("BCorder", coOffset + i) = 3;
                    vtemp.tuple(dim_c<9>, "BCbasis", coOffset + i) = mat3::identity();
                    vtemp.template tuple<3>("BCtarget", coOffset + i) = newX;
                    vtemp("BCfixed", coOffset + i) = (newX - x).l2NormSqr() == 0 ? 1 : 0;

                    vtemp("ws", coOffset + i) = avgNodeMass * augLagCoeff;
                    vtemp.tuple<3>("xtilde", coOffset + i) = newX;
                    vtemp.tuple<3>("lambda", coOffset + i) = vec3::zeros();
                    vtemp.tuple<3>("xn", coOffset + i) = x;
                    vtemp.tuple<3>("vn", coOffset + i) = (newX - x) / dt;
                    // vtemp.tuple<3>("xt", coOffset + i) = x;
                    vtemp.tuple<3>("xhat", coOffset + i) = x;
                    vtemp.tuple<3>("x0", coOffset + i) = coverts.pack<3>("x0", i);
                });
        }

    // spatial accel structs
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
        bvs.resize(stInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0, bvs);
        stBvh.build(pol, bvs);
        init_front(svInds, selfStFront);

        bvs.resize(seInds.size());
        retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, 0, bvs);
        seBvh.build(pol, bvs);
        init_front(seInds, selfSeFront);
    }
    if (coVerts)
        if (coVerts->size()) {
            bvs.resize(coEles->size());
            retrieve_bounding_volumes(pol, vtemp, "xn", *coEles, zs::wrapv<3>{}, coOffset, bvs);
            bouStBvh.build(pol, bvs);
            init_front(svInds, boundaryStFront);

            bvs.resize(coEdges->size());
            retrieve_bounding_volumes(pol, vtemp, "xn", *coEdges, zs::wrapv<2>{}, coOffset, bvs);
            bouSeBvh.build(pol, bvs);
            init_front(seInds, boundarySeFront);
        }

    updateWholeBoundingBoxSize(pol);
    /// update grad pn residual tolerance
    targetGRes = pnRel * std::sqrt(boxDiagSize2);
    // zeno::log_info("box diag size: {}, targetGRes: {}\n", std::sqrt(boxDiagSize2), targetGRes);

    /// for faster linear solve
    hess1.init(vtemp.get_allocator(), numDofs);
    hess2.init(PP.get_allocator(), estNumCps);
    hess3.init(PP.get_allocator(), estNumCps);
    hess4.init(PP.get_allocator(), estNumCps);
}
void IPCSystem::suggestKappa(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    auto cudaPol = zs::cuda_exec();
    if (kappa0 == 0) {
        /// kappaMin
        initKappa(cudaPol);
        /// adaptive kappa
        { // tet-oriented
            T H_b = computeHb((T)1e-16 * boxDiagSize2, this->dHat * this->dHat);
            kappa = 1e11 * avgNodeMass / (4e-16 * boxDiagSize2 * H_b);
            kappaMax = 100 * kappa;
            if (kappa < kappaMin)
                kappa = kappaMin;
            if (kappa > kappaMax)
                kappa = kappaMax;
        }
        { // surf oriented (use framedt here)
            auto kappaSurf = dt * dt * meanSurfaceArea / 3 * this->dHat * largestMu();
            // zeno::log_info("kappaSurf: {}, auto kappa: {}\n", kappaSurf, kappa);
            if (kappaSurf > kappa && kappaSurf < kappaMax) {
                kappa = kappaSurf;
            }
        }
        // boundaryKappa = kappa;
        // zeno::log_info("average node mass: {}, auto kappa: {} ({} - {})\n", avgNodeMass, this->kappa, this->kappaMin,
        //               this->kappaMax);
    }
}
void IPCSystem::advanceSubstep(zs::CudaExecutionPolicy &pol, typename IPCSystem::T ratio) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    // setup substep dt
    ++substep;
    dt = framedt * ratio;
    curRatio += ratio;

    projectDBC = false;
    BCsatisfied = false;
    pol(Collapse(coOffset), [vtemp = proxy<space>({}, vtemp), coOffset = coOffset, dt = dt] __device__(int vi) mutable {
        int BCorder = vtemp("BCorder", vi);
        auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
        auto projVec = [&BCbasis, BCorder](auto &dx) {
            dx = BCbasis.transpose() * dx;
            for (int d = 0; d != BCorder; ++d)
                dx[d] = 0;
            dx = BCbasis * dx;
        };
        auto xn = vtemp.pack(dim_c<3>, "xn", vi);
        vtemp.template tuple<3>("xhat", vi) = xn;
        auto deltaX = vtemp.pack(dim_c<3>, "vn", vi) * dt;
        if (BCorder > 0)
            projVec(deltaX);
        auto newX = xn + deltaX;
        vtemp.template tuple<3>("xtilde", vi) = newX;

        // update "BCfixed", "BCtarget" for dofs under boundary influence
        if (BCorder > 0) {
            vtemp.template tuple<3>("BCtarget", vi) = BCbasis.transpose() * newX;
            vtemp("BCfixed", vi) = deltaX.l2NormSqr() == 0 ? 1 : 0;
        }
    });
    if (coVerts)
        if (auto coSize = coVerts->size(); coSize)
            pol(Collapse(coSize),
                [vtemp = proxy<space>({}, vtemp), coverts = proxy<space>({}, *coVerts), coOffset = coOffset,
                 framedt = framedt, curRatio = curRatio] __device__(int i) mutable {
                    auto xhat = vtemp.pack(dim_c<3>, "xhat", coOffset + i);
                    auto xn = vtemp.pack(dim_c<3>, "xn", coOffset + i);
                    vtemp.template tuple<3>("xhat", coOffset + i) = xn;
                    vec3 newX{};
                    if (coverts.hasProperty("BCtarget"))
                        newX = coverts.pack<3>("BCtarget", i);
                    else {
                        auto v = coverts.pack<3>("v", i);
                        newX = xhat + v * framedt;
                    }
                    // auto xk = xhat + (newX - xhat) * curRatio;
                    auto xk = newX * curRatio + (1 - curRatio) * xhat;
                    vtemp.template tuple<3>("BCtarget", coOffset + i) = xk;
                    vtemp("BCfixed", coOffset + i) = (xk - xn).l2NormSqr() == 0 ? 1 : 0;
                    vtemp.template tuple<3>("xtilde", coOffset + i) = xk;
                });
    for (auto &primHandle : auxPrims) {
        if (primHandle.category == ZenoParticles::category_e::tracker) {
            const auto &eles = primHandle.getEles();
            pol(Collapse(eles.size()), [vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, eles),
                                        framedt = framedt, curRatio = curRatio] __device__(int ei) mutable {
                auto inds = eles.pack(dim_c<2>, "inds", ei).template reinterpret_bits<int>();
                // retrieve motion from associated boundary vert
                auto deltaX = vtemp.pack(dim_c<3>, "BCtarget", inds[1]) - vtemp.pack(dim_c<3>, "xhat", inds[1]);
                //
                auto xn = vtemp.pack(dim_c<3>, "xn", inds[0]);
                vtemp.template tuple<3>("BCtarget", inds[0]) = xn + deltaX;
                vtemp.tuple(dim_c<9>, "BCbasis", inds[0]) = mat3::identity();
                vtemp("BCfixed", inds[0]) = deltaX.l2NormSqr() == 0 ? 1 : 0;
                vtemp("BCorder", inds[0]) = 3;
                vtemp("BCsoft", inds[0]) = 0;
                vtemp.template tuple<3>("xtilde", inds[0]) = xn + deltaX;
            });
        }
    }
}
void IPCSystem::updateVelocities(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(zs::range(coOffset), [vtemp = proxy<space>({}, vtemp), dt = dt] __device__(int vi) mutable {
        auto newX = vtemp.pack<3>("xn", vi);
        auto dv = (newX - vtemp.pack<3>("xtilde", vi)) / dt;
        auto vn = vtemp.pack<3>("vn", vi);
        if (dv.length() > 4)
            dv = dv.normalized() * 4;
        vn += dv;
        int BCorder = vtemp("BCorder", vi);
        auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
        auto projVec = [&BCbasis, BCorder](auto &dx) {
            dx = BCbasis.transpose() * dx;
            for (int d = 0; d != BCorder; ++d)
                dx[d] = 0;
            dx = BCbasis * dx;
        };
        if (BCorder > 0)
            projVec(vn);
        vtemp.tuple<3>("vn", vi) = vn;
    });
}
void IPCSystem::writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    for (auto &primHandle : prims) {
        if (primHandle.isAuxiliary())
            continue;
        auto &verts = primHandle.getVerts();
        // update velocity and positions
        pol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts), dt = dt, vOffset = primHandle.vOffset,
             asBoundary = primHandle.isBoundary()] __device__(int vi) mutable {
                verts.tuple<3>("x", vi) = vtemp.pack<3>("xn", vOffset + vi);
                if (!asBoundary)
                    verts.tuple<3>("v", vi) = vtemp.pack<3>("vn", vOffset + vi);
            });
    }
    // not sure if this is necessary for numerical reasons
    if (coVerts && coLowResVerts)
        if (auto coSize = coVerts->size(); coSize)
            pol(Collapse(coSize),
                [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, *const_cast<dtiles_t *>(coVerts)),
                 loVerts = proxy<space>({}, *const_cast<tiles_t *>(coLowResVerts)),
                 coOffset = coOffset] ZS_LAMBDA(int vi) mutable {
                    auto newX = vtemp.pack(dim_c<3>, "xn", coOffset + vi);
                    verts.template tuple<3>("x", vi) = newX;
                    loVerts.template tuple<3>("x", vi) = newX;
                    // no need to update v here. positions are moved accordingly
                    // also, boundary velocies are set elsewhere
                });
}

struct MakeIPCSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto zstets = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        // auto zstets = get_input<ZenoParticles>("ZSParticles");
        std::shared_ptr<ZenoParticles> zsboundary;
        if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
            zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");

        auto cudaPol = zs::cuda_exec();
        for (auto zstet : zstets) {
            if (!zstet->hasImage(ZenoParticles::s_particleTag)) {
                auto &loVerts = zstet->getParticles();
                auto &verts = zstet->images[ZenoParticles::s_particleTag];
                verts = typename ZenoParticles::dtiles_t{loVerts.get_allocator(), loVerts.getPropertyTags(),
                                                         loVerts.size()};
                cudaPol(range(verts.size()), [loVerts = proxy<space>({}, loVerts),
                                              verts = proxy<space>({}, verts)] __device__(int vi) mutable {
                    // make sure there are no "inds"-like properties in verts!
                    for (int propid = 0; propid != verts._N; ++propid) {
                        auto propOffset = verts._tagOffsets[propid];
                        for (int chn = 0; chn != verts._tagSizes[propid]; ++chn)
                            verts(propOffset + chn, vi) = loVerts(propOffset + chn, vi);
                    }
                });
            }
        }
        if (zsboundary)
            if (!zsboundary->hasImage(ZenoParticles::s_particleTag)) {
                auto &loVerts = zsboundary->getParticles();
                auto &verts = zsboundary->images[ZenoParticles::s_particleTag];
                verts = typename ZenoParticles::dtiles_t{loVerts.get_allocator(), loVerts.getPropertyTags(),
                                                         loVerts.size()};
                cudaPol(range(verts.size()), [loVerts = proxy<space>({}, loVerts),
                                              verts = proxy<space>({}, verts)] __device__(int vi) mutable {
                    // make sure there are no "inds"-like properties in verts!
                    for (int propid = 0; propid != verts._N; ++propid) {
                        auto propOffset = verts._tagOffsets[propid];
                        for (int chn = 0; chn != verts._tagSizes[propid]; ++chn)
                            verts(propOffset + chn, vi) = loVerts(propOffset + chn, vi);
                    }
                });
            }

        const typename IPCSystem::dtiles_t *coVerts =
            zsboundary ? &zsboundary->images[ZenoParticles::s_particleTag] : nullptr;
        const typename IPCSystem::tiles_t *coLowResVerts = zsboundary ? &zsboundary->getParticles() : nullptr;
        const typename IPCSystem::tiles_t *coEdges =
            zsboundary ? &(*zsboundary)[ZenoParticles::s_surfEdgeTag] : nullptr;
        const typename IPCSystem::tiles_t *coEles = zsboundary ? &zsboundary->getQuadraturePoints() : nullptr;

        /// solver parameters
        auto input_est_num_cps = get_input2<int>("est_num_cps");
        auto input_withGround = get_input2<bool>("with_ground");
        auto input_withContact = get_input2<bool>("with_contact");
        auto input_withMollification = get_input2<bool>("with_mollification");
        auto input_contactEE = get_input2<bool>("contact_with_ee");
        auto input_contactSelf = get_input2<bool>("contact_with_self");
        auto input_dHat = get_input2<float>("dHat");
        auto input_epsv = get_input2<float>("epsv");
        auto input_kappa0 = get_input2<float>("kappa0");
        auto input_fricIterCap = get_input2<int>("fric_iter_cap");
        auto input_fric_mu = get_input2<float>("fric_mu");
        auto input_aug_coeff = get_input2<float>("aug_coeff");
        auto input_pn_rel = get_input2<float>("pn_rel");
        auto input_cg_rel = get_input2<float>("cg_rel");
        auto input_pn_cap = get_input2<int>("pn_iter_cap");
        auto input_cg_cap = get_input2<int>("cg_iter_cap");
        auto input_ccd_cap = get_input2<int>("ccd_iter_cap");
        auto input_gravity = get_input2<float>("gravity");
        auto dt = get_input2<float>("dt");
        auto groundNormal = get_input<zeno::NumericObject>("ground_normal")->get<zeno::vec3f>();
        if (auto len2 = lengthSquared(groundNormal); len2 > limits<float>::epsilon() * 10) {
            auto len = std::sqrt(len2);
            groundNormal /= len;
        } else
            groundNormal = zeno::vec3f{0, 1, 0}; // fallback to default up direction when degenerated

        auto A = std::make_shared<IPCSystem>(
            zstets, coVerts, coLowResVerts, coEdges, coEles, dt,
            (std::size_t)(input_est_num_cps ? input_est_num_cps : 1000000), input_withGround, input_withContact,
            input_withMollification, input_aug_coeff, input_pn_rel, input_cg_rel, input_pn_cap, input_cg_cap,
            input_ccd_cap, input_kappa0, input_fric_mu, input_dHat, input_epsv, groundNormal, input_gravity);
        A->enableContactEE = input_contactEE;
        A->enableContactSelf = input_contactSelf;
        A->fricIterCap = input_fricIterCap;

        set_output("ZSIPCSystem", A);
    }
};

ZENDEFNODE(MakeIPCSystem, {{
                               "ZSParticles",
                               "ZSBoundaryPrimitives",
                               {"int", "est_num_cps", "1000000"},
                               {"bool", "with_ground", "0"},
                               {"bool", "with_contact", "1"},
                               {"bool", "with_mollification", "1"},
                               {"bool", "contact_with_ee", "1"},
                               {"bool", "contact_with_self", "1"},
                               {"float", "dt", "0.01"},
                               {"float", "dHat", "0.001"},
                               {"vec3f", "ground_normal", "0,1,0"},
                               {"float", "epsv", "0.0"},
                               {"float", "kappa0", "0"},
                               {"int", "fric_iter_cap", "2"},
                               {"float", "fric_mu", "0"},
                               {"float", "aug_coeff", "1e2"},
                               {"float", "pn_rel", "0.01"},
                               {"float", "cg_rel", "0.001"},
                               {"int", "pn_iter_cap", "1000"},
                               {"int", "cg_iter_cap", "1000"},
                               {"int", "ccd_iter_cap", "20000"},
                               {"float", "gravity", "-9.8"},
                           },
                           {"ZSIPCSystem"},
                           {},
                           {"FEM"}});

} // namespace zeno