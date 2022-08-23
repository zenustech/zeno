#include "../Utils.hpp"
#include "Solver.cuh"

namespace zeno {

/// utilities
static constexpr std::size_t count_warps(std::size_t n) noexcept {
    return (n + 31) / 32;
}
static constexpr int warp_index(int n) noexcept {
    return n / 32;
}
static constexpr auto warp_mask(int i, int n) noexcept {
    int k = n % 32;
    const int tail = n - k;
    if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
    return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
}

template <typename T> static __forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst) {
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val += tmp;
    }
    if (locid == 0)
        zs::atomic_add(zs::exec_cuda, &dst, val);
}

template <typename T> static inline T computeHb(const T d2, const T dHat2) {
    if (d2 >= dHat2)
        return 0;
    T t2 = d2 - dHat2;
    return ((std::log(d2 / dHat2) * -2 - t2 * 4 / d2) + (t2 / d2) * (t2 / d2));
}

template <typename TileVecT, int codim = 3>
zs::Vector<zs::AABBBox<3, typename TileVecT::value_type>>
retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT &vtemp, const zs::SmallString &xTag,
                          const typename ZenoParticles::particles_t &eles, zs::wrapv<codim>, int voffset) {
    using namespace zs;
    using T = typename TileVecT::value_type;
    using bv_t = AABBBox<3, T>;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    zs::Vector<bv_t> ret{eles.get_allocator(), eles.size()};
    pol(range(eles.size()), [eles = proxy<space>({}, eles), bvs = proxy<space>(ret), vtemp = proxy<space>({}, vtemp),
                             codim_v = wrapv<codim>{}, xTag, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = RM_CVREF_T(codim_v)::value;
        auto inds = eles.template pack<dim>("inds", ei).template reinterpret_bits<int>() + voffset;
        auto x0 = vtemp.template pack<3>(xTag, inds[0]);
        bv_t bv{x0, x0};
        for (int d = 1; d != dim; ++d)
            merge(bv, vtemp.template pack<3>(xTag, inds[d]));
        bvs[ei] = bv;
    });
    return ret;
}
template <typename TileVecT0, typename TileVecT1, int codim = 3>
zs::Vector<zs::AABBBox<3, typename TileVecT0::value_type>>
retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT0 &verts, const zs::SmallString &xTag,
                          const typename ZenoParticles::particles_t &eles, zs::wrapv<codim>, const TileVecT1 &vtemp,
                          const zs::SmallString &dirTag, float stepSize, int voffset) {
    using namespace zs;
    using T = typename TileVecT0::value_type;
    using bv_t = AABBBox<3, T>;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    Vector<bv_t> ret{eles.get_allocator(), eles.size()};
    pol(zs::range(eles.size()), [eles = proxy<space>({}, eles), bvs = proxy<space>(ret),
                                 verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
                                 codim_v = wrapv<codim>{}, xTag, dirTag, stepSize, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = RM_CVREF_T(codim_v)::value;
        auto inds = eles.template pack<dim>("inds", ei).template reinterpret_bits<int>() + voffset;
        auto x0 = verts.template pack<3>(xTag, inds[0]);
        auto dir0 = vtemp.template pack<3>(dirTag, inds[0]);
        bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
        for (int d = 1; d != dim; ++d) {
            auto x = verts.template pack<3>(xTag, inds[d]);
            auto dir = vtemp.template pack<3>(dirTag, inds[d]);
            merge(bv, x);
            merge(bv, x + stepSize * dir);
        }
        bvs[ei] = bv;
    });
    return ret;
}
template <typename Op = std::plus<typename IPCSystem::T>>
static typename IPCSystem::T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<typename IPCSystem::T> &res,
                                    Op op = {}) {
    using namespace zs;
    using T = typename IPCSystem::T;
    Vector<T> ret{res.get_allocator(), 1};
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0, op);
    return ret.getVal();
}
typename IPCSystem::T dot(zs::CudaExecutionPolicy &cudaPol, typename IPCSystem::dtiles_t &vertData,
                          const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // Vector<double> res{vertData.get_allocator(), vertData.size()};
    Vector<double> res{vertData.get_allocator(), count_warps(vertData.size())};
    zs::memset(zs::mem_device, res.data(), 0, sizeof(double) * count_warps(vertData.size()));
    cudaPol(range(vertData.size()), [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0, tag1,
                                     n = vertData.size()] __device__(int pi) mutable {
        auto v0 = data.pack<3>(tag0, pi);
        auto v1 = data.pack<3>(tag1, pi);
        auto v = v0.dot(v1);
        // res[pi] = v;
        reduce_to(pi, n, v, res[pi / 32]);
    });
    return reduce(cudaPol, res, std::plus<double>{});
}
typename IPCSystem::T infNorm(zs::CudaExecutionPolicy &cudaPol, typename IPCSystem::dtiles_t &vertData,
                              const zs::SmallString tag = "dir") {
    using namespace zs;
    using T = typename IPCSystem::T;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), count_warps(vertData.size())};
    zs::memset(zs::mem_device, res.data(), 0, sizeof(T) * count_warps(vertData.size()));
    cudaPol(range(vertData.size()), [data = proxy<space>({}, vertData), res = proxy<space>(res), tag,
                                     n = vertData.size()] __device__(int pi) mutable {
        auto v = data.pack<3>(tag, pi);
        auto val = v.abs().max();

        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = zs::max(val, tmp);
        }
        if (locid == 0)
            res[pi / 32] = val;
    });
    return reduce(cudaPol, res, getmax<T>{});
}

IPCSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<2>)
    : zsprim{zsprim}, models{zsprim.getModel()}, verts{zsprim.getParticles<true>()}, eles{zsprim.getQuadraturePoints()},
      etemp{zsprim.getQuadraturePoints().get_allocator(), {{"He", 6 * 6}}, zsprim.numElements()},
      surfTris{zsprim.getQuadraturePoints()},  // this is fake!
      surfEdges{zsprim.getQuadraturePoints()}, // all elements are surface edges
      surfVerts{zsprim[ZenoParticles::s_surfVertTag]}, vOffset{vOffset},
      svtemp{zsprim.getQuadraturePoints().get_allocator(),
             {{"H", 3 * 3}, {"fn", 1}},
             zsprim[ZenoParticles::s_surfVertTag].size()},
      sfOffset{sfOffset}, seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::curve)
        throw std::runtime_error("dimension of 2 but is not curve");
    vOffset += verts.size();
    // sfOffset += 0; // no surface triangles
    seOffset += surfEdges.size();
    svOffset += surfVerts.size();
}
IPCSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<3>)
    : zsprim{zsprim}, models{zsprim.getModel()}, verts{zsprim.getParticles<true>()}, eles{zsprim.getQuadraturePoints()},
      etemp{zsprim.getQuadraturePoints().get_allocator(), {{"He", 9 * 9}}, zsprim.numElements()},
      surfTris{zsprim.getQuadraturePoints()}, surfEdges{zsprim[ZenoParticles::s_surfEdgeTag]},
      surfVerts{zsprim[ZenoParticles::s_surfVertTag]}, vOffset{vOffset},
      svtemp{zsprim.getQuadraturePoints().get_allocator(),
             {{"H", 3 * 3}, {"fn", 1}},
             zsprim[ZenoParticles::s_surfVertTag].size()},
      sfOffset{sfOffset}, seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::surface)
        throw std::runtime_error("dimension of 3 but is not surface");
    vOffset += verts.size();
    sfOffset += surfTris.size();
    seOffset += surfEdges.size();
    svOffset += surfVerts.size();
}
IPCSystem::PrimitiveHandle::PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset,
                                            std::size_t &seOffset, std::size_t &svOffset, zs::wrapv<4>)
    : zsprim{zsprim}, models{zsprim.getModel()}, verts{zsprim.getParticles<true>()}, eles{zsprim.getQuadraturePoints()},
      etemp{zsprim.getQuadraturePoints().get_allocator(), {{"He", 12 * 12}}, zsprim.numElements()},
      surfTris{zsprim[ZenoParticles::s_surfTriTag]}, surfEdges{zsprim[ZenoParticles::s_surfEdgeTag]},
      surfVerts{zsprim[ZenoParticles::s_surfVertTag]}, vOffset{vOffset},
      svtemp{zsprim.getQuadraturePoints().get_allocator(),
             {{"H", 3 * 3}, {"fn", 1}},
             zsprim[ZenoParticles::s_surfVertTag].size()},
      sfOffset{sfOffset}, seOffset{seOffset}, svOffset{svOffset}, category{zsprim.category} {
    if (category != ZenoParticles::tet)
        throw std::runtime_error("dimension of 4 but is not tetrahedra");
    vOffset += verts.size();
    sfOffset += surfTris.size();
    seOffset += surfEdges.size();
    svOffset += surfVerts.size();
}
typename IPCSystem::T IPCSystem::PrimitiveHandle::averageNodalMass(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprim.hasMeta(s_meanMassTag))
        return zsprim.readMeta(s_meanMassTag, zs::wrapt<T>{});
    Vector<T> masses{verts.get_allocator(), verts.size()};
    pol(Collapse{verts.size()}, [verts = proxy<space>({}, verts), masses = proxy<space>(masses)] ZS_LAMBDA(
                                    int vi) mutable { masses[vi] = verts("m", vi); });
    auto tmp = reduce(pol, masses) / masses.size();
    zsprim.setMeta(s_meanMassTag, tmp);
    return tmp;
}
typename IPCSystem::T IPCSystem::PrimitiveHandle::averageSurfEdgeLength(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprim.hasMeta(s_meanSurfEdgeLengthTag))
        return zsprim.readMeta(s_meanSurfEdgeLengthTag, zs::wrapt<T>{});
    auto &edges = surfEdges;
    Vector<T> edgeLengths{edges.get_allocator(), edges.size()};
    pol(Collapse{edges.size()}, [edges = proxy<space>({}, edges), verts = proxy<space>({}, verts),
                                 edgeLengths = proxy<space>(edgeLengths)] ZS_LAMBDA(int ei) mutable {
        auto inds = edges.template pack<2>("inds", ei).template reinterpret_bits<int>();
        edgeLengths[ei] = (verts.pack<3>("x0", inds[0]) - verts.pack<3>("x0", inds[1])).norm();
    });
    auto tmp = reduce(pol, edgeLengths) / edges.size();
    zsprim.setMeta(s_meanSurfEdgeLengthTag, tmp);
    return tmp;
}
typename IPCSystem::T IPCSystem::PrimitiveHandle::averageSurfArea(zs::CudaExecutionPolicy &pol) const {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (zsprim.category == ZenoParticles::curve)
        return (T)0;
    if (zsprim.hasMeta(s_meanSurfAreaTag))
        return zsprim.readMeta(s_meanSurfAreaTag, zs::wrapt<T>{});
    auto &tris = surfTris;
    Vector<T> surfAreas{tris.get_allocator(), tris.size()};
    pol(Collapse{surfAreas.size()}, [tris = proxy<space>({}, tris), verts = proxy<space>({}, verts),
                                     surfAreas = proxy<space>(surfAreas)] ZS_LAMBDA(int ei) mutable {
        auto inds = tris.template pack<3>("inds", ei).template reinterpret_bits<int>();
        surfAreas[ei] = (verts.pack<3>("x0", inds[1]) - verts.pack<3>("x0", inds[0]))
                            .cross(verts.pack<3>("x0", inds[2]) - verts.pack<3>("x0", inds[0]))
                            .norm() /
                        2;
    });
    auto tmp = reduce(pol, surfAreas) / tris.size();
    zsprim.setMeta(s_meanSurfAreaTag, tmp);
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
    if (coVerts.size()) {
        auto bouBv = bouSeBvh.getTotalBox(pol);
        merge(bv, bouBv._min);
        merge(bv, bouBv._max);
    }
    boxDiagSize2 = (bv._max - bv._min).l2NormSqr();
}
void IPCSystem::initKappa(zs::CudaExecutionPolicy &pol) {
#if 0
    // should be called after dHat set
    if (!s_enableContact)
        return;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
        vtemp.tuple<3>("p", i) = vec3::zeros();
        vtemp.tuple<3>("q", i) = vec3::zeros();
    });
    // inertial + elasticity
    computeInertialGradient(pol, "p");
    match([&](auto &elasticModel) { computeElasticGradientAndHessian(pol, elasticModel, "p", false); })(
        models.getElasticModel());
    // contacts
    findCollisionConstraints(pol, dHat, xi);
    auto prevKappa = kappa;
    kappa = 1;
    computeBarrierGradientAndHessian(pol, "q", false);
    computeBoundaryBarrierGradientAndHessian(pol, "q", false);
    kappa = prevKappa;

    auto gsum = dot(pol, vtemp, "p", "q");
    auto gsnorm = dot(pol, vtemp, "q", "q");
    if (gsnorm < limits<T>::min())
        kappaMin = 0;
    else
        kappaMin = -gsum / gsnorm;
    fmt::print("kappaMin: {}, gsum: {}, gsnorm: {}\n", kappaMin, gsum, gsnorm);
#endif
}

void IPCSystem::initialize(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    stInds = tiles_t{vtemp.get_allocator(), {{"inds", 3}}, sfOffset};
    seInds = tiles_t{vtemp.get_allocator(), {{"inds", 2}}, seOffset};
    svInds = tiles_t{vtemp.get_allocator(), {{"inds", 1}}, svOffset};

    meanEdgeLength = averageSurfEdgeLength(pol);
    meanSurfaceArea = averageSurfArea(pol);
    avgNodeMass = averageNodalMass(pol);
    for (auto &primHandle : prims) {
        auto &verts = primHandle.getVerts();
        // record surface (tri) indices
        if (primHandle.category != ZenoParticles::category_e::curve) {
            auto &tris = primHandle.getSurfTris();
            pol(Collapse(tris.size()),
                [stInds = proxy<space>({}, stInds), tris = proxy<space>({}, tris), voffset = primHandle.vOffset,
                 sfoffset = primHandle.sfOffset] __device__(int i) mutable {
                    stInds.template tuple<3>("inds", sfoffset + i) =
                        (tris.template pack<3>("inds", i).template reinterpret_bits<int>() + (int)voffset)
                            .template reinterpret_bits<float>();
                });
        }
        auto &edges = primHandle.getSurfEdges();
        pol(Collapse(edges.size()),
            [seInds = proxy<space>({}, seInds), edges = proxy<space>({}, edges), voffset = primHandle.vOffset,
             seoffset = primHandle.seOffset] __device__(int i) mutable {
                seInds.template tuple<2>("inds", seoffset + i) =
                    (edges.template pack<2>("inds", i).template reinterpret_bits<int>() + (int)voffset)
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

IPCSystem::IPCSystem(std::vector<ZenoParticles *> zsprims, const typename IPCSystem::dtiles_t &coVerts,
                     const typename IPCSystem::tiles_t &coEdges, const tiles_t &coEles, T dt, std::size_t estNumCps,
                     bool withGround, T augLagCoeff, T pnRel, T cgRel, int PNCap, int CGCap, int CCDCap, T kappa0,
                     T fricMu, T dHat, T epsv, T gravity)
    : coVerts{coVerts}, coEdges{coEdges}, coEles{coEles}, PP{estNumCps, zs::memsrc_e::um, 0},
      nPP{zsprims[0]->getParticles<true>().get_allocator(), 1}, tempPP{{{"H", 36}}, estNumCps, zs::memsrc_e::um, 0},
      PE{estNumCps, zs::memsrc_e::um, 0}, nPE{zsprims[0]->getParticles<true>().get_allocator(), 1},
      tempPE{{{"H", 81}}, estNumCps, zs::memsrc_e::um, 0}, PT{estNumCps, zs::memsrc_e::um, 0},
      nPT{zsprims[0]->getParticles<true>().get_allocator(), 1}, tempPT{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0},
      EE{estNumCps, zs::memsrc_e::um, 0}, nEE{zsprims[0]->getParticles<true>().get_allocator(), 1},
      tempEE{{{"H", 144}}, estNumCps, zs::memsrc_e::um, 0},
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
      ncsEE{zsprims[0]->getParticles<true>().get_allocator(), 1}, dt{dt}, framedt{dt}, curRatio{0},
      estNumCps{estNumCps}, s_enableGround{withGround},
      augLagCoeff{augLagCoeff}, pnRel{pnRel}, cgRel{cgRel}, PNCap{PNCap}, CGCap{CGCap}, CCDCap{CCDCap}, kappa{kappa0},
      kappa0{kappa0}, kappaMin{0}, kappaMax{kappa0}, fricMu{fricMu}, dHat{dHat}, epsv{epsv}, extForce{0, gravity, 0} {
    coOffset = sfOffset = seOffset = svOffset = 0;
    prevNumPP = prevNumPE = prevNumPT = prevNumEE = 0;
    for (auto primPtr : zsprims) {
        if (primPtr->category == ZenoParticles::category_e::curve) {
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<2>{});
        } else if (primPtr->category == ZenoParticles::category_e::surface)
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<3>{});
        else if (primPtr->category == ZenoParticles::category_e::tet)
            prims.emplace_back(*primPtr, coOffset, sfOffset, seOffset, svOffset, zs::wrapv<4>{});
    }
    fmt::print("num total obj <verts, surfV, surfE, surfT>: {}, {}, {}, {}\n", coOffset, svOffset, seOffset, sfOffset);

    numDofs = coOffset + coVerts.size();
    vtemp = dtiles_t{zsprims[0]->getParticles<true>().get_allocator(),
                     {{"grad", 3},
                      {"P", 9},
                      // dirichlet boundary condition type; 0: NOT, 1: ZERO, 2: NONZERO
                      {"BCorder", 1},
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
                      {"x0", 3}, // initial positions
                      {"xn0", 3},
                      {"xtilde", 3},
                      {"xhat", 3}, // initial positions at the current substep (constraint,
                                   // extforce)
                      {"temp", 3},
                      {"r", 3},
                      {"p", 3},
                      {"q", 3}},
                     numDofs};
    // inertial hessian
    tempPB = dtiles_t{vtemp.get_allocator(), {{"Hi", 9}}, coOffset};
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

    auto cudaPol = zs::cuda_exec();
    // average edge length (for CCD filtering)
    initialize(cudaPol);

    // adaptive dhat, targetGRes, kappa
    {
        updateWholeBoundingBoxSize(cudaPol);
        fmt::print("box diag size: {}\n", std::sqrt(boxDiagSize2));
        /// dHat
        this->dHat = dHat * std::sqrt(boxDiagSize2);
        /// grad pn residual tolerance
        targetGRes = pnRel * std::sqrt(boxDiagSize2);
        if (kappa0 == 0) {
            /// kappaMin
            initKappa(cudaPol);
            /// adaptive kappa
            { // tet-oriented
                T H_b = computeHb((T)1e-16 * boxDiagSize2, dHat * dHat);
                kappa = 1e11 * avgNodeMass / (4e-16 * boxDiagSize2 * H_b);
                kappaMax = 100 * kappa;
                if (kappa < kappaMin)
                    kappa = kappaMin;
                if (kappa > kappaMax)
                    kappa = kappaMax;
            }
            { // surf oriented (use framedt here)
                auto kappaSurf = dt * dt * meanSurfaceArea / 3 * dHat * largestMu();
                fmt::print("kappaSurf: {}, auto kappa: {}\n", kappaSurf, kappa);
                if (kappaSurf > kappa && kappaSurf < kappaMax) {
                    kappa = kappaSurf;
                }
            }
            // boundaryKappa = kappa;
            fmt::print("average node mass: {}, auto kappa: {} ({} - {})\n", avgNodeMass, this->kappa, this->kappaMin,
                       this->kappaMax);
        } else {
            fmt::print("manual kappa: {}\n", this->kappa);
        }
        // getchar();
    }
    // adaptive epsv
    if (epsv == 0) {
        this->epsv = this->dHat;
    } else {
        this->epsv *= this->dHat;
    }
    // output adaptive setups
    fmt::print("auto dHat: {}, targetGRes: {}, epsv (friction): {}\n", this->dHat, this->targetGRes, this->epsv);
}

void IPCSystem::reinitialize(zs::CudaExecutionPolicy &pol, typename IPCSystem::T framedt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    dt = framedt;
    this->framedt = framedt;
    curRatio = 0;

    projectDBC = false;
    BCsatisfied = false;
    useGD = false;

    for (auto &primHandle : prims) {
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
                BCtarget = verts.template pack<3>("BCtarget", i);
                BCbasis = verts.template pack<3, 3>("BCbasis", i);
                BCfixed = verts("BCfixed", i);
            }
            vtemp("BCorder", voffset + i) = BCorder;
            vtemp.template tuple<3>("BCtarget", voffset + i) = BCtarget;
            vtemp.template tuple<9>("BCbasis", voffset + i) = BCbasis;
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
    if (auto coSize = coVerts.size(); coSize) {
        fmt::print("in IPC solver: coSize is {} \n", coSize);
        pol(Collapse(coSize),
            [vtemp = proxy<space>({}, vtemp), coverts = proxy<space>({}, coVerts), coOffset = coOffset, dt = dt,
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
                vtemp.template tuple<9>("BCbasis", coOffset + i) = mat3::identity();
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
    {
        auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", stInds, zs::wrapv<3>{}, 0);
        stBvh.build(pol, triBvs);
        auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", seInds, zs::wrapv<2>{}, 0);
        seBvh.build(pol, edgeBvs);
    }
    if (coVerts.size()) {
        auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", coEles, zs::wrapv<3>{}, coOffset);
        bouStBvh.build(pol, triBvs);
        auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", coEdges, zs::wrapv<2>{}, coOffset);
        bouSeBvh.build(pol, edgeBvs);
    }
    puts("444");
}
void IPCSystem::advanceSubstep(zs::CudaExecutionPolicy &pol, typename IPCSystem::T ratio) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    // setup substep dt
    dt = framedt * ratio;
    curRatio += ratio;

    projectDBC = false;
    BCsatisfied = false;
    useGD = false;
    pol(Collapse(coOffset), [vtemp = proxy<space>({}, vtemp), coOffset = coOffset, dt = dt, ratio,
                             localRatio = ratio / (1 - curRatio + ratio)] __device__(int vi) mutable {
        int BCorder = vtemp("BCorder", vi);
        auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
        auto projVec = [&BCbasis, BCorder](auto &dx) {
            dx = BCbasis.transpose() * dx;
            for (int d = 0; d != BCorder; ++d)
                dx[d] = 0;
            dx = BCbasis * dx;
        };
        auto xn = vtemp.template pack<3>("xn", vi);
        vtemp.template tuple<3>("xhat", vi) = xn;
        auto deltaX = vtemp.template pack<3>("vn", vi) * dt;
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
    if (auto coSize = coVerts.size(); coSize)
        pol(Collapse(coSize), [vtemp = proxy<space>({}, vtemp), coverts = proxy<space>({}, coVerts),
                               coOffset = coOffset, framedt = framedt, curRatio = curRatio] __device__(int i) mutable {
            auto xhat = vtemp.template pack<3>("xhat", coOffset + i);
            auto xn = vtemp.template pack<3>("xn", coOffset + i);
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

        const typename IPCSystem::dtiles_t &coVerts =
            zsboundary ? zsboundary->images[ZenoParticles::s_particleTag] : typename IPCSystem::dtiles_t{};
        const typename IPCSystem::tiles_t &coEdges =
            zsboundary ? (*zsboundary)[ZenoParticles::s_surfEdgeTag] : typename IPCSystem::tiles_t{};
        const typename IPCSystem::tiles_t &coEles =
            zsboundary ? zsboundary->getQuadraturePoints() : typename IPCSystem::tiles_t{};

        /// solver parameters
        auto input_est_num_cps = get_input2<int>("est_num_cps");
        auto input_withGround = get_input2<int>("with_ground");
        auto input_dHat = get_input2<float>("dHat");
        auto input_epsv = get_input2<float>("epsv");
        auto input_kappa0 = get_input2<float>("kappa0");
        auto input_fric_mu = get_input2<float>("fric_mu");
        auto input_aug_coeff = get_input2<float>("aug_coeff");
        auto input_pn_rel = get_input2<float>("pn_rel");
        auto input_cg_rel = get_input2<float>("cg_rel");
        auto input_pn_cap = get_input2<int>("pn_iter_cap");
        auto input_cg_cap = get_input2<int>("cg_iter_cap");
        auto input_ccd_cap = get_input2<int>("ccd_iter_cap");
        auto input_gravity = get_input2<float>("gravity");
        auto dt = get_input2<float>("dt");

        auto A = std::make_shared<IPCSystem>(
            zstets, coVerts, coEdges, coEles, dt, (std::size_t)(input_est_num_cps ? input_est_num_cps : 1000000),
            input_withGround, input_aug_coeff, input_pn_rel, input_cg_rel, input_pn_cap, input_cg_cap, input_ccd_cap,
            input_kappa0, input_fric_mu, input_dHat, input_epsv, input_gravity);

        set_output("ZSIPCSystem", A);
    }
};

ZENDEFNODE(MakeIPCSystem, {{
                               "ZSParticles",
                               "ZSBoundaryPrimitives",
                               {"int", "est_num_cps", "1000000"},
                               {"int", "with_ground", "0"},
                               {"float", "dt", "0.01"},
                               {"float", "dHat", "0.001"},
                               {"float", "epsv", "0.0"},
                               {"float", "kappa0", "0"},
                               {"float", "fric_mu", "0"},
                               {"float", "aug_coeff", "1e2"},
                               {"float", "pn_rel", "0.01"},
                               {"float", "cg_rel", "0.0001"},
                               {"int", "pn_iter_cap", "1000"},
                               {"int", "cg_iter_cap", "1000"},
                               {"int", "ccd_iter_cap", "20000"},
                               {"float", "gravity", "-9.8"},
                           },
                           {"ZSIPCSystem"},
                           {},
                           {"FEM"}});

struct AdvanceIPCSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto A = get_input<IPCSystem>("ZSIPCSystem");

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);
        puts("reinitialize");
        for (int subi = 0; subi != nSubsteps; ++subi) {
            fmt::print("processing substep {}\n", subi);

            A->advanceSubstep(cudaPol, (typename IPCSystem::T)1 / nSubsteps);
        }

        set_output("ZSIPCSystem", A);
    }
};

ZENDEFNODE(AdvanceIPCSystem, {{
                                  "ZSIPCSystem",
                                  {"int", "num_substeps", "1"},
                                  {"float", "dt", "0.01"},
                              },
                              {"ZSIPCSystem"},
                              {},
                              {"FEM"}});

} // namespace zeno