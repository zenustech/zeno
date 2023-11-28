#pragma once
#include "SpatialAccel.cuh"
#include "Structures.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#if 1
#include "SolverUtils.cuh"
#endif

namespace zeno {

struct IPCSystem : IObject {
    using T = double;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using dtiles_t = zs::TileVector<T, 32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;
    using vec3f = zs::vec<float, 3>;
    using ivec3 = zs::vec<int, 3>;
    using ivec2 = zs::vec<int, 2>;
    using mat2 = zs::vec<T, 2, 2>;
    using mat3 = zs::vec<T, 3, 3>;
    using mat3f = zs::vec<float, 3, 3>;
    using pair_t = zs::vec<int, 2>;
    using pair3_t = zs::vec<int, 3>;
    using pair4_t = zs::vec<int, 4>;
    using dpair_t = zs::vec<Ti, 2>;
    using dpair3_t = zs::vec<Ti, 3>;
    using dpair4_t = zs::vec<Ti, 4>;
    // using bvh_t = zeno::ZenoLBvh<3, 32, int, T>;
    using bvh_t = zs::LBvh<3, int, T>;
    using bvfront_t = zs::BvttFront<int, int>;
    using bv_t = zs::AABBBox<3, T>;

    inline static const char s_meanMassTag[] = "MeanMass";
    inline static const char s_meanSurfEdgeLengthTag[] = "MeanSurfEdgeLength";
    inline static const char s_meanSurfAreaTag[] = "MeanSurfArea";
    static constexpr T s_constraint_residual = 1e-2;

    struct PrimitiveHandle {
        PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset, std::size_t &seOffset,
                        std::size_t &svOffset, zs::wrapv<2>);
        PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset, std::size_t &seOffset,
                        std::size_t &svOffset, zs::wrapv<3>);
        PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset, std::size_t &seOffset,
                        std::size_t &svOffset, zs::wrapv<4>);
        // soft springs: only elasticity matters
        PrimitiveHandle(std::shared_ptr<tiles_t> elesPtr, ZenoParticles::category_e);
        T averageNodalMass(zs::CudaExecutionPolicy &pol) const;
        T averageSurfEdgeLength(zs::CudaExecutionPolicy &pol) const;
        T averageSurfArea(zs::CudaExecutionPolicy &pol) const;

        auto getModelLameParams() const {
            T mu = 0, lam = 0;
            if (!isAuxiliary() && modelsPtr) {
                zs::match([&](const auto &model) {
                    mu = model.mu;
                    lam = model.lam;
                })(modelsPtr->getElasticModel());
            }
            return zs::make_tuple(mu, lam);
        }

        decltype(auto) getModels() const {
            if (!modelsPtr)
                throw std::runtime_error("primhandle models not available");
            return *modelsPtr;
        }
        decltype(auto) getVerts() const {
            if (!vertsPtr)
                throw std::runtime_error("primhandle verts not available");
            return *vertsPtr;
        }
        decltype(auto) getEles() const {
            if (!elesPtr)
                throw std::runtime_error("primhandle eles not available");
            return *elesPtr;
        }
        decltype(auto) getSurfTris() const {
            if (!surfTrisPtr)
                throw std::runtime_error("primhandle surf tris not available");
            return *surfTrisPtr;
        }
        decltype(auto) getSurfEdges() const {
            if (!surfEdgesPtr)
                throw std::runtime_error("primhandle surf edges not available");
            return *surfEdgesPtr;
        }
        decltype(auto) getSurfVerts() const {
            if (!surfVertsPtr)
                throw std::runtime_error("primhandle surf verts not available");
            return *surfVertsPtr;
        }
        bool isAuxiliary() const noexcept {
            if (zsprimPtr == nullptr)
                return true;
            return false;
        }
        bool isBoundary() const noexcept {
            if (zsprimPtr == nullptr) // auxiliary primitive for soft binding
                return true;
            return zsprimPtr->asBoundary;
        }
        bool hasBendingConstraints() const noexcept {
            return static_cast<bool>(bendingEdgesPtr);
        }

        std::shared_ptr<ZenoParticles> zsprimPtr{}; // nullptr if it is an auxiliary
        std::shared_ptr<const ZenoConstitutiveModel> modelsPtr;
        std::shared_ptr<ZenoParticles::dtiles_t> vertsPtr;
        std::shared_ptr<ZenoParticles::particles_t> elesPtr;
        std::shared_ptr<ZenoParticles::particles_t> bendingEdgesPtr;
        typename ZenoParticles::dtiles_t etemp, btemp; // elasticity, bending
        std::shared_ptr<ZenoParticles::particles_t> surfTrisPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfEdgesPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfVertsPtr;
        typename ZenoParticles::dtiles_t svtemp;
        const std::size_t vOffset, sfOffset, seOffset, svOffset;
        ZenoParticles::category_e category;
    };

    bool hasBoundary() const noexcept {
        if (coVerts != nullptr && coEdges != nullptr && coEles != nullptr)
            return (coVerts->size() > 0) && (coEdges->size() > 0) && (coEles->size() > 0);
        return false;
    }
    T averageNodalMass(zs::CudaExecutionPolicy &pol);
    T averageSurfEdgeLength(zs::CudaExecutionPolicy &pol);
    T averageSurfArea(zs::CudaExecutionPolicy &pol);
    T largestMu() const {
        T mu = 0;
        for (auto &&primHandle : prims) {
            auto [m, l] = primHandle.getModelLameParams();
            if (m > mu)
                mu = m;
        }
        return mu;
    }

    void pushBoundarySprings(std::shared_ptr<tiles_t> elesPtr, ZenoParticles::category_e category) {
        auxPrims.push_back(PrimitiveHandle{std::move(elesPtr), category});
    }
    bv_t updateWholeBoundingBoxSize(zs::CudaExecutionPolicy &pol);
    void initKappa(zs::CudaExecutionPolicy &pol);
    void initialize(zs::CudaExecutionPolicy &pol);
    IPCSystem(std::vector<ZenoParticles *> zsprims, const dtiles_t *coVerts, const tiles_t *coLowResVerts,
              const tiles_t *coEdges, const tiles_t *coEles, T dt, std::size_t ncps, bool withGround, bool withContact,
              bool withMollification, T augLagCoeff, T pnRel, T cgRel, int PNCap, int CGCap, int CCDCap, T kappa0,
              T fricMu, T dHat, T epsv, zeno::vec3f gn, T gravity);

    void reinitialize(zs::CudaExecutionPolicy &pol, T framedt);
    void suggestKappa(zs::CudaExecutionPolicy &pol);
    void advanceSubstep(zs::CudaExecutionPolicy &pol, T ratio);
    void updateVelocities(zs::CudaExecutionPolicy &pol);
    void writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol);

    /// pipeline
    bool newtonKrylov(zs::CudaExecutionPolicy &pol);
    // constraint
    void computeConstraints(zs::CudaExecutionPolicy &pol);
    bool areConstraintsSatisfied(zs::CudaExecutionPolicy &pol);
    T constraintResidual(zs::CudaExecutionPolicy &pol, bool maintainFixed = false);
    // contacts
    auto getCnts() const {
        return zs::make_tuple(nPP.getVal(), nPE.getVal(), nPT.getVal(), nEE.getVal(), nPPM.getVal(), nPEM.getVal(),
                              nEEM.getVal(), ncsPT.getVal(), ncsEE.getVal());
    }
    auto getCollisionCnts() const {
        return zs::make_tuple(ncsPT.getVal(), ncsEE.getVal());
    }
    void markSelfIntersectionPrimitives(zs::CudaExecutionPolicy &pol);
    void markSelfIntersectionPrimitives(zs::CudaExecutionPolicy &pol, zs::true_type);
    void findProximityPairs(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary);
    void findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, T xi = 0);
    void findCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary = false);
    void findBoundaryCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi);
    void precomputeFrictions(zs::CudaExecutionPolicy &pol, T dHat, T xi = 0); // called per optimization
    void findCCDConstraints(zs::CudaExecutionPolicy &pol, T alpha, T xi = 0);
    void findCCDConstraintsImpl(zs::CudaExecutionPolicy &pol, T alpha, T xi, bool withBoundary = false);
    // linear system setup
    void computeInertialAndGravityPotentialGradient(zs::CudaExecutionPolicy &cudaPol);
    void computeInertialPotentialGradient(zs::CudaExecutionPolicy &cudaPol,
                                          const zs::SmallString &gTag); // for kappaMin
    void computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag,
                                          bool includeHessian = true);
    void computeBendingGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag,
                                          bool includeHessian = true);
    void computeBoundaryBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, bool includeHessian = true);
    void computeBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag,
                                          bool includeHessian = true);
    void computeFrictionBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag,
                                                  bool includeHessian = true);

    void convertHessian(zs::CudaExecutionPolicy &pol);

    /// @note build linsys.spmat
    void initializeSystemHessian(zs::CudaExecutionPolicy &pol);
    // elasticity, bending, kinematic, external force potential, boundary motion, ground collision
    void updateInherentHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag);
    // mostly self-collision related
    void updateDynamicHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag);

    // krylov solver
    T infNorm(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag = "dir");
    T dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0, const zs::SmallString tag1);
    void project(zs::CudaExecutionPolicy &pol, zs::true_type, const zs::SmallString tag);
    void project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void precondition(zs::CudaExecutionPolicy &pol, zs::true_type, const zs::SmallString srcTag,
                      const zs::SmallString dstTag);
    void precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag, const zs::SmallString dstTag);

    void multiply(zs::CudaExecutionPolicy &pol, zs::true_type, const zs::SmallString dxTag,
                  const zs::SmallString bTag);
    void multiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag);
    void systemMultiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag);

    void cgsolve(zs::CudaExecutionPolicy &cudaPol, zs::true_type);
    void cgsolve(zs::CudaExecutionPolicy &cudaPol);
    void systemSolve(zs::CudaExecutionPolicy &cudaPol);

    void groundIntersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T &stepSize);
    void intersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T xi, T &stepSize);
    T energy(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void lineSearch(zs::CudaExecutionPolicy &cudaPol, T &alpha);

    // sim params
    std::size_t estNumCps = 1000000;
    bool enableGround = false;
    bool enableContact = true;
    bool enableMollification = true;
    bool enableContactEE = true;
    bool enableContactSelf = true;
    bool s_enableFriction = true;
    bool s_enableSelfFriction = true;
    vec3 s_groundNormal{0, 1, 0};
    T augLagCoeff = 1e4;
    T pnRel = 1e-2;
    T cgRel = 1e-2;
    int fricIterCap = 2;
    int PNCap = 1000;
    int CGCap = 500;
    int CCDCap = 20000;
    T kappa0 = 1e4;
    T fricMu = 0;
    T boundaryKappa = 1;
    T xi = 0; // 1e-2; // 2e-3;
    T dHat = 0.0025;
    T epsv = 0.0;
    vec3 extForce;

    T kappaMax = 1e8;
    T kappaMin = 1e4;
    T kappa = kappa0;
    bool projectDBC = false;
    bool BCsatisfied = false;
    bool needFricPrecompute = true;
    T updateZoneTol = 1e-1;
    T consTol = 1e-2;
    T armijoParam = 1e-4;
    T boxDiagSize2 = 0;
    T meanEdgeLength = 0, meanSurfaceArea = 0, avgNodeMass = 0;
    T targetGRes = 1e-2;

    //
    std::vector<PrimitiveHandle> prims;
    std::vector<PrimitiveHandle> auxPrims;
    std::size_t coOffset, numDofs, numBouDofs;
    std::size_t sfOffset, seOffset, svOffset;

    // (scripted) collision objects
    const dtiles_t *coVerts;
    const tiles_t *coLowResVerts, *coEdges, *coEles;
    dtiles_t vtemp;
    dtiles_t tempI;

    // self contacts
    zs::Vector<pair_t> PP;
    zs::Vector<int> nPP;
    dtiles_t tempPP;
    zs::Vector<pair3_t> PE;
    zs::Vector<int> nPE;
    dtiles_t tempPE;
    zs::Vector<pair4_t> PT;
    zs::Vector<int> nPT;
    dtiles_t tempPT;
    zs::Vector<pair4_t> EE;
    zs::Vector<int> nEE;
    dtiles_t tempEE;
    // mollifier
    zs::Vector<pair4_t> PPM;
    zs::Vector<int> nPPM;
    dtiles_t tempPPM;
    zs::Vector<pair4_t> PEM;
    zs::Vector<int> nPEM;
    dtiles_t tempPEM;
    zs::Vector<pair4_t> EEM;
    zs::Vector<int> nEEM;
    dtiles_t tempEEM;
    // friction
    zs::Vector<pair_t> FPP;
    zs::Vector<int> nFPP;
    dtiles_t fricPP;
    zs::Vector<pair3_t> FPE;
    zs::Vector<int> nFPE;
    dtiles_t fricPE;
    zs::Vector<pair4_t> FPT;
    zs::Vector<int> nFPT;
    dtiles_t fricPT;
    zs::Vector<pair4_t> FEE;
    zs::Vector<int> nFEE;
    dtiles_t fricEE;

    zs::Vector<zs::u8> exclSes, exclSts, exclBouSes, exclBouSts; // mark exclusion
    // end contacts

    zs::Vector<T> temp;   // generally 64-bit
    zs::Vector<bv_t> bvs; // as temporary buffer

    zs::Vector<pair4_t> csPT, csEE;
    zs::Vector<int> ncsPT, ncsEE;

    /// @brief solver state machine
    struct SolverState {
        void frameStepping() {
            frameNo++;
            substep = -1;
            curRatio = 0;
        }
        void subStepping(T ratio) {
            substep++;
            curRatio += ratio;
        }
        void reset() {
            substep = -1;
            frameNo = -1;
        }

        int getFrameNo() const noexcept {
            return frameNo;
        }
        int getSubstep() const noexcept {
            return substep;
        }

      private:
        int substep{-1};
        int frameNo{-1};
        T curRatio{0};
    } state;

    /// @brief for system hessian storage
    template <typename T_>
    struct SystemHessian {
        using T = T_;
        using vec3 = zs::vec<T, 3>;
        using mat3 = zs::vec<T, 3, 3>;
        using spmat_t = zs::SparseMatrix<mat3, true>;
        using hess2_t = HessianPiece<2, T>;
        using hess3_t = HessianPiece<3, T>;
        using hess4_t = HessianPiece<4, T>;

        /// @brief dynamic part, mainly for collision constraints
        bool initialized = false;
        /// @note initialization: hess.init(allocator, size)
        /// @note maintain: hess.reset(false, 0)    ->  hess.increaseCount(size)    ->  hess.hess/hess.inds
        HessianPiece<2, T> hess2;
        HessianPiece<3, T> hess3;
        HessianPiece<4, T> hess4;
        /// @brief static part
        spmat_t spmat{};
        /// @brief preconditioner
    };
    /// probably useful for all possible hessian maintenance?
    /// inherent + dynamic (discrete) 3x3 mat pieces?
    template <zs::execspace_e space, typename T_>
    struct SystemHessianView {
        using sys_hess_t = SystemHessian<T_>;
        using vec3 = zs::vec<T, 3>;
        using mat3 = zs::vec<T, 3, 3>;
        using spmat_t = zs::SparseMatrix<mat3, true>;
        using hess_t = HessianPiece<1, T>;

#if 0
        SystemHessianView(sys_hess_t &sys)
            : spmat{view<space>(sys.spmat, true_c)}, hess2{proxy<space>(sys.hess2)}, hess3{proxy<space>(sys.hess3)},
              hess4{proxy<space>(sys.hess4)} {
        }
#endif
        zs::SparseMatrixView<space, spmat_t, true> spmat;
        HessianView<hess_t> hess;
    };
    // for one-time static hessian topo build
    SystemHessian<T> linsys;

    // for faster linear system solve
    HessianPiece<1> hess1;
    HessianPiece<2> hess2;
    HessianPiece<3> hess3;
    HessianPiece<4> hess4;
    tiles_t cgtemp;

    // zs::SparseMatrix<mat3f, true> spmat{};

    // boundary contacts
    // auxiliary data (spatial acceleration)
    tiles_t stInds, seInds, svInds;
    using bvs_t = zs::LBvs<3, int, T>;
    bvh_t stBvh, seBvh;       // for simulated objects
    bvs_t stBvs, seBvs;       // STQ
    bvh_t bouStBvh, bouSeBvh; // for collision objects
    bvs_t bouStBvs, bouSeBvs; // STQ
    bvfront_t selfStFront, boundaryStFront;
    bvfront_t selfSeFront, boundarySeFront;
    bool frontManageRequired;
    std::optional<bv_t> wholeBv;
    T dt, framedt;
};

} // namespace zeno
