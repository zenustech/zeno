#pragma once
#include "SpatialAccel.cuh"
#include "Structures.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvtt.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/Vec.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#define s_useNewtonSolver 0 // 0 for gradient descent solver
#define s_useChebyshevAcc 1 // for GD solver
#define s_useGDDiagHess 1   // for GD solver
#define s_useLineSearch 1
#define s_debugOutput 0
#define s_useHardPhase 0
#define s_clothShearingCoeff 0.01f
#define s_silentMode 1
#define s_hardPhaseSilent 1
#define s_useMassSpring 0
#define s_debugRemoveHashTable 0
#define s_testLightCache 1
#define s_hardPhaseLinesearch 0
namespace zeno {

/// for cell-based collision detection

struct FastClothSystem : IObject {
    using T = float;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;

    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;
    using vec3f = zs::vec<float, 3>;
    using ivec3 = zs::vec<int, 3>;
    using ivec2 = zs::vec<int, 2>;
    using mat2 = zs::vec<T, 2, 2>;
    using mat3 = zs::vec<T, 3, 3>;
    using pair_t = zs::vec<int, 2>;
    using pair3_t = zs::vec<int, 3>;
    using pair4_t = zs::vec<int, 4>;
    using dpair_t = zs::vec<Ti, 2>;
    using dpair3_t = zs::vec<Ti, 3>;
    using dpair4_t = zs::vec<Ti, 4>;
    using bvh_t = zs::LBvh<3, int, T>;
    using sh_t = zs::SpatialHash<3, int, T>;
    using bv_t = typename bvh_t::Box;
#if !s_debugRemoveHashTable
    using etab_t = typename zs::bht<int, 2, int, 32>;
#endif
    static constexpr T s_constraint_residual = 1e-3;
    static constexpr T boundaryKappa = 1e1;
    inline static const char s_maxSurfEdgeLengthTag[] = "MaxEdgeLength";
    inline static const char s_meanMassTag[] = "MeanMass";
    inline static const char s_totalVolumeTag[] = "TotalVolume";

    // cloth, boundary
    struct PrimitiveHandle {
        // soft springs: only elasticity matters
        PrimitiveHandle(std::shared_ptr<tiles_t> elesPtr, ZenoParticles::category_e);
        PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset, Ti &svOffset, zs::wrapv<2>);
        PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset, Ti &svOffset, zs::wrapv<3>);
        PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset, Ti &svOffset, zs::wrapv<4>);

        T maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol, zs::Vector<T> &temp) const;
        T averageNodalMass(zs::CudaExecutionPolicy &pol) const;
        T totalVolume(zs::CudaExecutionPolicy &pol) const;

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

        std::shared_ptr<ZenoParticles> zsprimPtr{}; // nullptr if it is an auxiliary
        std::shared_ptr<const ZenoConstitutiveModel> modelsPtr;
        std::shared_ptr<tiles_t> vertsPtr;
        std::shared_ptr<ZenoParticles::particles_t> elesPtr;
        tiles_t etemp; // store elasticity hessian
        std::shared_ptr<ZenoParticles::particles_t> surfTrisPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfEdgesPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfVertsPtr;
        tiles_t svtemp; // surface vert hessian
        const int vOffset, sfOffset, seOffset, svOffset;
        ZenoParticles::category_e category;

        bool hasBC = false; 
    };

    bool hasBoundary() const noexcept {
        if (coVerts != nullptr)
            return coVerts->size() != 0;
        return false;
    }
    /// @note geometry queries, parameter setup
    T maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol, bool includeBoundary = true);
    T averageNodalMass(zs::CudaExecutionPolicy &pol);
    T totalVolume(zs::CudaExecutionPolicy &pol);
    auto largestLameParams() const {
        T mu = 0, lam = 0;
        for (auto &&primHandle : prims) {
            auto [m, l] = primHandle.getModelLameParams();
            if (m > mu)
                mu = m;
            if (l > lam)
                lam = l;
        }
        return zs::make_tuple(mu, lam);
    }
    void setupCollisionParams(zs::CudaExecutionPolicy &pol);

    void pushBoundarySprings(std::shared_ptr<tiles_t> elesPtr, ZenoParticles::category_e category) {
        auxPrims.push_back(PrimitiveHandle{std::move(elesPtr), category});
    }
    void updateWholeBoundingBoxSize(zs::CudaExecutionPolicy &pol);

    ///
    void initialize(zs::CudaExecutionPolicy &pol);
    FastClothSystem(std::vector<ZenoParticles *> zsprims, tiles_t *coVerts, tiles_t *coPoints, tiles_t *coEdges,
                    tiles_t *coEles, T dt, std::size_t ncps, bool withContact, T augLagCoeff, T pnRel, T cgRel,
                    int PNCap, int CGCap, T dHat, T gravity, int K, int IDyn, T BCStiffness, T mu, T LRef, T rho);

    /// @note initialize "ws" (mass), "yn", "vn" properties
    void reinitialize(zs::CudaExecutionPolicy &pol, T framedt);

    void updateVelocities(zs::CudaExecutionPolicy &pol);
    void writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol);

    /// collision
    void findConstraints(zs::CudaExecutionPolicy &pol, T dHat, const zs::SmallString &tag = "xinit");
    void lightCD(zs::CudaExecutionPolicy &pol, T dHat, const zs::SmallString &tag = "xinit");
    void findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, bool withBoundary);
    void lightFindCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat,
                                       bool withBoundary); // light version to be tested
    void lightFilterConstraints(zs::CudaExecutionPolicy &pol, T dHat, const zs::SmallString &tag);
    /// @note given "xinit", computes x^{k+1}
    void initialStepping(zs::CudaExecutionPolicy &pol);
    bool collisionStep(zs::CudaExecutionPolicy &pol, bool enableHardPhase); // given x^init (x^k) and y^{k+1}
    void softPhase(zs::CudaExecutionPolicy &pol);
    T hardPhase(zs::CudaExecutionPolicy &pol, T E0);
    bool constraintSatisfied(zs::CudaExecutionPolicy &pol, bool hasEps = true);
    T constraintEnergy(zs::CudaExecutionPolicy &pol);
    // void computeConstraintGradients(zs::CudaExecutionPolicy &cudaPol);

    /// pipeline
    void advanceSubstep(zs::CudaExecutionPolicy &pol, T ratio);
    void subStepping(zs::CudaExecutionPolicy &pol);
    // dynamics
    void computeInertialAndCouplingAndForceGradient(zs::CudaExecutionPolicy &cudaPol);
    void computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol);
    // boundary constraint
    void computeBoundaryConstraints(zs::CudaExecutionPolicy &pol);
    bool areBoundaryConstraintsSatisfied(zs::CudaExecutionPolicy &pol);
    T boundaryConstraintResidual(zs::CudaExecutionPolicy &pol);

    /// linear solve
    T dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0, const zs::SmallString tag1);
    T infNorm(zs::CudaExecutionPolicy &pol);
    T l2Norm(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag, const zs::SmallString dstTag);
    void multiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag);
    void cgsolve(zs::CudaExecutionPolicy &cudaPol);
    void newtonDynamicsStep(zs::CudaExecutionPolicy &pol);
    void gdDynamicsStep(zs::CudaExecutionPolicy &pol);
    T dynamicsEnergy(zs::CudaExecutionPolicy &pol);

    // for debug output data
    void writeFile(std::string filename, std::string info);
    int frameCnt = 0;

    // contacts
    auto getConstraintCnt() const {
        return std::make_tuple(nPP.getVal(), nE.getVal());
    }

    // sim params
    int substep = -1;
    std::size_t estNumCps = 1000000;

    bool firstStepping = true;
    T alpha = 0.3f; // step size
    T alphaMin = 0.10f;
    T alphaDecrease = 0.7f;
    T pnRel = 1e-2;
    T cgRel = 1e-2;
    int PNCap = 1000;
    int CGCap = 500;
    T armijoParam = 1e-4;
    T targetGRes = 1e-2;

    bool enableContact = true;
    bool enableContactSelf = true;
    bool projectDBC = false;
    T augLagCoeff = 1e4;
    T dHat = 8;
    vec3 extAccel;

    T boxDiagSize2 = 0;
    T avgNodeMass = 0, mu = 0, rho = 0;

    /// @brief for cloth collision handling
    /// @note all length in mm unit
    /// @note all vertex pair constraints are stored in <PP, nPP, tempPP>
    /// @note all edge constraints are stored in <E, nE, tempE>

    /// @note global upper length bound on all edges is L
    T L = 6 * zs::g_sqrt2;
    T LRef = 4.2, LAda = 6.5;
    /// @brief global lower bound on the distance between any two non-adjacent vertices in a mesh
    /// @brief (non-edge) vertex pair distance lower bound is B + Btight
    T B = 6, Btight = 0.5;
    /// @note vertex displacement upper boound is sqrt((B + Btight)^2 - B^2)

    /// @note small enough initial displacement for guaranted vertex displacement constraint
    /// @note i.e. || x^{k+1} - x^k || < 2 * || x^{init} - x^k ||

    /// @brief positive slack constant, used in proximity search
    /// @note sufficiently large enough to 1) faciliate convergence, 2) address missing pair issue
    /// @note sufficiently small enough to 1) reduce proximity search cost, 2) reduce early repulsion artifacts
    T epsSlack = 30; //21.75; // (B + Btight) * (B + Btight) + epsSlack + epsSlack < L^2
    // 42.25 + 72
    /// @brief hard phase constraint coefficients
    // constexpr T a0 = 0;
    // constexpr T a1 = 1;
    T a2 = 1 / (T)21.75, a3 = -1 / (T)473.0625;
    void updateHardPhaseFunctionCoefficients(T epsSlack) {
        a3 = -1 / (epsSlack * epsSlack);
        a2 = -epsSlack * a3;
    }

    /// @brief success condition constant (avoid near boundary numerical issues after the soft phase)
    T epsCond = 0.01;
    /// @brief initial displacement limit during the start of K iteration collision steps
    T D = 0.25;
    /// @brief coupling coefficients between cloth dynamics and collision dynamics
    T sigma = 160000; // s^{-2}
    /// @brief hard phase termination criteria
    T yita = 0.1;
    /// @brief counts: K [iterative steps], ISoft [soft phase steps], IHard [hard phase steps], IInit [x0 initialization]
    int K = 72, ISoft = 6 /*6~16*/, IHard = 8, IInit = 6;
    /// @brief counts: R [rollback steps for reduction], IDyn [cloth dynamics iters]
    int IDyn = 1 /*1~6*/, R = 8;
    T chebyOmega = 1.0f;
    T chebyRho = 0.99f;

    T proximityRadius() const {
        return std::sqrt((B + Btight) * (B + Btight) + epsSlack);
    }

    std::vector<PrimitiveHandle> prims;
    std::vector<PrimitiveHandle> auxPrims; // intended for hard constraint (tracker primitive)
    Ti coOffset, numDofs, numBouDofs;
    Ti sfOffset, seOffset, svOffset;

    // (scripted) collision objects
    /// @note allow (bisector) normal update on-the-fly, thus made modifiable
    tiles_t *coVerts, *coPoints, *coEdges, *coEles;

    tiles_t vtemp;        // solver data
    zs::Vector<T> temp;   // as temporary buffer
    zs::Vector<bv_t> bvs; // as temporary buffer

    // collision constraints (edge / non-edge)
    zs::Vector<pair_t> PP, E;
    zs::Vector<int> nPP, nE;
    // for light cache to be tested
    zs::Vector<pair_t> cPP, cE;
    zs::Vector<int> ncPP, ncE;
    int npp, ne;
    int ncpp, nce;
    tiles_t tempPP, tempE;
#if !s_debugRemoveHashTable
    etab_t eTab; // global surface edge hash table
#endif

#if 0
    zs::Vector<zs::u8> exclSes, exclSts, exclBouSes, exclBouSts; // mark exclusion
    zs::Vector<pair4_t> PT;
    zs::Vector<int> nPT;
    tiles_t tempPT;
    zs::Vector<pair4_t> EE;
    zs::Vector<int> nEE;
    tiles_t tempEE;

    zs::Vector<pair4_t> csPT, csEE;
    zs::Vector<int> ncsPT, ncsEE;
#endif
    // end contacts

    // boundary contacts
    // auxiliary data (spatial acceleration)
    tiles_t svInds, seInds, stInds;
    bvh_t svBvh;    // for simulated objects
    bvh_t bouSvBvh; // for collision objects
    sh_t svSh;
    sh_t bouSvSh;
    T dt, framedt, curRatio;

    zs::CppTimer timer;
    float auxTime[10]; // bvh build, bvh iter, sh build, sh iter
    float dynamicsTime[10];
    float collisionTime[10];
    int auxCnt[10];
    int dynamicsCnt[10];
    int collisionCnt[10];
    float initInterpolationTime;
    static constexpr bool s_enableProfile = true;

    T BCStiffness; 
#define s_testSh false
};

} // namespace zeno

#include "SolverUtils.cuh"