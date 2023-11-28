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

namespace zeno {

/// for cell-based collision detection

struct ClothSystem : IObject {
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
    using bvfront_t = zs::BvttFront<int, int>;
    using bv_t = typename bvh_t::Box;
    static constexpr T s_constraint_residual = 1e-2;
    static constexpr T boundaryKappa = 1e1;
    inline static const char s_maxSurfEdgeLengthTag[] = "MaxEdgeLength";
    inline static const char s_meanMassTag[] = "MeanMass";

    // cloth, boundary
    struct PrimitiveHandle {
        // soft springs: only elasticity matters
        PrimitiveHandle(std::shared_ptr<tiles_t> elesPtr, ZenoParticles::category_e);
        PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset, Ti &svOffset, zs::wrapv<2>);
        PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset, Ti &svOffset, zs::wrapv<3>);
        PrimitiveHandle(ZenoParticles &zsprim, Ti &vOffset, Ti &sfOffset, Ti &seOffset, Ti &svOffset, zs::wrapv<4>);

        T maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol, zs::Vector<T> &temp) const;
        T averageNodalMass(zs::CudaExecutionPolicy &pol) const;

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
    };

    bool hasBoundary() const noexcept {
        return coVerts != nullptr;
    }
    T maximumSurfEdgeLength(zs::CudaExecutionPolicy &pol, bool includeBoundary = true);
    T averageNodalMass(zs::CudaExecutionPolicy &pol);
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
    void initialize(zs::CudaExecutionPolicy &pol);
    ClothSystem(std::vector<ZenoParticles *> zsprims, tiles_t *coVerts, tiles_t *coEdges, tiles_t *coEles, T dt,
                std::size_t ncps, bool withContact, T augLagCoeff, T pnRel, T cgRel, int PNCap, int CGCap, T dHat,
                T gravity);

    void reinitialize(zs::CudaExecutionPolicy &pol, T framedt);

    void updateVelocities(zs::CudaExecutionPolicy &pol);
    void writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol);

    /// collision
    void markSelfIntersectionPrimitives(zs::CudaExecutionPolicy &pol);
    void findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat);
    void findCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, bool withBoundary);
    void findBoundaryCellCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat);

    /// pipeline
    void advanceSubstep(zs::CudaExecutionPolicy &pol, T ratio);
    void newtonKrylov(zs::CudaExecutionPolicy &pol);
    void computeInertialAndGravityGradientAndHessian(zs::CudaExecutionPolicy &cudaPol);
    void computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol);
    void computeCollisionGradientAndHessian(zs::CudaExecutionPolicy &cudaPol);

    // constraint
    void computeConstraints(zs::CudaExecutionPolicy &pol);
    bool areConstraintsSatisfied(zs::CudaExecutionPolicy &pol);
    T constraintResidual(zs::CudaExecutionPolicy &pol);

    /// linear solve
    T dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0, const zs::SmallString tag1);
    T infNorm(zs::CudaExecutionPolicy &pol);
    void project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag, const zs::SmallString dstTag);
    void multiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag);
    void cgsolve(zs::CudaExecutionPolicy &cudaPol);

    // contacts
    auto getCnts() const {
        return zs::make_tuple(nPP.getVal(), nPE.getVal(), nPT.getVal(), nEE.getVal(), ncsPT.getVal(), ncsEE.getVal());
    }
    auto getCollisionCnts() const {
        return zs::make_tuple(ncsPT.getVal(), ncsEE.getVal());
    }

    // sim params
    int substep = -1;
    std::size_t estNumCps = 1000000;

    T pnRel = 1e-2;
    T cgRel = 1e-2;
    int PNCap = 1000;
    int CGCap = 500;
    T armijoParam = 1e-4;
    T targetGRes = 1e-2;

    bool enableContact = true;
    bool enableContactEE = true;
    bool enableContactSelf = true;
    bool projectDBC = false;
    T augLagCoeff = 1e4;
    T dHat = 0.0025;
    vec3 extAccel;

    T boxDiagSize2 = 0;
    T avgNodeMass = 0;
    T maxMu, maxLam;

    /// @brief for cloth collision handling
    /// @note all length in mm unit
    /// @note all vertex pair constraints (edge or not) are stored in <PP, nPP, tempPP>

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
    T epsSlack = 21.75;
    /// @brief success condition constant (avoid near boundary numerical issues after the soft phase)
    T epsCond = 0.01;
    /// @brief initial displacement limit during the start of K iteration collision steps
    T D = 0.25;
    /// @brief coupling coefficients between cloth dynamics and collision dynamics
    T sigma = 80000; // s^{-2}
    /// @brief hard phase termination criteria
    T yita = 0.1;
    /// @brief counts: K [iterative steps], ISoft [soft phase steps], IHard [hard phase steps], IInit [x0 initialization]
    int K = 72, ISoft = 10 /*6~16*/, IHard = 8, IInit = 6;
    /// @brief counts: R [rollback steps for reduction], IDyn [cloth dynamics iters]
    int IDyn = 3 /*1~6*/, R = 8;

    ///
    /// initialization
    /// cloth step
    /// collision step
    ///     soft phase
    ///     hard phase

    T proximityRadius() const {
        return std::sqrt((B + Btight) * (B + Btight) + epsSlack);
    }

    ///

    //
    std::vector<PrimitiveHandle> prims;
    std::vector<PrimitiveHandle> auxPrims; // intended for hard constraint (tracker primitive)
    Ti coOffset, numDofs, numBouDofs;
    Ti sfOffset, seOffset, svOffset;

    // (scripted) collision objects
    /// @note allow (bisector) normal update on-the-fly, thus made modifiable
    tiles_t *coVerts, *coEdges, *coEles;

    tiles_t vtemp;      // solver data
    zs::Vector<T> temp; // as temporary buffer

    // self contacts
    zs::Vector<pair_t> PP;
    zs::Vector<int> nPP;
    tiles_t tempPP;
    zs::Vector<pair3_t> PE;
    zs::Vector<int> nPE;
    tiles_t tempPE;
    zs::Vector<pair4_t> PT;
    zs::Vector<int> nPT;
    tiles_t tempPT;
    zs::Vector<pair4_t> EE;
    zs::Vector<int> nEE;
    tiles_t tempEE;

    zs::Vector<zs::u8> exclSes, exclSts, exclBouSes, exclBouSts; // mark exclusion

    zs::Vector<pair4_t> csPT, csEE;
    zs::Vector<int> ncsPT, ncsEE;
    // end contacts

    // possibly accessed in compactHessian and cgsolve
    CsrMatrix<zs::vec<T, 3, 3>, int> linMat; // sparsity pattern update during hessian computation

    // boundary contacts
    // auxiliary data (spatial acceleration)
    tiles_t stInds, seInds, svInds;
    bvh_t stBvh, seBvh;       // for simulated objects
    bvh_t bouStBvh, bouSeBvh; // for collision objects
    bvfront_t selfStFront, boundaryStFront;
    bvfront_t selfSeFront, boundarySeFront;
    bool frontManageRequired;
    T dt, framedt, curRatio;
};

} // namespace zeno

#include "SolverUtils.cuh"