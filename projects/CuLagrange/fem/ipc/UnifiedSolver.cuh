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
#include "../SolverUtils.cuh"
#endif

#define USE_MAS 0
#define ENABLE_STQ 0

namespace zeno {

struct UnifiedIPCSystem : IObject {
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
    using bvs_t = zs::LBvs<3, int, T>;
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
        int getPrincipalAxis() {
            int axis = 0;
            T extent = bv._max[0] - bv._min[0];
            for (int d = 1; d != 3; ++d) {
                if (auto ext = bv._max[d] - bv._min[d]; ext > extent) {
                    extent = ext;
                    axis = d;
                }
            }
            return axis;
        }

        std::shared_ptr<ZenoParticles> zsprimPtr{}; // nullptr if it is an auxiliary
        std::shared_ptr<const ZenoConstitutiveModel> modelsPtr;
        std::shared_ptr<ZenoParticles::dtiles_t> vertsPtr;
        std::shared_ptr<ZenoParticles::particles_t> elesPtr;
        std::shared_ptr<ZenoParticles::particles_t> bendingEdgesPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfTrisPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfEdgesPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfVertsPtr;
        typename ZenoParticles::dtiles_t svtemp;
        const std::size_t vOffset, sfOffset, seOffset, svOffset;
        ZenoParticles::category_e category;
        ZenoParticles::bv_t bv;
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
    UnifiedIPCSystem(std::vector<ZenoParticles *> zsprims, const dtiles_t *coVerts, const tiles_t *coLowResVerts,
                     const tiles_t *coEdges, const tiles_t *coEles, T dt, std::size_t ncps, bool withGround,
                     bool withContact, bool withMollification, T augLagCoeff, T pnRel, T cgRel, int PNCap, int CGCap,
                     int CCDCap, T kappa0, T fricMu, T dHat, T epsv, zeno::vec3f gn, T gravity);

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
        return zs::make_tuple(PP.getCount(), PE.getCount(), PT.getCount(), EE.getCount(), PPM.getCount(),
                              PEM.getCount(), EEM.getCount(), csPT.getCount(), csEE.getCount());
    }
    auto getCollisionCnts() const {
        return zs::make_tuple(csPT.getCount(), csEE.getCount());
    }
    void findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, T xi = 0);
    void findCollisionConstraints2(zs::CudaExecutionPolicy &pol, T dHat, T xi = 0);
    void findCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary = false);
    void findCollisionConstraintsImpl2(zs::CudaExecutionPolicy &pol, T dHat, T xi, bool withBoundary = false);
    void findBoundaryCollisionConstraintsImpl(zs::CudaExecutionPolicy &pol, T dHat, T xi);
    void findBoundaryCollisionConstraintsImpl2(zs::CudaExecutionPolicy &pol, T dHat, T xi);
    void precomputeFrictions(zs::CudaExecutionPolicy &pol, T dHat, T xi = 0); // called per optimization
    void findCCDConstraints(zs::CudaExecutionPolicy &pol, T alpha, T xi = 0);
    void findCCDConstraintsImpl(zs::CudaExecutionPolicy &pol, T alpha, T xi, bool withBoundary = false);
#if ENABLE_STQ
    void findCCDConstraintsImplEE(zs::CudaExecutionPolicy &pol, T alpha, T xi);
#endif
    void findBoundaryCCDConstraintsImpl(zs::CudaExecutionPolicy &pol, T alpha, T xi);
    // linear system setup
    void computeInertialPotentialGradient(zs::CudaExecutionPolicy &cudaPol,
                                          const zs::SmallString &gTag); // for kappaMin
    void computeElasticGradient(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag);
    void computeBoundaryBarrierGradient(zs::CudaExecutionPolicy &pol);
    void computeBarrierGradient(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag);
    void computeFrictionBarrierGradient(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag);

    /// @note build linsys.spmat
    void initializeSystemHessian(zs::CudaExecutionPolicy &pol);
    // elasticity, bending, kinematic, external force potential, boundary motion, ground collision
    void updateInherentGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag);
    // mostly collision (non-ground) related
    void updateBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag);
    void updateFrictionBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag);
    void updateDynamicGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &gTag);
    void prepareDiagonalPreconditioner(zs::CudaExecutionPolicy &pol);

    // krylov solver
    T infNorm(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag = "dir");
    T dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0, const zs::SmallString tag1);
    void project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag, const zs::SmallString dstTag);

    void systemMultiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag);

    void systemSolve(zs::CudaExecutionPolicy &cudaPol);

    void groundIntersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T &stepSize);
    void intersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T xi, T &stepSize);
    T collisionEnergy(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    T energy(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void lineSearch(zs::CudaExecutionPolicy &cudaPol, T &alpha);

    // sim params
    std::size_t estNumCps = 100000;
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

    /// @brief collision
    template <typename ValT>
    struct DynamicBuffer {
        DynamicBuffer(std::size_t n = 0)
            : buf{n, zs::memsrc_e::device}, cnt{1, zs::memsrc_e::device}, prevCount{0} {
            reset();
        }
        ~DynamicBuffer() = default;

        std::size_t getCount() const {
            return cnt.getVal();
        }
        std::size_t getBufferCapacity() const {
            return buf.capacity();
        }
        ///
        void snapshot() {
            prevCount = cnt.size();
        }
        void resizeToCounter() {
            auto c = getCount();
            if (c <= getBufferCapacity())
                return;
            buf.resize(c);
        }
        void restartCounter() {
            cnt.setVal(prevCount);
        }
        void reset() {
            cnt.setVal(0);
        }
        ///
        void assignCounterFrom(const DynamicBuffer &o) {
            cnt = o.cnt;
        }
        ///
        int reserveFor(int inc) {
            int v = cnt.getVal();
            buf.resize((std::size_t)(v + inc));
            return v;
        }

        struct Port {
            ValT *buf;
            int *cnt;
            std::size_t cap;
            __forceinline__ __device__ void try_push(ValT &&val) {
                auto no = zs::atomic_add(zs::exec_cuda, cnt, 1);
                if (no < cap)
                    buf[no] = std::move(val);
            }
            __forceinline__ __device__ int next_index(zs::cg::thread_block_tile<8, zs::cg::thread_block> &tile) {
                int no = -1;
                if (tile.thread_rank() == 0)
                    no = zs::atomic_add(zs::exec_cuda, cnt, 1);
                no = tile.shfl(no, 0);
                return (no < cap ? no : -1);
            }
            __forceinline__ __device__ ValT &operator[](int i) {
                return buf[i];
            }
            __forceinline__ __device__ const ValT &operator[](int i) const {
                return buf[i];
            }
        };
        Port port() {
            return Port{buf.data(), cnt.data(), buf.capacity()}; // numDofs, tag;
        }

      protected:
        zs::Vector<ValT> buf;
        zs::Vector<int> cnt;
        std::size_t prevCount;
    };
    template <typename... DynBufs>
    void snapshot(DynBufs &...dynBufs) {
        (void)((void)dynBufs.snapshot(), ...);
    }
    template <typename... DynBufs>
    bool allFit(DynBufs &...dynBufs) {
        return ((dynBufs.getCount() <= dynBufs.getBufferCapacity()) && ...);
    }
    template <typename... DynBufs>
    void resizeAndRewind(DynBufs &...dynBufs) {
        (void)((void)dynBufs.resizeToCounter(), ...);
        (void)((void)dynBufs.restartCounter(), ...);
    }
    // self contacts
    DynamicBuffer<pair_t> PP;
    DynamicBuffer<pair3_t> PE;
    DynamicBuffer<pair4_t> PT;
    DynamicBuffer<pair4_t> EE;
    // mollifier
    DynamicBuffer<pair4_t> PPM;
    DynamicBuffer<pair4_t> PEM;
    DynamicBuffer<pair4_t> EEM;
    /// @brief friction
    DynamicBuffer<pair_t> FPP;
    dtiles_t fricPP;
    DynamicBuffer<pair3_t> FPE;
    dtiles_t fricPE;
    DynamicBuffer<pair4_t> FPT;
    dtiles_t fricPT;
    DynamicBuffer<pair4_t> FEE;
    dtiles_t fricEE;

    zs::Vector<zs::u8> exclDofs;                                 // mark dof collision exclusion
    zs::Vector<zs::u8> exclSes, exclSts, exclBouSes, exclBouSts; // mark exclusion
    // end contacts

    zs::Vector<T> temp;   // generally 64-bit
    zs::Vector<bv_t> bvs; // as temporary buffer

    DynamicBuffer<pair4_t> csPT, csEE;

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
        using pair_t = pair_t;
        using spmat_t = zs::SparseMatrix<mat3, true>;
        using dyn_hess_t = zs::tuple<pair_t, mat3>;

        /// @brief dynamic part, mainly for collision constraints
        bool initialized = false;

        // using hess2_t = HessianPiece<2, T>;
        // using hess3_t = HessianPiece<3, T>;
        // using hess4_t = HessianPiece<4, T>;
        /// @note initialization: hess.init(allocator, size)
        /// @note maintain: hess.reset(false, 0)    ->  hess.increaseCount(size)    ->  hess.hess/hess.inds
        // HessianPiece<2, T> hess2;
        // HessianPiece<3, T> hess3;
        // HessianPiece<4, T> hess4;

        DynamicBuffer<dyn_hess_t> dynHess;
        /// @brief inherent part
        spmat_t spmat{};     // _ptrs, _inds, _vals
        spmat_t neighbors{}; // _ptrs, _inds, _vals
        RM_CVREF_T(neighbors._inds) neighborInds;
        /// @brief preconditioner
        int nLevels;
        int nTotalEntries;
        zs::Vector<zs::vec<T, 96, 96>> Pm, inversePm;
        zs::Vector<zs::vec<T, 3>> Rm, Zm;

        int totalNodes;
        int levelnum;
        int totalNumberClusters;
        zs::vec<int, 2> h_clevelSize;

        zs::Vector<zs::vec<int, 2>> d_levelSize;
        zs::Vector<int> d_coarseSpaceTables;
        zs::Vector<int> d_prefixOriginal;
        zs::Vector<int> d_prefixSumOriginal;
        zs::Vector<int> d_goingNext;
        zs::Vector<int> d_denseLevel;
        zs::Vector<zs::vec<int, 4>> d_coarseTable;

        zs::Vector<unsigned int> d_fineConnectMask;
        zs::Vector<unsigned int> d_nextConnectMask;
        zs::Vector<unsigned int> d_nextPrefix;
        zs::Vector<unsigned int> d_nextPrefixSum;

        RM_CVREF_T(spmat._ptrs) traversed;

        void initializePreconditioner(zs::CudaExecutionPolicy &pol, UnifiedIPCSystem &system);
        int buildPreconditioner(zs::CudaExecutionPolicy &pol, UnifiedIPCSystem &system);
        void precondition(zs::CudaExecutionPolicy &pol, dtiles_t &vtemp, const zs::SmallString srcTag,
                          const zs::SmallString dstTag);

        void computeNumLevels(int vertNum);
        //void initPreconditioner(int vertNum, int totalNeighborNum, int4* m_collisonPairs);

        void BuildConnectMaskL0();
        void PreparePrefixSumL0();

        void BuildLevel1();
        void BuildConnectMaskLx(int level);
        void NextLevelCluster(int level);
        void PrefixSumLx(int level);
        void ComputeNextLevel(int level);
        void AggregationKernel();

        int ReorderRealtime(zs::CudaExecutionPolicy &pol, int dynNum);
        void PrepareHessian();

        void BuildCollisionConnection(zs::CudaExecutionPolicy &pol, zs::Vector<unsigned int> &connectionMsk,
                                      zs::Vector<int> &coarseTableSpace, int level);

        void BuildMultiLevelR(const double3 *R);
        void SchwarzLocalXSym();

        void CollectFinalZ(double3 *Z);
    };
    template <zs::execspace_e space, typename T_>
    struct SystemHessianView {
        using sys_hess_t = SystemHessian<T_>;
        using vec3 = typename sys_hess_t::vec3;
        using mat3 = typename sys_hess_t::mat3;
        using pair_t = typename sys_hess_t::pair_t;
        using spmat_t = typename sys_hess_t::spmat_t;
        using dyn_hess_t = typename sys_hess_t::dyn_hess_t;
        using dyn_buffer_t = DynamicBuffer<dyn_hess_t>;

        using spmat_view_t = RM_CVREF_T(zs::view<space>(std::declval<spmat_t &>(), zs::true_c));
        using dyn_buffer_view_t = RM_CVREF_T(std::declval<dyn_buffer_t &>().port());

        SystemHessianView(sys_hess_t &sys)
            : spmat{zs::view<space>(sys.spmat, zs::true_c)}, dynHess{sys.dynHess.port()} {
        }

        spmat_view_t spmat;
        dyn_buffer_view_t dynHess;
    };
    template <zs::execspace_e space, typename T_>
    auto port(SystemHessian<T_> &sys) {
        return SystemHessianView<space, T_>{sys};
    }
    // for one-time static hessian topo build
    SystemHessian<T> linsys;

    // boundary contacts
    // auxiliary data (spatial acceleration)
    tiles_t stInds, seInds, svInds;
    bvh_t stBvh, seBvh;       // for simulated objects
    bvh_t bouStBvh, bouSeBvh; // for collision objects

#if ENABLE_STQ
    int principalAxis;
    bvs_t stBvs, seBvs;
    bvs_t bouStBvs, bouSeBvs;
#endif

    std::optional<bv_t> wholeBv;
    T dt, framedt;
};

} // namespace zeno
