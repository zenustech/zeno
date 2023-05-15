#pragma once
#include "SpatialAccel.cuh"
#include "Structures.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvtt.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
namespace zeno {
struct RapidClothSystem : IObject {
    using T = float;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    constexpr static auto eps_c = zs::limits<T>::epsilon() * 1.0f; 
    constexpr static auto T_c = zs::float_c; 
    constexpr static auto enablePE_c = false; 
    constexpr static auto enablePP_c = false;
    constexpr static auto debugVis_c = true; 
    constexpr static auto enableProfile_c = false; 
    constexpr static auto showStatistics_c = false; 
    constexpr static auto silentMode_c = true; 
    T tinyDist = 1e-3; 
    T repulsionCoef = 1.f; 
    T repulsionRange = 2.f; 
    bool enableDegeneratedDist = true; 
    bool enableRepulsion = false; 
    bool enableDistConstraint = true; 
    bool enableFriction = false;
    bool enableSL = false; 
    T clothFricMu = 0.1f;
    T boundaryFricMu = 10.0f;   

    using primptr_t = typename std::shared_ptr<PrimitiveObject>; 
    using tiles_t = typename ZenoParticles::particles_t;
    using i2tab_t = typename zs::bht<int, 2, int>; 
    using itiles_t = zs::TileVector<int, 32>; 
    using vec3 = zs::vec<T, 3>;
    using vec3f = zs::vec<float, 3>;
    using ivec3 = zs::vec<int, 3>;
    using ivec2 = zs::vec<int, 2>;
    using bvec3 = zs::vec<bool, 3>; 
    using bvec4 = zs::vec<bool, 4>; 
    using mat2 = zs::vec<T, 2, 2>;
    using mat3 = zs::vec<T, 3, 3>;
    using pair_t = zs::vec<int, 2>;
    using pair3_t = zs::vec<int, 3>;
    using pair4_t = zs::vec<int, 4>;
    using bvh_t = zs::LBvh<3, int, T>;
    using bvfront_t = zs::BvttFront<int, int>;
    using spmat_t = zs::SparseMatrix<T, true>;
    using ispmat_t = zs::SparseMatrix<zs::u32, true>; 
    using bv_t = typename bvh_t::Box;

    static constexpr T s_constraint_residual = 1e-3;
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

    /// init 
    void initialize(zs::CudaExecutionPolicy &pol);
    // assume ncps < 6e5, normal choice: ncps = 1e5
    RapidClothSystem(std::vector<ZenoParticles *> zsprims, tiles_t *coVerts, tiles_t *coPoints, tiles_t *coEdges,
                    tiles_t *coEles, T dt, std::size_t spmatCps, std::size_t ncps, std::size_t bvhFrontCps, bool withContact, T augLagCoeff, T cgRel, T lcpTol, 
                    int PNCap, int CGCap, int lcpCap, T gravity, int L, T delta, T sigma, bool enableSL, T gamma, T eps, int maxVertCons, 
                    T BCStiffness, bool enableExclEdges, T repulsionCoef, bool enableDegeneratedDist, bool enableDistConstraint, 
                    T repulsionRange, T tinyDist, bool enableFric, float clothFricMu, float boundaryFricMu); 

    /// @note initialize "ws" (mass), "yn", "vn" properties
    void reinitialize(zs::CudaExecutionPolicy &pol, T framedt);
    void updateVelocities(zs::CudaExecutionPolicy &pol);
    void writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol);

    /// collision; TODO
    void consColoring(zs::CudaExecutionPolicy &pol);   
    bool checkConsColoring(zs::CudaExecutionPolicy &pol); 
    void findConstraintsImpl(zs::CudaExecutionPolicy &pol, T radius, bool withBoundary, const zs::SmallString &tag); 
    void findConstraints(zs::CudaExecutionPolicy &pol, T dist, const zs::SmallString &tag = "x(l)");
    void computeConstraints(zs::CudaExecutionPolicy &pol, const zs::SmallString& tag, T shrinking = 1.1f); // xl, cons -> c(xl), J(xl)     
    void solveLCP(zs::CudaExecutionPolicy &pol);        // yl, y[k], (c, J), xl -> lambda_{l+1}, y_{l+1} 
    void backwardStep(zs::CudaExecutionPolicy &pol);    // call cons + solveLCP 
    void forwardStep(zs::CudaExecutionPolicy &pol);     // async stepping  

    /// pipeline; TODO
    void advanceSubstep(zs::CudaExecutionPolicy &pol, T ratio);
    void subStepping(zs::CudaExecutionPolicy &pol);
    void subSteppingWithNewton(zs::CudaExecutionPolicy &pol); 
    void subSteppingWithGD(zs::CudaExecutionPolicy &pol); 

    // dynamics
    void computeInertialAndForceGradient(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &tag);
    void computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &tag);
    void computeRepulsionGradientAndHessian(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &tag); 

    /// linear solve
    T dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &tag0, const zs::SmallString &tag1, std::size_t n);
    template<class ValT, class tvT>
    ValT tvMax(zs::CudaExecutionPolicy &cudaPol, const tvT& tv, const zs::SmallString& tag, 
        std::size_t n, zs::wrapt<ValT> valWrapT = {}); 
    template<class ValT, class tvT>
    ValT tvMin(zs::CudaExecutionPolicy &cudaPol, const tvT& tv, const zs::SmallString& tag, 
        std::size_t n, zs::wrapt<ValT> valWrapT = {}); 
    template <int codim = 3>
    T infNorm(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag, std::size_t n, zs::wrapv<codim> = {});
    T l2Norm(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag, std::size_t n);
    void project(zs::CudaExecutionPolicy &pol, const zs::SmallString tag);
    void precondition(zs::CudaExecutionPolicy &pol, const zs::SmallString srcTag, const zs::SmallString dstTag);
    void multiply(zs::CudaExecutionPolicy &pol, const zs::SmallString dxTag, const zs::SmallString bTag);
    void cgsolve(zs::CudaExecutionPolicy &cudaPol);
    void newtonDynamicsStep(zs::CudaExecutionPolicy &pol);
    void gdDynamicsStep(zs::CudaExecutionPolicy &pol);
    T dynamicsEnergy(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag);

    // contacts
    void updateConstraintCnt() {
        std::tie(npp, npe, npt, nee, ne) = 
            std::make_tuple(nPP.getVal(), nPE.getVal(), nPT.getVal(), nEE.getVal(), nE.getVal());
    }

    // sim params
    bool enableExclEdges = false; 
    int substep = -1;
    std::size_t spmatCps = 1000000; 
    std::size_t estNumCps = 100000;
    std::size_t bvhFrontCps = 10000000; 
    T cgRel = 1e-2;
    int PNCap = 5;
    int CGCap = 250;
    T armijoParam = 1e-4;
    bool enableContact = true;
    bool enableContactSelf = true;
    T augLagCoeff = 1e4;
    vec3 gravAccel;

    // collision params 
    T avgNodeMass = 0; 
    T D_min, D_max, D, delta, sigma, gamma, eps;
    int L;

    // Chebyshev params
    T chebyOmega = 1.0f;
    T chebyRho = 0.99f;
    T chebyStep = 0.3f; // step size
    T chebyStepMin = 0.10f;
    T chebyStepDec = 0.7f;

    // geometries
    std::vector<PrimitiveHandle> prims;
    Ti coOffset, numDofs, numBouDofs;
    Ti sfOffset, seOffset, svOffset;
    // (scripted) collision objects
    /// @note allow (bisector) normal update on-the-fly, thus made modifiable
    tiles_t *coVerts, *coPoints, *coEdges, *coEles;

    // buffer
    itiles_t vCons;          // vertex -> constraint indices & constraints num 
    tiles_t vtemp;          // solver data
    zs::Vector<int> itemp; 
    zs::Vector<T> temp;     // as temporary buffer
    zs::Vector<bv_t> bvs;   // as temporary buffer

    // collision constraints (edge / non-edge)
    int lcpCap = 256; 
    T lcpTol = 1e-3; 
    zs::Vector<int> lcpConverged; 
    int maxVertCons = 32;
    int nConsColor = 0; 
    int nCons = 0; 
    int consDegree = 32 * 3;
    // TODO: make it an option
    i2tab_t exclTab; 
    spmat_t lcpMat; 
    ispmat_t lcpTopMat; 
    zs::Vector<int> lcpMatIs, lcpMatJs; 
    zs::Vector<int> lcpMatSize; 
    zs::Vector<zs::u32> colorMinWeights, colorWeights;
    zs::Vector<int> colorMaskOut, colors;
    itiles_t tempCons;       // LCP constraint storage
    tiles_t tempPP, tempPE, tempPT, tempEE, tempE; 
    zs::Vector<int> oPP, oPE, oPT, oEE, oE; 
    zs::Vector<int> nPP, nPE, nPT, nEE, nE;
    int opp, ope, opt, oee, oe;     // offsets
    int napp, nape, napt, naee, nae; 
    int npp, npe, npt, nee, ne;

    // auxiliary data (spatial acceleration)
    tiles_t svInds, seInds, stInds, spInds;
    bvh_t svBvh, stBvh, seBvh;    // for simulated objects // TODO: all svBvh -> stBvh & seBvh
    bvh_t bouStBvh, bouSeBvh; // for collision objects
    bvfront_t selfStFront, boundaryStFront;
    bvfront_t selfSeeFront, boundarySeeFront;
    bvfront_t selfSevFront, boundarySevFront;
    bvfront_t selfSvFront, boundarySvFront; 
    bool frontManageRequired; 
    T dt, framedt, curRatio;
    // boundary condition param 
    T BCStiffness = 1e6f; 

    //debug 
    primptr_t visPrim; 
};

    template <
        typename VecTA, typename VecTB,
        zs::enable_if_all<VecTA::dim == 1, zs::is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
    constexpr auto safe_dist2_ee(const zs::VecInterface<VecTA> &ea0, const zs::VecInterface<VecTA> &ea1,
                            const zs::VecInterface<VecTB> &eb0, const zs::VecInterface<VecTB> &eb1) noexcept {
        using T = zs::math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
        auto b = (ea1 - ea0).cross(eb1 - eb0);
        auto b2 = b.l2NormSqr(); 
        if (b2 < zs::limits<T>::epsilon() * 10.0f) // PE
            if (auto aLen2 = (ea0 - ea1).l2NormSqr(), bLen2 = (eb0 - eb1).l2NormSqr(); aLen2 < bLen2)
                return (ea0 - eb0).cross(ea0 - eb1).l2NormSqr() / bLen2;  
            else 
                return (eb0 - ea0).cross(eb0 - ea1).l2NormSqr() / aLen2; 
        T aTb = (eb0 - ea0).dot(b);
        return aTb * aTb / b.l2NormSqr();
    }

    template <
        typename VecTA, typename VecTB,
        zs::enable_if_all<VecTA::dim == 1, zs::is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
    constexpr auto safe_edge_edge_closest_point(const zs::VecInterface<VecTA> &ei0, const zs::VecInterface<VecTA> &ei1,
                            const zs::VecInterface<VecTB> &ej0, const zs::VecInterface<VecTB> &ej1) noexcept {
        using T = zs::math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
        constexpr auto eps_c = zs::limits<T>::epsilon(); 
        auto gammas = edge_edge_closest_point(ei0, ei1, ej0, ej1);
        if ((ei1 - ei0).cross(ej1 - ej0).l2NormSqr() / ((ei1 - ei0).l2NormSqr() * (ej1 - ej0).l2NormSqr() + eps_c) < eps_c)
        {
            // gammas = zs::vec<T, 2>{0.5f, 0.5f};
            auto ejProj = (ej1 - ej0).dot(ei1 - ei0); 
            auto ej0Proj = (ej0 - ei0).dot(ei1 - ei0); 
            auto ej1Proj = (ej1 - ei0).dot(ei1 - ei0); 
            auto ei2 = (ei1 - ei0).l2NormSqr(); 
            if (ej0Proj < 0 && ej1Proj < 0)
            {
                gammas[0] = 0; 
                if (ejProj > 0)
                    gammas[1] = 1; 
                else
                    gammas[1] = 0; 
            } else if (ej0Proj > ei2 && ej1Proj > ei2)
            {
                gammas[0] = 1; 
                if (ejProj > 0)
                    gammas[1] = 0;
                else 
                    gammas[1] = 1; 
            } else {
                auto minProj = zs::max(zs::min(ej0Proj, ej1Proj) / (ei2 + eps_c), 0.f); 
                auto maxProj = zs::min(zs::max(ej0Proj, ej1Proj) / (ei2 + eps_c), 1.f); 
                gammas[0] = (minProj + maxProj) * 0.5f; 
                gammas[1] = zs::min(zs::max((gammas[0] * ei2 - ej0Proj) / (ejProj + eps_c), 0.f), 1.f); 
            }
        }
        return gammas; 
    }

    template <
        typename VecTA, typename VecTB,
        zs::enable_if_all<VecTA::dim == 1, zs::is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
    constexpr auto safe_dist2_ee_unclassified(const zs::VecInterface<VecTA> &ea0,
                                        const zs::VecInterface<VecTA> &ea1,
                                        const zs::VecInterface<VecTB> &eb0,
                                        const zs::VecInterface<VecTB> &eb1) noexcept {
    using T = zs::math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
    T dist2{zs::limits<T>::max()};
    switch (ee_distance_type(ea0, ea1, eb0, eb1)) {
        case 0:
        dist2 = dist2_pp(ea0, eb0);
        break;
        case 1:
        dist2 = dist2_pp(ea0, eb1);
        break;
        case 2:
        dist2 = dist2_pe(ea0, eb0, eb1);
        break;
        case 3:
        dist2 = dist2_pp(ea1, eb0);
        break;
        case 4:
        dist2 = dist2_pp(ea1, eb1);
        break;
        case 5:
        dist2 = dist2_pe(ea1, eb0, eb1);
        break;
        case 6:
        dist2 = dist2_pe(eb0, ea0, ea1);
        break;
        case 7:
        dist2 = dist2_pe(eb1, ea0, ea1);
        break;
        case 8:
        dist2 = safe_dist2_ee(ea0, ea1, eb0, eb1);
        break;
        default:
        break;
    }
    return dist2;
    }

} // namespace zeno

#include "../SolverUtils.cuh"