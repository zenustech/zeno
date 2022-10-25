#pragma once
#include "Structures.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/Vec.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

struct PBDSystem : IObject {
    using T = float;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using dtiles_t = zs::TileVector<T, 32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;
    using ivec3 = zs::vec<int, 3>;
    using ivec2 = zs::vec<int, 2>;
    using bvh_t = zs::LBvh<3, int, T>;
    using bv_t = zs::AABBBox<3, T>;

    struct PrimitiveHandle {
        PrimitiveHandle(ZenoParticles &zsprim, std::size_t &vOffset, std::size_t &sfOffset, std::size_t &seOffset,
                        std::size_t &svOffset, zs::wrapv<4>);

        void initGeo();

        auto getModelLameParams() const {
            T mu, lam;
            zs::match([&](const auto &model) {
                mu = model.mu;
                lam = model.lam;
            })(models.getElasticModel());
            return zs::make_tuple(mu, lam);
        }

        decltype(auto) getVerts() const {
            return *vertsPtr;
        }
        decltype(auto) getEles() const {
            return *elesPtr;
        }
        decltype(auto) getEdges() const {
            return *edgesPtr;
        }
        decltype(auto) getSurfTris() const {
            return *surfTrisPtr;
        }
        decltype(auto) getSurfEdges() const {
            return *surfEdgesPtr;
        }
        decltype(auto) getSurfVerts() const {
            return *surfVertsPtr;
        }
        bool isBoundary() const noexcept {
            return zsprimPtr->asBoundary;
        }

        std::shared_ptr<ZenoParticles> zsprimPtr{}; // nullptr if it is an auxiliary object
        const ZenoConstitutiveModel &models;
        std::shared_ptr<dtiles_t> vertsPtr;
        std::shared_ptr<ZenoParticles::particles_t> elesPtr;
        std::shared_ptr<ZenoParticles::particles_t> edgesPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfTrisPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfEdgesPtr;
        std::shared_ptr<ZenoParticles::particles_t> surfVertsPtr;
        // typename ZenoParticles::dtiles_t etemp;
        // typename ZenoParticles::dtiles_t svtemp;
        const std::size_t vOffset, sfOffset, seOffset, svOffset;
        ZenoParticles::category_e category;
    };

    void initialize(zs::CudaExecutionPolicy &pol);
    PBDSystem(std::vector<ZenoParticles *> zsprims, vec3 extForce, T dt, int numSolveIters, T ec, T vc);

    void reinitialize(zs::CudaExecutionPolicy &pol, T framedt);
    void writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol);

    /// pipeline
    void preSolve(zs::CudaExecutionPolicy &pol);
    void solveEdge(zs::CudaExecutionPolicy &pol);
    void solveVolume(zs::CudaExecutionPolicy &pol);
    void postSolve(zs::CudaExecutionPolicy &pol);

    // sim params
    vec3 extForce;
    int solveIterCap = 50;
    T edgeCompliance = 0.001, volumeCompliance = 0.001;

    //
    std::vector<PrimitiveHandle> prims;
    std::size_t coOffset, numDofs;
    std::size_t sfOffset, seOffset, svOffset;

    dtiles_t vtemp;

    zs::Vector<T> temp;

    // boundary contacts
    // auxiliary data (spatial acceleration)
    tiles_t stInds, seInds, svInds;
    // using bvs_t = zs::LBvs<3, int, T>;
    bvh_t stBvh, seBvh; // for simulated objects
    // bvs_t stBvs, seBvs;       // STQ
    // bvh_t bouStBvh, bouSeBvh; // for collision objects
    // bvs_t bouStBvs, bouSeBvs; // STQ
    T dt, framedt;
};
// config compliance, num solve iters, edge/volume extforce, dt

} // namespace zeno