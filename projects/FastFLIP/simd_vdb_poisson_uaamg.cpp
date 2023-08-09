#include "simd_vdb_poisson_uaamg.h"

#include <atomic>
#include <immintrin.h>
#include <unordered_map>

#include "openvdb/tree/LeafManager.h"
#include "openvdb/tools/Interpolation.h"

#include "levelset_util.h"
#include "openvdb_grid_math_op.h"
#include "SIMD_UAAMG_Ops.h"
#include "tbb/concurrent_vector.h"
#include "vdb_SIMD_IO.h"

namespace simd_uaamg {

struct BuildPoissonRhs {
    BuildPoissonRhs(
        openvdb::FloatGrid::Ptr vx,
        openvdb::FloatGrid::Ptr vy,
        openvdb::FloatGrid::Ptr vz,
        openvdb::Vec3fGrid::Ptr weight,
        openvdb::Vec3fGrid::Ptr solidVel,
        const std::vector<openvdb::FloatTree::LeafNodeType*>& rhsLeaves, const float dx, const float dt) :
        mVelAxr{ vx->tree(),  vy->tree(), vz->tree() },
        mWeightAxr(weight->tree()), mSolidVelAxr(solidVel->tree()),
        mRhsLeaves(rhsLeaves), mDx(dx), mDt(dt) {}

    BuildPoissonRhs(const BuildPoissonRhs& other) :
        mVelAxr{ other.mVelAxr[0], other.mVelAxr[1], other.mVelAxr[2] },
        mWeightAxr(other.mWeightAxr), mSolidVelAxr(other.mSolidVelAxr),
        mRhsLeaves(other.mRhsLeaves), mDx(other.mDx), mDt(other.mDt)
    {
        //clear the cache
        for (int i = 0; i < 3; i++) {
            mVelAxr[i].clear();
        }
        mWeightAxr.clear();
        mSolidVelAxr.clear();
    }

    void operator()(const openvdb::Int32Tree::LeafNodeType& dofLeaf, openvdb::Index leafpos)  const {
        float invdx = 1.0f / mDx;
        for (auto iter = dofLeaf.beginValueOn(); iter; ++iter) {
            float rhs = 0, weight_sum = 0;
            openvdb::Coord globalCoord = iter.getCoord();
            bool hasNonZeroWeight = false;
            //x+ x- y+ y- z+ z-
            for (int i = 0; i < 6; i++) {
                int channel = i / 2;
                bool positiveDirection = (i % 2) == 0;

                auto nextCoord = globalCoord;
                nextCoord[channel] += int{ positiveDirection };

                //face weight
                float weight = mWeightAxr.getValue(nextCoord)[channel];
                weight_sum += weight;
                if (weight != 0.f) {
                    hasNonZeroWeight = true;
                }

                //liquid velocity iChannel
                float vel = mVelAxr[channel].getValue(nextCoord);

                //solid velocity iChannel
                float svel = mSolidVelAxr.getValue(nextCoord)[channel];

                //all elements there, write the rhs
                //write to the right hand side part
                if (positiveDirection) {
                    rhs -= invdx * (weight * vel
                        + (1.0f - weight) * svel);
                }
                else {
                    rhs += invdx * (weight * vel
                        + (1.0f - weight) * svel);
                }
            }//end for 6 faces of this voxel

            if (!hasNonZeroWeight || weight_sum < 0.1) {
                rhs = 0;
            }
            mRhsLeaves[leafpos]->setValueOn(iter.offset(), rhs);
        }//end for all dof 
    }//end operator

    mutable openvdb::FloatGrid::ConstUnsafeAccessor mVelAxr[3];
    mutable openvdb::Vec3fGrid::ConstUnsafeAccessor mWeightAxr, mSolidVelAxr;
    const std::vector<openvdb::FloatTree::LeafNodeType*>& mRhsLeaves;
    const float mDx, mDt;
};

struct BuildPoissonRhs_withTension {
    BuildPoissonRhs_withTension(
        openvdb::FloatGrid::Ptr vx,
        openvdb::FloatGrid::Ptr vy,
        openvdb::FloatGrid::Ptr vz,
        openvdb::FloatGrid::Ptr liquid_phi,
        openvdb::FloatGrid::Ptr curvature,
        openvdb::Vec3fGrid::Ptr weight,
        openvdb::Vec3fGrid::Ptr solidVel,
        const std::vector<openvdb::FloatTree::LeafNodeType*>& rhsLeaves,
        const float dx, const float dt, float tension_coef) :
        mVelAxr{ vx->tree(),  vy->tree(), vz->tree() },
        mPhiAxr(liquid_phi->tree()), mCurvAxr(curvature->tree()),
        mWeightAxr(weight->tree()), mSolidVelAxr(solidVel->tree()),
        mRhsLeaves(rhsLeaves),
        mDx(dx), mDt(dt), mTension_coef(tension_coef) {}

    BuildPoissonRhs_withTension(const BuildPoissonRhs_withTension& other) :
        mVelAxr{ other.mVelAxr[0], other.mVelAxr[1], other.mVelAxr[2] },
        mPhiAxr(other.mPhiAxr), mCurvAxr(other.mCurvAxr),
        mWeightAxr(other.mWeightAxr), mSolidVelAxr(other.mSolidVelAxr),
        mRhsLeaves(other.mRhsLeaves),
        mDx(other.mDx), mDt(other.mDt), mTension_coef(other.mTension_coef)
    {
        //clear the cache
        for (int i = 0; i < 3; i++) {
            mVelAxr[i].clear();
        }
        mPhiAxr.clear();
        mCurvAxr.clear();
        mWeightAxr.clear();
        mSolidVelAxr.clear();
    }

    void operator()(const openvdb::Int32Tree::LeafNodeType& dofLeaf, openvdb::Index leafpos)  const {
        float invdx = 1.0f / mDx;
        float dtOverDxSqr = mDt / (mDx * mDx);

        for (auto iter = dofLeaf.beginValueOn(); iter; ++iter) {
            float rhs = 0, weight_sum = 0;
            openvdb::Coord globalCoord = iter.getCoord();
            bool hasNonZeroWeight = false;

            float phiThis = mPhiAxr.getValue(globalCoord);;
            float phiOther = 0.f;

            float curvThis = mCurvAxr.getValue(globalCoord);
            float curvOther = 0.f;

            float weight = 0.f; //face weight

            //x+ x- y+ y- z+ z-
            for (int i = 0; i < 6; i++) {
                int channel = i / 2;
                bool positiveDirection = (i % 2) == 0;

                auto nextCoord = globalCoord;
                auto phiCoord = globalCoord;

                if (positiveDirection) {
                    nextCoord[channel]++;
                    phiCoord[channel]++;
                }
                else {
                    phiCoord[channel]--;
                }//end else positive direction
                weight = mWeightAxr.getValue(nextCoord)[channel];
                phiOther = mPhiAxr.getValue(phiCoord);
                curvOther = mCurvAxr.getValue(phiCoord);

                weight_sum += weight;
                
                if (weight != 0.f) {
                    hasNonZeroWeight = true;
                }

                //liquid velocity iChannel
                float vel = mVelAxr[channel].getValue(nextCoord);

                //solid velocity iChannel
                float svel = mSolidVelAxr.getValue(nextCoord)[channel];

                //all elements there, write the rhs
                //write to the right hand side part
                if (positiveDirection) {
                    rhs -= invdx * (weight * vel
                        + (1.0f - weight) * svel);
                }
                else {
                    rhs += invdx * (weight * vel
                        + (1.0f - weight) * svel);
                }

                //the other cell is an air cell
                if (phiThis < 0.f && phiOther >= 0.f) {
                    float theta = fraction_inside(phiThis, phiOther);
                    if (theta < 0.02f) theta = 0.02f;

                    rhs += dtOverDxSqr*weight*mTension_coef*(theta*curvOther + (1.f - theta)*curvThis)/theta;
                }
            }//end for 6 faces of this voxel

            if (!hasNonZeroWeight || weight_sum < 0.1) {
                rhs = 0;
            }
            mRhsLeaves[leafpos]->setValueOn(iter.offset(), rhs);
        }//end for all dof 
    }//end operator

    mutable openvdb::FloatGrid::ConstUnsafeAccessor mVelAxr[3];
    mutable openvdb::FloatGrid::ConstUnsafeAccessor mPhiAxr, mCurvAxr;
    mutable openvdb::Vec3fGrid::ConstUnsafeAccessor mWeightAxr, mSolidVelAxr;
    const std::vector<openvdb::FloatTree::LeafNodeType*>& mRhsLeaves;
    const float mDx, mDt, mTension_coef;
};

LaplacianWithLevel::Ptr LaplacianWithLevel::createPressurePoissonLaplacian(openvdb::FloatGrid::Ptr liquidPhi, openvdb::Vec3fGrid::Ptr faceWeight, const float dt)
{
    float dx = static_cast<float>(liquidPhi->voxelSize()[0]);
    return std::make_shared<LaplacianWithLevel>(liquidPhi, faceWeight, dt, dx);
}

openvdb::FloatGrid::Ptr LaplacianWithLevel::createPressurePoissonRightHandSide(
    openvdb::Vec3fGrid::Ptr faceWeight,
    openvdb::FloatGrid::Ptr vx, openvdb::FloatGrid::Ptr vy, openvdb::FloatGrid::Ptr vz,
    openvdb::Vec3fGrid::Ptr solidVel, float dt)
{
    
    if (mLevel != 0) {
        return openvdb::FloatGrid::create();
    }

    auto rhs = getZeroVectorGrid();

    std::vector<openvdb::FloatTree::LeafNodeType*> rhsLeaves;
    rhsLeaves.reserve(rhs->tree().leafCount());
    rhs->tree().getNodes(rhsLeaves);

    BuildPoissonRhs op(
        vx,
        vy,
        vz,
        faceWeight,
        solidVel,
        rhsLeaves, mDxThisLevel, dt);

    mDofLeafManager->foreach(op);
    return rhs;
}

openvdb::FloatGrid::Ptr LaplacianWithLevel::createPressurePoissonRightHandSide_withTension(
    openvdb::FloatGrid::Ptr liquid_phi,
    openvdb::FloatGrid::Ptr curvature,
    openvdb::Vec3fGrid::Ptr faceWeight,
    openvdb::FloatGrid::Ptr vx, openvdb::FloatGrid::Ptr vy, openvdb::FloatGrid::Ptr vz,
    openvdb::Vec3fGrid::Ptr solidVel, float dt, float tension_coef)
{
    
    if (mLevel != 0) {
        return openvdb::FloatGrid::create();
    }

    auto rhs = getZeroVectorGrid();

    std::vector<openvdb::FloatTree::LeafNodeType*> rhsLeaves;
    rhsLeaves.reserve(rhs->tree().leafCount());
    rhs->tree().getNodes(rhsLeaves);

    BuildPoissonRhs_withTension op(
        vx,
        vy,
        vz,
        liquid_phi,
        curvature,
        faceWeight,
        solidVel,
        rhsLeaves,
        mDxThisLevel, dt, tension_coef);

    mDofLeafManager->foreach(op);
    return rhs;
}

struct LaplacianTripletsReducer {
    LaplacianTripletsReducer(const LaplacianWithLevel* laplacian,
        const std::vector<openvdb::Int32Tree::LeafNodeType*>& dofLeaves)
        :mLaplacian(laplacian), mDofLeaves(dofLeaves) {
    }

    LaplacianTripletsReducer(const LaplacianTripletsReducer& other, tbb::split)
        :mLaplacian(other.mLaplacian), mDofLeaves(other.mDofLeaves) {
    }

    void operator()(const tbb::blocked_range<size_t>& r) {
        auto dofAxr = mLaplacian->mDofIndex->getConstUnsafeAccessor();
        auto diagonalAxr = mLaplacian->mDiagonal->getConstUnsafeAccessor();
        openvdb::FloatGrid::ConstUnsafeAccessor offDiagonalAxr[3] = {
            mLaplacian->mXEntry->tree(),
            mLaplacian->mYEntry->tree(),
            mLaplacian->mZEntry->tree() };

        //for each dof leaf
        for (size_t i = r.begin(); i != r.end(); ++i) {
            auto& leaf = *mDofLeaves[i];
            for (auto iter = leaf.beginValueOn(); iter; ++iter) {
                const auto globalCoord = iter.getCoord();
                const float diagonal = diagonalAxr.getValue(globalCoord);

                //the diagonal entry
                mTriplets.push_back(
                    Eigen::Triplet<float>(
                        iter.getValue(), iter.getValue(), diagonal));

                for (int iChannel = 0; iChannel < 3; iChannel++)
                {
                    //check lower neighbor
                    auto negCoord = globalCoord;
                    negCoord[iChannel]--;
                    if (dofAxr.isValueOn(negCoord)) {
                        mTriplets.push_back(
                            Eigen::Triplet<float>(
                                iter.getValue(), dofAxr.getValue(negCoord), offDiagonalAxr[iChannel].getValue(globalCoord)));
                    }
                    //check upper neighbor
                    auto posCoord = globalCoord;
                    posCoord[iChannel]++;
                    if (dofAxr.isValueOn(posCoord)) {
                        mTriplets.push_back(
                            Eigen::Triplet<float>(
                                iter.getValue(), dofAxr.getValue(posCoord), offDiagonalAxr[iChannel].getValue(posCoord)));
                    }

                }//end for three direction
            }//end for all active DOF in this leaf
        }//end for leaves
    }//end operator

    void join(const LaplacianTripletsReducer& other) {
        auto originalSize = mTriplets.size();
        auto otherSize = other.mTriplets.size();
        auto newSize = originalSize + otherSize;
        mTriplets.resize(newSize);
        std::copy(other.mTriplets.begin(), other.mTriplets.end(), mTriplets.begin() + originalSize);
    }

    const std::vector<openvdb::Int32Tree::LeafNodeType*>& mDofLeaves;
    const LaplacianWithLevel* mLaplacian;
    std::vector<Eigen::Triplet<float>> mTriplets;
};

void LaplacianWithLevel::getTriplets(std::vector<Eigen::Triplet<float>>& outTriplets) const
{
    std::vector<openvdb::Int32Tree::LeafNodeType*> dofLeaves;
    dofLeaves.reserve(mDofLeafManager->tree().leafCount());
    mDofLeafManager->getNodes(dofLeaves);
    size_t nleaves = dofLeaves.size();

    LaplacianTripletsReducer reducer(this, dofLeaves);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nleaves), reducer);
    outTriplets.swap(reducer.mTriplets);
}

LaplacianWithLevel::LaplacianWithLevel(openvdb::FloatGrid::Ptr liquidPhi,
    openvdb::Vec3fGrid::Ptr faceWeight,
    const float dt,
    const float dx)
{
    mDt = dt;
    mDxThisLevel = dx;
    mLevel = 0;

    float term = mDt / (mDxThisLevel * mDxThisLevel);

    //Note we do not set the background of the diagonal to be zero
    //later when we run the red-black Gauss seidel in simd instruction
    //we need to divide the result by the diagonal term
    //to avoid dividing by zero we set it to be one
    //then the mask of the diagonal term becomes very important
    //It is the only indicator to show this is a degree of freedom
    mDiagonal = openvdb::FloatGrid::create(6 * term);
    mDiagonal->setName("mDiagonal");
    mDiagonal->setTransform(liquidPhi->transformPtr());
    mDiagonal->setTree(std::make_shared<openvdb::FloatTree>(liquidPhi->tree(), 6 * term, openvdb::TopologyCopy()));

    //by default, the solid is sparse
    //so only a fraction of faces have weight<1
    //the default value of the vector form of face weight is 1
    //so the default value for each component is 1, but we directly
    //store the entry here
    mXEntry = openvdb::FloatGrid::create(-term);
    mXEntry->setName("mXEntry");
    mXEntry->setTransform(faceWeight->transform().copy());
    mXEntry->transform().postTranslate(openvdb::Vec3d(-0.5 * mDxThisLevel, 0.0, 0.0));
    mXEntry->setTree(std::make_shared<openvdb::FloatTree>(liquidPhi->tree(), -term, openvdb::TopologyCopy()));

    mYEntry = mXEntry->deepCopy();
    mYEntry->transform().postTranslate(openvdb::Vec3d(0.0, -0.5 * mDxThisLevel, 0.0));
    mYEntry->setName("mYEntry");

    mZEntry = mXEntry->deepCopy();
    mZEntry->transform().postTranslate(openvdb::Vec3d(0.0, 0.0, -0.5 * mDxThisLevel));
    mZEntry->setName("mZEntry");

    mDofIndex = openvdb::Int32Grid::create(-1);
    mDofIndex->setTransform(mDiagonal->transformPtr());
    initializeFinest(liquidPhi, faceWeight);
    initializeApplyOperator();
}

LaplacianWithLevel::LaplacianWithLevel(const LaplacianWithLevel& fineLevel, LaplacianWithLevel::Coarsening)
{
    initializeFromFineLevel(fineLevel);
}

void LaplacianWithLevel::initializeFromFineLevel(const LaplacianWithLevel& fineLevel)
{
    mDt = fineLevel.mDt;
    mDxThisLevel = 2.0f * fineLevel.mDxThisLevel;
    mLevel = fineLevel.mLevel + 1;

    //the laplacian diagonal and face terms is already trimmed
    //hence only the laplacian dof_idx keeps the actual layout
    //of the degree of freedoms
    auto coarseTransform = openvdb::math::Transform::createLinearTransform(mDxThisLevel);
    coarseTransform->postTranslate(openvdb::Vec3d(0.5f * fineLevel.mDxThisLevel));

    mDofIndex = openvdb::Int32Grid::create(-1);
    mDofIndex->setTransform(coarseTransform);

    //reduction touch leaves
    std::vector<openvdb::Int32Tree::LeafNodeType*> fineLevelLeaves;
    auto fineLevelLeafCount = fineLevel.mDofIndex->tree().leafCount();
    fineLevelLeaves.reserve(fineLevelLeafCount);
    fineLevel.mDofIndex->tree().getNodes(fineLevelLeaves);

    simd_uaamg::TouchCoarseLeafReducer leafToucher{ fineLevelLeaves };
    tbb::parallel_reduce(tbb::blocked_range<openvdb::Index32>(0, fineLevelLeafCount, /*grain size*/100), leafToucher);
    mDofIndex->setTree(leafToucher.mCoarseDofGrid->treePtr());

    mDofLeafManager = std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(mDofIndex->tree());

    //piecewise constant interpolation and restriction function
    //coarse voxel =8 fine voxels
    mDofLeafManager->foreach([&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto fineDofAxr{ fineLevel.mDofIndex->getConstUnsafeAccessor() };
        for (auto iter = leaf.beginValueAll(); iter; ++iter) {
            //the global coordinate in the coarse level
            auto coarseGlobalCoord{ iter.getCoord().asVec3i() };
            auto fineBaseCoord = coarseGlobalCoord * 2;

            bool fineDofNotFound = true;
            for (int ii = 0; ii < 2 && fineDofNotFound; ii++) {
                for (int jj = 0; jj < 2 && fineDofNotFound; jj++) {
                    for (int kk = 0; kk < 2 && fineDofNotFound; kk++) {
                        if (fineDofAxr.isValueOn(
                            openvdb::Coord(fineBaseCoord).offsetBy(ii, jj, kk))) {
                            fineDofNotFound = false;
                        }
                    }
                }
            }//for all fine voxels accociated

            if (fineDofNotFound) {
                iter.setValueOff();
            }
            else {
                iter.setValueOn();
            }
        }//end for all voxel in this leaf
        });
    setDofIndex(mDofIndex);

    float dtOverDxSqr = mDt / (mDxThisLevel * mDxThisLevel);
    //set up the full diagonal matrix, full face weight matrix

    mDiagonal = openvdb::FloatGrid::create(6.0f * dtOverDxSqr);
    mDiagonal->setTransform(coarseTransform);
    mDiagonal->setName("mDiagonal_level_" + std::to_string(mLevel));
    mDiagonal->setTree(
        std::make_shared<openvdb::FloatTree>(
            mDofIndex->tree(), 6.0f * dtOverDxSqr, openvdb::TopologyCopy()));

    mXEntry = openvdb::FloatGrid::create(-dtOverDxSqr);
    mXEntry->setName("mXEntry_level_" + std::to_string(mLevel));
    mXEntry->setTransform(mDiagonal->transformPtr());
    mXEntry->setTree(
        std::make_shared<openvdb::FloatTree>(
            mDofIndex->tree(), -dtOverDxSqr, openvdb::TopologyCopy()));

    mYEntry = mXEntry->deepCopy();
    mYEntry->setName("mYEntry_level_" + std::to_string(mLevel));

    mZEntry = mXEntry->deepCopy();
    mZEntry->setName("mZEntry_level_" + std::to_string(mLevel));


    //The prolongation operator is
    //fine = coarse
    //the restriction operator is
    //coarse = 1/8*(eight fine)
    //ideally, if the fine level diagonal is 6
    //then the coarse level diagonal should be 1.5, because dx_c=dx_f*2
    //1. a input coarse voxel turns on up to 8 fine voxels
    //2. each fine voxel scatters to its center 6, and its neighbor -1
    //3. the results are restricted
    //in a 2D example, original weight is 4, after scatter it becomes
    //-------------------------------------
    //|        |        |        |        |
    //|        |   -1   |   -1   |        |
    //|        |        |        |        |
    //-------------------------------------
    //|        |        |        |        |
    //|   -1   |    2   |    2   |   -1   |
    //|        |        |        |        |
    //-------------------------------------
    //|        |        |        |        |
    //|   -1   |    2   |    2   |   -1   |
    //|        |        |        |        |
    //-------------------------------------
    //|        |        |        |        |
    //|        |   -1   |   -1   |        |
    //|        |        |        |        |
    //-------------------------------------
    //
    // after restriction (sum and * 1/4) it becomes
    //-------------------------------------
    //|        |                 |        |
    //|        |      -0.5       |        |
    //|        |                 |        |
    //-------------------------------------
    //|        |                 |        |
    //|        |                 |        |
    //|        |                 |        |
    //|  -0.5  |        2        |  -0.5  |
    //|        |                 |        |
    //|        |                 |        |
    //|        |                 |        |
    //-------------------------------------
    //|        |                 |        |
    //|        |      -0.5       |        |
    //|        |                 |        |
    //-------------------------------------
    //it should additionally multiply by 0.5 
    //to get correct coefficients


    auto setCoarseLevelTerms = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto xAxr{ fineLevel.mXEntry->getConstUnsafeAccessor() };
        auto yAxr{ fineLevel.mYEntry->getConstUnsafeAccessor() };
        auto zAxr{ fineLevel.mZEntry->getConstUnsafeAccessor() };
        auto diagAxr{ fineLevel.mDiagonal->getConstUnsafeAccessor() };
        auto fineDofAxr{ fineLevel.mDofIndex->getConstUnsafeAccessor() };

        auto* diagLeaf = mDiagonal->tree().probeLeaf(leaf.origin());
        auto* xLeaf = mXEntry->tree().probeLeaf(leaf.origin());
        auto* yLeaf = mYEntry->tree().probeLeaf(leaf.origin());
        auto* zLeaf = mZEntry->tree().probeLeaf(leaf.origin());

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            //each voxel only record the flux of its and its below dof
            float diag = 0, x = 0, y = 0, z = 0;
            //the global coordinate in the coarse level
            auto coarseGlobalCoord{ iter.getCoord().asVec3i() };
            auto fineBaseCoord = openvdb::Coord(coarseGlobalCoord * 2);

            //loop over the 8 fine cells 
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        const auto fineCoord = fineBaseCoord.offsetBy(ii, jj, kk);

                        //contribution to diag
                        if (fineDofAxr.isValueOn(fineCoord)) {
                            diag += diagAxr.getValue(fineCoord);

                            //three face terms
                            if (fineDofAxr.isValueOn(fineCoord.offsetBy(-1, 0, 0))) {
                                if (ii == 0) {
                                    //contribute to the neg x term
                                    x += xAxr.getValue(fineCoord);
                                }
                                else {
                                    //this face will decrease two fine dofs
                                    //diagonal terms
                                    diag += 2 * xAxr.getValue(fineCoord);
                                }
                            }//end if x- neib on

                            if (fineDofAxr.isValueOn(fineCoord.offsetBy(0, -1, 0))) {
                                if (jj == 0) {
                                    y += yAxr.getValue(fineCoord);
                                }
                                else {
                                    diag += 2 * yAxr.getValue(fineCoord);
                                }
                            }//end if y- neib on

                            if (fineDofAxr.isValueOn(fineCoord.offsetBy(0, 0, -1))) {
                                if (kk == 0) {
                                    z += zAxr.getValue(fineCoord);
                                }
                                else {
                                    diag += 2 * zAxr.getValue(fineCoord);
                                }
                            }//end if z- neib on
                        }//end if this voxel is on
                    }//end fine kk
                }//end fine jj
            }//end fine ii

            //coefficient R accounts for 1/8
            //Then there is an additional 1/2
            auto offset = iter.offset();
            const float factor = 0.5f * (1.0f / 8.0f);
            diagLeaf->setValueOnly(offset, diag * factor);
            xLeaf->setValueOnly(offset, x * factor);
            yLeaf->setValueOnly(offset, y * factor);
            zLeaf->setValueOnly(offset, z * factor);
        }//end loop over all coarse dofs
    };//end set terms

    mDofLeafManager->foreach(setCoarseLevelTerms);

    trimDefaultNodes();
    initializeApplyOperator();
}

namespace {
struct BuildFinestMatrix {
    BuildFinestMatrix(
        openvdb::FloatGrid::Ptr in_liquid_phi,
        openvdb::Vec3fGrid::Ptr in_face_weights,
        const std::vector<openvdb::FloatTree::LeafNodeType*>& in_phi_leaf,
        const std::vector<openvdb::FloatTree::LeafNodeType*>& in_neg_x,
        const std::vector<openvdb::FloatTree::LeafNodeType*>& in_neg_y,
        const std::vector<openvdb::FloatTree::LeafNodeType*>& in_neg_z, float in_dt, float in_dx) :
        mPhiAxr(in_liquid_phi->tree()), mWeightAxr(in_face_weights->tree()),
        mPhiLeaves(in_phi_leaf), mNegX(in_neg_x), mNegY(in_neg_y), mNegZ(in_neg_z)
    {
        mDt = in_dt;
        mDx = in_dx;
    }

    BuildFinestMatrix(const BuildFinestMatrix& other) :
        mPhiAxr{ other.mPhiAxr }, mWeightAxr(other.mWeightAxr),
        mPhiLeaves(other.mPhiLeaves),
        mNegX(other.mNegX),
        mNegY(other.mNegY),
        mNegZ(other.mNegZ),
        mDt(other.mDt),
        mDx(other.mDx)
    {
        mPhiAxr.clear();
        mWeightAxr.clear();
    }

    void operator ()(openvdb::FloatTree::LeafNodeType& diagleaf, openvdb::Index leafpos) const {
        //in the initial state, the tree contains both phi<0 and phi>0
        //we only take the phi<0
        const auto& phiLeaf = *mPhiLeaves[leafpos];

        //everytime only write to the coefficient on the lower side
        //the positive side will be handled from the dof on the other side
        auto* xLeaf = mNegX[leafpos];
        auto* yLeaf = mNegY[leafpos];
        auto* zLeaf = mNegZ[leafpos];

        float dtOverDxSqr = mDt / (mDx * mDx);
        for (auto phiIter = phiLeaf.cbeginValueOn(); phiIter; ++phiIter) {
            if (phiIter.getValue() < 0.f) {
                //this is a valid dof
                auto globalCoord = phiIter.getCoord();

                float diagonal = 0.f;
                float xyzTerm[3] = { 0.f,0.f,0.f };
                float phiOther = 0.f;
                float weight = 0.f;
                openvdb::Coord weightCoord = globalCoord;
                openvdb::Coord phiCoord = globalCoord;

                //its six faces
                for (auto i_f = 0; i_f < 6; i_f++) {
                    //0,1,2 direction
                    int component = i_f / 2;
                    bool isPositiveDir = (i_f % 2 == 0);

                    //reset the other coordinate
                    weightCoord = globalCoord;
                    phiCoord = globalCoord;

                    if (isPositiveDir) {
                        weightCoord[component]++;
                        weight = mWeightAxr.getValue(weightCoord)[component];

                        phiCoord[component]++;
                        phiOther = mPhiAxr.getValue(phiCoord);
                    }
                    else {
                        weight = mWeightAxr.getValue(weightCoord)[component];
                        phiCoord[component]--;
                        phiOther = mPhiAxr.getValue(phiCoord);
                    }//end else positive direction
                    float term = weight * dtOverDxSqr;
                    //if other cell is a dof
                    if (phiOther < 0) {
                        diagonal += term;
                        //write to the negative term
                        if (!isPositiveDir) {
                            xyzTerm[component] = -term;
                        }
                    }
                    else {
                        //the other cell is an air cell
                        float theta = fraction_inside(phiIter.getValue(), phiOther);
                        if (theta < 0.02f) theta = 0.02f;
                        diagonal += term / theta;
                    }//end else other cell is dof
                }//end for 6 faces

                //for totally isolated DOF, do not include in the solving process.
                if (diagonal != 0.f) {
                    diagleaf.setValueOn(phiIter.offset(), diagonal);
                    xLeaf->setValueOn(phiIter.offset(), xyzTerm[0]);
                    yLeaf->setValueOn(phiIter.offset(), xyzTerm[1]);
                    zLeaf->setValueOn(phiIter.offset(), xyzTerm[2]);
                }
                else {
                    diagleaf.setValueOff(phiIter.offset());
                    xLeaf->setValueOff(phiIter.offset());
                    yLeaf->setValueOff(phiIter.offset());
                    zLeaf->setValueOff(phiIter.offset());
                }
            }//end if this voxel is liquid voxel
            else {
                diagleaf.setValueOff(phiIter.offset());
                xLeaf->setValueOff(phiIter.offset());
                yLeaf->setValueOff(phiIter.offset());
                zLeaf->setValueOff(phiIter.offset());
            }//else if this voxel is liquid voxel
        }//end for all touched liquid phi voxels
    }//end operator ()


    mutable openvdb::FloatGrid::ConstUnsafeAccessor mPhiAxr;
    mutable openvdb::Vec3fGrid::ConstUnsafeAccessor mWeightAxr;

    const std::vector<openvdb::FloatTree::LeafNodeType*>& mPhiLeaves;
    const std::vector<openvdb::FloatTree::LeafNodeType*>& mNegX;
    const std::vector<openvdb::FloatTree::LeafNodeType*>& mNegY;
    const std::vector<openvdb::FloatTree::LeafNodeType*>& mNegZ;
    float mDt, mDx;
};
}//end namespace

void LaplacianWithLevel::initializeFinest(
    openvdb::FloatGrid::Ptr in_liquid_phi,
    openvdb::Vec3fGrid::Ptr in_face_weights)
{
    std::vector<openvdb::FloatTree::LeafNodeType*> phiLeaves, xLeaves, yLeaves, zLeaves;
    auto nLeaf = in_liquid_phi->tree().leafCount();
    if (0 == nLeaf) {
        return;
    }
    phiLeaves.reserve(nLeaf);
    xLeaves.reserve(nLeaf);
    yLeaves.reserve(nLeaf);
    zLeaves.reserve(nLeaf);
    in_liquid_phi->tree().getNodes(phiLeaves);
    mXEntry->tree().getNodes(xLeaves);
    mYEntry->tree().getNodes(yLeaves);
    mZEntry->tree().getNodes(zLeaves);

    BuildFinestMatrix op(
        in_liquid_phi,
        in_face_weights,
        phiLeaves,
        xLeaves,
        yLeaves,
        zLeaves, mDt, mDxThisLevel);

    auto diagLeafMan = openvdb::tree::LeafManager<openvdb::FloatTree>(mDiagonal->tree());
    diagLeafMan.foreach(op, true, 10);

    //set the dof index
    mDofIndex = openvdb::Int32Grid::create(0);
    mDofIndex->setTree(std::make_shared<openvdb::Int32Tree>(mDiagonal->tree(), -1, openvdb::TopologyCopy()));
    mDofIndex->setTransform(mDiagonal->transformPtr());
    setDofIndex(mDofIndex);
    trimDefaultNodes();
}

void LaplacianWithLevel::setDofIndex(openvdb::Int32Grid::Ptr in_out_dofidx)
{
    auto dofLeafMan = openvdb::tree::LeafManager<openvdb::Int32Tree>(in_out_dofidx->tree());

    //first count how many dof in each leaf
    //then assign the global dof id
    std::vector<openvdb::Int32> dofEndInEachLeaf;
    dofEndInEachLeaf.assign(in_out_dofidx->tree().leafCount(), 0);

    auto dofCounter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
        dofEndInEachLeaf[leafpos] = leaf.onVoxelCount();
    };//end leaf active dof counter
    dofLeafMan.foreach(dofCounter);

    //scan through all leaves to determine
    for (size_t i = 1; i < dofEndInEachLeaf.size(); i++) {
        dofEndInEachLeaf[i] += dofEndInEachLeaf[i - 1];
    }

    auto set_dof_id = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
        openvdb::Int32 beginIndex = 0;
        if (leafpos != 0) {
            beginIndex = dofEndInEachLeaf[leafpos - 1];
        }
        leaf.fill(-1);
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.setValue(beginIndex);
            beginIndex++;
        }
    };
    dofLeafMan.foreach(set_dof_id);

    //return the total number of degree of freedom
    mNumDof = *dofEndInEachLeaf.crbegin();
}





//In normal mode applies L * lhs = result
//where lhs is the input
//In weighted jacobi mode
//apply one iteration to solve L * lhs = rhs
//and lhs and rhs are unchanged
struct alignas(32) LaplacianApplySIMD {
    enum class WorkingMode {
        WeightedJacobi,
        Residual,
        Laplacian,
        RedGS,
        BlackGS,
        SPAI0
    };
    struct LightWeightApplier {
        using LeafType = openvdb::FloatTree::LeafNodeType;
        using LeafArrayType = std::vector<LeafType*>;
        using LeafArrayPtrType = std::shared_ptr<LeafArrayType>;

        LightWeightApplier(LaplacianApplySIMD* in_parent,
            openvdb::FloatGrid::Ptr lhs,
            openvdb::FloatGrid::Ptr rhs,
            openvdb::FloatGrid::Ptr result, WorkingMode in_mode) {
            mMode = in_mode;
            mParent = in_parent;

            //fill result leaves
            mResultLeaves = std::make_shared<LeafArrayType>();
            mResultLeaves->reserve(result->tree().leafCount());
            result->tree().getNodes(*mResultLeaves);

            mLhsLeaves = std::make_shared<LeafArrayType>();
            mLhsLeaves->reserve(lhs->tree().leafCount());
            lhs->tree().getNodes(*mLhsLeaves);
            //adding the last element as nullptr so the leaf neighbor vector can point to this nullptr.
            mLhsLeaves->push_back(nullptr);

            //depending on the working mode, prepare leaf arrays
            if (mMode == WorkingMode::WeightedJacobi ||
                mMode == WorkingMode::SPAI0 ||
                mMode == WorkingMode::Residual ||
                mMode == WorkingMode::RedGS ||
                mMode == WorkingMode::BlackGS) {
                mRhsLeaves = std::make_shared<LeafArrayType>();
                mRhsLeaves->reserve(rhs->tree().leafCount());
                rhs->tree().getNodes(*mRhsLeaves);
            }
        }//light weight applier constructor

        //default copy constructor and destructor should be sufficient to do the job

        float* getData(openvdb::FloatTree::LeafNodeType* in_leaf) const {
            if (in_leaf) {
                return in_leaf->buffer().data();
            }
            else {
                return nullptr;
            }
        }

        const float* getData(const openvdb::FloatTree::LeafNodeType* in_leaf) const {
            if (in_leaf) {
                return in_leaf->buffer().data();
            }
            else {
                return nullptr;
            }
        }

        //this operator is to be applied on the dof index grid
        //this ensures operating on the correct mask
        //the computation only operates on 8 float at a time
        //using avx intrinsics
        //each leaf contains 512 float
        //offset = 64*x+8*y+z
        //hence each bulk need to loop over x and y
        //make sure not to use SSE instructions here to avoid context switching cost
        //called by the topology tree manager
        void operator()(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const {
            const float* rhsData;
            if (mMode == WorkingMode::WeightedJacobi ||
                mMode == WorkingMode::SPAI0 ||
                mMode == WorkingMode::Residual ||
                mMode == WorkingMode::RedGS ||
                mMode == WorkingMode::BlackGS) {
                rhsData = getData((*mRhsLeaves)[leafpos]);
            }
            float* resultData = getData((*mResultLeaves)[leafpos]);
            const float* diagData = getData(mParent->mDiagonalLeaves[leafpos]);
            const float* invdiagData = getData(mParent->mInvDiagLeaves[leafpos]);
            const float* SPAI0data;
            if (mMode == WorkingMode::SPAI0) SPAI0data = getData(mParent->mSPAI0Leaves[leafpos]);

            std::array<const float*, 7> lhsData;
            lhsData.fill(nullptr); lhsData[0] = getData((*mLhsLeaves)[leafpos]);
            for (int i = 0; i < 6; i++) {
                lhsData[i + 1] = getData((*mLhsLeaves)[mParent->mNeighborLeafOffset[leafpos][i]]);
            }

            const float* xTermData[2] = { nullptr,nullptr };
            xTermData[0] = getData(mParent->mXLeaves[leafpos]);
            xTermData[1] = getData(mParent->mXLeavesUpper[leafpos]);

            //0 self, 1 yp
            const float* yTermData[2] = { nullptr,nullptr };
            yTermData[0] = getData(mParent->mYLeaves[leafpos]);
            yTermData[1] = getData(mParent->mYLeavesUpper[leafpos]);

            //0 self, 1 zp
            const float* zTermData[2] = { nullptr,nullptr };
            zTermData[0] = getData(mParent->mZLeaves[leafpos]);
            zTermData[1] = getData(mParent->mZLeavesUpper[leafpos]);

            //loop over 64 vectors to conduct the computation
            // vectorid = x*8+y
            std::array<__m256, 7> faceResult;
            //uint32_t tempoffset;
            for (uint32_t vectorOffset = 0; vectorOffset < 512; vectorOffset += 8) {
                const uint8_t vectorMask = leaf.getValueMask().getWord<uint8_t>(vectorOffset / 8);
                if (vectorMask == uint8_t(0)) {
                    //there is no diagonal entry in this vector
                    continue;
                }

                {
                    __m256 xPlusLhs;
                    __m256 xPlusTerm;
                    __m256 xMinusLhs;
                    __m256 xMinusTerm;
                    //vector in the x directoin
                    /****************************************************************************/

                    //load current and lower leaf
                    if (vectorOffset < (7 * 64)) {
                        auto tempoffset = vectorOffset + 64;
                        vdb_SIMD_IO::get_simd_vector_unsafe(xPlusLhs, lhsData[0], tempoffset);
                        vdb_SIMD_IO::get_simd_vector(xPlusTerm, xTermData[0], tempoffset, mParent->mDefaultFaceTerm);
                        if (vectorOffset < 64) {
                            vdb_SIMD_IO::get_simd_vector(xMinusLhs, lhsData[2], vectorOffset + (7 * 64), _mm256_setzero_ps());
                        }
                        else {
                            vdb_SIMD_IO::get_simd_vector_unsafe(xMinusLhs, lhsData[0], vectorOffset - 64);
                        }
                    }
                    else {
                        //load next leaf
                        auto tempoffset = vectorOffset - (7 * 64);
                        vdb_SIMD_IO::get_simd_vector(xPlusLhs, lhsData[1], tempoffset, _mm256_setzero_ps());
                        vdb_SIMD_IO::get_simd_vector(xPlusTerm, xTermData[1], tempoffset, mParent->mDefaultFaceTerm);

                        //x-
                        vdb_SIMD_IO::get_simd_vector_unsafe(xMinusLhs, lhsData[0], vectorOffset - 64);
                    }

                    //vector in the x minux direction
                    /******************************************************************************/
                    vdb_SIMD_IO::get_simd_vector(xMinusTerm, xTermData[0], vectorOffset, mParent->mDefaultFaceTerm);
                    faceResult[1] = _mm256_mul_ps(xPlusLhs, xPlusTerm);
                    faceResult[2] = _mm256_mul_ps(xMinusLhs, xMinusTerm);
                }

                {
                    __m256 yPlusLhs;
                    __m256 yPlusTerm;
                    __m256 yMinusLhs;
                    __m256 yMinusTerm;

                    //vector in the y plus direction
                    /****************************************************************************/
                    //load current and lower
                    if ((vectorOffset & 63u) != 56u) {
                        auto tempoffset = vectorOffset + 8;
                        vdb_SIMD_IO::get_simd_vector_unsafe(yPlusLhs, lhsData[0], tempoffset);
                        vdb_SIMD_IO::get_simd_vector(yPlusTerm, yTermData[0], tempoffset, mParent->mDefaultFaceTerm);
                        if ((vectorOffset & 63) == 0) {
                            vdb_SIMD_IO::get_simd_vector(yMinusLhs, lhsData[4], vectorOffset + 56, _mm256_setzero_ps());
                        }
                        else {
                            vdb_SIMD_IO::get_simd_vector_unsafe(yMinusLhs, lhsData[0], vectorOffset - 8);
                        }
                    }
                    else {
                        //load next leaf
                        auto tempoffset = vectorOffset - 56;
                        vdb_SIMD_IO::get_simd_vector(yPlusLhs, lhsData[3], tempoffset, _mm256_setzero_ps());
                        vdb_SIMD_IO::get_simd_vector(yPlusTerm, yTermData[1], tempoffset, mParent->mDefaultFaceTerm);
                        //y-
                        vdb_SIMD_IO::get_simd_vector_unsafe(yMinusLhs, lhsData[0], vectorOffset - 8);
                    }

                    //vector in the y minus direction
                    /****************************************************************************/
                    vdb_SIMD_IO::get_simd_vector(yMinusTerm, yTermData[0], vectorOffset, mParent->mDefaultFaceTerm);
                    faceResult[3] = _mm256_mul_ps(yPlusLhs, yPlusTerm);
                    faceResult[4] = _mm256_mul_ps(yMinusLhs, yMinusTerm);
                }

                {
                    __m256 zPlusLhs;
                    __m256 zPlusTerm;
                    __m256 zMinusLhs;
                    __m256 zMinusTerm;

                    //Z lhs terms
                    /****************************************************************************/
                    vdb_SIMD_IO::get_pos_z_simd_vector(zPlusLhs, lhsData[0], lhsData[5], vectorOffset, _mm256_setzero_ps());
                    vdb_SIMD_IO::get_neg_z_simd_vector(zMinusLhs, lhsData[0], lhsData[6], vectorOffset, _mm256_setzero_ps());

                    //Z coefficient terms
                    /****************************************************************************/
                    vdb_SIMD_IO::get_pos_z_simd_vector(zPlusTerm, zTermData[0], zTermData[1], vectorOffset, mParent->mDefaultFaceTerm);
                    vdb_SIMD_IO::get_simd_vector(zMinusTerm, zTermData[0], vectorOffset, mParent->mDefaultFaceTerm);
                    //temp_result = _mm256_fmadd_ps(zMinusLhs, zMinusTerm, temp_result);

                    faceResult[5] = _mm256_mul_ps(zPlusLhs, zPlusTerm);
                    faceResult[6] = _mm256_mul_ps(zMinusLhs, zMinusTerm);

                    //collect
                    faceResult[1] = _mm256_add_ps(faceResult[1], faceResult[2]);
                    faceResult[3] = _mm256_add_ps(faceResult[3], faceResult[4]);
                    faceResult[5] = _mm256_add_ps(faceResult[5], faceResult[6]);

                    faceResult[1] = _mm256_add_ps(faceResult[1], faceResult[3]);
                    faceResult[1] = _mm256_add_ps(faceResult[1], faceResult[5]);
                }
                //now faced_result[1] contains all the off-diagonal results
                //collected form 1,2,3,4,5,6
                __m256 thisLhs;
                //__m256 thisDiag;
                //__m256 thisRhs;
                //__m256 residual;
                //__m256 thisInvDiag;
                //faced_result[0] stores the result of A*lhs
                //faced_result[6] = _mm256_add_ps(faced_result[4], faced_result[6]);
                vdb_SIMD_IO::get_simd_vector_unsafe(thisLhs, lhsData[0], vectorOffset);
                switch (mMode) {
                case WorkingMode::WeightedJacobi:
                {
                    __m256 thisDiag;
                    __m256 thisRhs;
                    __m256 residual;
                    __m256 thisInvDiag;
                    //v^1 = v^0 + w D^-1 r
                    // r = rhs - A*v^0
                    //v^0 is the input lhs
                    //v^1 is the output result
                    //the residual
                    //the diagonal term
                    /****************************************************************************/
                    vdb_SIMD_IO::get_simd_vector(thisDiag, diagData, vectorOffset, mParent->mDefaultDiagonal);
                    faceResult[0] = _mm256_fmadd_ps(thisLhs, thisDiag, faceResult[1]);
                    vdb_SIMD_IO::get_simd_vector_unsafe(thisRhs, rhsData, vectorOffset);
                    residual = _mm256_sub_ps(thisRhs, faceResult[0]);
                    vdb_SIMD_IO::get_simd_vector(thisInvDiag, invdiagData, vectorOffset, mParent->mDefaultInvDiagonal);
                    //thisInvDiag = _mm256_rcp_ps(thisDiag);
                    residual = _mm256_mul_ps(residual, thisInvDiag);
                    faceResult[0] = _mm256_fmadd_ps(mParent->mPackedWeightJacobi, residual, thisLhs);
                    break;
                }

                case WorkingMode::Residual:
                {
                    __m256 thisDiag;
                    __m256 thisRhs;
                    //the diagonal term
                    /****************************************************************************/
                    vdb_SIMD_IO::get_simd_vector(thisDiag, diagData, vectorOffset, mParent->mDefaultDiagonal);
                    faceResult[0] = _mm256_fmadd_ps(thisLhs, thisDiag, faceResult[1]);
                    vdb_SIMD_IO::get_simd_vector_unsafe(thisRhs, rhsData, vectorOffset);
                    faceResult[0] = _mm256_sub_ps(thisRhs, faceResult[0]);
                    break;
                }

                default:
                case WorkingMode::Laplacian:
                {
                    __m256 thisDiag;
                    //the diagonal term
                    /****************************************************************************/
                    vdb_SIMD_IO::get_simd_vector(thisDiag, diagData, vectorOffset, mParent->mDefaultDiagonal);
                    faceResult[0] = _mm256_fmadd_ps(thisLhs, thisDiag, faceResult[1]);
                    break;
                }
                case WorkingMode::RedGS:
                {
                    bool vector_even = ((vectorOffset >> 3) + (vectorOffset >> 6)) % 2 == 0;
                    __m256 residual;
                    __m256 thisRhs;
                    __m256 thisInvDiag;
                    vdb_SIMD_IO::get_simd_vector_unsafe(thisRhs, rhsData, vectorOffset);
                    residual = _mm256_sub_ps(thisRhs, faceResult[1]);
                    vdb_SIMD_IO::get_simd_vector(thisInvDiag, invdiagData, vectorOffset, mParent->mDefaultInvDiagonal);
                    residual = _mm256_mul_ps(residual, thisInvDiag);
                    residual = _mm256_mul_ps(residual, mParent->mPackedWeightSOR);
                    faceResult[0] = _mm256_fmadd_ps(thisLhs, mParent->mPackedOneMinusWeightSOR, residual);
                    //blend result
                    if (vector_even) {
                        faceResult[0] = _mm256_blend_ps(thisLhs, faceResult[0], 0b01010101);
                    }
                    else {
                        faceResult[0] = _mm256_blend_ps(thisLhs, faceResult[0], 0b10101010);
                    }
                    break;
                }
                case WorkingMode::BlackGS:
                {
                    bool vector_even = ((vectorOffset >> 3) + (vectorOffset >> 6)) % 2 == 0;
                    __m256 residual;
                    __m256 thisRhs;
                    __m256 thisInvDiag;
                    vdb_SIMD_IO::get_simd_vector_unsafe(thisRhs, rhsData, vectorOffset);
                    residual = _mm256_sub_ps(thisRhs, faceResult[1]);
                    vdb_SIMD_IO::get_simd_vector(thisInvDiag, invdiagData, vectorOffset, mParent->mDefaultInvDiagonal);
                    residual = _mm256_mul_ps(residual, thisInvDiag);
                    residual = _mm256_mul_ps(residual, mParent->mPackedWeightSOR);
                    faceResult[0] = _mm256_fmadd_ps(thisLhs, mParent->mPackedOneMinusWeightSOR, residual);
                    //blend result
                    if (vector_even) {
                        faceResult[0] = _mm256_blend_ps(thisLhs, faceResult[0], 0b10101010);
                    }
                    else {
                        faceResult[0] = _mm256_blend_ps(thisLhs, faceResult[0], 0b01010101);
                    }
                    break;
                }
                case WorkingMode::SPAI0:
                {
                    __m256 residual;
                    __m256 thisDiag;
                    __m256 thisRhs;
                    __m256 this_SPAI0;
                    //the diagonal term
                    /****************************************************************************/
                    vdb_SIMD_IO::get_simd_vector(thisDiag, diagData, vectorOffset, mParent->mDefaultDiagonal);
                    faceResult[0] = _mm256_fmadd_ps(thisLhs, thisDiag, faceResult[1]);
                    vdb_SIMD_IO::get_simd_vector_unsafe(thisRhs, rhsData, vectorOffset);
                    residual = _mm256_sub_ps(thisRhs, faceResult[0]);
                    vdb_SIMD_IO::get_simd_vector(this_SPAI0, SPAI0data, vectorOffset, mParent->mDefaultSPAI0);
                    faceResult[0] = _mm256_fmadd_ps(this_SPAI0, residual, thisLhs);
                    break;
                }
                }//end case working mode
                //faced_result[0] = _mm256_blend_ps(faced_result[0], this_lhs, vectormask);
                //write to the result

                _mm256_storeu_ps(resultData + vectorOffset, faceResult[0]);
                //make sure we write at the correct place
                for (int bit = 0; bit < 8; bit += 1) {
                    if (0 == ((vectorMask) & (1 << bit))) {
                        resultData[vectorOffset + bit] = 0.f;
                    }
                }

            }//end vectorid = [0 63]
        }//end operator()


        //lhs, rhs, result leaves having the same topology
        //these are the flattened leaves used for fast access of 
        LeafArrayPtrType mLhsLeaves;
        LeafArrayPtrType mRhsLeaves;
        LeafArrayPtrType mResultLeaves;

        WorkingMode mMode;

        LaplacianApplySIMD* mParent;
    };//end light weight evaluator

    LaplacianApplySIMD(
        openvdb::FloatGrid::Ptr in_Diagonal,
        openvdb::FloatGrid::Ptr in_Neg_x_entry,
        openvdb::FloatGrid::Ptr in_Neg_y_entry,
        openvdb::FloatGrid::Ptr in_Neg_z_entry,
        openvdb::Int32Grid::Ptr in_DOF,
        float in_default_term) : mDofLeafManager(in_DOF->tree()),
        mWeightSOR(1.2f) {
        mDiagonal = in_Diagonal;
        mXEntry = in_Neg_x_entry;
        mYEntry = in_Neg_y_entry;
        mZEntry = in_Neg_z_entry;
        mWeightJacobi = 6.0f / 7.0f;
        mDefaultFaceTerm = _mm256_set1_ps(-in_default_term);
        mDefaultDiagonal = _mm256_set1_ps(6.f * in_default_term);
        mDefaultInvDiagonal = _mm256_set1_ps(1.0f / (6.f * in_default_term));
        mDefaultSPAI0 = _mm256_set1_ps(1.0f / (7.f * in_default_term));
        mPackedWeightJacobi = _mm256_set1_ps(mWeightJacobi);
        mPackedWeightSOR = _mm256_set1_ps(mWeightSOR);
        mPackedOneMinusWeightSOR = _mm256_set1_ps(1.0f - mWeightSOR);
        mMode = WorkingMode::Laplacian;
        mInvDIagonal = openvdb::FloatGrid::create(1.0f / mDiagonal->background());
        mIsInvDiagonalInitialized = false;
        mSPAI0 = openvdb::FloatGrid::create(1.0f / (7.f * in_default_term));
        mIsSPAI0Initialized = false;
        initLinearizedLeaves();
    }

    LightWeightApplier getLightWeightApplier(openvdb::FloatGrid::Ptr in_out_result, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs) {
        return LightWeightApplier(this, in_lhs, in_rhs, in_out_result, mMode);
    }

    void Apply(LightWeightApplier in_applier) {
        mDofLeafManager.foreach(in_applier);
    }

    void initInvDiagonal() {
        if (mIsInvDiagonalInitialized) return;
        //set the inverse default term and inverse diagonal
        mInvDIagonal->setTree(std::make_shared<openvdb::FloatTree>(
            mDiagonal->tree(), 1.0f / mDiagonal->background(), openvdb::TopologyCopy()));
        auto leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(mInvDIagonal->tree());
        leafman.foreach([&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index) {
            auto* diagleaf = mDiagonal->tree().probeConstLeaf(leaf.origin());
            for (auto iter = leaf.beginValueOn(); iter; ++iter) {
                float diagValue = diagleaf->getValue(iter.offset());
                if (diagValue == 0) {
                    iter.setValue(0.f);
                }
                else {
                    iter.setValue(1.0f / diagValue);
                }

            }
            }
        );

        size_t nleaf = mDofLeafManager.leafCount();
        mInvDiagLeaves.resize(nleaf, nullptr);

        auto invdiag_leaves_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
            mInvDiagLeaves[leafpos] = mInvDIagonal->tree().probeConstLeaf(leaf.origin());
        };
        mDofLeafManager.foreach(invdiag_leaves_setter);

        mIsInvDiagonalInitialized = true;
    };

    void init_SPAI0() {
        if (mIsSPAI0Initialized) return;
        //set the SPAI0 matrix
        //by default the rhs has the correct pattern, which is not pruned
        // spai diagonal: akk/(ak)_2^2
        // diagonal entry over the row norm
        //set the inverse default term and inverse diagonal

        float defaultSPAI0Val = 6.0f / (7.f * mDiagonal->background());
        mSPAI0->setTree(std::make_shared<openvdb::FloatTree>(
            mDofLeafManager.tree(), defaultSPAI0Val, openvdb::TopologyCopy()));
        auto sqr = [](float in) {
            return in * in;
        };

        auto spai0Setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
            auto xAxr{ mXEntry->getConstUnsafeAccessor() };
            auto yAxr{ mYEntry->getConstUnsafeAccessor() };
            auto zAxr{ mZEntry->getConstUnsafeAccessor() };
            auto diagAxr{ mDiagonal->getConstUnsafeAccessor() };
            openvdb::Int32Grid::ConstUnsafeAccessor dofAxr(mDofLeafManager.tree());
            auto spai0Leaf = mSPAI0->tree().probeLeaf(leaf.origin());
            for (auto iter = leaf.beginValueOn(); iter; ++iter) {
                float rowNormSqr = 0;
                const auto globalCoord = iter.getCoord();
                //+x
                auto atCoord = globalCoord.offsetBy(1, 0, 0);
                if (dofAxr.isValueOn(atCoord)) {
                    rowNormSqr += sqr(xAxr.getValue(atCoord));
                }
                //-x
                atCoord = globalCoord.offsetBy(-1, 0, 0);
                if (dofAxr.isValueOn(atCoord)) {
                    rowNormSqr += sqr(xAxr.getValue(globalCoord));
                }
                //+y
                atCoord = globalCoord.offsetBy(0, 1, 0);
                if (dofAxr.isValueOn(atCoord)) {
                    rowNormSqr += sqr(yAxr.getValue(atCoord));
                }
                //-y
                atCoord = globalCoord.offsetBy(0, -1, 0);
                if (dofAxr.isValueOn(atCoord)) {
                    rowNormSqr += sqr(yAxr.getValue(globalCoord));
                }
                //+z
                atCoord = globalCoord.offsetBy(0, 0, 1);
                if (dofAxr.isValueOn(atCoord)) {
                    rowNormSqr += sqr(zAxr.getValue(atCoord));
                }
                //-z
                atCoord = globalCoord.offsetBy(0, 0, -1);
                if (dofAxr.isValueOn(atCoord)) {
                    rowNormSqr += sqr(zAxr.getValue(globalCoord));
                }
                float diagval = diagAxr.getValue(globalCoord);
                if (diagval == 0) {
                    spai0Leaf->setValueOn(iter.offset(), 0);
                }
                else {
                    spai0Leaf->setValueOn(iter.offset(), diagval / (sqr(diagval) + rowNormSqr));
                }
            }//end for all dof
        };//end spai setter

        mDofLeafManager.foreach(spai0Setter);
        LaplacianWithLevel::trimDefaultNodes(mSPAI0, defaultSPAI0Val, defaultSPAI0Val * 1e-5f);

        //set SPAI0 leaves after trim
        size_t nleaf = mDofLeafManager.leafCount();
        mSPAI0Leaves.resize(nleaf, nullptr);
        auto SPAI0LeavesSetter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
            mSPAI0Leaves[leafpos] = mSPAI0->tree().probeConstLeaf(leaf.origin());
        };
        mDofLeafManager.foreach(SPAI0LeavesSetter);

        mIsSPAI0Initialized = true;
    };

    void initLinearizedLeaves() {
        size_t nLeaf = mDofLeafManager.leafCount();

        //map DOF ptrs to their positions in the linearized array
        std::unordered_map<const openvdb::Int32Tree::LeafNodeType*, size_t> leafPtrToLeafposMap;
        std::vector<openvdb::Int32Tree::LeafNodeType*> dofLeafPtrs;
        dofLeafPtrs.reserve(nLeaf);
        mDofLeafManager.getNodes(dofLeafPtrs);
        for (size_t i = 0; i < nLeaf; i++) {
            leafPtrToLeafposMap[dofLeafPtrs[i]] = i;
        }
        mNeighborLeafOffset.resize(nLeaf);

        mDiagonalLeaves.resize(nLeaf, nullptr);
        mXLeaves.resize(nLeaf, nullptr);
        mYLeaves.resize(nLeaf, nullptr);
        mZLeaves.resize(nLeaf, nullptr);
        mXLeavesUpper.resize(nLeaf, nullptr);
        mYLeavesUpper.resize(nLeaf, nullptr);
        mZLeavesUpper.resize(nLeaf, nullptr);

        auto LinearArraySetter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
            //nLeaf+1 is the last element of linearized array
            //indicating nullptr
            mNeighborLeafOffset[leafpos].fill(nLeaf);

            mDiagonalLeaves[leafpos] = mDiagonal->tree().probeConstLeaf(leaf.origin());
            mXLeaves[leafpos] = mXEntry->tree().probeConstLeaf(leaf.origin());
            mYLeaves[leafpos] = mYEntry->tree().probeConstLeaf(leaf.origin());
            mZLeaves[leafpos] = mZEntry->tree().probeConstLeaf(leaf.origin());
            mXLeavesUpper[leafpos] = mXEntry->tree().probeConstLeaf(leaf.origin().offsetBy(8, 0, 0));
            mYLeavesUpper[leafpos] = mYEntry->tree().probeConstLeaf(leaf.origin().offsetBy(0, 8, 0));
            mZLeavesUpper[leafpos] = mZEntry->tree().probeConstLeaf(leaf.origin().offsetBy(0, 0, 8));

            for (int iFace = 0; iFace < 6; iFace++) {
                int component = iFace / 2;
                int isPositiveDir = (iFace % 2 == 0);

                auto neihborOrigin = leaf.origin();
                if (isPositiveDir) {
                    neihborOrigin[component] += 8;
                }
                else {
                    neihborOrigin[component] -= 8;
                }
                auto neighborLeaf = mDofLeafManager.tree().probeConstLeaf(neihborOrigin);
                if (neighborLeaf) {
                    mNeighborLeafOffset[leafpos][iFace] = leafPtrToLeafposMap[neighborLeaf];
                }
            }
        };
        mDofLeafManager.foreach(LinearArraySetter);
        initInvDiagonal();
    };

    void setLaplacianMode() { mMode = WorkingMode::Laplacian; }
    void setWeightedJacobiMode() { mMode = WorkingMode::WeightedJacobi; initInvDiagonal(); }
    void setSPAI0Mode() { mMode = WorkingMode::SPAI0; init_SPAI0(); }
    void setResidualMode() { mMode = WorkingMode::Residual; }
    void setWeightJacobi(float weight) {
        mWeightJacobi = weight;
        mPackedWeightJacobi = _mm256_set1_ps(mWeightJacobi);
    }

    void set_w_sor(float in_w_sor) {
        mWeightSOR = in_w_sor;
        mPackedWeightSOR = _mm256_set1_ps(mWeightSOR);
        mPackedOneMinusWeightSOR = _mm256_set1_ps(1.0f - mWeightSOR);
    }
    //gauss seidel updating the red points (iChannel+j+k) even
    void setRedGaussSeidelMode() { mMode = WorkingMode::RedGS; initInvDiagonal(); }
    //gauss seidel update the black points (iChannel+j+k) odd
    void setBlackGaussSeidelMode() { mMode = WorkingMode::BlackGS; initInvDiagonal(); }


    //over load new and delete for aligned allocation and free
    //this struct must be properly aligned for __m256 member
    void* operator new(size_t memsize) {
        size_t ptrAlloc = sizeof(void*);
        size_t alignSize = 32;
        size_t requestSize = sizeof(LaplacianApplySIMD) + alignSize;
        size_t needed = ptrAlloc + requestSize;

        void* alloc = ::operator new(needed);
        void* realAlloc = reinterpret_cast<void*>(reinterpret_cast<char*>(alloc) + ptrAlloc);
        void* ptr = std::align(alignSize, sizeof(LaplacianApplySIMD),
            realAlloc, requestSize);

        ((void**)ptr)[-1] = alloc; // save for delete calls to use
        return ptr;
    }

    void operator delete(void* ptr) {
        void* alloc = ((void**)ptr)[-1];
        ::operator delete(alloc);
    }

    //to be set in the constructor
    //when the requested leaf doesn't exist
    __m256 mDefaultFaceTerm;
    __m256 mDefaultDiagonal;
    __m256 mDefaultInvDiagonal;
    __m256 mDefaultSPAI0;

    //damped jacobi weight
    float mWeightJacobi;
    __m256 mPackedWeightJacobi;

    float mWeightSOR;
    __m256 mPackedWeightSOR, mPackedOneMinusWeightSOR;

    openvdb::FloatGrid::Ptr mDiagonal;
    openvdb::FloatGrid::Ptr mInvDIagonal;
    bool mIsInvDiagonalInitialized;
    openvdb::FloatGrid::Ptr mSPAI0;
    bool mIsSPAI0Initialized;
    openvdb::FloatGrid::Ptr mXEntry;
    openvdb::FloatGrid::Ptr mYEntry;
    openvdb::FloatGrid::Ptr mZEntry;

    //DOF manager that calls all function body over leaves
    openvdb::tree::LeafManager<openvdb::Int32Tree> mDofLeafManager;

    //linearized leaves for matrix coefficient
    //note some leaves may be nullptrs, because they are default values;
    //all leaves have length of DOF leaves
    std::vector<const openvdb::FloatTree::LeafNodeType*> mDiagonalLeaves, mInvDiagLeaves, mSPAI0Leaves;
    std::vector<const openvdb::FloatTree::LeafNodeType*> mXLeaves, mYLeaves, mZLeaves;
    std::vector<const openvdb::FloatTree::LeafNodeType*> mXLeavesUpper, mYLeavesUpper, mZLeavesUpper;

    //6 neighbor LHS leaves
    //x- x+ y- y+ z- z+
    //the offset referrs to the linearize LHS leaves in the light weight applier, length DOF leaves+1
    //the last element of that array is nullptr, so empty neighbor points to the last element.
    //this vector has leng of DOF leaves
    std::vector<std::array<size_t, 6>> mNeighborLeafOffset;

    WorkingMode mMode;
};//end LaplacianApplySIMD


void LaplacianWithLevel::initializeApplyOperator()
{
    //set the leaf manager
    mDofLeafManager = std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(mDofIndex->tree());

    mApplyOperator = std::shared_ptr<LaplacianApplySIMD>(new LaplacianApplySIMD(mDiagonal,
        mXEntry,
        mYEntry,
        mZEntry, mDofIndex,
        /*default term*/ mDt / (mDxThisLevel * mDxThisLevel)));
    mApplyOperator->setLaplacianMode();
}


void LaplacianWithLevel::residualApply(
    openvdb::FloatGrid::Ptr in_out_residual, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs)
{
    mApplyOperator->setResidualMode();
    _mm256_zeroupper();
    auto op = mApplyOperator->getLightWeightApplier(in_out_residual, in_lhs, in_rhs);
    {
        mApplyOperator->Apply(op);
    }
    _mm256_zeroupper();
}

void LaplacianWithLevel::laplacianApply(openvdb::FloatGrid::Ptr in_out_result, openvdb::FloatGrid::Ptr in_lhs)
{
    mApplyOperator->setLaplacianMode();
    _mm256_zeroupper();
    auto op = mApplyOperator->getLightWeightApplier(in_out_result, in_lhs, in_lhs);
    mApplyOperator->Apply(op);
    _mm256_zeroupper();
}

void LaplacianWithLevel::weightedJacobiApply(openvdb::FloatGrid::Ptr in_out_updated_lhs, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs)
{
    mApplyOperator->setWeightedJacobiMode();
    _mm256_zeroupper();
    auto op = mApplyOperator->getLightWeightApplier(in_out_updated_lhs, in_lhs, in_rhs);
    {
        mApplyOperator->Apply(op);
    }
    _mm256_zeroupper();
}
void LaplacianWithLevel::spai0Apply(openvdb::FloatGrid::Ptr in_out_updated_lhs, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs)
{
    mApplyOperator->setSPAI0Mode();
    _mm256_zeroupper();
    auto op = mApplyOperator->getLightWeightApplier(in_out_updated_lhs, in_lhs, in_rhs);
    mApplyOperator->Apply(op);
    _mm256_zeroupper();
}
template<bool red_first>
void LaplacianWithLevel::redBlackGaussSeidelApply(openvdb::FloatGrid::Ptr in_out_lhs, openvdb::FloatGrid::Ptr in_rhs)
{
    _mm256_zeroupper();
    if (red_first) {
        mApplyOperator->setRedGaussSeidelMode();
    }
    else {
        mApplyOperator->setBlackGaussSeidelMode();
    }

    auto op = mApplyOperator->getLightWeightApplier(in_out_lhs, in_out_lhs, in_rhs);
    mApplyOperator->Apply(op);

    //the scratch_pad has now red points updated
    if (red_first) {
        mApplyOperator->setBlackGaussSeidelMode();
    }
    else {
        mApplyOperator->setRedGaussSeidelMode();
    }

    op = mApplyOperator->getLightWeightApplier(in_out_lhs, in_out_lhs, in_rhs);
    mApplyOperator->Apply(op);

    _mm256_zeroupper();
}

openvdb::FloatGrid::Ptr LaplacianWithLevel::getZeroVectorGrid()
{
    openvdb::FloatGrid::Ptr result = openvdb::FloatGrid::create(0);
    result->setTree(std::make_shared<openvdb::FloatTree>(mDofIndex->tree(), 0.f, openvdb::TopologyCopy()));
    result->setTransform(mDofIndex->transformPtr());
    return result;
}

void LaplacianWithLevel::setGridToConstant(openvdb::FloatGrid::Ptr in_out_grid, float constant)
{
    openvdb::FloatTree::LeafNodeType constantLeaf;
    constantLeaf.fill(constant);

    auto set_constant_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* out_leaf = in_out_grid->tree().probeLeaf(leaf.origin());
        std::copy(constantLeaf.buffer().data(), constantLeaf.buffer().data() + constantLeaf.SIZE, out_leaf->buffer().data());
    };

    mDofLeafManager->foreach(set_constant_op);
}

void LaplacianWithLevel::setGridToResultAfterFirstJacobi(openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs)
{
    //assume the input has the same topology 
    //apply weighted jacobi iteration to result = (L^-1) rhs
    //assume the initial guess of result was 0
    //This amounts to v^1 = w*D^-1(b-A*0)=w*(invdiag)*rhs
    _mm256_zeroall();

    std::vector<openvdb::FloatTree::LeafNodeType*> resultLeaves, rhsLeaves;
    resultLeaves.reserve(mApplyOperator->mDofLeafManager.leafCount());
    rhsLeaves.reserve(mApplyOperator->mDofLeafManager.leafCount());
    in_out_grid->tree().getNodes(resultLeaves);
    in_rhs->tree().getNodes(rhsLeaves);

    auto setResultAfterFirstIter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
        auto* resultData = resultLeaves[leafpos]->buffer().data();
        const auto* rhsData = rhsLeaves[leafpos]->buffer().data();

        const auto* invDiagLeaf = mApplyOperator->mInvDIagonal->tree().probeConstLeaf(leaf.origin());
        const float* invDiagData = nullptr;
        if (invDiagLeaf) {
            invDiagData = invDiagLeaf->buffer().data();
        }

        __m256& defaultInvDiag = mApplyOperator->mDefaultInvDiagonal;
        __m256& packedWeight = mApplyOperator->mPackedWeightJacobi;

        __m256 result256;
        __m256 thisInvDiag;
        for (auto vectorOffset = 0; vectorOffset < leaf.SIZE; vectorOffset += 8) {
            const uint8_t mask = leaf.getValueMask().getWord<uint8_t>(vectorOffset / 8);
            if (mask == uint8_t(0)) {
                //there is no diagonal entry in this vector
                continue;
            }
            result256 = _mm256_loadu_ps(rhsData + vectorOffset);
            vdb_SIMD_IO::get_simd_vector(thisInvDiag, invDiagData, vectorOffset, defaultInvDiag);
            result256 = _mm256_mul_ps(thisInvDiag, result256);
            result256 = _mm256_mul_ps(packedWeight, result256);
            _mm256_storeu_ps(resultData + vectorOffset, result256);
        }
    };//end set_first_iter_op
    mDofLeafManager->foreach(setResultAfterFirstIter);
    _mm256_zeroall();
}

void LaplacianWithLevel::setGridToResultAfterFirstSPAI0(openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs)
{
    //assume the input has the same topology 
    //apply SPAI iteration to result = (L^-1) rhs
    //assume the initial guess of result was 0
    //This amounts to v^1 =SPAI0(b-A*0)=(SPAI0)*rhs
    _mm256_zeroall();

    auto setResultAfterFirstIter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        float* resultData = in_out_grid->tree().probeLeaf(leaf.origin())->buffer().data();
        const float* rhsData = in_rhs->tree().probeConstLeaf(leaf.origin())->buffer().data();
        const auto* SPAI0Leaf = mApplyOperator->mSPAI0->tree().probeConstLeaf(leaf.origin());
        const float* SPAI0_data = nullptr;
        if (SPAI0Leaf) {
            SPAI0_data = SPAI0Leaf->buffer().data();
        }
        __m256 defaultSPAI0 = mApplyOperator->mDefaultSPAI0;
        __m256 result256;
        __m256 SPAI0_256;
        for (auto vectorOffset = 0; vectorOffset < leaf.SIZE; vectorOffset += 8) {
            result256 = _mm256_loadu_ps(rhsData + vectorOffset);
            vdb_SIMD_IO::get_simd_vector(SPAI0_256, SPAI0_data, vectorOffset, defaultSPAI0);
            result256 = _mm256_mul_ps(SPAI0_256, result256);
            _mm256_storeu_ps(resultData + vectorOffset, result256);
        }

        for (auto iter = leaf.beginValueOff(); iter; ++iter) {
            resultData[iter.offset()] = 0.f;
        }
    };//end set_first_iter_op

    mDofLeafManager->foreach(setResultAfterFirstIter);
    _mm256_zeroall();
}

template<bool redFirst>
void LaplacianWithLevel::setGridToResultAfterFirstRBGS(openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs)
{
    //assume the initial guess is zero
    //then the first iteration only get red dof updated
    _mm256_zeroall();

    auto setResultAfterFirstIter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        float* resultData = in_out_grid->tree().probeLeaf(leaf.origin())->buffer().data();
        const float* rhsData = in_rhs->tree().probeConstLeaf(leaf.origin())->buffer().data();
        const auto* invDiagLeaf = mApplyOperator->mInvDIagonal->tree().probeConstLeaf(leaf.origin());
        const float* invDiagData = nullptr;
        if (invDiagLeaf) {
            invDiagData = invDiagLeaf->buffer().data();
        }

        __m256 defaultInvDiag = mApplyOperator->mDefaultInvDiagonal;
        __m256 packedWeightSOR = mApplyOperator->mPackedWeightSOR;

        __m256 result256;
        __m256 thisInvDiag;
        for (auto vector_offset = 0; vector_offset < leaf.SIZE; vector_offset += 8) {
            bool vector_even = ((vector_offset >> 3) + (vector_offset >> 6)) % 2 == 0;
            result256 = _mm256_loadu_ps(rhsData + vector_offset);
            vdb_SIMD_IO::get_simd_vector(thisInvDiag, invDiagData, vector_offset, defaultInvDiag);
            result256 = _mm256_mul_ps(thisInvDiag, result256);
            result256 = _mm256_mul_ps(packedWeightSOR, result256);

            if (redFirst) {
                //only update
                if (vector_even) {
                    result256 = _mm256_blend_ps(_mm256_setzero_ps(), result256, 0b01010101);
                }
                else {
                    result256 = _mm256_blend_ps(_mm256_setzero_ps(), result256, 0b10101010);
                }
            }
            else {
                if (vector_even) {
                    result256 = _mm256_blend_ps(_mm256_setzero_ps(), result256, 0b10101010);
                }
                else {
                    result256 = _mm256_blend_ps(_mm256_setzero_ps(), result256, 0b01010101);
                }
            }
            _mm256_storeu_ps(resultData + vector_offset, result256);
        }

        for (auto iter = leaf.beginValueOff(); iter; ++iter) {
            resultData[iter.offset()] = 0.f;
        }
    };//end set_first_iter_op

    mDofLeafManager->foreach(setResultAfterFirstIter);

    if (redFirst) {
        mApplyOperator->setBlackGaussSeidelMode();
    }
    else {
        mApplyOperator->setRedGaussSeidelMode();
    }

    auto op = mApplyOperator->getLightWeightApplier(in_out_grid, in_out_grid, in_rhs);
    mApplyOperator->Apply(op);

    _mm256_zeroall();
}

void LaplacianWithLevel::gridToVector(
    std::vector<float>& out_vector, openvdb::FloatGrid::Ptr in_grid)
{
    out_vector.resize(mNumDof);

    //it is assumed that the input grid has exactly the same topology as the DOF grid.
    auto setResult = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* valueLeaf = in_grid->tree().probeConstLeaf(leaf.origin());
        if (valueLeaf) {
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                if (leaf.isValueOn(offset)) {
                    out_vector[leaf.getValue(offset)] = valueLeaf->getValue(offset);
                }
            }
        }
    };
    mApplyOperator->mDofLeafManager.foreach(setResult);
}

void LaplacianWithLevel::vectorToGrid(openvdb::FloatGrid::Ptr out_grid, const std::vector<float>& in_vector)
{
    out_grid->setTree(std::make_shared<openvdb::FloatTree>(mDofIndex->tree(), 0.f, openvdb::TopologyCopy()));
    out_grid->setTransform(mDiagonal->transformPtr());
    vectorToGridUnsafe(out_grid, in_vector);
}

void LaplacianWithLevel::vectorToGridUnsafe(openvdb::FloatGrid::Ptr out_grid, const std::vector<float>& in_vector)
{
    //it is assumed that the input grid has exactly the same dof as the 
    auto setResult = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* valueLeaf = out_grid->tree().probeLeaf(leaf.origin());
        for (auto offset = 0; offset < leaf.SIZE; ++offset) {
            if (leaf.isValueOn(offset)) {
                valueLeaf->setValueOn(offset, in_vector[leaf.getValue(offset)]);
            }
        }
    };
    mApplyOperator->mDofLeafManager.foreach(setResult);
}

void LaplacianWithLevel::restriction(
    openvdb::FloatGrid::Ptr outCoarseGrid,
    openvdb::FloatGrid::Ptr inFineGrid,
    const LaplacianWithLevel& coarseLevel)
{
    //to be use by the dof idx manager at coarse level, the laplacian level
    auto collectFromFine = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* coarseValueLeaf = outCoarseGrid->tree().probeLeaf(leaf.origin());
        //fine voxel:   -4 -3 -2 -1 0 1 2 3 4 5
        //coarse voxel: -2 -2 -1 -1 0 0 1 1 2 2

        //each coarse leaf corresponds to 8 potential fine leaves that are active
        std::array<const openvdb::FloatTree::LeafNodeType*, 8> fineLeaves{ nullptr };
        int fineLeavesCounter = 0;
        auto fineBaseOrigin = openvdb::Coord(leaf.origin().asVec3i() * 2);
        for (int ii = 0; ii < 16; ii += 8) {
            for (int jj = 0; jj < 16; jj += 8) {
                for (int kk = 0; kk < 16; kk += 8) {
                    fineLeaves[fineLeavesCounter++] =
                        inFineGrid->tree().probeConstLeaf(fineBaseOrigin.offsetBy(ii, jj, kk));
                }
            }
        }

        for (auto iter = coarseValueLeaf->beginValueOn(); iter; ++iter) {
            //uint32_t at_fine_leaf = iter.offset();

            auto itercoord = coarseValueLeaf->offsetToLocalCoord(iter.offset());
            uint32_t atFineLeaf = 0;
            if (itercoord[2] >= 4) {
                atFineLeaf += 1;
            }
            if (itercoord[1] >= 4) {
                atFineLeaf += 2;
            }
            if (itercoord[0] >= 4) {
                atFineLeaf += 4;
            }

            //if there is possibly a dof in the fine leaf
            if (auto fine_leaf = fineLeaves[atFineLeaf]) {
                auto fine_base_voxel = openvdb::Coord(iter.getCoord().asVec3i() * 2);
                auto fine_base_offset = fine_leaf->coordToOffset(fine_base_voxel);
                float temp_sum = 0;
                for (int ii = 0; ii < 2; ii++) {
                    for (int jj = 0; jj < 2; jj++) {
                        for (int kk = 0; kk < 2; kk++) {
                            auto fine_offset = fine_base_offset + 64 * ii + 8 * jj + kk;
                            if (fine_leaf->isValueOn(fine_offset)) {
                                temp_sum += fine_leaf->getValue(fine_offset);
                            }
                        }//kk
                    }//jj
                }//ii
                iter.setValue(temp_sum * 0.125f);
            }//if fine leaf
        }//for all coarse on voxels
    };//end collect from fine

    coarseLevel.mDofLeafManager->foreach(collectFromFine);
}

template <bool inplace_add>
void LaplacianWithLevel::prolongation(
    openvdb::FloatGrid::Ptr outFineGrid, openvdb::FloatGrid::Ptr inCoarseGrid, float alpha)
{
    //to be use by the dof idx manager at coarse level, the laplacian level
    auto scatterToFine = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        const auto* coarseLeaf = inCoarseGrid->tree().probeConstLeaf(leaf.origin());
        //fine voxel:   -4 -3 -2 -1 0 1 2 3 4 5
        //coarse voxel: -2 -2 -1 -1 0 0 1 1 2 2

        //each coarse leaf corresponds to 8 potential fine leaves that are active
        std::array<openvdb::FloatTree::LeafNodeType*, 8> fineLeaves{ nullptr };
        int fineLeavesCounter = 0;
        auto fineBaseOrigin = openvdb::Coord(leaf.origin().asVec3i() * 2);
        for (int ii = 0; ii < 16; ii += 8) {
            for (int jj = 0; jj < 16; jj += 8) {
                for (int kk = 0; kk < 16; kk += 8) {
                    fineLeaves[fineLeavesCounter++] =
                        outFineGrid->tree().probeLeaf(fineBaseOrigin.offsetBy(ii, jj, kk));
                }
            }
        }

        for (auto iter = coarseLeaf->cbeginValueOn(); iter; ++iter) {
            //uint32_t at_fine_leaf = iter.offset();

            auto itercoord = coarseLeaf->offsetToLocalCoord(iter.offset());
            uint32_t atFineLeaf = 0;
            if (itercoord[2] >= 4) {
                atFineLeaf += 1;
            }
            if (itercoord[1] >= 4) {
                atFineLeaf += 2;
            }
            if (itercoord[0] >= 4) {
                atFineLeaf += 4;
            }

            float coarseValue = alpha * iter.getValue();
            //if there is possibly a dof in the fine leaf
            if (auto fineLeaf = fineLeaves[atFineLeaf]) {
                auto fineBaseVoxel = openvdb::Coord(iter.getCoord().asVec3i() * 2);
                auto fineBaseOffset = fineLeaf->coordToOffset(fineBaseVoxel);
                for (int ii = 0; ii < 2; ii++) {
                    for (int jj = 0; jj < 2; jj++) {
                        for (int kk = 0; kk < 2; kk++) {
                            auto fineOffset = fineBaseOffset + kk;
                            if (ii) fineOffset += 64;
                            if (jj) fineOffset += 8;
                            if (fineLeaf->isValueOn(fineOffset)) {
                                if (inplace_add) {
                                    fineLeaf->buffer().data()[fineOffset] += coarseValue;
                                }
                                else {
                                    fineLeaf->setValueOnly(fineOffset, coarseValue);
                                }

                            }
                        }//kk
                    }//jj
                }//ii
                //iter.setValue(temp_sum * 0.125f);
            }//if fine leaf
        }//for all coarse on voxels
    };//end collect from fine


    mDofLeafManager->foreach(scatterToFine);
}

void LaplacianWithLevel::trimDefaultNodes()
{
    //if a leaf node has the same value and equal to the default 
    //poisson equation term, remove the node
    float term = -mDt / (mDxThisLevel * mDxThisLevel);
    float diagonalTerm = -6.0f * term;
    float epsilon = 1e-5f;

    trimDefaultNodes(mDiagonal, diagonalTerm, diagonalTerm * epsilon);
    trimDefaultNodes(mXEntry, term, term * epsilon);
    trimDefaultNodes(mYEntry, term, term * epsilon);
    trimDefaultNodes(mZEntry, term, term * epsilon);
}

void LaplacianWithLevel::trimDefaultNodes(
    openvdb::FloatGrid::Ptr inOutGrid, float defaultValue, float epsilon)
{
    std::vector<openvdb::Coord> leafOrigins;
    std::vector<int> isQuasiUniform;
    epsilon = std::abs(epsilon);
    auto leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(inOutGrid->tree());
    leafOrigins.resize(leafman.leafCount());
    isQuasiUniform.resize(leafman.leafCount(), false);

    auto markUniform = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
        leafOrigins[leafpos] = leaf.origin();
        if (!leaf.isValueMaskOn()) {
            isQuasiUniform[leafpos] = false;
            //off-diagonal terms in the air are the same for active values
            //allow such leaves to be trimmed, so off-diagonal leaves
            //only exists near solid boundaries
            //uncomment the return retains such partially active homogeneous
            //leaves
            //return;
        }

        float maxError = 0;
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            float error = std::abs(iter.getValue() - defaultValue);
            if (error > maxError) {
                maxError = error;
            }
        }
        if (maxError <= epsilon) {
            isQuasiUniform[leafpos] = true;
        }
    };//end determine uniform

    leafman.foreach(markUniform);

    for (auto i = 0; i < leafOrigins.size(); ++i) {
        if (isQuasiUniform[i]) {
            delete inOutGrid->tree().stealNode<openvdb::FloatTree::LeafNodeType>(leafOrigins[i], defaultValue, false);
        }
    }
}

void PoissonSolver::constructMultigridHierarchy()
{
    
    int maxCoarsestDOF = 4000;
    while (mMultigridHierarchy.back()->mNumDof > maxCoarsestDOF) {
        //CSim::TimerMan::timer("Step/SIMD/levels/lv" + std::to_string(mMultigridHierarchy.size())).start();
        LaplacianWithLevel::Ptr coarserLevel = std::make_shared<LaplacianWithLevel>(
            *mMultigridHierarchy.back(), LaplacianWithLevel::Coarsening()
            );
        //CSim::TimerMan::timer("Step/SIMD/levels/lv" + std::to_string(mMultigridHierarchy.size())).stop();
        mMultigridHierarchy.push_back(coarserLevel);
    }

    //CSim::TimerMan::timer("Step/SIMD/levels/scratchpad").start();
    //the scratchpad for the v cycle to avoid
    for (int level = 0; level < mMultigridHierarchy.size(); level++) {
        //the solution at each level
        mMuCycleLHSs.push_back(mMultigridHierarchy[level]->getZeroVectorGrid());
        //the right hand side at each level
        mMuCycleRHSs.push_back(mMuCycleLHSs.back()->deepCopy());
        //the temporary result to store the jacobi iteration
        //use std::shared_ptr::swap to change the content
        mMuCycleTemps.push_back(mMuCycleLHSs.back()->deepCopy());
    }
    //CSim::TimerMan::timer("Step/SIMD/levels/scratchpad").stop();
    //CSim::TimerMan::timer("Step/SIMD/levels/solver").start();
    constructCoarsestLevelExactSolver();
    //CSim::TimerMan::timer("Step/SIMD/levels/solver").stop();
    printf("levels: %zd Dof:%d\n", mMultigridHierarchy.size(), mMultigridHierarchy[0]->mNumDof);
}

template<int mu_time, bool skip_first_iter>
void PoissonSolver::muCyclePreconditioner(const openvdb::FloatGrid::Ptr in_out_lhs, const openvdb::FloatGrid::Ptr in_rhs, const int level, int n)
{
    
    auto get_scheduled_weight = [n](int iteration) {
        std::array<float, 3> scheduled_weight;
        if (iteration >= n) {
            return 6.0f / 7.0f;
        }
        if (n == 1) {
            scheduled_weight[0] = 6.0f / 7.0f;
        }
        if (n == 2) {
            scheduled_weight[0] = 1.7319f;
            scheduled_weight[1] = 0.5695f;
        }
        if (n == 3) {
            scheduled_weight[0] = 2.2473f;
            scheduled_weight[1] = 0.8571f;
            scheduled_weight[2] = 0.5296f;
        }
        return scheduled_weight[iteration];
    };

    size_t nlevel = mMultigridHierarchy.size();

    if (level == nlevel - 1) {
        writeCoarsestEigenRhs(mCoarsestEigenRhs, in_rhs);
        mCoarsestEigenSolution = mCoarsestCGSolver->solve(mCoarsestEigenRhs);
        writeCoarsestGridSolution(in_out_lhs, mCoarsestEigenSolution);
        return;
    }

    mMuCycleLHSs[level] = in_out_lhs;
    mMuCycleRHSs[level] = in_rhs;

    if (skip_first_iter) {
        switch (mSmoother) {
        default:
        case SmootherOption::ScheduledRelaxedJacobi:
            //set result after first iter, with zero lhs guess
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(get_scheduled_weight(0));
            mMultigridHierarchy[level]->
                setGridToResultAfterFirstJacobi(mMuCycleTemps[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::WeightedJacobi:
            //set result after first iter, with zero lhs guess
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(6.0f / 7.0f);
            mMultigridHierarchy[level]->
                setGridToResultAfterFirstJacobi(mMuCycleTemps[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::RedBlackGaussSeidel:
            mMultigridHierarchy[level]->
                setGridToResultAfterFirstRBGS(mMuCycleLHSs[level], mMuCycleRHSs[level]);
            break;
        case SmootherOption::SPAI0:
            mMultigridHierarchy[level]->
                setGridToResultAfterFirstSPAI0(mMuCycleTemps[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        }
    }

    for (int i = (skip_first_iter ? 1 : 0); i < n; i++) {
        switch (mSmoother) {
        default:
        case SmootherOption::ScheduledRelaxedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(get_scheduled_weight(i));
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::WeightedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(6.0f / 7.0f);
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::RedBlackGaussSeidel:
            mMultigridHierarchy[level]->redBlackGaussSeidelApply</*red first*/true>(
                mMuCycleLHSs[level], mMuCycleRHSs[level]);
            break;
        case SmootherOption::SPAI0:
            mMultigridHierarchy[level]->spai0Apply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        }
    }

    mMultigridHierarchy[level]->residualApply(
        mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);

    int parent_level = level + 1;
    mMultigridHierarchy[level]->restriction(
        mMuCycleRHSs[parent_level], mMuCycleTemps[level], /*laplacian level*/ *mMultigridHierarchy[parent_level]);

    muCyclePreconditioner<mu_time,/*skip first*/true>(mMuCycleLHSs[parent_level], mMuCycleRHSs[parent_level], parent_level, n);
    for (int mu = 1; mu < mu_time; mu++) {
        muCyclePreconditioner<mu_time,/*skip first*/false>(mMuCycleLHSs[parent_level], mMuCycleRHSs[parent_level], parent_level, n);
    }

    mMultigridHierarchy[parent_level]->prolongation</*inplace add*/true>(
        mMuCycleLHSs[level], mMuCycleLHSs[parent_level]);

    for (int i = 0; i < n; i++) {
        switch (mSmoother) {
        default:
        case SmootherOption::ScheduledRelaxedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(get_scheduled_weight(n - 1 - i));
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::WeightedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(6.0f / 7.0f);
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::RedBlackGaussSeidel:
            mMultigridHierarchy[level]->redBlackGaussSeidelApply</*red first*/false>(
                mMuCycleLHSs[level], mMuCycleRHSs[level]);
            break;
        case SmootherOption::SPAI0:
            mMultigridHierarchy[level]->spai0Apply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        }
    }
}

template<int mu_time>
void PoissonSolver::muCycleIterative(const openvdb::FloatGrid::Ptr in_out_lhs, const openvdb::FloatGrid::Ptr in_rhs, const int level, const int n)
{
    
    auto get_scheduled_weight = [n](int iteration) {
        std::array<float, 3> scheduled_weight;
        scheduled_weight.fill(6.0f / 7.0f);
        if (iteration >= n) {
            return 6.0f / 7.0f;
        }
        if (n == 2) {
            scheduled_weight[0] = 1.7319f;
            scheduled_weight[1] = 0.5695f;
        }
        if (n >= 3) {
            scheduled_weight[0] = 2.2473f;
            scheduled_weight[1] = 0.8571f;
            scheduled_weight[2] = 0.5296f;
        }
        return scheduled_weight[iteration];
    };

    float sor = 1.0f;

    size_t nlevel = mMultigridHierarchy.size();

    if (level == nlevel - 1) {
        //must be an even number to make sure no actual swap happens
        int large_iter_n = 10 * n;
        for (int i = 0; i < large_iter_n; i++) {
            switch (mSmoother) {
            default:
            case SmootherOption::ScheduledRelaxedJacobi:
                mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(get_scheduled_weight(i));
                mMultigridHierarchy[level]->weightedJacobiApply(
                    mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
                mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
                break;
            case SmootherOption::WeightedJacobi:
                mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(6.0f / 7.0f);
                mMultigridHierarchy[level]->weightedJacobiApply(
                    mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
                mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
                break;
            case SmootherOption::RedBlackGaussSeidel:
                mMultigridHierarchy[level]->mApplyOperator->set_w_sor(1.0f);
                mMultigridHierarchy[level]->redBlackGaussSeidelApply</*red first*/true>(
                    mMuCycleLHSs[level], mMuCycleRHSs[level]);
                break;
            case SmootherOption::SPAI0:
                mMultigridHierarchy[level]->spai0Apply(
                    mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
                mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
                break;
            }
        }
        return;
    }

    mMuCycleLHSs[level] = in_out_lhs;
    mMuCycleRHSs[level] = in_rhs;

    for (int i = 0; i < n; i++) {
        switch (mSmoother) {
        default:
        case SmootherOption::ScheduledRelaxedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(get_scheduled_weight(i));
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::WeightedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(6.0f / 7.0f);
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::RedBlackGaussSeidel:
            mMultigridHierarchy[level]->mApplyOperator->set_w_sor(1.0f);
            mMultigridHierarchy[level]->redBlackGaussSeidelApply</*red first*/true>(
                mMuCycleLHSs[level], mMuCycleRHSs[level]);
            break;
        case SmootherOption::SPAI0:
            mMultigridHierarchy[level]->spai0Apply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        }
    }

    mMultigridHierarchy[level]->residualApply(
        mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);

    int parent_level = level + 1;
    mMultigridHierarchy[level]->restriction(
        mMuCycleRHSs[parent_level], mMuCycleTemps[level], /*laplacian level*/ *mMultigridHierarchy[parent_level]);

    for (int mu = 0; mu < mu_time; mu++) {
        muCycleIterative<mu_time>(mMuCycleLHSs[parent_level], mMuCycleRHSs[parent_level], parent_level, n);
    }

    mMultigridHierarchy[parent_level]->prolongation</*inplace add*/true>(
        mMuCycleLHSs[level], mMuCycleLHSs[parent_level],/*scaling*/0.5f);

    for (int i = 0; i < n; i++) {
        switch (mSmoother) {
        default:
        case SmootherOption::ScheduledRelaxedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(get_scheduled_weight(n - 1 - i));
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::WeightedJacobi:
            mMultigridHierarchy[level]->mApplyOperator->setWeightJacobi(6.0f / 7.0f);
            mMultigridHierarchy[level]->weightedJacobiApply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        case SmootherOption::RedBlackGaussSeidel:
            mMultigridHierarchy[level]->mApplyOperator->set_w_sor(1.0f);
            mMultigridHierarchy[level]->redBlackGaussSeidelApply</*red first*/false>(
                mMuCycleLHSs[level], mMuCycleRHSs[level]);
            break;
        case SmootherOption::SPAI0:
            mMultigridHierarchy[level]->spai0Apply(
                mMuCycleTemps[level], mMuCycleLHSs[level], mMuCycleRHSs[level]);
            mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
            break;
        }
    }
}



void PoissonSolver::constructCoarsestLevelExactSolver()
{
    std::vector<Eigen::Triplet<float>> triplets;
    mMultigridHierarchy.back()->getTriplets(triplets);

    int ndof = mMultigridHierarchy.back()->mNumDof;
    mCoarsestEigenMatrix.resize(ndof, ndof);

    mCoarsestEigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    //mCoarsestDirectSolver = std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>>>(mCoarsestEigenMatrix);
    mCoarsestCGSolver = std::make_shared<Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>>(mCoarsestEigenMatrix);
    mCoarsestCGSolver->setMaxIterations(10);
}

void PoissonSolver::writeCoarsestEigenRhs(Eigen::VectorXf& out_eigen_rhs, openvdb::FloatGrid::Ptr in_rhs)
{
    out_eigen_rhs.setZero(mMultigridHierarchy.back()->mNumDof);

    auto set_eigen_rhs = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto in_rhs_leaf = in_rhs->tree().probeConstLeaf(leaf.origin());

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            out_eigen_rhs[iter.getValue()] = in_rhs_leaf->getValue(iter.offset());
        }
    };
    mMultigridHierarchy.back()->mDofLeafManager->foreach(set_eigen_rhs);
}

void PoissonSolver::writeCoarsestGridSolution(openvdb::FloatGrid::Ptr in_out_result, const Eigen::VectorXf& in_eigen_solution)
{
    auto set_grid_solution = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* result_leaf = in_out_result->tree().probeLeaf(leaf.origin());

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            result_leaf->setValueOn(iter.offset(), in_eigen_solution[iter.getValue()]);
        }
    };
    mMultigridHierarchy.back()->mDofLeafManager->foreach(set_grid_solution);
}


int PoissonSolver::solveMultigridPCG(openvdb::FloatGrid::Ptr in_out_presssure, openvdb::FloatGrid::Ptr in_rhs)
{
    
    auto& level0 = *mMultigridHierarchy[0];
    mIterationTaken = 0;

    //according to mcadams algorithm 3

    //line2
    auto r = level0.getZeroVectorGrid();
    level0.residualApply(r, in_out_presssure, in_rhs);
    float nu = levelAbsMax(r);
    float initAbsoluteError = nu + 1e-16f;
    float numax = mRelativeTolerance * nu; //numax = std::min(numax, 1e-7f);
    printf("init error%e\n", nu/initAbsoluteError);
    //line3
    if (nu <= numax) {
        printf("iter:%d err:%e\n", mIterationTaken + 1, nu/initAbsoluteError);
        return PoissonSolver::SUCCESS;
    }

    //line4
    auto p = level0.getZeroVectorGrid();
    level0.setGridToConstant(p, 0);
    muCyclePreconditioner<2, true>(p, r, 0, 3);
    float rho = levelDot(p, r);

    auto z = level0.getZeroVectorGrid();
    //line 5
    float nu_old = nu;
    for (; mIterationTaken < mMaxIteration; mIterationTaken++) {
        //line6
        level0.laplacianApply(z, p);
        float sigma = levelDot(p, z);
        //line7
        float alpha = rho / sigma;
        //line8
        levelAlphaXPlusY(-alpha, z, r);
        nu_old = nu;
        nu = levelAbsMax(r); printf("iter:%d err:%e\n", mIterationTaken + 1, nu/initAbsoluteError);
        //line9
        if (nu <= numax) {
            //line10
            levelAlphaXPlusY(alpha, p, in_out_presssure);
            //line11
            //printf("iter:%d err:%e\n", mIterationTaken, nu);
            return PoissonSolver::SUCCESS;
            //line12
        }
        if (nu > nu_old && mIterationTaken > 3) {
            return PoissonSolver::FAILED;
        }
        //line13
        level0.setGridToConstant(z, 0);
        muCyclePreconditioner<2, true>(z, r, 0, 3);

        float rho_new = levelDot(z, r);

        //line14
        float beta = rho_new / rho;

        //line15
        rho = rho_new;
        //line16
        levelAlphaXPlusY(alpha, p, in_out_presssure);
        levelXPlusAlphaY(beta, z, p);
        //line17
    }

    //line18
    return PoissonSolver::FAILED;
}

int PoissonSolver::solvePureMultigrid(openvdb::FloatGrid::Ptr in_out_presssure, openvdb::FloatGrid::Ptr in_rhs)
{
    
    auto& level0 = *mMultigridHierarchy[0];
    mIterationTaken = 0;

    //according to mcadams algorithm 3

    //line2
    auto r = level0.getZeroVectorGrid();
    level0.residualApply(r, in_out_presssure, in_rhs);
    float nu = levelAbsMax(r);
    float initAbsoluteError = nu + 1e-16f;
    float numax = mRelativeTolerance * nu; //numax = std::min(numax, 1e-7f);

    //line3
    if (nu <= numax) {
        //printf("iter:%d err:%e\n", mIterationTaken, nu);
        return PoissonSolver::SUCCESS;
    }
    float nu_old = nu;
    for (; mIterationTaken < mMaxIteration; mIterationTaken++) {
        muCycleIterative<2>(in_out_presssure, in_rhs, 0, 3);
        level0.residualApply(r, in_out_presssure, in_rhs);
        nu_old = nu;
        nu = levelAbsMax(r);
        printf("iter:%d err:%e\n", mIterationTaken, nu/initAbsoluteError);
        if (nu <= numax) {
            //printf("iter:%d err:%e\n", mIterationTaken, nu);
            return PoissonSolver::SUCCESS;
        }
        if (nu > nu_old) {
            if (mIterationTaken > 8) {
                return PoissonSolver::FAILED;
            }
        }
    }

    return PoissonSolver::FAILED;
}


float PoissonSolver::levelAbsMax(openvdb::FloatGrid::Ptr in_lv0_grid, int level)
{
    
    auto op{ grid_abs_max_op(in_lv0_grid) };
    mMultigridHierarchy[level]->mDofLeafManager->reduce(op);
    return op.m_max;
}

float PoissonSolver::levelDot(openvdb::FloatGrid::Ptr a, openvdb::FloatGrid::Ptr b, int level)
{
    
    auto op{ grid_dot_op{ a, b} };
    mMultigridHierarchy[level]->mDofLeafManager->reduce(op);
    return op.dp_result;
}

void PoissonSolver::levelAlphaXPlusY(const float alpha, openvdb::FloatGrid::Ptr in_x, openvdb::FloatGrid::Ptr in_out_y, int level)
{
    
    //y = a*x + y
    auto add_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* xleaf = in_x->tree().probeConstLeaf(leaf.origin());
        auto* yleaf = in_out_y->tree().probeLeaf(leaf.origin());

        const float* xdata = xleaf->buffer().data();
        float* ydata = yleaf->buffer().data();
        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            ydata[iter.offset()] += alpha * xdata[iter.offset()];
        }
    };//end add_op

    mMultigridHierarchy[level]->mDofLeafManager->foreach(add_op);
}

void PoissonSolver::levelXPlusAlphaY(const float alpha, openvdb::FloatGrid::Ptr in_x, openvdb::FloatGrid::Ptr in_out_y, int level)
{
    
    //y = x + a*y
    auto add_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
        auto* xleaf = in_x->tree().probeConstLeaf(leaf.origin());
        auto* yleaf = in_out_y->tree().probeLeaf(leaf.origin());

        const float* xdata = xleaf->buffer().data();
        float* ydata = yleaf->buffer().data();
        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            ydata[iter.offset()] = xdata[iter.offset()] + alpha * ydata[iter.offset()];
        }
    };//end add_op

    mMultigridHierarchy[level]->mDofLeafManager->foreach(add_op);
}

}//end namespace simd_uaamg