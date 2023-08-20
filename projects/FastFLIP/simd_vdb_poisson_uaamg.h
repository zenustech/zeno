#ifndef SIMD_UAAMG_SIMD_VDB_POISSON
#define SIMD_UAAMG_SIMD_VDB_POISSON

#include "Eigen/Eigen"
#include "openvdb/openvdb.h"
#include "openvdb/tree/LeafManager.h"

namespace simd_uaamg {
struct alignas(32) LaplacianApplySIMD;

//7 point Laplacian for the Poisson's equation using Ng 2009 method
//An efficient fluid�solid coupling algorithm for single-phase flows
struct LaplacianWithLevel {
    /*
                  ^     y     |
                  |           |
                  |     d     |
                  |           |
                  |           |
      --------------------------------------
                  |           |
                  |           |
           a     mx     x     |      c
                  |           |
                  |           |
      -----------------my-------------------->x
                  |           |
                  |           |
                  |     b     |
                  |           |

      x indicates the location of unknown pressure variable
      mx and my indicates minus x and minus y faces
      a,b,c,d indicate the neighbor cells

      each cell carries the diagonal entry for x
      and face weight for the minus x and minus y (minus z for 3d).
      face weight = 1 if that face is purely liquid
      face weight = 0 if that face is purely solid

      for each of the cell in {a,b,c,d},
      if it is liquid voxel, add the term = (weight(x,{a,b,c,d}) * dt/(dx*dx))
      to the diagonal entry

      if it is an air voxel, add term / (fraction_liquid(x,{a,b,c,d})
      to the diagonal entry

      When this laplacian is applied to some value in x
      It scatters diagonal to x, and (- term({a,b,c,d})) to {a,b,c,d} if it is liquid
     */
    using Ptr = std::shared_ptr<LaplacianWithLevel>;
    struct Coarsening {};

    static Ptr createPressurePoissonLaplacian(
        openvdb::FloatGrid::Ptr in_liquid_phi,
        openvdb::Vec3fGrid::Ptr in_face_weights,
        const float in_dt);

    openvdb::FloatGrid::Ptr createPressurePoissonRightHandSide(
        openvdb::Vec3fGrid::Ptr in_face_weight,
        openvdb::FloatGrid::Ptr in_vx,
        openvdb::FloatGrid::Ptr in_vy,
        openvdb::FloatGrid::Ptr in_vz,
        openvdb::Vec3fGrid::Ptr in_solid_velocity,
        float in_dt);
    
    openvdb::FloatGrid::Ptr createPressurePoissonRightHandSide_withTension(
        openvdb::FloatGrid::Ptr in_liquid_phi,
        openvdb::FloatGrid::Ptr in_curvature,
        openvdb::Vec3fGrid::Ptr in_face_weight,
        openvdb::FloatGrid::Ptr in_vx,
        openvdb::FloatGrid::Ptr in_vy,
        openvdb::FloatGrid::Ptr in_vz,
        openvdb::Vec3fGrid::Ptr in_solid_velocity,
        float in_dt, float in_tension_coef);

    //Get (row, col, value) triplets for explicit matrix representation.
    void getTriplets(std::vector<Eigen::Triplet<float>>& out_triplets) const;

    //constructor from liquid phi, vector face weights, dx, dt
    LaplacianWithLevel(openvdb::FloatGrid::Ptr in_liquid_phi,
        openvdb::Vec3fGrid::Ptr in_face_weights,
        const float in_dt,
        const float in_dx);

    //construct the coarse level laplacian
    LaplacianWithLevel(const LaplacianWithLevel& child, Coarsening);

    void initializeFromFineLevel(const LaplacianWithLevel& child);

    void initializeFinest(openvdb::FloatGrid::Ptr in_liquid_phi,
        openvdb::Vec3fGrid::Ptr in_face_weights);

    //it assumes the dof idx tree already has the same topology as the
    //full diagonal tree before trim happens
    void setDofIndex(openvdb::Int32Grid::Ptr in_out_dofidx);

    void initializeApplyOperator();

    void residualApply(openvdb::FloatGrid::Ptr in_out_residual, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs);
    void laplacianApply(openvdb::FloatGrid::Ptr in_out_result, openvdb::FloatGrid::Ptr in_lhs);
    void weightedJacobiApply(openvdb::FloatGrid::Ptr in_out_updated_lhs, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs);
    void spai0Apply(openvdb::FloatGrid::Ptr in_out_updated_lhs, openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs);
    template<bool red_first>
    void redBlackGaussSeidelApply(openvdb::FloatGrid::Ptr in_out_lhs, openvdb::FloatGrid::Ptr in_rhs);

    //return a shared pointer of a float grid that matches the mask of diagonal
    openvdb::FloatGrid::Ptr getZeroVectorGrid();

    void setGridToConstant(openvdb::FloatGrid::Ptr in_out_grid, float constant);

    void setGridToResultAfterFirstJacobi(openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs);
    void setGridToResultAfterFirstSPAI0(openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs);
    template<bool redFirst = true>
    void setGridToResultAfterFirstRBGS(openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs);

    //turn the input grid to std vector of length ndof
    void gridToVector(std::vector<float>& out_vector, openvdb::FloatGrid::Ptr in_grid);
    void vectorToGrid(openvdb::FloatGrid::Ptr out_grid, const std::vector<float>& in_vector);
    //assume the out_grid already has the same topology as the degree of freedom index tree
    void vectorToGridUnsafe(openvdb::FloatGrid::Ptr out_grid, const std::vector<float>& in_vector);

    void restriction(openvdb::FloatGrid::Ptr out_coarse_grid, openvdb::FloatGrid::Ptr in_fine_grid, const LaplacianWithLevel& parent);
    template <bool inplace_add>
    void prolongation(openvdb::FloatGrid::Ptr out_fine_grid, openvdb::FloatGrid::Ptr in_coarse_grid, float alpha = 1.0f);

    void trimDefaultNodes();
    static void trimDefaultNodes(openvdb::FloatGrid::Ptr in_out_grid, float default_val, float epsilon);

    openvdb::FloatGrid::Ptr mDiagonal;
    openvdb::FloatGrid::Ptr mXEntry;
    openvdb::FloatGrid::Ptr mYEntry;
    openvdb::FloatGrid::Ptr mZEntry;

    openvdb::Int32Grid::Ptr mDofIndex;

    std::unique_ptr<openvdb::tree::LeafManager<openvdb::Int32Tree>> mDofLeafManager;
    std::shared_ptr<LaplacianApplySIMD> mApplyOperator;

    int mNumDof;

    float mDt;
    //dx at this level
    float mDxThisLevel;

    //level=0 means the finest grid for Laplacian matrix
    int mLevel;
};//end Laplacian with level

//use simd intrinsics to solve the poisson's equation
class PoissonSolver {
public:
    enum class SmootherOption {
        ScheduledRelaxedJacobi,
        WeightedJacobi,
        RedBlackGaussSeidel,
        SPAI0
    };
    using SuccessType = int;
    static const SuccessType SUCCESS = 0;
    static const SuccessType FAILED = 1;

    PoissonSolver(LaplacianWithLevel::Ptr in_finest_level_matrix) {
        mMultigridHierarchy.push_back(in_finest_level_matrix);
        constructMultigridHierarchy();
        mIterationTaken = 0;
        mMaxIteration = 100;
        mRelativeTolerance = 1e-7f;
        mSmoother = SmootherOption::ScheduledRelaxedJacobi;
    }

    SuccessType solveMultigridPCG(openvdb::FloatGrid::Ptr in_out_presssure, openvdb::FloatGrid::Ptr in_rhs);
    SuccessType solvePureMultigrid(openvdb::FloatGrid::Ptr in_out_presssure, openvdb::FloatGrid::Ptr in_rhs);

    int mIterationTaken;
    int mMaxIteration;
    float mRelativeTolerance;
    SmootherOption mSmoother;
    std::vector<LaplacianWithLevel::Ptr> mMultigridHierarchy;

private:
    //In the preconditioner version, the parent level Poisson matrix is effectively multiplied by 0.5
    //Therefore the propagated back error correction is about twice as large as 
    //the one obtained by solving the matrix built upon Galerkin coarsening principle
    //This helps accelerating preconditioned CG to converge faster
    //but the MG solver itself may not be convergent
    template<int mu_time, bool skip_first_iter>
    void muCyclePreconditioner(const openvdb::FloatGrid::Ptr in_out_lhs, const openvdb::FloatGrid::Ptr in_rhs, const int level, const int n);

    //If the mucycle is used as an iterative solver
    //The propagated error is scaled by 0.5 to cancel the effect of coarsening scaling of parent matrix
    template<int mu_time>
    void muCycleIterative(const openvdb::FloatGrid::Ptr in_out_lhs, const openvdb::FloatGrid::Ptr in_rhs, const int level, const int n, int postSmooth = 0);

    void constructMultigridHierarchy();
    void constructCoarsestLevelExactSolver();
    void writeCoarsestEigenRhs(Eigen::VectorXf& out_eigen_rhs, openvdb::FloatGrid::Ptr in_rhs);
    void writeCoarsestGridSolution(openvdb::FloatGrid::Ptr in_out_result, const Eigen::VectorXf& in_eigen_solution);

    float levelAbsMax(openvdb::FloatGrid::Ptr in_lv0_grid, int level = 0);
    float levelDot(openvdb::FloatGrid::Ptr a, openvdb::FloatGrid::Ptr b, int level = 0);
    //y = a*x+y;
    void levelAlphaXPlusY(const float alpha, openvdb::FloatGrid::Ptr in_x, openvdb::FloatGrid::Ptr in_out_y, int level = 0);
    //y = x + a*y;
    void levelXPlusAlphaY(const float alpha, openvdb::FloatGrid::Ptr in_x, openvdb::FloatGrid::Ptr in_out_y, int level = 0);

    //the direct solver for the coarsest level
    Eigen::SparseMatrix<float> mCoarsestEigenMatrix;
    Eigen::VectorXf mCoarsestEigenRhs, mCoarsestEigenSolution;
    std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>>> mCoarsestDirectSolver;
    std::shared_ptr<Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>> mCoarsestCGSolver;


    std::vector<openvdb::FloatGrid::Ptr> mMuCycleLHSs;
    std::vector<openvdb::FloatGrid::Ptr> mMuCycleRHSs;
    std::vector<openvdb::FloatGrid::Ptr> mMuCycleTemps;
};
}//end namespace simd_uaamg



#endif