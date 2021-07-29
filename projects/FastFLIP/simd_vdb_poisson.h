#pragma once
#include "Eigen/Eigen"
#include "openvdb/openvdb.h"
#include "openvdb/tree/LeafManager.h"
struct alignas(32) simd_laplacian_apply_op;

// use simd intrinsics to solve the poisson's equation
class simd_vdb_poisson {
public:
  using float_leaf_t = openvdb::FloatTree::LeafNodeType;
  using int_leaf_t = openvdb::Int32Tree::LeafNodeType;

  // 7 point Laplacian for the Poisson's equation using Ng 2009 method
  // An efficient fluid�solid coupling algorithm for single-phase flows
  struct Laplacian_with_level {
    /*
                    ^  y        |
                    |           |
                                    |     d     |
                                    |           |
                                    |           |
            --------------------------------------
                        |           |
                                    |           |
                    a      mx     x     |      c
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
            if it is liquid voxel, add the term = (weight(x,{a,b,c,d}) *
       dt/(dx*dx)) to the diagonal entry
            if it is an air voxel, add term / (fraction_liquid(x,{a,b,c,d})
            to the diagonal entry
            When this laplacian is applied to some value in x
            It scatters diagonal to x, and (- term({a,b,c,d})) to {a,b,c,d} if
       it is liquid
     */

    using Ptr = std::shared_ptr<Laplacian_with_level>;
    struct coarsening {};
    // constructor from liquid phi, vector face weights, dx, dt
    Laplacian_with_level(openvdb::FloatGrid::Ptr in_liquid_phi,
                         openvdb::Vec3fGrid::Ptr in_face_weights,
                         const float in_dt, const float in_dx);

    // construct the coarse level laplacian
    Laplacian_with_level(const Laplacian_with_level &parent, coarsening);

    void initialize_entries_from_parent(const Laplacian_with_level &parent);

    void initialize_finest(openvdb::FloatGrid::Ptr in_liquid_phi,
                           openvdb::Vec3fGrid::Ptr in_face_weights);

    // it assumes the dof idx tree already has the same topology as the
    // full diagonal tree before trim happens
    void set_dof_idx(openvdb::Int32Grid::Ptr in_out_dofidx);

    void initialize_evaluators();

    // return a shared pointer of the result of L*in_rhs where L is the
    // laplacian matrix
    openvdb::FloatGrid::Ptr apply(openvdb::FloatGrid::Ptr in_rhs);

    void residual_apply_assume_topo(openvdb::FloatGrid::Ptr in_out_residual,
                                    openvdb::FloatGrid::Ptr in_lhs,
                                    openvdb::FloatGrid::Ptr in_rhs);
    void Laplacian_apply_assume_topo(openvdb::FloatGrid::Ptr in_out_result,
                                     openvdb::FloatGrid::Ptr in_lhs);
    void weighted_jacobi_apply_assume_topo(
        openvdb::FloatGrid::Ptr in_out_updated_lhs,
        openvdb::FloatGrid::Ptr in_lhs, openvdb::FloatGrid::Ptr in_rhs);
    void SPAI0_apply_assume_topo(openvdb::FloatGrid::Ptr in_out_updated_lhs,
                                 openvdb::FloatGrid::Ptr in_lhs,
                                 openvdb::FloatGrid::Ptr in_rhs);
    template <bool red_first>
    void RBGS_apply_assume_topo_inplace(openvdb::FloatGrid::Ptr scratch_pad,
                                        openvdb::FloatGrid::Ptr in_out_lhs,
                                        openvdb::FloatGrid::Ptr in_rhs);
    void inplace_add_assume_topo(openvdb::FloatGrid::Ptr in_out_result,
                                 openvdb::FloatGrid::Ptr in_rhs);

    // return a shared pointer of a float grid that matches the mask of diagonal
    openvdb::FloatGrid::Ptr get_zero_vec_grid();

    void set_grid_constant_assume_topo(openvdb::FloatGrid::Ptr in_out_grid,
                                       float constant);

    void set_grid_to_result_after_first_jacobi_assume_topo(
        openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs);
    void set_grid_to_result_after_first_SPAI_assume_topo(
        openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs);
    template <bool red_first = true>
    void set_grid_to_result_after_first_RBGS_assume_topo(
        openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs);
    // turn the input grid to std vector of length ndof
    void grid2vector(std::vector<float> &out_vector,
                     openvdb::FloatGrid::Ptr in_grid);

    void vector2grid(openvdb::FloatGrid::Ptr out_grid,
                     const std::vector<float> &in_vector);

    // assume the out_grid already has the same topology as the degree of
    // freedom index tree
    void vector2grid_assume_topo(openvdb::FloatGrid::Ptr out_grid,
                                 const std::vector<float> &in_vector);

    void restriction(openvdb::FloatGrid::Ptr out_coarse_grid,
                     openvdb::FloatGrid::Ptr in_fine_grid,
                     const Laplacian_with_level &child);
    template <bool inplace_add>
    void prolongation(openvdb::FloatGrid::Ptr out_fine_grid,
                      openvdb::FloatGrid::Ptr in_coarse_grid,
                      const Laplacian_with_level &parent);

    void trim_default_nodes();
    static void trim_default_nodes(openvdb::FloatGrid::Ptr in_out_grid,
                                   float default_val, float epsilon);

    openvdb::FloatGrid::Ptr m_Diagonal;
    openvdb::FloatGrid::Ptr m_Neg_x_entry;
    openvdb::FloatGrid::Ptr m_Neg_y_entry;
    openvdb::FloatGrid::Ptr m_Neg_z_entry;

    openvdb::Int32Grid::Ptr m_dof_idx;

    std::unique_ptr<openvdb::tree::LeafManager<openvdb::Int32Tree>>
        m_dof_leafmanager;
    std::shared_ptr<simd_laplacian_apply_op> m_laplacian_evaluator;

    int m_ndof;

    float m_dt;
    // dx at this level
    float m_dx_this_level;
    // dx at the finest level
    float m_dx_finest;

    float m_diag_entry_min_threshold;

    // level=0 means the finest grid for Laplacian matrix
    int m_level;

    // only if the coarse level is constructed
    // from the same finest level make sense
    // this will be randomly generated
    float m_root_token;
  }; // end Laplacian with level

  simd_vdb_poisson(openvdb::FloatGrid::Ptr in_liquid_sdf,
                   openvdb::Vec3fGrid::Ptr in_face_weight,
                   openvdb::Vec3fGrid::Ptr in_velocity,
                   openvdb::Vec3fGrid::Ptr in_solid_velocity, float in_dt,
                   float in_dx) {
    m_liquid_sdf = in_liquid_sdf;
    m_face_weight = in_face_weight;
    m_velocity = in_velocity;
    m_solid_velocity = in_solid_velocity;
    dt = in_dt;
    m_dx = in_dx;
    m_iteration = 0;
    m_max_iter = 20;
  }
  std::vector<Laplacian_with_level::Ptr> m_laplacian_with_levels;
  std::vector<openvdb::FloatGrid::Ptr> m_v_cycle_lhss;
  std::vector<openvdb::FloatGrid::Ptr> m_v_cycle_rhss;
  std::vector<openvdb::FloatGrid::Ptr> m_v_cycle_temps;
  std::vector<openvdb::FloatGrid::Ptr> m_K_cycle_cs;
  std::vector<openvdb::FloatGrid::Ptr> m_K_cycle_vs;
  std::vector<openvdb::FloatGrid::Ptr> m_K_cycle_ds;
  std::vector<openvdb::FloatGrid::Ptr> m_K_cycle_ws;
  void construct_levels();
  void Vcycle(const openvdb::FloatGrid::Ptr in_out_lhs,
              const openvdb::FloatGrid::Ptr in_rhs, int n1 = 1, int n2 = 2,
              int ncoarse = 40);
  template <int mu_time, bool skip_first_iter>
  void mucycle_RBGS(const openvdb::FloatGrid::Ptr in_out_lhs,
                    const openvdb::FloatGrid::Ptr in_rhs, const int level,
                    int n1 = 2, int n2 = 2);

  // scheduled relaxed jacobi see Efficient relaxed-Jacobi smoothers for
  // multigrid on parallel computers
  template <int mu_time, bool skip_first_iter>
  void mucycle_SRJ(const openvdb::FloatGrid::Ptr in_out_lhs,
                   const openvdb::FloatGrid::Ptr in_rhs, const int level = 0,
                   int n = 3);

  // SPAI0 smoother mu cycle
  template <int mu_time, bool skip_first_iter>
  void mucycle_SPAI0(const openvdb::FloatGrid::Ptr in_out_lhs,
                     const openvdb::FloatGrid::Ptr in_rhs, const int level = 0,
                     int n = 3);

  template <bool skip_first_iter>
  void Kcycle_SRJ(const openvdb::FloatGrid::Ptr in_out_lhs,
                  const openvdb::FloatGrid::Ptr in_rhs, const int level = 0,
                  int n = 3);

  // the direct solver for the coarsest level
  Eigen::SparseMatrix<float> m_coarsest_eigen_matrix;
  Eigen::VectorXf m_coarsest_eigen_rhs, m_coarsest_eigen_solution;
  std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>>>
      m_coarsest_solver;
  void construct_coarsest_exact_solver();
  void write_coarsest_eigen_rhs(Eigen::VectorXf &out_eigen_rhs,
                                openvdb::FloatGrid::Ptr in_rhs);
  void write_coarsest_grid_solution(openvdb::FloatGrid::Ptr in_out_result,
                                    const Eigen::VectorXf &in_eigen_solution);

  void build_rhs();

  bool pcg_solve(openvdb::FloatGrid::Ptr in_out_presssure, float tolerance);
  void smooth_solve(openvdb::FloatGrid::Ptr in_out_presssure, int n);
  int iterations();
  openvdb::FloatGrid::Ptr m_rhs;
  void symmetry_test(int level = 0);

private:
  float lv_abs_max(openvdb::FloatGrid::Ptr in_lv0_grid, int level = 0);
  float lv_dot(openvdb::FloatGrid::Ptr a, openvdb::FloatGrid::Ptr b,
               int level = 0);
  // y = a*x+y;
  void lv_axpy(const float alpha, openvdb::FloatGrid::Ptr in_x,
               openvdb::FloatGrid::Ptr in_out_y, int level = 0);
  // y = x + a*y;
  void lv_xpay(const float alpha, openvdb::FloatGrid::Ptr in_x,
               openvdb::FloatGrid::Ptr in_out_y, int level = 0);
  // out = a*x + b*y;
  void lv_out_axby(openvdb::FloatGrid::Ptr out_result, const float alpha,
                   const float beta, openvdb::FloatGrid::Ptr in_x,
                   openvdb::FloatGrid::Ptr in_y, int level = 0);
  void lv_copyval(openvdb::FloatGrid::Ptr a, openvdb::FloatGrid::Ptr b,
                  int level = 0);
  int m_iteration;
  int m_max_iter;
  // the grid information used to construct the matrix and right hand side
  openvdb::FloatGrid::Ptr m_liquid_sdf;
  openvdb::Vec3fGrid::Ptr m_face_weight;
  openvdb::Vec3fGrid::Ptr m_velocity, m_solid_velocity;

  float dt;
  float m_dx;
};