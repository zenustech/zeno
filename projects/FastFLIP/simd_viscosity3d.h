#pragma once

#include "../zenvdb/include/zeno/packed3grids.h"
#include "openvdb/openvdb.h"
#include "openvdb/tree/LeafManager.h"
#include "Eigen/Eigen"

#include <immintrin.h>
namespace simd_uaamg {


struct L_with_level {
	enum class working_mode { JACOBI, SPAI0, NORMAL, RESIDUAL, RED_GS, BLACK_GS };
	using sparse_matrix_type = Eigen::SparseMatrix<float, Eigen::RowMajor>;
	//using coarse_solver_type = Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>>;
	using coarse_solver_type = Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::Upper | Eigen::Lower>;
	//using coarse_solver_type = Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::Upper|Eigen::Lower, Eigen::IncompleteCholesky<float>>;
	struct light_weight_applier {

		using leaf_type = openvdb::FloatTree::LeafNodeType;

		using leaf_vec_type = std::vector<leaf_type*>;
		using const_leaf_vec_type = std::vector<const leaf_type*>;
		using leaf_vec_ptr_type = std::shared_ptr<leaf_vec_type>;
		using const_leaf_vec_ptr_type = std::shared_ptr<const_leaf_vec_type>;

		light_weight_applier(L_with_level* parent,
			packed_FloatGrid3 out_result,
			packed_FloatGrid3 in_lhs,
			packed_FloatGrid3 in_rhs, L_with_level::working_mode in_working_mode);

		light_weight_applier(const light_weight_applier& other);

		float* leaf2data(openvdb::FloatTree::LeafNodeType* in_leaf) const {
			if (in_leaf) {
				return in_leaf->buffer().data();
			}
			else {
				return nullptr;
			}
		}

		const float* leaf2data(const openvdb::FloatTree::LeafNodeType* in_leaf) const {
			if (in_leaf) {
				return in_leaf->buffer().data();
			}
			else {
				return nullptr;
			}
		}

		template <int c>
		void component_operator(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const;

		void operator()(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const;

		//channel
		void set_channel(int in_channel) { channel = in_channel; }

		void set_working_mode(L_with_level::working_mode new_working_mode) { m_working_mode = new_working_mode; }

		int channel;

		L_with_level::working_mode m_working_mode;

		leaf_vec_ptr_type m_result_leaves[3];
		const_leaf_vec_ptr_type m_lhs_leaves[3], m_rhs_leaves[3];

		mutable openvdb::FloatGrid::ConstUnsafeAccessor
			m_diag_axr[3], m_inv_diag_axr[3], m_spai0_axr[3],
			m_negx_axr[3], m_negy_axr[3], m_negz_axr[3],
			Txyzd12_axr[3][2][2];

		mutable openvdb::FloatGrid::ConstUnsafeAccessor m_lhs_axr[3];

		L_with_level* m_parent;
	};

	//degree of freedom stored in three integer grid for each component
	struct DOF_tuple_t {
		openvdb::Int32Grid::Ptr idx[3];
	};
	using Ptr = std::shared_ptr<L_with_level>;
	struct Coarsening {};

	static Ptr create_viscosity_matrix(
		openvdb::FloatGrid::Ptr in_viscosity,
		openvdb::FloatGrid::Ptr in_liquid_sdf,
		openvdb::FloatGrid::Ptr in_solid_sdf, float in_dt, float in_rho);

	L_with_level(
		openvdb::FloatGrid::Ptr in_viscosity,
		openvdb::FloatGrid::Ptr in_liquid_sdf,
		openvdb::FloatGrid::Ptr in_solid_sdf, float in_dt, float in_rho);

	L_with_level(const L_with_level& child, L_with_level::Coarsening);

	void initialize_transforms();
	void initialize_grids();
	void calculate_subcell_sdf_volume();
	void calculate_matrix_coefficients_volume();
	void mark_dof();

	void set_dof(openvdb::Int32Grid::Ptr in_out_DOF, int& in_out_dof);
	void calculate_matrix_coefficients();

	void initialize_transforms_from_child(const L_with_level& child);
	void mark_dof_from_child(const L_with_level& child);
	void calculate_matrix_coefficients_from_child(const L_with_level& child);
	void build_SPAI0_matrix();
	void trim_default_nodes();

	packed_FloatGrid3 build_rhs(packed_FloatGrid3 in_velocity_field, openvdb::Vec3fGrid::Ptr in_solid_velocity) const;

	//convert a packed float grid to Eigen vector
	//based on the DOF
	//the input is assumed to have the same topology of the DOfs
	Eigen::VectorXf to_Eigenvector(packed_FloatGrid3 in_float3grid) const;
	void to_Eigenvector(Eigen::VectorXf& out_eigenvector, packed_FloatGrid3 in_float3grid) const;

	//turn a flat Eigen vector to packed float grid, assume the input vector
	//has ndof
	packed_FloatGrid3 to_packed_FloatGrid3(const Eigen::VectorXf& in_vector) const;
	void write_to_FloatGrid3(packed_FloatGrid3 out_grids, const Eigen::VectorXf& in_vector) const;
	void write_to_FloatGrid3_assume_topology(packed_FloatGrid3 out_grids, const Eigen::VectorXf& in_vector) const;

	packed_FloatGrid3 get_zero_vec() const;

	void IO(std::string fname);

	light_weight_applier get_light_weight_applier(packed_FloatGrid3 out_result, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs, L_with_level::working_mode in_working_mode);
	void run(light_weight_applier& Op);
	void L_apply(packed_FloatGrid3 out_result, packed_FloatGrid3 in_lhs);
	void residual_apply(packed_FloatGrid3 out_residual, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs);
	void Jacobi_apply(packed_FloatGrid3 out_updated_lhs, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs);
	void SPAI0_apply(packed_FloatGrid3 out_updated_lhs, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs);
	void XYZ_RBGS_apply(packed_FloatGrid3 out_updated_lhs, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs);
	void ZYX_RBGS_apply(packed_FloatGrid3 out_updated_lhs, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs);



	void set_grid_to_zero(packed_FloatGrid3 in_out_result);
	void set_grid_to_SPAI0_after_first_iteration(packed_FloatGrid3 out_lhs, packed_FloatGrid3 in_rhs);

	void restriction(packed_FloatGrid3 out_coarse_grid, packed_FloatGrid3 in_fine_grid, const L_with_level& parent);
	template <bool inplace_add>
	void prolongation(packed_FloatGrid3 out_fine_grid, packed_FloatGrid3 in_coarse_grid);

	void build_explicit_matrix();
	void get_triplets(std::vector<Eigen::Triplet<float>>& triplets);
	void construct_exact_solver();
	void solve_exact(packed_FloatGrid3 in_out_lhs, packed_FloatGrid3 in_rhs);

	float m_dx;
	float m_dt;
	float m_rho;
	float m_diag_frac_epsl;
	float m_default_diag, m_default_off_diag;
	int m_level;

	float m_damped_jacobi_coef;

	//For algebraic construction of levels, these information should only be used 
	//on the finest level
	openvdb::FloatGrid::Ptr m_viscosity;
	openvdb::FloatGrid::Ptr m_solid_sdf;
	openvdb::FloatGrid::Ptr m_liquid_sdf;

	//voxel vol is at voxel center
	openvdb::FloatGrid::Ptr m_voxel_vol;

	//8 subcell_dof points surround a subcell_vol voxel
	//8 subcell_vol voxels fill a liquid_sdf voxel.
	openvdb::FloatGrid::Ptr m_subcell_sdf;
	openvdb::FloatGrid::Ptr m_subcell_vol;

	//volume for each velocity dof
	//velocity volume is at voxel face center.
	packed_FloatGrid3 m_face_center_vol;

	//tyz is at x axis, txz is at y axis, txy is at z axis.
	//on the edge center
	openvdb::FloatGrid::Ptr m_edge_center_vol[3];

	//if a velocity sample can contribute to the variational integral if its volume is not zero
	//or related stress volume is not zero but it may be inside the solid.
	//If so, it contributes to the right hand side, instead of the equation.
	//The following indicator
	//shows if a channel is inside a solid
	openvdb::BoolGrid::Ptr m_is_solid[3];

	//on voxel indicates it is a DOF in the equation.
	DOF_tuple_t m_velocity_DOF;
	int m_ndof;
	std::unique_ptr<openvdb::tree::LeafManager<openvdb::Int32Tree>> m_dof_manager[3];

	//matrix coefficients
	//diagonal term for three channel
	packed_FloatGrid3 m_diagonal;

	//inverse diagonal for the smoothing procedure.
	packed_FloatGrid3 m_invdiag;

	packed_FloatGrid3 m_SPAI0;
	bool m_SPAI0_initialized;

	//same channel coefficients, between this channel and the channel in the negative direction
	//this channel refers to the bottom faces in this voxel.
	packed_FloatGrid3 m_neg_x, m_neg_y, m_neg_z;

	//cross channel coefficients between faces in a same voxel
	//the coefficients are at the midpoint between adjacent faces
	//corresponding to the same transformations defined below.
	//looking from positive x direction
	//> means Uy, ^ means Uz, * is the voxel center
	// 000,001,010,011 are the four positions of the cross channel terms
	//the starting zero means the face is positive x direction.
	//-------------------------------------
	//|        |        |        |        |
	//|        |        |        |        |
	//|        |        |        |        |
	//---------^--------z--------^---------
	//|        |        |        |        |
	//|        |        |  001   |  011   |
	//|        |        |        |        |
	//>--------*-------->--------*-------->
	//|        |        |        |        |
	//|        |        |  000   |  010   |
	//|        |        |        |        |
	//---------^-------(.)x------^--------y
	//|        |        |        |        |
	//|        |        |        |        |
	//|        |        |        |        |
	//-------------------------------------
	openvdb::FloatGrid::Ptr m_cross_channel[3][2][2];


	//transformation
	//reference voxel corner center across all levels
	openvdb::Vec3f m_voxel_corner_origin;

	openvdb::math::Transform::Ptr m_voxel_center_transform;
	openvdb::math::Transform::Ptr m_voxel_corner_transform;
	//face center transform for velocity channels
	openvdb::math::Transform::Ptr m_face_center_transform[3];
	//edge center transform for the tau_[yz,zx,xy] channels
	openvdb::math::Transform::Ptr m_edge_center_transform[3];
	//subcell corner transform for the subcell liquid sdf
	openvdb::math::Transform::Ptr m_subcell_corner_transform;
	//subcell center transform for the subcell volume
	openvdb::math::Transform::Ptr m_subcell_center_transform;

	//[positive normal direction x,y,z][first axis 0/1][second axis 0/1]
	/*
		//look from positive x direction first axis: y, second axis z
		//   ^z
		//(0,0,1)------(0,1,1)
		//   |            |
		//(0,0,0)------(0,1,0)>y

		//look from positive y direction first axis z, second axis x
		//   ^x
		//(1,0,0)------(1,0,1)
		//   |            |
		//(0,0,0)------(0,0,1)>z

		//look from positive z direction first axis x, second axis y
		//   ^y
		//(0,1,0)------(1,1,0)
		//   |            |
		//(0,0,0)------(1,0,0)>x

		each coefficient is associated with the voxel, not face.
	*/
	openvdb::math::Transform::Ptr m_crosssection_face_transform[3][2][2];

	sparse_matrix_type m_explicit_matrix;

	std::unique_ptr<coarse_solver_type> m_direct_solver;
	Eigen::VectorXf m_direct_solver_lhs, m_direct_solver_rhs;
};

struct simd_viscosity3d {

	simd_viscosity3d(openvdb::FloatGrid::Ptr in_viscosity,
		openvdb::FloatGrid::Ptr in_liquid_sdf,
		openvdb::FloatGrid::Ptr in_solid_sdf,
		packed_FloatGrid3 in_liquid_velocity,
		openvdb::Vec3fGrid::Ptr in_solid_velocity,
		float in_dt, float in_rho);

	simd_viscosity3d(L_with_level::Ptr level0,
		packed_FloatGrid3 in_liquid_velocity,
		openvdb::Vec3fGrid::Ptr in_solid_velocity);

	void pcg_solve(packed_FloatGrid3 in_lhs, float tolerance);

	float lv_abs_max(packed_FloatGrid3 in_grid, int level = 0);

	float lv_dot(packed_FloatGrid3 a, packed_FloatGrid3 b, int level = 0);

	//y = a*x + y;
	void lv_axpy(const float alpha, packed_FloatGrid3 x, packed_FloatGrid3 y, int level = 0);

	//y = x + a*y;
	void lv_xpay(const float alpha, packed_FloatGrid3 x, packed_FloatGrid3 y, int level = 0);

	void lv_copyval(packed_FloatGrid3 out_grid, packed_FloatGrid3 in_grid, int level = 0);

	void mucycle(packed_FloatGrid3 in_out_lhs, const packed_FloatGrid3 in_rhs, const int mu_time, const bool for_precon, const int level = 0, int n = 2);

	std::vector<std::shared_ptr<L_with_level>> m_matrix_levels;

	std::vector<packed_FloatGrid3> m_mucycle_lhss, m_mucycle_rhss, m_mucycle_temps;

	packed_FloatGrid3 m_rhs;

	int m_iteration;
	int m_max_iter;
};


}//end namespace simd_uaamg