#include <unordered_map>
#include "vdb_SIMD_IO.h"
#include "openvdb_grid_math_op.h"
#include "SIMD_UAAMG_Ops.h"
#include "simd_viscosity3d.h"
#include "simd_vdb_poisson_uaamg.h"
#include "openvdb/tree/LeafManager.h"
#include "openvdb/tools/Morphology.h"
#include "openvdb/tools/Interpolation.h"
#include "volume_fractions.h"

namespace simd_uaamg{

L_with_level::Ptr L_with_level::create_viscosity_matrix(
	openvdb::FloatGrid::Ptr in_viscosity,
	openvdb::FloatGrid::Ptr in_liquid_sdf,
	openvdb::FloatGrid::Ptr in_solid_sdf, float in_dt, float in_rho) 
{
	return std::make_shared<L_with_level>(in_viscosity, in_liquid_sdf, in_solid_sdf, in_dt, in_rho);
}

L_with_level::L_with_level(
	openvdb::FloatGrid::Ptr in_viscosity,
	openvdb::FloatGrid::Ptr in_liquid_sdf,
	openvdb::FloatGrid::Ptr in_solid_sdf, float in_dt, float in_rho)
{
	m_dx = (float)in_liquid_sdf->voxelSize()[0];
	m_dt = in_dt;
	m_rho = in_rho;
	m_liquid_sdf = in_liquid_sdf;
	m_solid_sdf = in_solid_sdf;
	m_viscosity = in_viscosity;
	m_diag_frac_epsl = 1e-1f;
	m_SPAI0_initialized = false;
	m_level = 0;
	m_damped_jacobi_coef = 10.0f / 10.0f;
	m_ndof = 0;
	m_voxel_corner_origin = openvdb::Vec3f(-0.5f * m_dx);
	initialize_transforms();
	initialize_grids();
	printf("lv0 subcell volume\n");
	calculate_subcell_sdf_volume();
	printf("lv0 coef volume\n");
	calculate_matrix_coefficients_volume();
	printf("lv0 markdof volume\n");
	mark_dof();
	printf("lv0 genmatrix volume\n");
	calculate_matrix_coefficients();
	printf("lv0 done\n");
}

L_with_level::L_with_level(const L_with_level& child, L_with_level::Coarsening)
{
	m_dx = child.m_dx * 2.0f;
	m_dt = child.m_dt;
	m_rho = child.m_rho;
	m_liquid_sdf = child.m_liquid_sdf;
	m_solid_sdf = child.m_solid_sdf;
	m_viscosity = child.m_viscosity;
	m_diag_frac_epsl = child.m_diag_frac_epsl;
	m_SPAI0_initialized = false;
	m_level = child.m_level + 1;
	m_damped_jacobi_coef = child.m_damped_jacobi_coef;
	m_ndof = 0;
	m_voxel_corner_origin = child.m_voxel_corner_origin;

	//the following values are not valid for child levels, they are only initialized.
	m_default_diag = 1.0f;
	m_default_off_diag = 1.0f;

	initialize_transforms();
	initialize_grids();
	mark_dof_from_child(child);
	//printf("lv%d gen matrix dof%d \n", mLevel, mNumDof);
	calculate_matrix_coefficients_from_child(child);
	//printf("lv%d done\n", mLevel);
	
}

void L_with_level::initialize_transforms()
{
	//voxel corner
	m_voxel_corner_transform = openvdb::math::Transform::createLinearTransform(m_dx);
	m_voxel_corner_transform->postTranslate(m_voxel_corner_origin);

	//voxel center is aligned with the world origin.
	m_voxel_center_transform = m_voxel_corner_transform->copy();
	m_voxel_center_transform->postTranslate(openvdb::Vec3f(0.5f * m_dx));

	for (int i = 0; i < 3; i++) {
		m_face_center_transform[i] = m_voxel_center_transform->copy();
		openvdb::Vec3d face_post_translation{ 0,0,0 };
		face_post_translation[i] -= 0.5 * m_dx;
		m_face_center_transform[i]->postTranslate(face_post_translation);

		m_edge_center_transform[i] = m_voxel_corner_transform->copy();
		openvdb::Vec3d edge_post_translation{ 0,0,0 };
		edge_post_translation[i] += 0.5 * m_dx;
		m_edge_center_transform[i]->postTranslate(edge_post_translation);

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
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				m_crosssection_face_transform[i][j][k] = m_voxel_center_transform->copy();
				switch (i) {
				case 0:
					m_crosssection_face_transform[i][j][k]->postTranslate(
						openvdb::Vec3d(0, -0.25 + j * 0.5, -0.25 + k * 0.5) * m_dx);
					break;
				case 1:
					m_crosssection_face_transform[i][j][k]->postTranslate(
						openvdb::Vec3d(-0.25 + k * 0.5, 0, -0.25 + j * 0.5) * m_dx);
					break;
				case 2:
					m_crosssection_face_transform[i][j][k]->postTranslate(
						openvdb::Vec3d(-0.25 + j * 0.5, -0.25 + k * 0.5, 0) * m_dx);
				}
			}
		}
	}

	m_subcell_corner_transform = openvdb::math::Transform::createLinearTransform(0.5 * m_dx);
	m_subcell_corner_transform->postTranslate(m_voxel_corner_origin);

	m_subcell_center_transform = m_subcell_corner_transform->copy();
	m_subcell_center_transform->postTranslate(openvdb::Vec3d(m_dx * 0.25f));
}

void L_with_level::initialize_grids()
{
	m_voxel_vol = openvdb::FloatGrid::create();
	m_voxel_vol->setTransform(m_voxel_center_transform);
	m_voxel_vol->setName("m_voxel_vol");

	m_subcell_sdf = openvdb::FloatGrid::create(1.5f * m_dx);
	m_subcell_sdf->setTransform(m_subcell_corner_transform);
	m_subcell_sdf->setName("m_subcell_sdf");
	m_subcell_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);

	m_subcell_vol = openvdb::FloatGrid::create();
	m_subcell_vol->setTransform(m_subcell_center_transform);
	m_subcell_vol->setName("m_subcell_vol");

	for (int i = 0; i < 3; i++) {
		m_face_center_vol.v[i] = openvdb::FloatGrid::create();
		m_face_center_vol.v[i]->setTransform(m_face_center_transform[i]);
		m_face_center_vol.v[i]->setName("m_face_center_volc" + std::to_string(i));

		m_edge_center_vol[i] = openvdb::FloatGrid::create();
		m_edge_center_vol[i]->setTransform(m_edge_center_transform[i]);
		m_edge_center_vol[i]->setName("m_edge_center_volc" + std::to_string(i));

		m_is_solid[i] = openvdb::BoolGrid::create();
		m_is_solid[i]->setTransform(m_face_center_transform[i]);
		m_is_solid[i]->setName("m_is_solidc" + std::to_string(i));

		m_velocity_DOF.idx[i] = openvdb::Int32Grid::create(-1);
		m_velocity_DOF.idx[i]->setTransform(m_face_center_transform[i]);
		m_velocity_DOF.idx[i]->setName("m_velocity_DOFc" + std::to_string(i));

		//matrix coefficients
		//diagonal term for each velocity channel
		m_diagonal.v[i] = openvdb::FloatGrid::create();
		m_diagonal.v[i]->setTransform(m_face_center_transform[i]);
		m_diagonal.v[i]->setName("m_diagonalc" + std::to_string(i));

		m_invdiag.v[i] = openvdb::FloatGrid::create();
		m_invdiag.v[i]->setTransform(m_face_center_transform[i]);
		m_invdiag.v[i]->setName("m_invdiagc" + std::to_string(i));

		m_SPAI0.v[i] = openvdb::FloatGrid::create();
		m_SPAI0.v[i]->setTransform(m_face_center_transform[i]);
		m_SPAI0.v[i]->setName("m_SPAI0c" + std::to_string(i));

		//coefficients of each velocity channel and its negative axis neighbor 
		//of the same channel
		m_neg_x.v[i] = openvdb::FloatGrid::create();
		m_neg_x.v[i]->setTransform(m_face_center_transform[i]->copy());
		openvdb::Vec3d neg_x_shift{ -0.5 * m_dx,0 , 0 };
		m_neg_x.v[i]->transform().postTranslate(neg_x_shift);
		m_neg_x.v[i]->setName("m_neg_xc" + std::to_string(i));

		m_neg_y.v[i] = openvdb::FloatGrid::create();
		m_neg_y.v[i]->setTransform(m_face_center_transform[i]->copy());
		openvdb::Vec3d neg_y_shift{ 0, -0.5 * m_dx,0 };
		m_neg_y.v[i]->transform().postTranslate(neg_y_shift);
		m_neg_y.v[i]->setName("m_neg_yc" + std::to_string(i));

		m_neg_z.v[i] = openvdb::FloatGrid::create();
		m_neg_z.v[i]->setTransform(m_face_center_transform[i]->copy());
		openvdb::Vec3d neg_z_shift{ 0, 0, -0.5 * m_dx };
		m_neg_z.v[i]->transform().postTranslate(neg_z_shift);
		m_neg_z.v[i]->setName("m_neg_zc" + std::to_string(i));

		//cross channel terms, looking from positive x direction
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
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				m_cross_channel[i][j][k] = openvdb::FloatGrid::create();
				m_cross_channel[i][j][k]->setTransform(m_crosssection_face_transform[i][j][k]);
				switch (i) {
				case 0:
					m_cross_channel[i][j][k]->setName("Tyz" + std::to_string(j) + std::to_string(k));
					break;
				case 1:
					m_cross_channel[i][j][k]->setName("Tzx" + std::to_string(j) + std::to_string(k));
					break;
				case 2:
					m_cross_channel[i][j][k]->setName("Txy" + std::to_string(j) + std::to_string(k));
					break;
				}

			}
		}
	}
}

void L_with_level::calculate_subcell_sdf_volume()
{
	auto dilated_liquid_sdf = m_liquid_sdf->deepCopy();
	openvdb::tools::dilateActiveValues(dilated_liquid_sdf->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
	for (auto iter = dilated_liquid_sdf->tree().beginLeaf(); iter; ++iter) {
		const auto base_coord = iter->origin() + iter->origin();
		for (int i = 0; i <= 8; i += 8) {
			for (int j = 0; j <= 8; j += 8) {
				for (int k = 0; k <= 8; k += 8) {
					m_subcell_sdf->tree().touchLeaf(base_coord.offsetBy(i, j, k));
					m_subcell_vol->tree().touchLeaf(base_coord.offsetBy(i, j, k));
				}
			}
		}
	}

	//sample the corner subcell liquid sdf
	auto subcell_sdf_setter = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index) {
		auto liquid_sdf_axr = m_liquid_sdf->getConstUnsafeAccessor();
		for (auto iter = leaf.beginValueAll(); iter; ++iter) {
			auto wpos = m_subcell_sdf->indexToWorld(iter.getCoord());
			auto ipos = m_liquid_sdf->worldToIndex(wpos);

			float sampled_sdf = openvdb::tools::BoxSampler::sample(liquid_sdf_axr, ipos);

			if (sampled_sdf < m_subcell_sdf->background()) {
				iter.setValue(sampled_sdf);
				iter.setValueOn();
			}
			else {
				iter.setValueOff();
			}
		}
	};//end subcell_sdf_setter
	openvdb::tree::LeafManager<openvdb::FloatTree> subcell_sdf_leafman(m_subcell_sdf->tree());
	subcell_sdf_leafman.foreach(subcell_sdf_setter);

	//set the volume of each subcell based on the liquid sdf
	auto subcell_vol_setter = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index ) {
		auto subcell_liquid_sdf_axr = m_subcell_sdf->getConstUnsafeAccessor();

		for (auto iter = leaf.beginValueAll(); iter; ++iter) {
			float phi[2][2][2];
			auto base_coord = iter.getCoord();
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					for (int k = 0; k < 2; k++) {
						phi[i][j][k] = subcell_liquid_sdf_axr.getValue(base_coord.offsetBy(i, j, k));
					}
				}
			}
			float liquid_fraction = volume_fraction(
				phi[0][0][0], phi[1][0][0],
				phi[0][1][0], phi[1][1][0],
				phi[0][0][1], phi[1][0][1],
				phi[0][1][1], phi[1][1][1]);

			if (liquid_fraction < 1e-1f) {
				//liquid_fraction = 0;
			}
			if (liquid_fraction > 0) {
				iter.setValue(liquid_fraction);
				iter.setValueOn();
			}
			else {
				iter.setValueOff();
			}

		}
	};//end subcell_vol_setter
	openvdb::tree::LeafManager<openvdb::FloatTree> subcell_vol_leafman(m_subcell_vol->tree());
	subcell_vol_leafman.foreach(subcell_vol_setter);
}

void L_with_level::calculate_matrix_coefficients_volume()
{
	openvdb::BoolGrid::Ptr copied_liquid_mask = openvdb::BoolGrid::create();
	copied_liquid_mask->setTree(std::make_shared<openvdb::BoolTree>(m_liquid_sdf->tree(), false, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(copied_liquid_mask->tree(), 2, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	m_voxel_vol->setTree(std::make_shared<openvdb::FloatTree>(copied_liquid_mask->tree(), 0.f, openvdb::TopologyCopy()));
	for (int i = 0; i < 3; i++) {
		m_face_center_vol.v[i]->setTree(std::make_shared<openvdb::FloatTree>(copied_liquid_mask->tree(), 0.f, openvdb::TopologyCopy()));
		m_edge_center_vol[i]->setTree(std::make_shared<openvdb::FloatTree>(copied_liquid_mask->tree(), 0.f, openvdb::TopologyCopy()));
	}

	auto term_volume_setter = [&](openvdb::BoolTree::LeafNodeType& leaf, openvdb::Index ) {
		openvdb::FloatTree::LeafNodeType* face_center_leaf[3], * edge_center_leaf[3], * voxel_center_leaf;
		voxel_center_leaf = m_voxel_vol->tree().probeLeaf(leaf.origin());
		for (int i = 0; i < 3; i++) {
			face_center_leaf[i] = m_face_center_vol.v[i]->tree().probeLeaf(leaf.origin());
			edge_center_leaf[i] = m_edge_center_vol[i]->tree().probeLeaf(leaf.origin());
		}
		auto subcell_vol_axr = m_subcell_vol->getConstUnsafeAccessor();
		for (auto iter = leaf.beginValueAll(); iter; ++iter) {
			float vol_c = 0;
			float vol_f[3] = { 0,0,0 };
			float vol_e[3] = { 0,0,0 };
			//for eight subcell volume cells
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					for (int k = 0; k < 2; k++) {
						//at the subcell level base coord
						auto at_volume_coord = iter.getCoord() + iter.getCoord();

						//voxel center volume
						at_volume_coord.offset(i, j, k);
						vol_c += subcell_vol_axr.getValue(at_volume_coord);
						//three channel or three axis
						for (int c = 0; c < 3; c++) {
							//voxel face volume
							auto at_face_coord = at_volume_coord;
							at_face_coord[c]--;
							vol_f[c] += subcell_vol_axr.getValue(at_face_coord);
							//voxel edge volume
							auto at_edge_coord = at_volume_coord.offsetBy(-1, -1, -1);
							at_edge_coord[c]++;
							vol_e[c] += subcell_vol_axr.getValue(at_edge_coord);
						}//channel c
					}//k
				}//j
			}//i
			if (vol_c > 0) {
				voxel_center_leaf->setValueOn(iter.offset(), vol_c * 0.125f);
			}
			else {
				voxel_center_leaf->setValueOff(iter.offset());
			}

			for (int c = 0; c < 3; c++) {
				if (vol_f[c] > 0) {
					face_center_leaf[c]->setValueOn(iter.offset(), vol_f[c] * 0.125f);
				}
				else {
					face_center_leaf[c]->setValueOff(iter.offset());
				}
				if (vol_e[c] > 0) {
					edge_center_leaf[c]->setValueOn(iter.offset(), vol_e[c] * 0.125f);
				}
				else {
					edge_center_leaf[c]->setValueOff(iter.offset());
				}
			}//end three channels
		}//end for all voxels
	};//end term_volume_setter

	openvdb::tree::LeafManager<openvdb::BoolTree> booltreeman(copied_liquid_mask->tree());
	booltreeman.foreach(term_volume_setter);
}

void L_with_level::mark_dof()
{
	//for all liquid cells, decide if a velocity face is inside the solid or not
	auto dilated_liquid = openvdb::BoolGrid::create();
	dilated_liquid->setTree(
		std::make_shared<openvdb::BoolTree>(
			m_liquid_sdf->tree(), false, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(dilated_liquid->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE, openvdb::tools::TilePolicy::EXPAND_TILES);

	for (int i = 0; i < 3; i++) {
		m_is_solid[i]->setTree(std::make_shared<openvdb::BoolTree>(dilated_liquid->tree(), false, openvdb::TopologyCopy()));

		auto solid_marker = [&](openvdb::BoolTree::LeafNodeType& leaf, openvdb::Index ) {
			auto solid_axr = m_solid_sdf->getConstUnsafeAccessor();

			for (auto iter = leaf.beginValueAll(); iter; ++iter) {
				auto face_wpos = m_is_solid[i]->indexToWorld(iter.getCoord());
				auto face_ipos = m_solid_sdf->worldToIndex(face_wpos);
				auto sampled_solid_sdf = openvdb::tools::BoxSampler::sample(solid_axr, face_ipos);
				if (sampled_solid_sdf < 0) {
					iter.setValue(true);
					iter.setValueOn();
				}
				else {
					iter.setValue(false);
					iter.setValueOff();
				}
			}
		};//end solid_marker
		openvdb::tree::LeafManager<openvdb::BoolTree> treeman(m_is_solid[i]->tree());
		treeman.foreach(solid_marker);

		//mark if a velocity sample that is not solid should be included in the equation system
		m_velocity_DOF.idx[i]->setTree(std::make_shared<openvdb::Int32Tree>(dilated_liquid->tree(), -1, openvdb::TopologyCopy()));

		auto dof_marker = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index ) {
			openvdb::FloatGrid::ConstUnsafeAccessor edge_vol_axr[3] = {
				m_edge_center_vol[0]->tree(), m_edge_center_vol[1]->tree(),m_edge_center_vol[2]->tree() };
			auto face_vol_axr = m_face_center_vol.v[i]->getConstUnsafeAccessor();
			auto voxel_vol_axr = m_voxel_vol->getConstUnsafeAccessor();
			auto is_solid_leaf = m_is_solid[i]->tree().probeConstLeaf(leaf.origin());

			//depending on the face direction
			int first_dir = (i + 1) % 3;
			int second_dir = (i + 2) % 3;
			openvdb::Coord dir0_vec{ 0,0,0 }; dir0_vec[i]++;
			openvdb::Coord dir1_vec{ 0,0,0 }; dir1_vec[first_dir]++;
			openvdb::Coord dir2_vec{ 0,0,0 }; dir2_vec[second_dir]++;

			for (auto iter = leaf.beginValueAll(); iter; ++iter) {
				if ((is_solid_leaf) && is_solid_leaf->isValueOn(iter.offset())) {
					iter.setValueOff();
					continue;
				}
				if (face_vol_axr.getValue(iter.getCoord()) > 0) {
					//velocity volume nonzero, make it a dof
					iter.setValueOn();
				}
				else {
					//velocity volume is zero, but its neighbor edge volume may be non-zero

					//diagonal stress term
					if (voxel_vol_axr.getValue(iter.getCoord()) > 0 || voxel_vol_axr.getValue(iter.getCoord() - dir0_vec) > 0) {
						iter.setValueOn();
					}
					else {
						//cross term
						//note the direction of the edge is orthogonal to the incremental positive direction.
						float vol_edge_dir1_neg = edge_vol_axr[first_dir].getValue(iter.getCoord());
						float vol_edge_dir1_pos = edge_vol_axr[first_dir].getValue(iter.getCoord() + dir2_vec);
						float vol_edge_dir2_neg = edge_vol_axr[second_dir].getValue(iter.getCoord());
						float vol_edge_dir2_pos = edge_vol_axr[second_dir].getValue(iter.getCoord() + dir1_vec);
						if (vol_edge_dir1_neg > 0 || vol_edge_dir1_pos > 0 || vol_edge_dir2_neg > 0 || vol_edge_dir2_pos > 0) {
							iter.setValueOn();
						}
						else {
							iter.setValueOff();
						}
					}
				}//end else face volume is zero
			}//end for all voxels
		};//end dof_marker
		//openvdb::tree::LeafManager<openvdb::Int32Tree> dof_leafman(m_velocity_DOF.idx[i]->tree());
		m_dof_manager[i] = std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(m_velocity_DOF.idx[i]->tree());
		m_dof_manager[i]->foreach(dof_marker);
		set_dof(m_velocity_DOF.idx[i], m_ndof);
		m_velocity_DOF.idx[i]->pruneGrid();
		m_dof_manager[i] = std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(m_velocity_DOF.idx[i]->tree());

	}//end for three channels
}

void L_with_level::set_dof(openvdb::Int32Grid::Ptr in_out_DOF, int& in_out_dof)
{
	openvdb::tree::LeafManager<openvdb::Int32Tree> leafman(in_out_DOF->tree());

	//first count how many dofs in each leaf
	std::vector<int> end_dof_at_this_leaf;
	if (leafman.leafCount() == 0) {
		return;
	}

	end_dof_at_this_leaf.resize(leafman.leafCount(), 0);

	auto per_leaf_dof_counter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
			end_dof_at_this_leaf[leafpos]++;
		}
	};
	leafman.foreach(per_leaf_dof_counter);

	//actual starting dof for each leaf.
	end_dof_at_this_leaf[0] += in_out_dof;
	for (int i = 1; i < end_dof_at_this_leaf.size(); i++) {
		end_dof_at_this_leaf[i] += end_dof_at_this_leaf[i - 1];
	}

	auto per_leaf_dof_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
		leaf.fill(-1);
		int starting_idx = in_out_dof;
		if (leafpos != 0) {
			starting_idx = end_dof_at_this_leaf[leafpos - 1];
		}
		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
			iter.setValue(starting_idx);
			starting_idx++;
		}
	};
	leafman.foreach(per_leaf_dof_setter);
	in_out_dof = end_dof_at_this_leaf.back();
}

namespace {
struct finest_level_matrix_setter {
	using leaf_type = openvdb::FloatTree::LeafNodeType;
	using leaves_vec_type = std::vector<leaf_type*>;
	using leaves_vec_ptr_t = leaves_vec_type*;

	finest_level_matrix_setter(
		leaves_vec_ptr_t in_diag_leaves, leaves_vec_ptr_t in_invdiag_leaves,
		leaves_vec_ptr_t in_neg_x_leaves, leaves_vec_ptr_t in_neg_y_leaves, leaves_vec_ptr_t in_neg_z_leaves,
		leaves_vec_ptr_t in_cross_term_leaves, L_with_level::DOF_tuple_t in_DOF, openvdb::FloatGrid::Ptr in_viscosity,
		packed_FloatGrid3 in_face_volume, openvdb::FloatGrid::Ptr in_edge_volume[3], openvdb::FloatGrid::Ptr in_voxel_volume, float in_dt, float in_rho, float in_diag_frac_epsl) :
		diag_leaves{ in_diag_leaves }, invdiag_leaves{ in_invdiag_leaves },
		neg_x_leaves{ in_neg_x_leaves },
		neg_y_leaves{ in_neg_y_leaves },
		neg_z_leaves{ in_neg_z_leaves },
		cross_term_leaves{ in_cross_term_leaves },
		m_dof_axr{ in_DOF.idx[0]->tree(), in_DOF.idx[1]->tree(), in_DOF.idx[2]->tree() },
		m_viscosity_axr(in_viscosity->tree()),
		m_face_volume_axr{ in_face_volume.v[0]->tree(),in_face_volume.v[1]->tree() ,in_face_volume.v[2]->tree() },
		m_edge_volume_axr{ in_edge_volume[0]->tree(),in_edge_volume[1]->tree() ,in_edge_volume[2]->tree() },
		m_volume_axr{ in_voxel_volume->tree() }, dt(in_dt), rho(in_rho), m_diag_frac_epsl(in_diag_frac_epsl) {
		dx = (float)in_voxel_volume->voxelSize()[0];
	}

	finest_level_matrix_setter(const finest_level_matrix_setter& other, tbb::split) :
		diag_leaves{ other.diag_leaves }, invdiag_leaves{ other.invdiag_leaves },
		neg_x_leaves{ other.neg_x_leaves },
		neg_y_leaves{ other.neg_y_leaves },
		neg_z_leaves{ other.neg_z_leaves },
		cross_term_leaves{ other.cross_term_leaves },
		m_dof_axr{ other.m_dof_axr[0], other.m_dof_axr[1], other.m_dof_axr[2] },
		m_viscosity_axr(other.m_viscosity_axr),
		m_face_volume_axr{ other.m_face_volume_axr[0], other.m_face_volume_axr[1], other.m_face_volume_axr[2] },
		m_edge_volume_axr{ other.m_edge_volume_axr[0],other.m_edge_volume_axr[1],other.m_edge_volume_axr[2] },
		m_volume_axr{ other.m_volume_axr }, dt(other.dt), rho(other.rho), dx(other.dx), m_diag_frac_epsl(other.m_diag_frac_epsl) {
		//clear accessor cache
		for (int i = 0; i < 3; i++) {
			m_dof_axr[i].clear();
			m_face_volume_axr[i].clear();
			m_edge_volume_axr[i].clear();
		}
		m_viscosity_axr.clear();
		m_volume_axr.clear();
	}

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
	//the actual operator for each channel
	template <int c>
	void channel_op(size_t leafpos) const {
		const int d0 = c;
		const int d1 = (c + 1) % 3;
		const int d2 = (c + 2) % 3;
		openvdb::Coord dir0{ 0 }, dir1{ 0 }, dir2{ 0 };
		dir0[d0]++; dir1[d1]++; dir2[d2]++;

		auto negxleaf = neg_x_leaves[d0][leafpos];
		auto negyleaf = neg_y_leaves[d0][leafpos];
		auto negzleaf = neg_z_leaves[d0][leafpos];

		//component is c
		//but the negative direction is d0, d1, d2, not necessarily x,y,z
		leaf_type* negdirleaf[3] = { nullptr,nullptr,nullptr };
		switch (c) {
		case 0:
			negdirleaf[0] = negxleaf;
			negdirleaf[1] = negyleaf;
			negdirleaf[2] = negzleaf;
			break;
		case 1:
			negdirleaf[0] = negyleaf;
			negdirleaf[1] = negzleaf;
			negdirleaf[2] = negxleaf;
			break;
		case 2:
			negdirleaf[0] = negzleaf;
			negdirleaf[1] = negxleaf;
			negdirleaf[2] = negyleaf;
		}

		float factor = dt / (rho * dx * dx);

		//the face
		for (auto iter = diag_leaves[d0][leafpos]->beginValueOn(); iter; ++iter) {
			const int this_dof = m_dof_axr[d0].getValue(iter.getCoord());
			if (this_dof != -1) {
				float temp_diag = 0;

				float face_vol = m_face_volume_axr[d0].getValue(iter.getCoord());

				if (face_vol < m_diag_frac_epsl) {
					face_vol = m_diag_frac_epsl;
				}
				//original velocity diagonal contribution
				temp_diag += face_vol;

				//+ d0 direction
				{
					float term = 2 * factor * m_volume_axr.getValue(iter.getCoord()) * m_viscosity_axr.getValue(iter.getCoord());
					temp_diag += term;
				}

				//- d0 direction
				{
					const int nd0dof = m_dof_axr[d0].getValue(iter.getCoord() - dir0);
					float term = 2 * factor * m_volume_axr.getValue(iter.getCoord() - dir0) * m_viscosity_axr.getValue(iter.getCoord() - dir0);
					temp_diag += term;
					if (nd0dof != -1) {
						negdirleaf[0]->setValueOn(iter.offset(), -term);
					}
					else {
						negdirleaf[0]->setValueOff(iter.offset(), 0.f);
					}
				}


				//index space face center for the viscosity sampling
				openvdb::Vec3f fc_ipos = iter.getCoord().asVec3s();
				fc_ipos[d0] -= 0.5f;

				//+ d1 direction
				{
					float edgevoldir2posp1 = m_edge_volume_axr[d2].getValue(iter.getCoord() + dir1);
					openvdb::Vec3f edgedir2posp1_ipos = fc_ipos;
					//move to positive d1 direction
					edgedir2posp1_ipos[d1] += 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir2posp1_ipos);
					float term = factor * edgevoldir2posp1 * sampled_viscosity;
					temp_diag += term;
					const int pd1dof = m_dof_axr[d0].getValue(iter.getCoord() + dir1);
				}

				//- d1 direction
				{
					float edgevoldir2posn1 = m_edge_volume_axr[d2].getValue(iter.getCoord());
					openvdb::Vec3f  edgedir2posn1_ipos = fc_ipos;
					//move to the negative d1 direction
					edgedir2posn1_ipos[d1] -= 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir2posn1_ipos);
					float term = factor * edgevoldir2posn1 * sampled_viscosity;
					temp_diag += term;
					const int nd1dof = m_dof_axr[d0].getValue(iter.getCoord() - dir1);
					if (nd1dof != -1) {
						negdirleaf[1]->setValueOn(iter.offset(), -term);
					}
					else {
						negdirleaf[1]->setValueOff(iter.offset(), 0.f);
					}
				}

				//+ d2 direction
				{
					float edgevold1posp2 = m_edge_volume_axr[d1].getValue(iter.getCoord() + dir2);
					openvdb::Vec3f edgedir1posp2_ipos = fc_ipos;
					//move to positive d2 direction
					edgedir1posp2_ipos[d2] += 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir1posp2_ipos);
					float term = factor * edgevold1posp2 * sampled_viscosity;
					temp_diag += term;
					const int pd2dof = m_dof_axr[d0].getValue(iter.getCoord() + dir2);
				}

				//- d2 direction
				{
					float edgevold1posn2 = m_edge_volume_axr[d1].getValue(iter.getCoord());
					openvdb::Vec3f edgedir1posn2_ipos = fc_ipos;
					//move to the negative d2 direction
					edgedir1posn2_ipos[d2] -= 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir1posn2_ipos);
					float term = factor * edgevold1posn2 * sampled_viscosity;
					temp_diag += term;
					const int nd2dof = m_dof_axr[d0].getValue(iter.getCoord() - dir2);
					if (nd2dof != -1) {
						negdirleaf[2]->setValueOn(iter.offset(), -term);
					}
					else {
						negdirleaf[2]->setValueOff(iter.offset(), 0.f);
					}
				}


				//write the diagonal term, as well as the inverse diagonal term
				//when viscosity=0, the diagonal can be zero
				/*if (temp_diag == 0) {
					temp_diag == m_diag_frac_epsl;
				}*/
				iter.setValue(temp_diag);
				iter.setValueOn();
				if (temp_diag != 0) {
					invdiag_leaves[c][leafpos]->setValueOn(iter.offset(), 1.0f / temp_diag);
				}
				else {
					invdiag_leaves[c][leafpos]->setValueOn(iter.offset(), 0.f);
				}

			}//end if this face voxel is a dof
			else {
				iter.setValueOff();
				invdiag_leaves[c][leafpos]->setValueOff(iter.offset(), 0.f);
				negdirleaf[0]->setValueOff(iter.offset(), 0.f);
				negdirleaf[1]->setValueOff(iter.offset(), 0.f);
				negdirleaf[2]->setValueOff(iter.offset(), 0.f);
			}//end else this face voxel is a dof

			//cross terms are between the other faces
			const int dof_face_dir1neg = m_dof_axr[d1].getValue(iter.getCoord());
			const int dof_face_dir1pos = m_dof_axr[d1].getValue(iter.getCoord() + dir1);
			const int dof_face_dir2neg = m_dof_axr[d2].getValue(iter.getCoord());
			const int dof_face_dir2pos = m_dof_axr[d2].getValue(iter.getCoord() + dir2);

			//00
			if (dof_face_dir1neg != -1 && dof_face_dir2neg != -1) {
				float edgevoldir0p00 = m_edge_volume_axr[d0].getValue(iter.getCoord());
				openvdb::Vec3f edgedir0p00_ipos = iter.getCoord().asVec3s();
				edgedir0p00_ipos[d1] -= 0.5f;
				edgedir0p00_ipos[d2] -= 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p00_ipos);
				float term = factor * edgevoldir0p00 * sampled_viscosity;
				cross_term_leaves[4 * d0][leafpos]->setValueOn(iter.offset(), term);
			}
			else {
				cross_term_leaves[4 * d0][leafpos]->setValueOff(iter.offset(), 0.f);
			}

			//01
			if (dof_face_dir1neg != -1 && dof_face_dir2pos != -1) {
				float edgevoldir0p01 = m_edge_volume_axr[d0].getValue(iter.getCoord() + dir2);
				openvdb::Vec3f edgedir0p01_ipos = iter.getCoord().asVec3s();
				edgedir0p01_ipos[d1] -= 0.5f;
				edgedir0p01_ipos[d2] += 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p01_ipos);
				float term = -factor * edgevoldir0p01 * sampled_viscosity;
				cross_term_leaves[4 * d0 + 1][leafpos]->setValueOn(iter.offset(), term);
			}
			else {
				cross_term_leaves[4 * d0 + 1][leafpos]->setValueOff(iter.offset(), 0.f);
			}

			//10
			if (dof_face_dir1pos != -1 && dof_face_dir2neg != -1) {
				float edgevoldir0p10 = m_edge_volume_axr[d0].getValue(iter.getCoord() + dir1);
				openvdb::Vec3f edgedir0p10_ipos = iter.getCoord().asVec3s();
				edgedir0p10_ipos[d1] += 0.5f;
				edgedir0p10_ipos[d2] -= 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p10_ipos);
				float term = -factor * edgevoldir0p10 * sampled_viscosity;
				cross_term_leaves[4 * d0 + 2][leafpos]->setValueOn(iter.offset(), term);
			}
			else {
				cross_term_leaves[4 * d0 + 2][leafpos]->setValueOff(iter.offset(), 0.f);
			}
			//11
			if (dof_face_dir1pos != -1 && dof_face_dir2pos != -1) {
				float edgevoldir0p11 = m_edge_volume_axr[d0].getValue(iter.getCoord() + dir1 + dir2);
				openvdb::Vec3f edgedir0p11_ipos = iter.getCoord().asVec3s();
				edgedir0p11_ipos[d1] += 0.5f;
				edgedir0p11_ipos[d2] += 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p11_ipos);
				float term = factor * edgevoldir0p11 * sampled_viscosity;
				cross_term_leaves[4 * d0 + 3][leafpos]->setValueOn(iter.offset(), term);
			}
			else {
				cross_term_leaves[4 * d0 + 3][leafpos]->setValueOff(iter.offset(), 0.f);
			}
		}//end for all active diagonal voxels
	}

	void operator()(const tbb::blocked_range<size_t>& r) const {
		for (size_t i = r.begin(); i != r.end(); ++i) {
			channel_op<0>(i);
		}
		for (size_t i = r.begin(); i != r.end(); ++i) {
			channel_op<1>(i);
		}
		for (size_t i = r.begin(); i != r.end(); ++i) {
			channel_op<2>(i);
		}
	}


	//matrix term leaves
	leaves_vec_type* diag_leaves, * invdiag_leaves,
		* neg_x_leaves, * neg_y_leaves, * neg_z_leaves;
	leaves_vec_type* cross_term_leaves;

	//index of degree of freedom for three channels
	mutable openvdb::Int32Grid::ConstUnsafeAccessor m_dof_axr[3];
	//viscosity is assumed to be voxel center variable.
	mutable openvdb::FloatGrid::ConstUnsafeAccessor m_viscosity_axr, m_face_volume_axr[3], m_edge_volume_axr[3], m_volume_axr;

	float dt;
	float rho;
	float dx;
	float m_diag_frac_epsl;
};
}//end namespace

void L_with_level::calculate_matrix_coefficients()
{
	//given the active state of the DOF
	//construct the matrix coefficients both explicly and the matrix free version
	openvdb::BoolGrid::Ptr dilated_DOF_pattern = openvdb::BoolGrid::create();
	for (int i = 0; i < 3; i++) {
		dilated_DOF_pattern->topologyUnion(*m_velocity_DOF.idx[i]);
	}
	openvdb::tools::dilateActiveValues(dilated_DOF_pattern->tree(), 2, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	using leaf_container_type = finest_level_matrix_setter::leaves_vec_type;
	std::unique_ptr<leaf_container_type[]> diag_leaves, invdiag_leaves,
		neg_x_leaves, neg_y_leaves, neg_z_leaves;
	diag_leaves = std::make_unique<leaf_container_type[]>(3);
	invdiag_leaves = std::make_unique<leaf_container_type[]>(3);
	neg_x_leaves = std::make_unique<leaf_container_type[]>(3);
	neg_y_leaves = std::make_unique<leaf_container_type[]>(3);
	neg_z_leaves = std::make_unique<leaf_container_type[]>(3);

	std::unique_ptr<leaf_container_type[]> cross_term_leaves;
	cross_term_leaves = std::make_unique<leaf_container_type[]>(3 * 2 * 2);
	int nleaf = dilated_DOF_pattern->tree().leafCount();


	//default values
	float factor = m_dt / (m_rho * m_dx * m_dx);
	m_default_off_diag = factor * m_viscosity->background();
	m_default_diag = 1.0f + 8.0f * m_default_off_diag;


	//set all coefficients matrix to the dilated pattern
	for (int i = 0; i < 3; i++) {
		m_diagonal.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), m_default_diag, openvdb::TopologyCopy()));
		diag_leaves[i].reserve(nleaf); m_diagonal.v[i]->tree().getNodes(diag_leaves[i]);

		m_invdiag.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
		invdiag_leaves[i].reserve(nleaf); m_invdiag.v[i]->tree().getNodes(invdiag_leaves[i]);

		if (i == 0) {
			m_neg_x.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), -2.0f * m_default_off_diag, openvdb::TopologyCopy()));
		}
		else {
			m_neg_x.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), -m_default_off_diag, openvdb::TopologyCopy()));
		}
		neg_x_leaves[i].reserve(nleaf); m_neg_x.v[i]->tree().getNodes(neg_x_leaves[i]);

		if (i == 1) {
			m_neg_y.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), -2.0f * m_default_off_diag, openvdb::TopologyCopy()));
		}
		else {
			m_neg_y.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), -m_default_off_diag, openvdb::TopologyCopy()));
		}
		neg_y_leaves[i].reserve(nleaf); m_neg_y.v[i]->tree().getNodes(neg_y_leaves[i]);

		if (i == 2) {
			m_neg_z.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), -2.0f * m_default_off_diag, openvdb::TopologyCopy()));
		}
		else {
			m_neg_z.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), -m_default_off_diag, openvdb::TopologyCopy()));
		}
		neg_z_leaves[i].reserve(nleaf); m_neg_z.v[i]->tree().getNodes(neg_z_leaves[i]);

		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				float default_cross_term = m_default_off_diag;
				if ((j + k) % 2 != 0) {
					default_cross_term = -m_default_off_diag;
				}
				m_cross_channel[i][j][k]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), default_cross_term, openvdb::TopologyCopy()));
				cross_term_leaves[4 * i + 2 * j + k].reserve(nleaf); m_cross_channel[i][j][k]->tree().getNodes(cross_term_leaves[4 * i + 2 * j + k]);
			}
		}
	}

	//generate the matrix coefficients
	finest_level_matrix_setter Op{
		diag_leaves.get(), invdiag_leaves.get(),
		neg_x_leaves.get(), neg_y_leaves.get(), neg_z_leaves.get(),
		cross_term_leaves.get(), m_velocity_DOF, m_viscosity,
		m_face_center_vol, m_edge_center_vol, m_voxel_vol, m_dt, m_rho, m_diag_frac_epsl };

	tbb::parallel_for(tbb::blocked_range<size_t>(0, nleaf), Op);
}

void L_with_level::initialize_transforms_from_child(const L_with_level& child)
{
	openvdb::Vec3f child_world_corner = child.m_voxel_corner_transform->indexToWorld(openvdb::Vec3f(0, 0, 0));
	openvdb::Vec3f naive_world_corner{ -m_dx * 0.5f };
	openvdb::Vec3f corner_shift = child_world_corner - naive_world_corner;

	//voxel center is aligned with the world origin.
	m_voxel_center_transform = openvdb::math::Transform::createLinearTransform(m_dx);
	m_voxel_center_transform->postTranslate(corner_shift);
	m_voxel_corner_transform = m_voxel_center_transform->copy();
	m_voxel_corner_transform->postTranslate(openvdb::Vec3d(-m_dx) * 0.5);

	for (int i = 0; i < 3; i++) {
		m_face_center_transform[i] = m_voxel_center_transform->copy();
		openvdb::Vec3d face_post_translation{ 0,0,0 };
		face_post_translation[i] -= 0.5 * m_dx;
		m_face_center_transform[i]->postTranslate(face_post_translation);

		m_edge_center_transform[i] = m_voxel_corner_transform->copy();
		openvdb::Vec3d edge_post_translation{ 0,0,0 };
		edge_post_translation[i] += 0.5 * m_dx;
		m_edge_center_transform[i]->postTranslate(edge_post_translation);

		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				m_crosssection_face_transform[i][j][k] = m_voxel_center_transform->copy();
				switch (i) {
				case 0:
					m_crosssection_face_transform[i][j][k]->postTranslate(
						openvdb::Vec3d(0, -0.25 + j * 0.5, -0.25 + k * 0.5) * m_dx);
					break;
				case 1:
					m_crosssection_face_transform[i][j][k]->postTranslate(
						openvdb::Vec3d(-0.25 + k * 0.5, 0, -0.25 + j * 0.5) * m_dx);
					break;
				case 2:
					m_crosssection_face_transform[i][j][k]->postTranslate(
						openvdb::Vec3d(-0.25 + j * 0.5, -0.25 + k * 0.5, 0) * m_dx);
				}
			}
		}
	}

	m_subcell_corner_transform = openvdb::math::Transform::createLinearTransform(0.5 * m_dx);
	m_subcell_corner_transform->postTranslate(openvdb::Vec3d(-m_dx) * 0.5);
	m_subcell_corner_transform->postTranslate(corner_shift);

	m_subcell_center_transform = openvdb::math::Transform::createLinearTransform(0.5 * m_dx);
	m_subcell_center_transform->postTranslate(openvdb::Vec3d(-m_dx) * 0.25);
	m_subcell_center_transform->postTranslate(corner_shift);
}

void L_with_level::mark_dof_from_child(const L_with_level& child)
{
	for (int i = 0; i < 3; i++) {
		//reduction touch leaves
		std::vector<openvdb::Int32Tree::LeafNodeType*> child_leaves;
		auto child_leafcount = child.m_velocity_DOF.idx[i]->tree().leafCount();
		child_leaves.reserve(child_leafcount);
		child.m_velocity_DOF.idx[i]->tree().getNodes(child_leaves);

		simd_uaamg::TouchCoarseLeafReducer coarse_reducer{ child_leaves };
		tbb::parallel_reduce(tbb::blocked_range<openvdb::Index32>(0, child_leafcount, 100), coarse_reducer);
		m_velocity_DOF.idx[i]->setTree(coarse_reducer.mCoarseDofGrid->treePtr());

		m_dof_manager[i] = std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(m_velocity_DOF.idx[i]->tree());

		//piecewise constant interpolation and restriction function
		//coarse voxel =8 fine voxels
		auto set_dof_mask_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto fine_dof_axr{ child.m_velocity_DOF.idx[i]->getConstUnsafeAccessor() };
			for (auto iter = leaf.beginValueAll(); iter; ++iter) {
				//the global coordinate in the coarse level
				auto C_gcoord{ iter.getCoord().asVec3i() };
				auto child_g_coord = C_gcoord * 2;

				bool no_target_dof = true;
				for (int ii = 0; ii < 2 && no_target_dof; ii++) {
					for (int jj = 0; jj < 2 && no_target_dof; jj++) {
						for (int kk = 0; kk < 2 && no_target_dof; kk++) {
							if (fine_dof_axr.isValueOn(
								openvdb::Coord(child_g_coord).offsetBy(ii, jj, kk))) {
								no_target_dof = false;
							}
						}
					}
				}//for all fine voxels accociated

				if (no_target_dof) {
					iter.setValueOff();
				}
				else {
					iter.setValueOn();
				}
			}//end for all voxel in this leaf
		};//end set dof_idx_op
		m_dof_manager[i]->foreach(set_dof_mask_op);

		set_dof(m_velocity_DOF.idx[i], m_ndof);
	}
}


namespace {
//set coefficient for each direction
//the related coefficients are set just as the
struct coarse_matrix_coef_setter {
	using leaf_type = openvdb::FloatTree::LeafNodeType;
	using leaves_vec_type = std::vector<leaf_type*>;
	using leaves_vec_ptr_type = leaves_vec_type*;

	coarse_matrix_coef_setter(
		leaves_vec_type* in_diag_leaves,
		leaves_vec_type* in_invdiag_leaves,
		leaves_vec_type** in_neg_dir_leaves_d012,
		leaves_vec_type* in_cross_term_leaves_d12,
		L_with_level::DOF_tuple_t in_coarse_dof_xyz,
		L_with_level::DOF_tuple_t in_fine_dof_xyz,
		openvdb::FloatGrid::Ptr in_fine_diag,
		openvdb::FloatGrid::Ptr in_fine_negdir_coef_d012[3],
		openvdb::FloatGrid::Ptr in_fine_cross_d12[4], int in_direction0) :
		m_diag_leaves(in_diag_leaves),
		m_invdiag_leaves(in_invdiag_leaves),
		neg_dir_leaves_d012(in_neg_dir_leaves_d012),
		m_cross_term_leaves_d12(in_cross_term_leaves_d12),
		m_coarse_dof_axr_xyz{ in_coarse_dof_xyz.idx[0]->tree(),in_coarse_dof_xyz.idx[1]->tree() ,in_coarse_dof_xyz.idx[2]->tree() },
		m_fine_dof_axr_xyz{ in_fine_dof_xyz.idx[0]->tree(),in_fine_dof_xyz.idx[1]->tree(),in_fine_dof_xyz.idx[2]->tree() },
		m_fine_diag_axr{ in_fine_diag->tree() },
		m_fine_neg_dir_axr_d012{ in_fine_negdir_coef_d012[0]->tree(),in_fine_negdir_coef_d012[1]->tree(),in_fine_negdir_coef_d012[2]->tree() },
		m_fine_cross_term_axr_d12{ in_fine_cross_d12[0]->tree(),in_fine_cross_d12[1]->tree(),in_fine_cross_d12[2]->tree(),in_fine_cross_d12[3]->tree() },
		d0{ in_direction0 }, d1{ (in_direction0 + 1) % 3 }, d2{ (in_direction0 + 2) % 3 }{
	}

	coarse_matrix_coef_setter(
		const coarse_matrix_coef_setter& other) :
		m_diag_leaves(other.m_diag_leaves),
		m_invdiag_leaves(other.m_invdiag_leaves),
		neg_dir_leaves_d012(other.neg_dir_leaves_d012),
		m_cross_term_leaves_d12(other.m_cross_term_leaves_d12),
		m_coarse_dof_axr_xyz{ other.m_coarse_dof_axr_xyz[0], other.m_coarse_dof_axr_xyz[1], other.m_coarse_dof_axr_xyz[2] },
		m_fine_dof_axr_xyz{ other.m_fine_dof_axr_xyz[0], other.m_fine_dof_axr_xyz[1], other.m_fine_dof_axr_xyz[2] },
		m_fine_diag_axr{ other.m_fine_diag_axr },
		m_fine_neg_dir_axr_d012{ other.m_fine_neg_dir_axr_d012[0], other.m_fine_neg_dir_axr_d012[1], other.m_fine_neg_dir_axr_d012[2] },
		m_fine_cross_term_axr_d12{ other.m_fine_cross_term_axr_d12[0], other.m_fine_cross_term_axr_d12[1], other.m_fine_cross_term_axr_d12[2], other.m_fine_cross_term_axr_d12[3] },
		d0{ other.d0 }, d1{ other.d1 }, d2{ other.d2 }{
		//clear accessor cache
		m_fine_diag_axr.clear();
		for (int i = 0; i < 3; i++) {
			m_coarse_dof_axr_xyz[i].clear();
			m_fine_dof_axr_xyz[i].clear();
			m_fine_neg_dir_axr_d012[i].clear();
		}
		for (int i = 0; i < 4; i++) {
			m_fine_cross_term_axr_d12[i].clear();
		}
	}

	//the operator runs over dilated DOF pattern of unioned channels, because some coefficients exist outside the
	//DOF pattern. 
	//All the coefficients leaves have dilated pattern, and would be turned off if no paired DOF exists
	void operator()(openvdb::BoolTree::LeafNodeType& dilated_leaf, openvdb::Index leafpos) const {
		leaf_type* diag_leaf = (*m_diag_leaves)[leafpos];
		leaf_type* invdiag_leaf = (*m_invdiag_leaves)[leafpos];
		leaf_type* neg_dir_leaf[3] = { (*neg_dir_leaves_d012[0])[leafpos],(*neg_dir_leaves_d012[1])[leafpos],(*neg_dir_leaves_d012[2])[leafpos] };
		leaf_type* cross_term_leaf[4] = { m_cross_term_leaves_d12[0][leafpos],
			m_cross_term_leaves_d12[1][leafpos] ,
			m_cross_term_leaves_d12[2][leafpos] ,
			m_cross_term_leaves_d12[3][leafpos] };
		openvdb::Coord dirxyz[3] = { openvdb::Coord(0),openvdb::Coord(0),openvdb::Coord(0) };
		dirxyz[0][d0]++; dirxyz[1][d1]++; dirxyz[2][d2]++;

		float rap_factor = 0.125f;
		for (auto iter = dilated_leaf.beginValueOn(); iter; ++iter) {
			const openvdb::Coord fine_base{ iter.getCoord().asVec3i() * 2 };
			//new diagonal term
			float new_diagonal_term = 0.f;
			//new negdir term
			float new_negdir_term_d012[3] = { 0.f,0.f,0.f };
			if (m_coarse_dof_axr_xyz[d0].getValue(iter.getCoord()) != -1) {
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						for (int kk = 0; kk < 2; kk++) {
							openvdb::Coord fine_coord = fine_base;
							fine_coord[d0] += ii;
							fine_coord[d1] += jj;
							fine_coord[d2] += kk;

							if (m_fine_dof_axr_xyz[d0].isValueOn(fine_coord)) {
								new_diagonal_term += m_fine_diag_axr.getValue(fine_coord);

								//for its six neighbors
								for (int negdir = 0; negdir < 3; negdir++) {
									float neg_term = m_fine_neg_dir_axr_d012[negdir].getValue(fine_coord);
									openvdb::Coord neg_neib = fine_coord - dirxyz[negdir];
									if (!m_fine_dof_axr_xyz[d0].isValueOn(neg_neib)) {
										continue;
									}
									switch (negdir) {
									case 0:
										if (0 == ii) {
											new_negdir_term_d012[negdir] += neg_term;
										}
										else {
											new_diagonal_term += neg_term * 2.0f;
										}
										break;
									case 1:
										if (0 == jj) {
											new_negdir_term_d012[negdir] += neg_term;
										}
										else {
											new_diagonal_term += neg_term * 2.0f;
										}
										break;
									case 2:
										if (0 == kk) {
											new_negdir_term_d012[negdir] += neg_term;
										}
										else {
											new_diagonal_term += neg_term * 2.0f;
										}
									}//end switch
								}//end for three negative direction
							}//end if fine coord is on
						}//end kk
					}//end jj
				}//end ii

				diag_leaf->setValueOn(iter.offset(), rap_factor * new_diagonal_term);
				float invdiag = 0.f;
				if (new_diagonal_term != 0) {
					invdiag = 1.0f / (rap_factor * new_diagonal_term);
				}
				invdiag_leaf->setValueOn(iter.offset(), invdiag);
				for (int negdir = 0; negdir < 3; negdir++) {
					auto neib_coord = iter.getCoord();
					neib_coord -= dirxyz[negdir];
					if (m_coarse_dof_axr_xyz[d0].isValueOn(neib_coord)) {
						neg_dir_leaf[negdir]->setValueOn(iter.offset(), rap_factor * new_negdir_term_d012[negdir]);
					}
					else {
						neg_dir_leaf[negdir]->setValueOff(iter.offset(), 0.f);
					}
				}
			}//if this channel is dof
			else {
				diag_leaf->setValueOff(iter.offset(), 0.f);
				invdiag_leaf->setValueOff(iter.offset(), 0.f);
				neg_dir_leaf[0]->setValueOff(iter.offset(), 0.f);
				neg_dir_leaf[1]->setValueOff(iter.offset(), 0.f);
				neg_dir_leaf[2]->setValueOff(iter.offset(), 0.f);
			}//else this channel is dof


			//new cross term
			//j along dir1
			//k along dir2
			//float new_cross_term[2][2] = { {0.f,0.f},{0.f,0.f} };
			for (int i = 0; i < 4; i++) {
				float new_cross_term = 0.f;
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						for (int kk = 0; kk < 2; kk++) {
							openvdb::Coord fine_coord = fine_base;
							fine_coord[d0] += ii;
							fine_coord[d1] += jj;
							fine_coord[d2] += kk;

							switch (i) {
							case 0:
								//00 term
								//00
								new_cross_term += m_fine_cross_term_axr_d12[0].getValue(fine_coord);
								//01+
								if (0 == kk) {
									new_cross_term += m_fine_cross_term_axr_d12[1].getValue(fine_coord);
								}
								//10
								if (0 == jj) {
									new_cross_term += m_fine_cross_term_axr_d12[2].getValue(fine_coord);
								}
								//11
								if (0 == kk && 0 == jj) {
									new_cross_term += m_fine_cross_term_axr_d12[3].getValue(fine_coord);
								}
								break;
							case 1:
								//01 term
								//00 has no contribution
								//01
								if (1 == kk) {
									new_cross_term += m_fine_cross_term_axr_d12[1].getValue(fine_coord);
								}
								//10 has no contribution
								//11
								if (0 == jj && 1 == kk) {
									new_cross_term += m_fine_cross_term_axr_d12[3].getValue(fine_coord);
								}
								break;
							case 2:
								//10 term
								//00 has no contribution
								//01 has no contribution
								//10
								if (1 == jj) {
									new_cross_term += m_fine_cross_term_axr_d12[2].getValue(fine_coord);
								}
								//11
								if (1 == jj && 0 == kk) {
									new_cross_term += m_fine_cross_term_axr_d12[3].getValue(fine_coord);
								}
								break;
							case 3:
								//11 term
								//00 has no contribution
								//01 has no contribution
								//10 has no contribution
								//11
								if (1 == jj && 1 == kk) {
									new_cross_term += m_fine_cross_term_axr_d12[3].getValue(fine_coord);
								}
							}//end switch four cross terms
						}//end kk
					}//end jj
				}//end ii

				//see if there is a pair of dof at coarse level
				int coarse_dof0 = -1, coarse_dof1 = -1;
				switch (i) {
				case 0:
					//00
					coarse_dof0 = m_coarse_dof_axr_xyz[d1].getValue(iter.getCoord());
					coarse_dof1 = m_coarse_dof_axr_xyz[d2].getValue(iter.getCoord());
					break;
				case 1:
					//01
					coarse_dof0 = m_coarse_dof_axr_xyz[d1].getValue(iter.getCoord());
					coarse_dof1 = m_coarse_dof_axr_xyz[d2].getValue(iter.getCoord() + dirxyz[2]);
					break;
				case 2:
					//10
					coarse_dof0 = m_coarse_dof_axr_xyz[d1].getValue(iter.getCoord() + dirxyz[1]);
					coarse_dof1 = m_coarse_dof_axr_xyz[d2].getValue(iter.getCoord());
					break;
				case 3:
					//11
					coarse_dof0 = m_coarse_dof_axr_xyz[d1].getValue(iter.getCoord() + dirxyz[1]);
					coarse_dof1 = m_coarse_dof_axr_xyz[d2].getValue(iter.getCoord() + dirxyz[2]);
				}//end switch cross terms

				if (coarse_dof0 != -1 && coarse_dof1 != -1) {
					cross_term_leaf[i]->setValueOn(iter.offset(), rap_factor * new_cross_term);
				}
				else {
					cross_term_leaf[i]->setValueOff(iter.offset(), 0.f);
				}
			}//end for i= 4 cross term
		}//end over all on voxels
	}

	//three directions
	//when setting a coefficients, 
	//d0 points outside of the screen, d1 to the right
	//d2 to the top
	const int d0, d1, d2;

	//coefficients leaves
	leaves_vec_type* m_diag_leaves, * m_invdiag_leaves;

	//three negative direction coefficients leaves
	//ordered in negative d0,d1,d2 direction
	//it is the re-ordered pointers or xyz
	//first deref goes to the xyz pointer
	//second deref goes to the corresponding vector
	leaves_vec_type** neg_dir_leaves_d012;

	//four cross terms
	//ordered in jk = 00 01 10 11;
	leaves_vec_type* m_cross_term_leaves_d12;

	//index of degree of freedom for three channels
	mutable openvdb::Int32Grid::ConstUnsafeAccessor m_coarse_dof_axr_xyz[3];
	mutable openvdb::Int32Grid::ConstUnsafeAccessor m_fine_dof_axr_xyz[3];

	mutable openvdb::FloatGrid::ConstUnsafeAccessor m_fine_diag_axr, m_fine_neg_dir_axr_d012[3], m_fine_cross_term_axr_d12[4];
};
}//end namespace

void L_with_level::calculate_matrix_coefficients_from_child(const L_with_level& child)
{
	auto dilated_DOF_pattern = openvdb::BoolGrid::create();
	for (int i = 0; i < 3; i++) {
		dilated_DOF_pattern->topologyUnion(*m_velocity_DOF.idx[i]);
	}

	openvdb::tools::dilateActiveValues(dilated_DOF_pattern->tree(), 3, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	openvdb::tree::LeafManager<openvdb::BoolTree> patternman(dilated_DOF_pattern->tree());

	using leaf_container_type = coarse_matrix_coef_setter::leaves_vec_type;
	std::unique_ptr<leaf_container_type[]> diag_leaves, invdiag_leaves,
		neg_x_leaves, neg_y_leaves, neg_z_leaves;
	diag_leaves = std::make_unique<leaf_container_type[]>(3);
	invdiag_leaves = std::make_unique<leaf_container_type[]>(3);
	neg_x_leaves = std::make_unique<leaf_container_type[]>(3);
	neg_y_leaves = std::make_unique<leaf_container_type[]>(3);
	neg_z_leaves = std::make_unique<leaf_container_type[]>(3);

	std::unique_ptr<leaf_container_type[]> cross_term_leaves;
	cross_term_leaves = std::make_unique<leaf_container_type[]>(3 * 2 * 2);
	int nleaf = dilated_DOF_pattern->tree().leafCount();

	//xyz ordered leaves
	for (int i = 0; i < 3; i++) {
		m_diagonal.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
		diag_leaves[i].reserve(nleaf); m_diagonal.v[i]->tree().getNodes(diag_leaves[i]);

		m_invdiag.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
		invdiag_leaves[i].reserve(nleaf); m_invdiag.v[i]->tree().getNodes(invdiag_leaves[i]);

		m_neg_x.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
		neg_x_leaves[i].reserve(nleaf); m_neg_x.v[i]->tree().getNodes(neg_x_leaves[i]);

		m_neg_y.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
		neg_y_leaves[i].reserve(nleaf); m_neg_y.v[i]->tree().getNodes(neg_y_leaves[i]);

		m_neg_z.v[i]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
		neg_z_leaves[i].reserve(nleaf); m_neg_z.v[i]->tree().getNodes(neg_z_leaves[i]);

		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				m_cross_channel[i][j][k]->setTree(std::make_shared<openvdb::FloatTree>(dilated_DOF_pattern->tree(), 0.f, openvdb::TopologyCopy()));
				cross_term_leaves[4 * i + 2 * j + k].reserve(nleaf); m_cross_channel[i][j][k]->tree().getNodes(cross_term_leaves[4 * i + 2 * j + k]);
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		leaf_container_type* d012_ordered_neg_leaves[3];
		openvdb::FloatGrid::Ptr fine_negdir_coef_d012[3];
		openvdb::FloatGrid::Ptr child_cross_term_d12[4];

		child_cross_term_d12[0] = child.m_cross_channel[i][0][0];
		child_cross_term_d12[1] = child.m_cross_channel[i][0][1];
		child_cross_term_d12[2] = child.m_cross_channel[i][1][0];
		child_cross_term_d12[3] = child.m_cross_channel[i][1][1];

		switch (i) {
		case 0:
			d012_ordered_neg_leaves[0] = neg_x_leaves.get();
			d012_ordered_neg_leaves[1] = neg_y_leaves.get();
			d012_ordered_neg_leaves[2] = neg_z_leaves.get();
			fine_negdir_coef_d012[0] = child.m_neg_x.v[0];
			fine_negdir_coef_d012[1] = child.m_neg_y.v[0];
			fine_negdir_coef_d012[2] = child.m_neg_z.v[0];
			break;
		case 1:
			d012_ordered_neg_leaves[0] = neg_y_leaves.get() + 1;
			d012_ordered_neg_leaves[1] = neg_z_leaves.get() + 1;
			d012_ordered_neg_leaves[2] = neg_x_leaves.get() + 1;
			fine_negdir_coef_d012[0] = child.m_neg_y.v[1];
			fine_negdir_coef_d012[1] = child.m_neg_z.v[1];
			fine_negdir_coef_d012[2] = child.m_neg_x.v[1];
			break;
		case 2:
			d012_ordered_neg_leaves[0] = neg_z_leaves.get() + 2;
			d012_ordered_neg_leaves[1] = neg_x_leaves.get() + 2;
			d012_ordered_neg_leaves[2] = neg_y_leaves.get() + 2;
			fine_negdir_coef_d012[0] = child.m_neg_z.v[2];
			fine_negdir_coef_d012[1] = child.m_neg_x.v[2];
			fine_negdir_coef_d012[2] = child.m_neg_y.v[2];
		}

		coarse_matrix_coef_setter Op(
			diag_leaves.get() + i,
			invdiag_leaves.get() + i,
			d012_ordered_neg_leaves,
			cross_term_leaves.get() + i * 4,
			m_velocity_DOF,
			child.m_velocity_DOF,
			child.m_diagonal.v[i],
			fine_negdir_coef_d012,
			child_cross_term_d12, i);

		patternman.foreach(Op);
	}
}

namespace {
struct SPAI0_collector {

	SPAI0_collector(
		std::vector<openvdb::FloatTree::LeafNodeType*>& in_SPAI0_leaves,
		openvdb::FloatGrid::Ptr in_diag,
		openvdb::FloatGrid::Ptr in_negx,
		openvdb::FloatGrid::Ptr in_negy,
		openvdb::FloatGrid::Ptr in_negz,
		openvdb::FloatGrid::Ptr cross_term[3][2][2],
		L_with_level::DOF_tuple_t in_dof, int in_channel) :
		m_negx_axr{ in_negx->tree() },
		m_negy_axr{ in_negy->tree() },
		m_negz_axr{ in_negz->tree() },
		m_dof_axr{ in_dof.idx[0]->tree(),in_dof.idx[1]->tree() ,in_dof.idx[2]->tree() },
		m_cross_axr{
			{{cross_term[0][0][0]->tree(),cross_term[0][0][1]->tree()},
			{cross_term[0][1][0]->tree(),cross_term[0][1][1]->tree()}},
			{{cross_term[1][0][0]->tree(),cross_term[1][0][1]->tree()},
			{cross_term[1][1][0]->tree(),cross_term[1][1][1]->tree()}},
			{{cross_term[2][0][0]->tree(),cross_term[2][0][1]->tree()},
			{cross_term[2][1][0]->tree(),cross_term[2][1][1]->tree()}} },
			m_diag(in_diag),
			m_SPAI0_leaves{ in_SPAI0_leaves }, channel(in_channel){}

	SPAI0_collector(const SPAI0_collector& other) :
		m_negx_axr{ other.m_negx_axr },
		m_negy_axr{ other.m_negy_axr },
		m_negz_axr{ other.m_negz_axr },
		m_dof_axr{ other.m_dof_axr[0],other.m_dof_axr[1],other.m_dof_axr[2] },
		m_cross_axr{
			{{other.m_cross_axr[0][0][0], other.m_cross_axr[0][0][1]},
			{other.m_cross_axr[0][1][0], other.m_cross_axr[0][1][1]}},
			{{other.m_cross_axr[1][0][0], other.m_cross_axr[1][0][1]},
			{other.m_cross_axr[1][1][0], other.m_cross_axr[1][1][1]}},
			{{other.m_cross_axr[2][0][0], other.m_cross_axr[2][0][1]},
			{other.m_cross_axr[2][1][0], other.m_cross_axr[2][1][1]}} },
			m_diag(other.m_diag),
			m_SPAI0_leaves(other.m_SPAI0_leaves), channel(other.channel) {
		//clear accessors
		m_negx_axr.clear();
		m_negy_axr.clear();
		m_negz_axr.clear();
		for (int i = 0; i < 3; i++) {
			m_dof_axr[i].clear();
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					m_cross_axr[i][j][k].clear();
				}
			}
		}
	}

	template<int c>
	void channel_operator(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const {
		const int d0 = c;
		const int d1 = (d0 + 1) % 3;
		const int d2 = (d0 + 2) % 3;

		openvdb::Coord dir[3] = { openvdb::Coord{0},openvdb::Coord{0},openvdb::Coord{0} };
		dir[0][d0]++; dir[1][d1]++; dir[2][d2]++;

		auto diag_leaf = m_diag->tree().probeConstLeaf(leaf.origin());
		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
			//diagonal term
			float denominator = 0.f;
			float numerator = 0.f;
			if (m_dof_axr[d0].isValueOn(iter.getCoord())) {

				{
					float this_diag = diag_leaf->getValue(iter.offset());
					denominator += this_diag * this_diag;
					numerator = this_diag;
				}

				//x+
				{
					openvdb::Coord neib = iter.getCoord();
					neib[0]++;
					if (m_dof_axr[d0].isValueOn(neib)) {
						float posx_term = m_negx_axr.getValue(neib);
						denominator += posx_term * posx_term;
					}
				}
				//x-
				{
					openvdb::Coord neib = iter.getCoord();
					neib[0]--;
					if (m_dof_axr[d0].isValueOn(neib)) {
						float negx_term = m_negx_axr.getValue(iter.getCoord());
						denominator += negx_term * negx_term;
					}
				}
				//y+
				{
					openvdb::Coord neib = iter.getCoord();
					neib[1]++;
					if (m_dof_axr[d0].isValueOn(neib)) {
						float posy_term = m_negy_axr.getValue(neib);
						denominator += posy_term * posy_term;
					}
				}
				//y-
				{
					openvdb::Coord neib = iter.getCoord();
					neib[1]--;
					if (m_dof_axr[d0].isValueOn(neib)) {
						float negy_term = m_negy_axr.getValue(iter.getCoord());
						denominator += negy_term * negy_term;
					}
				}
				//z+
				{
					openvdb::Coord neib = iter.getCoord();
					neib[2]++;
					if (m_dof_axr[d0].isValueOn(neib)) {
						float posz_term = m_negz_axr.getValue(neib);
						denominator += posz_term * posz_term;
					}
				}
				//z-
				{
					openvdb::Coord neib = iter.getCoord();
					neib[2]--;
					if (m_dof_axr[d0].isValueOn(neib)) {
						float negz_term = m_negz_axr.getValue(iter.getCoord());
						denominator += negz_term * negz_term;
					}
				}


				//Cross terms, see the light weight applier for details
				//Td2
				//Td200
				if (m_dof_axr[d1].isValueOn(iter.getCoord())) {
					float td200 = m_cross_axr[d2][0][0].getValue(iter.getCoord());
					denominator += td200 * td200;
				}
				//Td201
				if (m_dof_axr[d1].isValueOn(iter.getCoord() + dir[1])) {
					float td201 = m_cross_axr[d2][0][1].getValue(iter.getCoord());
					denominator += td201 * td201;
				}
				//Td210
				if (m_dof_axr[d1].isValueOn(iter.getCoord() - dir[0])) {
					float td210 = m_cross_axr[d2][1][0].getValue(iter.getCoord() - dir[0]);
					denominator += td210 * td210;
				}
				//Td211
				if (m_dof_axr[d1].isValueOn(iter.getCoord() - dir[0] + dir[1])) {
					float td211 = m_cross_axr[d2][1][1].getValue(iter.getCoord() - dir[0]);
					denominator += td211 * td211;
				}

				//Td1
				//Td100
				if (m_dof_axr[d2].isValueOn(iter.getCoord())) {
					float td100 = m_cross_axr[d1][0][0].getValue(iter.getCoord());
					denominator += td100 * td100;
				}
				//Td101
				if (m_dof_axr[d2].isValueOn(iter.getCoord() - dir[0])) {
					float td101 = m_cross_axr[d1][0][1].getValue(iter.getCoord() - dir[0]);
					denominator += td101 * td101;
				}
				//Td110
				if (m_dof_axr[d2].isValueOn(iter.getCoord() + dir[2])) {
					float td110 = m_cross_axr[d1][1][0].getValue(iter.getCoord());
					denominator += td110 * td110;
				}
				//Td111
				if (m_dof_axr[d2].isValueOn(iter.getCoord() + dir[2] - dir[0])) {
					float td111 = m_cross_axr[d1][1][1].getValue(iter.getCoord() - dir[0]);
					denominator += td111 * td111;
				}

				if (0 != denominator) {
					m_SPAI0_leaves[leafpos]->setValueOn(iter.offset(), numerator / denominator);
				}
				else {
					m_SPAI0_leaves[leafpos]->setValueOn(iter.offset(), 0.f);
				}
			}//end if there is diagonal term
		}//loop over all on voxel of a dof
	}//end component operator

	//loop over the dilated union DOF mask to include all coefficients
	void operator()(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const {
		switch (channel) {
		case 0:
			channel_operator<0>(leaf, leafpos);
			break;
		case 1:
			channel_operator<1>(leaf, leafpos);
			break;
		case 2:
			channel_operator<2>(leaf, leafpos);
			break;
		}
	}

	int channel;

	mutable openvdb::FloatGrid::ConstUnsafeAccessor m_negx_axr, m_negy_axr, m_negz_axr;
	mutable openvdb::Int32Grid::ConstUnsafeAccessor m_dof_axr[3];
	mutable openvdb::FloatGrid::ConstUnsafeAccessor m_cross_axr[3][2][2];

	openvdb::FloatGrid::Ptr m_diag;

	std::vector<openvdb::FloatTree::LeafNodeType*>& m_SPAI0_leaves;
};
}
void L_with_level::build_SPAI0_matrix()
{
	if (m_SPAI0_initialized) {
		return;
	}
	m_SPAI0_initialized = true;

	m_SPAI0 = get_zero_vec();
	m_SPAI0.setName("m_SPAI0");

	for (int i = 0; i < 3; i++) {
		std::vector<openvdb::FloatTree::LeafNodeType*> diag_leaves, spai0_leaves;
		size_t nleaf = m_dof_manager[i]->leafCount();
		spai0_leaves.reserve(nleaf);

		m_SPAI0.v[i]->tree().getNodes(spai0_leaves);

		SPAI0_collector collector(spai0_leaves, m_diagonal.v[i],
			m_neg_x.v[i],
			m_neg_y.v[i],
			m_neg_z.v[i],
			m_cross_channel,
			m_velocity_DOF, i);
		m_dof_manager[i]->foreach(collector);
	}
}

void L_with_level::trim_default_nodes()
{
	if (m_level != 0) {
		return;
	}
	float eps = 1e-3f;

	for (int c = 0; c < 3; c++) {
		//trim diagonal
		LaplacianWithLevel::trimDefaultNodes(m_diagonal.v[c], m_default_diag, m_default_diag * eps);

		//trim invdiag
		LaplacianWithLevel::trimDefaultNodes(m_invdiag.v[c], 1.0f / m_default_diag, 1.0f / m_default_diag * eps);

		//same channel off-diagonal terms
		LaplacianWithLevel::trimDefaultNodes(m_neg_x.v[c], m_neg_x.v[c]->background(), m_neg_x.v[c]->background() * eps);
		LaplacianWithLevel::trimDefaultNodes(m_neg_y.v[c], m_neg_y.v[c]->background(), m_neg_y.v[c]->background() * eps);
		LaplacianWithLevel::trimDefaultNodes(m_neg_z.v[c], m_neg_z.v[c]->background(), m_neg_z.v[c]->background() * eps);

		//trim cross-terms
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				LaplacianWithLevel::trimDefaultNodes(
					m_cross_channel[c][j][k],
					m_cross_channel[c][j][k]->background(),
					m_cross_channel[c][j][k]->background() * eps);
			}
		}
	}
}

namespace {
struct rhs_reducer {
	using leaf_type = openvdb::FloatTree::LeafNodeType;
	using leaves_vec_t = std::vector<leaf_type*>;
	using leaves_vec_ptr_t = leaves_vec_t*;

	rhs_reducer(const L_with_level::DOF_tuple_t in_dof,
		const openvdb::FloatGrid::Ptr in_viscosity,
		const openvdb::FloatGrid::Ptr in_edge_volume[3],
		const openvdb::FloatGrid::Ptr in_voxel_volume,
		const openvdb::Vec3fGrid::Ptr in_solid_vel,
		const openvdb::BoolGrid::Ptr in_is_solid[3],
		const std::vector<openvdb::BoolTree::LeafNodeType*>& in_dilated_solid_pattern_leaves, float in_dt, float in_rho) :
		m_dof_axr{ in_dof.idx[0]->tree(), in_dof.idx[1]->tree() ,in_dof.idx[2]->tree() },
		m_viscosity_axr{ in_viscosity->tree() },
		m_edge_volume_axr{ in_edge_volume[0]->tree(), in_edge_volume[1]->tree(), in_edge_volume[2]->tree() },
		m_volume_axr{ in_voxel_volume->tree() },
		m_solid_vel_axr{ in_solid_vel->tree() },
		m_is_solid_axr{ in_is_solid[0]->tree(),in_is_solid[1]->tree(),in_is_solid[2]->tree() },
		m_dilated_solid_leaves(in_dilated_solid_pattern_leaves)
	{
		m_rhs_contribution.clear();
		dt = in_dt;
		rho = in_rho;
		dx = (float)in_voxel_volume->voxelSize()[0];
	}

	rhs_reducer(const rhs_reducer& other, tbb::split) :
		m_dof_axr{ other.m_dof_axr[0], other.m_dof_axr[1], other.m_dof_axr[2] },
		m_viscosity_axr{ other.m_viscosity_axr },
		m_edge_volume_axr{ other.m_edge_volume_axr[0],other.m_edge_volume_axr[1],other.m_edge_volume_axr[2] },
		m_volume_axr{ other.m_volume_axr },
		m_solid_vel_axr{ other.m_solid_vel_axr },
		m_is_solid_axr{ other.m_is_solid_axr[0],other.m_is_solid_axr[1],other.m_is_solid_axr[2] },
		m_dilated_solid_leaves(other.m_dilated_solid_leaves), dt(other.dt), rho(other.rho), dx(other.dx)
	{
		m_rhs_contribution.clear();
		//clear the accessor
		for (int i = 0; i < 3; i++) {
			m_dof_axr[i].clear();
			m_edge_volume_axr[i].clear();
			m_is_solid_axr[i].clear();
		}
		m_viscosity_axr.clear();
		m_volume_axr.clear();
		m_solid_vel_axr.clear();
	}

	template <int c>
	void channel_op(size_t leafpos) {
		//int c;
		const int d0 = c;
		const int d1 = (c + 1) % 3;
		const int d2 = (c + 2) % 3;
		openvdb::Coord dir0{ 0 }, dir1{ 0 }, dir2{ 0 };
		dir0[d0]++; dir1[d1]++; dir2[d2]++;

		float factor = dt / (rho * dx * dx);
		//the face
		for (auto iter = m_dilated_solid_leaves[leafpos]->beginValueAll(); iter; ++iter) {
			const int this_dof = m_dof_axr[d0].getValue(iter.getCoord());

			if (this_dof != -1) {
				float temp_rhs = 0;

				//+ d0 direction
				{
					const int pd0dof = m_dof_axr[d0].getValue(iter.getCoord() + dir0);
					float term = 2 * factor * m_volume_axr.getValue(iter.getCoord()) * m_viscosity_axr.getValue(iter.getCoord());
					/*if (-1 == pd0dof) {*/
					if (-1 == pd0dof && m_is_solid_axr[d0].isValueOn(iter.getCoord() + dir0)) {
						temp_rhs += term * m_solid_vel_axr.getValue(iter.getCoord() + dir0)[d0];
					}
				}

				//- d0 direction
				{
					const int nd0dof = m_dof_axr[d0].getValue(iter.getCoord() - dir0);
					float term = 2 * factor * m_volume_axr.getValue(iter.getCoord() - dir0) * m_viscosity_axr.getValue(iter.getCoord() - dir0);
					/*if (-1 == nd0dof) {*/
					if (-1 == nd0dof && m_is_solid_axr[d0].isValueOn(iter.getCoord() - dir0)) {
						temp_rhs += term * m_solid_vel_axr.getValue(iter.getCoord() - dir0)[d0];
					}
				}


				//index space face center for the viscosity sampling
				openvdb::Vec3f fc_ipos = iter.getCoord().asVec3s();
				fc_ipos[d0] -= 0.5f;

				//+ d1 direction
				{
					float edgevoldir2posp1 = m_edge_volume_axr[d2].getValue(iter.getCoord() + dir1);
					openvdb::Vec3f edgedir2posp1_ipos = fc_ipos;
					//move to positive d1 direction
					edgedir2posp1_ipos[d1] += 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir2posp1_ipos);
					float term = factor * edgevoldir2posp1 * sampled_viscosity;
					const int pd1dof = m_dof_axr[d0].getValue(iter.getCoord() + dir1);
					/*if (-1 == pd1dof) {*/
					if (-1 == pd1dof && m_is_solid_axr[d0].isValueOn(iter.getCoord() + dir1)) {
						temp_rhs += term * m_solid_vel_axr.getValue(iter.getCoord() + dir1)[d0];
					}
				}

				//- d1 direction
				{
					float edgevoldir2posn1 = m_edge_volume_axr[d2].getValue(iter.getCoord());
					openvdb::Vec3f  edgedir2posn1_ipos = fc_ipos;
					//move to the negative d1 direction
					edgedir2posn1_ipos[d1] -= 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir2posn1_ipos);
					float term = factor * edgevoldir2posn1 * sampled_viscosity;
					const int nd1dof = m_dof_axr[d0].getValue(iter.getCoord() - dir1);
					/*if (-1 == nd1dof) {*/
					if (-1 == nd1dof && m_is_solid_axr[d0].isValueOn(iter.getCoord() - dir1)) {
						temp_rhs += term * m_solid_vel_axr.getValue(iter.getCoord() - dir1)[d0];
					}
				}

				//+ d2 direction
				{
					float edgevold1posp2 = m_edge_volume_axr[d1].getValue(iter.getCoord() + dir2);
					openvdb::Vec3f edgedir1posp2_ipos = fc_ipos;
					//move to positive d2 direction
					edgedir1posp2_ipos[d2] += 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir1posp2_ipos);
					float term = factor * edgevold1posp2 * sampled_viscosity;
					const int pd2dof = m_dof_axr[d0].getValue(iter.getCoord() + dir2);
					if (-1 == pd2dof && m_is_solid_axr[d0].isValueOn(iter.getCoord() + dir2)) {
						temp_rhs += term * m_solid_vel_axr.getValue(iter.getCoord() + dir2)[d0];
					}
				}

				//- d2 direction
				{
					float edgevold1posn2 = m_edge_volume_axr[d1].getValue(iter.getCoord());
					openvdb::Vec3f edgedir1posn2_ipos = fc_ipos;
					//move to the negative d2 direction
					edgedir1posn2_ipos[d2] -= 0.5f;
					float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir1posn2_ipos);
					float term = factor * edgevold1posn2 * sampled_viscosity;
					const int nd2dof = m_dof_axr[d0].getValue(iter.getCoord() - dir2);
					if (-1 == nd2dof && m_is_solid_axr[d0].isValueOn(iter.getCoord() - dir2)) {
						temp_rhs += term * m_solid_vel_axr.getValue(iter.getCoord() - dir2)[d0];
					}
				}
				//add the matrix rhs contribution to the original rhs term
				//iter.setValue(iter.getValue()+temp_rhs);

				if (temp_rhs != 0) {
					m_rhs_contribution.push_back({ this_dof, temp_rhs });
				}
			}//end if this face voxel is a dof

			//cross terms are between the other faces
			const int dof_face_dir1neg = m_dof_axr[d1].getValue(iter.getCoord());
			const int dof_face_dir1pos = m_dof_axr[d1].getValue(iter.getCoord() + dir1);
			const int dof_face_dir2neg = m_dof_axr[d2].getValue(iter.getCoord());
			const int dof_face_dir2pos = m_dof_axr[d2].getValue(iter.getCoord() + dir2);

			const bool is_solid_face_dir1neg = m_is_solid_axr[d1].isValueOn(iter.getCoord());
			const bool is_solid_face_dir1pos = m_is_solid_axr[d1].isValueOn(iter.getCoord() + dir1);
			const bool is_solid_face_dir2neg = m_is_solid_axr[d2].isValueOn(iter.getCoord());
			const bool is_solid_face_dir2pos = m_is_solid_axr[d2].isValueOn(iter.getCoord() + dir2);

			//00
			if (dof_face_dir1neg != -1 || dof_face_dir2neg != -1) {
				float edgevoldir0p00 = m_edge_volume_axr[d0].getValue(iter.getCoord());
				openvdb::Vec3f edgedir0p00_ipos = iter.getCoord().asVec3s();
				edgedir0p00_ipos[d1] -= 0.5f;
				edgedir0p00_ipos[d2] -= 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p00_ipos);
				float term = factor * edgevoldir0p00 * sampled_viscosity;

				/*m_matrix_triplets.push_back(tt(dof_face_dir1neg, dof_face_dir2neg, term));
				m_matrix_triplets.push_back(tt(dof_face_dir2neg, dof_face_dir1neg, term));*/
				if (dof_face_dir1neg == -1 && is_solid_face_dir1neg) {
					m_rhs_contribution.push_back({ dof_face_dir2neg,-term * m_solid_vel_axr.getValue(iter.getCoord())[d1] });
				}
				if (dof_face_dir2neg == -1 && is_solid_face_dir2neg) {
					m_rhs_contribution.push_back({ dof_face_dir1neg,-term * m_solid_vel_axr.getValue(iter.getCoord())[d2] });
				}
			}


			//01
			if (dof_face_dir1neg != -1 || dof_face_dir2pos != -1) {
				float edgevoldir0p01 = m_edge_volume_axr[d0].getValue(iter.getCoord() + dir2);
				openvdb::Vec3f edgedir0p01_ipos = iter.getCoord().asVec3s();
				edgedir0p01_ipos[d1] -= 0.5f;
				edgedir0p01_ipos[d2] += 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p01_ipos);
				float term = -factor * edgevoldir0p01 * sampled_viscosity;
				/*m_matrix_triplets.push_back(tt(dof_face_dir1neg, dof_face_dir2pos, term));
				m_matrix_triplets.push_back(tt(dof_face_dir2pos, dof_face_dir1neg, term));*/
				if (dof_face_dir1neg == -1 && is_solid_face_dir1neg) {
					m_rhs_contribution.push_back({ dof_face_dir2pos, -term * m_solid_vel_axr.getValue(iter.getCoord())[d1] });
				}
				if (dof_face_dir2pos == -1 && is_solid_face_dir2pos) {
					m_rhs_contribution.push_back({ dof_face_dir1neg, -term * m_solid_vel_axr.getValue(iter.getCoord() + dir2)[d2] });
				}
			}


			//10
			if (dof_face_dir1pos != -1 || dof_face_dir2neg != -1) {
				float edgevoldir0p10 = m_edge_volume_axr[d0].getValue(iter.getCoord() + dir1);
				openvdb::Vec3f edgedir0p10_ipos = iter.getCoord().asVec3s();
				edgedir0p10_ipos[d1] += 0.5f;
				edgedir0p10_ipos[d2] -= 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p10_ipos);
				float term = -factor * edgevoldir0p10 * sampled_viscosity;
				/*m_matrix_triplets.push_back(tt(dof_face_dir1pos, dof_face_dir2neg, term));
				m_matrix_triplets.push_back(tt(dof_face_dir2neg, dof_face_dir1pos, term));*/
				if (dof_face_dir1pos == -1 && is_solid_face_dir1pos) {
					m_rhs_contribution.push_back({ dof_face_dir2neg, -term * m_solid_vel_axr.getValue(iter.getCoord() + dir1)[d1] });
				}
				if (dof_face_dir2neg == -1 && is_solid_face_dir2neg) {
					m_rhs_contribution.push_back({ dof_face_dir1pos, -term * m_solid_vel_axr.getValue(iter.getCoord())[d2] });
				}
			}

			//11
			if (dof_face_dir1pos != -1 || dof_face_dir2pos != -1) {
				float edgevoldir0p11 = m_edge_volume_axr[d0].getValue(iter.getCoord() + dir1 + dir2);
				openvdb::Vec3f edgedir0p11_ipos = iter.getCoord().asVec3s();
				edgedir0p11_ipos[d1] += 0.5f;
				edgedir0p11_ipos[d2] += 0.5f;
				float sampled_viscosity = openvdb::tools::BoxSampler::sample(m_viscosity_axr, edgedir0p11_ipos);
				float term = factor * edgevoldir0p11 * sampled_viscosity;
				/*m_matrix_triplets.push_back(tt(dof_face_dir1pos, dof_face_dir2pos, term));
				m_matrix_triplets.push_back(tt(dof_face_dir2pos, dof_face_dir1pos, term));*/
				if (dof_face_dir1pos == -1 && is_solid_face_dir1pos) {
					m_rhs_contribution.push_back({ dof_face_dir2pos, -term * m_solid_vel_axr.getValue(iter.getCoord() + dir1)[d1] });
				}
				if (dof_face_dir2pos == -1 && is_solid_face_dir2pos) {
					m_rhs_contribution.push_back({ dof_face_dir1pos, -term * m_solid_vel_axr.getValue(iter.getCoord() + dir2)[d2] });
				}
			}
		}//end for all active diagonal voxels
	}

	void operator()(const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i) {
			channel_op<0>(i);
		}
		for (size_t i = r.begin(); i != r.end(); ++i) {
			channel_op<1>(i);
		}
		for (size_t i = r.begin(); i != r.end(); ++i) {
			channel_op<2>(i);
		}
	}

	void join(const rhs_reducer& other) {
		size_t original_size = m_rhs_contribution.size();
		size_t extra_size = other.m_rhs_contribution.size();
		m_rhs_contribution.resize(original_size + extra_size);
		std::copy(other.m_rhs_contribution.begin(), other.m_rhs_contribution.end(), m_rhs_contribution.begin() + original_size);
	}

	//idx of the rhs, contribution
	//this vector is to be processed after the reduction
	std::vector<std::pair<size_t, float>> m_rhs_contribution;

	const std::vector<openvdb::BoolTree::LeafNodeType*>& m_dilated_solid_leaves;

	//index of degree of freedom for three channels
	openvdb::Int32Grid::ConstUnsafeAccessor m_dof_axr[3];
	//viscosity is assumed to be voxel center variable.
	openvdb::FloatGrid::ConstUnsafeAccessor m_viscosity_axr, m_edge_volume_axr[3], m_volume_axr;

	openvdb::Vec3fGrid::ConstUnsafeAccessor m_solid_vel_axr;
	//is_solid_axr is used to determine if a face has prescribed velocity or not
	//dilated is solid just provide a way to know if current leaf is affected by prescribed velocity.
	openvdb::BoolGrid::ConstUnsafeAccessor m_is_solid_axr[3];

	float dt, rho, dx;
};//end rhs_builder

}//end namespace
packed_FloatGrid3 L_with_level::build_rhs(packed_FloatGrid3 in_velocity_field, openvdb::Vec3fGrid::Ptr in_solid_velocity) const
{
	packed_FloatGrid3 result = get_zero_vec();
	if (m_level != 0) {
		return result;
	}

	//the first part of the right hand side comes from the diagonal part of the original velocity
	for (int i = 0; i < 3; i++) {
		//called by DOF managers
		auto diagonal_rhs_builder = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto channel_vol_axr = m_face_center_vol.v[i]->getConstUnsafeAccessor();
			auto in_liquid_velocity_axr = in_velocity_field.v[i]->getConstUnsafeAccessor();
			auto result_leaf = result.v[i]->tree().probeLeaf(leaf.origin());

			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				float face_vol = channel_vol_axr.getValue(iter.getCoord());
				if (face_vol < m_diag_frac_epsl) {
					face_vol = m_diag_frac_epsl;
				}
				result_leaf->setValueOn(iter.offset(),
					in_liquid_velocity_axr.getValue(iter.getCoord()) * face_vol);
			}
		};
		m_dof_manager[i]->foreach(diagonal_rhs_builder);
	}

	auto dilated_is_solidmask = openvdb::BoolGrid::create();
	for (int i = 0; i < 3; i++) {
		dilated_is_solidmask->topologyUnion(*m_is_solid[i]);
	}

	openvdb::tools::dilateActiveValues(dilated_is_solidmask->tree(), 3,openvdb::tools::NearestNeighbors::NN_FACE, openvdb::tools::TilePolicy::EXPAND_TILES);

	std::vector<openvdb::BoolTree::LeafNodeType*> solid_pattern_leaves;
	solid_pattern_leaves.reserve(dilated_is_solidmask->tree().leafCount());
	dilated_is_solidmask->tree().getNodes(solid_pattern_leaves);


	//the second part comes from matrix terms
	//the input velocity field has more elements than the DOF
	//some of the velocity sample are inside the solid
	//they are treated as known variables, and contribute to the right hand side.
	rhs_reducer Op(m_velocity_DOF,
		m_viscosity,
		m_edge_center_vol,
		m_voxel_vol,
		in_solid_velocity,
		m_is_solid,
		solid_pattern_leaves, m_dt, m_rho);

	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, solid_pattern_leaves.size()), Op);

	//collect the rhs contribution into a hashmap
	//the matrix terms usually only exist near the solid part
	std::unordered_map<size_t, float> matrix_rhs_contribution;
	for (auto& contribution : Op.m_rhs_contribution) {
		auto existing_contribution = matrix_rhs_contribution.find(contribution.first);
		if (existing_contribution != matrix_rhs_contribution.end()) {
			existing_contribution->second += contribution.second;
		}
		else {
			matrix_rhs_contribution[contribution.first] = contribution.second;
		}
	}

	//collect the extra rhs from matrix
	for (int i = 0; i < 3; i++) {
		auto extra_dof_collector = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto result_leaf = result.v[i]->tree().probeLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				auto possible_matrix_contribution = matrix_rhs_contribution.find(iter.getValue());
				if (possible_matrix_contribution != matrix_rhs_contribution.end()) {
					float temp_contribution = possible_matrix_contribution->second;
					result_leaf->setValueOn(iter.offset(), result_leaf->getValue(iter.offset()) + temp_contribution);
				}
			}
		};
		m_dof_manager[i]->foreach(extra_dof_collector);
	}

	return result;
}

Eigen::VectorXf L_with_level::to_Eigenvector(packed_FloatGrid3 in_float3grid) const
{
	Eigen::VectorXf result;
	result.setZero(m_ndof);
	for (int i = 0; i < 3; i++) {
		auto value_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto component_leaf = in_float3grid.v[i]->tree().probeConstLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				result[iter.getValue()] = component_leaf->getValue(iter.offset());
			}
		};
		m_dof_manager[i]->foreach(value_setter);
	}

	return result;
}

void L_with_level::to_Eigenvector(Eigen::VectorXf& out_eigenvector, packed_FloatGrid3 in_float3grid) const
{
	out_eigenvector.setZero(m_ndof);
	for (int i = 0; i < 3; i++) {
		auto value_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto component_leaf = in_float3grid.v[i]->tree().probeConstLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				out_eigenvector[iter.getValue()] = component_leaf->getValue(iter.offset());
			}
		};
		m_dof_manager[i]->foreach(value_setter);
	}
}

packed_FloatGrid3 L_with_level::to_packed_FloatGrid3(const Eigen::VectorXf& in_vector) const
{
	packed_FloatGrid3 result;
	write_to_FloatGrid3(result, in_vector);
	return result;
}

void L_with_level::write_to_FloatGrid3(packed_FloatGrid3 result, const Eigen::VectorXf& in_vector) const
{
	for (int i = 0; i < 3; i++) {
		result.v[i]->setTransform(m_face_center_transform[i]);
		result.v[i]->setTree(std::make_shared<openvdb::FloatTree>(m_velocity_DOF.idx[i]->tree(), 0.f, openvdb::TopologyCopy()));

		auto value_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto component_leaf = result.v[i]->tree().probeLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				component_leaf->setValueOn(iter.offset(), in_vector[iter.getValue()]);
			}
		};
		m_dof_manager[i]->foreach(value_setter);
	}
}

void L_with_level::write_to_FloatGrid3_assume_topology(packed_FloatGrid3 out_grids, const Eigen::VectorXf& in_vector) const
{
	tbb::parallel_for(0, 3, [&](int i) {
		auto value_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto component_leaf = out_grids.v[i]->tree().probeLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				component_leaf->setValueOn(iter.offset(), in_vector[iter.getValue()]);
			}
		};
		m_dof_manager[i]->foreach(value_setter);
		});
}

packed_FloatGrid3 L_with_level::get_zero_vec() const
{
	
	packed_FloatGrid3 result;
	for (int i = 0; i < 3; i++) {
		result.v[i] = openvdb::FloatGrid::create();
		result.v[i]->setTransform(m_face_center_transform[i]);
		result.v[i]->setName("vc" + std::to_string(i));
		result.v[i]->setTree(std::make_shared<openvdb::FloatTree>(m_velocity_DOF.idx[i]->tree(), 0.f, openvdb::TopologyCopy()));
	}
	return result;
}

void L_with_level::IO(std::string fname)
{
	openvdb::io::File(fname).write({ m_liquid_sdf,m_subcell_sdf,m_subcell_vol,
		m_voxel_vol,
		m_face_center_vol.v[0],m_face_center_vol.v[1],m_face_center_vol.v[2],
		m_edge_center_vol[0],m_edge_center_vol[1],m_edge_center_vol[2],
		m_velocity_DOF.idx[0],m_velocity_DOF.idx[1],m_velocity_DOF.idx[2],
		m_diagonal.v[0], m_diagonal.v[1] ,m_diagonal.v[2],
		m_invdiag.v[0], m_invdiag.v[1] ,m_invdiag.v[2],
		m_SPAI0.v[0], m_SPAI0.v[1] ,m_SPAI0.v[2],
		m_neg_x.v[0],m_neg_x.v[1],m_neg_x.v[2],
		m_neg_y.v[0],m_neg_y.v[1],m_neg_y.v[2],
		m_neg_z.v[0],m_neg_z.v[1],m_neg_z.v[2],
		m_cross_channel[0][0][0], m_cross_channel[0][0][1], m_cross_channel[0][1][0], m_cross_channel[0][1][1],
		m_cross_channel[1][0][0], m_cross_channel[1][0][1], m_cross_channel[1][1][0], m_cross_channel[1][1][1],
		m_cross_channel[2][0][0], m_cross_channel[2][0][1], m_cross_channel[2][1][0], m_cross_channel[2][1][1] });
}

L_with_level::light_weight_applier L_with_level::get_light_weight_applier(
	packed_FloatGrid3 out_result,
	packed_FloatGrid3 in_lhs,
	packed_FloatGrid3 in_rhs, L_with_level::working_mode in_working_mode)
{
	return light_weight_applier(this, out_result, in_lhs, in_rhs, in_working_mode);
}

void L_with_level::run(light_weight_applier& Op)
{
	/*for (int i = 0; i < 3; i++) {
		Op.set_channel(i);
		m_dof_manager[i]->foreach(Op);
	}*/
	tbb::parallel_for(0, 3, [&](int i) {
		light_weight_applier channel_op(Op);
		channel_op.set_channel(i);
		m_dof_manager[i]->foreach(channel_op);
		});
}

void L_with_level::L_apply(packed_FloatGrid3 out_result, packed_FloatGrid3 in_lhs)
{
	
	auto Op = get_light_weight_applier(out_result, in_lhs, out_result, working_mode::NORMAL);
	run(Op);
}

void L_with_level::residual_apply(packed_FloatGrid3 out_residual, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs)
{
	
	auto Op = get_light_weight_applier(out_residual, in_lhs, in_rhs, working_mode::RESIDUAL);
	run(Op);
}

void L_with_level::Jacobi_apply(packed_FloatGrid3 out_updated_lhs, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs)
{
	
	auto Op = get_light_weight_applier(out_updated_lhs, in_lhs, in_rhs, working_mode::JACOBI);
	run(Op);
}

void L_with_level::SPAI0_apply(packed_FloatGrid3 out_updated_lhs, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs)
{
	
	build_SPAI0_matrix();
	auto Op = get_light_weight_applier(out_updated_lhs, in_lhs, in_rhs, working_mode::SPAI0);
	run(Op);
}


void L_with_level::XYZ_RBGS_apply(packed_FloatGrid3 dummy, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs)
{
	
	auto Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::RED_GS);
	Op.set_channel(0);
	m_dof_manager[0]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::BLACK_GS);
	Op.set_working_mode(working_mode::BLACK_GS);
	Op.set_channel(0);
	m_dof_manager[0]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::RED_GS);
	Op.set_working_mode(working_mode::RED_GS);
	Op.set_channel(1);
	m_dof_manager[1]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::BLACK_GS);
	Op.set_working_mode(working_mode::BLACK_GS);
	Op.set_channel(1);
	m_dof_manager[1]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::RED_GS);
	Op.set_working_mode(working_mode::RED_GS);
	Op.set_channel(2);
	m_dof_manager[2]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::BLACK_GS);
	Op.set_working_mode(working_mode::BLACK_GS);
	Op.set_channel(2);
	m_dof_manager[2]->foreach(Op);
}

void L_with_level::ZYX_RBGS_apply(packed_FloatGrid3 dummy, packed_FloatGrid3 in_lhs, packed_FloatGrid3 in_rhs)
{
	
	auto Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::BLACK_GS);
	Op.set_channel(2);
	m_dof_manager[2]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::RED_GS);
	Op.set_working_mode(working_mode::RED_GS);
	Op.set_channel(2);
	m_dof_manager[2]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::BLACK_GS);
	Op.set_working_mode(working_mode::BLACK_GS);
	Op.set_channel(1);
	m_dof_manager[1]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::RED_GS);
	Op.set_working_mode(working_mode::RED_GS);
	Op.set_channel(1);
	m_dof_manager[1]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::BLACK_GS);
	Op.set_working_mode(working_mode::BLACK_GS);
	Op.set_channel(0);
	m_dof_manager[0]->foreach(Op);

	//Op = get_light_weight_applier(in_lhs, in_lhs, in_rhs, working_mode::RED_GS);
	Op.set_working_mode(working_mode::RED_GS);
	Op.set_channel(0);
	m_dof_manager[0]->foreach(Op);
}

void L_with_level::set_grid_to_zero(packed_FloatGrid3 in_out_result)
{
	
	for (int i = 0; i < 3; i++) {
		auto zero_filler = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto result_leaf = in_out_result.v[i]->tree().probeLeaf(leaf.origin());
			result_leaf->fill(0);
		};
		m_dof_manager[i]->foreach(zero_filler);
	}
}

void L_with_level::set_grid_to_SPAI0_after_first_iteration(packed_FloatGrid3 out_lhs, packed_FloatGrid3 in_rhs)
{
	build_SPAI0_matrix();
	__m256 packed_w_jacobi = _mm256_set1_ps(m_damped_jacobi_coef);
	for (int i = 0; i < 3; i++) {
		auto SPAI0_result_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			const auto* rhs_leaf = in_rhs.v[i]->tree().probeConstLeaf(leaf.origin());
			const float* rhs_data = rhs_leaf->buffer().data();
			auto* lhs_leaf = out_lhs.v[i]->tree().probeLeaf(leaf.origin());
			float* lhs_data = lhs_leaf->buffer().data();
			const auto* SPAI0_leaf = m_SPAI0.v[i]->tree().probeConstLeaf(leaf.origin());
			const float* SPAI0_data = nullptr;
			if (SPAI0_leaf) {
				SPAI0_data = SPAI0_leaf->buffer().data();
			}
			for (uint32_t vector_offset = 0; vector_offset < 512; vector_offset += 8) {
				const uint8_t vectormask = leaf.getValueMask().getWord<uint8_t>(vector_offset / 8);
				if (vectormask == uint8_t(0)) {
					//there is no dof in this vector
					continue;
				}
				__m256 rhs_vector;
				__m256 SPAI0_vector;
				/****************************************************************************/
				vdb_SIMD_IO::get_simd_vector_unsafe(rhs_vector, rhs_data, vector_offset);
				vdb_SIMD_IO::get_simd_vector_unsafe(SPAI0_vector, SPAI0_data, vector_offset);
				//residual multiplied by SPAI0 diagonal
				rhs_vector = _mm256_mul_ps(rhs_vector, SPAI0_vector);
				//weighted residual update
				rhs_vector = _mm256_mul_ps(rhs_vector, packed_w_jacobi);
				_mm256_storeu_ps(lhs_data + vector_offset, rhs_vector);
			}
		};
		m_dof_manager[i]->foreach(SPAI0_result_setter);
	}
}

void L_with_level::restriction(packed_FloatGrid3 out_coarse_grid, packed_FloatGrid3 in_fine_grid, const L_with_level& parent)
{
	
	for (int i = 0; i < 3; i++) {
		//to be use by the dof idx manager at coarse level, the parent level
		auto collect_from_fine = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto* coarse_leaf = out_coarse_grid.v[i]->tree().probeLeaf(leaf.origin());
			//fine voxel:   -4 -3 -2 -1 0 1 2 3 4 5
			//coarse voxel: -2 -2 -1 -1 0 0 1 1 2 2

			//each coarse leaf corresponds to 8 potential fine leaves that are active
			std::array<const openvdb::FloatTree::LeafNodeType*, 8> fine_leaves{ nullptr };
			int fine_leaves_counter = 0;
			auto fine_base_origin = openvdb::Coord(leaf.origin().asVec3i() * 2);
			for (int ii = 0; ii < 16; ii += 8) {
				for (int jj = 0; jj < 16; jj += 8) {
					for (int kk = 0; kk < 16; kk += 8) {
						fine_leaves[fine_leaves_counter++] =
							in_fine_grid.v[i]->tree().probeConstLeaf(fine_base_origin.offsetBy(ii, jj, kk));
					}
				}
			}

			for (auto iter = coarse_leaf->beginValueOn(); iter; ++iter) {
				//uint32_t at_fine_leaf = iter.offset();

				auto itercoord = coarse_leaf->offsetToLocalCoord(iter.offset());
				uint32_t at_fine_leaf = 0;
				if (itercoord[2] >= 4) {
					at_fine_leaf += 1;
				}
				if (itercoord[1] >= 4) {
					at_fine_leaf += 2;
				}
				if (itercoord[0] >= 4) {
					at_fine_leaf += 4;
				}

				//if there is possibly a dof in the fine leaf
				if (auto fine_leaf = fine_leaves[at_fine_leaf]) {
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
					iter.setValue(temp_sum * 0.125f * 0.5f * 2.0f);
				}//if fine leaf
			}//for all coarse on voxels
		};//end collect from fine

		parent.m_dof_manager[i]->foreach(collect_from_fine);
	}
}
namespace {
struct explicit_matrix_reducer {
	using tt = Eigen::Triplet<float>;
	using leaf_type = openvdb::FloatTree::LeafNodeType;
	using leaves_vec_type = std::vector<leaf_type*>;
	using leaves_vec_ptr_type = leaves_vec_type*;

	explicit_matrix_reducer(
		const std::vector<const openvdb::Int32Tree::LeafNodeType*>& in_dof_leaves,
		L_with_level::DOF_tuple_t in_dof,
		openvdb::FloatGrid::Ptr in_diag,
		openvdb::FloatGrid::Ptr in_negx,
		openvdb::FloatGrid::Ptr in_negy,
		openvdb::FloatGrid::Ptr in_negz,
		openvdb::FloatGrid::Ptr in_cross_term[3][2][2],
		int in_channel
	) : m_dof{ in_dof.idx[0], in_dof.idx[1], in_dof.idx[2] },
		m_diag(in_diag), m_negx(in_negx), m_negy(in_negy), m_negz(in_negz),
		m_cross_term{
			{{in_cross_term[0][0][0],in_cross_term[0][0][1]},
			{in_cross_term[0][1][0],in_cross_term[0][1][1]}},
			{{in_cross_term[1][0][0],in_cross_term[1][0][1]},
			{in_cross_term[1][1][0],in_cross_term[1][1][1]}},
			{{in_cross_term[2][0][0],in_cross_term[2][0][1]},
			{in_cross_term[2][1][0],in_cross_term[2][1][1]}} },
			m_dof_leaves(in_dof_leaves),
			m_channel(in_channel){}

	explicit_matrix_reducer(const explicit_matrix_reducer& other, tbb::split) :
		m_dof{ other.m_dof[0], other.m_dof[1], other.m_dof[2] },
		m_diag(other.m_diag),
		m_negx(other.m_negx),
		m_negy(other.m_negy),
		m_negz(other.m_negz),
		m_cross_term{
			{{other.m_cross_term[0][0][0], other.m_cross_term[0][0][1]},
			{other.m_cross_term[0][1][0], other.m_cross_term[0][1][1]}},
			{{other.m_cross_term[1][0][0], other.m_cross_term[1][0][1]},
			{other.m_cross_term[1][1][0], other.m_cross_term[1][1][1]}},
			{{other.m_cross_term[2][0][0], other.m_cross_term[2][0][1]},
			{other.m_cross_term[2][1][0], other.m_cross_term[2][1][1]}} },
			m_dof_leaves(other.m_dof_leaves),
			m_channel(other.m_channel){}


	//to be called through the leaf range of each channel
	void operator()(const tbb::blocked_range<size_t>& r) {
		const int d0 = m_channel;
		const int d1 = (d0 + 1) % 3;
		const int d2 = (d0 + 2) % 3;

		openvdb::Coord dir[3] = { openvdb::Coord{0},openvdb::Coord{0},openvdb::Coord{0} };
		dir[0][d0]++; dir[1][d1]++; dir[2][d2]++;

		openvdb::FloatGrid::ConstUnsafeAccessor negxyz_axr[3] = {
		m_negx->tree(), m_negy->tree(), m_negz->tree() };

		openvdb::Int32Grid::ConstUnsafeAccessor m_dof_axr[3] = { m_dof[0]->tree(),m_dof[1]->tree(),m_dof[2]->tree() };
		openvdb::FloatGrid::ConstUnsafeAccessor diag_axr{ m_diag->tree() };
		openvdb::FloatGrid::ConstUnsafeAccessor m_cross_axr[3][2][2] = {
			{{m_cross_term[0][0][0]->tree(),m_cross_term[0][0][1]->tree()},
			{m_cross_term[0][1][0]->tree(),m_cross_term[0][1][1]->tree()}},
			{{m_cross_term[1][0][0]->tree(),m_cross_term[1][0][1]->tree()},
			{m_cross_term[1][1][0]->tree(),m_cross_term[1][1][1]->tree()}},
			{{m_cross_term[2][0][0]->tree(),m_cross_term[2][0][1]->tree()},
			{m_cross_term[2][1][0]->tree(),m_cross_term[2][1][1]->tree()}} };

		for (size_t ileaf = r.begin(); ileaf != r.end(); ++ileaf) {
			for (auto iter = m_dof_leaves[ileaf]->beginValueOn(); iter; ++iter) {
				//same channel related
				auto current_dof = iter.getValue();
				auto diag_val = diag_axr.getValue(iter.getCoord());
				//diagonal
				m_triplets.push_back({ current_dof, current_dof, diag_val });

				for (int i_face = 0; i_face < 6; i_face++) {
					openvdb::Coord neib = iter.getCoord();
					if (i_face % 2 == 0) {
						neib[i_face / 2]++;
					}
					else {
						neib[i_face / 2]--;
					}

					const auto neib_dof = m_dof_axr[d0].getValue(neib);
					if (-1 != neib_dof) {
						//get coefficients
						float term = negxyz_axr[i_face / 2].getValue(iter.getCoord());
						if (i_face % 2 == 0) {
							term = negxyz_axr[i_face / 2].getValue(neib);
						}
						m_triplets.push_back({ current_dof, neib_dof, term });
					}
				}//end for 6 faces

				//Cross terms, see the light weight applier for details
				//Td2
				//Td200
				{
					const auto dof200 = m_dof_axr[d1].getValue(iter.getCoord());
					if (-1 != dof200) {
						float td200 = m_cross_axr[d2][0][0].getValue(iter.getCoord());
						m_triplets.push_back({ current_dof, dof200, td200 });
					}
				}

				//Td201
				{
					const auto dof201 = m_dof_axr[d1].getValue(iter.getCoord() + dir[1]);
					if (-1 != dof201) {
						float td201 = m_cross_axr[d2][0][1].getValue(iter.getCoord());
						m_triplets.push_back({ current_dof, dof201,td201 });
					}
				}

				//Td210
				{
					const auto dof210 = m_dof_axr[d1].getValue(iter.getCoord() - dir[0]);
					if (-1 != dof210) {
						float td210 = m_cross_axr[d2][1][0].getValue(iter.getCoord() - dir[0]);
						m_triplets.push_back({ current_dof, dof210, td210 });
					}
				}

				//Td211
				{
					const auto dof211 = m_dof_axr[d1].getValue(iter.getCoord() - dir[0] + dir[1]);
					if (-1 != dof211) {
						float td211 = m_cross_axr[d2][1][1].getValue(iter.getCoord() - dir[0]);
						m_triplets.push_back({ current_dof, dof211, td211 });
					}
				}


				//Td1
				//Td100
				{
					const auto dof100 = m_dof_axr[d2].getValue(iter.getCoord());
					if (-1 != dof100) {
						float td100 = m_cross_axr[d1][0][0].getValue(iter.getCoord());
						m_triplets.push_back({ current_dof, dof100, td100 });
					}
				}

				//Td101
				{
					const auto dof101 = m_dof_axr[d2].getValue(iter.getCoord() - dir[0]);
					if (-1 != dof101) {
						float td101 = m_cross_axr[d1][0][1].getValue(iter.getCoord() - dir[0]);
						m_triplets.push_back({ current_dof, dof101, td101 });
					}
				}

				//Td110
				{
					const auto dof110 = m_dof_axr[d2].getValue(iter.getCoord() + dir[2]);
					if (-1 != dof110) {
						float td110 = m_cross_axr[d1][1][0].getValue(iter.getCoord());
						m_triplets.push_back({ current_dof, dof110, td110 });
					}
				}

				//Td111
				{
					const auto dof111 = m_dof_axr[d2].getValue(iter.getCoord() + dir[2] - dir[0]);
					if (-1 != dof111) {
						float td111 = m_cross_axr[d1][1][1].getValue(iter.getCoord() - dir[0]);
						m_triplets.push_back({ current_dof, dof111, td111 });
					}
				}
			}//end for all dof in this channel leaf
		}//end for every diagonal leaf

	}//end operator


	void join(const explicit_matrix_reducer& other) {
		size_t original_size = m_triplets.size();
		size_t extra_size = other.m_triplets.size();
		m_triplets.resize(original_size + extra_size);
		std::copy(other.m_triplets.begin(), other.m_triplets.end(), m_triplets.begin() + original_size);
	}

	int m_channel;

	openvdb::Int32Grid::Ptr m_dof[3];
	openvdb::FloatGrid::Ptr m_diag, m_negx, m_negy, m_negz;
	openvdb::FloatGrid::Ptr m_cross_term[3][2][2];

	const std::vector<const openvdb::Int32Tree::LeafNodeType*>& m_dof_leaves;
	std::vector<tt> m_triplets;
};//end explicit_matrix_reducer
}//end namespace

void L_with_level::build_explicit_matrix()
{
	std::vector<Eigen::Triplet<float>> triplets;

	get_triplets(triplets);

	m_explicit_matrix.resize(m_ndof, m_ndof);
	m_explicit_matrix.setFromTriplets(triplets.begin(), triplets.end());
}

void L_with_level::get_triplets(std::vector<Eigen::Triplet<float>>& triplets)
{
	for (int i = 0; i < 3; i++) {
		std::vector<const openvdb::Int32Tree::LeafNodeType*> dof_leaves;
		dof_leaves.reserve(m_dof_manager[i]->leafCount());
		m_dof_manager[i]->getNodes(dof_leaves);

		explicit_matrix_reducer reducer(
			dof_leaves,
			m_velocity_DOF,
			m_diagonal.v[i],
			m_neg_x.v[i],
			m_neg_y.v[i],
			m_neg_z.v[i],
			m_cross_channel,
			i);

		tbb::parallel_reduce(tbb::blocked_range<size_t>(0, dof_leaves.size()), reducer);

		auto original_size = triplets.size();
		auto extra_size = reducer.m_triplets.size();
		triplets.resize(original_size + extra_size);
		std::copy(reducer.m_triplets.begin(), reducer.m_triplets.end(), triplets.begin() + original_size);
	}
}

//end namespace
void L_with_level::construct_exact_solver()
{
	build_explicit_matrix();
	m_direct_solver = std::make_unique<coarse_solver_type>(m_explicit_matrix);
	m_direct_solver->setMaxIterations(10);
}

void L_with_level::solve_exact(packed_FloatGrid3 in_out_lhs, packed_FloatGrid3 in_rhs)
{
	to_Eigenvector(m_direct_solver_rhs, in_rhs);
	m_direct_solver_lhs = m_direct_solver->solve(m_direct_solver_rhs);
	write_to_FloatGrid3_assume_topology(in_out_lhs, m_direct_solver_lhs);
}

template<bool inplace_add>
void L_with_level::prolongation(packed_FloatGrid3 out_fine_grid, packed_FloatGrid3 in_coarse_grid)
{
	
	for (int i = 0; i < 3; i++) {
		//to be use by the dof idx manager at coarse level, the parent level
		auto scatter_to_fine = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			const auto* coarse_leaf = in_coarse_grid.v[i]->tree().probeConstLeaf(leaf.origin());
			//fine voxel:   -4 -3 -2 -1 0 1 2 3 4 5
			//coarse voxel: -2 -2 -1 -1 0 0 1 1 2 2

			//each coarse leaf corresponds to 8 potential fine leaves that are active
			std::array<openvdb::FloatTree::LeafNodeType*, 8> fine_leaves{ nullptr };
			int fine_leaves_counter = 0;
			auto fine_base_origin = openvdb::Coord(leaf.origin().asVec3i() * 2);
			for (int ii = 0; ii < 16; ii += 8) {
				for (int jj = 0; jj < 16; jj += 8) {
					for (int kk = 0; kk < 16; kk += 8) {
						fine_leaves[fine_leaves_counter++] =
							out_fine_grid.v[i]->tree().probeLeaf(fine_base_origin.offsetBy(ii, jj, kk));
					}
				}
			}

			for (auto iter = coarse_leaf->cbeginValueOn(); iter; ++iter) {
				//uint32_t at_fine_leaf = iter.offset();

				auto itercoord = coarse_leaf->offsetToLocalCoord(iter.offset());
				uint32_t at_fine_leaf = 0;
				if (itercoord[2] >= 4) {
					at_fine_leaf += 1;
				}
				if (itercoord[1] >= 4) {
					at_fine_leaf += 2;
				}
				if (itercoord[0] >= 4) {
					at_fine_leaf += 4;
				}

				float coarseval = iter.getValue();
				//if there is possibly a dof in the fine leaf
				if (auto fine_leaf = fine_leaves[at_fine_leaf]) {
					auto fine_base_voxel = openvdb::Coord(iter.getCoord().asVec3i() * 2);
					auto fine_base_offset = fine_leaf->coordToOffset(fine_base_voxel);
					for (int ii = 0; ii < 2; ii++) {
						for (int jj = 0; jj < 2; jj++) {
							for (int kk = 0; kk < 2; kk++) {
								auto fine_offset = fine_base_offset + kk;
								if (ii) fine_offset += 64;
								if (jj) fine_offset += 8;
								if (fine_leaf->isValueOn(fine_offset)) {
									if (inplace_add) {
										fine_leaf->buffer().data()[fine_offset] += coarseval;
									}
									else {
										fine_leaf->setValueOnly(fine_offset, coarseval);
									}

								}
							}//kk
						}//jj
					}//ii
					//iter.setValue(temp_sum * 0.125f);
				}//if fine leaf
			}//for all coarse on voxels
		};//end collect from fine


		m_dof_manager[i]->foreach(scatter_to_fine);
	}
}

L_with_level::light_weight_applier::light_weight_applier(
	L_with_level* parent,
	packed_FloatGrid3 out_result,
	packed_FloatGrid3 in_lhs,
	packed_FloatGrid3 in_rhs, L_with_level::working_mode in_working_mode) :m_parent(parent),
	m_diag_axr{ parent->m_diagonal.v[0]->tree(),
	parent->m_diagonal.v[1]->tree(),
	parent->m_diagonal.v[2]->tree(), },
	m_inv_diag_axr{ parent->m_invdiag.v[0]->tree(),
	parent->m_invdiag.v[1]->tree(),
	parent->m_invdiag.v[2]->tree(), },
	m_spai0_axr{ parent->m_SPAI0.v[0]->tree(),
	parent->m_SPAI0.v[1]->tree(),
	parent->m_SPAI0.v[2]->tree() },
	m_negx_axr{ parent->m_neg_x.v[0]->tree(),
	parent->m_neg_x.v[1]->tree(),
	parent->m_neg_x.v[2]->tree() },
	m_negy_axr{ parent->m_neg_y.v[0]->tree(),
	parent->m_neg_y.v[1]->tree(),
	parent->m_neg_y.v[2]->tree() },
	m_negz_axr{ parent->m_neg_z.v[0]->tree(),
	parent->m_neg_z.v[1]->tree(),
	parent->m_neg_z.v[2]->tree() },
	Txyzd12_axr{
	//0
		{{parent->m_cross_channel[0][0][0]->tree(),
	parent->m_cross_channel[0][0][1]->tree()},
		{parent->m_cross_channel[0][1][0]->tree(),
	parent->m_cross_channel[0][1][1]->tree()}},
	//1
		{{parent->m_cross_channel[1][0][0]->tree(),
	parent->m_cross_channel[1][0][1]->tree()},
		{parent->m_cross_channel[1][1][0]->tree(),
	parent->m_cross_channel[1][1][1]->tree()}},
	//2
		{{parent->m_cross_channel[2][0][0]->tree(),
	parent->m_cross_channel[2][0][1]->tree()},
		{parent->m_cross_channel[2][1][0]->tree(),
	parent->m_cross_channel[2][1][1]->tree()}} },
	m_lhs_axr{ in_lhs.v[0]->tree(),
	in_lhs.v[1]->tree(),
	in_lhs.v[2]->tree() },
	m_working_mode(in_working_mode)
{
	for (int i = 0; i < 3; i++) {
		m_rhs_leaves[i] = std::make_shared<const_leaf_vec_type>();
		m_lhs_leaves[i] = std::make_shared<const_leaf_vec_type>();
		m_result_leaves[i] = std::make_shared<leaf_vec_type>();
		size_t nleaf = parent->m_dof_manager[i]->leafCount();

		if (m_working_mode == working_mode::JACOBI ||
			m_working_mode == working_mode::SPAI0 ||
			m_working_mode == working_mode::RED_GS ||
			m_working_mode == working_mode::BLACK_GS ||
			m_working_mode == working_mode::RESIDUAL) {
			m_rhs_leaves[i]->reserve(nleaf);
			in_rhs.v[i]->tree().getNodes(*(m_rhs_leaves[i]));
		}

		m_lhs_leaves[i]->reserve(nleaf);
		in_lhs.v[i]->tree().getNodes(*(m_lhs_leaves[i]));

		m_result_leaves[i]->reserve(nleaf);
		out_result.v[i]->tree().getNodes(*(m_result_leaves[i]));
	}

	//processing result to c channel.
	channel = 0;
}

L_with_level::light_weight_applier::light_weight_applier(const light_weight_applier& other)
	:m_parent(other.m_parent),
	m_diag_axr{ other.m_parent->m_diagonal.v[0]->tree(),
	other.m_parent->m_diagonal.v[1]->tree(),
	other.m_parent->m_diagonal.v[2]->tree(), },
	m_inv_diag_axr{ other.m_parent->m_invdiag.v[0]->tree(),
	other.m_parent->m_invdiag.v[1]->tree(),
	other.m_parent->m_invdiag.v[2]->tree(), },
	m_spai0_axr{ other.m_parent->m_SPAI0.v[0]->tree(),
	other.m_parent->m_SPAI0.v[1]->tree(),
	other.m_parent->m_SPAI0.v[2]->tree() },
	m_negx_axr{ other.m_parent->m_neg_x.v[0]->tree(),
	other.m_parent->m_neg_x.v[1]->tree(),
	other.m_parent->m_neg_x.v[2]->tree() },
	m_negy_axr{ other.m_parent->m_neg_y.v[0]->tree(),
	other.m_parent->m_neg_y.v[1]->tree(),
	other.m_parent->m_neg_y.v[2]->tree() },
	m_negz_axr{ other.m_parent->m_neg_z.v[0]->tree(),
	other.m_parent->m_neg_z.v[1]->tree(),
	other.m_parent->m_neg_z.v[2]->tree() },
	Txyzd12_axr{
	//0
		{{other.m_parent->m_cross_channel[0][0][0]->tree(),
	other.m_parent->m_cross_channel[0][0][1]->tree()},
		{other.m_parent->m_cross_channel[0][1][0]->tree(),
	other.m_parent->m_cross_channel[0][1][1]->tree()}},
	//1
		{{other.m_parent->m_cross_channel[1][0][0]->tree(),
	other.m_parent->m_cross_channel[1][0][1]->tree()},
		{other.m_parent->m_cross_channel[1][1][0]->tree(),
	other.m_parent->m_cross_channel[1][1][1]->tree()}},
	//2
		{{other.m_parent->m_cross_channel[2][0][0]->tree(),
	other.m_parent->m_cross_channel[2][0][1]->tree()},
		{other.m_parent->m_cross_channel[2][1][0]->tree(),
	other.m_parent->m_cross_channel[2][1][1]->tree()}} },
	m_lhs_axr{ other.m_lhs_axr[0], other.m_lhs_axr[1] ,other.m_lhs_axr[2] },
	m_working_mode(other.m_working_mode),
	m_rhs_leaves{ other.m_rhs_leaves[0], other.m_rhs_leaves[1], other.m_rhs_leaves[2] },
	m_lhs_leaves{ other.m_lhs_leaves[0], other.m_lhs_leaves[1], other.m_lhs_leaves[2] },
	m_result_leaves{ other.m_result_leaves[0], other.m_result_leaves[1], other.m_result_leaves[2] },
	channel(other.channel)
{
	for (int i = 0; i < 3; i++) {
		m_lhs_axr[i].clear();
	}
}

template <int c>
void L_with_level::light_weight_applier::component_operator(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const
{
	const float* rhs_data = nullptr;
	const float* spai0_data = nullptr;
	if (m_working_mode == working_mode::JACOBI ||
		m_working_mode == working_mode::SPAI0 ||
		m_working_mode == working_mode::RED_GS ||
		m_working_mode == working_mode::BLACK_GS ||
		m_working_mode == working_mode::RESIDUAL) {
		rhs_data = leaf2data((*m_rhs_leaves[c])[leafpos]);
	}
	if (m_working_mode == working_mode::SPAI0) {
		spai0_data = leaf2data(m_spai0_axr[c].probeConstLeaf(leaf.origin()));
	}
	float* result_data = leaf2data((*m_result_leaves[c])[leafpos]);
	const float* diagdata = leaf2data(m_diag_axr[c].probeConstLeaf(leaf.origin()));
	const float* invdiagdata = leaf2data(m_inv_diag_axr[c].probeConstLeaf(leaf.origin()));

	//this leaf, xp xm, yp ym zp zm.
	std::array<const float*, 7> lhs_data;
	lhs_data.fill(nullptr); lhs_data[0] = leaf2data((*m_lhs_leaves[c])[leafpos]);
	for (int i_f = 0; i_f < 6; i_f++) {
		int component = i_f / 2;
		int positive_direction = (i_f % 2 == 0);
		auto neib_origin = leaf.origin();
		if (positive_direction) {
			neib_origin[component] += 8;
		}
		else {
			neib_origin[component] -= 8;
		}
		lhs_data[i_f + 1] = leaf2data(m_lhs_axr[c].probeConstLeaf(neib_origin));
	}

	//0 self, 1 xp
	const float* x_weight_data[2] = { nullptr,nullptr };
	x_weight_data[0] = leaf2data(m_negx_axr[c].probeConstLeaf(leaf.origin()));
	x_weight_data[1] = leaf2data(m_negx_axr[c].probeConstLeaf(leaf.origin().offsetBy(8, 0, 0)));

	//0 self, 1 yp
	const float* y_weight_data[2] = { nullptr,nullptr };
	y_weight_data[0] = leaf2data(m_negy_axr[c].probeConstLeaf(leaf.origin()));
	y_weight_data[1] = leaf2data(m_negy_axr[c].probeConstLeaf(leaf.origin().offsetBy(0, 8, 0)));

	//0 self, 1 zp
	const float* z_weight_data[2] = { nullptr,nullptr };
	z_weight_data[0] = leaf2data(m_negz_axr[c].probeConstLeaf(leaf.origin()));
	z_weight_data[1] = leaf2data(m_negz_axr[c].probeConstLeaf(leaf.origin().offsetBy(0, 0, 8)));

	//loop over 64 vectors to conduct the computation
	// vectorid = x*8+y
	//faced result stores the same channel laplacian
	//this, xp, xm, yp, ym, zp, zm
	std::array<__m256, 7> faced_result;


	/*
	* Here x is the dir0 direction, y is dir1, z is dir2.
	Assume we are processing the X component
	The associated cross terms are Txy and Tzx
	which are the cross terms when looking from positive Z direction
	and positive Y direction.
	So they are refereed to as Tdirection2, and Tdirection1 cross terms

					  -x +y      0x +y
	^ y          -----^------   -----^------
	|           |     uy T11  | T01  uy     |
	|           |             |             |
	|           |             >ux           |
	|           |             |             |
	|           |     uy T10  | T00  uy     |
	|            -----^------   -----^------
	(z)--------->x    -x 0y       0x 0y


	^x            ----------------
	|             |              |
	|             |              |
	|   0x 0z  uz >              > uz 0x +z
	|             |              |
	|             | T00  ux T10  |
	|             |------ ^------|
	|             | T01     T11  |
	|             |              |
	|   -x 0z  uz >              > uz -x +z
	|             |              |
	|             |              |
	|             |--------------|
	(y)--------->z
	*/

	//xyz yzx zxy
	std::array<__m256, 4> Td2_cross_result;
	std::array<__m256, 4> Td1_cross_result;

	//cross term coefficients leaves
	//the first elements corresponds to the negative dir0 direction leaf
	//the second elements corresponds to this leaf.
	const float* Td2_00, * Td2_01, * Td2_10[2], * Td2_11[2];
	const float* Td1_00, * Td1_01[2], * Td1_10, * Td1_11[2];

	//components, and positive directions
	const int d0 = c;
	const int d1 = (c + 1) % 3;
	const int d2 = (c + 2) % 3;
	openvdb::Coord dir[3] = { openvdb::Coord{0}, openvdb::Coord{0}, openvdb::Coord{0} };
	dir[0][d0]++; dir[1][d1]++; dir[2][d2]++;

	//leaves of coefficients
	openvdb::Coord dir0_leafshift{ 8 * dir[0].asVec3i() };

	//dir2 
	Td2_00 = leaf2data(Txyzd12_axr[d2][0][0].probeConstLeaf(leaf.origin()));
	Td2_01 = leaf2data(Txyzd12_axr[d2][0][1].probeConstLeaf(leaf.origin()));
	Td2_10[0] = leaf2data(Txyzd12_axr[d2][1][0].probeConstLeaf(leaf.origin() - dir0_leafshift));
	Td2_10[1] = leaf2data(Txyzd12_axr[d2][1][0].probeConstLeaf(leaf.origin()));
	Td2_11[0] = leaf2data(Txyzd12_axr[d2][1][1].probeConstLeaf(leaf.origin() - dir0_leafshift));
	Td2_11[1] = leaf2data(Txyzd12_axr[d2][1][1].probeConstLeaf(leaf.origin()));
	//dir1 
	Td1_00 = leaf2data(Txyzd12_axr[d1][0][0].probeConstLeaf(leaf.origin()));
	Td1_01[0] = leaf2data(Txyzd12_axr[d1][0][1].probeConstLeaf(leaf.origin() - dir0_leafshift));
	Td1_01[1] = leaf2data(Txyzd12_axr[d1][0][1].probeConstLeaf(leaf.origin()));
	Td1_10 = leaf2data(Txyzd12_axr[d1][1][0].probeConstLeaf(leaf.origin()));
	Td1_11[0] = leaf2data(Txyzd12_axr[d1][1][1].probeConstLeaf(leaf.origin() - dir0_leafshift));
	Td1_11[1] = leaf2data(Txyzd12_axr[d1][1][1].probeConstLeaf(leaf.origin()));

	//cross term velocity leaves
	//dir1:[negdir0, samedir0][samedir1 posdir1]
	//dir2:[samedir2, posdir2][negdir0, samedir0] 
	const float* dir1_lhs_data[2][2], * dir2_lhs_data[2][2];

	openvdb::Coord dir1_leafshift{ 8 * dir[1].asVec3i() };
	openvdb::Coord dir2_leafshift{ 8 * dir[2].asVec3i() };

	dir1_lhs_data[0][0] = leaf2data(m_lhs_axr[d1].probeConstLeaf(leaf.origin() - dir0_leafshift));
	dir1_lhs_data[0][1] = leaf2data(m_lhs_axr[d1].probeConstLeaf(leaf.origin() - dir0_leafshift + dir1_leafshift));
	dir1_lhs_data[1][0] = leaf2data(m_lhs_axr[d1].probeConstLeaf(leaf.origin()));
	dir1_lhs_data[1][1] = leaf2data(m_lhs_axr[d1].probeConstLeaf(leaf.origin() + dir1_leafshift));

	dir2_lhs_data[0][0] = leaf2data(m_lhs_axr[d2].probeConstLeaf(leaf.origin() - dir0_leafshift));
	dir2_lhs_data[0][1] = leaf2data(m_lhs_axr[d2].probeConstLeaf(leaf.origin()));
	dir2_lhs_data[1][0] = leaf2data(m_lhs_axr[d2].probeConstLeaf(leaf.origin() + dir2_leafshift - dir0_leafshift));
	dir2_lhs_data[1][1] = leaf2data(m_lhs_axr[d2].probeConstLeaf(leaf.origin() + dir2_leafshift));


	__m256 packed_w_jacobi = _mm256_set1_ps(m_parent->m_damped_jacobi_coef);
	__m256 packed_1minus_w_jacobi = _mm256_set1_ps(1.0f - m_parent->m_damped_jacobi_coef);

	__m256 default_diag = _mm256_set1_ps(m_parent->m_default_diag);
	__m256 default_invdiag = _mm256_set1_ps(1.0f / m_parent->m_default_diag);
	__m256 default_off_diag = _mm256_set1_ps(m_parent->m_default_off_diag);
	__m256 default_neg_off_diag = _mm256_set1_ps(-m_parent->m_default_off_diag);
	__m256 default_2neg_off_diag = _mm256_set1_ps(-2.0f * m_parent->m_default_off_diag);

	for (uint32_t vector_offset = 0; vector_offset < 512; vector_offset += 8) {
		const uint8_t vectormask = leaf.getValueMask().getWord<uint8_t>(vector_offset / 8);
		bool vector_even = ((vector_offset >> 3) + (vector_offset >> 6)) % 2 == 0;
		if (vectormask == uint8_t(0)) {
			//there is no dof in this vector
			continue;
		}

		/* x direction*/
		{
			__m256 xp_lhs, xp_weight, xm_lhs, xm_weight;
			//vector in the x plus directoin
			/****************************************************************************/
			if (vector_offset >= (7 * 64)) {
				//load the nextleaf
				uint32_t tempoffset = vector_offset - (7 * 64);

				vdb_SIMD_IO::get_simd_vector(xp_lhs, lhs_data[1], tempoffset, _mm256_setzero_ps());
				if (0 == c) {
					vdb_SIMD_IO::get_simd_vector(xp_weight, x_weight_data[1], tempoffset, default_2neg_off_diag);
				}
				else {
					vdb_SIMD_IO::get_simd_vector(xp_weight, x_weight_data[1], tempoffset, default_neg_off_diag);
				}
				

				//x-
				vdb_SIMD_IO::get_simd_vector_unsafe(xm_lhs, lhs_data[0], vector_offset - 64);
			}
			else {
				uint32_t tempoffset = vector_offset + 64;
				vdb_SIMD_IO::get_simd_vector_unsafe(xp_lhs, lhs_data[0], tempoffset);

				if (0 == c) {
					vdb_SIMD_IO::get_simd_vector(xp_weight, x_weight_data[0], tempoffset, default_2neg_off_diag);
				}
				else {
					vdb_SIMD_IO::get_simd_vector(xp_weight, x_weight_data[0], tempoffset, default_neg_off_diag);
				}
				
				if (vector_offset < 64) {
					vdb_SIMD_IO::get_simd_vector(xm_lhs, lhs_data[2], vector_offset + (7 * 64), _mm256_setzero_ps());
				}
				else {
					vdb_SIMD_IO::get_simd_vector_unsafe(xm_lhs, lhs_data[0], vector_offset - 64);
				}
			}
			faced_result[1] = _mm256_mul_ps(xp_lhs, xp_weight);
			//vector in the x minux direction
			/******************************************************************************/
			if (0 == c) {
				vdb_SIMD_IO::get_simd_vector(xm_weight, x_weight_data[0], vector_offset, default_2neg_off_diag);
			}
			else {
				vdb_SIMD_IO::get_simd_vector(xm_weight, x_weight_data[0], vector_offset, default_neg_off_diag);
			}
			
			faced_result[2] = _mm256_mul_ps(xm_lhs, xm_weight);
		}

		/*y direction*/
		{
			__m256 yp_lhs, yp_weight, ym_lhs, ym_weight;
			//vector in the y plus direction
			/****************************************************************************/
			//y==7 go to the next leaf and 
			if ((vector_offset & 63u) == 56u) {
				uint32_t tempoffset = vector_offset - 56;
				vdb_SIMD_IO::get_simd_vector(yp_lhs, lhs_data[3], tempoffset, _mm256_setzero_ps());
				if (1 == c) {
					vdb_SIMD_IO::get_simd_vector(yp_weight, y_weight_data[1], tempoffset, default_2neg_off_diag);
				}
				else {
					vdb_SIMD_IO::get_simd_vector(yp_weight, y_weight_data[1], tempoffset, default_neg_off_diag);
				}
				
				//y-
				vdb_SIMD_IO::get_simd_vector_unsafe(ym_lhs, lhs_data[0], vector_offset - 8);
			}
			else {
				uint32_t tempoffset = vector_offset + 8;
				vdb_SIMD_IO::get_simd_vector_unsafe(yp_lhs, lhs_data[0], tempoffset);
				if (1 == c) {
					vdb_SIMD_IO::get_simd_vector(yp_weight, y_weight_data[0], tempoffset, default_2neg_off_diag);
				}
				else {
					vdb_SIMD_IO::get_simd_vector(yp_weight, y_weight_data[0], tempoffset, default_neg_off_diag);
				}
				
				if ((vector_offset & 63) == 0) {
					vdb_SIMD_IO::get_simd_vector(ym_lhs, lhs_data[4], vector_offset + 56, _mm256_setzero_ps());
				}
				else {
					vdb_SIMD_IO::get_simd_vector_unsafe(ym_lhs, lhs_data[0], vector_offset - 8);
				}
			}
			faced_result[3] = _mm256_mul_ps(yp_lhs, yp_weight);
			//vector in the y minus direction
			/****************************************************************************/
			if (1 == c) {
				vdb_SIMD_IO::get_simd_vector(ym_weight, y_weight_data[0], vector_offset, default_2neg_off_diag);
			}
			else {
				vdb_SIMD_IO::get_simd_vector(ym_weight, y_weight_data[0], vector_offset, default_neg_off_diag);
			}
			
			faced_result[4] = _mm256_mul_ps(ym_lhs, ym_weight);
		}

		/*z direction*/
		{
			__m256 zp_lhs, zp_weight, zm_lhs, zm_weight;
			//vector in the z plus direction
			/****************************************************************************/
			vdb_SIMD_IO::get_pos_z_simd_vector(zp_lhs, lhs_data[0], lhs_data[5], vector_offset, _mm256_setzero_ps());
			if (2 == c) {
				vdb_SIMD_IO::get_pos_z_simd_vector(zp_weight, z_weight_data[0], z_weight_data[1], vector_offset, default_2neg_off_diag);
			}
			else {
				vdb_SIMD_IO::get_pos_z_simd_vector(zp_weight, z_weight_data[0], z_weight_data[1], vector_offset, default_neg_off_diag);
			}
			
			faced_result[5] = _mm256_mul_ps(zp_lhs, zp_weight);
			//vector in the z minus direction
			/****************************************************************************/
			vdb_SIMD_IO::get_neg_z_simd_vector(zm_lhs, lhs_data[0], lhs_data[6], vector_offset, _mm256_setzero_ps());
			if (2 == c) {
				vdb_SIMD_IO::get_simd_vector(zm_weight, z_weight_data[0], vector_offset, default_2neg_off_diag);
			}
			else {
				vdb_SIMD_IO::get_simd_vector(zm_weight, z_weight_data[0], vector_offset, default_neg_off_diag);
			}
			
			//temp_result = _mm256_fmadd_ps(zm_lhs, zm_weight, temp_result);
			faced_result[6] = _mm256_mul_ps(zm_lhs, zm_weight);
		}

		//collect same channel face result to faced0.
		//now faced_result[1] contains all the off-diagonal results
		//collected form 1,2,3,4,5,6
		faced_result[1] = _mm256_add_ps(faced_result[1], faced_result[2]);
		faced_result[3] = _mm256_add_ps(faced_result[3], faced_result[4]);
		faced_result[5] = _mm256_add_ps(faced_result[5], faced_result[6]);

		faced_result[1] = _mm256_add_ps(faced_result[1], faced_result[3]);
		faced_result[1] = _mm256_add_ps(faced_result[1], faced_result[5]);


		/*Cross terms*/

		//Td2_[][] terms
		//Td2_00
		{
			/****************************************************************************/
			__m256 td200, u_d1_0d0_0d1;
			vdb_SIMD_IO::get_simd_vector(td200, Td2_00, vector_offset, default_off_diag);
			vdb_SIMD_IO::get_simd_vector(u_d1_0d0_0d1, dir1_lhs_data[1][0], vector_offset, _mm256_setzero_ps());
			Td2_cross_result[0] = _mm256_mul_ps(u_d1_0d0_0d1, td200);
		}

		//Td2_01
		{
			/****************************************************************************/
			__m256 td201, u_d1_0d0_pd1;
			vdb_SIMD_IO::get_simd_vector(td201, Td2_01, vector_offset, default_neg_off_diag);
			switch (c) {
			case 0:
				//d1 is y
				//y==7 go to next leaf
				if ((vector_offset & 63u) == 56u) {
					uint32_t tempoffset = vector_offset - 56;
					vdb_SIMD_IO::get_simd_vector(u_d1_0d0_pd1, dir1_lhs_data[1][1], tempoffset, _mm256_setzero_ps());
				}
				else {
					uint32_t tempoffset = vector_offset + 8;
					vdb_SIMD_IO::get_simd_vector(u_d1_0d0_pd1, dir1_lhs_data[1][0], tempoffset, _mm256_setzero_ps());
				}
				break;
			case 1:
				//d1 is z
				vdb_SIMD_IO::get_pos_z_simd_vector(u_d1_0d0_pd1, dir1_lhs_data[1][0], dir1_lhs_data[1][1], vector_offset, _mm256_setzero_ps());
				break;
			default:
			case 2:
				//d1 is x
				//x==7 go to next leaf
				if (vector_offset >= (7 * 64)) {
					uint32_t tempoffset = vector_offset - (7 * 64);
					vdb_SIMD_IO::get_simd_vector(u_d1_0d0_pd1, dir1_lhs_data[1][1], tempoffset, _mm256_setzero_ps());
				}
				else {
					uint32_t tempoffset = vector_offset + 64;
					vdb_SIMD_IO::get_simd_vector(u_d1_0d0_pd1, dir1_lhs_data[1][0], tempoffset, _mm256_setzero_ps());
				}
				break;
			}
			Td2_cross_result[1] = _mm256_mul_ps(td201, u_d1_0d0_pd1);
		}

		//Td2_10
		{
			/****************************************************************************/
			__m256 td210, u_d1_nd0_0d1;
			switch (c) {
			case 0:
				//d0 is x
				//x==0 go to previoux leaf
				if (vector_offset < 64) {
					uint32_t tempoffset = vector_offset + (7 * 64);
					vdb_SIMD_IO::get_simd_vector(td210, Td2_10[0], tempoffset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d1_nd0_0d1, dir1_lhs_data[0][0], tempoffset, _mm256_setzero_ps());
				}
				else {
					uint32_t tempoffset = vector_offset - 64;
					vdb_SIMD_IO::get_simd_vector(td210, Td2_10[1], tempoffset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d1_nd0_0d1, dir1_lhs_data[1][0], tempoffset, _mm256_setzero_ps());
				}
				break;
			case 1:
				//d0 is y
				//y==0 go to previous leaf
				if ((vector_offset & 63) == 0) {
					uint32_t tempoffset = vector_offset + 56;
					vdb_SIMD_IO::get_simd_vector(td210, Td2_10[0], tempoffset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d1_nd0_0d1, dir1_lhs_data[0][0], tempoffset, _mm256_setzero_ps());
				}
				else {
					uint32_t tempoffset = vector_offset - 8;
					vdb_SIMD_IO::get_simd_vector(td210, Td2_10[1], tempoffset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d1_nd0_0d1, dir1_lhs_data[1][0], tempoffset, _mm256_setzero_ps());
				}
				break;
			case 2:
			default:
				//d0 is z
				vdb_SIMD_IO::get_neg_z_simd_vector(td210, Td2_10[1], Td2_10[0], vector_offset, default_neg_off_diag);
				vdb_SIMD_IO::get_neg_z_simd_vector(u_d1_nd0_0d1, dir1_lhs_data[1][0], dir1_lhs_data[0][0], vector_offset, _mm256_setzero_ps());
				break;
			}
			Td2_cross_result[2] = _mm256_mul_ps(td210, u_d1_nd0_0d1);
		}

		//Td2_11
		{
			/****************************************************************************/
			__m256 td211, u_d1_nd0_pd1;
			switch (c) {
			case 0:
				//d0 is x, d1 is y
				//x==0 go to previous leaf
				if (vector_offset < 64) {
					//coefficient
					uint32_t negdir0_offset = vector_offset + (7 * 64);
					vdb_SIMD_IO::get_simd_vector(td211, Td2_11[0], negdir0_offset, default_off_diag);
					//channel
					//y==7 go to next leaf
					if ((vector_offset & 63u) == 56u) {
						uint32_t negdir0_posdir1_offset = negdir0_offset - 56;
						vdb_SIMD_IO::get_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[0][1], negdir0_posdir1_offset, _mm256_setzero_ps());
					}
					else {
						uint32_t negdir0_posdir1_offset = negdir0_offset + 8;
						vdb_SIMD_IO::get_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[0][0], negdir0_posdir1_offset, _mm256_setzero_ps());
					}
				}
				else {
					//coefficient
					uint32_t negdir0_offset = vector_offset - 64;
					vdb_SIMD_IO::get_simd_vector(td211, Td2_11[1], negdir0_offset, default_off_diag);
					//channel
					//y==7 go to next leaf
					if ((vector_offset & 63u) == 56u) {
						uint32_t negdir0_posdir1_offset = negdir0_offset - 56;
						vdb_SIMD_IO::get_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[1][1], negdir0_posdir1_offset, _mm256_setzero_ps());
					}
					else {
						uint32_t negdir0_posdir1_offset = negdir0_offset + 8;
						vdb_SIMD_IO::get_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[1][0], negdir0_posdir1_offset, _mm256_setzero_ps());
					}
				}
				break;
			case 1:
				//d0 ix y, d1 is z
				//y==0 go to previous leaf
				if ((vector_offset & 63) == 0) {
					//coefficient
					uint32_t negdir0_offset = vector_offset + 56;
					vdb_SIMD_IO::get_simd_vector(td211, Td2_11[0], negdir0_offset, default_off_diag);
					//channel
					vdb_SIMD_IO::get_pos_z_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[0][0], dir1_lhs_data[0][1], negdir0_offset, _mm256_setzero_ps());
				}
				else {
					//coefficient
					uint32_t negdir0_offset = vector_offset - 8;
					vdb_SIMD_IO::get_simd_vector(td211, Td2_11[1], negdir0_offset, default_off_diag);
					//channel
					vdb_SIMD_IO::get_pos_z_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[1][0], dir1_lhs_data[1][1], negdir0_offset, _mm256_setzero_ps());
				}
				break;
			case 2:
			default:
				//d0 is z, d1 is x
				//coefficient
				vdb_SIMD_IO::get_neg_z_simd_vector(td211, Td2_11[1], Td2_11[0], vector_offset, default_off_diag);
				//if x==7, go to next leaf pairs
				if (vector_offset >= (7 * 64)) {
					uint32_t posdir2_offset = vector_offset - (7 * 64);
					vdb_SIMD_IO::get_neg_z_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[1][1], dir1_lhs_data[0][1], posdir2_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t posdir2_offset = vector_offset + 64;
					vdb_SIMD_IO::get_neg_z_simd_vector(u_d1_nd0_pd1, dir1_lhs_data[1][0], dir1_lhs_data[0][0], posdir2_offset, _mm256_setzero_ps());
				}
				break;
			}
			Td2_cross_result[3] = _mm256_mul_ps(td211, u_d1_nd0_pd1);
		}

		//Td1_[][] terms
		//Td1_00
		{
			/****************************************************************************/
			__m256 td100, u_d2_0d0_0d2;
			vdb_SIMD_IO::get_simd_vector(td100, Td1_00, vector_offset, default_off_diag);
			vdb_SIMD_IO::get_simd_vector(u_d2_0d0_0d2, dir2_lhs_data[0][1], vector_offset, _mm256_setzero_ps());
			Td1_cross_result[0] = _mm256_mul_ps(td100, u_d2_0d0_0d2);
		}

		//Td1_01
		{
			/****************************************************************************/
			__m256 td101, u_d2_nd0_0d2;
			switch (c) {
			case 0:
				//0 is x
				//x==0 to go previous leaf
				if (vector_offset < 64) {
					uint32_t negdir0_offset = vector_offset + (7 * 64);
					vdb_SIMD_IO::get_simd_vector(td101, Td1_01[0], negdir0_offset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d2_nd0_0d2, dir2_lhs_data[0][0], negdir0_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t negdir0_offset = vector_offset - 64;
					vdb_SIMD_IO::get_simd_vector(td101, Td1_01[1], negdir0_offset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d2_nd0_0d2, dir2_lhs_data[0][1], negdir0_offset, _mm256_setzero_ps());
				}
				break;
			case 1:
				//d0 is y
				//y==0 go to previous leaf
				if ((vector_offset & 63) == 0) {
					uint32_t negdir0_offset = vector_offset + 56;
					vdb_SIMD_IO::get_simd_vector(td101, Td1_01[0], negdir0_offset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d2_nd0_0d2, dir2_lhs_data[0][0], negdir0_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t negdir0_offset = vector_offset - 8;
					vdb_SIMD_IO::get_simd_vector(td101, Td1_01[1], negdir0_offset, default_neg_off_diag);
					vdb_SIMD_IO::get_simd_vector(u_d2_nd0_0d2, dir2_lhs_data[0][1], negdir0_offset, _mm256_setzero_ps());
				}
				break;
			default:
			case 2:
				//d0 is z
				vdb_SIMD_IO::get_neg_z_simd_vector(td101, Td1_01[1], Td1_01[0], vector_offset, default_neg_off_diag);
				vdb_SIMD_IO::get_neg_z_simd_vector(u_d2_nd0_0d2, dir2_lhs_data[0][1], dir2_lhs_data[0][0], vector_offset, _mm256_setzero_ps());
				break;
			}
			Td1_cross_result[1] = _mm256_mul_ps(td101, u_d2_nd0_0d2);
		}

		//Td1_10
		{
			/****************************************************************************/
			__m256 td110, u_d2_0d0_pd2;
			vdb_SIMD_IO::get_simd_vector(td110, Td1_10, vector_offset, default_neg_off_diag);
			switch (c) {
			case 0:
				//d2 is z
				vdb_SIMD_IO::get_pos_z_simd_vector(u_d2_0d0_pd2, dir2_lhs_data[0][1], dir2_lhs_data[1][1], vector_offset, _mm256_setzero_ps());
				break;
			case 1:
				//d2 is x
				//x==7 goto next leaf
				if (vector_offset >= (7 * 64)) {
					uint32_t posdir2_offset = vector_offset - (7 * 64);
					vdb_SIMD_IO::get_simd_vector(u_d2_0d0_pd2, dir2_lhs_data[1][1], posdir2_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t posdir2_offset = vector_offset + 64;
					vdb_SIMD_IO::get_simd_vector(u_d2_0d0_pd2, dir2_lhs_data[0][1], posdir2_offset, _mm256_setzero_ps());
				}
				break;
			default:
			case 2:
				//d2 is y
				//y==7 go to next leaf
				if ((vector_offset & 63u) == 56u) {
					uint32_t posdir2_offset = vector_offset - 56;
					vdb_SIMD_IO::get_simd_vector(u_d2_0d0_pd2, dir2_lhs_data[1][1], posdir2_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t posdir2_offset = vector_offset + 8;
					vdb_SIMD_IO::get_simd_vector(u_d2_0d0_pd2, dir2_lhs_data[0][1], posdir2_offset, _mm256_setzero_ps());
				}
				break;
			}
			Td1_cross_result[2] = _mm256_mul_ps(td110, u_d2_0d0_pd2);
		}

		//Td1_11
		{
			/****************************************************************************/
			__m256 t111, u_dir2_nd0_pd2;
			switch (c) {
			case 0:
				//d0 is x d2 is z
				if (vector_offset < 64) {
					uint32_t negdir0_offset = vector_offset + (7 * 64);
					vdb_SIMD_IO::get_simd_vector(t111, Td1_11[0], negdir0_offset, default_off_diag);
					vdb_SIMD_IO::get_pos_z_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[0][0], dir2_lhs_data[1][0], negdir0_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t negdir0_offset = vector_offset - 64;
					vdb_SIMD_IO::get_simd_vector(t111, Td1_11[1], negdir0_offset, default_off_diag);
					vdb_SIMD_IO::get_pos_z_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[0][1], dir2_lhs_data[1][1], negdir0_offset, _mm256_setzero_ps());
				}
				break;
			case 1:
				//d0 is y d2 is x
				//y==0 go to previous leaf
				if ((vector_offset & 63) == 0) {
					uint32_t negdir0_offset = vector_offset + 56;
					vdb_SIMD_IO::get_simd_vector(t111, Td1_11[0], negdir0_offset, default_off_diag);
					//x==7 go to next leaf
					if (vector_offset >= (7 * 64)) {
						uint32_t negdir0_posdir2_offset = negdir0_offset - (7 * 64);
						vdb_SIMD_IO::get_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[1][0], negdir0_posdir2_offset, _mm256_setzero_ps());
					}
					else {
						uint32_t negdir0_posdir2_offset = negdir0_offset + 64;
						vdb_SIMD_IO::get_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[0][0], negdir0_posdir2_offset, _mm256_setzero_ps());
					}
				}
				else {
					uint32_t negdir0_offset = vector_offset - 8;
					vdb_SIMD_IO::get_simd_vector(t111, Td1_11[1], negdir0_offset, default_off_diag);
					//x==7 go to next leaf
					if (vector_offset >= (7 * 64)) {
						uint32_t negdir0_posdir2_offset = negdir0_offset - (7 * 64);
						vdb_SIMD_IO::get_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[1][1], negdir0_posdir2_offset, _mm256_setzero_ps());
					}
					else {
						uint32_t negdir0_posdir2_offset = negdir0_offset + 64;
						vdb_SIMD_IO::get_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[0][1], negdir0_posdir2_offset, _mm256_setzero_ps());
					}
				}
				break;
			default:
			case 2:
				//d0 is z d2 is y
				vdb_SIMD_IO::get_neg_z_simd_vector(t111, Td1_11[1], Td1_11[0], vector_offset, default_off_diag);
				//y==7 go to next leaf
				if ((vector_offset & 63u) == 56u) {
					uint32_t posdir2_offset = vector_offset - 56;
					vdb_SIMD_IO::get_neg_z_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[1][1], dir2_lhs_data[1][0], posdir2_offset, _mm256_setzero_ps());
				}
				else {
					uint32_t posdir2_offset = vector_offset + 8;
					vdb_SIMD_IO::get_neg_z_simd_vector(u_dir2_nd0_pd2, dir2_lhs_data[0][1], dir2_lhs_data[0][0], posdir2_offset, _mm256_setzero_ps());
				}
				break;
			}
			Td1_cross_result[3] = _mm256_mul_ps(t111, u_dir2_nd0_pd2);
		}


		//collect Td2 results to 0
		Td2_cross_result[0] = _mm256_add_ps(Td2_cross_result[0], Td2_cross_result[1]);
		Td2_cross_result[2] = _mm256_add_ps(Td2_cross_result[2], Td2_cross_result[3]);

		Td2_cross_result[0] = _mm256_add_ps(Td2_cross_result[0], Td2_cross_result[2]);

		//collect Td1 results to 0
		Td1_cross_result[0] = _mm256_add_ps(Td1_cross_result[0], Td1_cross_result[1]);
		Td1_cross_result[2] = _mm256_add_ps(Td1_cross_result[2], Td1_cross_result[3]);

		Td1_cross_result[0] = _mm256_add_ps(Td1_cross_result[0], Td1_cross_result[2]);

		//collect cross term results to face result1
		//representing all off-diagonal terms
		faced_result[1] = _mm256_add_ps(faced_result[1], Td2_cross_result[0]);
		faced_result[1] = _mm256_add_ps(faced_result[1], Td1_cross_result[0]);

		//faced_result[0] stores the result of A*lhs
		//faced_result[6] = _mm256_add_ps(faced_result[4], faced_result[6]);
		__m256  this_lhs, this_diag;
		vdb_SIMD_IO::get_simd_vector_unsafe(this_lhs, lhs_data[0], vector_offset);
		vdb_SIMD_IO::get_simd_vector(this_diag, diagdata, vector_offset, default_diag);
		switch (m_working_mode) {
		case working_mode::BLACK_GS:
		{
			__m256 this_rhs;
			__m256 residual;
			__m256 this_invdiag;
			vdb_SIMD_IO::get_simd_vector_unsafe(this_rhs, rhs_data, vector_offset);
			residual = _mm256_sub_ps(this_rhs, faced_result[1]);
			vdb_SIMD_IO::get_simd_vector(this_invdiag, invdiagdata, vector_offset, default_invdiag);
			residual = _mm256_mul_ps(residual, this_invdiag);
			residual = _mm256_mul_ps(residual, packed_w_jacobi);
			faced_result[0] = _mm256_fmadd_ps(this_lhs, packed_1minus_w_jacobi, residual);
			//blend result
			if (vector_even) {
				faced_result[0] = _mm256_blend_ps(this_lhs, faced_result[0], 0b10101010);
			}
			else {
				faced_result[0] = _mm256_blend_ps(this_lhs, faced_result[0], 0b01010101);
			}
			break;
		}

		case working_mode::RED_GS:
		{
			__m256 this_rhs;
			__m256 residual;
			__m256 this_invdiag;
			vdb_SIMD_IO::get_simd_vector_unsafe(this_rhs, rhs_data, vector_offset);
			residual = _mm256_sub_ps(this_rhs, faced_result[1]);
			vdb_SIMD_IO::get_simd_vector(this_invdiag, invdiagdata, vector_offset, default_invdiag);
			residual = _mm256_mul_ps(residual, this_invdiag);
			residual = _mm256_mul_ps(residual, packed_w_jacobi);
			faced_result[0] = _mm256_fmadd_ps(this_lhs, packed_1minus_w_jacobi, residual);
			//blend result
			if (vector_even) {
				faced_result[0] = _mm256_blend_ps(this_lhs, faced_result[0], 0b01010101);
			}
			else {
				faced_result[0] = _mm256_blend_ps(this_lhs, faced_result[0], 0b10101010);
			}
			break;
		}
		case working_mode::JACOBI:
		{
			__m256 this_rhs;
			__m256 residual;
			__m256 this_invdiag;
			//v^1 = v^0 + w D^-1 r
			// r = rhs - A*v^0
			//v^0 is the input lhs
			//v^1 is the output result
			/****************************************************************************/
			faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
			vdb_SIMD_IO::get_simd_vector_unsafe(this_rhs, rhs_data, vector_offset);
			residual = _mm256_sub_ps(this_rhs, faced_result[0]);
			vdb_SIMD_IO::get_simd_vector(this_invdiag, invdiagdata, vector_offset, default_invdiag);
			residual = _mm256_mul_ps(residual, this_invdiag);
			faced_result[0] = _mm256_fmadd_ps(packed_w_jacobi, residual, this_lhs);
			break;
		}
		case working_mode::SPAI0:
		{
			__m256 this_rhs;
			__m256 residual;
			__m256 this_spai0;
			/****************************************************************************/
			faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
			vdb_SIMD_IO::get_simd_vector_unsafe(this_rhs, rhs_data, vector_offset);
			residual = _mm256_sub_ps(this_rhs, faced_result[0]);
			vdb_SIMD_IO::get_simd_vector(this_spai0, spai0_data, vector_offset, _mm256_setzero_ps());
			residual = _mm256_mul_ps(residual, this_spai0);
			faced_result[0] = _mm256_fmadd_ps(packed_w_jacobi, residual, this_lhs);
			break;
		}
		case working_mode::RESIDUAL:
		{
			__m256 this_rhs;
			/****************************************************************************/
			faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
			vdb_SIMD_IO::get_simd_vector_unsafe(this_rhs, rhs_data, vector_offset);
			faced_result[0] = _mm256_sub_ps(this_rhs, faced_result[0]);
			break;
		}
		default:
		case working_mode::NORMAL:
			/****************************************************************************/
			faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
			break;
		}//end case working mode

		//write to the result
		_mm256_storeu_ps(result_data + vector_offset, faced_result[0]);
		//make sure we write at the correct place
		for (int bit = 0; bit < 8; bit++) {
			if (0 == ((vectormask) & (1 << bit))) {
				result_data[vector_offset + bit] = 0;
			}
		}

	}//end vectorid = [0 63]
}

void L_with_level::light_weight_applier::operator()(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) const
{
	switch (channel) {
	case 0:
		component_operator<0>(leaf, leafpos);
		break;
	case 1:
		component_operator<1>(leaf, leafpos);
		break;
	case 2:
		component_operator<2>(leaf, leafpos);
		break;
	}
}


simd_viscosity3d::simd_viscosity3d(
	openvdb::FloatGrid::Ptr in_viscosity,
	openvdb::FloatGrid::Ptr in_liquid_sdf,
	openvdb::FloatGrid::Ptr in_solid_sdf,
	packed_FloatGrid3 in_liquid_velocity,
	openvdb::Vec3fGrid::Ptr in_solid_velocity,
	float in_dt, float in_rho)
{
	m_max_iter = 100;
	m_iteration = 0;
	std::shared_ptr<L_with_level> lv0 = std::make_shared<L_with_level>(
		in_viscosity,
		in_liquid_sdf,
		in_solid_sdf, in_dt, in_rho);

	m_rhs = lv0->build_rhs(in_liquid_velocity, in_solid_velocity);
	m_matrix_levels.push_back(lv0);

	int max_ndof = 4000;
	while (m_matrix_levels.back()->m_ndof > max_ndof) {
		std::shared_ptr<L_with_level> new_level = std::make_shared<L_with_level>(
			*m_matrix_levels.back(), L_with_level::Coarsening()
			);
		m_matrix_levels.push_back(new_level);
	}

	printf("level %zd, dof %d\n", m_matrix_levels.size(), m_matrix_levels[0]->m_ndof);
	m_matrix_levels[0]->trim_default_nodes();
	m_matrix_levels.back()->construct_exact_solver();
	
	//printf("gen_scratchpads\n");
	for (int i = 0; i < m_matrix_levels.size(); i++) {
		m_mucycle_lhss.push_back(m_matrix_levels[i]->get_zero_vec());
		m_mucycle_temps.push_back(m_matrix_levels[i]->get_zero_vec());
		m_mucycle_rhss.push_back(m_matrix_levels[i]->get_zero_vec());
	}
	//printf("simdsolvectordone\n");
}

simd_viscosity3d::simd_viscosity3d(L_with_level::Ptr level0,
	packed_FloatGrid3 in_liquid_velocity,
	openvdb::Vec3fGrid::Ptr in_solid_velocity)
{
	m_max_iter = 100;
	m_iteration = 0;

	m_rhs = level0->build_rhs(in_liquid_velocity, in_solid_velocity);
	m_matrix_levels.push_back(level0);

	int max_ndof = 4000;
	while (m_matrix_levels.back()->m_ndof > max_ndof) {
		std::shared_ptr<L_with_level> new_level = std::make_shared<L_with_level>(
			*m_matrix_levels.back(), L_with_level::Coarsening()
			);
		m_matrix_levels.push_back(new_level);
	}

	printf("level %zd, dof %d\n", m_matrix_levels.size(), m_matrix_levels[0]->m_ndof);
	m_matrix_levels[0]->trim_default_nodes();
	m_matrix_levels.back()->construct_exact_solver();

	//printf("gen_scratchpads\n");
	for (int i = 0; i < m_matrix_levels.size(); i++) {
		m_mucycle_lhss.push_back(m_matrix_levels[i]->get_zero_vec());
		m_mucycle_temps.push_back(m_matrix_levels[i]->get_zero_vec());
		m_mucycle_rhss.push_back(m_matrix_levels[i]->get_zero_vec());
	}
	//printf("simdsolvectordone\n");
}

void simd_viscosity3d::pcg_solve(packed_FloatGrid3 in_lhs, float tolerance)
{
	
	auto& level0 = *m_matrix_levels[0];

	m_iteration = 0;

	//according to mcadams algorithm 3

	//line2
	auto r = level0.get_zero_vec();
	level0.residual_apply(r, in_lhs, m_rhs);
	float nu = lv_abs_max(r);
	float initAbsoluteError = nu + 1e-16f;
	float numax = tolerance * nu; //numax = std::min(numax, 1e-7f);
	printf("init err:%e\n", 1.0f);
	//line3
	if (nu <= numax) {
		printf("iter:%d err:%e\n", m_iteration + 1, 1.0f);
		return;
	}

	//line4
	auto p = level0.get_zero_vec();
	level0.set_grid_to_zero(p);
	mucycle(p, r, /*mu_time*/2, /*for preconditioner*/true, 0, 2);
	float rho = lv_dot(p, r);

	auto z = level0.get_zero_vec();
	//line 5
	for (; m_iteration < m_max_iter; m_iteration++) {
		//line6
		level0.L_apply(z, p);
		float sigma = lv_dot(p, z);
		//line7
		float alpha = rho / sigma;
		//line8
		lv_axpy(-alpha, z, r);
		nu = lv_abs_max(r); printf("iter:%d err:%e\n", m_iteration + 1, nu/initAbsoluteError);
		//openvdb::io::File("CGresidual" + std::to_string(mIterationTaken) + ".vdb").write({ r.v[0], r.v[1], r.v[2] });
		//line9
		if (nu <= numax || m_iteration == (m_max_iter - 1)) {
			//line10
			lv_axpy(alpha, p, in_lhs);
			return;
		}
		//line13
		level0.set_grid_to_zero(z);
		mucycle(z, r, /*mu_time*/2, /*for preconditioner*/true, 0, 2);

		float rho_new = lv_dot(z, r);

		//line14
		float beta = rho_new / rho;
		//line15
		rho = rho_new;
		//line16
		lv_axpy(alpha, p, in_lhs);
		
		lv_xpay(beta, z, p);
		//line17
	}
	//line18
}

float simd_viscosity3d::lv_abs_max(packed_FloatGrid3 in_grid, int level)
{
	
	float result = 0;
	for (int i = 0; i < 3; i++) {
		auto op{ grid_abs_max_op(in_grid.v[i]) };
		m_matrix_levels[level]->m_dof_manager[i]->reduce(op);
		result = std::max(result, op.m_max);
	}

	return result;
}

float simd_viscosity3d::lv_dot(packed_FloatGrid3 a, packed_FloatGrid3 b, int level)
{
	
	float result = 0;
	for (int i = 0; i < 3; i++) {
		auto op{ grid_dot_op{ a.v[i], b.v[i]} };
		m_matrix_levels[level]->m_dof_manager[i]->reduce(op);
		result += op.dp_result;
	}

	return result;
}

void simd_viscosity3d::lv_axpy(const float alpha, packed_FloatGrid3 x, packed_FloatGrid3 y, int level)
{
	
	for (int i = 0; i < 3; i++) {
		//y = a*x + y
		auto add_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto* xleaf = x.v[i]->tree().probeConstLeaf(leaf.origin());
			auto* yleaf = y.v[i]->tree().probeLeaf(leaf.origin());

			const float* xdata = xleaf->buffer().data();
			float* ydata = yleaf->buffer().data();
			for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
				ydata[iter.offset()] += alpha * xdata[iter.offset()];
			}
		};//end add_op

		m_matrix_levels[level]->m_dof_manager[i]->foreach(add_op);
	}
}

void simd_viscosity3d::lv_xpay(const float alpha, packed_FloatGrid3 x, packed_FloatGrid3 y, int level)
{
	
	for (int i = 0; i < 3; i++) {
		//y = x + a*y
		auto add_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
			auto* xleaf = x.v[i]->tree().probeConstLeaf(leaf.origin());
			auto* yleaf = y.v[i]->tree().probeLeaf(leaf.origin());

			const float* xdata = xleaf->buffer().data();
			float* ydata = yleaf->buffer().data();
			for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
				ydata[iter.offset()] = xdata[iter.offset()] + alpha * ydata[iter.offset()];
			}
		};//end add_op

		m_matrix_levels[level]->m_dof_manager[i]->foreach(add_op);
	}
}

void simd_viscosity3d::lv_copyval(packed_FloatGrid3 out_grid, packed_FloatGrid3 in_grid, int level)
{
	
	for (int i = 0; i < 3; i++) {
		//it is assumed that the 
		auto copy_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index ) {
			auto* in_leaf = in_grid.v[i]->tree().probeLeaf(leaf.origin());
			auto* out_leaf = out_grid.v[i]->tree().probeLeaf(leaf.origin());
			std::copy(in_leaf->buffer().data(), in_leaf->buffer().data() + in_leaf->SIZE, out_leaf->buffer().data());
		};

		m_matrix_levels[level]->m_dof_manager[i]->foreach(copy_op);
	}
}

void simd_viscosity3d::mucycle(
	packed_FloatGrid3 in_out_lhs, const packed_FloatGrid3 in_rhs, const int mu_time, const bool for_precon, const int level, int n)
{
	
	static int iter = 0;
	int n1 = n;
	int n2 = n;
	size_t nlevel = m_matrix_levels.size();
	m_mucycle_lhss[level] = in_out_lhs;
	m_mucycle_rhss[level] = in_rhs;

	if (level == nlevel - 1) {
		m_matrix_levels[level]->solve_exact(in_out_lhs, in_rhs);
		return;
	}

	if (for_precon) {
		//m_matrix_levels[level]->set_grid_to_SPAI0_after_first_iteration(mMuCycleTemps[level], mMuCycleRHSs[level]);
		//mMuCycleTemps[level].swap(mMuCycleLHSs[level]);
		m_matrix_levels[level]->set_grid_to_zero(m_mucycle_lhss[level]);
	}

	for (int i = (for_precon ? 0 : 0); i < n1; i++) {
		m_matrix_levels[level]->XYZ_RBGS_apply(
			m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level]);
	}

	m_matrix_levels[level]->residual_apply(
		m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level]);

	int parent_level = level + 1;

	m_matrix_levels[level]->restriction(
		m_mucycle_rhss[parent_level], m_mucycle_temps[level], /*parent level*/ *m_matrix_levels[parent_level]);

	mucycle(m_mucycle_lhss[parent_level], m_mucycle_rhss[parent_level], mu_time, true, parent_level, n);
	for (int mu = 1; mu < mu_time; mu++) {
		mucycle(m_mucycle_lhss[parent_level], m_mucycle_rhss[parent_level], mu_time, false, parent_level, n);
	}

	m_matrix_levels[parent_level]->prolongation</*inplace add*/true>(
		m_mucycle_lhss[level], m_mucycle_lhss[parent_level]);

	for (int i = 0; i < n2; i++) {
		m_matrix_levels[level]->ZYX_RBGS_apply(
			m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level]);
	}
}

}//end namespace simd_uaamg