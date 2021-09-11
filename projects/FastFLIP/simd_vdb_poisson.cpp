#include "simd_vdb_poisson.h"
//#include "Timer.h"
#include "immintrin.h"
#include "levelset_util.h"
#include "openvdb/tools/Interpolation.h"
#include "openvdb/tree/LeafManager.h"
#include "tbb/concurrent_vector.h"
#include <atomic>
simd_vdb_poisson::Laplacian_with_level::Laplacian_with_level(
    openvdb::FloatGrid::Ptr in_liquid_phi,
    openvdb::Vec3fGrid::Ptr in_face_weights, const float in_dt,
    const float in_dx) {
  // random token to identify the finest level
  std::random_device device;
  std::mt19937 generator(/*seed=*/device());
  std::uniform_real_distribution<> distribution(-0.5, 0.5);
  m_root_token = distribution(generator);

  m_dt = in_dt;
  m_dx_finest = in_dx;
  m_dx_this_level = in_dx;
  m_level = 0;

  float term = m_dt / (m_dx_this_level * m_dx_this_level);
  m_diag_entry_min_threshold = term * 1e-1;
  // Note we do not set the background of the diagonal to be zero
  // later when we run the red-black Gauss seidel in simd instruction
  // we need to divide the result by the diagonal term
  // to avoid dividing by zero we set it to be one
  // then the mask of the diagonal term becomes very important
  // It is the only indicator to show this is a degree of freedom
  m_Diagonal = openvdb::FloatGrid::create(6 * term);
  m_Diagonal->setName("Poisson_Diagonal");
  m_Diagonal->setTransform(in_liquid_phi->transformPtr());
  m_Diagonal->setTree(std::make_shared<openvdb::FloatTree>(
      in_liquid_phi->tree(), 6 * term, openvdb::TopologyCopy()));

  // by default, the solid is sparse
  // so only a fraction of faces have weight<1
  // the default value of the vector form of face weight is 1
  // so the default value for each component is 1, but we directly
  // store the entry here
  m_Neg_x_entry = openvdb::FloatGrid::create(-term);
  m_Neg_x_entry->setName("Neg_x_term");
  m_Neg_x_entry->setTransform(in_face_weights->transformPtr());
  m_Neg_x_entry->setTree(std::make_shared<openvdb::FloatTree>(
      in_liquid_phi->tree(), -term, openvdb::TopologyCopy()));

  m_Neg_y_entry = m_Neg_x_entry->deepCopy();
  m_Neg_y_entry->setName("Neg_y_term");

  m_Neg_z_entry = m_Neg_x_entry->deepCopy();
  m_Neg_z_entry->setName("Neg_z_term");

  initialize_finest(in_liquid_phi, in_face_weights);

  initialize_evaluators();
}

simd_vdb_poisson::Laplacian_with_level::Laplacian_with_level(
    const Laplacian_with_level &parent, coarsening) {
  // copy the token
  m_root_token = parent.m_root_token;

  m_dt = parent.m_dt;
  m_dx_finest = parent.m_dx_finest;
  m_dx_this_level = 2.0f * parent.m_dx_this_level;
  m_level = parent.m_level + 1;

  initialize_entries_from_parent(parent);
}

void simd_vdb_poisson::Laplacian_with_level::initialize_entries_from_parent(
    const Laplacian_with_level &parent) {
  // the parent diagonal and face terms is already trimmed
  // hence only the parent dof_idx keeps the actual layout
  // of the degree of freedoms
  auto coarse_xform =
      openvdb::math::Transform::createLinearTransform(m_dx_this_level);
  coarse_xform->postTranslate(openvdb::Vec3d(0.5 * parent.m_dx_this_level));

  m_dof_idx = openvdb::Int32Grid::create(-1);
  m_dof_idx->setTransform(coarse_xform);
  for (auto iter = parent.m_dof_idx->tree().cbeginLeaf(); iter; ++iter) {
    m_dof_idx->tree().touchLeaf(openvdb::Coord(iter->origin().asVec3i() / 2));
  }

  m_dof_leafmanager =
      std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(
          m_dof_idx->tree());

  // piecewise constant interpolation and restriction function
  // coarse voxel =8 fine voxels
  auto set_dof_mask_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                             openvdb::Index leafpos) {
    auto fine_dof_axr{parent.m_dof_idx->getConstAccessor()};
    for (auto iter = leaf.beginValueAll(); iter; ++iter) {
      // the global coordinate in the coarse level
      auto C_gcoord{iter.getCoord().asVec3i()};
      auto parent_g_coord = C_gcoord * 2;

      bool no_target_dof = true;
      for (int ii = 0; ii < 2 && no_target_dof; ii++) {
        for (int jj = 0; jj < 2 && no_target_dof; jj++) {
          for (int kk = 0; kk < 2 && no_target_dof; kk++) {
            if (fine_dof_axr.isValueOn(
                    openvdb::Coord(parent_g_coord).offsetBy(ii, jj, kk))) {
              no_target_dof = false;
            }
          }
        }
      } // for all fine voxels accociated

      if (no_target_dof) {
        iter.setValueOff();
      } else {
        iter.setValueOn();
      }
    } // end for all voxel in this leaf
  };  // end set dof_idx_op
  m_dof_leafmanager->foreach (set_dof_mask_op);

  set_dof_idx(m_dof_idx);

  float dt_over_dxsqr = m_dt / (m_dx_this_level * m_dx_this_level);
  // set up the full diagonal matrix, full face weight matrix

  m_Diagonal = openvdb::FloatGrid::create(6.0f * dt_over_dxsqr);
  m_Diagonal->setTransform(coarse_xform);
  m_Diagonal->setName("Poisson_Diagonal_level_" + std::to_string(m_level));
  m_Diagonal->setTree(std::make_shared<openvdb::FloatTree>(
      m_dof_idx->tree(), 6.0f * dt_over_dxsqr, openvdb::TopologyCopy()));

  m_Neg_x_entry = openvdb::FloatGrid::create(-dt_over_dxsqr);
  m_Neg_x_entry->setName("Neg_x_term_level_" + std::to_string(m_level));
  m_Neg_x_entry->setTransform(m_Diagonal->transformPtr());
  m_Neg_x_entry->setTree(std::make_shared<openvdb::FloatTree>(
      m_dof_idx->tree(), -dt_over_dxsqr, openvdb::TopologyCopy()));

  m_Neg_y_entry = m_Neg_x_entry->deepCopy();
  m_Neg_y_entry->setName("Neg_y_term_level_" + std::to_string(m_level));

  m_Neg_z_entry = m_Neg_x_entry->deepCopy();
  m_Neg_z_entry->setName("Neg_z_term_level_" + std::to_string(m_level));

  // The prolongation operator is
  // fine = coarse
  // the restriction operator is
  // coarse = 1/8*(eight fine)
  // ideally, if the fine level diagonal is 6
  // then the coarse level diagonal should be 1.5, because dx_c=dx_f*2
  // 1. a input coarse voxel turns on up to 8 fine voxels
  // 2. each fine voxel scatters to its center 6, and its neighbor -1
  // 3. the results are restricted
  // in a 2D example, original weight is 4, after scatter it becomes
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
  // it should additionally multiply by 0.5
  // to get correct coefficients

  auto set_terms = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                       openvdb::Index leafpos) {
    auto x_axr{parent.m_Neg_x_entry->getConstAccessor()};
    auto y_axr{parent.m_Neg_y_entry->getConstAccessor()};
    auto z_axr{parent.m_Neg_z_entry->getConstAccessor()};
    auto diag_axr{parent.m_Diagonal->getConstAccessor()};
    auto fine_dof_axr{parent.m_dof_idx->getConstAccessor()};

    auto *diagleaf = m_Diagonal->tree().probeLeaf(leaf.origin());
    auto *xleaf = m_Neg_x_entry->tree().probeLeaf(leaf.origin());
    auto *yleaf = m_Neg_y_entry->tree().probeLeaf(leaf.origin());
    auto *zleaf = m_Neg_z_entry->tree().probeLeaf(leaf.origin());

    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      // each voxel only record the flux of its and its below dof
      float voxel_diag = 0, voxel_x_term = 0, voxel_y_term = 0,
            voxel_z_term = 0;
      // the global coordinate in the coarse level
      auto C_gcoord{iter.getCoord().asVec3i()};
      auto fine_g_coord = openvdb::Coord(C_gcoord * 2);

      // loop over the 8 fine cells
      for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
          for (int kk = 0; kk < 2; kk++) {
            auto at_voxel = fine_g_coord.offsetBy(ii, jj, kk);

            // contribution to diag
            if (fine_dof_axr.isValueOn(at_voxel)) {
              voxel_diag += diag_axr.getValue(at_voxel);

              // three face terms
              auto neg_x_voxel = at_voxel;
              neg_x_voxel[0]--;
              if (fine_dof_axr.isValueOn(neg_x_voxel)) {
                if (ii == 0) {
                  // contribute to the neg x term
                  voxel_x_term += x_axr.getValue(at_voxel);
                } else {
                  // this face will decrease two fine dofs
                  // diagonal terms
                  voxel_diag += 2 * x_axr.getValue(at_voxel);
                }
              } // end if x- neib on

              auto neg_y_voxel = at_voxel;
              neg_y_voxel[1]--;
              if (fine_dof_axr.isValueOn(neg_y_voxel)) {
                if (jj == 0) {
                  voxel_y_term += y_axr.getValue(at_voxel);
                } else {
                  voxel_diag += 2 * y_axr.getValue(at_voxel);
                }
              } // end if y- neib on

              auto neg_z_voxel = at_voxel;
              neg_z_voxel[2]--;
              if (fine_dof_axr.isValueOn(neg_z_voxel)) {
                if (kk == 0) {
                  voxel_z_term += z_axr.getValue(at_voxel);
                } else {
                  voxel_diag += 2 * z_axr.getValue(at_voxel);
                }
              } // end if z- neib on
            }   // end if this voxel is on

          } // end fine kk
        }   // end fine jj
      }     // end fine ii

      // coefficient R accounts for 1/8
      // Then there is an additional 1/2
      auto offset = iter.offset();
      const float factor = 0.5f * (1.0f / 8.0f);
      diagleaf->setValueOnly(offset, voxel_diag * factor);
      xleaf->setValueOnly(offset, voxel_x_term * factor);
      yleaf->setValueOnly(offset, voxel_y_term * factor);
      zleaf->setValueOnly(offset, voxel_z_term * factor);
    } // end loop over all coarse dofs
  };  // end set terms
  m_dof_leafmanager->foreach (set_terms);

  initialize_evaluators();
  trim_default_nodes();
}

void simd_vdb_poisson::Laplacian_with_level::initialize_finest(
    openvdb::FloatGrid::Ptr in_liquid_phi,
    openvdb::Vec3fGrid::Ptr in_face_weights) {
  // really create the poisson matrix

  float dt_over_dxsqr = m_dt / (m_dx_this_level * m_dx_this_level);

  // this function is to be used with the leaf manager of diagonal term
  auto set_coeffs = [&](float_leaf_t &leaf, openvdb::Index leafpos) {
    // in the initial state, the tree contains both phi<0 and phi>0
    // we only take the phi<0
    const auto &phi_leaf = *in_liquid_phi->tree().probeConstLeaf(leaf.origin());
    auto phi_axr{in_liquid_phi->getConstAccessor()};
    auto weight_axr{in_face_weights->getConstAccessor()};

    // everytime only write to the coefficient on the lower side
    // the positive side will be handled from the dof on the other side
    auto *x_entry_leaf = m_Neg_x_entry->tree().probeLeaf(leaf.origin());
    auto *y_entry_leaf = m_Neg_y_entry->tree().probeLeaf(leaf.origin());
    auto *z_entry_leaf = m_Neg_z_entry->tree().probeLeaf(leaf.origin());

    if (!(x_entry_leaf && y_entry_leaf && z_entry_leaf)) {
      printf("dof leaf exists, but weight entry leaf is not ready\n");
      exit(-1);
    }

    for (auto phi_iter = phi_leaf.cbeginValueOn(); phi_iter; ++phi_iter) {
      if (phi_iter.getValue() < 0) {
        // this is a valid dof
        auto voxel_gcoord = phi_iter.getCoord();

        float this_diag_entry = 0;
        float this_xyz_term[3] = {0, 0, 0};
        float phi_other_cell = 0;
        float weight_other_cell = 0;
        openvdb::Coord other_weight_coord = voxel_gcoord;
        openvdb::Coord other_phi_coord = voxel_gcoord;

        // its six faces
        for (auto i_f = 0; i_f < 6; i_f++) {
          // 0,1,2 direction
          int component = i_f / 2;
          bool positive_dir = (i_f % 2 == 0);

          // reset the other coordinate
          other_weight_coord = voxel_gcoord;
          other_phi_coord = voxel_gcoord;

          if (positive_dir) {
            other_weight_coord[component]++;
            weight_other_cell =
                weight_axr.getValue(other_weight_coord)[component];

            other_phi_coord[component]++;
            phi_other_cell = phi_axr.getValue(other_phi_coord);
          } else {
            weight_other_cell =
                weight_axr.getValue(other_weight_coord)[component];
            other_phi_coord[component]--;
            phi_other_cell = phi_axr.getValue(other_phi_coord);
          } // end else positive direction
          float term = weight_other_cell * dt_over_dxsqr;
          // if other cell is a dof
          if (phi_other_cell < 0) {
            this_diag_entry += term;
            // write to the negative term
            if (!positive_dir) {
              this_xyz_term[component] = -term;
            }
          } else {
            // the other cell is an air cell
            float theta = fraction_inside(phi_iter.getValue(), phi_other_cell);
            if (theta < 0.02f) { theta = 0.02f; }
            this_diag_entry += term / theta;
          } // end else other cell is dof
        }// end for 6 faces
        if (this_diag_entry == 0.f) {
            //for totally isolated voxel
            //mark it as inactive
            leaf.setValueOff(phi_iter.offset());
            x_entry_leaf->setValueOff(phi_iter.offset(), 0.f);
            y_entry_leaf->setValueOff(phi_iter.offset(), 0.f);
            z_entry_leaf->setValueOff(phi_iter.offset(), 0.f);
        }
        else {
            leaf.setValueOn(phi_iter.offset(), this_diag_entry);
            x_entry_leaf->setValueOn(phi_iter.offset(), this_xyz_term[0]);
            y_entry_leaf->setValueOn(phi_iter.offset(), this_xyz_term[1]);
            z_entry_leaf->setValueOn(phi_iter.offset(), this_xyz_term[2]);
        }
      } // end if this voxel is liquid voxel
      else {
        leaf.setValueOff(phi_iter.offset());
        x_entry_leaf->setValueOff(phi_iter.offset());
        y_entry_leaf->setValueOff(phi_iter.offset());
        z_entry_leaf->setValueOff(phi_iter.offset());
      } // else if this voxel is liquid voxel
    }   // end for all touched liquid phi voxels
  };    // end set diagonal

  auto diag_leafman =
      openvdb::tree::LeafManager<openvdb::FloatTree>(m_Diagonal->tree());
  diag_leafman.foreach (set_coeffs);

  // set the dof index
  m_dof_idx = openvdb::Int32Grid::create(0);
  m_dof_idx->setTree(std::make_shared<openvdb::Int32Tree>(
      m_Diagonal->tree(), -1, openvdb::TopologyCopy()));
  m_dof_idx->setTransform(m_Diagonal->transformPtr());

  set_dof_idx(m_dof_idx);
  trim_default_nodes();
}

void simd_vdb_poisson::Laplacian_with_level::set_dof_idx(
    openvdb::Int32Grid::Ptr in_out_dofidx) {  // crash on 0 particles?
  auto dof_leafman =
      openvdb::tree::LeafManager<openvdb::Int32Tree>(in_out_dofidx->tree());

  // first count how many dof in each leaf
  // then assign the global dof id
  std::vector<openvdb::Int32> dof_end_in_each_leaf;
  dof_end_in_each_leaf.assign(in_out_dofidx->tree().leafCount(), 0);

  auto leaf_active_dof_counter = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                                     openvdb::Index leafpos) {
    dof_end_in_each_leaf[leafpos] = leaf.onVoxelCount();
  }; // end leaf active dof counter
  dof_leafman.foreach (leaf_active_dof_counter);

  // scan through all leaves to determine
  for (size_t i = 1; i < dof_end_in_each_leaf.size(); i++) {
    dof_end_in_each_leaf[i] += dof_end_in_each_leaf[i - 1];
  }

  auto set_dof_id = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                        openvdb::Index leafpos) {
    openvdb::Int32 begin_idx = 0;
    if (leafpos != 0) {
      begin_idx = dof_end_in_each_leaf[leafpos - 1];
    }
    leaf.fill(-1);
    for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
      iter.setValue(begin_idx);
      begin_idx++;
    }
  };
  dof_leafman.foreach (set_dof_id);

  // return the total number of degree of freedom
  m_ndof = *dof_end_in_each_leaf.crbegin();
}

// In normal mode applies L * lhs = result
// where lhs is the input
// In weighted jacobi mode
// apply one iteration to solve L * lhs = rhs
// and lhs and rhs are unchanged
struct alignas(32) simd_laplacian_apply_op {
  enum class working_mode {
    NORMAL_LAPLACIAN,
    WEIGHTED_JACOBI,
    SPAI0,
    RESIDUAL,
    RED_GS,
    BLACK_GS
  };

  // default term is dt_over_dxsqr
  simd_laplacian_apply_op(openvdb::FloatGrid::Ptr in_Diagonal,
                          openvdb::FloatGrid::Ptr in_Neg_x_entry,
                          openvdb::FloatGrid::Ptr in_Neg_y_entry,
                          openvdb::FloatGrid::Ptr in_Neg_z_entry,
                          openvdb::FloatGrid::Ptr in_out_lhs,
                          openvdb::FloatGrid::Ptr in_out_rhs,
                          openvdb::FloatGrid::Ptr in_out_result,
                          float in_default_term)
      : m_sor(1.2) {
    m_Diagonal = in_Diagonal;
    m_Neg_x_entry = in_Neg_x_entry;
    m_Neg_y_entry = in_Neg_y_entry;
    m_Neg_z_entry = in_Neg_z_entry;
    m_rhs = in_out_rhs;
    m_w_jacobi = 6.0 / 7.0;
    m_lhs = in_out_lhs;
    m_result = in_out_result;
    default_lhs = _mm256_set1_ps(0);
    default_rhs = _mm256_set1_ps(0);
    default_face_term = _mm256_set1_ps(-in_default_term);
    default_diag = _mm256_set1_ps(6.f * in_default_term);
    default_inv_diagonal = _mm256_set1_ps(1.0f / (6.f * in_default_term));
    default_SPAI0 = _mm256_set1_ps(1.0f / (7.f * in_default_term));
    packed_w_jacobi = _mm256_set1_ps(m_w_jacobi);
    packed_sor = _mm256_set1_ps(m_sor);
    packed_1msor = _mm256_set1_ps(1 - m_sor);
    m_mode = working_mode::NORMAL_LAPLACIAN;
    m_inv_diagonal =
        openvdb::FloatGrid::create(1.0f / m_Diagonal->background());
    m_invdiag_initialized = false;
    m_SPAI0 = openvdb::FloatGrid::create(1.0f / (7.f * in_default_term));
    m_spai_initialized = false;
  }

  void init_invdiag() {
    if (m_invdiag_initialized)
      return;
    // set the inverse default term and inverse diagonal
    m_inv_diagonal->setTree(std::make_shared<openvdb::FloatTree>(
        m_Diagonal->tree(), 1.0f / m_Diagonal->background(),
        openvdb::TopologyCopy()));
    float diag_small_threshold = m_Diagonal->background() / 6.0f * 1e-1;
    auto leafman =
        openvdb::tree::LeafManager<openvdb::FloatTree>(m_inv_diagonal->tree());
    leafman.foreach (
        [&](openvdb::FloatTree::LeafNodeType &leaf, openvdb::Index leafpos) {
          auto *diagleaf = m_Diagonal->tree().probeConstLeaf(leaf.origin());
          for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            float diag_val = diagleaf->getValue(iter.offset());
            if (diag_val == 0) {
              iter.setValue(1);
            } else {
              iter.setValue(1.0f / diag_val);
            }
          }
        });
    m_invdiag_initialized = true;
  };
  void init_SPAI0() {
    if (m_spai_initialized)
      return;
    // set the SPAI0 matrix
    // by default the rhs has the correct pattern, which is not pruned
    // spai diagonal: akk/(ak)_2^2
    // diagonal entry over the row norm
    // set the inverse default term and inverse diagonal

    float default_spai0_val = 6.0f / (7.f * m_Diagonal->background());
    m_SPAI0->setTree(std::make_shared<openvdb::FloatTree>(
        m_rhs->tree(), default_spai0_val, openvdb::TopologyCopy()));
    auto sqr = [](float in) { return in * in; };
    auto spai0_setter = [&](openvdb::FloatTree::LeafNodeType &leaf,
                            openvdb::Index leafpos) {
      auto x_axr{m_Neg_x_entry->getConstAccessor()};
      auto y_axr{m_Neg_y_entry->getConstAccessor()};
      auto z_axr{m_Neg_z_entry->getConstAccessor()};
      auto diag_axr{m_Diagonal->getConstAccessor()};
      auto dof_axr{m_rhs->getConstAccessor()};
      auto spai_leaf = m_SPAI0->tree().probeLeaf(leaf.origin());
      for (auto iter = leaf.beginValueOn(); iter; ++iter) {
        float row_normsqr = 0;
        auto gcoord = iter.getCoord();
        //+x
        auto at_coord = gcoord.offsetBy(1, 0, 0);
        if (dof_axr.isValueOn(at_coord)) {
          row_normsqr += sqr(x_axr.getValue(at_coord));
        }
        //-x
        at_coord = gcoord.offsetBy(-1, 0, 0);
        if (dof_axr.isValueOn(at_coord)) {
          row_normsqr += sqr(x_axr.getValue(gcoord));
        }
        //+y
        at_coord = gcoord.offsetBy(0, 1, 0);
        if (dof_axr.isValueOn(at_coord)) {
          row_normsqr += sqr(y_axr.getValue(at_coord));
        }
        //-y
        at_coord = gcoord.offsetBy(0, -1, 0);
        if (dof_axr.isValueOn(at_coord)) {
          row_normsqr += sqr(y_axr.getValue(gcoord));
        }
        //+z
        at_coord = gcoord.offsetBy(0, 0, 1);
        if (dof_axr.isValueOn(at_coord)) {
          row_normsqr += sqr(z_axr.getValue(at_coord));
        }
        //-z
        at_coord = gcoord.offsetBy(0, 0, -1);
        if (dof_axr.isValueOn(at_coord)) {
          row_normsqr += sqr(z_axr.getValue(gcoord));
        }
        float diagval = diag_axr.getValue(gcoord);
        if (diagval == 0) {
          spai_leaf->setValueOn(iter.offset(), 0);
        } else {
          spai_leaf->setValueOn(iter.offset(),
                                diagval / (sqr(diagval) + row_normsqr));
        }
      } // end for all dof
    };  // end spai setter

    auto dofman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_rhs->tree());
    dofman.foreach (spai0_setter);

    simd_vdb_poisson::Laplacian_with_level::trim_default_nodes(
        m_SPAI0, default_spai0_val, default_spai0_val * 1e-5);
    m_spai_initialized = true;
  };

  void set_normal_mode() { m_mode = working_mode::NORMAL_LAPLACIAN; }
  void set_jacobi_mode() {
    m_mode = working_mode::WEIGHTED_JACOBI;
    init_invdiag();
  }
  void set_SPAI0_mode() {
    m_mode = working_mode::SPAI0;
    init_SPAI0();
  }
  void set_residual_mode() { m_mode = working_mode::RESIDUAL; }
  void set_w_jacobi(float in_w_jacobi) {
    m_w_jacobi = in_w_jacobi;
    packed_w_jacobi = _mm256_set1_ps(m_w_jacobi);
  }
  // gauss seidel updating the red points (i+j+k) even
  void set_red_GS() {
    m_mode = working_mode::RED_GS;
    init_invdiag();
  }
  // gauss seidel update the black points (i+j+k) odd
  void set_black_GS() {
    m_mode = working_mode::BLACK_GS;
    init_invdiag();
  }

  void set_input_lhs(openvdb::FloatGrid::Ptr in_lhs) { m_lhs = in_lhs; }
  void set_input_rhs(openvdb::FloatGrid::Ptr in_rhs) { m_rhs = in_rhs; }
  void set_output_result(openvdb::FloatGrid::Ptr out_result) {
    m_result = out_result;
  }

  // this operator is to be applied on the dof index grid
  // this ensures operating on the correct mask
  // the computation only operates on 8 float at a time
  // using avx intrinsics
  // each leaf contains 512 float
  // offset = 64*x+8*y+z
  // hence each bulk need to loop over x and y
  // make sure not to use SSE instructions here to avoid context switching cost
  void operator()(openvdb::Int32Tree::LeafNodeType &leaf,
                  openvdb::Index leafpos) const {
    const auto *rhs_leaf = m_rhs->tree().probeConstLeaf(leaf.origin());

    auto *result_leaf = m_result->tree().probeLeaf(leaf.origin());
    result_leaf->setValueMask(leaf.getValueMask());

    const auto *diagleaf = m_Diagonal->tree().probeConstLeaf(leaf.origin());
    const auto *invdiagleaf =
        m_inv_diagonal->tree().probeConstLeaf(leaf.origin());
    const auto *SPAI0leaf = m_SPAI0->tree().probeConstLeaf(leaf.origin());
    // the six neighbor leafs for the pressure
    // 0     1   2   3   4   5   6
    // self, xp, xm, yp, ym, zp, zm
    std::array<const simd_vdb_poisson::float_leaf_t *, 7> lhs_leaf;
    lhs_leaf.fill(nullptr);
    lhs_leaf[0] = m_lhs->tree().probeConstLeaf(leaf.origin());

    for (int i_f = 0; i_f < 6; i_f++) {
      int component = i_f / 2;
      int positive_direction = (i_f % 2 == 0);

      auto neib_origin = leaf.origin();
      if (positive_direction) {
        neib_origin[component] += 8;
      } else {
        neib_origin[component] -= 8;
      }
      lhs_leaf[size_t(i_f + 1)] = m_lhs->tree().probeConstLeaf(neib_origin);
    }

    // 0 self, 1 xp
    const simd_vdb_poisson::float_leaf_t *x_weight_leaf[2] = {nullptr, nullptr};
    x_weight_leaf[0] = m_Neg_x_entry->tree().probeConstLeaf(leaf.origin());
    x_weight_leaf[1] =
        m_Neg_x_entry->tree().probeConstLeaf(leaf.origin().offsetBy(8, 0, 0));

    // 0 self, 1 yp
    const simd_vdb_poisson::float_leaf_t *y_weight_leaf[2] = {nullptr, nullptr};
    y_weight_leaf[0] = m_Neg_y_entry->tree().probeConstLeaf(leaf.origin());
    y_weight_leaf[1] =
        m_Neg_y_entry->tree().probeConstLeaf(leaf.origin().offsetBy(0, 8, 0));

    // 0 self, 1 zp
    const simd_vdb_poisson::float_leaf_t *z_weight_leaf[2] = {nullptr, nullptr};
    z_weight_leaf[0] = m_Neg_z_entry->tree().probeConstLeaf(leaf.origin());
    z_weight_leaf[1] =
        m_Neg_z_entry->tree().probeConstLeaf(leaf.origin().offsetBy(0, 0, 8));

    // loop over 64 lanes to conduct the computation
    // laneid = x*8+y
    __m256 xp_lhs;
    __m256 xp_weight;
    __m256 xm_lhs;
    __m256 xm_weight;
    __m256 yp_lhs;
    __m256 yp_weight;
    __m256 ym_lhs;
    __m256 ym_weight;
    __m256 zp_lhs;
    __m256 zp_weight;
    __m256 zm_lhs;
    __m256 zm_weight;
    __m256 this_lhs;
    __m256 this_diag;
    __m256 this_rhs;
    __m256 residual;
    __m256 this_invdiag;
    std::array<__m256, 7> faced_result;
    uint32_t tempoffset;
    for (uint32_t laneoffset = 0; laneoffset < 512; laneoffset += 8) {
      const uint8_t lanemask =
          leaf.getValueMask().getWord<uint8_t>(laneoffset / 8);
      bool lane_even = ((laneoffset >> 3) + (laneoffset >> 6)) % 2 == 0;
      if (lanemask == uint8_t(0)) {
        // there is no diagonal entry in this lane
        continue;
      }

      // lane in the x plus directoin
      /****************************************************************************/

      if (laneoffset >= (7 * 64)) {
        // load the nextleaf
        tempoffset = laneoffset - (7 * 64);
        get_x_y_lane(xp_lhs, lhs_leaf[1], tempoffset, default_lhs);
        get_x_y_lane(xp_weight, x_weight_leaf[1], tempoffset,
                     default_face_term);

        // x-
        get_x_y_lane_unsafe(xm_lhs, lhs_leaf[0], laneoffset - 64, default_lhs);
      } else {
        tempoffset = laneoffset + 64;
        get_x_y_lane_unsafe(xp_lhs, lhs_leaf[0], tempoffset, default_lhs);
        get_x_y_lane(xp_weight, x_weight_leaf[0], tempoffset,
                     default_face_term);
        if (laneoffset < 64) {
          get_x_y_lane(xm_lhs, lhs_leaf[2], laneoffset + (7 * 64), default_lhs);
        } else {
          get_x_y_lane_unsafe(xm_lhs, lhs_leaf[0], laneoffset - 64,
                              default_lhs);
        }
      }
      faced_result[1] = _mm256_mul_ps(xp_lhs, xp_weight);
      // lane in the x minux direction
      /******************************************************************************/
      /*if (laneoffset < 64) {
              get_x_y_lane(xm_lhs, lhs_leaf[2], laneoffset + (7 * 64),
      default_lhs);
      }
      else {
              get_x_y_lane(xm_lhs, lhs_leaf[0], laneoffset - 64, default_lhs);
      }*/
      get_x_y_lane(xm_weight, x_weight_leaf[0], laneoffset, default_face_term);
      faced_result[2] = _mm256_mul_ps(xm_lhs, xm_weight);
      // lane in the y plus direction
      /****************************************************************************/
      // y==7 go to the next leaf and
      if ((laneoffset & 63u) == 56u) {
        tempoffset = laneoffset - 56;
        get_x_y_lane(yp_lhs, lhs_leaf[3], tempoffset, default_lhs);
        get_x_y_lane(yp_weight, y_weight_leaf[1], tempoffset,
                     default_face_term);
        // y-
        get_x_y_lane_unsafe(ym_lhs, lhs_leaf[0], laneoffset - 8, default_lhs);
      } else {
        tempoffset = laneoffset + 8;
        get_x_y_lane_unsafe(yp_lhs, lhs_leaf[0], tempoffset, default_lhs);
        get_x_y_lane(yp_weight, y_weight_leaf[0], tempoffset,
                     default_face_term);
        if ((laneoffset & 63) == 0) {
          get_x_y_lane(ym_lhs, lhs_leaf[4], laneoffset + 56, default_lhs);
        } else {
          get_x_y_lane_unsafe(ym_lhs, lhs_leaf[0], laneoffset - 8, default_lhs);
        }
      }
      faced_result[3] = _mm256_mul_ps(yp_lhs, yp_weight);
      // lane in the y minus direction
      /****************************************************************************/

      get_x_y_lane(ym_weight, y_weight_leaf[0], laneoffset, default_face_term);
      faced_result[4] = _mm256_mul_ps(ym_lhs, ym_weight);
      // lane in the z plus direction
      /****************************************************************************/
      get_pos_z_lane(zp_lhs, lhs_leaf[0], lhs_leaf[5], laneoffset, default_lhs);
      get_pos_z_lane(zp_weight, z_weight_leaf[0], z_weight_leaf[1], laneoffset,
                     default_face_term);
      faced_result[5] = _mm256_mul_ps(zp_lhs, zp_weight);
      // lane in the z minus direction
      /****************************************************************************/
      get_neg_z_lane(zm_lhs, lhs_leaf[0], lhs_leaf[6], laneoffset, default_lhs);
      get_x_y_lane(zm_weight, z_weight_leaf[0], laneoffset, default_face_term);
      // temp_result = _mm256_fmadd_ps(zm_lhs, zm_weight, temp_result);
      faced_result[6] = _mm256_mul_ps(zm_lhs, zm_weight);

      // collect
      faced_result[1] = _mm256_add_ps(faced_result[1], faced_result[2]);
      faced_result[3] = _mm256_add_ps(faced_result[3], faced_result[4]);
      faced_result[5] = _mm256_add_ps(faced_result[5], faced_result[6]);

      faced_result[1] = _mm256_add_ps(faced_result[1], faced_result[3]);
      faced_result[1] = _mm256_add_ps(faced_result[1], faced_result[5]);

      // now faced_result[1] contains all the off-diagonal results
      // collected form 1,2,3,4,5,6

      // faced_result[0] stores the result of A*lhs
      // faced_result[6] = _mm256_add_ps(faced_result[4], faced_result[6]);
      get_x_y_lane_unsafe(this_lhs, lhs_leaf[0], laneoffset, default_lhs);
      switch (m_mode) {
      case working_mode::BLACK_GS:
        get_x_y_lane_unsafe(this_rhs, rhs_leaf, laneoffset, default_rhs);
        residual = _mm256_sub_ps(this_rhs, faced_result[1]);
        get_x_y_lane(this_invdiag, invdiagleaf, laneoffset,
                     default_inv_diagonal);
        residual = _mm256_mul_ps(residual, this_invdiag);
        residual = _mm256_mul_ps(residual, packed_sor);
        faced_result[0] = _mm256_fmadd_ps(this_lhs, packed_1msor, residual);
        // blend result
        if (lane_even) {
          faced_result[0] =
              _mm256_blend_ps(this_lhs, faced_result[0], 0b10101010);
        } else {
          faced_result[0] =
              _mm256_blend_ps(this_lhs, faced_result[0], 0b01010101);
        }
        break;
      case working_mode::RED_GS:
        get_x_y_lane_unsafe(this_rhs, rhs_leaf, laneoffset, default_rhs);
        residual = _mm256_sub_ps(this_rhs, faced_result[1]);
        get_x_y_lane(this_invdiag, invdiagleaf, laneoffset,
                     default_inv_diagonal);
        residual = _mm256_mul_ps(residual, this_invdiag);
        residual = _mm256_mul_ps(residual, packed_sor);
        faced_result[0] = _mm256_fmadd_ps(this_lhs, packed_1msor, residual);
        // blend result
        if (lane_even) {
          faced_result[0] =
              _mm256_blend_ps(this_lhs, faced_result[0], 0b01010101);
        } else {
          faced_result[0] =
              _mm256_blend_ps(this_lhs, faced_result[0], 0b10101010);
        }
        break;
      case working_mode::WEIGHTED_JACOBI:
        // v^1 = v^0 + w D^-1 r
        // r = rhs - A*v^0
        // v^0 is the input lhs
        // v^1 is the output result
        // the residual
        // the diagonal term
        /****************************************************************************/
        get_x_y_lane(this_diag, diagleaf, laneoffset, default_diag);
        faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
        get_x_y_lane_unsafe(this_rhs, rhs_leaf, laneoffset, default_rhs);
        residual = _mm256_sub_ps(this_rhs, faced_result[0]);
        get_x_y_lane(this_invdiag, invdiagleaf, laneoffset,
                     default_inv_diagonal);
        residual = _mm256_mul_ps(residual, this_invdiag);
        faced_result[0] = _mm256_fmadd_ps(packed_w_jacobi, residual, this_lhs);
        break;
      case working_mode::SPAI0:
        // the diagonal term
        /****************************************************************************/
        get_x_y_lane(this_diag, diagleaf, laneoffset, default_diag);
        faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
        get_x_y_lane_unsafe(this_rhs, rhs_leaf, laneoffset, default_rhs);
        residual = _mm256_sub_ps(this_rhs, faced_result[0]);
        get_x_y_lane(this_invdiag, SPAI0leaf, laneoffset, default_SPAI0);
        faced_result[0] = _mm256_fmadd_ps(this_invdiag, residual, this_lhs);
        break;
      case working_mode::RESIDUAL:
        // the diagonal term
        /****************************************************************************/
        get_x_y_lane(this_diag, diagleaf, laneoffset, default_diag);
        faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
        get_x_y_lane_unsafe(this_rhs, rhs_leaf, laneoffset, default_rhs);
        faced_result[0] = _mm256_sub_ps(this_rhs, faced_result[0]);
        break;
      default:
      case working_mode::NORMAL_LAPLACIAN:
        // the diagonal term
        /****************************************************************************/
        get_x_y_lane(this_diag, diagleaf, laneoffset, default_diag);
        faced_result[0] = _mm256_fmadd_ps(this_lhs, this_diag, faced_result[1]);
        break;
      } // end case working mode
      // write to the result
      _mm256_storeu_ps(result_leaf->buffer().data() + laneoffset,
                       faced_result[0]);
    } // end laneid = [0 63]

    // set the off elements to zero
    if (result_leaf->isValueMaskOn()) {
      return;
    }
    for (auto offiter = result_leaf->beginValueOff(); offiter; ++offiter) {
      offiter.setValue(0);
    }

  } // end operator

  void get_x_y_lane(__m256 &out_lane,
                    const simd_vdb_poisson::float_leaf_t *leaf_ptr,
                    uint32_t load_offset, const __m256 &default_val) const {
    if (!leaf_ptr) {
      out_lane = default_val;
    } else {
      out_lane = _mm256_loadu_ps(leaf_ptr->buffer().data() + load_offset);
    }
  };
  void get_x_y_lane_unsafe(__m256 &out_lane,
                           const simd_vdb_poisson::float_leaf_t *leaf_ptr,
                           uint32_t load_offset, const __m256) const {
    out_lane = _mm256_loadu_ps(leaf_ptr->buffer().data() + load_offset);
  };
  // get a z lane that looks liks taking from a lane
  // that is shifted 1 element towards negative z direction
  // with respect to current leaf
  // for example, assume in a one dimensional storage
  //  neg z neib        this
  // | 0 1 2 ... 7 | 8 9 ... 15 | 16 ...
  // return the result of
  // | 7 8 9 ... 14 |
  // this is done by shuffle and blend
  // (1) first use avx instruction to load the previous lane
  // |0 1 2 3 4 5 6 7|
  // permute the above lane: using _mm256_permute_ps
  // |3 0 1 2 7 4 5 6|
  //
  // (2) load the middle m256 lane:
  // |8 9 10 11 12 13 14 15 |
  // permute each 128 lane: using _mm256_permute_ps
  // |11 8 9 10 15 12 13 14 |
  //
  // (3) we need to replace the 11 and 15 in the above result
  // to prepare for that, we extract the lower part of the above result:
  // we need the following lane:
  // |7 4 5 6 11 8 9 10 |
  // to get it we then blend it with the original using _mm256_permute2f128_ps
  //
  // (4) finally we use _mm256_blend_ps to blend the result
  // this function would only be called to retrieve the lhs leaf
  void get_neg_z_lane(__m256 &out_lane,
                      const simd_vdb_poisson::float_leaf_t *this_leaf,
                      const simd_vdb_poisson::float_leaf_t *neg_neib_z_leaf,
                      uint32_t lane256_offset,
                      const __m256 &default_val) const {
    // the beginning of the x y lane in a leaf

    // the data of this lane
    //| 8 9 ... 15 |
    __m256 original_this_lane;
    if (this_leaf) {
      original_this_lane =
          _mm256_loadu_ps(this_leaf->buffer().data() + lane256_offset);
    } else {
      original_this_lane = default_val;
      if (!neg_neib_z_leaf) {
        out_lane = default_val;
        return;
      }
    }

    // the data in the negative neighbor
    //|0 1 2 3 4 5 6 7|
    __m256 neg_neib_lane;
    if (neg_neib_z_leaf) {
      neg_neib_lane =
          _mm256_loadu_ps(neg_neib_z_leaf->buffer().data() + lane256_offset);
    } else {
      neg_neib_lane = default_val;
    }

    //(1),(2)
    // permute each lane, this happens within each 128 half of the 256 lane
    // lowz  highz
    //|0 1 2 3| -> |3 0 1 2|
    // hence imm8[1:0]=3
    // imm8[3:2] = 0
    // imm8[5:4] = 1
    // imm8[7:6] = 2
    // use _mm256_permute_ps
    const uint8_t permute_imm8 = 0b10010011;
    original_this_lane = _mm256_permute_ps(original_this_lane, permute_imm8);
    neg_neib_lane = _mm256_permute_ps(neg_neib_lane, permute_imm8);

    //(3)
    // the lower part of this is the high part of the permuted neg neiber
    // the higher part of this is the lower part of the permuted original
    // use _mm256_permute2f128_ps
    /*
    DEFINE SELECT4(src1, src2, control) {
    CASE(control[1:0]) OF
    0:	tmp[127:0] := src1[127:0]
    1:	tmp[127:0] := src1[255:128]
    2:	tmp[127:0] := src2[127:0]
    3:	tmp[127:0] := src2[255:128]
    ESAC
    IF control[3]
            tmp[127:0] := 0
    FI
    RETURN tmp[127:0]
    }
    dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
    dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
    dst[MAX:256] := 0
    */

    // we use src1 = neg neib
    //       src2 = original
    // lower 128 is high src1, imm8[3:0] = 1
    // higher 128 is low src2, imm8[7:4] = 2
    __m256 high_neg_neib_low_original;

    const uint8_t interweave_imm8 = 0b00100001;
    high_neg_neib_low_original = _mm256_permute2f128_ps(
        neg_neib_lane, original_this_lane, interweave_imm8);

    // blend the permuted original lane with the
    // the 0 and 4 position will be from the high_neg_neib
    // the rest bits are from the permuted original
    const uint8_t blend_imm8 = 0b00010001;
    out_lane = _mm256_blend_ps(original_this_lane, high_neg_neib_low_original,
                               blend_imm8);
  } // end get_neg_z_lane

  // get a z lane that looks liks taking from a lane
  // that is shifted 1 element towards positive z direction
  // with respect to current leaf
  // for example, assume in a one dimensional storage
  //      this     positive neib
  // | 0 1 2 ... 7 | 8 9 ... 15 | 16 ...
  // return the result of
  // | 1 2 3 4 5 6 7 8 |
  // this is done by shuffle and blend
  // (1) first use avx instruction to load the positive lane
  // |8 9 10 11 12 13 14 15|
  // permute the above lane: using _mm256_permute_ps
  // |9 10 11 8 15 12 13 14|
  //
  // (2) load the middle m256 lane:
  // | 0 1 2 3 4 5 6 7 |
  // permute each 128 lane: using _mm256_permute_ps
  // | 1 2 3 0 5 6 7 4 |
  //
  // (3) we need to replace the 0 and 4 in the above result
  // to prepare for that, we extract the upper part of the above result:
  // we need the following lane:
  // |5 6 7 4, 9 10 11 8 |
  //  high original, lower pos neib
  // to get it we then blend it with the original using _mm256_permute2f128_ps
  //
  // (4) finally we use _mm256_blend_ps to blend the result
  // this function would only be called to retrieve the lhs leaf
  inline void
  get_pos_z_lane(__m256 &out_lane,
                 const simd_vdb_poisson::float_leaf_t *this_leaf,
                 const simd_vdb_poisson::float_leaf_t *pos_neib_z_leaf,
                 uint32_t lane256_offset, const __m256 &default_val) const {

    // the data of this lane
    //| 8 9 ... 15 |
    __m256 original_this_lane;
    if (this_leaf) {
      original_this_lane =
          _mm256_loadu_ps(this_leaf->buffer().data() + lane256_offset);
    } else {
      original_this_lane = default_val;
      if (!pos_neib_z_leaf) {
        out_lane = default_val;
        return;
      }
    }

    // the data in the negative neighbor
    //|0 1 2 3 4 5 6 7|
    __m256 pos_neib_lane;
    if (pos_neib_z_leaf) {
      pos_neib_lane =
          _mm256_loadu_ps(pos_neib_z_leaf->buffer().data() + lane256_offset);
    } else {
      pos_neib_lane = default_val;
    }

    //(1),(2)
    // permute each lane, this happens within each 128 half of the 256 lane
    // lowz  highz
    //|0 1 2 3| -> |1 2 3 0|
    // hence imm8[1:0]=1
    // imm8[3:2] = 2
    // imm8[5:4] = 3
    // imm8[7:6] = 0
    // use _mm256_permute_ps
    const uint8_t permute_imm8 = 0b00111001;
    original_this_lane = _mm256_permute_ps(original_this_lane, permute_imm8);
    pos_neib_lane = _mm256_permute_ps(pos_neib_lane, permute_imm8);

    //(3)
    // the lower part of this is the high part of the permuted neg neiber
    // the higher part of this is the lower part of the permuted original
    // use _mm256_permute2f128_ps
    /*
    DEFINE SELECT4(src1, src2, control) {
    CASE(control[1:0]) OF
    0:	tmp[127:0] := src1[127:0]
    1:	tmp[127:0] := src1[255:128]
    2:	tmp[127:0] := src2[127:0]
    3:	tmp[127:0] := src2[255:128]
    ESAC
    IF control[3]
            tmp[127:0] := 0
    FI
    RETURN tmp[127:0]
    }
    dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
    dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
    dst[MAX:256] := 0
    */

    // we use src1 = original
    //       src2 = high neib
    // lower 128 is high src1 imm8[3:0] = 1
    // higher 128 is low src2, imm8[7:4] = 2
    __m256 high_original_low_pos_neib;

    //
    const uint8_t interweave_imm8 = 0b00100001;
    high_original_low_pos_neib = _mm256_permute2f128_ps(
        original_this_lane, pos_neib_lane, interweave_imm8);

    // blend the permuted original lane with the
    // the 3 and 7 position will be from the high_neg_neib
    // the rest bits are from the permuted original
    const uint8_t blend_imm8 = 0b10001000;
    out_lane = _mm256_blend_ps(original_this_lane, high_original_low_pos_neib,
                               blend_imm8);
  } // end get_pos_z_lane

  // over load new and delete for aligned allocation and free
  void *operator new(size_t memsize) {
    size_t ptr_alloc = sizeof(void *);
    size_t align_size = 32;
    size_t request_size = sizeof(simd_laplacian_apply_op) + align_size;
    size_t needed = ptr_alloc + request_size;

    void *alloc = ::operator new(needed);
    void *real_alloc =
        reinterpret_cast<void *>(reinterpret_cast<char *>(alloc) + ptr_alloc);
    void *ptr = std::align(align_size, sizeof(simd_laplacian_apply_op),
                           real_alloc, request_size);

    ((void **)ptr)[-1] = alloc; // save for delete calls to use
    return ptr;
  }

  void operator delete(void *ptr) {
    void *alloc = ((void **)ptr)[-1];
    ::operator delete(alloc);
  }

  // the default face term when the faces are all liquid
  // -dt_over_dxsqr
  float default_term;

  // to be set in the constructor
  // when the requested leaf doesn't exist
  __m256 default_lhs;
  __m256 default_rhs;
  __m256 default_face_term;
  __m256 default_diag;
  __m256 default_inv_diagonal;
  __m256 default_SPAI0;
  std::array<__m256, 7> m_zeros_for_result;

  // damped jacobi weight
  float m_w_jacobi;
  __m256 packed_w_jacobi;

  const float m_sor;
  __m256 packed_sor, packed_1msor;

  openvdb::FloatGrid::Ptr m_Diagonal;
  openvdb::FloatGrid::Ptr m_inv_diagonal;
  bool m_invdiag_initialized;
  openvdb::FloatGrid::Ptr m_SPAI0;
  bool m_spai_initialized;
  openvdb::FloatGrid::Ptr m_Neg_x_entry;
  openvdb::FloatGrid::Ptr m_Neg_y_entry;
  openvdb::FloatGrid::Ptr m_Neg_z_entry;

  openvdb::FloatGrid::Ptr m_rhs;
  openvdb::FloatGrid::Ptr m_lhs;
  openvdb::FloatGrid::Ptr m_result;
  working_mode m_mode;
}; // end simd_laplacian_apply_op

void simd_vdb_poisson::Laplacian_with_level::initialize_evaluators() {
  // set the leaf manager
  m_dof_leafmanager =
      std::make_unique<openvdb::tree::LeafManager<openvdb::Int32Tree>>(
          m_dof_idx->tree());

  auto zerovec = get_zero_vec_grid();

  m_laplacian_evaluator =
      std::shared_ptr<simd_laplacian_apply_op>(new simd_laplacian_apply_op(
          m_Diagonal, m_Neg_x_entry, m_Neg_y_entry, m_Neg_z_entry, zerovec,
          zerovec, zerovec,
          /*default term*/ m_dt / (m_dx_this_level * m_dx_this_level)));
  m_laplacian_evaluator->set_normal_mode();
}

openvdb::FloatGrid::Ptr
simd_vdb_poisson::Laplacian_with_level::apply(openvdb::FloatGrid::Ptr in_lhs) {
  _mm256_zeroall();

  auto result = get_zero_vec_grid();
  // set the input and out put
  m_laplacian_evaluator->set_input_lhs(in_lhs);
  m_laplacian_evaluator->set_output_result(result);
  // CSim::TimerMan::timer("Sim.step/vdbflip/mulvsimd/apply").start();
  // diag_leafman.foreach(apply_op);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);
  // CSim::TimerMan::timer("Sim.step/vdbflip/mulvsimd/apply").stop();
  _mm256_zeroall();
  return result;
}

void simd_vdb_poisson::Laplacian_with_level::residual_apply_assume_topo(
    openvdb::FloatGrid::Ptr in_out_residual, openvdb::FloatGrid::Ptr in_lhs,
    openvdb::FloatGrid::Ptr in_rhs) {
  m_laplacian_evaluator->set_residual_mode();
  _mm256_zeroupper();
  m_laplacian_evaluator->set_input_lhs(in_lhs);
  m_laplacian_evaluator->set_input_rhs(in_rhs);
  m_laplacian_evaluator->set_output_result(in_out_residual);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);
  _mm256_zeroupper();
}

void simd_vdb_poisson::Laplacian_with_level::Laplacian_apply_assume_topo(
    openvdb::FloatGrid::Ptr in_out_result, openvdb::FloatGrid::Ptr in_lhs) {
  m_laplacian_evaluator->set_normal_mode();
  _mm256_zeroupper();
  m_laplacian_evaluator->set_input_lhs(in_lhs);
  m_laplacian_evaluator->set_input_rhs(in_lhs);
  m_laplacian_evaluator->set_output_result(in_out_result);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);
  _mm256_zeroupper();
}

void simd_vdb_poisson::Laplacian_with_level::weighted_jacobi_apply_assume_topo(
    openvdb::FloatGrid::Ptr in_out_updated_lhs, openvdb::FloatGrid::Ptr in_lhs,
    openvdb::FloatGrid::Ptr in_rhs) {
  m_laplacian_evaluator->set_jacobi_mode();
  _mm256_zeroupper();
  m_laplacian_evaluator->set_input_lhs(in_lhs);
  m_laplacian_evaluator->set_input_rhs(in_rhs);
  m_laplacian_evaluator->set_output_result(in_out_updated_lhs);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);
  _mm256_zeroupper();
}
void simd_vdb_poisson::Laplacian_with_level::SPAI0_apply_assume_topo(
    openvdb::FloatGrid::Ptr in_out_updated_lhs, openvdb::FloatGrid::Ptr in_lhs,
    openvdb::FloatGrid::Ptr in_rhs) {
  m_laplacian_evaluator->set_SPAI0_mode();
  _mm256_zeroupper();
  m_laplacian_evaluator->set_input_lhs(in_lhs);
  m_laplacian_evaluator->set_input_rhs(in_rhs);
  m_laplacian_evaluator->set_output_result(in_out_updated_lhs);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);
  _mm256_zeroupper();
}
template <bool red_first>
void simd_vdb_poisson::Laplacian_with_level::RBGS_apply_assume_topo_inplace(
    openvdb::FloatGrid::Ptr scratch_pad, openvdb::FloatGrid::Ptr in_out_lhs,
    openvdb::FloatGrid::Ptr in_rhs) {
  openvdb::FloatGrid::Ptr output_updated_lhs = scratch_pad;
  _mm256_zeroupper();
  if (red_first) {
    m_laplacian_evaluator->set_red_GS();
  } else {
    m_laplacian_evaluator->set_black_GS();
  }
  m_laplacian_evaluator->set_input_lhs(in_out_lhs);
  m_laplacian_evaluator->set_input_rhs(in_rhs);
  m_laplacian_evaluator->set_output_result(scratch_pad);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);

  // the updated lhs has now red points updated
  if (red_first) {
    m_laplacian_evaluator->set_black_GS();
  } else {
    m_laplacian_evaluator->set_red_GS();
  }
  m_laplacian_evaluator->set_input_lhs(scratch_pad);
  m_laplacian_evaluator->set_output_result(in_out_lhs);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);

  // now the in_out_lhs is updated, let's copy the result to the output
  auto copy_to_result = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                            openvdb::Index leafpos) {
    auto *source_leaf = in_out_lhs->tree().probeLeaf(leaf.origin());
    auto *target_leaf = output_updated_lhs->tree().probeLeaf(leaf.origin());

    std::copy(source_leaf->buffer().data(),
              source_leaf->buffer().data() + leaf.SIZE,
              target_leaf->buffer().data());
  }; // end copy to result

  // m_dof_leafmanager->foreach(copy_to_result);
  _mm256_zeroupper();
}

void simd_vdb_poisson::Laplacian_with_level::inplace_add_assume_topo(
    openvdb::FloatGrid::Ptr in_out_result, openvdb::FloatGrid::Ptr in_rhs) {
  // assume the input has the same topology
  _mm256_zeroall();

  auto set_first_iter_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto *result_leaf = in_out_result->tree().probeLeaf(leaf.origin());
    const auto *rhs_leaf = in_rhs->tree().probeConstLeaf(leaf.origin());

    __m256 result_256;
    __m256 rhs_256;
    for (auto laneoffset = 0; laneoffset < leaf.SIZE; laneoffset += 8) {
      result_256 = _mm256_loadu_ps(result_leaf->buffer().data() + laneoffset);
      rhs_256 = _mm256_loadu_ps(rhs_leaf->buffer().data() + laneoffset);
      result_256 = _mm256_add_ps(rhs_256, result_256);
      _mm256_storeu_ps(result_leaf->buffer().data() + laneoffset, result_256);
    }

    for (auto iter = result_leaf->beginValueOff(); iter; ++iter) {
      iter.setValue(0);
    }
  }; // end set_first_iter_op

  m_dof_leafmanager->foreach (set_first_iter_op);

  _mm256_zeroall();
}

openvdb::FloatGrid::Ptr
simd_vdb_poisson::Laplacian_with_level::get_zero_vec_grid() {
  openvdb::FloatGrid::Ptr result = openvdb::FloatGrid::create(0);
  result->setTree(std::make_shared<openvdb::FloatTree>(
      m_dof_idx->tree(), 0.f, openvdb::TopologyCopy()));
  result->setTransform(m_dof_idx->transformPtr());
  return result;
}

void simd_vdb_poisson::Laplacian_with_level::set_grid_constant_assume_topo(
    openvdb::FloatGrid::Ptr in_out_grid, float constant) {
  openvdb::FloatTree::LeafNodeType zeroleaf;
  zeroleaf.fill(constant);

  // it is assumed that the
  auto set_constant_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                             openvdb::Index leafpos) {
    auto *out_leaf = in_out_grid->tree().probeLeaf(leaf.origin());
    std::copy(zeroleaf.buffer().data(),
              zeroleaf.buffer().data() + zeroleaf.SIZE,
              out_leaf->buffer().data());
    for (auto iter = out_leaf->beginValueOff(); iter; ++iter) {
      iter.setValue(0);
    }
  };

  m_dof_leafmanager->foreach (set_constant_op);
}

void simd_vdb_poisson::Laplacian_with_level::
    set_grid_to_result_after_first_jacobi_assume_topo(
        openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs) {
  // assume the input has the same topology
  // apply weighted jacobi iteration to result = (L^-1) rhs
  // assume the initial guess of result was 0
  // This amounts to v^1 = w*D^-1(b-A*0)=w*(invdiag)*rhs
  float w = m_laplacian_evaluator->m_w_jacobi;
  _mm256_zeroall();

  auto set_first_iter_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto *result_leaf = in_out_grid->tree().probeLeaf(leaf.origin());
    const auto *rhs_leaf = in_rhs->tree().probeConstLeaf(leaf.origin());
    const auto *invdiag_leav =
        m_laplacian_evaluator->m_inv_diagonal->tree().probeConstLeaf(
            leaf.origin());
    __m256 default_invdiag = m_laplacian_evaluator->default_inv_diagonal;
    __m256 packed_w = m_laplacian_evaluator->packed_w_jacobi;

    __m256 result_256;
    __m256 this_invdiag;
    for (auto laneoffset = 0; laneoffset < leaf.SIZE; laneoffset += 8) {
      result_256 = _mm256_loadu_ps(rhs_leaf->buffer().data() + laneoffset);
      m_laplacian_evaluator->get_x_y_lane(this_invdiag, invdiag_leav,
                                          laneoffset, default_invdiag);
      result_256 = _mm256_mul_ps(this_invdiag, result_256);
      result_256 = _mm256_mul_ps(packed_w, result_256);
      _mm256_storeu_ps(result_leaf->buffer().data() + laneoffset, result_256);
    }

    for (auto iter = result_leaf->beginValueOff(); iter; ++iter) {
      iter.setValue(0);
    }
  }; // end set_first_iter_op

  m_dof_leafmanager->foreach (set_first_iter_op);

  _mm256_zeroall();
}

void simd_vdb_poisson::Laplacian_with_level::
    set_grid_to_result_after_first_SPAI_assume_topo(
        openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs) {
  // assume the input has the same topology
  // apply SPAI iteration to result = (L^-1) rhs
  // assume the initial guess of result was 0
  // This amounts to v^1 =SPAI0(b-A*0)=(SPAI0)*rhs
  _mm256_zeroall();

  auto set_first_iter_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto *result_leaf = in_out_grid->tree().probeLeaf(leaf.origin());
    const auto *rhs_leaf = in_rhs->tree().probeConstLeaf(leaf.origin());
    const auto *SPAI0_leaf =
        m_laplacian_evaluator->m_inv_diagonal->tree().probeConstLeaf(
            leaf.origin());
    __m256 default_SPAI0 = m_laplacian_evaluator->default_SPAI0;
    __m256 packed_w = m_laplacian_evaluator->packed_w_jacobi;

    __m256 result_256;
    __m256 SPAI0_256;
    for (auto laneoffset = 0; laneoffset < leaf.SIZE; laneoffset += 8) {
      result_256 = _mm256_loadu_ps(rhs_leaf->buffer().data() + laneoffset);
      m_laplacian_evaluator->get_x_y_lane(SPAI0_256, SPAI0_leaf, laneoffset,
                                          default_SPAI0);
      result_256 = _mm256_mul_ps(SPAI0_256, result_256);
      _mm256_storeu_ps(result_leaf->buffer().data() + laneoffset, result_256);
    }

    for (auto iter = result_leaf->beginValueOff(); iter; ++iter) {
      iter.setValue(0);
    }
  }; // end set_first_iter_op

  m_dof_leafmanager->foreach (set_first_iter_op);

  _mm256_zeroall();
}

template <bool red_first>
void simd_vdb_poisson::Laplacian_with_level::
    set_grid_to_result_after_first_RBGS_assume_topo(
        openvdb::FloatGrid::Ptr in_out_grid, openvdb::FloatGrid::Ptr in_rhs) {
  // assume the initial guess is zero
  // then the first iteration only get red dof updated
  _mm256_zeroall();

  auto set_first_iter_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto *result_leaf = in_out_grid->tree().probeLeaf(leaf.origin());
    const auto *rhs_leaf = in_rhs->tree().probeConstLeaf(leaf.origin());
    const auto *invdiag_leav =
        m_laplacian_evaluator->m_inv_diagonal->tree().probeConstLeaf(
            leaf.origin());
    __m256 default_invdiag = m_laplacian_evaluator->default_inv_diagonal;
    __m256 packed_sor = m_laplacian_evaluator->packed_sor;

    __m256 result_256;
    __m256 this_invdiag;
    for (auto laneoffset = 0; laneoffset < leaf.SIZE; laneoffset += 8) {
      bool lane_even = ((laneoffset >> 3) + (laneoffset >> 6)) % 2 == 0;
      result_256 = _mm256_loadu_ps(rhs_leaf->buffer().data() + laneoffset);
      m_laplacian_evaluator->get_x_y_lane(this_invdiag, invdiag_leav,
                                          laneoffset, default_invdiag);
      result_256 = _mm256_mul_ps(this_invdiag, result_256);
      result_256 = _mm256_mul_ps(packed_sor, result_256);

      if (red_first) {
        // only update
        if (lane_even) {
          result_256 =
              _mm256_blend_ps(_mm256_setzero_ps(), result_256, 0b01010101);
        } else {
          result_256 =
              _mm256_blend_ps(_mm256_setzero_ps(), result_256, 0b10101010);
        }
      } else {
        if (lane_even) {
          result_256 =
              _mm256_blend_ps(_mm256_setzero_ps(), result_256, 0b10101010);
        } else {
          result_256 =
              _mm256_blend_ps(_mm256_setzero_ps(), result_256, 0b01010101);
        }
      }
      _mm256_storeu_ps(result_leaf->buffer().data() + laneoffset, result_256);
    }

    for (auto iter = result_leaf->beginValueOff(); iter; ++iter) {
      iter.setValue(0);
    }
  }; // end set_first_iter_op

  m_dof_leafmanager->foreach (set_first_iter_op);

  // try inplace?
  if (red_first) {
    m_laplacian_evaluator->set_black_GS();
  } else {
    m_laplacian_evaluator->set_red_GS();
  }
  m_laplacian_evaluator->set_input_lhs(in_out_grid);
  m_laplacian_evaluator->set_output_result(in_out_grid);
  m_laplacian_evaluator->set_input_rhs(in_rhs);
  m_dof_leafmanager->foreach (*m_laplacian_evaluator);

  _mm256_zeroall();
}

void simd_vdb_poisson::Laplacian_with_level::grid2vector(
    std::vector<float> &out_vector, openvdb::FloatGrid::Ptr in_grid) {
  out_vector.resize(m_ndof);

  // it is assumed that the input grid has exactly the same dof as the
  auto dof_leafman =
      openvdb::tree::LeafManager<openvdb::Int32Tree>(m_dof_idx->tree());
  auto set_result = [&](int_leaf_t &leaf, openvdb::Index leafpos) {
    auto *val_leaf = in_grid->tree().probeConstLeaf(leaf.origin());
    if (!val_leaf) {
      printf("input grid doesn't have a leaf that the dof idx tree has\n");
      exit(-1);
    }
    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
      if (leaf.isValueOn(offset)) {
        out_vector[leaf.getValue(offset)] = val_leaf->getValue(offset);
      }
    }
  };
  dof_leafman.foreach (set_result);
}

void simd_vdb_poisson::Laplacian_with_level::vector2grid(
    openvdb::FloatGrid::Ptr out_grid, const std::vector<float> &in_vector) {
  out_grid->setTree(std::make_shared<openvdb::FloatTree>(
      m_dof_idx->tree(), 0.f, openvdb::TopologyCopy()));
  out_grid->setTransform(m_Diagonal->transformPtr());
  vector2grid_assume_topo(out_grid, in_vector);
}

void simd_vdb_poisson::Laplacian_with_level::vector2grid_assume_topo(
    openvdb::FloatGrid::Ptr out_grid, const std::vector<float> &in_vector) {
  // it is assumed that the input grid has exactly the same dof as the
  auto dof_leafman =
      openvdb::tree::LeafManager<openvdb::Int32Tree>(m_dof_idx->tree());
  auto set_result = [&](int_leaf_t &leaf, openvdb::Index leafpos) {
    auto *val_leaf = out_grid->tree().probeLeaf(leaf.origin());
    if (!val_leaf) {
      printf("input grid doesn't have a leaf that the dof idx tree has\n");
      exit(-1);
    }
    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
      if (leaf.isValueOn(offset)) {
        val_leaf->setValueOn(offset, in_vector[leaf.getValue(offset)]);
      }
    }
  };
  dof_leafman.foreach (set_result);
}

void simd_vdb_poisson::Laplacian_with_level::restriction(
    openvdb::FloatGrid::Ptr out_coarse_grid,
    openvdb::FloatGrid::Ptr in_fine_grid, const Laplacian_with_level &child) {
  if (child.m_root_token != m_root_token) {
    printf("restriction error, level0 mismatch\n");
    exit(-1);
  }
  if (child.m_level != (m_level + 1)) {
    printf("not restrict to a child\n");
    exit(-1);
  }
  bool topology_check = false;
  if (topology_check) {
    if (!in_fine_grid->tree().hasSameTopology(m_dof_idx->tree())) {
      printf("input fine grid does not match this level\n");
      exit(-1);
    }
    if (!out_coarse_grid->tree().hasSameTopology(child.m_dof_idx->tree())) {
      printf("output coarse grid does not match chile level\n");
      exit(-1);
    }
  }

  // to be use by the dof idx manager at coarse level, the child level
  auto collect_from_fine = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto *coarse_leaf = out_coarse_grid->tree().probeLeaf(leaf.origin());
    // fine voxel:   -4 -3 -2 -1 0 1 2 3 4 5
    // coarse voxel: -2 -2 -1 -1 0 0 1 1 2 2

    // each coarse leaf corresponds to 8 potential fine leaves that are active
    std::array<const openvdb::FloatTree::LeafNodeType *, 8> fine_leaves{
        nullptr};
    int fine_leaves_counter = 0;
    auto fine_base_origin = openvdb::Coord(leaf.origin().asVec3i() * 2);
    for (int ii = 0; ii < 16; ii += 8) {
      for (int jj = 0; jj < 16; jj += 8) {
        for (int kk = 0; kk < 16; kk += 8) {
          fine_leaves[fine_leaves_counter++] =
              in_fine_grid->tree().probeConstLeaf(
                  fine_base_origin.offsetBy(ii, jj, kk));
        }
      }
    }

    for (auto iter = coarse_leaf->beginValueOn(); iter; ++iter) {
      // uint32_t at_fine_leaf = iter.offset();

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

      // if there is possibly a dof in the fine leaf
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
            } // kk
          }   // jj
        }     // ii
        iter.setValue(temp_sum * 0.125f);
      } // if fine leaf
    }   // for all coarse on voxels
  };    // end collect from fine

  child.m_dof_leafmanager->foreach (collect_from_fine);
}

template <bool inplace_add>
void simd_vdb_poisson::Laplacian_with_level::prolongation(
    openvdb::FloatGrid::Ptr out_fine_grid,
    openvdb::FloatGrid::Ptr in_coarse_grid,
    const Laplacian_with_level &parent) {
  if (parent.m_root_token != m_root_token) {
    printf("prolongation error, level0 mismatch\n");
    exit(-1);
  }
  if (parent.m_level != (m_level - 1)) {
    printf("prolongation error, not to a parent\n");
    exit(-1);
  }
  bool topology_check = false;
  if (topology_check) {
    if (!in_coarse_grid->tree().hasSameTopology(m_dof_idx->tree())) {
      printf("input coarse grid does not match this level\n");
      exit(-1);
    }
    if (!out_fine_grid->tree().hasSameTopology(parent.m_dof_idx->tree())) {
      printf("output fine grid does not match parent level\n");
      exit(-1);
    }
  }

  // to be used by the fine dof idx manager, which is the parent
  auto collect_from_coarse = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                                 openvdb::Index leafpos) {
    const auto *coarse_leaf = in_coarse_grid->tree().probeConstLeaf(
        openvdb::Coord(leaf.origin().asVec3i() / 2));
    auto *fine_leaf = out_fine_grid->tree().probeLeaf(leaf.origin());

    auto base_coarse_voxel = openvdb::Coord(leaf.origin().asVec3i() / 2);
    for (auto iter = fine_leaf->beginValueOn(); iter; ++iter) {
      auto coarse_shift_coord =
          fine_leaf->offsetToLocalCoord(iter.offset()).asVec3i() / 2;
      iter.setValue(coarse_leaf->getValue(
          openvdb::Coord(base_coarse_voxel + coarse_shift_coord)));
    }
  }; // end collect from coarse

  // parent.m_dof_leafmanager->foreach(collect_from_coarse);

  // to be use by the dof idx manager at coarse level, the child level
  auto scatter_to_fine = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                             openvdb::Index leafpos) {
    const auto *coarse_leaf =
        in_coarse_grid->tree().probeConstLeaf(leaf.origin());
    // fine voxel:   -4 -3 -2 -1 0 1 2 3 4 5
    // coarse voxel: -2 -2 -1 -1 0 0 1 1 2 2

    // each coarse leaf corresponds to 8 potential fine leaves that are active
    std::array<openvdb::FloatTree::LeafNodeType *, 8> fine_leaves{nullptr};
    int fine_leaves_counter = 0;
    auto fine_base_origin = openvdb::Coord(leaf.origin().asVec3i() * 2);
    for (int ii = 0; ii < 16; ii += 8) {
      for (int jj = 0; jj < 16; jj += 8) {
        for (int kk = 0; kk < 16; kk += 8) {
          fine_leaves[fine_leaves_counter++] = out_fine_grid->tree().probeLeaf(
              fine_base_origin.offsetBy(ii, jj, kk));
        }
      }
    }

    for (auto iter = coarse_leaf->cbeginValueOn(); iter; ++iter) {
      // uint32_t at_fine_leaf = iter.offset();

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
      // if there is possibly a dof in the fine leaf
      if (auto fine_leaf = fine_leaves[at_fine_leaf]) {
        auto fine_base_voxel = openvdb::Coord(iter.getCoord().asVec3i() * 2);
        auto fine_base_offset = fine_leaf->coordToOffset(fine_base_voxel);
        for (int ii = 0; ii < 2; ii++) {
          for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 2; kk++) {
              auto fine_offset = fine_base_offset + kk;
              if (ii)
                fine_offset += 64;
              if (jj)
                fine_offset += 8;
              if (fine_leaf->isValueOn(fine_offset)) {
                if (inplace_add) {
                  fine_leaf->buffer().data()[fine_offset] += coarseval;
                } else {
                  fine_leaf->setValueOnly(fine_offset, coarseval);
                }
              }
            } // kk
          }   // jj
        }     // ii
        // iter.setValue(temp_sum * 0.125f);
      } // if fine leaf
    }   // for all coarse on voxels
  };    // end collect from fine

  m_dof_leafmanager->foreach (scatter_to_fine);
}

void simd_vdb_poisson::Laplacian_with_level::trim_default_nodes() {
  // if a leaf node has the same value and equal to the default
  // poisson equation term, remove the node
  float term = -m_dt / (m_dx_this_level * m_dx_this_level);
  float diag_term = -6.0f * term;
  float threshold = diag_term * 1e-5f;

  trim_default_nodes(m_Diagonal, diag_term, diag_term * 1e-5);
  trim_default_nodes(m_Neg_x_entry, term, term * 1e-5);
  trim_default_nodes(m_Neg_y_entry, term, term * 1e-5);
  trim_default_nodes(m_Neg_z_entry, term, term * 1e-5);
}

void simd_vdb_poisson::Laplacian_with_level::trim_default_nodes(
    openvdb::FloatGrid::Ptr in_out_grid, float default_value, float epsilon) {
  std::vector<openvdb::Coord> leaf_origins;
  std::vector<bool> quasi_uniform;
  epsilon = std::abs(epsilon);
  auto leafman =
      openvdb::tree::LeafManager<openvdb::FloatTree>(in_out_grid->tree());
  leaf_origins.resize(leafman.leafCount());
  quasi_uniform.resize(leafman.leafCount(), false);

  auto determine_uniform = [&](float_leaf_t &leaf, openvdb::Index leafpos) {
    leaf_origins[leafpos] = leaf.origin();
    if (!leaf.isValueMaskOn()) {
      quasi_uniform[leafpos] = false;
      return;
    }

    float max_error = 0;
    for (auto offset = 0; offset < 512; ++offset) {
      float curr_error = std::abs(leaf.getValue(offset) - default_value);
      if (curr_error > max_error) {
        max_error = curr_error;
      }
    }
    if (max_error < epsilon) {
      quasi_uniform[leafpos] = true;
    }
  }; // end determine uniform

  leafman.foreach (determine_uniform);

  for (auto i = 0; i < leaf_origins.size(); ++i) {
    if (quasi_uniform[i]) {
      delete in_out_grid->tree().stealNode<float_leaf_t>(leaf_origins[i],
                                                         default_value, false);
    }
  }
}

void simd_vdb_poisson::construct_levels() {
  // build the first level
  Laplacian_with_level::Ptr level0 = std::make_shared<Laplacian_with_level>(
      m_liquid_sdf, m_face_weight, dt, m_dx);

  m_laplacian_with_levels.push_back(level0);

  int max_dof = 4000;
  while (m_laplacian_with_levels.back()->m_ndof > max_dof) {
    Laplacian_with_level::Ptr next_level =
        std::make_shared<Laplacian_with_level>(
            *m_laplacian_with_levels.back(),
            Laplacian_with_level::coarsening());
    m_laplacian_with_levels.push_back(next_level);
  }

  // the scratchpad for the v cycle to avoid
  for (int level = 0; level < m_laplacian_with_levels.size(); level++) {
    // the solution at each level
    m_v_cycle_lhss.push_back(
        m_laplacian_with_levels[level]->get_zero_vec_grid());
    // the right hand side at each level
    m_v_cycle_rhss.push_back(
        m_laplacian_with_levels[level]->get_zero_vec_grid());
    // the temporary result to store the jacobi iteration
    // use std::shared_ptr::swap to change the content
    m_v_cycle_temps.push_back(
        m_laplacian_with_levels[level]->get_zero_vec_grid());
    if (level > 0) {
      m_K_cycle_cs.push_back(
          m_laplacian_with_levels[level]->get_zero_vec_grid());
      m_K_cycle_vs.push_back(
          m_laplacian_with_levels[level]->get_zero_vec_grid());
      m_K_cycle_ds.push_back(
          m_laplacian_with_levels[level]->get_zero_vec_grid());
      m_K_cycle_ws.push_back(
          m_laplacian_with_levels[level]->get_zero_vec_grid());
    } else {
      // the finest level is never used in Kcycle
      m_K_cycle_cs.push_back(openvdb::FloatGrid::create());
      m_K_cycle_vs.push_back(openvdb::FloatGrid::create());
      m_K_cycle_ds.push_back(openvdb::FloatGrid::create());
      m_K_cycle_ws.push_back(openvdb::FloatGrid::create());
    }
  }
  construct_coarsest_exact_solver();
  printf("levels: %zd Dof:%d\n", m_laplacian_with_levels.size(),
         m_laplacian_with_levels[0]->m_ndof);
}

void simd_vdb_poisson::Vcycle(const openvdb::FloatGrid::Ptr in_out_lhs,
                              const openvdb::FloatGrid::Ptr in_rhs, int n1,
                              int n2, int ncoarse) {
  bool topology_check = false;
  if (topology_check) {
    if (!in_out_lhs->tree().hasSameTopology(
            m_laplacian_with_levels[0]->m_dof_idx->tree())) {
      printf("Vcyel lhs grid does not match level 0\n");
      exit(-1);
    }
    if (!in_rhs->tree().hasSameTopology(
            m_laplacian_with_levels[0]->m_dof_idx->tree())) {
      printf("Vcyel rhs grid does not match level 0\n");
      exit(-1);
    }
  }
  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V").start();
  size_t nlevel = m_laplacian_with_levels.size();
  m_v_cycle_lhss[0] = in_out_lhs;
  m_v_cycle_rhss[0] = in_rhs;

  // on the finest level, relax a few iterations
  // in general in_out_lhs as the initial guess is not zero vector
  for (int i = 0; i < n1; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth0").start();
    m_laplacian_with_levels[0]->RBGS_apply_assume_topo_inplace<true>(
        m_v_cycle_temps[0], m_v_cycle_lhss[0], m_v_cycle_rhss[0]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth0").stop();
    // the updated lhs is in m_v_cycle_temps
    // we need to update the v_cycle_lhs
    // m_v_cycle_lhss[0].swap(m_v_cycle_temps[0]);
    // now the lhs is updated
  }

  // pass the residual to the next level
  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/residual0").start();
  m_laplacian_with_levels[0]->residual_apply_assume_topo(
      m_v_cycle_temps[0], m_v_cycle_lhss[0], m_v_cycle_rhss[0]);
  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/residual0").stop();

  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/R0").start();
  m_laplacian_with_levels[0]->restriction(m_v_cycle_rhss[1], m_v_cycle_temps[0],
                                          *m_laplacian_with_levels[1]);
  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/R0").stop();
  // pass get the residual, restrict the residual to the next level
  for (int level = 1; level < (nlevel - 1); level++) {
    // relax
    // the initial of the lhs for this level was supposed to be 0
    // but passing 0 to one iteration of weighted jacobi
    // gives weighted right handside
    m_laplacian_with_levels[level]
        ->set_grid_to_result_after_first_RBGS_assume_topo(
            m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    /*m_laplacian_with_levels[level]->set_grid_constant_assume_topo(
            m_v_cycle_lhss[level], 0);*/
    for (int i = 0; i < n1; i++) {
      // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth"+std::to_string(level)).start();
      m_laplacian_with_levels[level]->RBGS_apply_assume_topo_inplace<true>(
          m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
      // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth" +
      // std::to_string(level)).stop();
      // m_v_cycle_lhss[level].swap(m_v_cycle_temps[level]);
    }
    // calculate residual
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/residual" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->residual_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/residual" +
    // std::to_string(level)).stop();

    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/R" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->restriction(
        m_v_cycle_rhss[size_t(level + 1)], m_v_cycle_temps[level],
        *m_laplacian_with_levels[size_t(level + 1)]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/R" +
    // std::to_string(level)).stop(); pass to next level
  } // end for 0 to the last second level

  // at the coarsest level
  int level = nlevel - 1;

  bool use_iter = true;

  if (nlevel != m_laplacian_with_levels.size()) {
    use_iter = true;
  }

  if (use_iter) {
    m_laplacian_with_levels[level]
        ->set_grid_to_result_after_first_jacobi_assume_topo(
            m_v_cycle_lhss[level], m_v_cycle_rhss[level]);

    /*m_laplacian_with_levels[level]->set_grid_constant_assume_topo(
            m_v_cycle_lhss[level], 0);*/

    for (int i = 0; i < ncoarse; i++) {
      m_laplacian_with_levels[level]->RBGS_apply_assume_topo_inplace<true>(
          m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
      // m_v_cycle_lhss[level].swap(m_v_cycle_temps[level]);
    }
  } else {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/Direct").start();
    write_coarsest_eigen_rhs(m_coarsest_eigen_rhs, m_v_cycle_rhss[level]);
    m_coarsest_eigen_solution = m_coarsest_solver->solve(m_coarsest_eigen_rhs);
    write_coarsest_grid_solution(m_v_cycle_lhss[level],
                                 m_coarsest_eigen_solution);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/Direct").stop();
  }

  // get the error correction back
  for (level = nlevel - 2; level >= 0; level--) {
    int child_level = level + 1;
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/P" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[child_level]->prolongation</*inplace add*/ true>(
        m_v_cycle_lhss[level], m_v_cycle_lhss[child_level],
        *m_laplacian_with_levels[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/P" +
    // std::to_string(level)).stop();
    /*m_laplacian_with_levels[level]->inplace_add_assume_topo(
            m_v_cycle_lhss[level], m_v_cycle_temps[level]);*/

    for (int i = 0; i < n2; i++) {
      // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth" +
      // std::to_string(level)).start();
      m_laplacian_with_levels[level]->RBGS_apply_assume_topo_inplace<false>(
          m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
      // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth" +
      // std::to_string(level)).stop();
      // m_v_cycle_lhss[level].swap(m_v_cycle_temps[level]);
    }
  }

  // at this point the m_v_cycle_lhss[0] contains the solution
  // we need to write it back to the input
  // the input tree is among the m_v_cycle_lhss and m_v_cycle_temps
  // and is already modified multiple times
  // it's worthless to to distinguish, just copy.
  auto copy_to_result = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                            openvdb::Index leafpos) {
    auto *source_leaf = m_v_cycle_lhss[0]->tree().probeLeaf(leaf.origin());
    auto *target_leaf = in_out_lhs->tree().probeLeaf(leaf.origin());

    std::copy(source_leaf->buffer().data(),
              source_leaf->buffer().data() + leaf.SIZE,
              target_leaf->buffer().data());
  }; // end copy to result

  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/copy").start();
  // m_laplacian_with_levels[0]->m_dof_leafmanager->foreach(copy_to_result);
  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/copy").stop();
  if ((n1 + n2) % 2 == 1) {
    // m_v_cycle_lhss[0].swap(m_v_cycle_temps[0]);
  }
  // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V").stop();
}

template <int mu_time, bool skip_first_iter>
void simd_vdb_poisson::mucycle_RBGS(const openvdb::FloatGrid::Ptr in_out_lhs,
                                    const openvdb::FloatGrid::Ptr in_rhs,
                                    const int level, int n1, int n2) {
  size_t nlevel = m_laplacian_with_levels.size();
  if (level == nlevel - 1) {
    write_coarsest_eigen_rhs(m_coarsest_eigen_rhs, in_rhs);
    m_coarsest_eigen_solution = m_coarsest_solver->solve(m_coarsest_eigen_rhs);
    write_coarsest_grid_solution(in_out_lhs, m_coarsest_eigen_solution);
    return;
  }

  m_v_cycle_lhss[level] = in_out_lhs;
  m_v_cycle_rhss[level] = in_rhs;

  if (skip_first_iter) {
    // set result after first iter, with zero lhs guess
    m_laplacian_with_levels[level]
        ->set_grid_to_result_after_first_RBGS_assume_topo(
            m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
  }

  for (int i = (skip_first_iter ? 1 : 0); i < n1; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->RBGS_apply_assume_topo_inplace<true>(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
  }
  m_laplacian_with_levels[level]->residual_apply_assume_topo(
      m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);

  int child_level = level + 1;
  m_laplacian_with_levels[level]->restriction(
      m_v_cycle_rhss[child_level], m_v_cycle_temps[level],
      /*child level*/ *m_laplacian_with_levels[child_level]);

  // set the initial guess for the next level
  m_laplacian_with_levels[child_level]
      ->set_grid_to_result_after_first_RBGS_assume_topo(
          m_v_cycle_lhss[child_level], m_v_cycle_rhss[child_level]);
  mucycle_RBGS<mu_time, true>(m_v_cycle_lhss[child_level],
                              m_v_cycle_rhss[child_level], child_level, n1, n2);
  for (int mu = 1; mu < mu_time; mu++) {
    mucycle_RBGS<mu_time, false>(m_v_cycle_lhss[child_level],
                                 m_v_cycle_rhss[child_level], child_level,
                                 n1 + 1, n2);
  }

  m_laplacian_with_levels[child_level]->prolongation</*inplace add*/ true>(
      m_v_cycle_lhss[level], m_v_cycle_lhss[child_level],
      /*parent level*/ *m_laplacian_with_levels[level]);

  for (int i = 0; i < n2; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->RBGS_apply_assume_topo_inplace<false>(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).stop();
  }
}

template <int mu_time, bool skip_first_iter>
void simd_vdb_poisson::mucycle_SRJ(const openvdb::FloatGrid::Ptr in_out_lhs,
                                   const openvdb::FloatGrid::Ptr in_rhs,
                                   const int level, int n) {
  /*lv_copyval(in_out_lhs, in_rhs);
  return;*/
  int n1 = n, n2 = n;
  std::array<float, 3> scheduled_weight;
  if (n == 1) {
    scheduled_weight[0] = 2.0f / 3.0f;
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

  size_t nlevel = m_laplacian_with_levels.size();
  if (level == nlevel - 1) {
    write_coarsest_eigen_rhs(m_coarsest_eigen_rhs, in_rhs);
    m_coarsest_eigen_solution = m_coarsest_solver->solve(m_coarsest_eigen_rhs);
    write_coarsest_grid_solution(in_out_lhs, m_coarsest_eigen_solution);
    return;
  }

  m_v_cycle_lhss[level] = in_out_lhs;
  m_v_cycle_rhss[level] = in_rhs;

  if (skip_first_iter) {
    // set result after first iter, with zero lhs guess
    m_laplacian_with_levels[level]->m_laplacian_evaluator->set_w_jacobi(
        scheduled_weight[0]);
    m_laplacian_with_levels[level]
        ->set_grid_to_result_after_first_jacobi_assume_topo(
            m_v_cycle_temps[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
  }

  for (int i = (skip_first_iter ? 1 : 0); i < n1; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->m_laplacian_evaluator->set_w_jacobi(
        scheduled_weight[i]);
    m_laplacian_with_levels[level]->weighted_jacobi_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
  }
  m_laplacian_with_levels[level]->residual_apply_assume_topo(
      m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);

  int child_level = level + 1;
  m_laplacian_with_levels[level]->restriction(
      m_v_cycle_rhss[child_level], m_v_cycle_temps[level],
      /*child level*/ *m_laplacian_with_levels[child_level]);

  mucycle_SRJ<mu_time, /*skip first*/ true>(
      m_v_cycle_lhss[child_level], m_v_cycle_rhss[child_level], child_level, n);
  for (int mu = 1; mu < mu_time; mu++) {
    mucycle_SRJ<mu_time, /*skip first*/ false>(m_v_cycle_lhss[child_level],
                                               m_v_cycle_rhss[child_level],
                                               child_level, n);
  }

  m_laplacian_with_levels[child_level]->prolongation</*inplace add*/ true>(
      m_v_cycle_lhss[level], m_v_cycle_lhss[child_level],
      /*parent level*/ *m_laplacian_with_levels[level]);

  for (int i = 0; i < n2; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->m_laplacian_evaluator->set_w_jacobi(
        scheduled_weight[size_t(n2 - 1 - i)]);
    m_laplacian_with_levels[level]->weighted_jacobi_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).stop();
  }
}

template <int mu_time, bool skip_first_iter>
void simd_vdb_poisson::mucycle_SPAI0(const openvdb::FloatGrid::Ptr in_out_lhs,
                                     const openvdb::FloatGrid::Ptr in_rhs,
                                     const int level, int n) {
  int n1 = n, n2 = n;

  size_t nlevel = m_laplacian_with_levels.size();
  if (level == nlevel - 1) {
    write_coarsest_eigen_rhs(m_coarsest_eigen_rhs, in_rhs);
    m_coarsest_eigen_solution = m_coarsest_solver->solve(m_coarsest_eigen_rhs);
    write_coarsest_grid_solution(in_out_lhs, m_coarsest_eigen_solution);
    return;
  }

  m_v_cycle_lhss[level] = in_out_lhs;
  m_v_cycle_rhss[level] = in_rhs;

  if (skip_first_iter) {
    // set result after first iter, with zero lhs guess
    m_laplacian_with_levels[level]
        ->set_grid_to_result_after_first_SPAI_assume_topo(
            m_v_cycle_temps[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
  }

  for (int i = (skip_first_iter ? 1 : 0); i < n1; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->SPAI0_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
  }
  m_laplacian_with_levels[level]->residual_apply_assume_topo(
      m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);

  int child_level = level + 1;
  m_laplacian_with_levels[level]->restriction(
      m_v_cycle_rhss[child_level], m_v_cycle_temps[level],
      /*child level*/ *m_laplacian_with_levels[child_level]);

  mucycle_SPAI0<mu_time, /*skip first*/ true>(
      m_v_cycle_lhss[child_level], m_v_cycle_rhss[child_level], child_level, n);
  for (int mu = 1; mu < mu_time; mu++) {
    mucycle_SPAI0<mu_time, /*skip first*/ false>(m_v_cycle_lhss[child_level],
                                                 m_v_cycle_rhss[child_level],
                                                 child_level, n);
  }

  m_laplacian_with_levels[child_level]->prolongation</*inplace add*/ true>(
      m_v_cycle_lhss[level], m_v_cycle_lhss[child_level],
      /*parent level*/ *m_laplacian_with_levels[level]);

  for (int i = 0; i < n2; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->SPAI0_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).stop();
  }
}

/*
https://www.sciencedirect.com/science/article/pii/S0898122114004143
A GPU accelerated aggregation algebraic multigrid method
*/
template <bool skip_first_iter>
void simd_vdb_poisson::Kcycle_SRJ(const openvdb::FloatGrid::Ptr in_out_lhs,
                                  const openvdb::FloatGrid::Ptr in_rhs,
                                  const int level, int n) {

  int n1 = n, n2 = n;
  std::array<float, 3> scheduled_weight;
  if (n == 1) {
    scheduled_weight[0] = 2.0f / 3.0f;
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

  size_t nlevel = m_laplacian_with_levels.size();

  m_v_cycle_lhss[level] = in_out_lhs;
  m_v_cycle_rhss[level] = in_rhs;

  if (skip_first_iter) {
    // set result after first iter, with zero lhs guess
    m_laplacian_with_levels[level]->m_laplacian_evaluator->set_w_jacobi(
        scheduled_weight[0]);
    m_laplacian_with_levels[level]
        ->set_grid_to_result_after_first_jacobi_assume_topo(
            m_v_cycle_temps[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
  }

  for (int i = (skip_first_iter ? 1 : 0); i < n1; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->m_laplacian_evaluator->set_w_jacobi(
        scheduled_weight[i]);
    m_laplacian_with_levels[level]->weighted_jacobi_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
  }
  m_laplacian_with_levels[level]->residual_apply_assume_topo(
      m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);

  int child_level = level + 1;
  m_laplacian_with_levels[level]->restriction(
      m_v_cycle_rhss[child_level], m_v_cycle_temps[level],
      /*child level*/ *m_laplacian_with_levels[child_level]);

  if (child_level == nlevel - 1) {
    write_coarsest_eigen_rhs(m_coarsest_eigen_rhs, m_v_cycle_rhss[child_level]);
    m_coarsest_eigen_solution = m_coarsest_solver->solve(m_coarsest_eigen_rhs);
    write_coarsest_grid_solution(m_v_cycle_lhss[child_level],
                                 m_coarsest_eigen_solution);
  } else {
    // first Krylov iteration

    Kcycle_SRJ<true>(m_v_cycle_lhss[child_level], m_v_cycle_rhss[child_level],
                     child_level);

    lv_copyval(m_K_cycle_cs[child_level], m_v_cycle_lhss[child_level],
               child_level);
    m_laplacian_with_levels[child_level]->Laplacian_apply_assume_topo(
        m_K_cycle_vs[child_level], m_K_cycle_cs[child_level]);
    float rho1 = lv_dot(m_K_cycle_vs[child_level], m_K_cycle_cs[child_level],
                        child_level);
    float alpha1 = lv_dot(m_v_cycle_rhss[child_level],
                          m_K_cycle_cs[child_level], child_level);

    float r_child_norm = lv_abs_max(m_v_cycle_rhss[child_level], child_level);
    lv_axpy(-alpha1 / rho1, m_K_cycle_vs[child_level],
            m_v_cycle_rhss[child_level], child_level);
    float r_tilde_child_norm =
        lv_abs_max(m_v_cycle_rhss[child_level], child_level);
    // printf("level:%d rnorm:%e, rtildenorm:%e\n", level, r_child_norm,
    // r_tilde_child_norm);
    if (r_tilde_child_norm < 0.25 * r_child_norm) {
      lv_axpy(alpha1 / rho1 - 1, m_v_cycle_lhss[child_level],
              m_v_cycle_lhss[child_level], child_level);
    } else {
      // the second Krylov iteration
      // the error will be stored in temps, instead of lhss, because the latter
      // stores previous step results
      Kcycle_SRJ<true>(m_v_cycle_lhss[child_level], m_v_cycle_rhss[child_level],
                       child_level);
      lv_copyval(m_K_cycle_ds[child_level], m_v_cycle_lhss[child_level],
                 child_level);
      m_laplacian_with_levels[child_level]->Laplacian_apply_assume_topo(
          m_K_cycle_ws[child_level], m_K_cycle_ds[child_level]);
      float gamma = lv_dot(m_K_cycle_ds[child_level], m_K_cycle_vs[child_level],
                           child_level);
      float beta = lv_dot(m_K_cycle_ds[child_level], m_K_cycle_ws[child_level],
                          child_level);
      float alpha2 = lv_dot(m_K_cycle_ds[child_level],
                            m_v_cycle_rhss[child_level], child_level);
      float rho2 = beta - gamma * gamma / rho1;
      float coef1 = alpha1 / rho1 - gamma * alpha2 / (rho1 * rho2);
      float coef2 = alpha2 / rho2;
      lv_out_axby(m_v_cycle_lhss[child_level], coef1, coef2,
                  m_K_cycle_cs[child_level], m_K_cycle_ds[child_level],
                  child_level);
    }
  }

  m_laplacian_with_levels[child_level]->prolongation</*inplace add*/ true>(
      m_v_cycle_lhss[level], m_v_cycle_lhss[child_level],
      /*parent level*/ *m_laplacian_with_levels[level]);

  for (int i = 0; i < n2; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).start();
    m_laplacian_with_levels[level]->m_laplacian_evaluator->set_w_jacobi(
        scheduled_weight[i]);
    m_laplacian_with_levels[level]->weighted_jacobi_apply_assume_topo(
        m_v_cycle_temps[level], m_v_cycle_lhss[level], m_v_cycle_rhss[level]);
    m_v_cycle_temps[level].swap(m_v_cycle_lhss[level]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/W/smooth" +
    // std::to_string(level)).stop();
  }
}

void simd_vdb_poisson::construct_coarsest_exact_solver() {
  // the coarse level usually contains <1000 dofs
  tbb::concurrent_vector<Eigen::Triplet<float>> ijval;

  const Laplacian_with_level &laplacian = *m_laplacian_with_levels.back();

  auto build_triplet = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                           openvdb::Index leafpos) {
    auto diag_axr{laplacian.m_Diagonal->getConstAccessor()};
    auto x_axr{laplacian.m_Neg_x_entry->getConstAccessor()};
    auto y_axr{laplacian.m_Neg_y_entry->getConstAccessor()};
    auto z_axr{laplacian.m_Neg_z_entry->getConstAccessor()};
    auto dof_axr{laplacian.m_dof_idx->getConstAccessor()};

    float diag_epsl = laplacian.m_Diagonal->background() * 1e-2f;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      auto gcoord = iter.getCoord();

      float diagval = diag_axr.getValue(gcoord);
      if (diagval == 0) {
        diagval = diag_epsl;
      }
      // the diagonal entry
      ijval.push_back(
          Eigen::Triplet<float>(iter.getValue(), iter.getValue(), diagval));

      // check lower neighbor
      auto negx_coord = gcoord.offsetBy(-1, 0, 0);
      if (dof_axr.isValueOn(negx_coord)) {
        ijval.push_back(Eigen::Triplet<float>(iter.getValue(),
                                              dof_axr.getValue(negx_coord),
                                              x_axr.getValue(gcoord)));
        ijval.push_back(Eigen::Triplet<float>(dof_axr.getValue(negx_coord),
                                              iter.getValue(),
                                              x_axr.getValue(gcoord)));
      } // end negx dof on

      auto negy_coord = gcoord.offsetBy(0, -1, 0);
      if (dof_axr.isValueOn(negy_coord)) {
        ijval.push_back(Eigen::Triplet<float>(iter.getValue(),
                                              dof_axr.getValue(negy_coord),
                                              y_axr.getValue(gcoord)));
        ijval.push_back(Eigen::Triplet<float>(dof_axr.getValue(negy_coord),
                                              iter.getValue(),
                                              y_axr.getValue(gcoord)));
      } // end negy dof on

      auto negz_coord = gcoord.offsetBy(0, 0, -1);
      if (dof_axr.isValueOn(negz_coord)) {
        ijval.push_back(Eigen::Triplet<float>(iter.getValue(),
                                              dof_axr.getValue(negz_coord),
                                              z_axr.getValue(gcoord)));
        ijval.push_back(Eigen::Triplet<float>(dof_axr.getValue(negz_coord),
                                              iter.getValue(),
                                              z_axr.getValue(gcoord)));
      } // end negz dof on
    }   // end for all on dof in this leaf
  };    // end gen triplet

  m_laplacian_with_levels.back()->m_dof_leafmanager->foreach (build_triplet);
  printf("triplet gen\n");
  int ndof = laplacian.m_ndof;
  m_coarsest_eigen_matrix.resize(ndof, ndof);
  // std::vector<Eigen::Triplet<float>> triplets;
  // triplets.reserve(ijval.size());
  // for (int i = 0; i < ijval.size(); i++) {
  //	triplets.push_back(ijval[i]);
  //}
  m_coarsest_eigen_matrix.setFromTriplets(ijval.begin(), ijval.end());
  printf("factor\n");
  m_coarsest_solver =
      std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>>>(
          m_coarsest_eigen_matrix);
}

void simd_vdb_poisson::write_coarsest_eigen_rhs(
    Eigen::VectorXf &out_eigen_rhs, openvdb::FloatGrid::Ptr in_rhs) {
  out_eigen_rhs.setZero(m_laplacian_with_levels.back()->m_ndof);

  auto set_eigen_rhs = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                           openvdb::Index leafpos) {
    auto in_rhs_leaf = in_rhs->tree().probeConstLeaf(leaf.origin());

    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      out_eigen_rhs[iter.getValue()] = in_rhs_leaf->getValue(iter.offset());
    }
  };
  m_laplacian_with_levels.back()->m_dof_leafmanager->foreach (set_eigen_rhs);
}

void simd_vdb_poisson::write_coarsest_grid_solution(
    openvdb::FloatGrid::Ptr in_out_result,
    const Eigen::VectorXf &in_eigen_solution) {
  auto set_grid_solution = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto *result_leaf = in_out_result->tree().probeLeaf(leaf.origin());

    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      result_leaf->setValueOn(iter.offset(),
                              in_eigen_solution[iter.getValue()]);
    }
  };
  m_laplacian_with_levels.back()->m_dof_leafmanager->foreach (
      set_grid_solution);
}

void simd_vdb_poisson::build_rhs() {
  if (m_laplacian_with_levels.empty()) {
    construct_levels();
  }

  m_rhs = m_laplacian_with_levels[0]->get_zero_vec_grid();
  std::atomic<size_t> isolated_cell{0};
  auto build_rhs_op = [&](const openvdb::Int32Tree::LeafNodeType &idxleaf,
                          openvdb::Index leafpos) {
    auto *rhs_leaf = m_rhs->tree().probeLeaf(idxleaf.origin());
    auto weight_axr = m_face_weight->getConstUnsafeAccessor();
    auto vel_axr = m_velocity->getConstUnsafeAccessor();
    auto svel_axr = m_solid_velocity->getConstUnsafeAccessor();

    float invdx = 1.0f / m_dx;

    for (auto offset = 0; offset < idxleaf.SIZE; offset++) {
      if (!idxleaf.isValueMaskOn(offset)) {
        continue;
      }
      const auto lcoord = idxleaf.offsetToLocalCoord(offset);
      const auto gcoord = lcoord + idxleaf.origin();

      float local_rhs = 0;
      openvdb::Vec3f this_vel = vel_axr.getValue(gcoord);
      bool has_non_zero_weight = false;

      // x+ x- y+ y- z+ z-
      for (int i_face = 0; i_face < 6; i_face++) {
        int channel = i_face / 2;
        bool positive_dir = (i_face % 2) == 0;

        auto vneib_c = gcoord;
        vneib_c[channel] += positive_dir;

        //face weight
        float weight = weight_axr.getValue(vneib_c)[channel];

        if (weight != 0.f) {
            has_non_zero_weight = true;
        }

        //liquid velocity channel
        float vel = vel_axr.getValue(vneib_c)[channel];

        //solid velocity channel
        float svel = svel_axr.getValue(vneib_c)[channel];

        // all elements there, write the rhs
        if (positive_dir) {
          local_rhs -= invdx * (weight * vel + (1.0f - weight) * svel);
        } else {
          local_rhs += invdx * (weight * vel + (1.0f - weight) * svel);
        }

      } // end for 6 faces of this voxel

      if (!has_non_zero_weight) {
          local_rhs = 0;
          isolated_cell++;
      }
      rhs_leaf->setValueOn(offset, local_rhs);
    } // for all active dof in this leaf
  };  // end build rhs

  m_laplacian_with_levels[0]->m_dof_leafmanager->foreach (build_rhs_op);
  printf("isolated cell:%d\n", int(isolated_cell));
}
void simd_vdb_poisson::smooth_solve(openvdb::FloatGrid::Ptr in_out_presssure,
                                    int n) {
  auto &level0 = *m_laplacian_with_levels[0];

  if (!m_rhs) {
    build_rhs();
  }
  if (!in_out_presssure->tree().hasSameTopology(level0.m_dof_idx->tree())) {
    printf("input guess pressure does not match dof idx pattern\n");
    exit(-1);
  }
  auto r = level0.get_zero_vec_grid();
  lv_copyval(m_v_cycle_temps[0], r, 0);
  lv_copyval(m_v_cycle_lhss[0], r, 0);
  level0.residual_apply_assume_topo(r, in_out_presssure, m_rhs);

  m_v_cycle_rhss[0] = r;
  for (int i = 0; i < n; i++) {
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth0").start();
    m_laplacian_with_levels[0]->RBGS_apply_assume_topo_inplace<true>(
        m_v_cycle_temps[0], m_v_cycle_lhss[0], m_v_cycle_rhss[0]);
    // CSim::TimerMan::timer("Sim.step/vdbflip/simdpcg/V/smooth0").stop();
    // the updated lhs is in m_v_cycle_temps
    // we need to update the v_cycle_lhs
    // m_v_cycle_lhss[0].swap(m_v_cycle_temps[0]);
    // now the lhs is updated
  }
  lv_axpy(1, m_v_cycle_lhss[0], in_out_presssure);
}
bool simd_vdb_poisson::pcg_solve(openvdb::FloatGrid::Ptr in_out_presssure,
                                 float tolerance) {
  auto &level0 = *m_laplacian_with_levels[0];
  if (!in_out_presssure->tree().hasSameTopology(level0.m_dof_idx->tree())) {
    printf("input guess pressure does not match dof idx pattern\n");
    exit(-1);
  }

  if (!m_rhs) {
    build_rhs();
  }
  m_iteration = 0;

  // according to mcadams algorithm 3

  // line2
  auto r = level0.get_zero_vec_grid();
  level0.residual_apply_assume_topo(r, in_out_presssure, m_rhs);
  float nu = lv_abs_max(r);
  float numax = tolerance * nu;
  numax = std::min(numax, 1e-7f);

  // line3
  if (nu <= numax) {
    return true;
  }

  // line4
  auto p = level0.get_zero_vec_grid();
  level0.set_grid_constant_assume_topo(p, 0);
  mucycle_SRJ<2, true>(p, r, 0);
  // Kcycle_SRJ<true>(p, r);
  // Vcycle(p, r,4,4,200);
  float rho = lv_dot(p, r);

  auto z = level0.get_zero_vec_grid();
  // line 5
  for (; m_iteration < m_max_iter; m_iteration++) {
    // line6
    level0.Laplacian_apply_assume_topo(z, p);
    float sigma = lv_dot(p, z);
    // line7
    float alpha = rho / sigma;
    // line8
    lv_axpy(-alpha, z, r);
    nu = lv_abs_max(r);
    printf("iter:%d err:%e\n", m_iteration, nu);
    // line9
    if (nu <= numax || m_iteration == (m_max_iter - 1)) {
      // line10
      lv_axpy(alpha, p, in_out_presssure);
      // line11
      return (nu <= numax) && (m_iteration <= (m_max_iter - 1));
      // line12
    }
    // line13
    level0.set_grid_constant_assume_topo(z, 0);
    mucycle_SRJ<2, true>(z, r, 0);
    // Kcycle_SRJ<true>(z, r);
    // Vcycle(z, r,4,4,200);
    float rho_new = lv_dot(z, r);

    // line14
    float beta = rho_new / rho;
    // line15
    rho = rho_new;
    // line16
    lv_axpy(alpha, p, in_out_presssure);
    lv_xpay(beta, z, p);
    // line17
  }
  // line18
}

int simd_vdb_poisson::iterations() { return m_iteration; }
void simd_vdb_poisson::symmetry_test(int level) {
  if (m_laplacian_with_levels.empty()) {
    construct_levels();
  }

  if (level >= m_laplacian_with_levels.size()) {
    level = m_laplacian_with_levels.size() - 1;
  }
  // initialize two random vectors
  auto a = m_laplacian_with_levels[level]->get_zero_vec_grid();
  auto b = m_laplacian_with_levels[level]->get_zero_vec_grid();

  auto randomize = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                       openvdb::Index leafpos) {
    std::random_device device;
    std::mt19937 generator(/*seed=*/device());
    std::uniform_real_distribution<> distribution(-0.5, 0.5);
    auto aleaf = a->tree().probeLeaf(leaf.origin());
    auto bleaf = b->tree().probeLeaf(leaf.origin());
    for (auto iter = leaf.beginValueOn(); iter; ++iter) {
      aleaf->setValueOnly(iter.offset(), distribution(generator));
      bleaf->setValueOnly(iter.offset(), distribution(generator));
    }
  };

  m_laplacian_with_levels[level]->m_dof_leafmanager->foreach (randomize);

  auto tempval = m_laplacian_with_levels[level]->get_zero_vec_grid();
  m_laplacian_with_levels[level]->Laplacian_apply_assume_topo(tempval, b);
  float aTLb = lv_dot(a, tempval, level);
  m_laplacian_with_levels[level]->Laplacian_apply_assume_topo(tempval, a);
  float bTLa = lv_dot(b, tempval, level);

  printf("aTLb:%e, bTLa:%e, difference:%e\n", aTLb, bTLa,
         std::abs(aTLb - bTLa));
}
namespace {
struct grid_abs_max_op {
  grid_abs_max_op(openvdb::FloatGrid::Ptr in_grid) {
    m_max = 0;
    m_grid = in_grid;
  }

  grid_abs_max_op(const grid_abs_max_op &other, tbb::split) {
    m_max = 0;
    m_grid = other.m_grid;
  }

  // used by level0 dof leafmanager
  void operator()(openvdb::Int32Tree::LeafNodeType &leaf,
                  openvdb::Index leafpos) {
    auto *float_leaf = m_grid->tree().probeConstLeaf(leaf.origin());
    for (auto iter = float_leaf->cbeginValueOn(); iter; ++iter) {
      m_max = std::max(m_max, std::abs(iter.getValue()));
    }
  }

  void join(grid_abs_max_op &other) { m_max = std::max(m_max, other.m_max); }

  openvdb::FloatGrid::Ptr m_grid;
  float m_max;
};
} // namespace
float simd_vdb_poisson::lv_abs_max(openvdb::FloatGrid::Ptr in_lv0_grid,
                                   int level) {
  auto op{grid_abs_max_op(in_lv0_grid)};
  m_laplacian_with_levels[level]->m_dof_leafmanager->reduce(op);
  return op.m_max;
}
namespace {
struct grid_dot_op {
  grid_dot_op(openvdb::FloatGrid::Ptr in_a, openvdb::FloatGrid::Ptr in_b) {
    m_a = in_a;
    m_b = in_b;
    dp_result = 0;
  }

  grid_dot_op(const grid_dot_op &other, tbb::split) {
    m_a = other.m_a;
    m_b = other.m_b;
    dp_result = 0;
  }

  void operator()(openvdb::Int32Tree::LeafNodeType &leaf,
                  openvdb::Index leafpos) {
    auto *aleaf = m_a->tree().probeConstLeaf(leaf.origin());
    auto *bleaf = m_b->tree().probeConstLeaf(leaf.origin());

    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      dp_result +=
          aleaf->getValue(iter.offset()) * bleaf->getValue(iter.offset());
    }
  }

  void join(grid_dot_op &other) { dp_result += other.dp_result; }

  float dp_result;
  openvdb::FloatGrid::Ptr m_a;
  openvdb::FloatGrid::Ptr m_b;
};
} // namespace
float simd_vdb_poisson::lv_dot(openvdb::FloatGrid::Ptr a,
                               openvdb::FloatGrid::Ptr b, int level) {
  auto op{grid_dot_op{a, b}};
  m_laplacian_with_levels[level]->m_dof_leafmanager->reduce(op);
  return op.dp_result;
}

void simd_vdb_poisson::lv_axpy(const float alpha, openvdb::FloatGrid::Ptr in_x,
                               openvdb::FloatGrid::Ptr in_out_y, int level) {
  // y = a*x + y
  auto add_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                    openvdb::Index leafpos) {
    auto *xleaf = in_x->tree().probeConstLeaf(leaf.origin());
    auto *yleaf = in_out_y->tree().probeLeaf(leaf.origin());

    const float *xdata = xleaf->buffer().data();
    float *ydata = yleaf->buffer().data();
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      ydata[iter.offset()] += alpha * xdata[iter.offset()];
    }
  }; // end add_op

  m_laplacian_with_levels[level]->m_dof_leafmanager->foreach (add_op);
}

void simd_vdb_poisson::lv_xpay(const float alpha, openvdb::FloatGrid::Ptr in_x,
                               openvdb::FloatGrid::Ptr in_out_y, int level) {
  // y = x + a*y
  auto add_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                    openvdb::Index leafpos) {
    auto *xleaf = in_x->tree().probeConstLeaf(leaf.origin());
    auto *yleaf = in_out_y->tree().probeLeaf(leaf.origin());

    const float *xdata = xleaf->buffer().data();
    float *ydata = yleaf->buffer().data();
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      ydata[iter.offset()] =
          xdata[iter.offset()] + alpha * ydata[iter.offset()];
    }
  }; // end add_op

  m_laplacian_with_levels[level]->m_dof_leafmanager->foreach (add_op);
}

void simd_vdb_poisson::lv_out_axby(openvdb::FloatGrid::Ptr out_result,
                                   const float alpha, const float beta,
                                   openvdb::FloatGrid::Ptr in_x,
                                   openvdb::FloatGrid::Ptr in_y, int level) {
  // y = x + a*y
  auto add_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                    openvdb::Index leafpos) {
    auto *xleaf = in_x->tree().probeConstLeaf(leaf.origin());
    auto *yleaf = in_y->tree().probeConstLeaf(leaf.origin());
    auto *outleaf = out_result->tree().probeLeaf(leaf.origin());

    const float *xdata = xleaf->buffer().data();
    const float *ydata = yleaf->buffer().data();
    float *outdata = outleaf->buffer().data();
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
      outdata[iter.offset()] =
          alpha * xdata[iter.offset()] + beta * ydata[iter.offset()];
    }
  }; // end add_op

  m_laplacian_with_levels[level]->m_dof_leafmanager->foreach (add_op);
}

void simd_vdb_poisson::lv_copyval(openvdb::FloatGrid::Ptr out_grid,
                                  openvdb::FloatGrid::Ptr in_grid, int level) {
  // it is assumed that the
  auto copy_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                     openvdb::Index leafpos) {
    auto *in_leaf = in_grid->tree().probeLeaf(leaf.origin());
    auto *out_leaf = out_grid->tree().probeLeaf(leaf.origin());
    std::copy(in_leaf->buffer().data(),
              in_leaf->buffer().data() + in_leaf->SIZE,
              out_leaf->buffer().data());
  };

  m_laplacian_with_levels[level]->m_dof_leafmanager->foreach (copy_op);
}
