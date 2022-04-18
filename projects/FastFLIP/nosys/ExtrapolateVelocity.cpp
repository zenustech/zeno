#include "openvdb/openvdb.h"
#include <omp.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tree/LeafManager.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
class vdb_velocity_extrapolator {
public:
  static void extrapolate(int n_layer,
                          openvdb::Vec3fGrid::Ptr &in_out_vel_grid);
};

class velocity_extrapolate_functor {
  using vel_tree_node_t = openvdb::Vec3fTree::LeafNodeType;
  // using valid_tree_node_t = openvdb::BoolTree::LeafNodeType;
public:
  velocity_extrapolate_functor(
      const std::vector<vel_tree_node_t *> &in_vel_tree_node_ptrs,
      openvdb::Vec3SGrid::Ptr in_vel_grid,
      openvdb::BoolGrid::Ptr in_empty_indicator)
      : m_vel_tree_node_ptrs(in_vel_tree_node_ptrs) {
    m_vel_has_empty_voxel = in_empty_indicator;
    m_vel_grid = in_vel_grid;
    m_extrapolate_vel_grid =
        openvdb::Vec3SGrid::create(openvdb::Vec3s{0, 0, 0});
    m_extrapolate_weight_grid = openvdb::Int32Grid::create(0);
  };

  void seven_neib_operator(const tbb::blocked_range<openvdb::Index32> &r) {
    auto target_vel_axr = m_extrapolate_vel_grid->getAccessor();
    auto target_weight_axr = m_extrapolate_weight_grid->getAccessor();
    auto const_vel_axr = m_vel_grid->getConstAccessor();

    auto const_empty_indicator_axr = m_vel_has_empty_voxel->getConstAccessor();

    for (size_t i_node = r.begin(); i_node != r.end(); ++i_node) {
      const vel_tree_node_t &vel_node = *m_vel_tree_node_ptrs[i_node];
      const auto N = vel_node.dim();

      openvdb::Coord leaf_coord =
          openvdb::Coord(vel_node.origin().asVec3i() / 8);

      // only perform the search if there is empty cells to fill
      if (const_empty_indicator_axr.getValue(leaf_coord)) {
        for (openvdb::Index offset = 0; offset < vel_node.size(); offset++) {
          // collect operation
          /************1 interpolate vels in this node**********************/
          openvdb::Coord ijk = vel_node.offsetToLocalCoord(offset);
          if (vel_node.isValueMaskOff(offset)) {
            // only collect the contribution from this node,
            openvdb::Vec3s tempvel{0, 0, 0};
            openvdb::Int32 valcount = 0;
            openvdb::Coord c{ijk};
            auto coffset = offset;

            for (int i_xyz = 0; i_xyz < 3; i_xyz++) {
              // xyz-
              c = ijk;
              c[i_xyz]--;
              if (c[i_xyz] >= 0) {
                coffset = vel_node.coordToOffset(c);
                if (vel_node.isValueMaskOn(coffset)) {
                  tempvel += vel_node.getValue(coffset);
                  valcount++;
                }
              }
              // xyz+
              c = ijk;
              c[i_xyz]++;
              if (c[i_xyz] < N) {
                coffset = vel_node.coordToOffset(c);
                if (vel_node.isValueMaskOn(coffset)) {
                  tempvel += vel_node.getValue(coffset);
                  valcount++;
                }
              }
            } // end for three directions

            // write to the result grid
            if (valcount != 0) {
              target_vel_axr.setValue(vel_node.offsetToGlobalCoord(offset),
                                      tempvel);
              target_weight_axr.setValue(vel_node.offsetToGlobalCoord(offset),
                                         valcount);
            }
          } // end if velocity not set
        }   // end for all voxels
      }     // end if has empty voxel

      bool found_empty_neib = false;

      for (int ii = -1; ii <= 1 && !found_empty_neib; ii++) {
        for (int jj = -1; jj <= 1 && !found_empty_neib; jj++) {
          for (int kk = -1; kk <= 1 && !found_empty_neib; kk++) {
            if (ii == 0 && jj == 0 && kk == 0) {
              // skip the center cell
              continue;
            }
            found_empty_neib |= const_empty_indicator_axr.getValue(
                openvdb::Coord{leaf_coord[0] + ii, leaf_coord[1] + jj,
                               leaf_coord[2] + kk});
          }
        }
      }
      if (found_empty_neib) {
        for (openvdb::Index offset = 0; offset < vel_node.size(); offset++) {
          openvdb::Coord ijk = vel_node.offsetToLocalCoord(offset);
          /************2 interpolate vels to the neighbor of this node******/
          for (int i_xyz = 0; i_xyz < 3; i_xyz++) {
            if (((ijk[i_xyz] == 0) || (ijk[i_xyz] == N - 1)) &&
                vel_node.isValueOn(offset)) {

              // this voxel will possibly contribute to the leaf node on the -/+
              // side
              openvdb::Coord neib_global_coord = vel_node.origin() + ijk;
              openvdb::Coord neib_global_coord_div8{leaf_coord};
              if (ijk[i_xyz] == 0) {
                neib_global_coord[i_xyz]--;
                neib_global_coord_div8[i_xyz]--;
              } else {
                neib_global_coord[i_xyz]++;
                neib_global_coord_div8[i_xyz]++;
              }

              if (const_empty_indicator_axr.getValue(neib_global_coord_div8) !=
                  true) {
                continue;
              }

              if (!const_vel_axr.isValueOn(neib_global_coord)) {
                // really influencing an empty voxel
                auto tempvel = target_vel_axr.getValue(neib_global_coord);
                auto valcount = target_weight_axr.getValue(neib_global_coord);
                tempvel += vel_node.getValue(offset);
                valcount++;
                target_vel_axr.setValue(neib_global_coord, tempvel);
                target_weight_axr.setValue(neib_global_coord, valcount);
              } // end if the influenced voxel is not valid
            }   // end if near the boundary
          }     // end for three directions
        }       // end if found leaf nodes that has empty voxels

      } // end loop over all voxels in this leaf node

    } // end loop over all leaf nodes
  }

  void
  twenty_seven_neib_operator(const tbb::blocked_range<openvdb::Index32> &r) {
    auto target_vel_axr = m_extrapolate_vel_grid->getAccessor();
    auto target_weight_axr = m_extrapolate_weight_grid->getAccessor();
    auto const_vel_axr = m_vel_grid->getConstAccessor();

    auto const_empty_indicator_axr = m_vel_has_empty_voxel->getConstAccessor();

    for (size_t i_node = r.begin(); i_node != r.end(); ++i_node) {
      const vel_tree_node_t &vel_node = *m_vel_tree_node_ptrs[i_node];
      const int N = vel_node.dim();
      openvdb::Coord leaf_coord =
          openvdb::Coord(vel_node.origin().asVec3i() / 8);

      // only perform the search if there is empty cells to fill
      if (const_empty_indicator_axr.getValue(leaf_coord)) {
        for (openvdb::Index offset = 0; offset < vel_node.size(); offset++) {
          // collect operation
          /************1 interpolate vels in this node**********************/
          openvdb::Coord ijk = vel_node.offsetToLocalCoord(offset);
          if (vel_node.isValueMaskOff(offset)) {
            // only collect the contribution from this node,
            openvdb::Vec3s tempvel{0, 0, 0};
            openvdb::Int32 valcount = 0;
            openvdb::Coord c{ijk};
            auto coffset = offset;

            int ib = std::max(0, c[0] - 1), ie = std::min(N, c[0] + 2);
            int jb = std::max(0, c[1] - 1), je = std::min(N, c[1] + 2);
            int kb = std::max(0, c[2] - 1), ke = std::min(N, c[2] + 2);

            // loop over its 27 neighbors inside the leaf
            for (int ii = ib; ii < ie; ii++) {
              for (int jj = jb; jj < je; jj++) {
                for (int kk = kb; kk < ke; kk++) {
                  coffset = vel_node.coordToOffset(openvdb::Coord{ii, jj, kk});
                  if (vel_node.isValueMaskOn(coffset)) {
                    tempvel += vel_node.getValue(coffset);
                    valcount++;
                  }
                } // end kk
              }   // end jj
            }     // end ii

            ////loop over its 27 neighbors inside the leaf
            // for (int ii = std::max(0, c[0]-1); ii < std::min(N, c[0]+2);
            // ii++) { 	for (int jj = std::max(0, c[1] - 1); jj < std::min(N, c[1]
            //+ 2); jj++) { 		for (int kk = std::max(0, c[2] - 1); kk <
            //std::min(N, c[2] + 2); kk++) { 			coffset =
            //vel_node.coordToOffset(openvdb::Coord{ ii,jj,kk }); 			if
            //(vel_node.isValueMaskOn(coffset)) { 				tempvel +=
            //vel_node.getValue(coffset); 				valcount++;
            //			}
            //		}//end kk
            //	}//end jj
            //}//end ii

            // write to the result grid
            if (valcount != 0) {
              target_vel_axr.setValue(vel_node.offsetToGlobalCoord(offset),
                                      tempvel);
              target_weight_axr.setValue(vel_node.offsetToGlobalCoord(offset),
                                         valcount);
            }
          } // end if velocity not set
        }   // end for all voxels
      }     // end if has empty voxel

      bool found_empty_neib = false;

      for (int ii = -1; ii <= 1 && !found_empty_neib; ii++) {
        for (int jj = -1; jj <= 1 && !found_empty_neib; jj++) {
          for (int kk = -1; kk <= 1 && !found_empty_neib; kk++) {
            if (ii == 0 && jj == 0 && kk == 0) {
              // skip the center cell
              continue;
            }
            found_empty_neib |= const_empty_indicator_axr.getValue(
                openvdb::Coord{leaf_coord[0] + ii, leaf_coord[1] + jj,
                               leaf_coord[2] + kk});
          }
        }
      }
      if (found_empty_neib) {
        int Nm1 = vel_node.DIM - 1;
        for (openvdb::Index offset = 0; offset < vel_node.size(); offset++) {
          const openvdb::Coord ijk = vel_node.offsetToLocalCoord(offset);
          bool on_border = false;
          if (ijk[0] == 0 || ijk[1] == 0 || ijk[2] == 0 || ijk[0] == Nm1 ||
              ijk[1] == Nm1 || ijk[2] == Nm1) {
            on_border = true;
          }

          if (on_border && vel_node.isValueOn(offset)) {
            // loop its 27 neighbors
            for (int ii = ijk[0] - 1; ii <= ijk[0] + 1; ii++) {
              for (int jj = ijk[1] - 1; jj <= ijk[1] + 1; jj++) {
                for (int kk = ijk[2] - 1; kk <= ijk[2] + 1; kk++) {
                  // still writing to this leaf node
                  if (ii >= 0 && ii < N && jj >= 0 && jj < N && kk >= 0 &&
                      kk < N) {
                    continue;
                  }
                  // writing to neighbors
                  openvdb::Coord neib_global_coord =
                      vel_node.origin().offsetBy(ii, jj, kk);
                  openvdb::Coord neib_global_coord_div8{leaf_coord};

                  if (ii == -1) {
                    neib_global_coord_div8[0]--;
                  } else if (ii == N) {
                    neib_global_coord_div8[0]++;
                  }

                  if (jj == -1) {
                    neib_global_coord_div8[1]--;
                  } else if (jj == N) {
                    neib_global_coord_div8[1]++;
                  }

                  if (kk == -1) {
                    neib_global_coord_div8[2]--;
                  } else if (kk == N) {
                    neib_global_coord_div8[2]++;
                  }

                  if (const_empty_indicator_axr.getValue(
                          neib_global_coord_div8) != true) {
                    continue;
                  }

                  if (!const_vel_axr.isValueOn(neib_global_coord)) {
                    // really influencing an empty voxel
                    auto tempvel = target_vel_axr.getValue(neib_global_coord);
                    auto valcount =
                        target_weight_axr.getValue(neib_global_coord);
                    tempvel += vel_node.getValue(offset);
                    valcount++;
                    target_vel_axr.setValue(neib_global_coord, tempvel);
                    target_weight_axr.setValue(neib_global_coord, valcount);
                  } // end if the influenced voxel is not valid

                } // end kk
              }   // end jj
            }     // end ii
          }       // end if on border

        } // end loop over all voxels in this leaf node

      } // end if found leaf nodes that has empty voxels

    } // end loop over all leaf nodes
  }

  void operator()(const tbb::blocked_range<openvdb::Index32> &r) {
    seven_neib_operator(r);
  }

  velocity_extrapolate_functor(const velocity_extrapolate_functor &rhs,
                               tbb::split)
      : m_vel_tree_node_ptrs(rhs.m_vel_tree_node_ptrs) {
    m_vel_grid = rhs.m_vel_grid;
    m_vel_has_empty_voxel = rhs.m_vel_has_empty_voxel;
    m_extrapolate_vel_grid =
        openvdb::Vec3SGrid::create(openvdb::Vec3s{0, 0, 0});
    m_extrapolate_weight_grid = openvdb::Int32Grid::create(0);
  }

  void join(velocity_extrapolate_functor &rhs) { join2(rhs); }

  void join1(velocity_extrapolate_functor &rhs) {
    // join the velocity and weight
    openvdb::tools::compSum(*m_extrapolate_vel_grid,
                            *rhs.m_extrapolate_vel_grid);
    openvdb::tools::compSum(*m_extrapolate_weight_grid,
                            *rhs.m_extrapolate_weight_grid);
  }

  void join2(velocity_extrapolate_functor &rhs) {
    // touch and add
    // velocity
    for (auto leaf = rhs.m_extrapolate_vel_grid->tree().beginLeaf(); leaf;
         ++leaf) {
      auto *newLeaf = m_extrapolate_vel_grid->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        auto &tree =
            const_cast<openvdb::Vec3fGrid &>(*rhs.m_extrapolate_vel_grid)
                .tree();
        m_extrapolate_vel_grid->tree().addLeaf(
            tree.template stealNode<openvdb::Vec3fGrid::TreeType::LeafNodeType>(
                leaf->origin(), openvdb::Vec3f{0}, false));
      } else {
        // otherwise increment existing values
        for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
          auto val = newLeaf->getValue(iter.offset());
          val += *iter;
          newLeaf->setValueOn(iter.offset(), val);
        }
      }
    }

    // velocity neighbor count
    for (auto leaf = rhs.m_extrapolate_weight_grid->tree().beginLeaf(); leaf;
         ++leaf) {
      auto *newLeaf =
          m_extrapolate_weight_grid->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        auto &tree =
            const_cast<openvdb::Int32Grid &>(*rhs.m_extrapolate_weight_grid)
                .tree();
        m_extrapolate_weight_grid->tree().addLeaf(
            tree.template stealNode<openvdb::Int32Grid::TreeType::LeafNodeType>(
                leaf->origin(), 0, false));
      } else {
        // otherwise increment existing values
        for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
          auto val = newLeaf->getValue(iter.offset());
          val += *iter;
          newLeaf->setValueOn(iter.offset(), val);
        }
      }
    }
  }

  // constant source part
  const std::vector<vel_tree_node_t *> &m_vel_tree_node_ptrs;

  openvdb::Vec3SGrid::Ptr m_vel_grid;
  // const std::vector<valid_tree_node_t*>& m_valid_tree_node_ptrs;

  // empty indicator
  // takes leaf.origin()/8 as argument
  openvdb::BoolGrid::Ptr m_vel_has_empty_voxel;

public:
  // reduced result
  // temporary velocity grid storing the extrapolated value
  openvdb::Vec3SGrid::Ptr m_extrapolate_vel_grid;
  // marks if the voxel is the result of interpolateion
  openvdb::Int32Grid::Ptr m_extrapolate_weight_grid;
}; // end extrapolate functor

struct add_extrapolated_vel_op {

  add_extrapolated_vel_op(
      std::vector<openvdb::Vec3fGrid::TreeType::LeafNodeType *>
          &in_result_vel_nodes,
      std::vector<openvdb::Int32Tree::LeafNodeType *> &in_count_nodes)
      : m_result_vel_nodes(in_result_vel_nodes),
        m_count_nodes(in_count_nodes){};

  // this operator is designed for the leaf manager
  // of m_extrapolate_vel_grid
  // the m_result_velocity_grid must contain the leafs of the
  // extrapolated_velocity
  void
  operator()(openvdb::Vec3SGrid::TreeType::LeafNodeType &extrapolated_vel_leaf,
             openvdb::Index leafpos) const {
    auto *result_leaf = m_result_vel_nodes[leafpos];
    auto *count_leaf = m_count_nodes[leafpos];
    /*if (!result_leaf||!count_leaf) {
            printf("cannot write to target velocity field because the leaf
    doesn't exist\n"); exit(-1);
    }

    if (result_leaf->origin() != extrapolated_vel_leaf.origin()) {
            printf("returned leaf are not the same in space\n");
            exit(-1);
    }*/

    // real writing
    for (auto offset = 0; offset < count_leaf->SIZE; offset++) {
      if (!count_leaf->isValueMaskOn(offset)) {
        continue;
      }

      auto count = count_leaf->getValue(offset);
      /*if (count == 0) {
              printf("extrapolated counter = 0 but is touched\n");
              exit(-1);
      }*/
      if (count != 0) {
        result_leaf->setValueOn(offset, extrapolated_vel_leaf.getValue(offset) /
                                            float(count));
      }
    }
  }

  // velocity grid storing the extrapolated value unnormalized
  // openvdb::Vec3SGrid::Ptr m_result_velocity_grid;
  // count of neighborhood voxels
  // openvdb::Int32Grid::Ptr m_extrapolate_weight_grid;

  // leafs
  std::vector<openvdb::Vec3fGrid::TreeType::LeafNodeType *> m_result_vel_nodes;
  std::vector<openvdb::Int32Tree::LeafNodeType *> m_count_nodes;
};

void vdb_velocity_extrapolator::extrapolate(
    int n_layer, openvdb::Vec3fGrid::Ptr &in_out_vel_grid) {
  using vel_tree_node_t = openvdb::Vec3fTree::LeafNodeType;
  // using valid_tree_node_t = openvdb::BoolTree::LeafNodeType;

  // a node mask indicating if this node in current in_out_velocity grid
  // contains empty voxels to be extrapolated
  auto empty_indicator = openvdb::BoolGrid::create(true);
  // by default, all nodes can be potentially empty.

  auto empty_indicator_axr = empty_indicator->getAccessor();

  auto update_indicator = [&]() {
    empty_indicator->clear();
    for (auto itr = in_out_vel_grid->tree().cbeginLeaf(); itr; ++itr) {
      auto coord = itr->origin();
      empty_indicator_axr.setValue(openvdb::Coord(coord.asVec3i() / 8),
                                   !itr->isValueMaskOn());
    }
  };

  for (; n_layer > 0; n_layer--) {
    update_indicator();
    // m_extrapolate_vel_grid->clear();
    // m_extrapolate_weight_grid->clear();

    // list of all nodes of the original velocity
    auto vel_leaf_count = in_out_vel_grid->treePtr()->leafCount();
    std::vector<openvdb::Vec3fGrid::TreeType::LeafNodeType *>
        vel_leaf_node_ptrs;
    vel_leaf_node_ptrs.reserve(vel_leaf_count);
    in_out_vel_grid->treePtr()->getNodes(vel_leaf_node_ptrs);

    auto vel_extrapolator = velocity_extrapolate_functor{
        vel_leaf_node_ptrs, in_out_vel_grid, empty_indicator};

    tbb::parallel_reduce(
        tbb::blocked_range<openvdb::Index>(0, vel_leaf_count, 300),
        vel_extrapolator);
    printf("extrapolate done\n");
    auto temp_extrapolated_vel_grid = openvdb::Vec3fGrid::create();

    ////add the influence of the extrapolated velocity
    // temp_extrapolated_vel_grid->treePtr()->combine2Extended(
    //	vel_extrapolator.m_extrapolate_vel_grid->tree(),
    //	vel_extrapolator.m_extrapolate_weight_grid->tree(),
    //	divide_weight_functor(), false);

    // in_out_vel_grid->treePtr()->combineExtended(temp_extrapolated_vel_grid->tree(),
    // add_extrapolated_velocity_functor(), false);

    auto extrapolator_leafmanager =
        openvdb::tree::LeafManager<openvdb::Vec3fTree>(
            vel_extrapolator.m_extrapolate_vel_grid->tree());

    // prepare the nodes
    std::vector<openvdb::Vec3fGrid::TreeType::LeafNodeType *> result_vel_nodes;
    std::vector<openvdb::Int32Tree::LeafNodeType *> count_nodes;

    auto n_leafcount =
        vel_extrapolator.m_extrapolate_weight_grid->tree().leafCount();
    result_vel_nodes.reserve(n_leafcount);
    count_nodes.reserve(n_leafcount);

    auto set_nodes =
        [&](openvdb::Vec3SGrid::TreeType::LeafNodeType &extrapolated_vel_leaf,
            openvdb::Index leafpos) {
          auto *vel_leaf =
              in_out_vel_grid->tree().touchLeaf(extrapolated_vel_leaf.origin());
          auto *count_leaf =
              vel_extrapolator.m_extrapolate_weight_grid->tree().touchLeaf(
                  extrapolated_vel_leaf.origin());
          result_vel_nodes.push_back(vel_leaf);
          count_nodes.push_back(count_leaf);
        };

    extrapolator_leafmanager.foreach (set_nodes, false);

    auto finalizer = add_extrapolated_vel_op(result_vel_nodes, count_nodes);

    extrapolator_leafmanager.foreach (finalizer);
  }
}

namespace zeno {

struct Vec3FieldExtrapolate : zeno::INode {
  virtual void apply() override {
    int n = get_param<int>("NumIterates"));
    auto velocity = get_input("Field")->as<VDBFloat3Grid>();

    vdb_velocity_extrapolator::extrapolate(n, velocity->m_grid);
  }
};

static int defVec3FieldExtrapolate = zeno::defNodeClass<Vec3FieldExtrapolate>(
    "Vec3FieldExtrapolate", {/* inputs: */ {
                                 "Field",
                             },
                             /* outputs: */ {},
                             /* params: */
                             {
                                 {"int", "NumIterates", "1"},
                             },

                             /* category: */
                             {
                                 "FLIPSolver",
                             }});

} // namespace zeno