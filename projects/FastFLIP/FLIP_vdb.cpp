#include "FLIP_vdb.h"
#include "levelset_util.h"
#include "openvdb/points/PointAdvect.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/Morphology.h"
#include "openvdb/tree/LeafManager.h"
#include "tbb/scalable_allocator.h"
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>

#include <atomic>
// intrinsics
#include "simd_vdb_poisson.h"
#include <openvdb/tools/Interpolation.h>
#include <xmmintrin.h>
// #include "BPS3D_volume_vel_cache.h" // decouple Datan BEM works for now

#include "tbb/blocked_range3d.h"
#include <thread>
namespace {
struct BoxSampler {
  template <typename axr_t>
  static void get_eight_data(openvdb::Vec3f *data, axr_t &axr,
                             openvdb::Coord gcoord) {
    auto &i = gcoord[0];
    auto &j = gcoord[1];
    auto &k = gcoord[2];
    // 000
    data[0] = axr.getValue(gcoord);
    // 001
    k++;
    data[1] = axr.getValue(gcoord);
    // 011
    j++;
    data[3] = axr.getValue(gcoord);
    // 010
    k--;
    data[2] = axr.getValue(gcoord);
    // 110
    i++;
    data[6] = axr.getValue(gcoord);
    // 111
    k++;
    data[7] = axr.getValue(gcoord);
    // 101
    j--;
    data[5] = axr.getValue(gcoord);
    // 100
    k--;
    data[4] = axr.getValue(gcoord);
  }

  // w=0->a
  // w=1->b
  static float mix(float a, float b, float w) { return a + (b - a) * w; }

  // channel mix
  template <int c>
  static float mixc(const openvdb::Vec3f &a, const openvdb::Vec3f &b, float w) {
    return a[c] + (b[c] - a[c]) * w;
  }

  /*void trilinear_interpolate(openvdb::Vec3f& out, openvdb::Vec3f* data, float
  wx, float wy, float wz) { out = mix( mix(mix(data[0], data[1], wz),
  mix(data[2], data[3], wz), wy), mix(mix(data[4], data[5], wz), mix(data[6],
  data[7], wz), wy), wx);
  }*/

  // channel interpolation
  template <int c>
  static void trilinear_interpolatec(float &out, const openvdb::Vec3f *data,
                                     float wx, float wy, float wz) {
    out = mix(mix(mix(data[0][c], data[1][c], wz),
                  mix(data[2][c], data[3][c], wz), wy),
              mix(mix(data[4][c], data[5][c], wz),
                  mix(data[6][c], data[7][c], wz), wy),
              wx);
  }

  // channel interpolation
  template <int c, typename axr_t>
  static float samplec(axr_t &axr, const openvdb::Vec3f &ixyz) {
    using namespace openvdb::tools::local_util;
    openvdb::Coord base{floorVec3(ixyz)};
    openvdb::Vec3f data[8];
    get_eight_data(data, axr, base);
    float result;
    trilinear_interpolatec<c>(result, data, ixyz.x() - base.x(),
                              ixyz.y() - base.y(), ixyz.z() - base.z());
    return result;
  };
};

struct StaggeredBoxSampler {
  template <typename axr_t>
  static openvdb::Vec3f sample(axr_t &axr, const openvdb::Vec3f &xyz) {
    float vx = BoxSampler::samplec<0>(
        axr, openvdb::Vec3f{xyz.x() + 0.5f, xyz.y(), xyz.z()});
    float vy = BoxSampler::samplec<1>(
        axr, openvdb::Vec3f{xyz.x(), xyz.y() + 0.5f, xyz.z()});
    float vz = BoxSampler::samplec<2>(
        axr, openvdb::Vec3f{xyz.x(), xyz.y(), xyz.z() + 0.5f});

    return openvdb::Vec3f{vx, vy, vz};
  };
};
} // namespace
namespace {

// reduction style p2g operator designed for the leaf manager class
struct particle_to_grid_reducer {

  // constructor
  particle_to_grid_reducer(openvdb::Vec3fGrid::Ptr in_velocity,
                           openvdb::Vec3fGrid::Ptr in_velocity_weights,
                           openvdb::FloatGrid::Ptr in_liquid_sdf,
                           float in_particle_radius) {
    m_particle_radius = in_particle_radius;
    m_transform = in_velocity->transformPtr();
    in_velocity->clear();
    in_velocity_weights->clear();
    in_liquid_sdf->clear();

    // the final result will be directly modify the input argument
    m_velocity = in_velocity;
    m_velocity_weights = in_velocity_weights;
    m_liquid_sdf = in_liquid_sdf;

    // set the loop order and offset of the u v w phi sampling point
    // with respect to the center voxel center
    x_offset_to_center_voxel_center =
        std::make_shared<std::array<__m128, 27>>();
    y_offset_to_center_voxel_center =
        std::make_shared<std::array<__m128, 27>>();
    z_offset_to_center_voxel_center =
        std::make_shared<std::array<__m128, 27>>();
    loop_order = std::make_shared<std::array<openvdb::Coord, 27>>();

    // u:  (-0.5, 0, 0)
    // v:  (0, -0.5, 0)
    // w:  (0, 0, -0.5)
    // phi:(0, 0, 0)
    // note we use set so the
    // R0 R1 R2 R3 correspond to the 0,1,2,3 argument
    // when later extracting using _mm_storer_ps, we get float[0] = arg0
    // correctly see
    // http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_store.htm
    // see
    // http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_set.htm

    __m128 xpack = _mm_set_ps(-0.5f, 0.f, 0.f, 0.f);
    __m128 ypack = _mm_set_ps(0.f, -0.5f, 0.f, 0.f);
    __m128 zpack = _mm_set_ps(0.f, 0.f, -0.5f, 0.f);
    for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
      int ijk = ivoxel;
      int basex = ijk / 9;
      ijk -= 9 * basex;
      int basey = ijk / 3;
      ijk -= 3 * basey;
      int basez = ijk;
      // becomes -1 -> 1

      basex -= 1;
      basey -= 1;
      basez -= 1;
      // broadcast four float as the base
      __m128 basex4 = _mm_set_ps1(float(basex));
      __m128 basey4 = _mm_set_ps1(float(basey));
      __m128 basez4 = _mm_set_ps1(float(basez));

      loop_order->at(ivoxel) = openvdb::Coord{basex, basey, basez};

      x_offset_to_center_voxel_center->at(ivoxel) = _mm_add_ps(basex4, xpack);
      y_offset_to_center_voxel_center->at(ivoxel) = _mm_add_ps(basey4, ypack);
      z_offset_to_center_voxel_center->at(ivoxel) = _mm_add_ps(basez4, zpack);
    }
  }

  // split constructor
  particle_to_grid_reducer(particle_to_grid_reducer &other, tbb::split) {
    m_particle_radius = other.m_particle_radius;
    m_transform = other.m_transform;

    m_velocity = other.m_velocity->copyWithNewTree();
    m_velocity_weights = other.m_velocity_weights->copyWithNewTree();
    m_liquid_sdf = other.m_liquid_sdf->copyWithNewTree();

    x_offset_to_center_voxel_center = other.x_offset_to_center_voxel_center;
    y_offset_to_center_voxel_center = other.y_offset_to_center_voxel_center;
    z_offset_to_center_voxel_center = other.z_offset_to_center_voxel_center;
    loop_order = other.loop_order;
  }

  // operator
  // this opeartor is called inside a range loop inthe leaf manager
  void operator()(
      openvdb::points::PointDataGrid::TreeType::LeafNodeType &particle_leaf,
      openvdb::Index leafpos) {

    const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
    const __m128 float1x4 = _mm_set_ps1(float(1));
    const __m128 float0x4 = _mm_set_ps1(float(0));

    // attribute reader
    // Extract the position attribute from the leaf by name (P is position).
    const openvdb::points::AttributeArray &positionArray =
        particle_leaf.constAttributeArray("P");
    // Extract the velocity attribute from the leaf by name (v is velocity).
    const openvdb::points::AttributeArray &velocityArray =
        particle_leaf.constAttributeArray("v");

    // Create read-only handles for position and velocity.
    openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>
        positionHandle(positionArray);
    openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>
        velocityHandle(velocityArray);

    // scatter style transfer
    // loop over all particles
    float tophix, tophiy, tophiz;
    float dx = m_transform->voxelSize()[0];

    // accessor to the three grids
    auto vel_accessor = m_velocity->getAccessor();
    auto vel_weights_accessor = m_velocity_weights->getAccessor();
    auto liquid_sdf_accessor = m_liquid_sdf->getAccessor();

    // most of the time the particle will write to the current leaf
    auto *local_vel_leaf = vel_accessor.touchLeaf(particle_leaf.origin());
    auto *local_vel_weights_leaf =
        vel_weights_accessor.touchLeaf(particle_leaf.origin());
    auto *local_sdf_leaf =
        liquid_sdf_accessor.touchLeaf(particle_leaf.origin());

    neib_sdf_leaf_ptr_cache.fill(nullptr);
    neib_vel_leaf_ptr_cache.fill(nullptr);
    neib_vel_weights_leaf_ptr_cache.fill(nullptr);

    // set the center cache
    neib_sdf_leaf_ptr_cache[9 + 3 + 1] = local_sdf_leaf;
    neib_vel_leaf_ptr_cache[9 + 3 + 1] = local_vel_leaf;
    neib_vel_weights_leaf_ptr_cache[9 + 3 + 1] = local_vel_weights_leaf;

    for (auto piter = particle_leaf.beginIndexOn(); piter; ++piter) {

      auto voxelpos = positionHandle.get(*piter);
      auto pvel = velocityHandle.get(*piter);

      // broadcast the variables
      __m128 particle_x = _mm_set_ps1(voxelpos[0]);
      __m128 particle_y = _mm_set_ps1(voxelpos[1]);
      __m128 particle_z = _mm_set_ps1(voxelpos[2]);

      // calculate the distance to the 27 neib uvw phi samples
      for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
        // calculate the distance
        // arg(A,B): ret A-B
        // the absolute value trick: abs_mask: 01111111..32bit..1111 x 4
        // _mm_and_ps(abs_mask(), v);
        x_dist_particle_to_sample = _mm_and_ps(
            absmask, _mm_sub_ps(x_offset_to_center_voxel_center->at(ivoxel),
                                particle_x));
        y_dist_particle_to_sample = _mm_and_ps(
            absmask, _mm_sub_ps(y_offset_to_center_voxel_center->at(ivoxel),
                                particle_y));
        z_dist_particle_to_sample = _mm_and_ps(
            absmask, _mm_sub_ps(z_offset_to_center_voxel_center->at(ivoxel),
                                particle_z));

        // the distance to the phi variable
        _mm_store_ss(&tophix, x_dist_particle_to_sample);
        _mm_store_ss(&tophiy, y_dist_particle_to_sample);
        _mm_store_ss(&tophiz, z_dist_particle_to_sample);
        dist_to_phi_sample =
            dx * std::sqrt(tophix * tophix + tophiy * tophiy + tophiz * tophiz);

        // the uvw weights trilinear
        // transfer the distance to weight at the 27 voxels
        //(1-dist)
        // the far points now becomes negative
        x_dist_particle_to_sample =
            _mm_sub_ps(float1x4, x_dist_particle_to_sample);
        y_dist_particle_to_sample =
            _mm_sub_ps(float1x4, y_dist_particle_to_sample);
        z_dist_particle_to_sample =
            _mm_sub_ps(float1x4, z_dist_particle_to_sample);

        // turn everything positive or zero
        // now the dist_to_sample is actually the component-wise weight on the
        // voxel time to multiply them together
        x_dist_particle_to_sample =
            _mm_max_ps(float0x4, x_dist_particle_to_sample);
        y_dist_particle_to_sample =
            _mm_max_ps(float0x4, y_dist_particle_to_sample);
        z_dist_particle_to_sample =
            _mm_max_ps(float0x4, z_dist_particle_to_sample);

        // turn them into weights reduce to x
        x_dist_particle_to_sample =
            _mm_mul_ps(x_dist_particle_to_sample, y_dist_particle_to_sample);
        x_dist_particle_to_sample =
            _mm_mul_ps(x_dist_particle_to_sample, z_dist_particle_to_sample);
        //}//end for 27 voxel

        ////write to the grid
        // for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
        alignas(16) float packed_weight[4];
        _mm_storer_ps(packed_weight, x_dist_particle_to_sample);

        openvdb::Coord write_coord = piter.getCoord() + loop_order->at(ivoxel);

        // check if the write position is local
        auto c = write_coord - particle_leaf.origin();
        int in_neib = 9 + 3 + 1;
        if (c[0] < 0) {
          in_neib -= 9;
        } else if (c[0] >= 8) {
          in_neib += 9;
        }

        if (c[1] < 0) {
          in_neib -= 3;
        } else if (c[1] >= 8) {
          in_neib += 3;
        }

        if (c[2] < 0) {
          in_neib -= 1;
        } else if (c[2] >= 8) {
          in_neib += 1;
        }

        // neib_offset3d could hav negative items
        // bool is_local = (c[0] >= 0) && (c[1] >= 0) && (c[2] >= 0) && (c[0] <
        // 8) && (c[1] < 8) && (c[2] < 8);

        openvdb::Vec3fGrid::TreeType::LeafNodeType *new_vel_leaf =
            local_vel_leaf;
        openvdb::Vec3fGrid::TreeType::LeafNodeType *new_vel_weights_leaf =
            local_vel_weights_leaf;
        openvdb::FloatGrid::TreeType::LeafNodeType *new_sdf_leaf =
            local_sdf_leaf;

        // if (!is_local) {
        // cache the neighbor leafs
        if (!neib_sdf_leaf_ptr_cache[in_neib]) {
          neib_vel_leaf_ptr_cache[in_neib] =
              vel_accessor.touchLeaf(write_coord);
          neib_vel_weights_leaf_ptr_cache[in_neib] =
              vel_weights_accessor.touchLeaf(write_coord);
          neib_sdf_leaf_ptr_cache[in_neib] =
              liquid_sdf_accessor.touchLeaf(write_coord);
        }

        {
          new_vel_leaf = neib_vel_leaf_ptr_cache[in_neib];
          new_vel_weights_leaf = neib_vel_weights_leaf_ptr_cache[in_neib];
          new_sdf_leaf = neib_sdf_leaf_ptr_cache[in_neib];
        }
        //}

        openvdb::Index write_offset = new_vel_leaf->coordToOffset(write_coord);
        openvdb::Vec3f weights{packed_weight[0], packed_weight[1],
                               packed_weight[2]};
        openvdb::Vec3f weighted_vel = pvel * weights;

        // write weighted velocity
        auto original_weighted_vel = new_vel_leaf->getValue(write_offset);
        new_vel_leaf->setValueOn(write_offset,
                                 weighted_vel + original_weighted_vel);

        // write weights
        auto original_weights = new_vel_weights_leaf->getValue(write_offset);
        new_vel_weights_leaf->setValueOn(write_offset,
                                         weights + original_weights);

        // phi
        // compare
        float original_sdf = new_sdf_leaf->getValue(write_offset);
        new_sdf_leaf->setValueOn(
            write_offset,
            std::min(original_sdf, dist_to_phi_sample - m_particle_radius));
      } // end for its 27 neighbor

    } // end for all points in this leaf
  }   // end operator()

  // join operator
  void join(const particle_to_grid_reducer &other) {
    // velocity
    for (auto leaf = other.m_velocity->tree().beginLeaf(); leaf; ++leaf) {
      auto *newLeaf = m_velocity->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        auto &tree = const_cast<openvdb::Vec3fGrid &>(*other.m_velocity).tree();
        m_velocity->tree().addLeaf(
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

    // velocity weights
    for (auto leaf = other.m_velocity_weights->tree().beginLeaf(); leaf;
         ++leaf) {
      auto *newLeaf = m_velocity_weights->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        auto &tree =
            const_cast<openvdb::Vec3fGrid &>(*other.m_velocity_weights).tree();
        m_velocity_weights->tree().addLeaf(
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

    // liquid sdf
    for (auto leaf = other.m_liquid_sdf->tree().beginLeaf(); leaf; ++leaf) {
      auto *newLeaf = m_liquid_sdf->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        // it doesn't matter what const it put back in the other leaf, since it
        // is going to be deleted anyway
        auto &tree =
            const_cast<openvdb::FloatGrid &>(*other.m_liquid_sdf).tree();
        m_liquid_sdf->tree().addLeaf(
            tree.template stealNode<openvdb::FloatGrid::TreeType::LeafNodeType>(
                leaf->origin(), 1.f, false));
      } else {
        // update current sdf
        for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
          auto val = newLeaf->getValue(iter.offset());
          val = std::min(val, *iter);
          newLeaf->setValueOn(iter.offset(), val);
        }
      }
    }
  }

  // particle radius
  float m_particle_radius;

  // neighbor leaf pointer cache
  std::array<openvdb::Vec3fGrid::TreeType::LeafNodeType *, 27>
      neib_vel_leaf_ptr_cache;
  std::array<openvdb::Vec3fGrid::TreeType::LeafNodeType *, 27>
      neib_vel_weights_leaf_ptr_cache;
  std::array<openvdb::FloatGrid::TreeType::LeafNodeType *, 27>
      neib_sdf_leaf_ptr_cache;

  // constant distances to the center voxel
  std::shared_ptr<std::array<__m128, 27>> x_offset_to_center_voxel_center;
  std::shared_ptr<std::array<__m128, 27>> y_offset_to_center_voxel_center;
  std::shared_ptr<std::array<__m128, 27>> z_offset_to_center_voxel_center;
  std::shared_ptr<std::array<openvdb::Coord, 27>> loop_order;

  // pre-allocated space for computation
  __m128 x_dist_particle_to_sample;
  __m128 y_dist_particle_to_sample;
  __m128 z_dist_particle_to_sample;

  float dist_to_phi_sample;

  // reduction result
  openvdb::Vec3fGrid::Ptr m_velocity;
  openvdb::Vec3fGrid::Ptr m_velocity_weights;
  openvdb::FloatGrid::Ptr m_liquid_sdf;

  // transform information
  openvdb::math::Transform::Ptr m_transform;
};

// first step, for some of the velocity voxel
// one of its weights is non zero because some particle touches it
// however the velocity of other velocity is missing
// deduce it from its 27 neighboring voxel that has non-zero weights
// on this component
struct deduce_missing_velocity_and_normalize {
  deduce_missing_velocity_and_normalize(
      openvdb::Vec3fGrid::Ptr in_velocity_weights,
      openvdb::Vec3fGrid::Ptr in_original_velocity)
      : original_velocity_accessor(in_original_velocity->getConstAccessor()),
        original_weights_accessor(in_velocity_weights->getConstAccessor()) {
    m_velocity_weights = in_velocity_weights;
    m_original_velocity = in_original_velocity;
  }

  void operator()(openvdb::Vec3fGrid::TreeType::LeafNodeType &vel_leaf,
                  openvdb::Index leafpos) const {
    const auto *oweight_leaf =
        original_weights_accessor.probeConstLeaf(vel_leaf.origin());
    const auto *ovelocity_leaf =
        original_velocity_accessor.probeConstLeaf(vel_leaf.origin());

    if (!oweight_leaf) {
      printf("velocity voxel on, weight voxel not on. exit\n");
      exit(-1);
    }
    float epsl = 1e-4f;
    float weight_threshold = 1e-1f;
    for (openvdb::Index offset = 0; offset < vel_leaf.SIZE; offset++) {
      if (ovelocity_leaf->isValueMaskOff(offset)) {
        continue;
      }

      openvdb::Vec3f vel = ovelocity_leaf->getValue(offset);
      // check its three component
      for (int i_component = 0; i_component < 3; i_component++) {
        float weight = oweight_leaf->getValue(offset)[i_component];
        // if (weight < weight_threshold) {
        //   // we have a voxel that is touched but does not have weights and
        //   // velocity deduce from neighbor
        //   openvdb::Coord c = ovelocity_leaf->offsetToGlobalCoord(offset);

        //   float total_weights = 0;
        //   float weighted_component = 0;

        //   for (int ii = -2; ii <= 2; ii++) {
        //     for (int jj = -2; jj <= 2; jj++) {
        //       for (int kk = -2; kk <= 2; kk++) {
        //         total_weights += original_weights_accessor.getValue(
        //             c + openvdb::Coord{ii, jj, kk})[i_component];
        //         weighted_component += original_velocity_accessor.getValue(
        //             c + openvdb::Coord{ii, jj, kk})[i_component];
        //       }
        //     }
        //   }

          // if (total_weights == 0) {
          //   printf("a voxel is touched and has zero channel weight, but its "
          //          "neighbor voxel dont have touched channel\n");
          //   std::cout << "coordinate: " << c << std::endl;
          //   exit(-1);
          // }

          //vel[i_component] = weighted_component / (total_weights);
       // } else {
          // weight !=0
          
            vel[i_component] /= weight+1e-4;
        //}
      } // end for three component find missing velocity

      // store the weighted version
      vel_leaf.setValueOn(offset, vel);
    } // end for all voxels in this leaf node
  }   // end operator()

  openvdb::Vec3fGrid::ConstAccessor original_weights_accessor;
  openvdb::Vec3fGrid::ConstAccessor original_velocity_accessor;

  openvdb::Vec3fGrid::Ptr m_velocity_weights;
  openvdb::Vec3fGrid::Ptr m_original_velocity;
};
} // namespace

void FLIP_vdb::Advect(float dt, float dx,
                      openvdb::points::PointDataGrid::Ptr &particles,
                      openvdb::Vec3fGrid::Ptr &velocity,
                      openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                      float pic_component, int RK_ORDER) {
  auto update_FLIP_velocity = [&](openvdb::points::PointDataTree::LeafNodeType
                                      &leaf,
                                  openvdb::Index leafpos) {
    // Attribute reader
    // Extract the position attribute from the leaf by name (P is position).
    const openvdb::points::AttributeArray &positionArray =
        leaf.attributeArray("P");
    // Extract the velocity attribute from the leaf by name (v is velocity).
    openvdb::points::AttributeArray &velocityArray = leaf.attributeArray("v");

    // Create read handles for position and velocity
    openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>
        positionHandle(positionArray);
    openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                          FLIP_vdb::VelocityCodec>
        velocityHandle(velocityArray);

    auto ovaxr{velocity_after_p2g->getConstAccessor()};
    auto vaxr{velocity->getConstAccessor()};
    for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
      openvdb::Vec3R index_gpos =
          iter.getCoord().asVec3d() + positionHandle.get(*iter);
      auto original_vel =
          openvdb::tools::StaggeredBoxSampler::sample(ovaxr, index_gpos);
      auto updated_vel =
          openvdb::tools::StaggeredBoxSampler::sample(vaxr, index_gpos);
      auto old_pvel = velocityHandle.get(*iter);
      velocityHandle.set(*iter,
                         (pic_component)*updated_vel +
                             (1.0f - pic_component) *
                                 (updated_vel - original_vel + old_pvel));
    }
  };

  auto particle_man =
      openvdb::tree::LeafManager<openvdb::points::PointDataTree>(
          particles->tree());
  FLIP_vdb::custom_move_points_and_set_flip_vel(
      *particles, *velocity, *velocity_after_p2g, pic_component, dt, dx,
      RK_ORDER);
}
namespace {
struct custom_integrator {

  custom_integrator(const openvdb::Vec3fGrid &in_velocity, float in_dx,
                    float in_dt)
      : vaxr(in_velocity.tree()), m_velocity(in_velocity) {
    dx = in_dx;
    invdx = 1.0f / dx;
    dt = in_dt;
    dtinvx = dt / in_dx;
  }

  void integrate1(openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) const {
    ipos += V0 * dtinvx;
  }

  void integrate2(openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) const {
    openvdb::Vec3f V1 = openvdb::tools::StaggeredBoxSampler::sample(
        vaxr, ipos + 0.5f * V0 * dtinvx);
    ipos += V1 * dtinvx;
  }

  void integrate3(openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) const {
    openvdb::Vec3f V1 = openvdb::tools::StaggeredBoxSampler::sample(
        vaxr, ipos + 0.5f * V0 * dtinvx);
    openvdb::Vec3f V2 = openvdb::tools::StaggeredBoxSampler::sample(
        vaxr, ipos + dtinvx * (2.0f * V1 - V0));
    ipos += dtinvx * (V0 + 4.0f * V1 + V2) * (1.0f / 6.0f);
  }

  void integrate4(openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) const {
    openvdb::Vec3f V1 = openvdb::tools::StaggeredBoxSampler::sample(
        vaxr, ipos + 0.5f * V0 * dtinvx);
    openvdb::Vec3f V2 = openvdb::tools::StaggeredBoxSampler::sample(
        vaxr, ipos + 0.5f * V1 * dtinvx);
    openvdb::Vec3f V3 =
        openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + V2 * dtinvx);
    ipos += dtinvx * (V0 + 2.0f * (V1 + V2) + V3) * (1.0f / 6.0f);
  }

  float dx;
  float invdx;
  float dtinvx;
  float dt;
  openvdb::tree::ValueAccessor<const openvdb::Vec3fTree> vaxr;
  const openvdb::Vec3fGrid &m_velocity;
};
struct point_to_counter_reducer {

  point_to_counter_reducer(
      const float in_dt, const float in_dx,
      const openvdb::Vec3fGrid &in_velocity,
      const openvdb::Vec3fGrid &in_old_velocity, float pic_component,
      const std::vector<openvdb::points::PointDataTree::LeafNodeType *>
          &in_particles,
      int RK_order)
      : dt(in_dt), m_rk_order(RK_order), m_dx(in_dx), m_invdx(1.0f / in_dx),
        m_velocity(in_velocity), m_old_velocity(in_old_velocity),
        m_pic_component(pic_component), m_particles(in_particles) {
    m_integrator =
        std::make_shared<custom_integrator>(in_velocity, in_dx, in_dt);
    m_counter_grid = openvdb::points::PointDataGrid::create();
  }

  point_to_counter_reducer(const point_to_counter_reducer &other, tbb::split)
      : dt(other.dt), m_rk_order(other.m_rk_order), m_dx(other.m_dx),
        m_invdx(other.m_invdx), m_velocity(other.m_velocity),
        m_old_velocity(other.m_old_velocity),
        m_pic_component(other.m_pic_component), m_particles(other.m_particles) {
    m_integrator = std::make_shared<custom_integrator>(m_velocity, m_dx, dt);
    m_counter_grid = openvdb::points::PointDataGrid::create();
  }

  // loop over ranges of flattened particle leaves
  void operator()(const tbb::blocked_range<openvdb::Index> &r) {
    using namespace openvdb::tools::local_util;
    auto counter_axr{m_counter_grid->getAccessor()};
    auto vaxr{m_velocity.getConstUnsafeAccessor()};
    auto old_vaxr{m_old_velocity.getConstUnsafeAccessor()};

    std::function<void(openvdb::Vec3f & ipos, const openvdb::Vec3f &V0)>
        movefunc;

    switch (m_rk_order) {
    case 1:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate1(ipos, V0);
      };
      break;
    case 2:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate2(ipos, V0);
      };
      break;
    case 3:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate3(ipos, V0);
      };
      break;
    case 4:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate4(ipos, V0);
      };
      break;
    default:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate1(ipos, V0);
      };
    }

    // leaf iter
    for (auto liter = r.begin(); liter != r.end(); ++liter) {
      auto &leaf = *m_particles[liter];

      // attributes
      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      openvdb::points::AttributeArray &positionArray = leaf.attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      openvdb::points::AttributeArray &velocityArray = leaf.attributeArray("v");

      // Create read handles for position and velocity
      openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                            FLIP_vdb::PositionCodec>
          positionHandle(positionArray);
      openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                            FLIP_vdb::VelocityCodec>
          velocityHandle(velocityArray);

      // point index space source position
      openvdb::Vec3f pIspos;
      // advection velocity
      openvdb::Vec3f adv_vel;
      // old velocity
      openvdb::Vec3f old_vel;
      // particle velocity
      openvdb::Vec3f particle_vel;

      // point index space target position after move
      openvdb::Vec3f pItpos;

      // point world pos
      openvdb::Vec3f pWtpos;

      // the target voxel coordinate
      openvdb::Coord ptCoord;
      // loop over all particles
      openvdb::points::PointDataTree::LeafNodeType *writing_leaf;
      // std::vector<
      //	std::pair<openvdb::Vec3f, openvdb::Vec3f>>* writing_vector;

      std::vector<std::tuple<uint16_t, openvdb::Index32, openvdb::Index32>>
          *writing_offset_index_leafpos;

      // old leaf beging and end
      openvdb::Coord olbegin{openvdb::Coord::max()};
      openvdb::Coord olend{openvdb::Coord::min()};
      float flip_component = (1.0f - m_pic_component);
      for (auto piter = leaf.beginIndexOn(); piter; ++piter) {
        pIspos = piter.getCoord().asVec3s() + positionHandle.get(*piter);
        particle_vel = velocityHandle.get(*piter);

        /*adv_vel = openvdb::tools::StaggeredBoxSampler::sample(vaxr, pIspos);
        old_vel = openvdb::tools::StaggeredBoxSampler::sample(old_vaxr,
        pIspos);*/
        adv_vel = StaggeredBoxSampler::sample(vaxr, pIspos);
        old_vel = StaggeredBoxSampler::sample(old_vaxr, pIspos);
        // update the velocity of the particle
        /*particle_vel = (m_pic_component)*adv_vel + (1.0f - m_pic_component) *
         * (adv_vel - old_vel + particle_vel);*/
        particle_vel = adv_vel + flip_component * (-old_vel + particle_vel);
        // pItpos = pIspos + dt * adv_vel * m_invdx;
        pItpos = pIspos;
        movefunc(pItpos, adv_vel);

        ptCoord = openvdb::Coord{floorVec3(pItpos + openvdb::Vec3f{0.5f})};

        // directly change the original attribute to the target voxel position
        // later it will be transfered to the new position
        positionHandle.set(*piter, pItpos - ptCoord);
        velocityHandle.set(*piter, particle_vel);
        // check if we are writing to the previous leaf?

        if ((ptCoord[0] >= olbegin[0]) && (ptCoord[1] >= olbegin[1]) &&
            (ptCoord[2] >= olbegin[2]) && (ptCoord[0] < olend[0]) &&
            (ptCoord[1] < olend[1]) && (ptCoord[2] < olend[2])) {
          // increment the counter
          uint16_t toffset = writing_leaf->coordToOffset(ptCoord);
          writing_leaf->setOffsetOn(toffset,
                                    writing_leaf->getValue(toffset) + 1);

          // append the velocity and index space position
          // writing_vector->push_back(std::make_pair(pItpos,
          // velocityHandle.get(*piter)));

          writing_offset_index_leafpos->push_back(
              std::make_tuple(toffset, *piter, liter));
        } // end if writing to same leaf
        else {
          // try to probe it to check if we have it in this tree already
          if (writing_leaf = counter_axr.probeLeaf(ptCoord)) {
          } else {
            writing_leaf = counter_axr.touchLeaf(ptCoord);
            toffset_oindex_oleafpos_hashmap[writing_leaf->origin()] =
                std::make_unique<std::vector<std::tuple<
                    uint16_t, openvdb::Index32, openvdb::Index32>>>();
          }

          // increment the counter
          uint16_t toffset = writing_leaf->coordToOffset(ptCoord);
          writing_leaf->setOffsetOn(toffset,
                                    writing_leaf->getValue(toffset) + 1);

          writing_offset_index_leafpos =
              toffset_oindex_oleafpos_hashmap[writing_leaf->origin()].get();
          writing_offset_index_leafpos->push_back(
              std::make_tuple(toffset, *piter, liter));
          // set the bounding box
          olbegin = writing_leaf->origin();
          olend = olbegin + openvdb::Coord{8};
        } // end else writing to the same leaf
      }   // end loop over all particles
    }     // end for range leaves
  }       // end operator

  void join(point_to_counter_reducer &other) {
    auto &grid = *other.m_counter_grid;
    // merge the counter grid
    for (auto leaf = grid.tree().beginLeaf(); leaf; ++leaf) {
      auto *newLeaf = m_counter_grid->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        // if the leaf doesn't yet exist in the new tree, steal it
        auto &tree = const_cast<openvdb::points::PointDataGrid &>(grid).tree();
        m_counter_grid->tree().addLeaf(
            tree.template stealNode<
                openvdb::points::PointDataTree::LeafNodeType>(leaf->origin(), 0,
                                                              false));
      } else {
        // otherwise increment existing values
        for (auto iter = leaf->beginValueOn(); iter; ++iter) {
          // auto original_counter = newLeaf->getValue(iter.offset());
          // newLeaf->setOffsetOn(iter.offset(), original_counter +
          // leaf->getValue(iter.offset()));
          //*(newLeaf->buffer().data()+iter.offset()) =
          //*(newLeaf->buffer().data() + iter.offset()) + iter.getValue();
          newLeaf->setOffsetOn(iter.offset(),
                               *(newLeaf->buffer().data() + iter.offset()) +
                                   iter.getValue());
        }
      }
    }

    for (auto tuplevec = other.toffset_oindex_oleafpos_hashmap.begin();
         tuplevec != other.toffset_oindex_oleafpos_hashmap.end(); ++tuplevec) {
      auto itr_in_this = toffset_oindex_oleafpos_hashmap.find(tuplevec->first);
      if (itr_in_this != toffset_oindex_oleafpos_hashmap.end()) {
        auto original_size = itr_in_this->second->size();
        itr_in_this->second->resize(original_size + tuplevec->second->size());
        std::copy(tuplevec->second->begin(), tuplevec->second->end(),
                  itr_in_this->second->begin() + original_size);
      } else {
        toffset_oindex_oleafpos_hashmap[tuplevec->first] =
            std::move(tuplevec->second);
      }
    }
  }

  // velocity integrator
  std::shared_ptr<custom_integrator> m_integrator;

  // time step
  const float dt;

  const int m_rk_order;
  // index to world transform
  // for the particles as well as the velocity
  const float m_dx;
  const float m_invdx;

  // the velocity field used to advect the particles
  const openvdb::Vec3fGrid &m_velocity;
  const openvdb::Vec3fGrid &m_old_velocity;
  float m_pic_component;

  // the source particles
  const std::vector<openvdb::points::PointDataTree::LeafNodeType *>
      &m_particles;

  // this is the reduction result
  openvdb::points::PointDataGrid::Ptr m_counter_grid;

  // hashmap storing target voxel offset, original attribute index, original
  // leaf position
  std::unordered_map<openvdb::Coord,
                     std::unique_ptr<std::vector<std::tuple<
                         uint16_t, openvdb::Index32, openvdb::Index32>>>>
      toffset_oindex_oleafpos_hashmap;
};

struct point_to_counter_reducer2 {

  point_to_counter_reducer2(
      const float in_dt, const float in_dx,
      const openvdb::FloatGrid &in_liquidsdf,
      const openvdb::Vec3fGrid &in_velocity,
      const openvdb::Vec3fGrid &in_velocity_to_be_advected,
      const openvdb::Vec3fGrid &in_old_velocity,
      const openvdb::FloatGrid &in_solid_sdf,
      const openvdb::Vec3fGrid &in_center_solid_grad,
      const openvdb::FloatGrid &in_center_solid_vel_n, float pic_component,
      const std::vector<openvdb::points::PointDataTree::LeafNodeType *>
          &in_particles,
      float surfacedist, int RK_order)
      : dt(in_dt), m_rk_order(RK_order), m_dx(in_dx), m_invdx(1.0f / in_dx),
        m_velocity(in_velocity),
        m_velocity_to_be_advected(in_velocity_to_be_advected),
        m_old_velocity(in_old_velocity), m_solid_sdf(in_solid_sdf),
        m_center_solidgrad(in_center_solid_grad),
        m_center_solidveln(in_center_solid_vel_n),
        m_pic_component(pic_component), m_particles(in_particles),
        m_liquidsdf(in_liquidsdf), m_surfacedist(surfacedist) {

    m_integrator =
        std::make_shared<custom_integrator>(in_velocity, in_dx, in_dt);
    m_counter_grid = openvdb::points::PointDataGrid::create();
  }

  point_to_counter_reducer2(const point_to_counter_reducer2 &other, tbb::split)
      : dt(other.dt), m_rk_order(other.m_rk_order), m_dx(other.m_dx),
        m_invdx(other.m_invdx), m_velocity(other.m_velocity),
        m_velocity_to_be_advected(other.m_velocity_to_be_advected),
        m_old_velocity(other.m_old_velocity),
        m_pic_component(other.m_pic_component), m_solid_sdf(other.m_solid_sdf),
        m_center_solidgrad(other.m_center_solidgrad),
        m_center_solidveln(other.m_center_solidveln),
        m_particles(other.m_particles), m_surfacedist(other.m_surfacedist),
        m_liquidsdf(other.m_liquidsdf) {
    m_integrator = std::make_shared<custom_integrator>(m_velocity, m_dx, dt);
    m_counter_grid = openvdb::points::PointDataGrid::create();
  }

  // loop over ranges of flattened particle leaves
  void operator()(const tbb::blocked_range<openvdb::Index> &r) {
    using namespace openvdb::tools::local_util;
    auto liquid_sdf_axr{m_liquidsdf.getConstUnsafeAccessor()};
    auto counter_axr{m_counter_grid->getAccessor()};
    auto vaxr{m_velocity.getConstUnsafeAccessor()};
    auto v_tobe_adv_axr{m_velocity_to_be_advected.getConstAccessor()};
    auto old_vaxr{m_old_velocity.getConstUnsafeAccessor()};

    auto sdfaxr{m_solid_sdf.getConstUnsafeAccessor()};
    auto naxr{m_center_solidgrad.getConstUnsafeAccessor()};
    auto vnaxr{m_center_solidveln.getConstUnsafeAccessor()};

    std::function<void(openvdb::Vec3f & ipos, const openvdb::Vec3f &V0)>
        movefunc;

    switch (m_rk_order) {
    case 1:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate1(ipos, V0);
      };
      break;
    case 2:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate2(ipos, V0);
      };
      break;
    case 3:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate3(ipos, V0);
      };
      break;
    case 4:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate4(ipos, V0);
      };
      break;
    default:
      movefunc = [&](openvdb::Vec3f &ipos, const openvdb::Vec3f &V0) {
        m_integrator->integrate1(ipos, V0);
      };
    }

    // leaf iter
    for (auto liter = r.begin(); liter != r.end(); ++liter) {
      auto &leaf = *m_particles[liter];

      // attributes
      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      openvdb::points::AttributeArray &positionArray = leaf.attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      openvdb::points::AttributeArray &velocityArray = leaf.attributeArray("v");

      // Create read handles for position and velocity
      openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                            FLIP_vdb::PositionCodec>
          positionHandle(positionArray);
      openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                            FLIP_vdb::VelocityCodec>
          velocityHandle(velocityArray);

      // point index space source position
      openvdb::Vec3f pIspos;
      // advection velocity
      openvdb::Vec3f adv_vel;

      // carried_velocity velocity
      openvdb::Vec3f carried_vel;

      // old velocity
      openvdb::Vec3f old_vel;

      // particle velocity
      openvdb::Vec3f particle_vel;

      // point index space target position after move
      openvdb::Vec3f pItpos;

      // point world pos
      openvdb::Vec3f pWtpos;

      // the target voxel coordinate
      openvdb::Coord ptCoord;
      // loop over all particles
      openvdb::points::PointDataTree::LeafNodeType *writing_leaf;
      // std::vector<
      //	std::pair<openvdb::Vec3f, openvdb::Vec3f>>* writing_vector;

      std::vector<std::tuple<uint16_t, openvdb::Index32, openvdb::Index32>>
          *writing_offset_index_leafpos;

      // old leaf beging and end
      openvdb::Coord olbegin{openvdb::Coord::max()};
      openvdb::Coord olend{openvdb::Coord::min()};
      float flip_component = (1.0f - m_pic_component);
      float deep_threshold = -4.0 * m_dx;
      float touch_threshold = 0.6f * m_dx;
      float invdx = 1.0f / m_dx;
      float dtinvdx = dt / m_dx;

      bool adv_same_field =
          m_velocity.treePtr() == m_velocity_to_be_advected.treePtr();
      for (auto piter = leaf.beginIndexOn(); piter; ++piter) {
        pIspos = piter.getCoord().asVec3s() + positionHandle.get(*piter);
        particle_vel = velocityHandle.get(*piter);

        /*adv_vel = openvdb::tools::StaggeredBoxSampler::sample(vaxr, pIspos);
        old_vel = openvdb::tools::StaggeredBoxSampler::sample(old_vaxr,
        pIspos);*/
        adv_vel = StaggeredBoxSampler::sample(vaxr, pIspos);
        old_vel = StaggeredBoxSampler::sample(old_vaxr, pIspos);
        float p_liquidsdf =
            openvdb::tools::BoxSampler::sample(liquid_sdf_axr, pIspos);
        float t_coef = 1;
        if (p_liquidsdf < 0 && p_liquidsdf >= -m_surfacedist) {
          t_coef = p_liquidsdf / -m_surfacedist;
          t_coef = std::min(std::max(t_coef, 0.0f), 1.0f);
        }
        if (p_liquidsdf >= 0) {
          t_coef = 0;
        }
        if (m_surfacedist > 0)
          flip_component = t_coef * flip_component + (1.0f - t_coef) * 0.95f;
        if (adv_same_field) {
          carried_vel = adv_vel;
        } else {
          carried_vel = StaggeredBoxSampler::sample(v_tobe_adv_axr, pIspos);
        }

        // update the velocity of the particle
        /*particle_vel = (m_pic_component)*adv_vel + (1.0f - m_pic_component) *
         * (adv_vel - old_vel + particle_vel);*/

        particle_vel = adv_vel + flip_component * (-old_vel + particle_vel);

        pItpos = pIspos;
        movefunc(pItpos, adv_vel);
        ptCoord = openvdb::Coord{floorVec3(pItpos + openvdb::Vec3f{0.5f})};

        float new_pos_solid_sdf = openvdb::tools::BoxSampler::sample(
            sdfaxr, pItpos + openvdb::Vec3f{0.5});
        // collision detection, see if this coordinate is inside the solid
        if (new_pos_solid_sdf < 0) {

          if (new_pos_solid_sdf < deep_threshold) {
            // too deep, just continue to next particle
            continue;
          }
          // inside, but still can be saved, use the surface normal to move back
          // the particle
          auto snormal = naxr.getValue(ptCoord);
          // move particle out
          pItpos -= new_pos_solid_sdf * snormal * invdx * 1.0f;
          ptCoord = openvdb::Coord{floorVec3(pItpos + openvdb::Vec3f{0.5f})};
          // handle velocity bounces
          particle_vel += 
              (vnaxr.getValue(ptCoord) - snormal.dot(particle_vel)) * snormal;

        } // end if surface normal exist

        // pItpos = pIspos + dt * adv_vel * m_invdx;

        // directly change the original attribute to the target voxel position
        // later it will be transfered to the new position
        positionHandle.set(*piter, pItpos - ptCoord);
        velocityHandle.set(*piter, particle_vel);
        // check if we are writing to the previous leaf?

        if ((ptCoord[0] >= olbegin[0]) && (ptCoord[1] >= olbegin[1]) &&
            (ptCoord[2] >= olbegin[2]) && (ptCoord[0] < olend[0]) &&
            (ptCoord[1] < olend[1]) && (ptCoord[2] < olend[2])) {
          // increment the counter
          uint16_t toffset = writing_leaf->coordToOffset(ptCoord);
          auto existing_par = writing_leaf->getValue(toffset);
          if (existing_par > 27) {
            continue;
          }
          writing_leaf->setOffsetOn(toffset, existing_par + 1);

          // append the velocity and index space position
          // writing_vector->push_back(std::make_pair(pItpos,
          // velocityHandle.get(*piter)));

          writing_offset_index_leafpos->push_back(
              std::make_tuple(toffset, *piter, liter));
        } // end if writing to same leaf
        else {
          // try to probe it to check if we have it in this tree already
          if (writing_leaf = counter_axr.probeLeaf(ptCoord)) {
          } else {
            writing_leaf = counter_axr.touchLeaf(ptCoord);
            toffset_oindex_oleafpos_hashmap[writing_leaf->origin()] =
                std::make_unique<std::vector<std::tuple<
                    uint16_t, openvdb::Index32, openvdb::Index32>>>();
          }
          writing_offset_index_leafpos =
              toffset_oindex_oleafpos_hashmap[writing_leaf->origin()].get();

          // set the bounding box
          olbegin = writing_leaf->origin();
          olend = olbegin + openvdb::Coord{8};

          // increment the counter
          uint16_t toffset = writing_leaf->coordToOffset(ptCoord);
          auto existing_par = writing_leaf->getValue(toffset);
          if (existing_par > 27) {
            continue;
          }

          writing_leaf->setOffsetOn(toffset, existing_par + 1);
          writing_offset_index_leafpos->push_back(
              std::make_tuple(toffset, *piter, liter));

        } // end else writing to the same leaf
      }   // end loop over all particles
    }     // end for range leaves
  }       // end operator

  void join(point_to_counter_reducer2 &other) {
    auto &grid = *other.m_counter_grid;
    // merge the counter grid
    for (auto leaf = grid.tree().beginLeaf(); leaf; ++leaf) {
      auto *newLeaf = m_counter_grid->tree().probeLeaf(leaf->origin());
      if (!newLeaf) {
        // if the leaf doesn't yet exist in the new tree, steal it
        auto &tree = const_cast<openvdb::points::PointDataGrid &>(grid).tree();
        m_counter_grid->tree().addLeaf(
            tree.template stealNode<
                openvdb::points::PointDataTree::LeafNodeType>(leaf->origin(), 0,
                                                              false));
      } else {
        // otherwise increment existing values
        for (auto iter = leaf->beginValueOn(); iter; ++iter) {
          // auto original_counter = newLeaf->getValue(iter.offset());
          // newLeaf->setOffsetOn(iter.offset(), original_counter +
          // leaf->getValue(iter.offset()));
          //*(newLeaf->buffer().data()+iter.offset()) =
          //*(newLeaf->buffer().data() + iter.offset()) + iter.getValue();
          newLeaf->setOffsetOn(iter.offset(),
                               *(newLeaf->buffer().data() + iter.offset()) +
                                   iter.getValue());
        }
      }
    }

    for (auto tuplevec = other.toffset_oindex_oleafpos_hashmap.begin();
         tuplevec != other.toffset_oindex_oleafpos_hashmap.end(); ++tuplevec) {
      auto itr_in_this = toffset_oindex_oleafpos_hashmap.find(tuplevec->first);
      if (itr_in_this != toffset_oindex_oleafpos_hashmap.end()) {
        auto original_size = itr_in_this->second->size();
        itr_in_this->second->resize(original_size + tuplevec->second->size());
        std::copy(tuplevec->second->begin(), tuplevec->second->end(),
                  itr_in_this->second->begin() + original_size);
      } else {
        toffset_oindex_oleafpos_hashmap[tuplevec->first] =
            std::move(tuplevec->second);
      }
    }
  }

  // velocity integrator
  std::shared_ptr<custom_integrator> m_integrator;

  // time step
  const float dt;
  const float m_surfacedist;
  const int m_rk_order;
  // index to world transform
  // for the particles as well as the velocity
  const float m_dx;
  const float m_invdx;

  // the velocity field used to advect the particles
  const openvdb::Vec3fGrid &m_velocity;
  const openvdb::FloatGrid &m_liquidsdf;
  const openvdb::Vec3fGrid &m_velocity_to_be_advected;
  const openvdb::Vec3fGrid &m_old_velocity;
  float m_pic_component;

  const openvdb::FloatGrid &m_solid_sdf;
  const openvdb::Vec3fGrid &m_center_solidgrad;
  const openvdb::FloatGrid &m_center_solidveln;

  // the source particles
  const std::vector<openvdb::points::PointDataTree::LeafNodeType *>
      &m_particles;

  // this is the reduction result
  openvdb::points::PointDataGrid::Ptr m_counter_grid;

  // hashmap storing target voxel offset, original attribute index, original
  // leaf position
  std::unordered_map<openvdb::Coord,
                     std::unique_ptr<std::vector<std::tuple<
                         uint16_t, openvdb::Index32, openvdb::Index32>>>>
      toffset_oindex_oleafpos_hashmap;
};

} // namespace
void FLIP_vdb::custom_move_points_and_set_flip_vel(
    openvdb::points::PointDataGrid &in_out_points,
    const openvdb::Vec3fGrid &in_velocity_field,
    const openvdb::Vec3fGrid &in_old_velocity, float PIC_component, float dt,
    float dx, int RK_order) {

  std::vector<openvdb::points::PointDataTree::LeafNodeType *> particle_leaves;
  in_out_points.tree().getNodes(particle_leaves);

  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").start();
  auto reducer = std::make_unique<point_to_counter_reducer>(
      dt, dx, in_velocity_field, in_old_velocity, PIC_component,
      particle_leaves, RK_order);
  tbb::parallel_reduce(
      tbb::blocked_range<openvdb::Index>(0, particle_leaves.size(), 10),
      *reducer);
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").stop();
  // compose the result

  auto newTree_leafman{
      openvdb::tree::LeafManager<openvdb::points::PointDataTree>(
          reducer->m_counter_grid->tree())};

  auto p_desc = openvdb::points::AttributeSet::Descriptor::create(
      position_attribute::attributeType());
  auto pv_desc =
      p_desc->duplicateAppend("v", velocity_attribute::attributeType());
  auto set_new_attribute_list =
      [&](openvdb::points::PointDataTree::LeafNodeType &leaf,
          openvdb::Index leafpos) {
        using namespace openvdb::tools::local_util;

        std::vector<int> voxel_particle_count;
        voxel_particle_count.resize(leaf.size());
        std::vector<openvdb::PointDataIndex32> index_ends;
        index_ends.assign(leaf.size(), 0);
        voxel_particle_count[0] = leaf.getValue(0);
        index_ends[0] = voxel_particle_count[0];
        for (auto offset = 1; offset < leaf.size(); ++offset) {
          voxel_particle_count[offset] = leaf.getValue(offset);
          index_ends[offset] =
              index_ends[offset - 1] + voxel_particle_count[offset];
        }

        // according to the index space leaf position, assign the particles to
        // the final attribute list
        // printf("leafcounter %d, attrcounter:%d\n", *index_ends.rbegin(),
        // reducer.toffset_oindex_oleafpos_hashmap[leaf.origin()]->size());
        // create the attribute set

        // auto local_pv_descriptor = pv_descriptor();
        leaf.initializeAttributes(p_desc, *index_ends.rbegin());
        leaf.appendAttribute(leaf.attributeSet().descriptor(), pv_desc, 1);

        // attribute writer
        leaf.setOffsets(index_ends);

        // set the positions and velocities
        // get the new attribute arrays
        openvdb::points::AttributeArray &posarray = leaf.attributeArray("P");
        openvdb::points::AttributeArray &varray = leaf.attributeArray("v");

        // Create read handles for position and velocity
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::PositionCodec>
            posWHandle(posarray);
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::VelocityCodec>
            vWHandle(varray);

        openvdb::Vec3i pcoord;
        int writing_offset;
        int writer_index;

        // move from the original leaves
        for (size_t i = 0;
             i <
             reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()]->size();
             i++) {
          auto &tooiol_vec =
              *reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()];
          writing_offset = std::get<0>(tooiol_vec[i]);
          writer_index =
              index_ends[writing_offset] - voxel_particle_count[writing_offset];
          voxel_particle_count[writing_offset]--;

          posarray.set(
              writer_index,
              particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("P"),
              std::get<1>(tooiol_vec[i]));
          varray.set(
              writer_index,
              particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("v"),
              std::get<1>(tooiol_vec[i]));
        }
      };
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose").start();
  // printf("compose_start_for_each\n");
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/compose").start();
  newTree_leafman.foreach (set_new_attribute_list);
  // printf("compose_end_for_each\n");
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/compose").stop();
  reducer->m_counter_grid->setName("new_counter_grid");
  auto voxel_center_transform =
      openvdb::math::Transform::createLinearTransform(dx);
  reducer->m_counter_grid->setTransform(voxel_center_transform);
  // openvdb::io::File("new_advect.vdb").write({reducer.m_counter_grid});
  // printf("compose_start_replace_tree\n");
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/replace").start();
  in_out_points.setTree(reducer->m_counter_grid->treePtr());
  // printf("compose_end_replace_tree\n");
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/replace").stop();
  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose").stop();
}

namespace {
struct p2g_collector {
  p2g_collector(openvdb::FloatGrid::Ptr in_liquid_sdf,
                openvdb::Vec3fGrid::Ptr in_velocity,
                openvdb::Vec3fGrid::Ptr in_velocity_weight,
                openvdb::points::PointDataGrid::Ptr in_particles,
                float in_particle_radius) {
    m_liquid_sdf = in_liquid_sdf;
    m_velocity = in_velocity;
    m_velocity_weight = in_velocity_weight;
    m_particles = in_particles;
    m_particle_radius = in_particle_radius;
    // set the loop order and offset of the u v w phi sampling point
    // with respect to the center voxel center

    // u:  (-0.5, 0, 0)
    // v:  (0, -0.5, 0)
    // w:  (0, 0, -0.5)
    // phi:(0, 0, 0)
    // note we use set so the
    // R0 R1 R2 R3 correspond to the 0,1,2,3 argument
    // when later extracting using _mm_storer_ps, we get float[0] = arg0
    // correctly see
    // http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_store.htm
    // see
    // http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_set.htm

    __m128 xpack = _mm_set_ps(-0.5f, 0.f, 0.f, 0.f);
    __m128 ypack = _mm_set_ps(0.f, -0.5f, 0.f, 0.f);
    __m128 zpack = _mm_set_ps(0.f, 0.f, -0.5f, 0.f);
    for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
      int ijk = ivoxel;
      int basex = ijk / 9;
      ijk -= 9 * basex;
      int basey = ijk / 3;
      ijk -= 3 * basey;
      int basez = ijk;
      // becomes -1 -> 1

      basex -= 1;
      basey -= 1;
      basez -= 1;
      // broadcast four float as the base
      __m128 basex4 = _mm_set_ps1(float(basex));
      __m128 basey4 = _mm_set_ps1(float(basey));
      __m128 basez4 = _mm_set_ps1(float(basez));

      loop_order.at(ivoxel) = openvdb::Coord{basex, basey, basez};

      x_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basex4, xpack);
      y_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basey4, ypack);
      z_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basez4, zpack);
    }
  }

  // packed structure storing a particle leaf
  // its attribute handle
  struct particle_leaf {
    particle_leaf() {
      m_leafptr = nullptr;
      m_p_handle_ptr.reset();
      m_v_handle_ptr.reset();
    }

    // iterator over points in a voxel
    struct particle_iter {
      particle_iter() : m_parent(nullptr) {
        m_item = 0;
        m_indexend = 0;
        m_indexbegin = 0;
      }

      particle_iter(const particle_leaf *in_parent, openvdb::Index in_offset) {
        set(in_parent, in_offset);
      }

      particle_iter(const particle_iter &other) = default;

      void set(const particle_leaf *in_parent, openvdb::Index in_offset) {
        m_parent = in_parent;

        if (!in_parent->m_leafptr) {
          m_item = 0;
          m_indexbegin = 0;
          m_indexend = 0;
          return;
        }
        m_indexend = in_parent->m_leafptr->getValue(in_offset);
        if (in_offset == 0) {
          m_indexbegin = 0;
        } else {
          m_indexbegin = in_parent->m_leafptr->getValue(in_offset - 1);
        }
        m_item = m_indexbegin;
      }

      void operator=(const particle_iter &other) {
        m_item = other.m_item;
        m_parent = other.m_parent;
        m_indexbegin = other.m_indexbegin;
        m_indexend = other.m_indexend;
      }

      operator bool() const { return m_item < m_indexend; }

      particle_iter &operator++() {
        m_item++;
        return *this;
      }

      openvdb::Index operator*() const { return m_item; }

      openvdb::Index m_item;
      openvdb::Index m_indexbegin, m_indexend;
      const particle_leaf *m_parent;
    };

    particle_leaf(
        const openvdb::points::PointDataTree::LeafNodeType *in_leaf_ptr)
        : m_leafptr(in_leaf_ptr) {
      if (in_leaf_ptr) {
        // Attribute reader
        // Extract the position attribute from the leaf by name (P is position).
        const openvdb::points::AttributeArray &positionArray =
            in_leaf_ptr->attributeArray("P");
        // Extract the velocity attribute from the leaf by name (v is velocity).
        const openvdb::points::AttributeArray &velocityArray =
            in_leaf_ptr->attributeArray("v");

        //// Create read handles for position and velocity
        // openvdb::points::AttributeHandle<openvdb::Vec3f,
        // FLIP_vdb::PositionCodec> positionHandle(positionArray);
        // openvdb::points::AttributeHandle<openvdb::Vec3f,
        // FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);
        m_p_handle_ptr = openvdb::points::AttributeHandle<
            openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
        m_v_handle_ptr = openvdb::points::AttributeHandle<
            openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);
      }
    }

    // default?
    particle_leaf(const particle_leaf &other) {
      m_leafptr = other.m_leafptr;
      m_p_handle_ptr = other.m_p_handle_ptr;
      m_v_handle_ptr = other.m_v_handle_ptr;
    }

    particle_iter get_particle_iter(openvdb::Index offset) const {
      return particle_iter(this, offset);
    }

    particle_iter get_particle_iter(const openvdb::Coord &lgxyz) const {
      return particle_iter(this, m_leafptr->coordToOffset(lgxyz));
    }

    operator bool() const { return m_leafptr != nullptr; }

    openvdb::Vec3f get_p(openvdb::Index pos) const {
      return m_p_handle_ptr->get(pos);
    }

    openvdb::Vec3f get_v(openvdb::Index pos) const {
      return m_v_handle_ptr->get(pos);
    }

    openvdb::points::AttributeHandle<
        openvdb::Vec3f, FLIP_vdb::PositionCodec>::Ptr m_p_handle_ptr;
    openvdb::points::AttributeHandle<
        openvdb::Vec3f, FLIP_vdb::VelocityCodec>::Ptr m_v_handle_ptr;
    const openvdb::points::PointDataTree::LeafNodeType *m_leafptr;
  };

  void fill_particle_leafs(
      std::array<particle_leaf, 27> &in_out_leaves,
      const openvdb::Coord &center_origin,
      const openvdb::points::PointDataTree &in_particle_tree) const {
    // for all its neighbor nodes
    int counter = 0;
    for (int ii = -8; ii <= 8; ii += 8) {
      for (int jj = -8; jj <= 8; jj += 8) {
        for (int kk = -8; kk <= 8; kk += 8) {
          auto leafptr = in_particle_tree.probeConstLeaf(
              center_origin.offsetBy(ii, jj, kk));
          in_out_leaves[counter++] = particle_leaf(leafptr);
        }
      }
    }
  }

  // given the 27 particle leaves,
  // iterate all particles that could possibly contribute to the center leaf
  // each iterator can return the position, velocity, local coordinate relative
  // to the center leaf
  struct all_particle_iterator {
    all_particle_iterator(const std::array<particle_leaf, 27> &in_leaves)
        : m_leaves(in_leaves) {
      m_leafpos = 0;
      at_voxel = 0;
      // initially on the corner
      m_leafxyz = {-8, -8, -8};
      // it runs from [{-1,-1,-1}, {9,9,9} )
      // the size is 1+8+1 = 10
      // so there are at most 1000 voxels to iterate
      m_center_xyz = openvdb::Coord{-1, -1, -1};
      auto offset = 511u;
      // printf("setting voxel iter at offset%d\n",offset);
      m_voxel_particle_iter.set(&m_leaves[m_leafpos], offset);
      m_at_interior = false;
      // printf("voxel_iter set\n");
      if (!m_voxel_particle_iter) {
        move_to_next_on_voxel();
      }
    }

    void move_to_next_on_voxel() {
      do {
        // ran out of attribute in this voxel
        // find the next on voxel that will contribute
        // to the center voxel
        // allowed range: [-1,8] in x,y,z
        m_center_xyz[2]++;

        // only when it freshly turns 0 and 8 it cross the leaf border
        if (m_center_xyz[2] == 0) {
          m_leafpos++;
        } else if (m_center_xyz[2] == 8) {
          m_leafpos++;
        } else if (m_center_xyz[2] == 9) {
          m_center_xyz[2] = -1;
          m_leafpos -= 2;

          m_center_xyz[1]++;
          if (m_center_xyz[1] == 0) {
            m_leafpos += 3;
          } else if (m_center_xyz[1] == 8) {
            m_leafpos += 3;
          } else if (m_center_xyz[1] == 9) {
            m_center_xyz[1] = -1;
            m_leafpos -= 6;

            m_center_xyz[0]++;
            if (m_center_xyz[0] == 0) {
              m_leafpos += 9;
            } else if (m_center_xyz[0] == 8) {
              m_leafpos += 9;
            } else if (m_center_xyz[0] == 9) {
              m_leafpos += 9;
              if (m_leafpos != 27) {
                printf("particle iterator increment error! actual leafpos:%d\n",
                       m_leafpos);
                exit(-1);
              }
            };
          } // end if y overflow
        }   // end if z overflow

        // check leaf bount
        if (m_leafpos >= 27)
          return;

        // this leaf is empty
        if (m_leaves[m_leafpos].m_leafptr == nullptr) {
          continue;
        };

        int offset = m_leaves[m_leafpos].m_leafptr->coordToOffset(m_center_xyz);
        if (m_leaves[m_leafpos].m_leafptr->isValueOn(offset)) {
          m_voxel_particle_iter.set(&m_leaves[m_leafpos], offset);
          auto itercoord = m_center_xyz;
          m_at_interior =
              (itercoord[0] >= 1 && itercoord[0] <= 6 && itercoord[1] >= 1 &&
               itercoord[1] <= 6 && itercoord[2] >= 1 && itercoord[2] <= 6);
          return;
        }
      } while (true);
    } // end move to next on voxel

    bool is_interior_voxel() const { return m_at_interior; }

    openvdb::Coord getCoord() const { return m_center_xyz; }

    openvdb::Vec3f getP() const {
      return m_leaves[m_leafpos].get_p(*m_voxel_particle_iter);
    }

    openvdb::Vec3f getv() const {
      return m_leaves[m_leafpos].get_v(*m_voxel_particle_iter);
    }

    all_particle_iterator &operator++() {
      ++m_voxel_particle_iter;
      // printf("advancing index %d at voxel %d\n", *m_voxel_particle_iter,
      // at_voxel);
      if (!m_voxel_particle_iter) {
        move_to_next_on_voxel();
      } // end if ran out of particles in this voxel
      return *this;
    }

    operator bool() const { return m_voxel_particle_iter; }

    // local coordinates relative to the center leaf
    openvdb::Coord m_center_xyz;
    bool m_at_interior;
    openvdb::Coord m_g_xyz;
    int at_voxel;
    // attribute index in corresponding
    particle_leaf::particle_iter m_voxel_particle_iter;
    // the offset in each voxel
    uint32_t m_voxel_offset;
    // 0-26, the position in the particle leaf
    uint32_t m_leafpos;
    openvdb::Coord m_leafxyz;
    std::array<int, 27> visit_order;
    const std::array<particle_leaf, 27> &m_leaves;
  };

  // to be use by the velocity grid manager
  // collect the contribution of particles in current leaf
  // and its neighbor leafs
  void operator()(openvdb::Vec3fTree::LeafNodeType &leaf,
                  openvdb::Index leafpos) const {
    auto new_vel_leaf = m_velocity->tree().probeLeaf(leaf.origin());
    auto new_vel_weights_leaf =
        m_velocity_weight->tree().probeLeaf(leaf.origin());
    auto new_sdf_leaf = m_liquid_sdf->tree().probeLeaf(leaf.origin());

    const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
    const __m128 float1x4 = _mm_set_ps1(float(1));
    const __m128 float0x4 = _mm_set_ps1(float(0));

    // scatter style transfer
    // loop over all particles
    float tophix, tophiy, tophiz;
    float dx = m_liquid_sdf->voxelSize()[0];

    std::array<particle_leaf, 27> particle_leaves;
    fill_particle_leafs(particle_leaves, leaf.origin(), m_particles->tree());

    auto iter{all_particle_iterator(particle_leaves)};
    // pre-allocated space for computation
    __m128 x_dist_particle_to_sample;
    __m128 y_dist_particle_to_sample;
    __m128 z_dist_particle_to_sample;
    __m128 comp_result;
    __m128 comp_target;
    // the distance to the other cell center will always be greater
    // than zero, hence the last predicate is always true
    comp_target = _mm_set_ps(1.f, 1.f, 1.f, -1.f);
    float dist_to_phi_sample;
    // particle iterator
    for (; iter; ++iter) {

      auto voxelpos = iter.getP();
      auto pvel = iter.getv();
      // broadcast the variables
      __m128 particle_x = _mm_set_ps1(voxelpos[0]);
      __m128 particle_y = _mm_set_ps1(voxelpos[1]);
      __m128 particle_z = _mm_set_ps1(voxelpos[2]);

      // calculate the distance to the 27 neib uvw phi samples
      for (int ivoxel = 0; ivoxel < 27; ivoxel++) {

        // is it writing to this leaf?
        openvdb::Coord head = iter.getCoord() + loop_order.at(ivoxel);

        if (head[0] < 0 || head[1] < 0 || head[2] < 0 || head[0] > 7 ||
            head[1] > 7 || head[2] > 7) {
          continue;
        }

        auto write_offset = leaf.coordToOffset(head);

        // calculate the distance
        // arg(A,B): ret A-B
        // the absolute value trick: abs_mask: 01111111..32bit..1111 x 4
        // _mm_and_ps(abs_mask(), v);
        x_dist_particle_to_sample = _mm_and_ps(
            absmask,
            _mm_sub_ps(x_offset_to_center_voxel_center.at(ivoxel), particle_x));
        y_dist_particle_to_sample = _mm_and_ps(
            absmask,
            _mm_sub_ps(y_offset_to_center_voxel_center.at(ivoxel), particle_y));
        z_dist_particle_to_sample = _mm_and_ps(
            absmask,
            _mm_sub_ps(z_offset_to_center_voxel_center.at(ivoxel), particle_z));

        // the distance to the phi variable
        _mm_store_ss(&tophix, x_dist_particle_to_sample);
        _mm_store_ss(&tophiy, y_dist_particle_to_sample);
        _mm_store_ss(&tophiz, z_dist_particle_to_sample);
        dist_to_phi_sample =
            dx * std::sqrt(tophix * tophix + tophiy * tophiy + tophiz * tophiz);

        // phi
        // compare
        float original_sdf = new_sdf_leaf->getValue(write_offset);
        new_sdf_leaf->setValueOn(
            write_offset,
            std::min(original_sdf, dist_to_phi_sample - m_particle_radius));

        do {
          // if for [x,y,z]distance_to_sample, the first 3 values are all
          // greater than 1 then the weight is 0, so skip it
          uint32_t test = _mm_movemask_ps(
              _mm_cmpgt_ps(x_dist_particle_to_sample, comp_target));
          if (test == 0b00001111) {
            break;
          }
          test = _mm_movemask_ps(
              _mm_cmpgt_ps(y_dist_particle_to_sample, comp_target));
          if (test == 0b00001111) {
            break;
          }
          test = _mm_movemask_ps(
              _mm_cmpgt_ps(z_dist_particle_to_sample, comp_target));
          if (test == 0b00001111) {
            break;
          }

          // the uvw weights trilinear
          // transfer the distance to weight at the 27 voxels
          //(1-dist)
          // the far points now becomes negative
          x_dist_particle_to_sample =
              _mm_sub_ps(float1x4, x_dist_particle_to_sample);
          y_dist_particle_to_sample =
              _mm_sub_ps(float1x4, y_dist_particle_to_sample);
          z_dist_particle_to_sample =
              _mm_sub_ps(float1x4, z_dist_particle_to_sample);

          // turn everything positive or zero
          // now the dist_to_sample is actually the component-wise weight on the
          // voxel time to multiply them together
          x_dist_particle_to_sample =
              _mm_max_ps(float0x4, x_dist_particle_to_sample);
          y_dist_particle_to_sample =
              _mm_max_ps(float0x4, y_dist_particle_to_sample);
          z_dist_particle_to_sample =
              _mm_max_ps(float0x4, z_dist_particle_to_sample);

          // turn them into weights reduce to x
          x_dist_particle_to_sample =
              _mm_mul_ps(x_dist_particle_to_sample, y_dist_particle_to_sample);
          x_dist_particle_to_sample =
              _mm_mul_ps(x_dist_particle_to_sample, z_dist_particle_to_sample);
          //}//end for 27 voxel

          ////write to the grid
          // for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
          alignas(16) float packed_weight[4];

          _mm_storer_ps(packed_weight, x_dist_particle_to_sample);

          openvdb::Vec3f weights{packed_weight[0], packed_weight[1],
                                 packed_weight[2]};
          // write weighted velocity

          openvdb::Vec3f weighted_vel = pvel * weights;
          auto original_weighted_vel = new_vel_leaf->getValue(write_offset);
          new_vel_leaf->setValueOn(write_offset,
                                   weighted_vel + original_weighted_vel);
          // new_vel_leaf->modifyValue(write_offset,
          // [&weighted_vel](openvdb::Vec3f& in_out) {in_out += weighted_vel;
          // }); write weights
          auto original_weights = new_vel_weights_leaf->getValue(write_offset);
          new_vel_weights_leaf->setValueOn(write_offset,
                                           weights + original_weights);
          // new_vel_weights_leaf->modifyValue(write_offset,
          // [&weights](openvdb::Vec3f& in_out) {in_out += weights; });

        } while (false); // test if the velocity weight is zero
      }                  // end for 27 voxels
    }                    // end for all particles influencing this leaf
  }                      // end operator

  // particle radius
  float m_particle_radius;

  // neighbor leaf pointer cache

  // constant distances to the center voxel
  std::array<__m128, 27> x_offset_to_center_voxel_center;
  std::array<__m128, 27> y_offset_to_center_voxel_center;
  std::array<__m128, 27> z_offset_to_center_voxel_center;
  std::array<openvdb::Coord, 27> loop_order;

  openvdb::FloatGrid::Ptr m_liquid_sdf;
  openvdb::Vec3fGrid::Ptr m_velocity, m_velocity_weight;
  openvdb::points::PointDataGrid::Ptr m_particles;
};
} // namespace
void FLIP_vdb::particle_to_grid_collect_style(
    openvdb::points::PointDataGrid::Ptr &particles,
    openvdb::Vec3fGrid::Ptr &velocity,
    openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
    openvdb::Vec3fGrid::Ptr &velocity_weights,
    openvdb::FloatGrid::Ptr &liquid_sdf,
    openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf, float dx) {
  float particle_radius = 0.8f * dx * 1.01;
  velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
      particles->tree(), openvdb::Vec3f{0}, openvdb::TopologyCopy()));
  openvdb::tools::dilateActiveValues(
      velocity->tree(), 1,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);

  velocity_weights = velocity->deepCopy();

  auto voxel_center_transform =
      openvdb::math::Transform::createLinearTransform(dx);
  liquid_sdf->setTransform(voxel_center_transform);
  liquid_sdf->setTree(std::make_shared<openvdb::FloatTree>(
      velocity->tree(), 0.6f * dx * 1.01, openvdb::TopologyCopy()));

  auto collector_op{p2g_collector(liquid_sdf, velocity, velocity_weights,
                                  particles, particle_radius)};

  auto vleafman =
      openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity->tree());

  vleafman.foreach (collector_op, true);

  openvdb::Vec3fGrid::Ptr original_unweighted_velocity = velocity->deepCopy();

  openvdb::tree::LeafManager<openvdb::Vec3fGrid::TreeType>
      velocity_grid_manager(velocity->tree());

  auto velocity_normalizer = deduce_missing_velocity_and_normalize(
      velocity_weights, original_unweighted_velocity);

  velocity_grid_manager.foreach (velocity_normalizer, true, 1);

  //velocity = original_unweighted_velocity->deepCopy();
  // store the velocity just after the transfer
  velocity_after_p2g = velocity->deepCopy();
  velocity_after_p2g->setName("Velocity_After_P2G");
  pushed_out_liquid_sdf = liquid_sdf->deepCopy();
}

namespace {

using namespace openvdb;
bool boxes_overlap(Coord begin0, Coord end0, Coord begin1, Coord end1) {
  if (end0[0] <= begin1[0])
    return false; // a is left of b
  if (begin0[0] >= end1[0])
    return false; // a is right of b
  if (end0[1] <= begin1[1])
    return false; // a is above b
  if (begin0[1] >= end1[1])
    return false; // a is below b
  if (end0[2] <= begin1[2])
    return false; // a is in front of b
  if (begin0[2] >= end1[2])
    return false; // a is behind b
  return true;    // boxes overlap
}

// for each of the leaf nodes that overlaps the fill zone
// fill all voxels that are within the fill zone and bottoms
// are below the sea level
// the newly generated particle positions and offset in the attribute array
// will be recorded in std vectors associated with each leaf nodes
// it also removes all particles that are
// 1:outside the bounding box
// 2:in the fill layer but are above the sealevel

struct particle_fill_kill_op {
  using pos_offset_t = std::pair<openvdb::Vec3f, openvdb::Index>;
  using node_t = openvdb::points::PointDataGrid::TreeType::LeafNodeType;

  particle_fill_kill_op(tbb::concurrent_vector<node_t *> &in_nodes,
                        openvdb::FloatGrid::Ptr in_boundary_fill_kill_volume,
                        const openvdb::Coord &in_domain_begin,
                        const openvdb::Coord &in_domain_end, int fill_layer,
                        openvdb::math::Transform::Ptr in_transform,
                        openvdb::FloatGrid::Ptr in_solid_grid,
                        openvdb::Vec3fGrid::Ptr in_velocity_volume,
                        FLIP_vdb::descriptor_t::Ptr p_descriptor,
                        FLIP_vdb::descriptor_t::Ptr pv_descriptor)
      : m_leaf_nodes(in_nodes), m_p_descriptor(p_descriptor),
        m_pv_descriptor(pv_descriptor) {
    m_boundary_fill_kill_volume = in_boundary_fill_kill_volume;
    m_domain_begin = in_domain_begin;
    m_domain_end = in_domain_end;
    m_int_begin = in_domain_begin + openvdb::Coord{fill_layer};
    m_int_end = in_domain_end - openvdb::Coord{fill_layer};
    m_transform = in_transform;
    m_solid = in_solid_grid;
    m_velocity_volume = in_velocity_volume;
  } // end constructor

  bool voxel_inside_inner_box(const openvdb::Coord &voxel_coord) const {
    return (voxel_coord[0] >= m_int_begin[0] &&
            voxel_coord[1] >= m_int_begin[1] &&
            voxel_coord[2] >= m_int_begin[2] && voxel_coord[0] < m_int_end[0] &&
            voxel_coord[1] < m_int_end[1] && voxel_coord[2] < m_int_end[2]);
  }

  bool voxel_inside_outer_box(const openvdb::Coord &voxel_coord) const {
    return (voxel_coord[0] >= m_domain_begin[0] &&
            voxel_coord[1] >= m_domain_begin[1] &&
            voxel_coord[2] >= m_domain_begin[2] &&
            voxel_coord[0] < m_domain_end[0] &&
            voxel_coord[1] < m_domain_end[1] &&
            voxel_coord[2] < m_domain_end[2]);
  }
  // the leaf nodes here are assumed to either contain solid voxels
  // or has any corner outside the interior box
  void operator()(const tbb::blocked_range<openvdb::Index> &r) const {

    std::vector<openvdb::PointDataIndex32> new_attribute_offsets;
    std::vector<openvdb::Vec3f> new_positions;
    std::vector<openvdb::Vec3f> new_velocity;

    // solid is assumed to point to the minimum corner of this voxel
    auto solid_axr = m_solid->getConstAccessor();
    auto velocity_axr = m_velocity_volume->getConstAccessor();
    auto boundary_volume_axr = m_boundary_fill_kill_volume->getConstAccessor();

    float voxel_include_threshold = m_transform->voxelSize()[0] * 0.5;
    float particle_radius =
        0.55 * m_transform->voxelSize()[0] * std::sqrt(3) / 2.0;
    // minimum number of particle per voxel
    const int min_np = 8;
    // max number of particle per voxel
    const int max_np = 16;

    new_positions.reserve(max_np * 512);
    new_velocity.reserve(max_np * 512);
    // loop over the leaf nodes
    for (auto ileaf = r.begin(); ileaf != r.end(); ++ileaf) {
      node_t &leaf = *m_leaf_nodes[ileaf];

      // If this leaf is totally outside of the bounding box
      if (!boxes_overlap(leaf.origin(), leaf.origin() + openvdb::Coord{8},
                         m_domain_begin, m_domain_end)) {
        leaf.clearAttributes();
        continue;
      }

      // this leaf could potentially be a empty node without attribute
      // if so, create the position and velocity attribute
      if (leaf.attributeSet().size() == 0) {
        auto local_pv_descriptor = m_pv_descriptor;
        leaf.initializeAttributes(m_p_descriptor, 0);

        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);
      }

      // Randomize the point positions.
      std::random_device device;
      std::uniform_int_distribution<> intdistrib(0, 21474836);
      std::mt19937 gen(/*seed=*/device());
      size_t index = intdistrib(gen);
      // std::random_device device;
      // std::mt19937 generator(/*seed=*/device());
      // std::uniform_real_distribution<> distribution(-0.5, 0.5);

      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      const openvdb::points::AttributeArray &positionArray =
          leaf.attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      const openvdb::points::AttributeArray &velocityArray =
          leaf.attributeArray("v");

      // Create read handles for position and velocity
      openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>
          positionHandle(positionArray);
      openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>
          velocityHandle(velocityArray);

      // clear the new offset to be assigned
      new_attribute_offsets.clear();
      new_positions.clear();
      new_velocity.clear();
      openvdb::Index current_particle_count = 0;

      // at least reserve the original particles spaces
      new_positions.reserve(positionArray.size());
      new_velocity.reserve(velocityArray.size());
      new_attribute_offsets.reserve(leaf.SIZE);
      // emit new particles and transfer old particles
      for (openvdb::Index offset = 0; offset < leaf.SIZE; offset++) {
        openvdb::Index original_attribute_begin = 0;
        if (offset != 0) {
          original_attribute_begin = leaf.getValue(offset - 1);
        }
        const openvdb::Index original_attribute_end = leaf.getValue(offset);
        const auto voxel_gcoord = leaf.offsetToGlobalCoord(offset);

        // 1**********************************
        // domain boundary check
        if (!voxel_inside_outer_box(voxel_gcoord)) {
          new_attribute_offsets.push_back(current_particle_count);
          continue;
        }

        // 2**********************************
        // if the particle voxel has eight solid vertices
        bool all_solid = true;
        for (int ii = 0; ii < 2 && all_solid; ii++) {
          for (int jj = 0; jj < 2 && all_solid; jj++) {
            for (int kk = 0; kk < 2 && all_solid; kk++) {
              if (solid_axr.getValue(voxel_gcoord +
                                     openvdb::Coord{ii, jj, kk}) > 0) {
                all_solid = false;
              }
            }
          }
        }
        if (all_solid) {
          new_attribute_offsets.push_back(current_particle_count);
          continue;
        }

        // 3************************************
        // if it is not inside the fill zone
        // just emit the original particles
        if (voxel_inside_inner_box(voxel_gcoord)) {
          // test if it has any solid
          // if so, remove the particles with solid phi<0
          bool all_liquid = true;
          for (int ii = 0; ii < 2 && all_liquid; ii++) {
            for (int jj = 0; jj < 2 && all_liquid; jj++) {
              for (int kk = 0; kk < 2 && all_liquid; kk++) {
                if (solid_axr.getValue(voxel_gcoord +
                                       openvdb::Coord{ii, jj, kk}) < 0) {
                  all_liquid = false;
                }
              }
            }
          }

          for (int i_emit = original_attribute_begin;
               i_emit < original_attribute_end; i_emit++) {
            auto current_pos = positionHandle.get(i_emit);
            if (all_liquid) {
              new_positions.push_back(current_pos);
              new_velocity.push_back(velocityHandle.get(i_emit));
              current_particle_count++;
            } else {
              if (openvdb::tools::BoxSampler::sample(
                      solid_axr, voxel_gcoord + current_pos) > 0) {
                new_positions.push_back(current_pos);
                new_velocity.push_back(velocityHandle.get(i_emit));
                current_particle_count++;
              } // end if the particle position if outside solid
            }
          }
          // current_particle_count += original_attribute_end -
          // original_attribute_begin;
        } else {
          // the fill / kill zone

          // test if this voxel has at least half below the sea level
          // min_np particles will fill the lower part of this voxel
          // automatically generating dense particles near sea level
          // particle voxel has lattice at its center
          auto voxel_particle_wpos = m_transform->indexToWorld(voxel_gcoord);
          auto voxel_boundary_ipos =
              m_boundary_fill_kill_volume->worldToIndex(voxel_particle_wpos);
          if (openvdb::tools::BoxSampler::sample(boundary_volume_axr,
                                                 voxel_boundary_ipos) <
              voxel_include_threshold) {
            // this voxel is below the sea
            // emit the original particles in this voxel until max capacity
            // some particles will not be emitted because it is
            int emitter_end = std::min(original_attribute_end,
                                       original_attribute_begin + max_np);
            int this_voxel_emitted = 0;
            for (int i_emit = emitter_end; i_emit > original_attribute_begin;
                 i_emit--) {
              int ie = i_emit - 1;
              const openvdb::Vec3f particle_voxel_pos = positionHandle.get(ie);
              auto point_particle_wpos =
                  m_transform->indexToWorld(particle_voxel_pos + voxel_gcoord);
              auto point_boundary_ipos =
                  m_boundary_fill_kill_volume->worldToIndex(
                      point_particle_wpos);
              if (openvdb::tools::BoxSampler::sample(boundary_volume_axr,
                                                     point_boundary_ipos) < 0) {
                new_positions.push_back(particle_voxel_pos);
                new_velocity.push_back(velocityHandle.get(ie));
                current_particle_count++;
                this_voxel_emitted++;
              }
            } // end for original particles in this voxel

            // additionally check if we need to emit new particles to fill this
            // voxel emit particles up to min_np
            for (; this_voxel_emitted < min_np;) {
              // generate new position
              openvdb::Vec3f pos{randomTable[(index++) % 21474836]};
              auto new_point_particle_wpos =
                  m_transform->indexToWorld(pos + voxel_gcoord);
              auto new_point_boundary_ipos =
                  m_boundary_fill_kill_volume->worldToIndex(
                      new_point_particle_wpos);
              if (openvdb::tools::BoxSampler::sample(boundary_volume_axr,
                                                     new_point_boundary_ipos) <
                  -particle_radius) {
                // fully emit
                pos[0] = randomTable[(index++) % 21474836];
                pos[2] = randomTable[(index++) % 21474836];
                new_positions.push_back(pos);

                const openvdb::Vec3f vel = openvdb::tools::BoxSampler::sample(
                    m_velocity_volume->tree(),
                    m_velocity_volume->worldToIndex(
                        m_transform->indexToWorld(pos + voxel_gcoord)));
                new_velocity.push_back(vel);

                /*m_pos_offsets[ileaf].push_back(
                        std::make_pair(m_transform->indexToWorld(pos +
                   voxel_gcoord), current_particle_count));*/

                current_particle_count++;
              } // end if particle below voxel sea level
              this_voxel_emitted++;
            } // end for fill new particles in voxel

          } // end if this voxel center is below sealevel

          // at this point the voxel is in the air, so do not emit particles
        } // end else voxel inside inner box

        // emit finished
        // record the new offset for this voxel
        new_attribute_offsets.push_back(current_particle_count);
      } // end for all voxels

      // Below this line, the positions and offsets of points are set
      // update the offset in this leaf and attribute array

      if (new_positions.empty()) {
        leaf.clearAttributes();
      } else {
        auto local_pv_descriptor = m_pv_descriptor;
        leaf.initializeAttributes(m_p_descriptor, new_positions.size());

        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);

        leaf.setOffsets(new_attribute_offsets);

        // set the positions and velocities
        // get the new attribute arrays
        openvdb::points::AttributeArray &posarray = leaf.attributeArray("P");
        openvdb::points::AttributeArray &varray = leaf.attributeArray("v");

        // Create read handles for position and velocity
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::PositionCodec>
            posWHandle(posarray);
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::VelocityCodec>
            vWHandle(varray);

        for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
          posWHandle.set(*iter, new_positions[*iter]);
          vWHandle.set(*iter, new_velocity[*iter]);
        }
      } // end if has points to write to attribute

    } // end for range leaf
  }   // end operator

  const FLIP_vdb::descriptor_t::Ptr m_p_descriptor;
  const FLIP_vdb::descriptor_t::Ptr m_pv_descriptor;

  // outside box minmax, and inside box minmax
  openvdb::Coord m_domain_begin, m_domain_end, m_int_begin, m_int_end;

  // functions used to determine if a point is below sea_level
  openvdb::FloatGrid::Ptr m_boundary_fill_kill_volume;

  // size: number of leaf nodes intersecting the fillzone
  tbb::concurrent_vector<node_t *> &m_leaf_nodes;

  // solid sdf function used to detect voxels fully merged in solid
  openvdb::FloatGrid::Ptr m_solid;

  // velocity volume function used to set the velocity of the newly emitted
  // particles
  openvdb::Vec3fGrid::Ptr m_velocity_volume;

  // transform of the particle grid
  openvdb::math::Transform::Ptr m_transform;

  // size: same as m_leaf_nodes to modify the velocity later
  // originally designed for batch velocity interpolation
  // now replaced by velocity volume
  // std::vector<std::vector<pos_offset_t>>& m_pos_offsets;
}; // end fill kil operator

struct get_fill_kill_nodes_op {
  using node_t = openvdb::points::PointDataGrid::TreeType::LeafNodeType;

  get_fill_kill_nodes_op(openvdb::points::PointDataGrid::Ptr in_particles,
                         openvdb::FloatGrid::Ptr in_solid,
                         tbb::concurrent_vector<node_t *> &in_out_leaf_nodes,
                         const openvdb::Coord &domain_begin,
                         const openvdb::Coord &domain_end,
                         openvdb::FloatGrid::Ptr in_boundary_fill_kill_volume)
      : m_leaf_nodes(in_out_leaf_nodes) {
    m_particles = in_particles;
    m_solid = in_solid;
    fill_layer = 4, m_domain_begin = domain_begin;
    m_domain_end = domain_end;
    int_begin = m_domain_begin + openvdb::Coord{fill_layer};
    int_end = m_domain_end - openvdb::Coord{fill_layer};
    m_boundary_fill_kill_volume = in_boundary_fill_kill_volume;
    touch_fill_nodes();
    m_leaf_nodes.clear();
  }

  // this is intended to be used with leaf manager of particles
  void operator()(node_t &leaf, openvdb::Index leafpos) const {
    auto solid_axr = m_solid->getConstAccessor();

    auto worth_dealing = [&](const openvdb::Coord &origin) {
      // the node must either include active solid voxels
      // so it will be used to delete particles
      // or is beyond the inner minmax so it will be used for filling
      // and for deleting particles
      if (solid_axr.probeConstLeaf(origin))
        return true;
      if (solid_axr.getValue(origin) < 0)
        return true;

      for (int ii = 0; ii < 16; ii += 8) {
        for (int jj = 0; jj < 16; jj += 8) {
          for (int kk = 0; kk < 16; kk += 8) {
            for (int ic = 0; ic < 3; ic++) {
              if (origin[ic] + openvdb::Coord{ii, jj, kk}[ic] < int_begin[ic]) {
                return true;
              }
              if (origin[ic] + openvdb::Coord{ii, jj, kk}[ic] > int_end[ic]) {
                return true;
              }
            }
          }
        }
      }

      return false;
    }; // end leaf worth dealing

    if (worth_dealing(leaf.origin())) {
      m_leaf_nodes.push_back(&leaf);
    }
  }

  void touch_fill_nodes() {
    // the first fill node
    auto first_leaf = m_particles->treePtr()->touchLeaf(m_domain_begin);
    const auto leaf_idxmin = first_leaf->origin();

    auto solid_axr = m_solid->getConstAccessor();
    // loop over all leaf nodes intersecting the big bounding box
    for (int x = leaf_idxmin[0]; x < m_domain_end[0]; x += 8) {
      for (int y = leaf_idxmin[1]; y < m_domain_end[1]; y += 8) {
        for (int z = leaf_idxmin[2]; z < m_domain_end[2]; z += 8) {
          openvdb::Coord origin{x, y, z};
          if (intersect_fill_layer(origin)) {
            m_particles->tree().touchLeaf(origin);
          }
        }
      }
    }
    // note there might be nodes with particles but are
    // beyond the big bounding box, hence touching the fill nodes
    // and detect solid leaves within the bounding box may not find them
    // all nodes are discovered in the operator() function
  }

  bool intersect_fill_layer(const openvdb::Coord &origin) {
    bool outside_smallbox = false;

    for (int ii = 0; ii < 16 && !outside_smallbox; ii += 8) {
      for (int jj = 0; jj < 16 && !outside_smallbox; jj += 8) {
        for (int kk = 0; kk < 16 && !outside_smallbox; kk += 8) {
          // any of the eight corner is outside of the small box,
          auto test_coord = origin + openvdb::Coord{ii, jj, kk};
          for (int ic = 0; ic < 3 && !outside_smallbox; ic++) {
            if (test_coord[ic] < int_begin[ic] ||
                test_coord[ic] > int_end[ic]) {
              outside_smallbox = true;
            }
          }
        }
      }
    }

    // it is purely liquid
    if (!outside_smallbox)
      return false;

    // by default, this leaf intersects the big bounding box
    // check bottom sealevel
    float voxel_sdf_threshold = m_particles->transform().voxelSize()[0];
    for (int ii = 0; ii < 8; ii += 1) {
      for (int kk = 0; kk < 8; kk += 1) {
        auto worigin = m_particles->indexToWorld(origin.offsetBy(ii, 0, kk));
        auto iorigin = m_boundary_fill_kill_volume->worldToIndex(worigin);
        if (openvdb::tools::BoxSampler::sample(
                m_boundary_fill_kill_volume->tree(), iorigin) <
            voxel_sdf_threshold) {
          return true;
        }
      }
    }

    // all points above the waterline
    return false;
  }; // end intersect_fill_layer

  openvdb::FloatGrid::Ptr m_boundary_fill_kill_volume;

  openvdb::Coord m_domain_begin, m_domain_end, int_begin, int_end;

  int fill_layer;

  openvdb::points::PointDataGrid::Ptr m_particles;

  openvdb::FloatGrid::Ptr m_solid;

  // size: number of leaf nodes intersecting the fillzone
  tbb::concurrent_vector<node_t *> &m_leaf_nodes;
};

} // namespace

namespace {
struct tbbcoordhash {
  std::size_t hash(const openvdb::Coord &c) const { return c.hash(); }

  bool equal(const openvdb::Coord &a, const openvdb::Coord &b) const {
    return a == b;
  }
};
} // namespace

namespace {
struct narrow_band_copy_iterator {

  narrow_band_copy_iterator(const std::vector<openvdb::Index> &in_idx_of_source)
      : m_idx_of_source(in_idx_of_source) {
    m_it = 0;
  }

  narrow_band_copy_iterator(const narrow_band_copy_iterator &other)
      : m_idx_of_source(other.m_idx_of_source), m_it(other.m_it) {}
  operator bool() const { return m_it < m_idx_of_source.size(); }

  narrow_band_copy_iterator &operator++() {
    ++m_it;
    return *this;
  }

  openvdb::Index sourceIndex() const { return m_idx_of_source[m_it]; }

  Index targetIndex() const { return m_it; }

  // idx in the attribute array
  //[target_idx]=source_idx
  size_t m_it;
  const std::vector<openvdb::Index> &m_idx_of_source;
};
} // namespace

namespace {
// given input target solid sdf and source solid sdf
// touch all target solid sdf where solid sdf is active
// or has negative tiles
// leaf iterator cannot find tiles
struct solid_leaf_tile_toucher {

  solid_leaf_tile_toucher(openvdb::FloatGrid::Ptr in_target_solid_sdf,
                          openvdb::FloatGrid::Ptr in_source_solid_sdf) {
    m_target_solid_sdf = in_target_solid_sdf;
    m_source_solid_sdf = in_source_solid_sdf;
  }
  template <typename IterT> inline bool operator()(IterT &iter) {
    using namespace openvdb::tools::local_util;
    typename IterT::NonConstValueType value;
    typename IterT::ChildNodeType *child = iter.probeChild(value);
    // std::cout << "touch leaf" << iter.getCoord() << std::endl;
    if (child == nullptr) {
      // std::cout << "Tile with value " << value << std::endl;
      if (value < 0) {
        if (iter.parent().getLevel() == 1) {
          auto wpos = m_source_solid_sdf->indexToWorld(iter.getCoord());
          auto ipos = m_target_solid_sdf->worldToIndex(wpos);
          m_target_solid_sdf->tree().touchLeaf(openvdb::Coord(floorVec3(ipos)));
        }
      }
      return true; // no child to visit, so stop descending
    }
    // std::cout << "Level-" << child->getLevel() << " node" << std::endl;
    if (child->getLevel() == 0) {
      auto wpos = m_source_solid_sdf->indexToWorld(iter.getCoord());
      auto ipos = m_target_solid_sdf->worldToIndex(wpos);
      m_target_solid_sdf->tree().touchLeaf(openvdb::Coord(floorVec3(ipos)));
      return true; // don't visit leaf nodes
    }
    return false;
  }
  // The generic method, above, calls iter.probeChild(), which is not defined
  // for LeafNode::ChildAllIter.  These overloads ensure that the generic
  // method template doesn't get instantiated for LeafNode iterators.
  bool operator()(openvdb::FloatTree::LeafNodeType::ChildAllIter &) {
    return true;
  }
  bool operator()(openvdb::FloatTree::LeafNodeType::ChildAllCIter &) {
    return true;
  }

  openvdb::FloatGrid::Ptr m_target_solid_sdf;
  openvdb::FloatGrid::Ptr m_source_solid_sdf;
};
} // namespace
void FLIP_vdb::update_solid_sdf(
    std::vector<openvdb::FloatGrid::Ptr> &moving_solids,
    openvdb::FloatGrid::Ptr &solid_sdf,
    openvdb::points::PointDataGrid::Ptr &particles) {
  using namespace openvdb::tools::local_util;
  for (auto &solidsdfptr : moving_solids) {
    for (auto leafiter = solidsdfptr->tree().cbeginLeaf(); leafiter;
         ++leafiter) {
      auto source_ipos = leafiter->origin();
      auto source_wpos = solidsdfptr->indexToWorld(source_ipos);
      auto target_ipos =
          openvdb::Coord(floorVec3(solid_sdf->worldToIndex(source_wpos)));
      solid_sdf->tree().touchLeaf(target_ipos);
    }
    /*auto toucher = solid_leaf_tile_toucher(m_solid_sdf, solidsdfptr);
    solidsdfptr->tree().visit(toucher);*/
  }
  // also retrieve all possible liquids trapped inside solid
  tbb::concurrent_vector<openvdb::Coord> neib_liquid_origins;
  auto collect_liquid_origins =
      [&](openvdb::points::PointDataTree::LeafNodeType &leaf,
          openvdb::Index leafpos) {
        for (auto &solidsdfptr : moving_solids) {
          auto axr{solidsdfptr->getConstUnsafeAccessor()};
          for (int ii = 0; ii <= 8; ii += 8) {
            for (int jj = 0; jj <= 8; jj += 8) {
              for (int kk = 0; kk <= 8; kk += 8) {
                auto at_coord = leaf.origin().offsetBy(ii, jj, kk);
                auto wpos = particles->indexToWorld(at_coord);
                auto ipos = solidsdfptr->worldToIndex(wpos);
                if (openvdb::tools::BoxSampler::sample(axr, ipos) < 0) {
                  neib_liquid_origins.push_back(leaf.origin());
                  return;
                }
              } // end kk
            }   // end jj
          }     // end ii
        }       // end for all solids
      };        // end collect liquid origin
  auto partman = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(
      particles->tree());
  partman.foreach (collect_liquid_origins);

  // touch those liquid leafs as well
  for (auto coord : neib_liquid_origins) {
    solid_sdf->tree().touchLeaf(coord);
  }

  auto update_solid_op = [&](openvdb::FloatTree::LeafNodeType &leaf,
                             openvdb::Index leafpos) {
    for (const auto &solidsdfptr : moving_solids) {
      auto source_solid_axr{solidsdfptr->getConstAccessor()};

      // get the sdf from the moving solids
      for (auto offset = 0; offset < leaf.SIZE; ++offset) {
        float current_sdf = leaf.getValue(offset);
        auto current_Ipos = leaf.offsetToGlobalCoord(offset);
        auto current_wpos = solid_sdf->indexToWorld(current_Ipos);
        // interpolate the value at the
        float source_sdf = openvdb::tools::BoxSampler::sample(
            source_solid_axr, solidsdfptr->worldToIndex(current_wpos));
        leaf.setValueOn(offset, std::min(current_sdf, source_sdf));
      } // end for target solid voxel
    }   // end for all moving solids
  };    // end update solid operator

  auto solid_leafman =
      openvdb::tree::LeafManager<openvdb::FloatTree>(solid_sdf->tree());

  solid_leafman.foreach (update_solid_op);
}
void FLIP_vdb::reseed_fluid(
    openvdb::points::PointDataGrid::Ptr &in_out_particles,
    openvdb::FloatGrid::Ptr &liquid_sdf, openvdb::Vec3fGrid::Ptr &velocity) {
  using namespace openvdb::tools::local_util;
  float dx = in_out_particles->transform().voxelSize()[0];
  std::vector<openvdb::points::PointDataTree::LeafNodeType *> particle_leaves;
  in_out_particles->tree().getNodes(particle_leaves);

  auto pnamepair = FLIP_vdb::position_attribute::attributeType();
  auto m_position_attribute_descriptor =
      openvdb::points::AttributeSet::Descriptor::create(pnamepair);

  auto vnamepair = FLIP_vdb::velocity_attribute::attributeType();
  auto m_pv_attribute_descriptor =
      m_position_attribute_descriptor->duplicateAppend("v", vnamepair);

  // use existing descriptors if we already have particles
  if (!in_out_particles->empty()) {
    for (auto it = in_out_particles->tree().cbeginLeaf(); it; ++it) {
      if (it->getLastValue() != 0) {
        // a non-empty leaf with attributes
        m_pv_attribute_descriptor = it->attributeSet().descriptorPtr();
        break;
      }
    }
  }

  // CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").start();
  auto seed_leaf2 = [&](const tbb::blocked_range<size_t> &r) {
    auto liquid_sdf_axr{liquid_sdf->getConstUnsafeAccessor()};
    auto vaxr{velocity->getConstUnsafeAccessor()};
    // std::random_device device;
    // std::mt19937 generator(/*seed=*/device());
    // std::uniform_real_distribution<> distribution(-0.5, 0.5);
    std::random_device device;
    std::uniform_int_distribution<> intdistrib(0, 21474836);
    std::mt19937 gen(/*seed=*/device());
    size_t index = intdistrib(gen);
    // leaf iter
    for (auto liter = r.begin(); liter != r.end(); ++liter) {
      auto &leaf = *particle_leaves[liter];
      std::vector<openvdb::Vec3f> new_pos, new_vel;
      std::vector<openvdb::PointDataIndex32> new_idxend;

      // check if the last element is zero
      // if so, it's a new leaf
      if (leaf.getLastValue() == 0) {
        // initialize the attribute descriptor
        auto local_pv_descriptor = m_pv_attribute_descriptor;
        leaf.initializeAttributes(m_position_attribute_descriptor, 0);
        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);
      }

      // reader
      const openvdb::points::AttributeArray &positionArray =
          leaf.attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      const openvdb::points::AttributeArray &velocityArray =
          leaf.attributeArray("v");
      auto p_handle_ptr = openvdb::points::AttributeHandle<
          openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
      auto v_handle_ptr = openvdb::points::AttributeHandle<
          openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

      openvdb::Index32 emitted_particle = 0;
      for (auto offset = 0; offset < leaf.SIZE; ++offset) {
        openvdb::Index32 idxbegin = 0;
        if (offset != 0) {
          idxbegin = leaf.getValue(offset - 1);
        }
        openvdb::Index32 idxend = leaf.getValue(offset);
        int count = 0;
        // used to indicate sub_voxel(8 cells) occupancy by particles
        unsigned char sub_voxel_occupancy = 0;

        // first emit original particles
        openvdb::Index32 this_voxel_emitted = 0;
        for (auto idx = idxbegin; idx < idxend; ++idx) {
          auto p = p_handle_ptr->get(idx);
          unsigned char sv_pos =
              ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
          // only emit uniformly, otherwise skip it
          // if (sub_voxel_occupancy & (1 << sv_pos)) {
          // 	//that bit is active, there is already an particle
          // 	//skip it
          // 	continue;
          // }
          sub_voxel_occupancy |= (1 << sv_pos);
          new_pos.push_back(p);
          new_vel.push_back(v_handle_ptr->get(idx));
          emitted_particle++;
          this_voxel_emitted++;
        }

        auto voxelwpos =
            in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
        auto voxelipos = liquid_sdf->worldToIndex(voxelwpos);
        float liquid_phi = openvdb::tools::BoxSampler::sample(
            liquid_sdf->getConstAccessor(), voxelipos);
        if (liquid_phi < -dx && this_voxel_emitted < 4) {
          const int max_emit_trial = 16;
          for (int trial = 0; this_voxel_emitted < 6 && trial < max_emit_trial;
               trial++) {
            openvdb::Vec3d particle_pipos{randomTable[(index++) % 21474836],
                                          randomTable[(index++) % 21474836],
                                          randomTable[(index++) % 21474836]};
            auto &p = particle_pipos;
            unsigned char sv_pos =
                ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
            // only emit uniformly, otherwise skip it
            if (sub_voxel_occupancy & (1 << sv_pos)) {
              // that bit is active, there is already an particle
              // skip it
              continue;
            }
            auto particlewpos =
                in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
            auto velsamplepos = velocity->worldToIndex(particlewpos);
            sub_voxel_occupancy |= 1 << sv_pos;
            new_pos.push_back(particle_pipos);
            new_vel.push_back(StaggeredBoxSampler::sample(vaxr, velsamplepos));
            emitted_particle++;
            this_voxel_emitted++;
          } // end for 16 emit trials
        }
        new_idxend.push_back(emitted_particle);
      }

      auto local_pv_descriptor = m_pv_attribute_descriptor;
      leaf.initializeAttributes(m_position_attribute_descriptor,
                                emitted_particle);
      leaf.appendAttribute(leaf.attributeSet().descriptor(),
                           local_pv_descriptor, 1);
      leaf.setOffsets(new_idxend);

      // set the positions and velocities
      // get the new attribute arrays
      openvdb::points::AttributeArray &posarray = leaf.attributeArray("P");
      openvdb::points::AttributeArray &varray = leaf.attributeArray("v");

      // Create read handles for position and velocity
      openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                            FLIP_vdb::PositionCodec>
          posWHandle(posarray);
      openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                            FLIP_vdb::VelocityCodec>
          vWHandle(varray);

      for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
        posWHandle.set(*iter, new_pos[*iter]);
        vWHandle.set(*iter, new_vel[*iter]);
      } // end for all on voxels
    }
  };

  tbb::parallel_for(tbb::blocked_range<size_t>(0, particle_leaves.size()),
                    seed_leaf2);
  // printf("added particles:%zd\n", size_t(added_particles));
  // printf("original leaf count:%d\n", m_particles->tree().leafCount());
  for (auto leaf : particle_leaves) {
    if (leaf != in_out_particles->tree().probeLeaf(leaf->origin())) {
      in_out_particles->tree().addLeaf(leaf);
    }
  }
  // openvdb::io::File("dbg.vdb").write({ m_particles });
  // printf("new leafcount:%d\n", m_particles->tree().leafCount());
  in_out_particles->pruneGrid();
}

void FLIP_vdb::emit_liquid(
    openvdb::points::PointDataGrid::Ptr &in_out_particles,
    openvdb::FloatGrid::Ptr &sdf, openvdb::Vec3fGrid::Ptr &vel,
    openvdb::FloatGrid::Ptr &liquid_sdf, float vx, float vy, float vz) {
  using namespace openvdb::tools::local_util;

  // alias
  auto in_sdf = sdf;
  auto in_vel = vel;
  bool use_vel_volume = false;
  // the input is assumed to be narrowband boxes
  auto inputbbox = openvdb::CoordBBox();
  auto is_valid = in_sdf->tree().evalLeafBoundingBox(inputbbox);
  if (!is_valid) {
    return;
  }
  if (vel != nullptr) {
    use_vel_volume = true;
  }
  // bounding box in the world coordinates
  auto worldinputbbox = openvdb::BBoxd(in_sdf->indexToWorld(inputbbox.min()),
                                       in_sdf->indexToWorld(inputbbox.max()));

  // map the bounding box to the particle index space
  auto loopbegin =
      floorVec3(in_out_particles->worldToIndex(worldinputbbox.min())) -
      openvdb::Vec3i(8);
  auto loopend =
      ceilVec3(in_out_particles->worldToIndex(worldinputbbox.max())) +
      openvdb::Vec3i(8);

  auto tempgrid = openvdb::BoolGrid::create();
  tempgrid->setTransform(in_out_particles->transformPtr());
  auto templeaf = tempgrid->tree().touchLeaf(Coord(loopbegin));

  loopbegin = templeaf->origin().asVec3i() / 8;
  templeaf = tempgrid->tree().touchLeaf(Coord(loopend));
  loopend = templeaf->origin().asVec3i() / 8;

  using point_leaf_t = openvdb::points::PointDataTree::LeafNodeType;
  tbb::concurrent_hash_map<openvdb::Coord, point_leaf_t *, tbbcoordhash>
      target_point_leaves;

  auto pispace_range = tbb::blocked_range3d<int, int, int>(
      loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(),
      loopend.z());

  float dx = in_out_particles->transform().voxelSize()[0];
  printf("%f\n", dx);
  auto mark_active_point_leaves =
      [&](const tbb::blocked_range3d<int, int, int> &r) {
        auto solid_axr{in_sdf->getConstAccessor()};

        for (int i = r.pages().begin(); i < r.pages().end(); i += 1) {
          for (int j = r.rows().begin(); j < r.rows().end(); j += 1) {
            for (int k = r.cols().begin(); k < r.cols().end(); k += 1) {
              // test if any of the corner of this box touches the
              int has_solid = 0;
              int i8 = i * 8, j8 = j * 8, k8 = k * 8;
              for (int ii = 0; ii <= 8 && !has_solid; ii += 1) {
                for (int jj = 0; jj <= 8 && !has_solid; jj += 1) {
                  for (int kk = 0; kk <= 8 && !has_solid; kk += 1) {
                    auto particle_idx_pos =
                        openvdb::Vec3i(i8 + ii, j8 + jj, k8 + kk);
                    has_solid =
                        (openvdb::tools::BoxSampler::sample(
                             solid_axr, in_sdf->worldToIndex(
                                            in_out_particles->indexToWorld(
                                                particle_idx_pos))) < 0);
                  }
                }
              }

              if (has_solid) {
                auto floor_leaf = in_out_particles->tree().probeLeaf(
                    openvdb::Coord(i8, j8, k8));
                if (floor_leaf) {
                  target_point_leaves.insert(
                      std::make_pair(floor_leaf->origin(), floor_leaf));
                } else {
                  auto newleaff =
                      new openvdb::points::PointDataTree::LeafNodeType(
                          openvdb::Coord(i8, j8, k8), 0, true);
                  auto new_origin = newleaff->origin();
                  bool success = target_point_leaves.insert(
                      std::make_pair(newleaff->origin(), newleaff));
                  if (!success) {
                    delete newleaff;
                  }
                } // end else found existing particle leaf
              }   // end if touch solid
            }     // loop k
          }       // loop j
        }         // loop i
      };          // end mark active point elaves
  tbb::parallel_for(pispace_range, mark_active_point_leaves);

  // for parallel process
  std::vector<point_leaf_t *> leafarray;
  for (auto &i : target_point_leaves) {
    leafarray.push_back(i.second);
  }
  // printf("touched_leafs:%zd\n", leafarray.size());

  std::atomic<size_t> added_particles{0};

  auto pnamepair = FLIP_vdb::position_attribute::attributeType();
  auto m_position_attribute_descriptor =
      openvdb::points::AttributeSet::Descriptor::create(pnamepair);

  auto vnamepair = FLIP_vdb::velocity_attribute::attributeType();
  auto m_pv_attribute_descriptor =
      m_position_attribute_descriptor->duplicateAppend("v", vnamepair);

  // use existing descriptors if we already have particles
  if (!in_out_particles->empty()) {
    for (auto it = in_out_particles->tree().cbeginLeaf(); it; ++it) {
      if (it->getLastValue() != 0) {
        // a non-empty leaf with attributes
        m_pv_attribute_descriptor = it->attributeSet().descriptorPtr();
        break;
      }
    }
  }

  // seed each leaf
  auto seed_leaf = [&](const tbb::blocked_range<size_t> &r) {
    if (use_vel_volume) {
      auto in_sdf_axr{in_sdf->getConstAccessor()};
      auto in_vel_axr{in_vel->getConstAccessor()};

      float sdf_threshold = -dx * 0.1;

      // std::random_device device;
      // std::mt19937 generator(/*seed=*/device());
      // std::uniform_real_distribution<> distribution(-0.5, 0.5);
      std::random_device device;
      std::uniform_int_distribution<> intdistrib(0, 21474836);
      std::mt19937 gen(/*seed=*/device());
      size_t index = intdistrib(gen);

      for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
        auto &leaf = *leafarray[ileaf];
        std::vector<openvdb::Vec3f> new_pos, new_vel;
        std::vector<openvdb::PointDataIndex32> new_idxend;

        // check if the last element is zero
        // if so, it's a new leaf
        if (leaf.getLastValue() == 0) {
          // initialize the attribute descriptor
          auto local_pv_descriptor = m_pv_attribute_descriptor;
          leaf.initializeAttributes(m_position_attribute_descriptor, 0);
          leaf.appendAttribute(leaf.attributeSet().descriptor(),
                               local_pv_descriptor, 1);
        }

        // reader
        const openvdb::points::AttributeArray &positionArray =
            leaf.attributeArray("P");
        // Extract the velocity attribute from the leaf by name (v is velocity).
        const openvdb::points::AttributeArray &velocityArray =
            leaf.attributeArray("v");
        auto p_handle_ptr = openvdb::points::AttributeHandle<
            openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
        auto v_handle_ptr = openvdb::points::AttributeHandle<
            openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

        openvdb::Index32 emitted_particle = 0;
        for (auto offset = 0; offset < leaf.SIZE; ++offset) {
          openvdb::Index32 idxbegin = 0;
          if (offset != 0) {
            idxbegin = leaf.getValue(offset - 1);
          }
          openvdb::Index32 idxend = leaf.getValue(offset);

          // used to indicate sub_voxel(8 cells) occupancy by particles
          unsigned char sub_voxel_occupancy = 0;

          // first emit original particles
          openvdb::Index32 this_voxel_emitted = 0;
          for (auto idx = idxbegin; idx < idxend; ++idx) {
            auto p = p_handle_ptr->get(idx);
            unsigned char sv_pos =
                ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
            // only emit uniformly, otherwise skip it
            // if (sub_voxel_occupancy & (1 << sv_pos)) {
            // 	//that bit is active, there is already an particle
            // 	//skip it
            // 	continue;
            // }
            sub_voxel_occupancy |= (1 << sv_pos);
            new_pos.push_back(p);
            new_vel.push_back(v_handle_ptr->get(idx));
            emitted_particle++;
            this_voxel_emitted++;
          }

          // emit to fill the sdf
          auto voxelwpos =
              in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
          auto voxelipos = in_sdf->worldToIndex(voxelwpos);
          float liquid_phi = 10000;
          if (liquid_sdf != nullptr) {
            liquid_phi = openvdb::tools::BoxSampler::sample(
                liquid_sdf->getConstAccessor(), voxelipos);
          }
          if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < dx &&
              liquid_phi >= 0) {
            const int max_emit_trial = 16;
            for (int trial = 0;
                 this_voxel_emitted < 8 && trial < max_emit_trial; trial++) {
              openvdb::Vec3d particle_pipos{randomTable[(index++) % 21474836],
                                            randomTable[(index++) % 21474836],
                                            randomTable[(index++) % 21474836]};
              auto &p = particle_pipos;
              unsigned char sv_pos =
                  ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
              // only emit uniformly, otherwise skip it
              if (sub_voxel_occupancy & (1 << sv_pos)) {
                // that bit is active, there is already an particle
                // skip it
                continue;
              }
              auto particlewpos =
                  in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
              auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
              if (openvdb::tools::BoxSampler::sample(
                      in_sdf_axr, particle_sdfipos) < sdf_threshold) {
                sub_voxel_occupancy |= 1 << sv_pos;
                new_pos.push_back(particle_pipos);
                new_vel.push_back(openvdb::tools::BoxSampler::sample(
                    in_vel_axr, particle_sdfipos));
                emitted_particle++;
                this_voxel_emitted++;
                added_particles++;
              }
            } // end for 16 emit trials
          }   // if voxel inside emitter

          new_idxend.push_back(emitted_particle);
        } // end emit original particles and new particles

        // set the new index ends, attributes
        // initialize the attribute descriptor
        auto local_pv_descriptor = m_pv_attribute_descriptor;
        leaf.initializeAttributes(m_position_attribute_descriptor,
                                  emitted_particle);
        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);
        leaf.setOffsets(new_idxend);

        // set the positions and velocities
        // get the new attribute arrays
        openvdb::points::AttributeArray &posarray = leaf.attributeArray("P");
        openvdb::points::AttributeArray &varray = leaf.attributeArray("v");

        // Create read handles for position and velocity
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::PositionCodec>
            posWHandle(posarray);
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::VelocityCodec>
            vWHandle(varray);

        for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
          posWHandle.set(*iter, new_pos[*iter]);
          vWHandle.set(*iter, new_vel[*iter]);
        } // end for all on voxels
      }   // end for range leaf
    } else {
      auto in_sdf_axr{in_sdf->getConstAccessor()};
      float sdf_threshold = -dx * 0.1;

      // std::random_device device;
      // std::mt19937 generator(/*seed=*/device());
      // std::uniform_real_distribution<> distribution(-0.5, 0.5);
      std::random_device device;
      std::uniform_int_distribution<> intdistrib(0, 21474836);
      std::mt19937 gen(/*seed=*/device());
      size_t index = intdistrib(gen);

      for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
        auto &leaf = *leafarray[ileaf];
        std::vector<openvdb::Vec3f> new_pos, new_vel;
        std::vector<openvdb::PointDataIndex32> new_idxend;

        // check if the last element is zero
        // if so, it's a new leaf
        if (leaf.getLastValue() == 0) {
          // initialize the attribute descriptor
          auto local_pv_descriptor = m_pv_attribute_descriptor;
          leaf.initializeAttributes(m_position_attribute_descriptor, 0);
          leaf.appendAttribute(leaf.attributeSet().descriptor(),
                               local_pv_descriptor, 1);
        }

        // reader
        const openvdb::points::AttributeArray &positionArray =
            leaf.attributeArray("P");
        // Extract the velocity attribute from the leaf by name (v is velocity).
        const openvdb::points::AttributeArray &velocityArray =
            leaf.attributeArray("v");
        auto p_handle_ptr = openvdb::points::AttributeHandle<
            openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
        auto v_handle_ptr = openvdb::points::AttributeHandle<
            openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

        openvdb::Index32 emitted_particle = 0;
        for (auto offset = 0; offset < leaf.SIZE; ++offset) {
          openvdb::Index32 idxbegin = 0;
          if (offset != 0) {
            idxbegin = leaf.getValue(offset - 1);
          }
          openvdb::Index32 idxend = leaf.getValue(offset);

          // used to indicate sub_voxel(8 cells) occupancy by particles
          unsigned char sub_voxel_occupancy = 0;

          // first emit original particles
          openvdb::Index32 this_voxel_emitted = 0;
          for (auto idx = idxbegin; idx < idxend; ++idx) {
            auto p = p_handle_ptr->get(idx);
            unsigned char sv_pos =
                ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
            // only emit uniformly, otherwise skip it
            // if (sub_voxel_occupancy & (1 << sv_pos)) {
            // 	//that bit is active, there is already an particle
            // 	//skip it
            // 	continue;
            // }
            sub_voxel_occupancy |= (1 << sv_pos);
            new_pos.push_back(p);
            new_vel.push_back(v_handle_ptr->get(idx));
            emitted_particle++;
            this_voxel_emitted++;
          }

          // emit to fill the sdf
          auto voxelwpos =
              in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
          auto voxelipos = in_sdf->worldToIndex(voxelwpos);
          float liquid_phi = 10000;
          if (liquid_sdf != nullptr) {
            liquid_phi = openvdb::tools::BoxSampler::sample(
                liquid_sdf->getConstAccessor(), voxelipos);
          }
          if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < dx &&
              liquid_phi >= 0) {
            const int max_emit_trial = 16;
            for (int trial = 0;
                 this_voxel_emitted < 8 && trial < max_emit_trial; trial++) {
              openvdb::Vec3d particle_pipos{randomTable[(index++) % 21474836],
                                            randomTable[(index++) % 21474836],
                                            randomTable[(index++) % 21474836]};
              auto &p = particle_pipos;
              unsigned char sv_pos =
                  ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
              // only emit uniformly, otherwise skip it
              if (sub_voxel_occupancy & (1 << sv_pos)) {
                // that bit is active, there is already an particle
                // skip it
                continue;
              }
              auto particlewpos =
                  in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
              auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
              if (openvdb::tools::BoxSampler::sample(
                      in_sdf_axr, particle_sdfipos) < sdf_threshold) {
                sub_voxel_occupancy |= 1 << sv_pos;
                new_pos.push_back(particle_pipos);
                new_vel.push_back(openvdb::Vec3d(vx, vy, vz));
                emitted_particle++;
                this_voxel_emitted++;
                added_particles++;
              }
            } // end for 16 emit trials
          }   // if voxel inside emitter

          new_idxend.push_back(emitted_particle);
        } // end emit original particles and new particles

        // set the new index ends, attributes
        // initialize the attribute descriptor
        auto local_pv_descriptor = m_pv_attribute_descriptor;
        leaf.initializeAttributes(m_position_attribute_descriptor,
                                  emitted_particle);
        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);
        leaf.setOffsets(new_idxend);

        // set the positions and velocities
        // get the new attribute arrays
        openvdb::points::AttributeArray &posarray = leaf.attributeArray("P");
        openvdb::points::AttributeArray &varray = leaf.attributeArray("v");

        // Create read handles for position and velocity
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::PositionCodec>
            posWHandle(posarray);
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::VelocityCodec>
            vWHandle(varray);

        for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
          posWHandle.set(*iter, new_pos[*iter]);
          vWHandle.set(*iter, new_vel[*iter]);
        } // end for all on voxels
      }   // end for range leaf
    }
  }; // end seed particle

  tbb::parallel_for(tbb::blocked_range<size_t>(0, leafarray.size()), seed_leaf);
  // printf("added particles:%zd\n", size_t(added_particles));
  // printf("original leaf count:%d\n", m_particles->tree().leafCount());
  for (auto leaf : leafarray) {
    if (leaf != in_out_particles->tree().probeLeaf(leaf->origin())) {
      in_out_particles->tree().addLeaf(leaf);
    }
  }
  // openvdb::io::File("dbg.vdb").write({ m_particles });
  // printf("new leafcount:%d\n", m_particles->tree().leafCount());
  in_out_particles->pruneGrid();
}

void FLIP_vdb::calculate_face_weights(openvdb::Vec3fGrid::Ptr &face_weight,
                                      openvdb::FloatGrid::Ptr &liquid_sdf,
                                      openvdb::FloatGrid::Ptr &solid_sdf) {
  face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{1.0f});
  face_weight->setTree(std::make_shared<openvdb::Vec3fTree>(
      liquid_sdf->tree(), openvdb::Vec3f{1.0f}, openvdb::TopologyCopy()));
  openvdb::tools::dilateActiveValues(
      face_weight->tree(), 1,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
  face_weight->setName("Face_Weights");
  face_weight->setTransform(liquid_sdf->transformPtr());
  face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);
  auto set_face_weight_op = [&](openvdb::Vec3fTree::LeafNodeType &leaf,
                                openvdb::Index leafpos) {
    auto solid_axr{solid_sdf->getConstAccessor()};
    // solid sdf
    float ssdf[2][2][2];
    openvdb::Vec3f uvwweight{1.f};
    for (auto offset = 0; offset < leaf.SIZE; offset++) {
      if (leaf.isValueMaskOff(offset)) {
        continue;
      }
      for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
          for (int kk = 0; kk < 2; kk++) {
            ssdf[ii][jj][kk] = solid_axr.getValue(
                leaf.offsetToGlobalCoord(offset) + openvdb::Coord{ii, jj, kk});
          }
        }
      } // end retrieve eight solid sdfs on the voxel corners

      // fraction_inside(bl,br,tl,tr)
      // tl-----tr
      //|      |
      //|      |
      // bl-----br

      // look from positive x direction

      //   ^z
      //(0,0,1)------(0,1,1)
      //   |            |
      //(0,0,0)------(0,1,0)>y
      // uweight
      uvwweight[0] = 1.0f - fraction_inside(ssdf[0][0][0], ssdf[0][1][0],
                                            ssdf[0][0][1], ssdf[0][1][1]);
      uvwweight[0] = std::max(0.f, std::min(uvwweight[0], 1.f));

      // look from positive y direction
      //   ^x
      //(1,0,0)------(1,0,1)
      //   |            |
      //(0,0,0)------(0,0,1)>z
      // vweight
      uvwweight[1] = 1.0f - fraction_inside(ssdf[0][0][0], ssdf[0][0][1],
                                            ssdf[1][0][0], ssdf[1][0][1]);
      uvwweight[1] = std::max(0.f, std::min(uvwweight[1], 1.f));

      // look from positive z direction
      //   ^y
      //(0,1,0)------(1,1,0)
      //   |            |
      //(0,0,0)------(1,0,0)>x
      // wweight
      uvwweight[2] = 1.0f - fraction_inside(ssdf[0][0][0], ssdf[1][0][0],
                                            ssdf[0][1][0], ssdf[1][1][0]);
      uvwweight[2] = std::max(0.f, std::min(uvwweight[2], 1.f));
      leaf.setValueOn(offset, uvwweight);
    } // end for all offset
  };  // end set face weight op

  auto leafman =
      openvdb::tree::LeafManager<openvdb::Vec3fTree>(face_weight->tree());
  leafman.foreach (set_face_weight_op);
}

void FLIP_vdb::clamp_liquid_phi_in_solids(
    openvdb::FloatGrid::Ptr &liquid_sdf, openvdb::FloatGrid::Ptr &solid_sdf,
    openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf, float dx) {
  openvdb::tools::dilateActiveValues(
      liquid_sdf->tree(), 1,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
  auto correct_liquid_phi_in_solid = [&](openvdb::FloatTree::LeafNodeType &leaf,
                                         openvdb::Index leafpos) {
    // detech if there is solid
    if (solid_sdf->tree().probeConstLeaf(leaf.origin())) {
      auto const_solid_axr{solid_sdf->getConstAccessor()};
      auto shift{openvdb::Vec3R{0.5}};

      for (auto offset = 0; offset < leaf.SIZE; offset++) {
        if (leaf.isValueMaskOff(offset)) {
          continue;
        }
        auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
            solid_sdf->tree(),
            leaf.offsetToGlobalCoord(offset).asVec3d() + shift);

        if (voxel_solid_sdf < 0) {
          leaf.setValueOn(offset, dx);
        }
      } // end for all voxel
    }   // end there is solid in this leaf
  };    // end correct_liquid_phi_in_solid

  auto phi_manager =
      openvdb::tree::LeafManager<openvdb::FloatTree>(liquid_sdf->tree());

  phi_manager.foreach (correct_liquid_phi_in_solid);

  pushed_out_liquid_sdf = liquid_sdf->deepCopy();

  // this is to be called by a solid manager
  auto immerse_liquid_into_solid = [&](openvdb::FloatTree::LeafNodeType &leaf,
                                       openvdb::Index leafpos) {
    // detech if there is solid
    if (solid_sdf->tree().probeConstLeaf(leaf.origin())) {
      auto pushed_out_liquid_axr{pushed_out_liquid_sdf->getConstAccessor()};
      auto const_solid_axr{solid_sdf->getConstAccessor()};
      auto shift{openvdb::Vec3R{0.5}};

      for (auto iter = leaf.beginValueOn(); iter; ++iter) {
        auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
            solid_sdf->tree(),
            leaf.offsetToGlobalCoord(iter.offset()).asVec3d() + shift);

        if (voxel_solid_sdf < 0) {
          bool found_liquid_neib = false;
          for (int i_neib = 0; i_neib < 6 && !found_liquid_neib; i_neib++) {
            int component = i_neib / 2;
            int positive_dir = (i_neib % 2 == 0);

            auto test_coord = iter.getCoord();

            if (positive_dir) {
              test_coord[component]++;
            } else {
              test_coord[component]--;
            }
            found_liquid_neib |=
                (pushed_out_liquid_axr.getValue(test_coord) < 0);
          } // end for 6 direction

          // mark this voxel as liquid
          if (found_liquid_neib) {
            float current_sdf = iter.getValue();
            iter.setValue(-0.5f * dx);
            // iter.setValue(-dx * 0.01);
            iter.setValueOn();
          }
        } // end if this voxel is inside solid

      } // end for all voxel
    }   // end there is solid in this leaf
  };    // end immerse_liquid_into_solid

  phi_manager.foreach (immerse_liquid_into_solid);
}

namespace {
// return the total number of degree of freedom
// and set the degree of freedom tree
openvdb::Int32
fill_idx_tree_from_liquid_sdf(openvdb::Int32Tree::Ptr dof_tree,
                              openvdb::FloatGrid::Ptr liquid_sdf_tree) {

  auto dof_leafman = openvdb::tree::LeafManager<openvdb::Int32Tree>(*dof_tree);

  // first count how many dof in each leaf
  // then assign the global dof id
  std::vector<openvdb::Int32> dof_end_in_each_leaf;
  dof_end_in_each_leaf.assign(liquid_sdf_tree->tree().leafCount(), 0);

  // mark the active state of dofs
  auto mark_dof_active_op = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                                openvdb::Index leafpos) {
    auto *sdf_leaf = liquid_sdf_tree->tree().probeConstLeaf(leaf.origin());
    leaf.setValueMask(sdf_leaf->getValueMask());
    for (auto iter = leaf.cbeginValueOn(); iter != leaf.cendValueOn(); ++iter) {
      // mark the position where liquid sdf non-negative as inactive
      if (sdf_leaf->getValue(iter.offset()) >= 0) {
        leaf.setValueOff(iter.offset());
      }
    }
  }; // end mark_dof_active_op
  dof_leafman.foreach (mark_dof_active_op);

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
  return *dof_end_in_each_leaf.crbegin();
} // end fill idx tree from liquid sdf

} // namespace
void FLIP_vdb::apply_pressure_gradient(
    openvdb::FloatGrid::Ptr &liquid_sdf, openvdb::FloatGrid::Ptr &solid_sdf,
    openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf,
    openvdb::FloatGrid::Ptr &pressure, openvdb::Vec3fGrid::Ptr &face_weight,
    openvdb::Vec3fGrid::Ptr &velocity, openvdb::Vec3fGrid::Ptr &solid_velocity,
    float dx, float in_dt) {

  // given the solved pressure, update the velocity
  // this is to be used by the velocity(post_pressure) leaf manager
  auto velocity_update_op = [&](openvdb::Vec3fTree::LeafNodeType &leaf,
                                openvdb::Index leafpos) {
    auto phi_axr{liquid_sdf->getConstUnsafeAccessor()};
    auto true_phi_axr{pushed_out_liquid_sdf->getConstUnsafeAccessor()};
    auto weight_axr{face_weight->getConstUnsafeAccessor()};
    auto solid_vel_axr{solid_velocity->getConstUnsafeAccessor()};
    auto solid_sdf_axr{solid_sdf->getConstUnsafeAccessor()};
    auto pressure_axr{pressure->getConstUnsafeAccessor()};
    // auto update_axr{ m_velocity_update->getAccessor() };

    for (auto offset = 0; offset < leaf.SIZE; offset++) {
      if (leaf.isValueMaskOff(offset)) {
        continue;
      }
      // we are looking at a velocity sample previously obtained from p2g
      // possibly with extrapolation

      // set this velocity as off unless we have an update
      // on any component of its velocity
      bool has_any_update = false;
      auto gcoord = leaf.offsetToGlobalCoord(offset);
      auto original_vel = leaf.getValue(offset);
      auto solid_vel = solid_vel_axr.getValue(gcoord);

      // face_weight=0 in solid
      auto face_weights = weight_axr.getValue(gcoord);
      auto vel_update = 0 * original_vel;
      // three velocity channel
      for (int ic = 0; ic < 3; ic++) {
        auto lower_gcoord = gcoord;
        lower_gcoord[ic] -= 1;
        if (face_weights[ic] > 0) {
          // this face has liquid component
          // does it have any dof on its side?
          auto phi_this = phi_axr.getValue(gcoord);
          auto phi_below = phi_axr.getValue(lower_gcoord);
          float p_this = pressure_axr.getValue(gcoord);
          float p_below = pressure_axr.getValue(lower_gcoord);

          bool update_this_velocity = false;
          if (phi_this < 0 && phi_below < 0) {
            update_this_velocity = true;
          } else {
            if (phi_this > 0 && phi_below > 0) {
              update_this_velocity = false;
            } // end if all outside liquid
            else {
              // one of them is inside the liquid, one is outside
              if (phi_this >= 0) {
                // this point is outside the liquid, possibly free air or free
                // air in the liquid if so, set the pressure to be ghost value
                // if (phi_this+phi_below<0) {
                if (true_phi_axr.getValue(lower_gcoord) < 0) {
                  update_this_velocity = true;
                }
                //}
              }
              if (phi_below >= 0) {
                // this point below is outside the liquid, possibly free air
                // if so, set the pressure to be ghost value
                // if (phi_this+phi_below<0) {
                if (true_phi_axr.getValue(gcoord) < 0) {
                  update_this_velocity = true;
                }
                //}
              }
            } // end else all outside liquid
          }   // end all inside liquid

          if (update_this_velocity) {
            float theta = 1;
            if (phi_this >= 0 || phi_below >= 0) {
              theta = fraction_inside(phi_below, phi_this);
              if (theta < 0.02f)
                theta = 0.02f;
            }

            original_vel[ic] -= in_dt * (float)(p_this - p_below) / dx / theta;
            vel_update[ic] = in_dt * (float)(p_this - p_below) / dx / theta;
            if (face_weights[ic] < 1) {
              // mix the solid velocity and fluid velocity if friction is
              // expected
              float solid_fraction = (1 - face_weights[ic]);
              original_vel[ic] = (1 - solid_fraction) * original_vel[ic] +
                                 (solid_fraction)*solid_vel[ic];
            }
            has_any_update = true;
          } // end if any dofs on two sides
        }   // end if face_weight[ic]>0
        else {
          // this face is inside solid
          // just let it be the solid velocity
          // original_vel[ic] = solid_vel[ic];
          // has_any_update = true;
        } // end else face_weight[ic]>0
      }   // end for three component

      if (!has_any_update) {
        leaf.setValueOff(offset, openvdb::Vec3R(0, 0, 0));
        // update_axr.setValueOff(leaf.offsetToGlobalCoord(offset));
      } else {
        leaf.setValueOn(offset, original_vel);
        // update_axr.setValue(leaf.offsetToGlobalCoord(offset), vel_update);
      }
    } // end for all voxels
  };  // end velocity_update_op

  auto vel_leafman =
      openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity->tree());
  vel_leafman.foreach (velocity_update_op);
}

void FLIP_vdb::solve_pressure_simd(
    openvdb::FloatGrid::Ptr &liquid_sdf,
    openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf,
    openvdb::FloatGrid::Ptr &rhsgrid, openvdb::FloatGrid::Ptr &curr_pressure,
    openvdb::Vec3fGrid::Ptr &face_weight, openvdb::Vec3fGrid::Ptr &velocity,
    openvdb::Vec3fGrid::Ptr &solid_velocity, float dt, float dx) {
  // CSim::TimerMan::timer("Sim.step/vdbflip/pressure/buildlevel").start();

    //skip if there is no dof to solve
	if (liquid_sdf->tree().leafCount() == 0) {
		return;
	}

  auto simd_solver =
      simd_vdb_poisson(liquid_sdf, face_weight, velocity,
                       solid_velocity, dt, dx);

  simd_solver.construct_levels();
  simd_solver.build_rhs();
  // CSim::TimerMan::timer("Sim.step/vdbflip/pressure/buildlevel").stop();

  auto pressure = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
  pressure->setName("Pressure");

  // use the previous pressure as warm start
  auto set_warm_pressure = [&](openvdb::Int32Tree::LeafNodeType &leaf,
                               openvdb::Index leafpos) {
    auto old_pressure_axr{curr_pressure->getConstAccessor()};
    auto *new_pressure_leaf = pressure->tree().probeLeaf(leaf.origin());

    for (auto iter = new_pressure_leaf->beginValueOn(); iter; ++iter) {
      float old_pressure = old_pressure_axr.getValue(iter.getCoord());
      if (std::isfinite(old_pressure)) {
        iter.setValue(old_pressure);
      }
    }
  }; // end set_warm_pressure

  // simd_solver.m_laplacian_with_levels[0]->m_dof_leafmanager->foreach(set_warm_pressure);

  // CSim::TimerMan::timer("Sim.step/vdbflip/pressure/simdpcg").start();
  bool converged = simd_solver.pcg_solve(pressure, 1e-7);
  // CSim::TimerMan::timer("Sim.step/vdbflip/pressure/simdpcg").stop();
  if (converged)
    curr_pressure.swap(pressure);
  else {
    pressure = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
    simd_solver.m_laplacian_with_levels[0]->m_dof_leafmanager->foreach (
        set_warm_pressure);
    simd_solver.smooth_solve(pressure, 200);
    curr_pressure.swap(pressure);
  }
  // CSim::TimerMan::timer("Sim.step/vdbflip/pressure/updatevel").start();
  // apply_pressure_gradient(dt);
  // CSim::TimerMan::timer("Sim.step/vdbflip/pressure/updatevel").stop();

  // simd_solver.build_rhs();
  rhsgrid = simd_solver.m_rhs;
  // m_rhsgrid = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
  // simd_solver.m_laplacian_with_levels[0]->set_grid_constant_assume_topo(m_rhsgrid,
  // 1);
  // simd_solver.m_laplacian_with_levels[0]->Laplacian_apply_assume_topo(m_rhsgrid,
  // m_rhsgrid->deepCopy()); m_rhsgrid =
  // simd_solver.m_laplacian_with_levels[0]->m_Neg_z_entry;
  rhsgrid->setName("RHS");
}

void FLIP_vdb::field_add_vector(openvdb::Vec3fGrid::Ptr &velocity_field,
                                openvdb::Vec3fGrid::Ptr &face_weight, float x,
                                float y, float z, float dt) {
  if (face_weight != nullptr) {
    auto add_gravity = [&](openvdb::Vec3fTree::LeafNodeType &leaf,
                           openvdb::Index leafpos) {
      auto face_weight_axr{face_weight->getConstAccessor()};
      for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
        // if (face_weight_axr.getValue(iter.getCoord())[0] > 0)
        {
          iter.modifyValue([&](openvdb::Vec3f &v) { v[0] += dt * x; });
        }
        // if (face_weight_axr.getValue(iter.getCoord())[1] > 0)
        {
          iter.modifyValue([&](openvdb::Vec3f &v) { v[1] += dt * y; });
        }
        // if (face_weight_axr.getValue(iter.getCoord())[2] > 0)
        {
          iter.modifyValue([&](openvdb::Vec3f &v) { v[2] += dt * z; });
        }
      }
    };
    auto velman =
        openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity_field->tree());
    velman.foreach (add_gravity);
  } else {
    auto add_gravity = [&](openvdb::Vec3fTree::LeafNodeType &leaf,
                           openvdb::Index leafpos) {
      for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
        iter.modifyValue([&](openvdb::Vec3f &v) { v[0] += dt * x; });

        iter.modifyValue([&](openvdb::Vec3f &v) { v[1] += dt * y; });
        iter.modifyValue([&](openvdb::Vec3f &v) { v[2] += dt * z; });
      }
    };
    auto velman =
        openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity_field->tree());
    velman.foreach (add_gravity);
  }
}

float FLIP_vdb::cfl(openvdb::Vec3fGrid::Ptr &vel) {
  std::vector<float> max_v_per_leaf;
  max_v_per_leaf.assign(vel->tree().leafCount(), 0);

  auto find_per_leaf_max_vel = [&](openvdb::Vec3fTree::LeafNodeType &leaf,
                                   openvdb::Index leafpos) {
    float max_v = 0;
    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
      if (leaf.isValueOn(offset)) {
        for (int ic = 0; ic < 3; ic++) {
          if (max_v < std::abs(leaf.getValue(offset)[ic])) {
            max_v = std::abs(leaf.getValue(offset)[ic]);
          }
        }
      }
    }
    max_v_per_leaf[leafpos] = max_v;
  }; // end find per leaf max vel

  auto leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(vel->tree());
  leafman.foreach (find_per_leaf_max_vel);

  float max_v = 0;
  if (max_v_per_leaf.empty()) {
    return std::numeric_limits<float>::max() / 2;
  }
  max_v = max_v_per_leaf[0];

  // dont take the maximum number
  // take the 90%?
  int nleaf = max_v_per_leaf.size();
  int top90 = nleaf * 99 / 100;
  std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + nleaf - 1,
                   max_v_per_leaf.end());
  std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + top90,
                   max_v_per_leaf.end());

  /*for (const auto& v : max_v_per_leaf) {
          if (v > max_v) {
                  max_v = v;
          }
  }*/
  max_v = max_v_per_leaf[nleaf - 1];
  printf("max velocity component:%f\n", max_v_per_leaf.back());
  printf("cfl velocity:%f\n", max_v);
  float dx = vel->voxelSize()[0];
  return dx / (std::abs(max_v) + 1e-6f);
}

void FLIP_vdb::Advect(float dt, float dx,
                      openvdb::points::PointDataGrid::Ptr &particles,
                      openvdb::Vec3fGrid::Ptr &velocity,
                      openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                      openvdb::FloatGrid::Ptr &solid_sdf,
                      openvdb::Vec3fGrid::Ptr &solid_vel, float pic_component,
                      int RK_ORDER) {

  custom_move_points_and_set_flip_vel(
      particles, nullptr, velocity, velocity, velocity_after_p2g, solid_sdf,
      solid_vel, pic_component, dt, 0, /*RK order*/ RK_ORDER);
}
void FLIP_vdb::AdvectSheetty(float dt, float dx, float surfacedist,
                             openvdb::points::PointDataGrid::Ptr &particles,
                             openvdb::FloatGrid::Ptr &liquid_sdf,
                             openvdb::Vec3fGrid::Ptr &velocity,
                             openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                             openvdb::FloatGrid::Ptr &solid_sdf,
                             openvdb::Vec3fGrid::Ptr &solid_vel,
                             float pic_component, int RK_ORDER) {

  custom_move_points_and_set_flip_vel(
      particles, liquid_sdf, velocity, velocity, velocity_after_p2g, solid_sdf,
      solid_vel, pic_component, dt, surfacedist, /*RK order*/ RK_ORDER);
}
void FLIP_vdb::custom_move_points_and_set_flip_vel(
    openvdb::points::PointDataGrid::Ptr in_out_points,
    const openvdb::FloatGrid::Ptr in_liquid_sdf,
    const openvdb::Vec3fGrid::Ptr in_velocity_field,
    const openvdb::Vec3fGrid::Ptr in_velocity_field_to_be_advected,
    const openvdb::Vec3fGrid::Ptr in_old_velocity,
    openvdb::FloatGrid::Ptr in_solid_sdf, openvdb::Vec3fGrid::Ptr in_solid_vel,
    float PIC_component, float dt, float surfacedist, int RK_order) {
  if (!in_out_points) {
    return;
  }

  float dx = in_out_points->transform().voxelSize()[0];
  bool has_valid_solid = true;
  if (in_solid_sdf == nullptr) {
    auto corner_transform = openvdb::math::Transform::createLinearTransform(dx);
    in_solid_vel = openvdb::Vec3fGrid::create(openvdb::Vec3f(0));
    in_solid_vel->setTransform(corner_transform);

    corner_transform->postTranslate(openvdb::Vec3d{-0.5, -0.5, -0.5} *
                                    double(dx));
    in_solid_sdf = openvdb::FloatGrid::create(3 * dx);
    in_solid_sdf->setTransform(corner_transform);
    has_valid_solid = false;
  } else if (in_solid_vel == nullptr) {
    auto corner_transform = openvdb::math::Transform::createLinearTransform(dx);
    in_solid_vel = openvdb::Vec3fGrid::create(openvdb::Vec3f(0));
    in_solid_vel->setTransform(corner_transform);
  }
  // prepare voxel center solid outward normal direction
  // in case a particle falls into solid during advection, this will help
  // pushing the particle out.
  openvdb::Vec3fGrid::Ptr voxel_center_solid_normal =
      openvdb::Vec3fGrid::create(openvdb::Vec3f(0));
  openvdb::FloatGrid::Ptr voxel_center_solid_vn = openvdb::FloatGrid::create(0);

  if (has_valid_solid) {
    voxel_center_solid_normal->setTree(std::make_shared<openvdb::Vec3fTree>(
        in_solid_sdf->tree(), /*bgval*/ openvdb::Vec3f(0),
        openvdb::TopologyCopy()));
  }

  openvdb::tools::dilateActiveValues(
      voxel_center_solid_normal->tree(), 5,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
  voxel_center_solid_normal->setTransform(in_out_points->transformPtr());
  voxel_center_solid_normal->setName("solidnormal");

  voxel_center_solid_vn->setTree(std::make_shared<openvdb::FloatTree>(
      voxel_center_solid_normal->tree(), /*bgval*/ 0.f,
      openvdb::TopologyCopy()));
  voxel_center_solid_vn->setTransform(in_out_points->transformPtr());
  voxel_center_solid_vn->setName("vn");

  Eigen::MatrixXf AT;
  Eigen::Vector4f invATA;
  AT.resize(4, 8);
  for (int i = 0; i < 8; i++) {
    int a = i / 4;
    int b = (i - a * 4) / 2;
    int c = i - a * 4 - b * 2;
    AT.col(i) = Eigen::Vector4f{a - 0.5f, b - 0.5f, c - 0.5f, 1.0f};
  }

  AT *= dx;
  invATA.setZero(4);
  invATA(0) = 0.5f;
  invATA(1) = 0.5f;
  invATA(2) = 0.5f;
  invATA(3) = 0.125f;
  invATA *= 1.0f / (dx * dx);

  auto set_voxel_center_solid_grad_and_vn =
      [&](openvdb::Vec3fTree::LeafNodeType &leaf, openvdb::Index leafpos) {
        auto vnleaf = voxel_center_solid_vn->tree().probeLeaf(leaf.origin());
        auto sdfaxr = in_solid_sdf->getConstAccessor();
        auto velaxr = in_solid_vel->getConstAccessor();

        Eigen::VectorXf data;
        data.setZero(8);

        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
          auto gcoord = iter.getCoord();
          // calculate the sdf at this voxel center
          auto &i = gcoord[0];
          auto &j = gcoord[1];
          auto &k = gcoord[2];
          data[0] = sdfaxr.getValue(gcoord); // 000
          k++;
          data[1] = sdfaxr.getValue(gcoord); // 001
          j++;
          data[3] = sdfaxr.getValue(gcoord); // 011
          k--;
          data[2] = sdfaxr.getValue(gcoord); // 010
          i++;
          data[6] = sdfaxr.getValue(gcoord); // 110
          k++;
          data[7] = sdfaxr.getValue(gcoord); // 111
          j--;
          data[5] = sdfaxr.getValue(gcoord); // 101
          k--;
          data[4] = sdfaxr.getValue(gcoord); // 100

          Eigen::VectorXf abcd;
          abcd = invATA.array() * (AT * data).array();

          // d is the sdf at center
          if (abcd[3] < 0.5f) {
            openvdb::Vec3f abc{abcd[0], abcd[1], abcd[2]};
            abc.normalize();
            leaf.setValueOn(iter.offset(), abc);

            // solid velocity on this direction
            auto solidvel = velaxr.getValue(gcoord);
            vnleaf->setValueOn(iter.offset(), solidvel.dot(abc));
          } else {
            leaf.setValueOff(iter.offset());
            vnleaf->setValueOff(iter.offset());
          }
        } // for all on voxels
      };  // end set_voxel_center_solid_grad_and_vn
  auto voxel_center_solid_normalman =
      openvdb::tree::LeafManager<openvdb::Vec3fTree>(
          voxel_center_solid_normal->tree());
  if (has_valid_solid) {
    voxel_center_solid_normalman.foreach (set_voxel_center_solid_grad_and_vn);
  }

  voxel_center_solid_normal->pruneGrid();
  voxel_center_solid_vn->pruneGrid();

  // openvdb::io::File("vn.vdb").write({ voxel_center_solid_vn,
  // voxel_center_solid_normal });

  std::vector<openvdb::points::PointDataTree::LeafNodeType *> particle_leaves;
  in_out_points->tree().getNodes(particle_leaves);

  auto to_use_liquid_sdf = openvdb::FloatGrid::create(dx);
  to_use_liquid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
  to_use_liquid_sdf->setTransform(
      openvdb::math::Transform::createLinearTransform(dx));
  if (in_liquid_sdf != nullptr) {
    to_use_liquid_sdf = in_liquid_sdf->deepCopy();
  }
  auto reducer = std::make_unique<point_to_counter_reducer2>(
      dt, dx, *to_use_liquid_sdf, *in_velocity_field,
      *in_velocity_field_to_be_advected, *in_old_velocity, *in_solid_sdf,
      *voxel_center_solid_normal, *voxel_center_solid_vn, PIC_component,
      particle_leaves, surfacedist, RK_order);
  tbb::parallel_reduce(
      tbb::blocked_range<openvdb::Index>(0, particle_leaves.size(), 10),
      *reducer);

  // compose the result

  auto newTree_leafman{
      openvdb::tree::LeafManager<openvdb::points::PointDataTree>(
          reducer->m_counter_grid->tree())};

  auto pnamepair = position_attribute::attributeType();
  auto position_attribute_descriptor =
      openvdb::points::AttributeSet::Descriptor::create(pnamepair);

  auto vnamepair = velocity_attribute::attributeType();
  auto pv_attribute_descriptor =
      position_attribute_descriptor->duplicateAppend("v", vnamepair);

  auto set_new_attribute_list =
      [&](openvdb::points::PointDataTree::LeafNodeType &leaf,
          openvdb::Index leafpos) {
        using namespace openvdb::tools::local_util;

        std::vector<int> voxel_particle_count;
        voxel_particle_count.resize(leaf.size());
        std::vector<openvdb::PointDataIndex32> index_ends;
        index_ends.assign(leaf.size(), 0);
        voxel_particle_count[0] = leaf.getValue(0);
        index_ends[0] = voxel_particle_count[0];
        for (auto offset = 1; offset < leaf.size(); ++offset) {
          voxel_particle_count[offset] = leaf.getValue(offset);
          index_ends[offset] =
              index_ends[offset - 1] + voxel_particle_count[offset];
        }

        // according to the index space leaf position, assign the particles to
        // the final attribute list
        // printf("leafcounter %d, attrcounter:%d\n", *index_ends.rbegin(),
        // reducer.toffset_oindex_oleafpos_hashmap[leaf.origin()]->size());
        // create the attribute set

        auto local_pv_descriptor = pv_attribute_descriptor;
        leaf.initializeAttributes(position_attribute_descriptor,
                                  *index_ends.rbegin());
        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);

        // attribute writer
        leaf.setOffsets(index_ends);

        // set the positions and velocities
        // get the new attribute arrays
        openvdb::points::AttributeArray &posarray = leaf.attributeArray("P");
        openvdb::points::AttributeArray &varray = leaf.attributeArray("v");

        // Create read handles for position and velocity
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::PositionCodec>
            posWHandle(posarray);
        openvdb::points::AttributeWriteHandle<openvdb::Vec3f,
                                              FLIP_vdb::VelocityCodec>
            vWHandle(varray);

        openvdb::Vec3i pcoord;
        int writing_offset;
        int writer_index;

        // move from the original leaves
        for (size_t i = 0;
             i <
             reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()]->size();
             i++) {
          auto &tooiol_vec =
              *reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()];
          writing_offset = std::get<0>(tooiol_vec[i]);
          writer_index =
              index_ends[writing_offset] - voxel_particle_count[writing_offset];
          voxel_particle_count[writing_offset]--;

          posarray.set(
              writer_index,
              particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("P"),
              std::get<1>(tooiol_vec[i]));
          varray.set(
              writer_index,
              particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("v"),
              std::get<1>(tooiol_vec[i]));
        }
      };

  // printf("compose_start_for_each\n");

  newTree_leafman.foreach (set_new_attribute_list);
  // printf("compose_end_for_each\n");

  reducer->m_counter_grid->setName("new_counter_grid");
  reducer->m_counter_grid->setTransform(in_out_points->transformPtr());
  // openvdb::io::File("new_advect.vdb").write({reducer.m_counter_grid});
  // printf("compose_start_replace_tree\n");

  in_out_points->setTree(reducer->m_counter_grid->treePtr());
  // printf("compose_end_replace_tree\n");
}

void FLIP_vdb::point_integrate_vector(
    openvdb::points::PointDataGrid::Ptr &in_out_particles, openvdb::Vec3R &dx,
    std::string channel) {
  std::vector<openvdb::points::PointDataTree::LeafNodeType *> leafs;
  in_out_particles->tree().getNodes(leafs);
  auto transform = in_out_particles->transformPtr();

  tbb::parallel_for(
      tbb::blocked_range<openvdb::Index>(0, leafs.size(), 1),
      [&](tbb::blocked_range<openvdb::Index> r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          auto leaf = leafs[i];
          // attributes
          // Attribute reader
          // Extract the position attribute from the leaf by name (P is
          // position).
          openvdb::points::AttributeArray &positionArray =
              leaf->attributeArray("P");
          // Extract the velocity attribute from the leaf by name (v is
          // velocity).
          openvdb::points::AttributeArray &velocityArray =
              leaf->attributeArray("v");

          using PositionCodec =
              openvdb::points::FixedPointCodec</*one byte*/ false>;
          using VelocityCodec = openvdb::points::TruncateCodec;
          // Create read handles for position and velocity
          openvdb::points::AttributeHandle<openvdb::Vec3f, PositionCodec>
              positionHandle(positionArray);
          openvdb::points::AttributeHandle<openvdb::Vec3f, VelocityCodec>
              velocityHandle(velocityArray);
          openvdb::points::AttributeWriteHandle<openvdb::Vec3f, VelocityCodec>
              velocityWHandle(velocityArray);

          for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            openvdb::Vec3R v = velocityHandle.get(*iter);
            if (channel == std::string("vel")) {
              v += dx;
            }
            velocityWHandle.set(*iter, v);
          }
        }
      });
}
