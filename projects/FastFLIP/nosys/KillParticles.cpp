#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include "FLIP_vdb.h"
#include "levelset_util.h"
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tree/LeafManager.h>

namespace zeno {

static void kill_particles_inside(
        openvdb::points::PointDataGrid::Ptr &points,
        openvdb::FloatGrid::Ptr &solid_sdf) {

   		auto pnamepair = FLIP_vdb::position_attribute::attributeType();
   		auto m_p_descriptor =
   openvdb::points::AttributeSet::Descriptor::create(pnamepair);

   		auto vnamepair = FLIP_vdb::velocity_attribute::attributeType();
   		auto m_pv_descriptor =
   m_p_descriptor->duplicateAppend("v", vnamepair);

  using pos_offset_t = std::pair<openvdb::Vec3f, openvdb::Index>;
  using node_t = openvdb::points::PointDataGrid::TreeType::LeafNodeType;

  auto kill_particles_op = [&](openvdb::points::PointDataTree::LeafNodeType &leaf,
          openvdb::Index leafpos) {

    auto solid_axr = solid_sdf->getConstAccessor();

    std::vector<openvdb::PointDataIndex32> new_attribute_offsets;
    std::vector<openvdb::Vec3f> new_positions;
    std::vector<openvdb::Vec3f> new_velocity;

    // minimum number of particle per voxel
    const int min_np = 8;
    // max number of particle per voxel
    const int max_np = 16;

    new_positions.reserve(max_np * 512);
    new_velocity.reserve(max_np * 512);

      /*// If this leaf is totally outside of the bounding box
      if (!boxes_overlap(leaf.origin(), leaf.origin() + openvdb::Coord{8},
                         m_domain_begin, m_domain_end)) {
        leaf.clearAttributes();
        continue;
      }*/

      // this leaf could potentially be a empty node without attribute
      // if so, create the position and velocity attribute
      if (leaf.attributeSet().size() == 0) {
        auto local_pv_descriptor = m_pv_descriptor;
        leaf.initializeAttributes(m_p_descriptor, 0);

        leaf.appendAttribute(leaf.attributeSet().descriptor(),
                             local_pv_descriptor, 1);
      }

      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      const openvdb::points::AttributeArray &positionArray =
          leaf.attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      const openvdb::points::AttributeArray &velocityArray =
          leaf.attributeArray("v");

      openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>
          positionHandle(positionArray);
      openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>
          velocityHandle(velocityArray);

      // clear the new offset to be assigned
      new_attribute_offsets.clear();
      new_positions.clear();
      new_velocity.clear();

      openvdb::Index current_particle_count = 0;
      new_positions.reserve(positionArray.size());
      new_velocity.reserve(velocityArray.size());

      for (openvdb::Index offset = 0; offset < leaf.SIZE; offset++) {
        openvdb::Index original_attribute_begin = 0;
        if (offset != 0) {
          original_attribute_begin = leaf.getValue(offset - 1);
        const openvdb::Index original_attribute_end = leaf.getValue(offset);
        const auto voxel_gcoord = leaf.offsetToGlobalCoord(offset);

          for (int i_emit = original_attribute_begin;
               i_emit < original_attribute_end; i_emit++) {
            auto current_pos = positionHandle.get(i_emit);
              if (openvdb::tools::BoxSampler::sample(
                      solid_axr, voxel_gcoord + current_pos) > 0) {
                new_positions.push_back(current_pos);
                new_velocity.push_back(velocityHandle.get(i_emit));
                current_particle_count++;
              } // end if the particle position if outside solid
            }
          }
          new_attribute_offsets.push_back(current_particle_count);
        }
          // current_particle_count += original_attribute_end -
          // original_attribute_begin;

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
  };

  openvdb::tree::LeafManager<openvdb::points::PointDataTree>
    points_leafman(points->tree());
  points_leafman.foreach (kill_particles_op);
}

struct KillParticlesInSDF : zeno::INode {
    virtual void apply() override {
        auto points = get_input<VDBPointsGrid>("Particles");
        auto sdf = get_input<VDBFloatGrid>("KillerSDF");

        kill_particles_inside(points->m_grid, sdf->m_grid);
        set_output("Particles", std::move(points));
    }
};

ZENDEFNODE(KillParticlesInSDF, {
    {"Particles", "KillerSDF"},
    {"Particles"},
    {},
    {"FLIPSolver"},
});

}
