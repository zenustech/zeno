#include "tbb/scalable_allocator.h"
#include "FLIP_vdb.h"
#include <openvdb/points/PointCount.h>
#include "openvdb/tree/LeafManager.h"
#include "openvdb/points/PointAdvect.h"
#include "openvdb/tools/Morphology.h"
#include "openvdb/tools/MeshToVolume.h"
#include "SimOptions.h"
#include "Timer.h"
#include "levelset_util.h"

#include <atomic>
//intrinsics
#include <xmmintrin.h>
#include "simd_vdb_poisson.h"
// #include "BPS3D_volume_vel_cache.h" // decouple Datan BEM works for now

#include <thread>
#include "tbb/blocked_range3d.h"
namespace {
	struct BoxSampler {
		template <typename axr_t>
		static void get_eight_data(openvdb::Vec3f* data,
			axr_t& axr,
			openvdb::Coord gcoord) {
			auto& i = gcoord[0];
			auto& j = gcoord[1];
			auto& k = gcoord[2];
			//000
			data[0] = axr.getValue(gcoord);
			//001
			k++;
			data[1] = axr.getValue(gcoord);
			//011
			j++;
			data[3] = axr.getValue(gcoord);
			//010
			k--;
			data[2] = axr.getValue(gcoord);
			//110
			i++;
			data[6] = axr.getValue(gcoord);
			//111
			k++;
			data[7] = axr.getValue(gcoord);
			//101
			j--;
			data[5] = axr.getValue(gcoord);
			//100
			k--;
			data[4] = axr.getValue(gcoord);
		}

		//w=0->a
		//w=1->b
		static float mix(float a, float b, float w) {
			return a + (b - a) * w;
		}

		//channel mix
		template<int c>
		static float mixc(const openvdb::Vec3f& a, const openvdb::Vec3f& b, float w) {
			return a[c] + (b[c] - a[c]) * w;
		}

		/*void trilinear_interpolate(openvdb::Vec3f& out, openvdb::Vec3f* data, float wx, float wy, float wz) {
			out = mix(
				mix(mix(data[0], data[1], wz), mix(data[2], data[3], wz), wy),
				mix(mix(data[4], data[5], wz), mix(data[6], data[7], wz), wy),
				wx);
		}*/

		//channel interpolation
		template<int c>
		static void trilinear_interpolatec(float& out, const openvdb::Vec3f* data, float wx, float wy, float wz) {
			out = mix(
				mix(mix(data[0][c], data[1][c], wz), mix(data[2][c], data[3][c], wz), wy),
				mix(mix(data[4][c], data[5][c], wz), mix(data[6][c], data[7][c], wz), wy),
				wx);
		}

		//channel interpolation
		template<int c, typename axr_t>
		static float samplec(axr_t& axr, const openvdb::Vec3f& ixyz) {
			using  namespace openvdb::tools::local_util;
			openvdb::Coord base{ floorVec3(ixyz) };
			openvdb::Vec3f data[8];
			get_eight_data(data, axr, base);
			float result;
			trilinear_interpolatec<c>(result, data, ixyz.x() - base.x(), ixyz.y() - base.y(), ixyz.z() - base.z());
			return result;
		};
	};


	struct StaggeredBoxSampler {
		template<typename axr_t>
		static openvdb::Vec3f sample(axr_t& axr, const openvdb::Vec3f& xyz) {
			float vx = BoxSampler::samplec<0>(axr, openvdb::Vec3f{ xyz.x() + 0.5f, xyz.y(), xyz.z() });
			float vy = BoxSampler::samplec<1>(axr, openvdb::Vec3f{ xyz.x(), xyz.y() + 0.5f, xyz.z() });
			float vz = BoxSampler::samplec<2>(axr, openvdb::Vec3f{ xyz.x(), xyz.y(), xyz.z() + 0.5f });

			return openvdb::Vec3f{ vx,vy,vz };
		};
	};
}
FLIP_vdb::FLIP_vdb(const init_arg_t& in_arg)
{
	openvdb::initialize();
	using  namespace openvdb::tools::local_util;
	m_framenumber = 0;

	m_collide_with_domain = true;

	m_dx = in_arg.m_dx;
	m_cfl = 1.f;
	m_particle_radius = 0.5f * std::sqrt(3.0f) * m_dx * 1.01f;

	//construct the index minimum and maximum that defines the simulation domain
	m_domain_index_begin = floorVec3(in_arg.m_domain_vecf_min / m_dx);
	m_domain_index_end = ceilVec3(in_arg.m_domain_vecf_max / m_dx);

	m_voxel_center_transform = openvdb::math::Transform::createLinearTransform(m_dx);
	m_voxel_vertex_transform = openvdb::math::Transform::createLinearTransform(m_dx);
	m_voxel_vertex_transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(m_dx));

	initialize_attribute_descriptor();
	init_grids();
	//set the original velocity volume;
	//particle_to_grid_collect_style();
	//extrapolate_velocity(5);
	//particle_to_grid_reduce_style();
	//take_shapshot();
}

void FLIP_vdb::init_grids()
{
	//init_boundary_fill_kill_volume();
	//init_particles();
	//init_empty_point_grid();

	//initialize all grids
	m_pressure = openvdb::FloatGrid::create(float(0));
	m_pressure->setTransform(m_voxel_center_transform);
	m_pressure->setGridClass(openvdb::GridClass::GRID_FOG_VOLUME);
	m_pressure->setName("Pressure");

	m_rhsgrid = m_pressure->deepCopy();
	m_rhsgrid->setName("RHS");

	//velocity
	m_velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0 });
	m_velocity->setTransform(m_voxel_center_transform);
	m_velocity->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	m_velocity->setName("Velocity");

	m_velocity_after_p2g = m_velocity->deepCopy();

	m_velocity_snapshot = m_velocity->deepCopy();
	m_velocity_snapshot->setName("Velocity_Snapshot");

	//solid velocity
	m_solid_velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
	m_solid_velocity->setTransform(m_voxel_center_transform);
	m_solid_velocity->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	m_solid_velocity->setName("Solid_Velocity");

	//velocity weights for p2g
	m_velocity_weights = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
	m_velocity_weights->setTransform(m_voxel_center_transform);
	m_velocity_weights->setName("Velocity_P2G_Weights");

	//face weights for pressure
	m_face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
	m_face_weight->setName("Face_Weights");
	m_face_weight->setTransform(m_voxel_center_transform);
	m_face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);

	//liquid sdf
	m_liquid_sdf = openvdb::FloatGrid::create(0.9f * m_dx);
	m_liquid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	m_liquid_sdf->setTransform(m_voxel_center_transform);
	m_liquid_sdf->setName("Liquid_SDF");

	//pushed out liquid sdf used to identify true liquid
	m_pushed_out_liquid_sdf = m_liquid_sdf->deepCopy();

	//solid sdf
	//set background value to be positive to show it is away from the solid
	//treat it as the narrow band width
	m_solid_sdf = openvdb::FloatGrid::create(3.f * m_dx);
	m_solid_sdf->setTransform(m_voxel_vertex_transform);
	m_solid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	m_solid_sdf->setName("Solid_SDF");

	//init_domain_solid_sdf();
	//update_solid_sdf();
}


// void FLIP_vdb::init_empty_point_grid()
// {
// 	m_particles = openvdb::points::PointDataGrid::create();
// 	m_particles->setTransform(m_voxel_center_transform);
// 	m_particles->setName("Particles");
// 	auto first_leaf = m_particles->treePtr()->touchLeaf(openvdb::Coord(m_domain_index_begin));

// 	//add descriptors
// 	first_leaf->initializeAttributes(pos_descriptor(), 0);
// 	first_leaf->appendAttribute(first_leaf->attributeSet().descriptor(), m_pv_attribute_descriptor, 1);

// 	printf("original leaf count:%d\n", m_particles->tree().leafCount());
// }

void FLIP_vdb::write_points(const std::string& fname) const
{
	// Create a VDB file object and write out the grid.
	openvdb::io::File(fname).write({ m_particles });
}

bool FLIP_vdb::test()
{

	float target_t = Options::doubleValue("time-step");

	if (m_framenumber * target_t > Options::doubleValue("simulation-time")) {
    return false;
	}
	float stepped = 0;
	while (stepped < target_t) {
		auto cfl_dt = 4 * cfl_and_regularize_velocity();
		float estimated_steps = std::ceil((target_t - stepped) / cfl_dt);
		float estimated_dt = (target_t - stepped) / estimated_steps * 1.1;
		float dt = std::min(estimated_dt, cfl_dt);
		//dont do that if we can just finish in cfl_dt
		if (stepped + cfl_dt > target_t) {
			dt = cfl_dt;
		}

		if (stepped + dt > target_t) {
			dt = target_t - stepped;
		}

		printf("stepped:%.4f, dt:%.4f cfl:%.4f\n", stepped, dt, cfl_dt);
		//update teh solid
		//update_solid_sdf();

		CSim::TimerMan::timer("Sim.step/vdbflip/advection").start();
		advection(dt);
		CSim::TimerMan::timer("Sim.step/vdbflip/advection").stop();
		printf("advection done\n");

		CSim::TimerMan::timer("Sim.step/vdbflip/seed").start();
		fill_kill_particles();
		CSim::TimerMan::timer("Sim.step/vdbflip/seed").stop();
		printf("fill_kill_particles done\n");
		CSim::TimerMan::timer("Sim.step/vdbflip/p2gcollect").start();
		particle_to_grid_collect_style();
		extrapolate_velocity(1);
		//particle_to_grid_reduce_style();
		calculate_face_weights();
		clamp_liquid_phi_in_solids();
		printf("particle_to_grid done\n");
		CSim::TimerMan::timer("Sim.step/vdbflip/p2gcollect").stop();

		CSim::TimerMan::timer("Sim.step/vdbflip/set_face_weights").start();
		apply_body_force(dt);
		set_solid_velocity();
		CSim::TimerMan::timer("Sim.step/vdbflip/set_face_weights").stop();

		CSim::TimerMan::timer("Sim.step/vdbflip/pressure").start();
		//solve_pressure(dt);

		solve_pressure_simd(dt);
;
		CSim::TimerMan::timer("Sim.step/vdbflip/pressure").stop();
		CSim::TimerMan::timer("Sim.step/vdbflip/extrapolate").start();
		extrapolate_velocity();
		CSim::TimerMan::timer("Sim.step/vdbflip/extrapolate").stop();
		stepped += dt;
		cfl();
	}

	m_framenumber++;
  return true;

}

// void FLIP_vdb::step(float in_dt)
// {
// 	float target_t = in_dt;
// 	float stepped = 0.f;
// 	while (stepped < target_t) {
// 		auto cfl_dt = m_cfl * cfl_and_regularize_velocity();
// 		float estimated_steps = std::ceil((target_t - stepped) / cfl_dt);
// 		float estimated_dt = (target_t - stepped) / estimated_steps * 1.1;
// 		float dt = std::min(estimated_dt, cfl_dt);
// 		if (stepped + cfl_dt > target_t) {
// 			dt = cfl_dt;
// 		}

// 		if (stepped + dt > target_t) {
// 			dt = target_t - stepped;
// 		}

// 		printf("1\n");
		
// 		advection(dt);
// 		printf("2\n");
		
// 		//emit new fluid particles
// 		//emit_liquid(m_particles,m_source_sdfvel, m_domain_index_begin*m_dx, m_domain_index_end*m_dx);
// 		printf("3\n");
		
// 		//remove particles outside of the boundary, or in the sink zone.
// 		//solids are also a type of sink
// 		//sink_liquid(m_sink_sdfs);
// 		printf("4\n");

// 		//update_solid_sdf_vel(m_solid_sdfvel);
// 		printf("5\n");

// 		particle_to_grid_collect_style();
// 		printf("6\n");
// 		extrapolate_velocity(1);
// 		printf("7\n");
// 		calculate_face_weights();
// 		printf("8\n");
// 		clamp_liquid_phi_in_solids();
// 		printf("9\n");
// 		apply_body_force(dt);
// 		printf("10\n");
// 		solve_pressure_simd(dt);
// 		printf("11\n");
// 		extrapolate_velocity();
// 		printf("12\n");
// 		stepped += dt;
// 		cfl();
// 		printf("13\n");
// 	}
// }

// void FLIP_vdb::IO(std::string filename)
// {
// 	//auto copied_particles = narrow_band_particles();

// 	auto nb_particles = openvdb::points::PointDataGrid::create();
// 	auto interior_sdf = openvdb::FloatGrid::create();
// 	//narrow_band_particles(nb_particles, interior_sdf, m_particles);

// 	auto copied_pressure = m_pressure->deepCopy();
// 	auto copied_solid_sdf = m_solid_sdf->deepCopy();
// 	auto copied_liquid_sdf = m_pushed_out_liquid_sdf->deepCopy();
// 	auto copied_velocity = m_velocity->deepCopy();
// 	//auto copied_old_velocity = m_velocity_after_p2g->deepCopy();
// 	//auto copied_face_weights = m_face_weight->deepCopy();
// 	//auto copied_update = m_velocity_update->deepCopy();
// 	//auto copied_RHS = m_rhsgrid->deepCopy();

// 	auto save_vdb = [&]() {
// 		/*auto diff_pressure = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
// 				iter.setValue(iter.getValue() + 9.81 * iter.getCoord()[1] * m_dx);
// 			}
// 		};
// 		openvdb::tree::LeafManager<openvdb::FloatTree>(copied_pressure->tree()).foreach(diff_pressure);*/
// 		printf("async_write frame %d\n", m_framenumber);
// 		CSim::TimerMan::timer("Sim.step/vdbflip/IO").start();
// 		std::stringstream s;
// 		s << filename << m_framenumber << ".vdb";
// 		m_boundary_velocity_volume->setName("boundary_velocity_volume");
// 		//openvdb::io::File(s.str()).write({ m_particles, m_velocity, m_velocity_weights, m_liquid_sdf,  m_solid_sdf, m_pressure, m_solid_velocity });
// 		//openvdb::io::File(s.str()).write({ copied_particles , copied_liquid_sdf,copied_solid_sdf, copied_velocity, copied_old_velocity, copied_pressure, copied_face_weights, copied_update, copied_RHS ,m_solid_velocity->deepCopy() });
// 		openvdb::io::File(s.str()).write({ copied_particles ,m_shrinked_liquid_sdf });
// 		CSim::TimerMan::timer("Sim.step/vdbflip/IO").stop();
// 	};

// 	auto save_thread{ std::thread(save_vdb) };
// 	save_thread.join();
// }

namespace {

	//reduction style p2g operator designed for the leaf manager class
	struct particle_to_grid_reducer {

		//constructor
		particle_to_grid_reducer(
			openvdb::Vec3fGrid::Ptr in_velocity,
			openvdb::Vec3fGrid::Ptr in_velocity_weights,
			openvdb::FloatGrid::Ptr in_liquid_sdf,
			float in_particle_radius
		) {
			m_particle_radius = in_particle_radius;
			m_transform = in_velocity->transformPtr();
			in_velocity->clear();
			in_velocity_weights->clear();
			in_liquid_sdf->clear();

			//the final result will be directly modify the input argument
			m_velocity = in_velocity;
			m_velocity_weights = in_velocity_weights;
			m_liquid_sdf = in_liquid_sdf;

			//set the loop order and offset of the u v w phi sampling point
			//with respect to the center voxel center
			x_offset_to_center_voxel_center = std::make_shared<std::array<__m128, 27>>();
			y_offset_to_center_voxel_center = std::make_shared<std::array<__m128, 27>>();
			z_offset_to_center_voxel_center = std::make_shared<std::array<__m128, 27>>();
			loop_order = std::make_shared<std::array<openvdb::Coord, 27>>();

			//u:  (-0.5, 0, 0)
			//v:  (0, -0.5, 0)
			//w:  (0, 0, -0.5)
			//phi:(0, 0, 0)
			//note we use set so the 
			//R0 R1 R2 R3 correspond to the 0,1,2,3 argument
			//when later extracting using _mm_storer_ps, we get float[0] = arg0 correctly
			//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_store.htm
			//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_set.htm

			__m128 xpack = _mm_set_ps(-0.5f, 0.f, 0.f, 0.f);
			__m128 ypack = _mm_set_ps(0.f, -0.5f, 0.f, 0.f);
			__m128 zpack = _mm_set_ps(0.f, 0.f, -0.5f, 0.f);
			for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
				int ijk = ivoxel;
				int basex = ijk / 9; ijk -= 9 * basex;
				int basey = ijk / 3; ijk -= 3 * basey;
				int basez = ijk;
				//becomes -1 -> 1

				basex -= 1; basey -= 1; basez -= 1;
				//broadcast four float as the base
				__m128 basex4 = _mm_set_ps1(float(basex));
				__m128 basey4 = _mm_set_ps1(float(basey));
				__m128 basez4 = _mm_set_ps1(float(basez));

				loop_order->at(ivoxel) = openvdb::Coord{ basex, basey, basez };

				x_offset_to_center_voxel_center->at(ivoxel) = _mm_add_ps(basex4, xpack);
				y_offset_to_center_voxel_center->at(ivoxel) = _mm_add_ps(basey4, ypack);
				z_offset_to_center_voxel_center->at(ivoxel) = _mm_add_ps(basez4, zpack);
			}
		}

		//split constructor
		particle_to_grid_reducer(particle_to_grid_reducer& other, tbb::split) {
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


		//operator
		//this opeartor is called inside a range loop inthe leaf manager
		void operator() (
			openvdb::points::PointDataGrid::TreeType::LeafNodeType& particle_leaf,
			openvdb::Index leafpos) {

			const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
			const __m128 float1x4 = _mm_set_ps1(float(1));
			const __m128 float0x4 = _mm_set_ps1(float(0));

			//attribute reader
			// Extract the position attribute from the leaf by name (P is position).
			const openvdb::points::AttributeArray& positionArray =
				particle_leaf.constAttributeArray("P");
			// Extract the velocity attribute from the leaf by name (v is velocity).
			const openvdb::points::AttributeArray& velocityArray =
				particle_leaf.constAttributeArray("v");

			// Create read-only handles for position and velocity.
			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);

			//scatter style transfer
			//loop over all particles
			float tophix, tophiy, tophiz;
			float dx = m_transform->voxelSize()[0];

			//accessor to the three grids
			auto vel_accessor = m_velocity->getAccessor();
			auto vel_weights_accessor = m_velocity_weights->getAccessor();
			auto liquid_sdf_accessor = m_liquid_sdf->getAccessor();


			//most of the time the particle will write to the current leaf
			auto* local_vel_leaf = vel_accessor.touchLeaf(particle_leaf.origin());
			auto* local_vel_weights_leaf = vel_weights_accessor.touchLeaf(particle_leaf.origin());
			auto* local_sdf_leaf = liquid_sdf_accessor.touchLeaf(particle_leaf.origin());


			neib_sdf_leaf_ptr_cache.fill(nullptr);
			neib_vel_leaf_ptr_cache.fill(nullptr);
			neib_vel_weights_leaf_ptr_cache.fill(nullptr);

			//set the center cache
			neib_sdf_leaf_ptr_cache[9 + 3 + 1] = local_sdf_leaf;
			neib_vel_leaf_ptr_cache[9 + 3 + 1] = local_vel_leaf;
			neib_vel_weights_leaf_ptr_cache[9 + 3 + 1] = local_vel_weights_leaf;

			for (auto piter = particle_leaf.beginIndexOn(); piter; ++piter) {

				auto voxelpos = positionHandle.get(*piter);
				auto pvel = velocityHandle.get(*piter);

				//broadcast the variables
				__m128 particle_x = _mm_set_ps1(voxelpos[0]);
				__m128 particle_y = _mm_set_ps1(voxelpos[1]);
				__m128 particle_z = _mm_set_ps1(voxelpos[2]);


				//calculate the distance to the 27 neib uvw phi samples
				for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
					//calculate the distance
					//arg(A,B): ret A-B
					// the absolute value trick: abs_mask: 01111111..32bit..1111 x 4
					// _mm_and_ps(abs_mask(), v);
					x_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(x_offset_to_center_voxel_center->at(ivoxel), particle_x));
					y_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(y_offset_to_center_voxel_center->at(ivoxel), particle_y));
					z_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(z_offset_to_center_voxel_center->at(ivoxel), particle_z));

					//the distance to the phi variable
					_mm_store_ss(&tophix, x_dist_particle_to_sample);
					_mm_store_ss(&tophiy, y_dist_particle_to_sample);
					_mm_store_ss(&tophiz, z_dist_particle_to_sample);
					dist_to_phi_sample = dx * std::sqrt(tophix * tophix + tophiy * tophiy + tophiz * tophiz);


					//the uvw weights trilinear
					//transfer the distance to weight at the 27 voxels
					//(1-dist)
					//the far points now becomes negative
					x_dist_particle_to_sample = _mm_sub_ps(float1x4, x_dist_particle_to_sample);
					y_dist_particle_to_sample = _mm_sub_ps(float1x4, y_dist_particle_to_sample);
					z_dist_particle_to_sample = _mm_sub_ps(float1x4, z_dist_particle_to_sample);

					//turn everything positive or zero
					//now the dist_to_sample is actually the component-wise weight on the voxel
					//time to multiply them together
					x_dist_particle_to_sample = _mm_max_ps(float0x4, x_dist_particle_to_sample);
					y_dist_particle_to_sample = _mm_max_ps(float0x4, y_dist_particle_to_sample);
					z_dist_particle_to_sample = _mm_max_ps(float0x4, z_dist_particle_to_sample);

					//turn them into weights reduce to x
					x_dist_particle_to_sample = _mm_mul_ps(x_dist_particle_to_sample, y_dist_particle_to_sample);
					x_dist_particle_to_sample = _mm_mul_ps(x_dist_particle_to_sample, z_dist_particle_to_sample);
					//}//end for 27 voxel

					////write to the grid
					//for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
					alignas(16) float packed_weight[4];
					_mm_storer_ps(packed_weight, x_dist_particle_to_sample);

					openvdb::Coord write_coord = piter.getCoord() + loop_order->at(ivoxel);

					//check if the write position is local
					auto c = write_coord - particle_leaf.origin();
					int in_neib = 9 + 3 + 1;
					if (c[0] < 0) {
						in_neib -= 9;
					}
					else if (c[0] >= 8) {
						in_neib += 9;
					}

					if (c[1] < 0) {
						in_neib -= 3;
					}
					else if (c[1] >= 8) {
						in_neib += 3;
					}

					if (c[2] < 0) {
						in_neib -= 1;
					}
					else if (c[2] >= 8) {
						in_neib += 1;
					}

					//neib_offset3d could hav negative items
					//bool is_local = (c[0] >= 0) && (c[1] >= 0) && (c[2] >= 0) && (c[0] < 8) && (c[1] < 8) && (c[2] < 8);

					openvdb::Vec3fGrid::TreeType::LeafNodeType* new_vel_leaf = local_vel_leaf;
					openvdb::Vec3fGrid::TreeType::LeafNodeType* new_vel_weights_leaf = local_vel_weights_leaf;
					openvdb::FloatGrid::TreeType::LeafNodeType* new_sdf_leaf = local_sdf_leaf;

					//if (!is_local) {
						//cache the neighbor leafs
					if (!neib_sdf_leaf_ptr_cache[in_neib]) {
						neib_vel_leaf_ptr_cache[in_neib] = vel_accessor.touchLeaf(write_coord);
						neib_vel_weights_leaf_ptr_cache[in_neib] = vel_weights_accessor.touchLeaf(write_coord);
						neib_sdf_leaf_ptr_cache[in_neib] = liquid_sdf_accessor.touchLeaf(write_coord);
					}

					{
						new_vel_leaf = neib_vel_leaf_ptr_cache[in_neib];
						new_vel_weights_leaf = neib_vel_weights_leaf_ptr_cache[in_neib];
						new_sdf_leaf = neib_sdf_leaf_ptr_cache[in_neib];
					}
					//}



					openvdb::Index write_offset = new_vel_leaf->coordToOffset(write_coord);
					openvdb::Vec3f weights{ packed_weight[0],packed_weight[1],packed_weight[2] };
					openvdb::Vec3f weighted_vel = pvel * weights;

					//write weighted velocity
					auto original_weighted_vel = new_vel_leaf->getValue(write_offset);
					new_vel_leaf->setValueOn(write_offset, weighted_vel + original_weighted_vel);

					//write weights
					auto original_weights = new_vel_weights_leaf->getValue(write_offset);
					new_vel_weights_leaf->setValueOn(write_offset, weights + original_weights);

					//phi
					//compare 
					float original_sdf = new_sdf_leaf->getValue(write_offset);
					new_sdf_leaf->setValueOn(write_offset, std::min(original_sdf, dist_to_phi_sample - m_particle_radius));
				}//end for its 27 neighbor

			}//end for all points in this leaf
		}//end operator()

		//join operator
		void join(const particle_to_grid_reducer& other) {
			//velocity
			for (auto leaf = other.m_velocity->tree().beginLeaf(); leaf; ++leaf) {
				auto* newLeaf = m_velocity->tree().probeLeaf(leaf->origin());
				if (!newLeaf) {
					auto& tree = const_cast<openvdb::Vec3fGrid&>(*other.m_velocity).tree();
					m_velocity->tree().addLeaf(
						tree.template stealNode<openvdb::Vec3fGrid::TreeType::LeafNodeType>(
							leaf->origin(), openvdb::Vec3f{ 0 }, false));
				}
				else {
					// otherwise increment existing values
					for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
						auto val = newLeaf->getValue(iter.offset());
						val += *iter;
						newLeaf->setValueOn(iter.offset(), val);
					}
				}
			}

			//velocity weights
			for (auto leaf = other.m_velocity_weights->tree().beginLeaf(); leaf; ++leaf) {
				auto* newLeaf = m_velocity_weights->tree().probeLeaf(leaf->origin());
				if (!newLeaf) {
					auto& tree = const_cast<openvdb::Vec3fGrid&>(*other.m_velocity_weights).tree();
					m_velocity_weights->tree().addLeaf(
						tree.template stealNode<openvdb::Vec3fGrid::TreeType::LeafNodeType>(
							leaf->origin(), openvdb::Vec3f{ 0 }, false));
				}
				else {
					// otherwise increment existing values
					for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
						auto val = newLeaf->getValue(iter.offset());
						val += *iter;
						newLeaf->setValueOn(iter.offset(), val);
					}
				}
			}

			//liquid sdf
			for (auto leaf = other.m_liquid_sdf->tree().beginLeaf(); leaf; ++leaf) {
				auto* newLeaf = m_liquid_sdf->tree().probeLeaf(leaf->origin());
				if (!newLeaf) {
					//it doesn't matter what const it put back in the other leaf, since it is going to be deleted anyway
					auto& tree = const_cast<openvdb::FloatGrid&>(*other.m_liquid_sdf).tree();
					m_liquid_sdf->tree().addLeaf(
						tree.template stealNode<openvdb::FloatGrid::TreeType::LeafNodeType>(
							leaf->origin(), 1.f, false));
				}
				else {
					//update current sdf
					for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
						auto val = newLeaf->getValue(iter.offset());
						val = std::min(val, *iter);
						newLeaf->setValueOn(iter.offset(), val);
					}
				}
			}
		}

		//particle radius 
		float m_particle_radius;



		//neighbor leaf pointer cache
		std::array<openvdb::Vec3fGrid::TreeType::LeafNodeType*, 27> neib_vel_leaf_ptr_cache;
		std::array<openvdb::Vec3fGrid::TreeType::LeafNodeType*, 27> neib_vel_weights_leaf_ptr_cache;
		std::array<openvdb::FloatGrid::TreeType::LeafNodeType*, 27> neib_sdf_leaf_ptr_cache;


		//constant distances to the center voxel
		std::shared_ptr<std::array<__m128, 27>> x_offset_to_center_voxel_center;
		std::shared_ptr<std::array<__m128, 27>> y_offset_to_center_voxel_center;
		std::shared_ptr<std::array<__m128, 27>> z_offset_to_center_voxel_center;
		std::shared_ptr<std::array<openvdb::Coord, 27>> loop_order;

		//pre-allocated space for computation
		__m128 x_dist_particle_to_sample;
		__m128 y_dist_particle_to_sample;
		__m128 z_dist_particle_to_sample;

		float dist_to_phi_sample;

		//reduction result
		openvdb::Vec3fGrid::Ptr m_velocity;
		openvdb::Vec3fGrid::Ptr m_velocity_weights;
		openvdb::FloatGrid::Ptr m_liquid_sdf;

		//transform information
		openvdb::math::Transform::Ptr m_transform;
	};

	//first step, for some of the velocity voxel
	//one of its weights is non zero because some particle touches it
	//however the velocity of other velocity is missing
	//deduce it from its 27 neighboring voxel that has non-zero weights
	//on this component
	struct deduce_missing_velocity_and_normalize {
		deduce_missing_velocity_and_normalize(
			openvdb::Vec3fGrid::Ptr in_velocity_weights,
			openvdb::Vec3fGrid::Ptr in_original_velocity
		) : original_velocity_accessor(in_original_velocity->getConstAccessor()),
			original_weights_accessor(in_velocity_weights->getConstAccessor()) {
			m_velocity_weights = in_velocity_weights;
			m_original_velocity = in_original_velocity;
		}

		void operator()(openvdb::Vec3fGrid::TreeType::LeafNodeType& vel_leaf, openvdb::Index leafpos) const {
			const auto* oweight_leaf = original_weights_accessor.probeConstLeaf(vel_leaf.origin());
			const auto* ovelocity_leaf = original_velocity_accessor.probeConstLeaf(vel_leaf.origin());

			if (!oweight_leaf) {
				printf("velocity voxel on, weight voxel not on. exit\n");
				exit(-1);
			}
			float epsl = 1e-5f;
			float weight_threshold = 1e-1f;
			for (openvdb::Index offset = 0; offset < vel_leaf.SIZE; offset++) {
				if (ovelocity_leaf->isValueMaskOff(offset)) {
					continue;
				}

				openvdb::Vec3f vel = ovelocity_leaf->getValue(offset);
				//check its three component
				for (int i_component = 0; i_component < 3; i_component++) {
					float weight = oweight_leaf->getValue(offset)[i_component];
					if (weight < weight_threshold) {
						//we have a voxel that is touched but does not have weights and velocity
						//deduce from neighbor
						openvdb::Coord c = ovelocity_leaf->offsetToGlobalCoord(offset);

						float total_weights = 0;
						float weighted_component = 0;

						for (int ii = -2; ii <= 2; ii++) {
							for (int jj = -2; jj <= 2; jj++) {
								for (int kk = -2; kk <= 2; kk++) {
									total_weights += original_weights_accessor.getValue(c + openvdb::Coord{ ii,jj,kk })[i_component];
									weighted_component += original_velocity_accessor.getValue(c + openvdb::Coord{ ii,jj,kk })[i_component];
								}
							}
						}

						if (total_weights == 0) {
							printf("a voxel is touched and has zero channel weight, but its neighbor voxel dont have touched channel\n");
							std::cout << "coordinate: " << c << std::endl;
							exit(-1);
						}

						vel[i_component] = weighted_component / (total_weights);
					}
					else {
						//weight !=0
						vel[i_component] /= weight;
					}
				}//end for three component find missing velocity

				//store the weighted version
				vel_leaf.setValueOn(offset, vel);
			}//end for all voxels in this leaf node
		}//end operator()

		openvdb::Vec3fGrid::ConstAccessor original_weights_accessor;
		openvdb::Vec3fGrid::ConstAccessor original_velocity_accessor;

		openvdb::Vec3fGrid::Ptr m_velocity_weights;
		openvdb::Vec3fGrid::Ptr m_original_velocity;
	};
}

void FLIP_vdb::add_solid_sdf(openvdb::FloatGrid::Ptr in_solid_sdf)
{
	openvdb::tools::signedFloodFill(in_solid_sdf->tree());
	m_moving_solids.push_back(in_solid_sdf);
}

void FLIP_vdb::add_propeller_sdf(openvdb::FloatGrid::Ptr in_propeller_sdf)
{
	m_propeller_sdf.push_back(in_propeller_sdf);
}

void FLIP_vdb::take_shapshot()
{
	m_velocity_snapshot = m_velocity->deepCopy();
	m_liquid_sdf_snapshot = m_pushed_out_liquid_sdf->deepCopy();
}

void FLIP_vdb::advection(float dt)
{

	float pic_component = 0.05;
	//update the FLIP particle velocity

	auto update_FLIP_velocity = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		// Attribute reader
		// Extract the position attribute from the leaf by name (P is position).
		const openvdb::points::AttributeArray& positionArray =
			leaf.attributeArray("P");
		// Extract the velocity attribute from the leaf by name (v is velocity).
		openvdb::points::AttributeArray& velocityArray =
			leaf.attributeArray("v");

		// Create read handles for position and velocity
		openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
		openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);

		auto ovaxr{ m_velocity_after_p2g->getConstAccessor() };
		auto vaxr{ m_velocity->getConstAccessor() };
		for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
			openvdb::Vec3R index_gpos = iter.getCoord().asVec3d() + positionHandle.get(*iter);
			auto original_vel = openvdb::tools::StaggeredBoxSampler::sample(ovaxr, index_gpos);
			auto updated_vel = openvdb::tools::StaggeredBoxSampler::sample(vaxr, index_gpos);
			auto old_pvel = velocityHandle.get(*iter);
			velocityHandle.set(*iter, (pic_component)*updated_vel + (1.0f - pic_component) * (updated_vel - original_vel + old_pvel));
		}

	};//end update flip velocity

	auto particle_man = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(m_particles->tree());
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/updatevel").start();
	//particle_man.foreach(update_FLIP_velocity);
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/updatevel").stop();
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/adv").start();
	//advect them in the velocity field
	//openvdb::points::advectPoints(*m_particles, *m_velocity, /*RK order*/ 1, /*time step*/dt, /*number of steps*/1);
	//std::cout << " particle mem " << m_particles->memUsage() << std::endl;
	custom_move_points_and_set_flip_vel(*m_particles, *m_velocity, *m_velocity_after_p2g, pic_component, dt, /*RK order*/ 1);
	//openvdb::points::advectPoints(*m_particles, *m_velocity, /*RK order*/ 4, /*time step*/dt, /*number of steps*/1);
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/adv").stop();

}

void FLIP_vdb::Advect(float dt, float dx, openvdb::points::PointDataGrid::Ptr particles, openvdb::Vec3fGrid::Ptr velocity,
openvdb::Vec3fGrid::Ptr velocity_after_p2g, float pic_component, int RK_ORDER)
{
		auto update_FLIP_velocity = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		// Attribute reader
		// Extract the position attribute from the leaf by name (P is position).
		const openvdb::points::AttributeArray& positionArray =
			leaf.attributeArray("P");
		// Extract the velocity attribute from the leaf by name (v is velocity).
		openvdb::points::AttributeArray& velocityArray =
			leaf.attributeArray("v");

		// Create read handles for position and velocity
		openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
		openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);

		auto ovaxr{ velocity_after_p2g->getConstAccessor() };
		auto vaxr{ velocity->getConstAccessor() };
		for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
			openvdb::Vec3R index_gpos = iter.getCoord().asVec3d() + positionHandle.get(*iter);
			auto original_vel = openvdb::tools::StaggeredBoxSampler::sample(ovaxr, index_gpos);
			auto updated_vel = openvdb::tools::StaggeredBoxSampler::sample(vaxr, index_gpos);
			auto old_pvel = velocityHandle.get(*iter);
			velocityHandle.set(*iter, (pic_component)*updated_vel + (1.0f - pic_component) * (updated_vel - original_vel + old_pvel));
		}
		auto particle_man = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(particles->tree());
		FLIP_vdb::custom_move_points_and_set_flip_vel(*particles, *velocity, *velocity_after_p2g, pic_component, dt, dx, RK_ORDER);
	};
}
namespace {
	struct custom_integrator {

		custom_integrator(const openvdb::Vec3fGrid& in_velocity, float in_dx, float in_dt) :
			vaxr(in_velocity.tree()), m_velocity(in_velocity) {
			dx = in_dx;
			invdx = 1.0f / dx;
			dt = in_dt;
			dtinvx = dt / in_dx;
		}

		void integrate1(openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) const {
			ipos += V0 * dtinvx;
		}

		void integrate2(openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) const {
			openvdb::Vec3f V1 = openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V0 * dtinvx);
			ipos += V1 * dtinvx;
		}

		void integrate3(openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) const {
			openvdb::Vec3f V1 = openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V0 * dtinvx);
			openvdb::Vec3f V2 = openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + dtinvx * (2.0f * V1 - V0));
			ipos += dtinvx * (V0 + 4.0f * V1 + V2) * (1.0f / 6.0f);
		}

		void integrate4(openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) const {
			openvdb::Vec3f V1 = openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V0 * dtinvx);
			openvdb::Vec3f V2 = openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V1 * dtinvx);
			openvdb::Vec3f V3 = openvdb::tools::StaggeredBoxSampler::sample(vaxr, ipos + V2 * dtinvx);
			ipos += dtinvx * (V0 + 2.0f * (V1 + V2) + V3) * (1.0f / 6.0f);
		}

		float dx;
		float invdx;
		float dtinvx;
		float dt;
		openvdb::tree::ValueAccessor<const openvdb::Vec3fTree> vaxr;
		const openvdb::Vec3fGrid& m_velocity;
	};
	struct point_to_counter_reducer {

		point_to_counter_reducer(
			const float in_dt,
			const float in_dx,
			const openvdb::Vec3fGrid& in_velocity,
			const openvdb::Vec3fGrid& in_old_velocity,
			float pic_component,
			const std::vector<openvdb::points::PointDataTree::LeafNodeType*>& in_particles,
			int RK_order) : dt(in_dt),
			m_rk_order(RK_order), m_dx(in_dx), m_invdx(1.0f / in_dx),
			m_velocity(in_velocity),
			m_old_velocity(in_old_velocity),
			m_pic_component(pic_component),
			m_particles(in_particles)
		{
			m_integrator = std::make_shared<custom_integrator>(in_velocity, in_dx, in_dt);
			m_counter_grid = openvdb::points::PointDataGrid::create();
		}

		point_to_counter_reducer(const point_to_counter_reducer& other, tbb::split) :
			dt(other.dt), m_rk_order(other.m_rk_order), m_dx(other.m_dx), m_invdx(other.m_invdx),
			m_velocity(other.m_velocity),
			m_old_velocity(other.m_old_velocity),
			m_pic_component(other.m_pic_component),
			m_particles(other.m_particles) {
			m_integrator = std::make_shared<custom_integrator>(m_velocity, m_dx, dt);
			m_counter_grid = openvdb::points::PointDataGrid::create();
		}


		//loop over ranges of flattened particle leaves
		void operator()(const tbb::blocked_range<openvdb::Index>& r) {
			using  namespace openvdb::tools::local_util;
			auto counter_axr{ m_counter_grid->getAccessor() };
			auto vaxr{ m_velocity.getConstUnsafeAccessor() };
			auto old_vaxr{ m_old_velocity.getConstUnsafeAccessor() };

			std::function<void(openvdb::Vec3f& ipos, const openvdb::Vec3f& V0)> movefunc;

			switch (m_rk_order) {
			case 1: movefunc = [&](openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) {
				m_integrator->integrate1(ipos, V0); }; break;
			case 2: movefunc = [&](openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) {
				m_integrator->integrate2(ipos, V0); }; break;
			case 3: movefunc = [&](openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) {
				m_integrator->integrate3(ipos, V0); }; break;
			case 4: movefunc = [&](openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) {
				m_integrator->integrate4(ipos, V0); }; break;
			default: movefunc = [&](openvdb::Vec3f& ipos, const openvdb::Vec3f& V0) {
				m_integrator->integrate1(ipos, V0); };
			}

			//leaf iter
			for (auto liter = r.begin(); liter != r.end(); ++liter) {
				auto& leaf = *m_particles[liter];

				//attributes
				// Attribute reader
				// Extract the position attribute from the leaf by name (P is position).
				openvdb::points::AttributeArray& positionArray =
					leaf.attributeArray("P");
				// Extract the velocity attribute from the leaf by name (v is velocity).
				openvdb::points::AttributeArray& velocityArray =
					leaf.attributeArray("v");

				// Create read handles for position and velocity
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);

				//point index space source position
				openvdb::Vec3f pIspos;
				//advection velocity
				openvdb::Vec3f adv_vel;
				//old velocity
				openvdb::Vec3f old_vel;
				//particle velocity
				openvdb::Vec3f particle_vel;

				//point index space target position after move
				openvdb::Vec3f pItpos;

				//point world pos
				openvdb::Vec3f pWtpos;

				//the target voxel coordinate
				openvdb::Coord ptCoord;
				//loop over all particles
				openvdb::points::PointDataTree::LeafNodeType* writing_leaf;
				//std::vector<
				//	std::pair<openvdb::Vec3f, openvdb::Vec3f>>* writing_vector;

				std::vector<
					std::tuple<uint16_t, openvdb::Index32, openvdb::Index32>>*writing_offset_index_leafpos;

				//old leaf beging and end
				openvdb::Coord olbegin{ openvdb::Coord::max() };
				openvdb::Coord olend{ openvdb::Coord::min() };
				float flip_component = (1.0f - m_pic_component);
				for (auto piter = leaf.beginIndexOn(); piter; ++piter) {
					pIspos = piter.getCoord().asVec3s() + positionHandle.get(*piter);
					particle_vel = velocityHandle.get(*piter);

					/*adv_vel = openvdb::tools::StaggeredBoxSampler::sample(vaxr, pIspos);
					old_vel = openvdb::tools::StaggeredBoxSampler::sample(old_vaxr, pIspos);*/
					adv_vel = StaggeredBoxSampler::sample(vaxr, pIspos);
					old_vel = StaggeredBoxSampler::sample(old_vaxr, pIspos);
					//update the velocity of the particle
					/*particle_vel = (m_pic_component)*adv_vel + (1.0f - m_pic_component) * (adv_vel - old_vel + particle_vel);*/
					particle_vel = adv_vel + flip_component * (- old_vel + particle_vel);
					//pItpos = pIspos + dt * adv_vel * m_invdx;
					pItpos = pIspos;
					movefunc(pItpos, adv_vel);

					ptCoord = openvdb::Coord{ floorVec3(pItpos + openvdb::Vec3f{ 0.5f }) };

					//directly change the original attribute to the target voxel position
					//later it will be transfered to the new position
					positionHandle.set(*piter, pItpos - ptCoord);
					velocityHandle.set(*piter, particle_vel);
					//check if we are writing to the previous leaf?

					if ((ptCoord[0] >= olbegin[0]) &&
						(ptCoord[1] >= olbegin[1]) &&
						(ptCoord[2] >= olbegin[2]) &&
						(ptCoord[0] < olend[0]) &&
						(ptCoord[1] < olend[1]) &&
						(ptCoord[2] < olend[2])) {
						//increment the counter
						uint16_t toffset = writing_leaf->coordToOffset(ptCoord);
						writing_leaf->setOffsetOn(toffset, writing_leaf->getValue(toffset) + 1);

						//append the velocity and index space position
						//writing_vector->push_back(std::make_pair(pItpos, velocityHandle.get(*piter)));

						writing_offset_index_leafpos->push_back(std::make_tuple(toffset, *piter, liter));
					}//end if writing to same leaf
					else {
						//try to probe it to check if we have it in this tree already
						if (writing_leaf = counter_axr.probeLeaf(ptCoord)) {}
						else {
							writing_leaf = counter_axr.touchLeaf(ptCoord);
							toffset_oindex_oleafpos_hashmap[writing_leaf->origin()] = std::make_unique<std::vector<
								std::tuple<uint16_t, openvdb::Index32, openvdb::Index32>>>();
						}

						//increment the counter
						uint16_t toffset = writing_leaf->coordToOffset(ptCoord);
						writing_leaf->setOffsetOn(toffset, writing_leaf->getValue(toffset) + 1);

						writing_offset_index_leafpos = toffset_oindex_oleafpos_hashmap[writing_leaf->origin()].get();
						writing_offset_index_leafpos->push_back(std::make_tuple(toffset, *piter, liter));
						//set the bounding box
						olbegin = writing_leaf->origin();
						olend = olbegin + openvdb::Coord{ 8 };
					}//end else writing to the same leaf
				}//end loop over all particles
			}//end for range leaves
		}//end operator

		void join(point_to_counter_reducer& other) {
			auto& grid = *other.m_counter_grid;
			//merge the counter grid
			for (auto leaf = grid.tree().beginLeaf(); leaf; ++leaf) {
				auto* newLeaf = m_counter_grid->tree().probeLeaf(leaf->origin());
				if (!newLeaf) {
					// if the leaf doesn't yet exist in the new tree, steal it
					auto& tree = const_cast<openvdb::points::PointDataGrid&>(grid).tree();
					m_counter_grid->tree().addLeaf(tree.template stealNode<openvdb::points::PointDataTree::LeafNodeType>(leaf->origin(),
						0, false));
				}
				else {
					// otherwise increment existing values
					for (auto iter = leaf->beginValueOn(); iter; ++iter) {
						//auto original_counter = newLeaf->getValue(iter.offset());
						//newLeaf->setOffsetOn(iter.offset(), original_counter + leaf->getValue(iter.offset()));
						//*(newLeaf->buffer().data()+iter.offset()) = *(newLeaf->buffer().data() + iter.offset()) + iter.getValue();
						newLeaf->setOffsetOn(iter.offset(), *(newLeaf->buffer().data() + iter.offset()) + iter.getValue());
					}
				}
			}


			for (auto tuplevec = other.toffset_oindex_oleafpos_hashmap.begin();
				tuplevec != other.toffset_oindex_oleafpos_hashmap.end(); ++tuplevec) {
				auto itr_in_this = toffset_oindex_oleafpos_hashmap.find(tuplevec->first);
				if (itr_in_this != toffset_oindex_oleafpos_hashmap.end()) {
					auto original_size = itr_in_this->second->size();
					itr_in_this->second->resize(original_size + tuplevec->second->size());
					std::copy(tuplevec->second->begin(), tuplevec->second->end(), itr_in_this->second->begin() + original_size);
				}
				else {
					toffset_oindex_oleafpos_hashmap[tuplevec->first] = std::move(tuplevec->second);
				}
			}
		}


		//velocity integrator
		std::shared_ptr<custom_integrator> m_integrator;


		//time step
		const float dt;

		const int m_rk_order;
		//index to world transform
		//for the particles as well as the velocity
		const float m_dx;
		const float m_invdx;

		//the velocity field used to advect the particles
		const openvdb::Vec3fGrid& m_velocity;
		const openvdb::Vec3fGrid& m_old_velocity;
		float m_pic_component;

		//the source particles
		const std::vector<openvdb::points::PointDataTree::LeafNodeType*>& m_particles;

		//this is the reduction result
		openvdb::points::PointDataGrid::Ptr m_counter_grid;

		//hashmap storing target voxel offset, original attribute index, original leaf position
		std::unordered_map <
			openvdb::Coord, std::unique_ptr<
			std::vector<
			std::tuple<uint16_t, openvdb::Index32, openvdb::Index32>>>> toffset_oindex_oleafpos_hashmap;
	};
}
void FLIP_vdb::custom_move_points_and_set_flip_vel(
	openvdb::points::PointDataGrid& in_out_points,
	const openvdb::Vec3fGrid& in_velocity_field,
	const openvdb::Vec3fGrid& in_old_velocity,
	float PIC_component, float dt, float dx, int RK_order)
{

	std::vector<openvdb::points::PointDataTree::LeafNodeType*> particle_leaves;
	in_out_points.tree().getNodes(particle_leaves);


	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").start();
	auto reducer = std::make_unique<point_to_counter_reducer>(
		dt, dx, in_velocity_field, in_old_velocity, PIC_component, particle_leaves, RK_order);
	tbb::parallel_reduce(tbb::blocked_range<openvdb::Index>(0, particle_leaves.size(), 10), *reducer);
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").stop();
	//compose the result
	
	auto newTree_leafman{ openvdb::tree::LeafManager<openvdb::points::PointDataTree>(reducer->m_counter_grid->tree()) };

	auto p_desc = openvdb::points::AttributeSet::Descriptor::create(position_attribute::attributeType());
	auto pv_desc = p_desc->duplicateAppend("v", velocity_attribute::attributeType());
	auto set_new_attribute_list = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		using  namespace openvdb::tools::local_util;

		std::vector<int> voxel_particle_count; voxel_particle_count.resize(leaf.size());
		std::vector<openvdb::PointDataIndex32> index_ends; index_ends.assign(leaf.size(), 0);
		voxel_particle_count[0] = leaf.getValue(0);
		index_ends[0] = voxel_particle_count[0];
		for (auto offset = 1; offset < leaf.size(); ++offset) {
			voxel_particle_count[offset] = leaf.getValue(offset);
			index_ends[offset] = index_ends[offset - 1] + voxel_particle_count[offset];
		}

		//according to the index space leaf position, assign the particles to
		//the final attribute list
		//printf("leafcounter %d, attrcounter:%d\n", *index_ends.rbegin(), reducer.toffset_oindex_oleafpos_hashmap[leaf.origin()]->size());
		//create the attribute set
		
		//auto local_pv_descriptor = pv_descriptor();
		leaf.initializeAttributes(p_desc, *index_ends.rbegin());
		leaf.appendAttribute(leaf.attributeSet().descriptor(), pv_desc, 1);

		//attribute writer
		leaf.setOffsets(index_ends);

		//set the positions and velocities
		//get the new attribute arrays
		openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
		openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

		// Create read handles for position and velocity
		openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
		openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

		openvdb::Vec3i pcoord;
		int writing_offset;
		int writer_index;

		//move from the original leaves
		for (size_t i = 0; i < reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()]->size(); i++) {
			auto& tooiol_vec = *reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()];
			writing_offset = std::get<0>(tooiol_vec[i]);
			writer_index = index_ends[writing_offset] - voxel_particle_count[writing_offset];
			voxel_particle_count[writing_offset]--;

			posarray.set(writer_index, particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("P"), std::get<1>(tooiol_vec[i]));
			varray.set(writer_index, particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("v"), std::get<1>(tooiol_vec[i]));
		}
	};
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose").start();
	//printf("compose_start_for_each\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/compose").start();
	newTree_leafman.foreach(set_new_attribute_list);
	//printf("compose_end_for_each\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/compose").stop();
	reducer->m_counter_grid->setName("new_counter_grid");
	auto voxel_center_transform = openvdb::math::Transform::createLinearTransform(dx);
	reducer->m_counter_grid->setTransform(voxel_center_transform);
	//openvdb::io::File("new_advect.vdb").write({reducer.m_counter_grid});
	//printf("compose_start_replace_tree\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/replace").start();
	in_out_points.setTree(reducer->m_counter_grid->treePtr());
	//printf("compose_end_replace_tree\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/replace").stop();
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose").stop();
}
void FLIP_vdb::custom_move_points_and_set_flip_vel(
	openvdb::points::PointDataGrid& in_out_points,
	const openvdb::Vec3fGrid& in_velocity_field,
	const openvdb::Vec3fGrid& in_old_velocity,
	float PIC_component, float dt,int RK_order)
{
	std::vector<openvdb::points::PointDataTree::LeafNodeType*> particle_leaves;
	in_out_points.tree().getNodes(particle_leaves);


	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").start();
	auto reducer = std::make_unique<point_to_counter_reducer>(
		dt, m_dx, in_velocity_field, in_old_velocity, PIC_component, particle_leaves, RK_order);
	tbb::parallel_reduce(tbb::blocked_range<openvdb::Index>(0, particle_leaves.size(), 10), *reducer);
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/reduce").stop();
	//compose the result

	auto newTree_leafman{ openvdb::tree::LeafManager<openvdb::points::PointDataTree>(reducer->m_counter_grid->tree()) };

	auto set_new_attribute_list = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		using  namespace openvdb::tools::local_util;

		std::vector<int> voxel_particle_count; voxel_particle_count.resize(leaf.size());
		std::vector<openvdb::PointDataIndex32> index_ends; index_ends.assign(leaf.size(), 0);
		voxel_particle_count[0] = leaf.getValue(0);
		index_ends[0] = voxel_particle_count[0];
		for (auto offset = 1; offset < leaf.size(); ++offset) {
			voxel_particle_count[offset] = leaf.getValue(offset);
			index_ends[offset] = index_ends[offset - 1] + voxel_particle_count[offset];
		}

		//according to the index space leaf position, assign the particles to
		//the final attribute list
		//printf("leafcounter %d, attrcounter:%d\n", *index_ends.rbegin(), reducer.toffset_oindex_oleafpos_hashmap[leaf.origin()]->size());
		//create the attribute set

		auto local_pv_descriptor = pv_descriptor();
		leaf.initializeAttributes(pos_descriptor(), *index_ends.rbegin());
		leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);

		//attribute writer
		leaf.setOffsets(index_ends);

		//set the positions and velocities
		//get the new attribute arrays
		openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
		openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

		// Create read handles for position and velocity
		openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
		openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

		openvdb::Vec3i pcoord;
		int writing_offset;
		int writer_index;

		//move from the original leaves
		for (size_t i = 0; i < reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()]->size(); i++) {
			auto& tooiol_vec = *reducer->toffset_oindex_oleafpos_hashmap[leaf.origin()];
			writing_offset = std::get<0>(tooiol_vec[i]);
			writer_index = index_ends[writing_offset] - voxel_particle_count[writing_offset];
			voxel_particle_count[writing_offset]--;

			posarray.set(writer_index, particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("P"), std::get<1>(tooiol_vec[i]));
			varray.set(writer_index, particle_leaves[std::get<2>(tooiol_vec[i])]->attributeArray("v"), std::get<1>(tooiol_vec[i]));
		}
	};
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose").start();
	//printf("compose_start_for_each\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/compose").start();
	newTree_leafman.foreach(set_new_attribute_list);
	//printf("compose_end_for_each\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/compose").stop();
	reducer->m_counter_grid->setName("new_counter_grid");
	reducer->m_counter_grid->setTransform(m_voxel_center_transform);
	//openvdb::io::File("new_advect.vdb").write({reducer.m_counter_grid});
	//printf("compose_start_replace_tree\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/replace").start();
	in_out_points.setTree(reducer->m_counter_grid->treePtr());
	//printf("compose_end_replace_tree\n");
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose/replace").stop();
	//CSim::TimerMan::timer("Sim.step/vdbflip/advection/compose").stop();
}

void FLIP_vdb::extrapolate_velocity(int layer)
{
	//BPS3D_volume_vel_cache::extrapolate(layer, m_velocity);
}

void FLIP_vdb::particle_to_grid_reduce_style()
{

	//for the given particles grid
	//turns it into velocity grid and liquid phi grid

	//initialize the grids so that it's possible to steal some nodes

	openvdb::tree::LeafManager<openvdb::points::PointDataTree> particle_manager(m_particles->tree());

	//maybe it's not necessary to allocate these
	//particle_manager.foreach(touch_velocity_phi_leaves, false);

	auto p2gop = particle_to_grid_reducer(m_velocity, m_velocity_weights, m_liquid_sdf, m_particle_radius);


	particle_manager.reduce(p2gop, true, 20);


	openvdb::Vec3fGrid::Ptr original_unweighted_velocity = m_velocity->deepCopy();

	openvdb::tree::LeafManager<openvdb::Vec3fGrid::TreeType> velocity_grid_manager(m_velocity->tree());

	auto velocity_normalizer = deduce_missing_velocity_and_normalize(m_velocity_weights, original_unweighted_velocity);

	velocity_grid_manager.foreach(velocity_normalizer, true, 1);

	//store the velocity just after the transfer
	m_velocity_after_p2g = m_velocity->deepCopy();
	m_velocity_after_p2g->setName("Velocity_After_P2G");
	//m_velocity_snapshot = m_velocity->deepCopy();
	//m_velocity_snapshot->setName("Velocity_After_Projection");
	//extrapolate_velocity();
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
			//set the loop order and offset of the u v w phi sampling point
			//with respect to the center voxel center

			//u:  (-0.5, 0, 0)
			//v:  (0, -0.5, 0)
			//w:  (0, 0, -0.5)
			//phi:(0, 0, 0)
			//note we use set so the 
			//R0 R1 R2 R3 correspond to the 0,1,2,3 argument
			//when later extracting using _mm_storer_ps, we get float[0] = arg0 correctly
			//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_store.htm
			//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_set.htm

			__m128 xpack = _mm_set_ps(-0.5f, 0.f, 0.f, 0.f);
			__m128 ypack = _mm_set_ps(0.f, -0.5f, 0.f, 0.f);
			__m128 zpack = _mm_set_ps(0.f, 0.f, -0.5f, 0.f);
			for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
				int ijk = ivoxel;
				int basex = ijk / 9; ijk -= 9 * basex;
				int basey = ijk / 3; ijk -= 3 * basey;
				int basez = ijk;
				//becomes -1 -> 1

				basex -= 1; basey -= 1; basez -= 1;
				//broadcast four float as the base
				__m128 basex4 = _mm_set_ps1(float(basex));
				__m128 basey4 = _mm_set_ps1(float(basey));
				__m128 basez4 = _mm_set_ps1(float(basez));

				loop_order.at(ivoxel) = openvdb::Coord{ basex, basey, basez };

				x_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basex4, xpack);
				y_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basey4, ypack);
				z_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basez4, zpack);
			}
		}


		//packed structure storing a particle leaf
		//its attribute handle
		struct particle_leaf {
			particle_leaf() {
				m_leafptr = nullptr;
				m_p_handle_ptr.reset();
				m_v_handle_ptr.reset();
			}

			//iterator over points in a voxel
			struct particle_iter {
				particle_iter() :m_parent(nullptr) {
					m_item = 0;
					m_indexend = 0;
					m_indexbegin = 0;
				}

				particle_iter(const particle_leaf* in_parent, openvdb::Index in_offset) {
					set(in_parent, in_offset);
				}

				particle_iter(const particle_iter& other) = default;

				void set(const particle_leaf* in_parent, openvdb::Index in_offset) {
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
					}
					else {
						m_indexbegin = in_parent->m_leafptr->getValue(in_offset - 1);
					}
					m_item = m_indexbegin;
				}

				void operator =(const particle_iter& other) {
					m_item = other.m_item;
					m_parent = other.m_parent;
					m_indexbegin = other.m_indexbegin;
					m_indexend = other.m_indexend;
				}

				operator bool() const {
					return m_item < m_indexend;
				}

				particle_iter& operator ++() {
					m_item++;
					return *this;
				}

				openvdb::Index operator*()  const {
					return m_item;
				}

				openvdb::Index m_item;
				openvdb::Index m_indexbegin, m_indexend;
				const particle_leaf* m_parent;
			};


			particle_leaf(const openvdb::points::PointDataTree::LeafNodeType* in_leaf_ptr) : m_leafptr(in_leaf_ptr) {
				if (in_leaf_ptr) {
					// Attribute reader
					// Extract the position attribute from the leaf by name (P is position).
					const openvdb::points::AttributeArray& positionArray =
						in_leaf_ptr->attributeArray("P");
					// Extract the velocity attribute from the leaf by name (v is velocity).
					const openvdb::points::AttributeArray& velocityArray =
						in_leaf_ptr->attributeArray("v");

					//// Create read handles for position and velocity
					//openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
					//openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);
					m_p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
					m_v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);
				}
			}

			//default?
			particle_leaf(const particle_leaf& other) {
				m_leafptr = other.m_leafptr;
				m_p_handle_ptr = other.m_p_handle_ptr;
				m_v_handle_ptr = other.m_v_handle_ptr;
			}

			particle_iter get_particle_iter(openvdb::Index offset) const {
				return particle_iter(this, offset);
			}

			particle_iter get_particle_iter(const openvdb::Coord& lgxyz) const {
				return particle_iter(this, m_leafptr->coordToOffset(lgxyz));
			}

			operator bool() const {
				return m_leafptr!=nullptr;
			}

			openvdb::Vec3f get_p(openvdb::Index pos) const {
				return m_p_handle_ptr->get(pos);
			}

			openvdb::Vec3f get_v(openvdb::Index pos) const {
				return m_v_handle_ptr->get(pos);
			}

			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>::Ptr m_p_handle_ptr;
			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>::Ptr m_v_handle_ptr;
			const openvdb::points::PointDataTree::LeafNodeType* m_leafptr;
		};

		void fill_particle_leafs(std::array<particle_leaf, 27>& in_out_leaves, const openvdb::Coord& center_origin, const openvdb::points::PointDataTree& in_particle_tree) const {
			//for all its neighbor nodes
			int counter = 0;
			for (int ii = -8; ii <= 8; ii += 8) {
				for (int jj = -8; jj <= 8; jj += 8) {
					for (int kk = -8; kk <= 8; kk += 8) {
						auto leafptr = in_particle_tree.probeConstLeaf(center_origin.offsetBy(ii, jj, kk));
						in_out_leaves[counter++] = particle_leaf(leafptr);
					}
				}
			}
		}

		//given the 27 particle leaves, 
		//iterate all particles that could possibly contribute to the center leaf
		//each iterator can return the position, velocity, local coordinate relative
		//to the center leaf
		struct all_particle_iterator {
			all_particle_iterator(const std::array<particle_leaf, 27>& in_leaves) :
				m_leaves(in_leaves) {
				m_leafpos = 0;
				at_voxel = 0;
				//initially on the corner
				m_leafxyz = { -8,-8,-8 };
				//it runs from [{-1,-1,-1}, {9,9,9} )
				//the size is 1+8+1 = 10
				//so there are at most 1000 voxels to iterate
				m_center_xyz = openvdb::Coord{ -1,-1,-1 };
				auto offset = 511u;
				//printf("setting voxel iter at offset%d\n",offset);
				m_voxel_particle_iter.set(&m_leaves[m_leafpos], offset);
				m_at_interior = false;
				//printf("voxel_iter set\n");
				if (!m_voxel_particle_iter) {
					move_to_next_on_voxel();
				}
			}

			void move_to_next_on_voxel() {
				do {
					//ran out of attribute in this voxel
					//find the next on voxel that will contribute
					//to the center voxel
					//allowed range: [-1,8] in x,y,z
					m_center_xyz[2]++;
					
					//only when it freshly turns 0 and 8 it cross the leaf border
					if (m_center_xyz[2] == 0) { m_leafpos++; }
					else if (m_center_xyz[2] == 8) { m_leafpos++; }
					else
					if (m_center_xyz[2] == 9) {
						m_center_xyz[2] = -1; m_leafpos -= 2;

						m_center_xyz[1]++;
						if (m_center_xyz[1] == 0) { m_leafpos += 3; }
						else if (m_center_xyz[1] == 8) { m_leafpos += 3; }
						else
						if (m_center_xyz[1] == 9) {
							m_center_xyz[1] = -1; m_leafpos -= 6;

							m_center_xyz[0]++;
							if (m_center_xyz[0] == 0) { m_leafpos += 9; }
							else if (m_center_xyz[0] == 8) { m_leafpos += 9; }
							else
							if (m_center_xyz[0] == 9) {
								m_leafpos += 9;
								if (m_leafpos != 27) {
									printf("particle iterator increment error! actual leafpos:%d\n",m_leafpos);
									exit(-1);
								}
							};
						}//end if y overflow
					}//end if z overflow

					//check leaf bount
					if (m_leafpos >= 27) return;

					//this leaf is empty
					if (m_leaves[m_leafpos].m_leafptr==nullptr) { continue; };

					int offset = m_leaves[m_leafpos].m_leafptr->coordToOffset(m_center_xyz);
					if (m_leaves[m_leafpos].m_leafptr->isValueOn(offset)) {
						m_voxel_particle_iter.set(&m_leaves[m_leafpos], offset);
						auto itercoord = m_center_xyz;
						m_at_interior = (itercoord[0] >= 1 && itercoord[0] <= 6 && itercoord[1] >= 1 && itercoord[1] <= 6 && itercoord[2] >= 1 && itercoord[2] <= 6);
						return;
					}
				} while (true);
			}//end move to next on voxel

			bool is_interior_voxel() const {
				return m_at_interior;
			}

			openvdb::Coord getCoord() const {
				return m_center_xyz;
			}

			openvdb::Vec3f getP() const {
				return m_leaves[m_leafpos].get_p(*m_voxel_particle_iter);
			}

			openvdb::Vec3f getv() const {
				return m_leaves[m_leafpos].get_v(*m_voxel_particle_iter);
			}

			all_particle_iterator& operator++() {
				++m_voxel_particle_iter;
				//printf("advancing index %d at voxel %d\n", *m_voxel_particle_iter, at_voxel);
				if (!m_voxel_particle_iter) {
					move_to_next_on_voxel();
				}//end if ran out of particles in this voxel
				return *this;
			}

			operator bool() const {
				return m_voxel_particle_iter;
			}

			//local coordinates relative to the center leaf
			openvdb::Coord m_center_xyz;
			bool m_at_interior;
			openvdb::Coord m_g_xyz;
			int at_voxel;
			//attribute index in corresponding 
			particle_leaf::particle_iter m_voxel_particle_iter;
			//the offset in each voxel
			uint32_t m_voxel_offset;
			//0-26, the position in the particle leaf
			uint32_t m_leafpos;
			openvdb::Coord m_leafxyz;
			std::array<int, 27> visit_order;
			const std::array<particle_leaf, 27>& m_leaves;
		};

		//to be use by the velocity grid manager
		//collect the contribution of particles in current leaf
		//and its neighbor leafs
		void operator() (openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) const {
			auto new_vel_leaf = m_velocity->tree().probeLeaf(leaf.origin());
			auto new_vel_weights_leaf = m_velocity_weight->tree().probeLeaf(leaf.origin());
			auto new_sdf_leaf = m_liquid_sdf->tree().probeLeaf(leaf.origin());

			const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
			const __m128 float1x4 = _mm_set_ps1(float(1));
			const __m128 float0x4 = _mm_set_ps1(float(0));

			//scatter style transfer
			//loop over all particles
			float tophix, tophiy, tophiz;
			float dx = m_liquid_sdf->voxelSize()[0];

			std::array<particle_leaf, 27> particle_leaves;
			fill_particle_leafs(particle_leaves, leaf.origin(), m_particles->tree());

			auto iter{ all_particle_iterator(particle_leaves) };
			//pre-allocated space for computation
			__m128 x_dist_particle_to_sample;
			__m128 y_dist_particle_to_sample;
			__m128 z_dist_particle_to_sample;
			__m128 comp_result;
			__m128 comp_target;
			//the distance to the other cell center will always be greater
			//than zero, hence the last predicate is always true
			comp_target = _mm_set_ps(1.f, 1.f, 1.f, -1.f);
			float dist_to_phi_sample;
			//particle iterator
			for (; iter; ++iter) {
				
				auto voxelpos = iter.getP();
				auto pvel = iter.getv();
				//broadcast the variables
				__m128 particle_x = _mm_set_ps1(voxelpos[0]);
				__m128 particle_y = _mm_set_ps1(voxelpos[1]);
				__m128 particle_z = _mm_set_ps1(voxelpos[2]);

				//calculate the distance to the 27 neib uvw phi samples
				for (int ivoxel = 0; ivoxel < 27; ivoxel++) {

					//is it writing to this leaf?
					openvdb::Coord head = iter.getCoord() + loop_order.at(ivoxel);
					
					if (head[0] < 0 || head[1] < 0 || head[2] < 0 || head[0]>7 || head[1]>7 || head[2]>7) {
						continue;
					}

					auto write_offset = leaf.coordToOffset(head);

					//calculate the distance
					//arg(A,B): ret A-B
					// the absolute value trick: abs_mask: 01111111..32bit..1111 x 4
					// _mm_and_ps(abs_mask(), v);
					x_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(x_offset_to_center_voxel_center.at(ivoxel), particle_x));
					y_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(y_offset_to_center_voxel_center.at(ivoxel), particle_y));
					z_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(z_offset_to_center_voxel_center.at(ivoxel), particle_z));

					//the distance to the phi variable
					_mm_store_ss(&tophix, x_dist_particle_to_sample);
					_mm_store_ss(&tophiy, y_dist_particle_to_sample);
					_mm_store_ss(&tophiz, z_dist_particle_to_sample);
					dist_to_phi_sample = dx * std::sqrt(tophix * tophix + tophiy * tophiy + tophiz * tophiz);

					//phi
					//compare 
					float original_sdf = new_sdf_leaf->getValue(write_offset);
					new_sdf_leaf->setValueOn(write_offset, std::min(original_sdf, dist_to_phi_sample - m_particle_radius));

					do {
						//if for [x,y,z]distance_to_sample, the first 3 values are all greater than 1
						//then the weight is 0, so skip it
						uint32_t test = _mm_movemask_ps(_mm_cmpgt_ps(x_dist_particle_to_sample, comp_target));
						if (test == 0b00001111) {
							break;
						}
						test = _mm_movemask_ps(_mm_cmpgt_ps(y_dist_particle_to_sample, comp_target));
						if (test == 0b00001111) {
							break;
						}
						test = _mm_movemask_ps(_mm_cmpgt_ps(z_dist_particle_to_sample, comp_target));
						if (test == 0b00001111) {
							break;
						}
						
						//the uvw weights trilinear
						//transfer the distance to weight at the 27 voxels
						//(1-dist)
						//the far points now becomes negative
						x_dist_particle_to_sample = _mm_sub_ps(float1x4, x_dist_particle_to_sample);
						y_dist_particle_to_sample = _mm_sub_ps(float1x4, y_dist_particle_to_sample);
						z_dist_particle_to_sample = _mm_sub_ps(float1x4, z_dist_particle_to_sample);

						//turn everything positive or zero
						//now the dist_to_sample is actually the component-wise weight on the voxel
						//time to multiply them together
						x_dist_particle_to_sample = _mm_max_ps(float0x4, x_dist_particle_to_sample);
						y_dist_particle_to_sample = _mm_max_ps(float0x4, y_dist_particle_to_sample);
						z_dist_particle_to_sample = _mm_max_ps(float0x4, z_dist_particle_to_sample);

						//turn them into weights reduce to x
						x_dist_particle_to_sample = _mm_mul_ps(x_dist_particle_to_sample, y_dist_particle_to_sample);
						x_dist_particle_to_sample = _mm_mul_ps(x_dist_particle_to_sample, z_dist_particle_to_sample);
						//}//end for 27 voxel

						////write to the grid
						//for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
						alignas(16) float packed_weight[4];

						_mm_storer_ps(packed_weight, x_dist_particle_to_sample);

						openvdb::Vec3f weights{ packed_weight[0],packed_weight[1],packed_weight[2] };
						//write weighted velocity

						openvdb::Vec3f weighted_vel = pvel * weights;
						auto original_weighted_vel = new_vel_leaf->getValue(write_offset);
						new_vel_leaf->setValueOn(write_offset, weighted_vel + original_weighted_vel);
						//new_vel_leaf->modifyValue(write_offset, [&weighted_vel](openvdb::Vec3f& in_out) {in_out += weighted_vel; });
						//write weights
						auto original_weights = new_vel_weights_leaf->getValue(write_offset);
						new_vel_weights_leaf->setValueOn(write_offset, weights + original_weights);
						//new_vel_weights_leaf->modifyValue(write_offset, [&weights](openvdb::Vec3f& in_out) {in_out += weights; });

					} while (false);//test if the velocity weight is zero
				}//end for 27 voxels
			}//end for all particles influencing this leaf
		}//end operator

		//particle radius 
		float m_particle_radius;

		//neighbor leaf pointer cache

		//constant distances to the center voxel
		std::array<__m128, 27> x_offset_to_center_voxel_center;
		std::array<__m128, 27> y_offset_to_center_voxel_center;
		std::array<__m128, 27> z_offset_to_center_voxel_center;
		std::array<openvdb::Coord, 27> loop_order;

		openvdb::FloatGrid::Ptr m_liquid_sdf;
		openvdb::Vec3fGrid::Ptr m_velocity, m_velocity_weight;
		openvdb::points::PointDataGrid::Ptr m_particles;
	};
}
void FLIP_vdb::particle_to_grid_collect_style(
	openvdb::points::PointDataGrid::Ptr particles,
	openvdb::Vec3fGrid::Ptr velocity,
	openvdb::Vec3fGrid::Ptr velocity_after_p2g,
	openvdb::Vec3fGrid::Ptr velocity_weights,
	openvdb::FloatGrid::Ptr liquid_sdf,
	openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
	float dx)
{
	float particle_radius = 0.5f * std::sqrt(3.0f) * dx * 1.01;
	velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
		particles->tree(), openvdb::Vec3f{ 0 }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(velocity->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	velocity_weights = velocity->deepCopy();
	auto voxel_center_transform = openvdb::math::Transform::createLinearTransform(dx);
	liquid_sdf->setTransform(voxel_center_transform);
	liquid_sdf->setTree(std::make_shared<openvdb::FloatTree>(
		velocity->tree(), liquid_sdf->background(), openvdb::TopologyCopy()));

	auto collector_op{ p2g_collector(liquid_sdf,
			velocity,
			velocity_weights,
			particles,
			particle_radius) };
	
	auto vleafman = 
	openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity->tree());

	vleafman.foreach(collector_op,true);
	
	openvdb::Vec3fGrid::Ptr original_unweighted_velocity = velocity->deepCopy();

	openvdb::tree::LeafManager<openvdb::Vec3fGrid::TreeType> velocity_grid_manager(velocity->tree());

	auto velocity_normalizer = deduce_missing_velocity_and_normalize(velocity_weights, original_unweighted_velocity);
	
	velocity_grid_manager.foreach(velocity_normalizer, true, 1);
	
	//store the velocity just after the transfer
	velocity_after_p2g = velocity->deepCopy();
	velocity_after_p2g->setName("Velocity_After_P2G");
	pushed_out_liquid_sdf = liquid_sdf;	
}
void FLIP_vdb::particle_to_grid_collect_style()
{

	printf("collect start\n");
	//allocate the tree for transfered velocity and liquid phi

	m_velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
		m_particles->tree(), openvdb::Vec3f{ 0 }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(m_velocity->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	//velocity weights
	
	m_velocity_weights = m_velocity->deepCopy();

	m_liquid_sdf->setTransform(m_voxel_center_transform);
	m_liquid_sdf->setTree(std::make_shared<openvdb::FloatTree>(
		m_velocity->tree(), m_liquid_sdf->background(), openvdb::TopologyCopy()));


	auto collector_op{ p2g_collector(m_liquid_sdf,
			m_velocity,
			m_velocity_weights,
			m_particles,
			m_particle_radius) };

	auto vleafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_velocity->tree());


	vleafman.foreach(collector_op,true);
	
	openvdb::Vec3fGrid::Ptr original_unweighted_velocity = m_velocity->deepCopy();

	openvdb::tree::LeafManager<openvdb::Vec3fGrid::TreeType> velocity_grid_manager(m_velocity->tree());

	auto velocity_normalizer = deduce_missing_velocity_and_normalize(m_velocity_weights, original_unweighted_velocity);
	
	velocity_grid_manager.foreach(velocity_normalizer, true, 1);
	
	//store the velocity just after the transfer
	m_velocity_after_p2g = m_velocity->deepCopy();
	m_velocity_after_p2g->setName("Velocity_After_P2G");
	m_pushed_out_liquid_sdf = m_liquid_sdf;

}

namespace {

	using namespace openvdb;
	bool boxes_overlap(Coord begin0, Coord end0, Coord begin1, Coord end1) {
		if (end0[0] <= begin1[0]) return false; // a is left of b
		if (begin0[0] >= end1[0]) return false; // a is right of b
		if (end0[1] <= begin1[1]) return false; // a is above b
		if (begin0[1] >= end1[1]) return false; // a is below b
		if (end0[2] <= begin1[2]) return false; // a is in front of b
		if (begin0[2] >= end1[2]) return false; // a is behind b
		return true; // boxes overlap
	}

	//for each of the leaf nodes that overlaps the fill zone
	//fill all voxels that are within the fill zone and bottoms
	//are below the sea level
	//the newly generated particle positions and offset in the attribute array
	//will be recorded in std vectors associated with each leaf nodes
	//it also removes all particles that are
	// 1:outside the bounding box
	// 2:in the fill layer but are above the sealevel

	struct particle_fill_kill_op {
		using pos_offset_t = std::pair<openvdb::Vec3f, openvdb::Index>;
		using node_t = openvdb::points::PointDataGrid::TreeType::LeafNodeType;

		particle_fill_kill_op(tbb::concurrent_vector<node_t*>& in_nodes,
			openvdb::FloatGrid::Ptr in_boundary_fill_kill_volume,
			const openvdb::Coord& in_domain_begin, const openvdb::Coord& in_domain_end,
			int fill_layer, openvdb::math::Transform::Ptr in_transform,
			openvdb::FloatGrid::Ptr in_solid_grid,
			openvdb::Vec3fGrid::Ptr in_velocity_volume,
			FLIP_vdb::descriptor_t::Ptr p_descriptor,
			FLIP_vdb::descriptor_t::Ptr pv_descriptor
		) :m_leaf_nodes(in_nodes),
			m_p_descriptor(p_descriptor),
			m_pv_descriptor(pv_descriptor) {
			m_boundary_fill_kill_volume = in_boundary_fill_kill_volume;
			m_domain_begin = in_domain_begin;
			m_domain_end = in_domain_end;
			m_int_begin = in_domain_begin + openvdb::Coord{ fill_layer };
			m_int_end = in_domain_end - openvdb::Coord{ fill_layer };
			m_transform = in_transform;
			m_solid = in_solid_grid;
			m_velocity_volume = in_velocity_volume;
		}//end constructor

		bool voxel_inside_inner_box(const openvdb::Coord& voxel_coord) const {
			return (voxel_coord[0] >= m_int_begin[0] &&
				voxel_coord[1] >= m_int_begin[1] &&
				voxel_coord[2] >= m_int_begin[2] &&
				voxel_coord[0] < m_int_end[0] &&
				voxel_coord[1] < m_int_end[1] &&
				voxel_coord[2] < m_int_end[2]);
		}

		bool voxel_inside_outer_box(const openvdb::Coord& voxel_coord) const {
			return (voxel_coord[0] >= m_domain_begin[0] &&
				voxel_coord[1] >= m_domain_begin[1] &&
				voxel_coord[2] >= m_domain_begin[2] &&
				voxel_coord[0] < m_domain_end[0] &&
				voxel_coord[1] < m_domain_end[1] &&
				voxel_coord[2] < m_domain_end[2]);
		}
		//the leaf nodes here are assumed to either contain solid voxels
		//or has any corner outside the interior box
		void operator()(const tbb::blocked_range<openvdb::Index>& r) const {
			std::vector<openvdb::PointDataIndex32> new_attribute_offsets;
			std::vector<openvdb::Vec3f> new_positions;
			std::vector<openvdb::Vec3f> new_velocity;

			//solid is assumed to point to the minimum corner of this voxel
			auto solid_axr = m_solid->getConstAccessor();
			auto velocity_axr = m_velocity_volume->getConstAccessor();
			auto boundary_volume_axr = m_boundary_fill_kill_volume->getConstAccessor();

			float voxel_include_threshold = m_transform->voxelSize()[0]*0.5;
			float particle_radius = 0.55*m_transform->voxelSize()[0] * std::sqrt(3) / 2.0;
			//minimum number of particle per voxel
			const int min_np = 8;
			//max number of particle per voxel
			const int max_np = 16;

			new_positions.reserve(max_np * 512);
			new_velocity.reserve(max_np * 512);
			//loop over the leaf nodes
			for (auto ileaf = r.begin(); ileaf != r.end(); ++ileaf) {
				node_t& leaf = *m_leaf_nodes[ileaf];

				// If this leaf is totally outside of the bounding box
				if (!boxes_overlap(leaf.origin(), leaf.origin() + openvdb::Coord{ 8 },
					m_domain_begin, m_domain_end)) {
					leaf.clearAttributes();
					continue;
				}

				//this leaf could potentially be a empty node without attribute
				//if so, create the position and velocity attribute
				if (leaf.attributeSet().size() == 0) {
					auto local_pv_descriptor = m_pv_descriptor;
					leaf.initializeAttributes(m_p_descriptor, 0);

					leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
				}

				// Randomize the point positions.
				std::random_device device;
				std::mt19937 generator(/*seed=*/device());
				std::uniform_real_distribution<> distribution(-0.5, 0.5);

				// Attribute reader
				// Extract the position attribute from the leaf by name (P is position).
				const openvdb::points::AttributeArray& positionArray =
					leaf.attributeArray("P");
				// Extract the velocity attribute from the leaf by name (v is velocity).
				const openvdb::points::AttributeArray& velocityArray =
					leaf.attributeArray("v");

				// Create read handles for position and velocity
				openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
				openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);

				//clear the new offset to be assigned
				new_attribute_offsets.clear();
				new_positions.clear();
				new_velocity.clear();
				openvdb::Index current_particle_count = 0;

				//at least reserve the original particles spaces
				new_positions.reserve(positionArray.size());
				new_velocity.reserve(velocityArray.size());
				new_attribute_offsets.reserve(leaf.SIZE);
				//emit new particles and transfer old particles
				for (openvdb::Index offset = 0; offset < leaf.SIZE; offset++) {
					openvdb::Index original_attribute_begin = 0;
					if (offset != 0) {
						original_attribute_begin = leaf.getValue(offset - 1);
					}
					const openvdb::Index original_attribute_end = leaf.getValue(offset);
					const auto voxel_gcoord = leaf.offsetToGlobalCoord(offset);

					//1**********************************
					//domain boundary check
					if (!voxel_inside_outer_box(voxel_gcoord)) {
						new_attribute_offsets.push_back(current_particle_count);
						continue;
					}

					//2**********************************
					//if the particle voxel has eight solid vertices
					bool all_solid = true;
					for (int ii = 0; ii < 2 && all_solid; ii++) {
						for (int jj = 0; jj < 2 && all_solid; jj++) {
							for (int kk = 0; kk < 2 && all_solid; kk++) {
								if (solid_axr.getValue(voxel_gcoord + openvdb::Coord{ ii,jj,kk }) > 0) {
									all_solid = false;
								}
							}
						}
					}
					if (all_solid) {
						new_attribute_offsets.push_back(current_particle_count);
						continue;
					}


					//3************************************
					//if it is not inside the fill zone
					//just emit the original particles
					if (voxel_inside_inner_box(voxel_gcoord)) {
						//test if it has any solid
						//if so, remove the particles with solid phi<0
						bool all_liquid = true;
						for (int ii = 0; ii < 2 && all_liquid; ii++) {
							for (int jj = 0; jj < 2 && all_liquid; jj++) {
								for (int kk = 0; kk < 2 && all_liquid; kk++) {
									if (solid_axr.getValue(voxel_gcoord + openvdb::Coord{ ii,jj,kk }) < 0) {
										all_liquid = false;
									}
								}
							}
						}

						for (int i_emit = original_attribute_begin; i_emit < original_attribute_end; i_emit++) {
							auto current_pos = positionHandle.get(i_emit);
							if (all_liquid) {
								new_positions.push_back(current_pos);
								new_velocity.push_back(velocityHandle.get(i_emit));
								current_particle_count++;
							}
							else {
								if (openvdb::tools::BoxSampler::sample(solid_axr, voxel_gcoord + current_pos) > 0) {
									new_positions.push_back(current_pos);
									new_velocity.push_back(velocityHandle.get(i_emit));
									current_particle_count++;
								}//end if the particle position if outside solid
							}

						}
						//current_particle_count += original_attribute_end - original_attribute_begin;
					}
					else {
						//the fill / kill zone

						//test if this voxel has at least half below the sea level
						//min_np particles will fill the lower part of this voxel
						//automatically generating dense particles near sea level
						//particle voxel has lattice at its center
						auto voxel_particle_wpos = m_transform->indexToWorld(voxel_gcoord);
						auto voxel_boundary_ipos = m_boundary_fill_kill_volume->worldToIndex(voxel_particle_wpos);
						if (openvdb::tools::BoxSampler::sample(boundary_volume_axr,voxel_boundary_ipos)< voxel_include_threshold) {
							//this voxel is below the sea
							//emit the original particles in this voxel until max capacity
							//some particles will not be emitted because it is 
							int emitter_end = std::min(original_attribute_end, original_attribute_begin + max_np);
							int this_voxel_emitted = 0;
							for (int i_emit = emitter_end; i_emit > original_attribute_begin; i_emit--) {
								int ie = i_emit - 1;
								const openvdb::Vec3f particle_voxel_pos = positionHandle.get(ie);
								auto point_particle_wpos = m_transform->indexToWorld(particle_voxel_pos + voxel_gcoord);
								auto point_boundary_ipos = m_boundary_fill_kill_volume->worldToIndex(point_particle_wpos);
								if (openvdb::tools::BoxSampler::sample(boundary_volume_axr, point_boundary_ipos) < 0) {
									new_positions.push_back(particle_voxel_pos);
									new_velocity.push_back(velocityHandle.get(ie));
									current_particle_count++;
									this_voxel_emitted++;
								}
							}//end for original particles in this voxel

							//additionally check if we need to emit new particles to fill this voxel
							//emit particles up to min_np
							for (; this_voxel_emitted < min_np; ) {
								//generate new position
								openvdb::Vec3f pos{ distribution(generator) };
								auto new_point_particle_wpos = m_transform->indexToWorld(pos + voxel_gcoord);
								auto new_point_boundary_ipos = m_boundary_fill_kill_volume->worldToIndex(new_point_particle_wpos);
								if (openvdb::tools::BoxSampler::sample(boundary_volume_axr, new_point_boundary_ipos) < -particle_radius) {
									//fully emit
									pos[0] = distribution(generator);
									pos[2] = distribution(generator);
									new_positions.push_back(pos);

									const openvdb::Vec3f vel = openvdb::tools::BoxSampler::sample(
										m_velocity_volume->tree(),
										m_velocity_volume->worldToIndex(
											m_transform->indexToWorld(pos + voxel_gcoord)));
									new_velocity.push_back(vel);

									/*m_pos_offsets[ileaf].push_back(
										std::make_pair(m_transform->indexToWorld(pos + voxel_gcoord),
											current_particle_count));*/
									
									current_particle_count++;
								}//end if particle below voxel sea level
								this_voxel_emitted++;
							}//end for fill new particles in voxel

						}//end if this voxel center is below sealevel

						//at this point the voxel is in the air, so do not emit particles
					}//end else voxel inside inner box

					//emit finished
					//record the new offset for this voxel
					new_attribute_offsets.push_back(current_particle_count);
				}//end for all voxels

				// Below this line, the positions and offsets of points are set
				// update the offset in this leaf and attribute array

				if (new_positions.empty()) {
					leaf.clearAttributes();
				}
				else {
					auto local_pv_descriptor = m_pv_descriptor;
					leaf.initializeAttributes(m_p_descriptor, new_positions.size());

					leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);

					leaf.setOffsets(new_attribute_offsets);


					//set the positions and velocities
					//get the new attribute arrays
					openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
					openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

					// Create read handles for position and velocity
					openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
					openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

					for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
						posWHandle.set(*iter, new_positions[*iter]);
						vWHandle.set(*iter, new_velocity[*iter]);
					}
				}//end if has points to write to attribute

			}//end for range leaf
		}//end operator


		const FLIP_vdb::descriptor_t::Ptr m_p_descriptor;
		const FLIP_vdb::descriptor_t::Ptr m_pv_descriptor;

		//outside box minmax, and inside box minmax
		openvdb::Coord m_domain_begin, m_domain_end, m_int_begin, m_int_end;

		//functions used to determine if a point is below sea_level
		openvdb::FloatGrid::Ptr m_boundary_fill_kill_volume;

		//size: number of leaf nodes intersecting the fillzone
		tbb::concurrent_vector<node_t*>& m_leaf_nodes;

		//solid sdf function used to detect voxels fully merged in solid
		openvdb::FloatGrid::Ptr m_solid;

		//velocity volume function used to set the velocity of the newly emitted particles
		openvdb::Vec3fGrid::Ptr m_velocity_volume;

		//transform of the particle grid
		openvdb::math::Transform::Ptr m_transform;

		//size: same as m_leaf_nodes to modify the velocity later
		//originally designed for batch velocity interpolation
		//now replaced by velocity volume
		//std::vector<std::vector<pos_offset_t>>& m_pos_offsets;
	};//end fill kil operator

	struct get_fill_kill_nodes_op {
		using node_t = openvdb::points::PointDataGrid::TreeType::LeafNodeType;

		get_fill_kill_nodes_op(openvdb::points::PointDataGrid::Ptr in_particles,
			openvdb::FloatGrid::Ptr in_solid,
			tbb::concurrent_vector<node_t*>& in_out_leaf_nodes,
			const openvdb::Coord& domain_begin, const openvdb::Coord& domain_end,
			openvdb::FloatGrid::Ptr in_boundary_fill_kill_volume
		) : m_leaf_nodes(in_out_leaf_nodes) {
			m_particles = in_particles;
			m_solid = in_solid;
			fill_layer = 4,
				m_domain_begin = domain_begin;
			m_domain_end = domain_end;
			int_begin = m_domain_begin + openvdb::Coord{ fill_layer };
			int_end = m_domain_end - openvdb::Coord{ fill_layer };
			m_boundary_fill_kill_volume = in_boundary_fill_kill_volume;
			touch_fill_nodes();
			m_leaf_nodes.clear();
		}


		//this is intended to be used with leaf manager of particles
		void operator()(node_t& leaf, openvdb::Index leafpos) const {
			auto solid_axr = m_solid->getConstAccessor();

			auto worth_dealing = [&](const openvdb::Coord& origin) {
				//the node must either include active solid voxels
				//so it will be used to delete particles
				//or is beyond the inner minmax so it will be used for filling
				//and for deleting particles
				if (solid_axr.probeConstLeaf(origin)) return true;
				if (solid_axr.getValue(origin) < 0) return true;

				for (int ii = 0; ii < 16; ii += 8) {
					for (int jj = 0; jj < 16; jj += 8) {
						for (int kk = 0; kk < 16; kk += 8) {
							for (int ic = 0; ic < 3; ic++) {
								if (origin[ic] + openvdb::Coord{ ii,jj,kk }[ic] < int_begin[ic]) {
									return true;
								}
								if (origin[ic] + openvdb::Coord{ ii,jj,kk }[ic] > int_end[ic]) {
									return true;
								}
							}
						}
					}
				}

				return false;
			};//end leaf worth dealing

			if (worth_dealing(leaf.origin())) {
				m_leaf_nodes.push_back(&leaf);
			}
		}

		void touch_fill_nodes() {
			//the first fill node
			auto first_leaf = m_particles->treePtr()->touchLeaf(m_domain_begin);
			const auto leaf_idxmin = first_leaf->origin();

			auto solid_axr = m_solid->getConstAccessor();
			//loop over all leaf nodes intersecting the big bounding box
			for (int x = leaf_idxmin[0]; x < m_domain_end[0]; x += 8) {
				for (int y = leaf_idxmin[1]; y < m_domain_end[1]; y += 8) {
					for (int z = leaf_idxmin[2]; z < m_domain_end[2]; z += 8) {
						openvdb::Coord origin{ x,y,z };
						if (intersect_fill_layer(origin)) {
							m_particles->tree().touchLeaf(origin);
						}
					}
				}
			}
			//note there might be nodes with particles but are
			//beyond the big bounding box, hence touching the fill nodes
			//and detect solid leaves within the bounding box may not find them
			//all nodes are discovered in the operator() function
		}

		bool intersect_fill_layer(const openvdb::Coord& origin) {
			bool outside_smallbox = false;

			for (int ii = 0; ii < 16 && !outside_smallbox; ii += 8) {
				for (int jj = 0; jj < 16 && !outside_smallbox; jj += 8) {
					for (int kk = 0; kk < 16 && !outside_smallbox; kk += 8) {
						//any of the eight corner is outside of the small box,
						auto test_coord = origin + openvdb::Coord{ ii,jj,kk };
						for (int ic = 0; ic < 3 && !outside_smallbox; ic++) {
							if (test_coord[ic] < int_begin[ic] || test_coord[ic] > int_end[ic]) {
								outside_smallbox = true;
							}
						}
					}
				}
			}

			//it is purely liquid
			if (!outside_smallbox) return false;

			//by default, this leaf intersects the big bounding box
			//check bottom sealevel 
			float voxel_sdf_threshold = m_particles->transform().voxelSize()[0];
			for (int ii = 0; ii < 8; ii += 1) {
				for (int kk = 0; kk < 8; kk += 1) {
					auto worigin = m_particles->indexToWorld(origin.offsetBy(ii, 0, kk));
					auto iorigin = m_boundary_fill_kill_volume->worldToIndex(worigin);
					if (openvdb::tools::BoxSampler::sample(m_boundary_fill_kill_volume->tree(),iorigin)< voxel_sdf_threshold) {
						return true;
					}
				}
			}

			//all points above the waterline
			return false;
		};// end intersect_fill_layer

		openvdb::FloatGrid::Ptr m_boundary_fill_kill_volume;

		openvdb::Coord m_domain_begin, m_domain_end, int_begin, int_end;

		int fill_layer;

		openvdb::points::PointDataGrid::Ptr m_particles;

		openvdb::FloatGrid::Ptr m_solid;

		//size: number of leaf nodes intersecting the fillzone
		tbb::concurrent_vector<node_t*>& m_leaf_nodes;
	};

}

void FLIP_vdb::fill_kill_particles()
{
	using partleaf_t = get_fill_kill_nodes_op::node_t;
	tbb::concurrent_vector<partleaf_t*> fill_kill_nodes;


	auto gen_nodes_op = get_fill_kill_nodes_op(
		m_particles,
		m_solid_sdf,
		fill_kill_nodes,
		openvdb::Coord(m_domain_index_begin),
		openvdb::Coord(m_domain_index_end),
		m_boundary_fill_kill_volume);

	auto pleavs_manager = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(m_particles->tree());

	pleavs_manager.foreach(gen_nodes_op);
	//fill and kill the particles

	auto filler = particle_fill_kill_op(fill_kill_nodes,
		m_boundary_fill_kill_volume,
		openvdb::Coord(m_domain_index_begin), openvdb::Coord(m_domain_index_end),
		/*fill_layer*/4, m_particles->transformPtr(),
		m_solid_sdf,
		m_boundary_velocity_volume,
		pos_descriptor(),
		pv_descriptor()
	);

	tbb::parallel_for(tbb::blocked_range<openvdb::Index>(0, fill_kill_nodes.size(), 8), filler);
	m_particles->pruneGrid(0);
	//write_points("pointsdbg.vdb");
}
namespace {
	struct tbbcoordhash {
		std::size_t hash(const openvdb::Coord& c) const {
			return c.hash();
		}

		bool equal(const openvdb::Coord& a, const openvdb::Coord& b) const {
			return a == b;
		}
	};
}

void FLIP_vdb::seed_liquid(openvdb::FloatGrid::Ptr in_sdf, const openvdb::Vec3f& init_vel)
{
	//the input is assumed to be narrowband boxes
	auto inputbbox = openvdb::CoordBBox();
	auto is_valid = in_sdf->tree().evalLeafBoundingBox(inputbbox);
	if (!is_valid) {
		return;
	}
	using point_leaf_t = openvdb::points::PointDataTree::LeafNodeType;
	tbb::concurrent_hash_map<openvdb::Coord, point_leaf_t*, tbbcoordhash> target_point_leaves;

	//check bounding box
	auto particle_bbox = openvdb::CoordBBox(openvdb::Coord(m_domain_index_begin-8), openvdb::Coord(m_domain_index_end - 1 + 8));
	
	auto insdfman = openvdb::tree::LeafManager<openvdb::FloatTree>(in_sdf->tree());
	auto mark_active_point_leaves = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		using  namespace openvdb::tools::local_util;
		auto target_wposbegin = in_sdf->indexToWorld(leaf.origin());
		auto target_wposend = in_sdf->indexToWorld(leaf.origin().offsetBy(8));
		auto target_iposbegin = m_particles->worldToIndex(target_wposbegin);
		auto target_iposend = m_particles->worldToIndex(target_wposend);

		for (auto target_ipos = target_iposbegin; target_ipos < target_iposend; target_ipos += 8) {
			if (target_ipos[0] >= particle_bbox.min()[0] && target_ipos[0] <= particle_bbox.max()[0]
				&& target_ipos[1] >= particle_bbox.min()[1] && target_ipos[2] <= particle_bbox.max()[1]
				&& target_ipos[2] >= particle_bbox.min()[2] && target_ipos[2] <= particle_bbox.max()[2]) {

				auto gcoordf = Coord(floorVec3(target_ipos));
				auto gcoordc = Coord(ceilVec3(target_ipos));
				auto floor_leaf = m_particles->tree().probeLeaf(gcoordf);
				auto ceil_leaf = m_particles->tree().probeLeaf(gcoordc);
				if (floor_leaf) {
					target_point_leaves.insert(std::make_pair(floor_leaf->origin(), floor_leaf));
				}
				if (ceil_leaf && (ceil_leaf != floor_leaf)) {
					target_point_leaves.insert(std::make_pair(ceil_leaf->origin(), ceil_leaf));
				}

				if (floor_leaf || ceil_leaf) {
					return;
				}
				else {
					auto newleafc = new openvdb::points::PointDataTree::LeafNodeType(gcoordc, 0, true);
					auto new_origin = newleafc->origin();
					bool success = target_point_leaves.insert(std::make_pair(newleafc->origin(), newleafc));
					if (!success) {
						delete newleafc;
						
					}

					//if the floor index belongs to another leaf
					//create another leaf
					if ((gcoordf < new_origin)) {
						auto newleaff = new openvdb::points::PointDataTree::LeafNodeType(gcoordf, 0, true);
						bool success = target_point_leaves.insert(std::make_pair(newleaff->origin(), newleaff));
						if (!success) {
							delete newleaff;
						}
					}
				}
			}//end if this point is inside the bounding box of particle domain
		}//end loop over the in sdf leaf mapped into particle space
	};//end mark active point elaves
	insdfman.foreach(mark_active_point_leaves);

	std::vector<point_leaf_t*> leafarray;
	for (auto& i : target_point_leaves) {
		leafarray.push_back(i.second);
	}
	printf("touched_leafs:%zd\n", leafarray.size());

	std::atomic<size_t> added_particles{ 0 };

	//seed each leaf
	auto seed_leaf = [&](const tbb::blocked_range<size_t>& r) {
		auto solid_axr{ m_solid_sdf->getConstAccessor() };
		auto in_sdf_axr{ in_sdf->getConstAccessor() };
		float sdf_threshold = -m_particle_radius * 0.55;

		std::random_device device;
		std::mt19937 generator(/*seed=*/device());
		std::uniform_real_distribution<> distribution(-0.5, 0.5);

		for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
			auto& leaf = *leafarray[ileaf];
			std::vector<openvdb::Vec3f> new_pos, new_vel;
			std::vector<openvdb::PointDataIndex32> new_idxend;

			//check if the last element is zero
			//if so, it's a new leaf
			if (leaf.getLastValue() == 0) {
				//initialize the attribute descriptor
				auto local_pv_descriptor = pv_descriptor();
				leaf.initializeAttributes(pos_descriptor(), 0);
				leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
			}

			//reader
			const openvdb::points::AttributeArray& positionArray =
				leaf.attributeArray("P");
			// Extract the velocity attribute from the leaf by name (v is velocity).
			const openvdb::points::AttributeArray& velocityArray =
				leaf.attributeArray("v");
			auto p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
			auto v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

			openvdb::Index32 emitted_particle = 0;
			for (auto offset = 0; offset < leaf.SIZE;++offset) {
				openvdb::Index32 idxbegin = 0;
				if (offset != 0) {
					idxbegin = leaf.getValue(offset - 1);
				}
				openvdb::Index32 idxend = leaf.getValue(offset);
				//first emit original particles
				openvdb::Index32 this_voxel_emitted = 0;
				for (auto idx = idxbegin; idx < idxend; ++idx) {
					new_pos.push_back(p_handle_ptr->get(idx));
					new_vel.push_back(v_handle_ptr->get(idx));
					emitted_particle++;
					this_voxel_emitted++;
				}

				//emit to fill the sdf
				auto voxelwpos = m_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
				auto voxelipos = in_sdf->worldToIndex(voxelwpos);
				if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < m_dx) {
					for (; this_voxel_emitted < 8; this_voxel_emitted++) {
						openvdb::Vec3d particle_pipos{ distribution(generator) ,distribution(generator) ,distribution(generator) };
						auto particlewpos = m_particles->indexToWorld(particle_pipos) + voxelwpos;
						auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
						if (openvdb::tools::BoxSampler::sample(in_sdf_axr, particle_sdfipos) < sdf_threshold) {
							new_pos.push_back(particle_pipos);
							new_vel.push_back(init_vel);
							emitted_particle++;
							added_particles++;
						}
					}
				}
				
				new_idxend.push_back(emitted_particle);
			}//end emit original particles and new particles

			//set the new index ends, attributes
			//initialize the attribute descriptor
			auto local_pv_descriptor = pv_descriptor();
			leaf.initializeAttributes(pos_descriptor(), emitted_particle);
			leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);

			leaf.setOffsets(new_idxend);
			
			//set the positions and velocities
			//get the new attribute arrays
			openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
			openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

			// Create read handles for position and velocity
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

			for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
				posWHandle.set(*iter, new_pos[*iter]);
				vWHandle.set(*iter, new_vel[*iter]);
			}//end for all on voxels
		}//end for range leaf
	};//end seed particle 

	tbb::parallel_for(tbb::blocked_range<size_t>(0, leafarray.size()), seed_leaf);
	printf("added particles:%zd\n", size_t(added_particles));
	printf("original leaf count:%d\n", m_particles->tree().leafCount());
	for (auto leaf : leafarray) {
		m_particles->tree().addLeaf(leaf);
	}
	printf("new leafcount:%d\n", m_particles->tree().leafCount());
	
	//write_points("pointsdbg.vdb");
	particle_to_grid_collect_style();
	extrapolate_velocity(5);
}

// void FLIP_vdb::emit_liquid(openvdb::points::PointDataGrid::Ptr in_out_particles, 
// 	const std::vector<sdf_vel_pair>& in_sdf_vels, 
// 	const openvdb::Vec3f& in_emit_world_min, 
// 	const openvdb::Vec3f& in_emit_world_max)
// {
// 	for (const auto& sv : in_sdf_vels) {
// 		emit_liquid(in_out_particles, sv, in_emit_world_min, in_emit_world_max);
// 	}
// }

// void FLIP_vdb::emit_liquid(openvdb::points::PointDataGrid::Ptr in_out_particles, const sdf_vel_pair& in_sdf_vel, const openvdb::Vec3f& in_emit_world_min, const openvdb::Vec3f& in_emit_world_max)
// {
// 	using  namespace openvdb::tools::local_util;
// 	if (!in_sdf_vel.m_activated) {
// 		return;
// 	}
// 	//alias
// 	auto in_sdf = in_sdf_vel.m_sdf;
// 	auto in_vel = in_sdf_vel.m_vel;

// 	//the input is assumed to be narrowband boxes
// 	auto inputbbox = openvdb::CoordBBox();
// 	auto is_valid = in_sdf->tree().evalLeafBoundingBox(inputbbox);
// 	if (!is_valid) {
// 		return;
// 	}

// 	//bounding box in the world coordinates
// 	auto worldinputbbox = openvdb::BBoxd(in_sdf->indexToWorld(inputbbox.min()), in_sdf->indexToWorld(inputbbox.max()));
// 	for (int ic = 0; ic < 3; ic++) {
// 		if (worldinputbbox.min()[ic] < in_emit_world_min[ic]) {
// 			worldinputbbox.min()[ic] = in_emit_world_min[ic];
// 		}
// 		if (worldinputbbox.max()[ic] > in_emit_world_max[ic]) {
// 			worldinputbbox.max()[ic] = in_emit_world_max[ic];
// 		}
// 	}

// 	//map the bounding box to the particle index space
// 	auto loopbegin = floorVec3(in_out_particles->worldToIndex(worldinputbbox.min())) - openvdb::Vec3i(8);
// 	auto loopend = ceilVec3(in_out_particles->worldToIndex(worldinputbbox.max())) + openvdb::Vec3i(8);

// 	auto tempgrid = openvdb::BoolGrid::create();
// 	tempgrid->setTransform(in_out_particles->transformPtr());
// 	auto templeaf =  tempgrid->tree().touchLeaf(Coord(loopbegin));

// 	loopbegin = templeaf->origin().asVec3i()/8;
// 	templeaf = tempgrid->tree().touchLeaf(Coord(loopend));
// 	loopend = templeaf->origin().asVec3i()/8;

// 	using point_leaf_t = openvdb::points::PointDataTree::LeafNodeType;
// 	tbb::concurrent_hash_map<openvdb::Coord, point_leaf_t*, tbbcoordhash> target_point_leaves;

// 	auto pispace_range = tbb::blocked_range3d<int, int, int>(loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(), loopend.z());

// 	float dx = in_out_particles->transform().voxelSize()[0];
// 	auto mark_active_point_leaves = [&](const tbb::blocked_range3d<int,int,int>& r) {
// 		auto solid_axr{ in_sdf->getConstAccessor() };


// 		for (int i = r.pages().begin(); i < r.pages().end(); i +=1) {
// 			for (int j = r.rows().begin(); j < r.rows().end(); j +=1) {
// 				for (int k = r.cols().begin(); k < r.cols().end(); k +=1) {
// 					//test if any of the corner of this box touches the 
// 					int has_solid = 0;
// 					int i8 = i * 8, j8 = j * 8, k8 = k * 8;
// 					for (int ii = 0; ii <= 8 && !has_solid; ii += 8) {
// 						for (int jj = 0; jj <= 8 && !has_solid; jj += 8) {
// 							for (int kk = 0; kk <= 8 && !has_solid; kk += 8) {
// 								auto particle_idx_pos = openvdb::Vec3i(i8 + ii, j8 + jj, k8 + kk);
// 								has_solid = (openvdb::tools::BoxSampler::sample(solid_axr, in_sdf->worldToIndex(in_out_particles->indexToWorld(particle_idx_pos))) < 0);
// 							}
// 						}
// 					}

// 					if (has_solid) {
// 						auto floor_leaf = in_out_particles->tree().probeLeaf(openvdb::Coord(i8,j8,k8));
// 						if (floor_leaf) {
// 							target_point_leaves.insert(std::make_pair(floor_leaf->origin(), floor_leaf));
// 						}
// 						else {
// 							auto newleaff = new openvdb::points::PointDataTree::LeafNodeType(openvdb::Coord(i8, j8, k8), 0, true);
// 							auto new_origin = newleaff->origin();
// 							bool success = target_point_leaves.insert(std::make_pair(newleaff->origin(), newleaff));
// 							if (!success) {
// 								delete newleaff;
// 							}
// 						}//end else found existing particle leaf
// 					}//end if touch solid
// 				}//loop k
// 			}//loop j
// 		}//loop i
// 	};//end mark active point elaves
// 	tbb::parallel_for(pispace_range, mark_active_point_leaves);

// 	//for parallel process
// 	std::vector<point_leaf_t*> leafarray;
// 	for (auto& i : target_point_leaves) {
// 		leafarray.push_back(i.second);
// 	}
// 	//printf("touched_leafs:%zd\n", leafarray.size());

// 	std::atomic<size_t> added_particles{ 0 };

// 	auto pnamepair = FLIP_vdb::position_attribute::attributeType();
// 	auto m_position_attribute_descriptor = openvdb::points::AttributeSet::Descriptor::create(pnamepair);
	
// 	auto vnamepair = velocity_attribute::attributeType();
// 	auto m_pv_attribute_descriptor = m_position_attribute_descriptor->duplicateAppend("v", vnamepair);

// 	//use existing descriptors if we already have particles
// 	if (!in_out_particles->empty()) {
// 		for (auto it = in_out_particles->tree().cbeginLeaf(); it; ++it) {
// 			if (it->getLastValue() != 0) {
// 				//a non-empty leaf with attributes
// 				m_pv_attribute_descriptor = it->attributeSet().descriptorPtr();
// 				break;
// 			}
// 		}
// 	}

// 	//seed each leaf
// 	auto seed_leaf = [&](const tbb::blocked_range<size_t>& r) {
// 		auto in_sdf_axr{ in_sdf->getConstAccessor() };
// 		auto in_vel_axr{ in_vel->getConstAccessor() };
// 		float sdf_threshold = -dx * 0.55;

// 		std::random_device device;
// 		std::mt19937 generator(/*seed=*/device());
// 		std::uniform_real_distribution<> distribution(-0.5, 0.5);

// 		for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
// 			auto& leaf = *leafarray[ileaf];
// 			std::vector<openvdb::Vec3f> new_pos, new_vel;
// 			std::vector<openvdb::PointDataIndex32> new_idxend;

// 			//check if the last element is zero
// 			//if so, it's a new leaf
// 			if (leaf.getLastValue() == 0) {
// 				//initialize the attribute descriptor
// 				auto local_pv_descriptor = m_pv_attribute_descriptor;
// 				leaf.initializeAttributes(m_position_attribute_descriptor, 0);
// 				leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
// 			}

// 			//reader
// 			const openvdb::points::AttributeArray& positionArray =
// 				leaf.attributeArray("P");
// 			// Extract the velocity attribute from the leaf by name (v is velocity).
// 			const openvdb::points::AttributeArray& velocityArray =
// 				leaf.attributeArray("v");
// 			auto p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
// 			auto v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

// 			openvdb::Index32 emitted_particle = 0;
// 			for (auto offset = 0; offset < leaf.SIZE; ++offset) {
// 				openvdb::Index32 idxbegin = 0;
// 				if (offset != 0) {
// 					idxbegin = leaf.getValue(offset - 1);
// 				}
// 				openvdb::Index32 idxend = leaf.getValue(offset);


// 				//used to indicate sub_voxel(8 cells) occupancy by particles
// 				unsigned char sub_voxel_occupancy = 0;
				
// 				//first emit original particles
// 				openvdb::Index32 this_voxel_emitted = 0;
// 				for (auto idx = idxbegin; idx < idxend; ++idx) {
// 					auto p = p_handle_ptr->get(idx);
// 					unsigned char sv_pos = ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
// 					//only emit uniformly, otherwise skip it
// 					if (sub_voxel_occupancy & (1 << sv_pos)) {
// 						//that bit is active, there is already an particle
// 						//skip it
// 						continue;
// 					}
// 					sub_voxel_occupancy |= (1 << sv_pos);
// 					new_pos.push_back(p);
// 					new_vel.push_back(v_handle_ptr->get(idx));
// 					emitted_particle++;
// 					this_voxel_emitted++;
// 				}

// 				//emit to fill the sdf
// 				auto voxelwpos = in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
// 				auto voxelipos = in_sdf->worldToIndex(voxelwpos);
// 				if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < dx) {
// 					const int max_emit_trial = 16;
// 					for (int trial = 0; this_voxel_emitted < 8 && trial<max_emit_trial; trial++) {
// 						openvdb::Vec3d particle_pipos{ distribution(generator) ,distribution(generator) ,distribution(generator) };
// 						auto& p = particle_pipos;
// 						unsigned char sv_pos = ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
// 						//only emit uniformly, otherwise skip it
// 						if (sub_voxel_occupancy & (1 << sv_pos)) {
// 							//that bit is active, there is already an particle
// 							//skip it
// 							continue;
// 						}
// 						auto particlewpos = in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
// 						auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
// 						if (openvdb::tools::BoxSampler::sample(in_sdf_axr, particle_sdfipos) < sdf_threshold) {
// 							sub_voxel_occupancy |= 1 << sv_pos;
// 							new_pos.push_back(particle_pipos);
// 							new_vel.push_back(openvdb::tools::BoxSampler::sample(in_vel_axr,particle_sdfipos));
// 							emitted_particle++;
// 							this_voxel_emitted++;
// 							added_particles++;
// 						}
// 					}//end for 16 emit trials
// 				}//if voxel inside emitter

// 				new_idxend.push_back(emitted_particle);
// 			}//end emit original particles and new particles

// 			//set the new index ends, attributes
// 			//initialize the attribute descriptor
// 			auto local_pv_descriptor = m_pv_attribute_descriptor;
// 			leaf.initializeAttributes(m_position_attribute_descriptor, emitted_particle);
// 			leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
// 			leaf.setOffsets(new_idxend);

// 			//set the positions and velocities
// 			//get the new attribute arrays
// 			openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
// 			openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

// 			// Create read handles for position and velocity
// 			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
// 			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

// 			for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
// 				posWHandle.set(*iter, new_pos[*iter]);
// 				vWHandle.set(*iter, new_vel[*iter]);
// 			}//end for all on voxels
// 		}//end for range leaf
// 	};//end seed particle 

// 	tbb::parallel_for(tbb::blocked_range<size_t>(0, leafarray.size()), seed_leaf);
// 	//printf("added particles:%zd\n", size_t(added_particles));
// 	//printf("original leaf count:%d\n", m_particles->tree().leafCount());
// 	for (auto leaf : leafarray) {
// 		if (leaf != in_out_particles->tree().probeLeaf(leaf->origin())) {
// 			in_out_particles->tree().addLeaf(leaf);
// 		}
// 	}
// 	//openvdb::io::File("dbg.vdb").write({ m_particles });
// 	//printf("new leafcount:%d\n", m_particles->tree().leafCount());
// 	in_out_particles->pruneGrid();
// }

// void FLIP_vdb::sink_liquid(const std::vector<openvdb::FloatGrid::Ptr>& in_sdfs)
// {
// 	//remove all particles inside solids, in sink zone, and outside of the bounding box
// 	auto partman = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(m_particles->tree());

// 	std::vector<openvdb::points::PointDataTree::LeafNodeType*> leaves;
// 	partman.getNodes(leaves);

// 	openvdb::Vec3d idx_domain_min = m_domain_index_begin - openvdb::Vec3d{ 0.5 }, 
// 		idx_domain_max= m_domain_index_end - openvdb::Vec3d{ 0.5 };

// 	auto sink_leaf = [&](const tbb::blocked_range<size_t>& r) {
// 		std::vector<openvdb::tree::ValueAccessor<const openvdb::FloatTree>> sink_axrs;
// 		for (size_t isink = 0; isink < in_sdfs.size(); isink++) {
// 			sink_axrs.push_back(openvdb::tree::ValueAccessor<const openvdb::FloatTree>{ in_sdfs[isink]->tree() });
// 		}

// 		auto particle_voxel_inside_domain = [&](const openvdb::Coord& voxel_coord) {
// 			return (voxel_coord[0] >= m_domain_index_begin[0] &&
// 				voxel_coord[1] >= m_domain_index_begin[1] &&
// 				voxel_coord[2] >= m_domain_index_begin[2] &&
// 				voxel_coord[0] < m_domain_index_end[0] &&
// 				voxel_coord[1] < m_domain_index_end[1] &&
// 				voxel_coord[2] < m_domain_index_end[2]);
// 		};

// 		auto touch_sink = [&](const openvdb::Vec3d& pipos, float delta = 0) {
// 			//check domain boundary in index space
// 			if (pipos[0] <= idx_domain_min[0] || pipos[0] >= idx_domain_max[0]
// 				|| pipos[1] <= idx_domain_min[1] || pipos[1] >= idx_domain_max[1]
// 				|| pipos[2] <= idx_domain_min[2] || pipos[2] >= idx_domain_max[2]) {
// 				return true;
// 			}

// 			openvdb::Vec3d wpos = m_particles->indexToWorld(pipos);

// 			//check sinks
// 			for (int isink = 0; isink < in_sdfs.size(); isink++) {
// 				openvdb::Vec3d sinkipos = in_sdfs[isink]->worldToIndex(wpos);
// 				if (openvdb::tools::BoxSampler::sample(sink_axrs[isink], sinkipos) < delta) {
// 					return true;
// 				}
// 			}

// 			return false;
// 		};//end touch_sink

// 		for (auto leafpos = r.begin(); leafpos != r.end(); leafpos++) {
// 			auto leaforigin = leaves[leafpos]->origin();

// 			bool corner_sink = false;
// 			//anyone of its corners touch sink zone then it is worth processing
// 			for (int i = 0; i <= 8 && !corner_sink; i += 8) {
// 				for (int j = 0; j <= 8 && !corner_sink; j += 8) {
// 					for (int k = 0; k <= 8 && !corner_sink; k += 8) {
// 						corner_sink |= touch_sink(leaforigin.offset(i, j, k).asVec3d() - openvdb::Vec3d{ 0.5 });
// 					}//k
// 				}//j
// 			}//i

// 			if (!corner_sink) { continue; }
// 			//below this line, at least one corner of the particle leaf touches sink zone
// 			//loop over all particles and remove those in sink zone

// 			//this bulk is empty so skip the deletion
// 			if (leaves[leafpos]->getLastValue() == 0) {
// 				continue;
// 			}
			
// 			//new p v arrays
// 			std::vector<openvdb::Vec3f> new_p, new_v;
// 			std::vector<openvdb::PointDataIndex32> new_idxoffset;
// 			new_idxoffset.reserve(leaves[leafpos]->SIZE);

// 			//pv readers
// 			const openvdb::points::AttributeArray& positionArray =
// 				leaves[leafpos]->attributeArray("P");
// 			// Extract the velocity attribute from the leaf by name (v is velocity).
// 			const openvdb::points::AttributeArray& velocityArray =
// 				leaves[leafpos]->attributeArray("v");

// 			// Create read handles for position and velocity
// 			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
// 			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);


// 			int emitted_particles = 0;
// 			//relaxed criteria for selecting voxels
// 			//allow further criteria for particles
// 			float voxeldelta = 0.5 * m_particles->transform().voxelSize()[0];

// 			//loop over all voxels
// 			for (auto offset = 0; offset < leaves[leafpos]->SIZE;++offset) {
// 				int idxend = leaves[leafpos]->getValue(offset);
// 				int idxbegin = 0;
// 				if (offset != 0) {
// 					idxbegin = leaves[leafpos]->getValue(offset - 1);
// 				}

// 				if (idxend == idxbegin) {
// 					new_idxoffset.push_back(emitted_particles);
// 					continue;
// 				}

// 				auto voxelipos =leaves[leafpos]->offsetToGlobalCoord(offset);

// 				if (!particle_voxel_inside_domain(voxelipos)) {
// 					new_idxoffset.push_back(emitted_particles);
// 					continue;
// 				}

// 				if (touch_sink(voxelipos.asVec3d(), voxeldelta)) {
// 					//emit those are qualified
// 					for (auto idx = idxbegin; idx < idxend; idx++) {
// 						auto particleipos = positionHandle.get(idx) + voxelipos;
// 						if (!touch_sink(particleipos)) {
// 							new_p.push_back(positionHandle.get(idx));
// 							new_v.push_back(velocityHandle.get(idx));
// 							emitted_particles++;
// 						}
// 					}
// 				}
// 				else {
// 					//emit as usual
// 					for (auto idx = idxbegin; idx < idxend; idx++) {
// 						new_p.push_back(positionHandle.get(idx));
// 						new_v.push_back(velocityHandle.get(idx));
// 						emitted_particles++;
// 					}
// 				}
// 				new_idxoffset.push_back(emitted_particles);
// 			}//end for all voxels

// 			//assign new attributes
// 			if (new_p.empty()) {
// 				leaves[leafpos]->clearAttributes();
// 			}
// 			else {
// 				auto local_pv_descriptor = m_pv_attribute_descriptor;
// 				leaves[leafpos]->initializeAttributes(m_position_attribute_descriptor, new_p.size());
// 				leaves[leafpos]->appendAttribute(leaves[leafpos]->attributeSet().descriptor(), local_pv_descriptor, 1);
// 				leaves[leafpos]->setOffsets(new_idxoffset);

// 				//set the positions and velocities
// 				//get the new attribute arrays
// 				openvdb::points::AttributeArray& posarray = leaves[leafpos]->attributeArray("P");
// 				openvdb::points::AttributeArray& varray = leaves[leafpos]->attributeArray("v");

// 				// Create read handles for position and velocity
// 				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
// 				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

// 				for (auto iter = leaves[leafpos]->beginIndexOn(); iter; ++iter) {
// 					posWHandle.set(*iter, new_p[*iter]);
// 					vWHandle.set(*iter, new_v[*iter]);
// 				}
// 			}//end if has points to write to attribute
// 		}//end for ranged leaf pos
// 	};//end sink_leaf lambda

// 	tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), sink_leaf);

// 	m_particles->pruneGrid();
// }

namespace {
	struct narrow_band_copy_iterator {

		narrow_band_copy_iterator(const std::vector<openvdb::Index>& in_idx_of_source):
			m_idx_of_source(in_idx_of_source) {
			m_it = 0;
		}

		narrow_band_copy_iterator(const narrow_band_copy_iterator& other):
		m_idx_of_source(other.m_idx_of_source),
		m_it(other.m_it) {

		}
		operator bool() const { return m_it < m_idx_of_source.size(); }

		narrow_band_copy_iterator& operator++() { ++m_it; return *this; }

		openvdb::Index sourceIndex() const
		{
			return m_idx_of_source[m_it];
		}

		Index targetIndex() const
		{
			return m_it;
		}

		//idx in the attribute array
		//[target_idx]=source_idx
		size_t m_it;
		const std::vector<openvdb::Index>& m_idx_of_source;
	};
}
// void FLIP_vdb::narrow_band_particles(openvdb::points::PointDataGrid::Ptr out_nb_particles, 
// 	openvdb::FloatGrid::Ptr out_interior_sdf, 
// 	openvdb::points::PointDataGrid::Ptr in_particles, int nlayer)
// {
// 	//particles are stored in particle voxels
// 	//1 dilate particle voxels and mark existing part as inactive to get the surface mask
// 	//2 dilate the surface mask nlayer-1 to get the erosion mask
// 	//3 dilate the surface mask nlayer to get the output particle mask
// 	//4 use the erosion mask to generate interior sdf (every voxel has -0.5dx)
// 	//5 use the output particle mask to generate the narrow banded particles

// 	using point_leaf_t = openvdb::points::PointDataTree::LeafNodeType;

// 	//1.1 dilate particle voxels to get surface mask
// 	auto surface_mask = openvdb::BoolGrid::create();
// 	surface_mask->setTree(std::make_shared<openvdb::BoolTree>(in_particles->tree(), /*inactive value*/false,/*active value*/true, openvdb::TopologyCopy()));
// 	openvdb::tools::dilateActiveValues(surface_mask->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);

// 	//1.2 hollow the surface mask
// 	auto hollow_surface_mask_lambda = [in_particles](openvdb::BoolTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		auto particle_leaf = in_particles->tree().probeConstLeaf(leaf.origin());
// 		if (!particle_leaf) {
// 			return;
// 		}

// 		for (auto offset = 0; offset < leaf.size(); ++offset) {
// 			if (particle_leaf->isValueOn(offset)) {
// 				leaf.setValueOff(offset);
// 			}
// 		}
// 	};//end hollow surface mask lambda

// 	auto surface_mask_man = openvdb::tree::LeafManager<openvdb::BoolTree>(surface_mask->tree());
// 	surface_mask_man.foreach(hollow_surface_mask_lambda);
// 	surface_mask->pruneGrid();

// 	//2 dilate the surface mask nlayer-1 to get the erosion mask
// 	auto erosion_mask = surface_mask->deepCopy();
// 	openvdb::tools::dilateActiveValues(erosion_mask->tree(), nlayer - 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);

// 	//3 dilate the erosion mask to get the output particle mask
// 	auto output_particle_mask = erosion_mask->deepCopy();
// 	openvdb::tools::dilateActiveValues(output_particle_mask->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);

// 	//4 use the erosion mask to generate interior sdf (every voxel has -0.5dx)
// 	float dx = in_particles->transform().voxelSize()[0];

// 	auto interior_sdf = openvdb::FloatGrid::create(0.5 * dx);
// 	interior_sdf->setTree(std::make_shared<openvdb::FloatTree>(in_particles->tree(), /*inactive value*/0.5f*dx,/*active value*/-0.5f*dx, openvdb::TopologyCopy()));

// 	auto erosion_lambda = [erosion_mask, dx](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		auto erosionleaf = erosion_mask->tree().probeConstLeaf(leaf.origin());
// 		if (!erosionleaf) {
// 			return;
// 		}

// 		for (auto offset = 0; offset < leaf.size(); ++offset) {
// 			if (erosionleaf->isValueOn(offset)) {
// 				leaf.setValueOff(offset, 0.5 * dx);
// 			}
// 		}
// 	};//end erosion lambda
// 	auto interior_sdf_man = openvdb::tree::LeafManager<openvdb::FloatTree>(interior_sdf->tree());
// 	interior_sdf_man.foreach(erosion_lambda);
	
// 	//5 use the output particle mask to generate the narrow banded particles
// 	auto nb_particles = openvdb::points::PointDataGrid::create();
// 	nb_particles->setTree(std::make_shared<openvdb::points::PointDataTree>(output_particle_mask->tree(), /*background value*/0, openvdb::TopologyCopy()));

// 	auto populate_nb_particle_lambda = [in_particles](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		auto original_leaf = in_particles->tree().probeConstLeaf(leaf.origin());
// 		if (!original_leaf) {
// 			leaf.setValuesOff();
// 			return;
// 		}

// 		if (original_leaf->getLastValue() == 0) {
// 			//original leaf is empty, possibly not having any attribute sets
// 			leaf.setValuesOff();
// 			return;
// 		}

// 		//write idx ends
// 		int emitted = 0;
// 		std::vector<openvdb::Index> original_particle_idx;

// 		for (auto offset=0; offset < leaf.size(); ++offset) {
// 			auto idxend = original_leaf->getValue(offset);
// 			auto idxbegin = 0;
// 			if (offset != 0) {
// 				idxbegin = original_leaf->getValue(offset - 1);
// 			}

// 			//emit if mask on
// 			if (leaf.isValueOn(offset)) {
// 				for (auto it = idxbegin; it < idxend; ++it) {
// 					original_particle_idx.push_back(it);
// 				}
// 				emitted += idxend - idxbegin;
// 			}
// 			leaf.setOffsetOnly(offset, emitted);
// 		}
		
// 		//copy the attributes
// 		leaf.replaceAttributeSet(new openvdb::points::AttributeSet(original_leaf->attributeSet(), emitted), true);
// 		for (const auto& it : original_leaf->attributeSet().descriptor().map()) {
// 			const auto attributeIndex = static_cast<openvdb::Index>(it.second);
			
// 			auto& target_array = leaf.attributeArray(attributeIndex);
			
// 			narrow_band_copy_iterator copyiter(original_particle_idx);

// 			target_array.copyValues(original_leaf->attributeArray(attributeIndex), copyiter);
// 		}
// 	};//end populate narrow band particles lambda
// 	auto nb_manager = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(nb_particles->tree());
// 	nb_manager.foreach(populate_nb_particle_lambda,true);
	
// 	//write to the output
// 	interior_sdf->pruneGrid();
// 	nb_particles->pruneGrid();

// 	out_interior_sdf->setTree(interior_sdf->treePtr());
// 	out_interior_sdf->setTransform(in_particles->transformPtr());
// 	out_interior_sdf->setName("surface");
// 	out_interior_sdf->setGridClass(openvdb::GRID_LEVEL_SET);
// 	out_nb_particles->setTree(nb_particles->treePtr());
// 	out_nb_particles->setTransform(in_particles->transformPtr());
// 	out_nb_particles->setName("particles");

// }


bool FLIP_vdb::below_waterline(float in_height)
{
	return in_height < m_waterline;
}

bool FLIP_vdb::below_sea_level(const openvdb::Vec3f& P)
{
	return below_waterline(P[1]);
}

float FLIP_vdb::index_space_sea_level(const openvdb::Coord& xyz) const
{
	return 1e-3f;
}

// void FLIP_vdb::init_velocity_volume()
// {
// 	openvdb::Vec3f init_vel{
// 	(float)Options::doubleValue("tank_flow_x"),
// 	(float)Options::doubleValue("tank_flow_y"),
// 	(float)Options::doubleValue("tank_flow_z")
// 	};
// 	//set initial zero velocity volume
// 	m_boundary_velocity_volume = openvdb::Vec3fGrid::create(openvdb::Vec3f(0));

// 	m_boundary_velocity_volume->setTransform(m_voxel_center_transform);
// 	//use the domain sdf to set inflow-outflow velocity
// 	for (auto iter = m_domain_solid_sdf->tree().cbeginLeaf(); iter; ++iter) {
// 		auto* p = m_boundary_velocity_volume->tree().touchLeaf(iter->origin());
// 		p->fill(init_vel);
// 	}
// }

void FLIP_vdb::init_boundary_fill_kill_volume()
{
	m_boundary_fill_kill_volume = openvdb::FloatGrid::create(1.0f);
	//create a box and run mesh to volume conversion
	float extension_band = 8.f;
	auto boxmin = m_domain_index_begin - openvdb::Vec3f{ extension_band };
	auto boxmax = m_domain_index_end + openvdb::Vec3f{ extension_band };
	boxmin = m_voxel_center_transform->indexToWorld(boxmin);
	boxmax = m_voxel_center_transform->indexToWorld(boxmax);
	//set the sea level
	boxmax[1] = 0;
	openvdb::Vec3f box_size = boxmax - boxmin;
	//create 6 quads
	std::vector<openvdb::Vec4I> faces; faces.resize(6);
	std::vector<openvdb::Vec3f> vtx; vtx.resize(8);

	//set 8 vertices
	for (int ii = 0; ii < 2; ii++) {
		for (int jj = 0; jj < 2; jj++) {
			for (int kk = 0; kk < 2; kk++) {
				vtx[ii * 4 + jj * 2 + kk] =
					boxmin + openvdb::Vec3f{ ii * box_size[0],jj * box_size[1],kk * box_size[2] };
			}
		}
	}

	//bottom: y=0
	// ^x
	// |  4    5
	// |
	// |  0    1
	// ---------->z
	//top y=1
	// ^x
	// |  6    7
	// |
	// |  2    3
	// ---------->z

	//x- face
	faces[0] = openvdb::Vec4I{ 0,1,3,2 };
	//x+
	faces[1] = openvdb::Vec4I{ 5,4,6,7 };
	//y-
	faces[2] = openvdb::Vec4I{ 1,0,4,5 };
	//y+
	faces[3] = openvdb::Vec4I{ 2,3,7,6 };
	//z-
	faces[4] = openvdb::Vec4I{ 4,0,2,6 };
	//z+
	faces[5] = openvdb::Vec4I{ 1,5,7,3 };

	m_boundary_fill_kill_volume = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*m_voxel_center_transform, vtx, faces, 3.0f);
	m_boundary_fill_kill_volume->setName("boundary_volume");
	m_boundary_fill_kill_volume->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	//openvdb::io::File("boundary_volume.vdb").write({ m_boundary_fill_kill_volume });
}

// void FLIP_vdb::init_domain_solid_sdf()
// {
// 	m_domain_solid_sdf = openvdb::FloatGrid::create(3.f * m_dx);
// 	m_domain_solid_sdf->setTransform(m_voxel_vertex_transform);
// 	m_domain_solid_sdf->setName("Domain_SDF");
// 	m_domain_solid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);

// 	int narrow_band_width = 8;

// 	auto shrinked_domain_begin = m_domain_index_begin + openvdb::Vec3i{ narrow_band_width };
// 	auto shrinked_domain_end = m_domain_index_end - openvdb::Vec3i{ narrow_band_width };

// 	auto extended_domain_begin = m_domain_index_begin - openvdb::Vec3i{ narrow_band_width };
// 	auto extended_domain_end = m_domain_index_end + openvdb::Vec3i{ narrow_band_width };

// 	//first node
// 	auto* first_node = m_domain_solid_sdf->tree().touchLeaf(openvdb::Coord{ extended_domain_begin });
// 	auto touch_domain_begin = first_node->origin();
// 	auto touch_domain_end = extended_domain_end;

// 	auto has_point_outside_domain = [&](const openvdb::Coord& leaf_origin) {
// 		for (int ii = 0; ii <= 8; ii += 8) {
// 			for (int jj = 0; jj <= 8; jj += 8) {
// 				for (int kk = 0; kk <= 8; kk += 8) {
// 					for (int ic = 0; ic < 3; ic++) {
// 						if (leaf_origin[ic] + openvdb::Coord{ ii,jj,kk }[ic] < shrinked_domain_begin[ic]) {
// 							return true;
// 						}
// 						if (leaf_origin[ic] + openvdb::Coord{ ii,jj,kk }[ic] > shrinked_domain_end[ic]) {
// 							return true;
// 						}
// 					}
// 				}
// 			}
// 		}
// 		return false;
// 	};

// 	for (int x = touch_domain_begin[0]; x <= touch_domain_end[0]; x++) {
// 		for (int y = touch_domain_begin[1]; y <= touch_domain_end[1]; y++) {
// 			for (int z = touch_domain_begin[2]; z <= touch_domain_end[2]; z++) {
// 				if (has_point_outside_domain(openvdb::Coord{ x,y,z })) {
// 					m_domain_solid_sdf->tree().touchLeaf(openvdb::Coord{ x,y,z });
// 				}
// 			}
// 		}
// 	}

// 	auto leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_domain_solid_sdf->tree());

// 	openvdb::Vec3f domain_center = 0.5f * (m_domain_index_begin + m_domain_index_end);
// 	openvdb::Vec3f domain_extend = openvdb::Vec3f{ m_domain_index_end } - domain_center;

// 	auto set_leaf_sdf = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		for (auto offset = 0; offset < leaf.SIZE; offset++) {
// 			//find the projection point on the boundary
// 			auto cpos = leaf.offsetToGlobalCoord(offset);
// 			//projecte cpos
// 			auto ppos = cpos;
// 			bool voxel_is_outside = false;
// 			for (int ic = 0; ic < 3; ic++) {
// 				if (ppos[ic] < m_domain_index_begin[ic]) {
// 					ppos[ic] = m_domain_index_begin[ic];
// 					voxel_is_outside = true;
// 				}
// 				//note the domain contains end-begin voxels
// 				//the end voxel is outside of the domain
// 				//but its solid vertex happens to be on the corner of the domain
// 				//it is still counted as inside the domain
// 				if (ppos[ic] > m_domain_index_end[ic]) {
// 					ppos[ic] = m_domain_index_end[ic];
// 					voxel_is_outside = true;
// 				}
// 			}

// 			if (voxel_is_outside) {
// 				//outside of the domain means it's inside the solid container, set negaive distance
// 				leaf.setValueOn(offset, -m_dx * (ppos - cpos).asVec3s().length());
// 			}
// 			else {
// 				//find the closest way to move this point to the boundary
// 				int effort[6];
// 				//negx
// 				effort[0] = cpos[0] - m_domain_index_begin[0];
// 				//negy
// 				effort[1] = cpos[1] - m_domain_index_begin[1];
// 				//negz
// 				effort[2] = cpos[2] - m_domain_index_begin[2];

// 				//posx
// 				effort[3] = m_domain_index_end[0] - cpos[0];
// 				//posy
// 				effort[4] = m_domain_index_end[1] - cpos[1];
// 				//posz
// 				effort[5] = m_domain_index_end[2] - cpos[2];

// 				int min_effort = effort[0];

// 				for (int ieffort = 1; ieffort < 6; ieffort++) {
// 					if (effort[ieffort] < min_effort) {
// 						min_effort = effort[ieffort];
// 					}
// 				}

// 				//inside the domain the value is positive;
// 				leaf.setValueOn(offset, m_dx * min_effort);
// 			}
// 		}//end for all voxel
// 	};//end set leaf boundary sdf

// 	leafman.foreach(set_leaf_sdf);
// }

namespace {
	//given input target solid sdf and source solid sdf
	//touch all target solid sdf where solid sdf is active
	// or has negative tiles
	// leaf iterator cannot find tiles
	struct solid_leaf_tile_toucher
	{
		
		solid_leaf_tile_toucher(openvdb::FloatGrid::Ptr in_target_solid_sdf, openvdb::FloatGrid::Ptr in_source_solid_sdf) {
			m_target_solid_sdf = in_target_solid_sdf;
			m_source_solid_sdf = in_source_solid_sdf;
		}
		template<typename IterT>
		inline bool operator()(IterT& iter)
		{
			using  namespace openvdb::tools::local_util;
			typename IterT::NonConstValueType value;
			typename IterT::ChildNodeType* child = iter.probeChild(value);
			//std::cout << "touch leaf" << iter.getCoord() << std::endl;
			if (child == nullptr) {
				//std::cout << "Tile with value " << value << std::endl;
				if (value < 0) {
					if (iter.parent().getLevel() == 1) {
						auto wpos = m_source_solid_sdf->indexToWorld(iter.getCoord());
						auto ipos = m_target_solid_sdf->worldToIndex(wpos);
						m_target_solid_sdf->tree().touchLeaf(openvdb::Coord(floorVec3(ipos)));
					}
				}
				return true; // no child to visit, so stop descending
			}
			//std::cout << "Level-" << child->getLevel() << " node" << std::endl;
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
		bool operator()(openvdb::FloatTree::LeafNodeType::ChildAllIter&) { return true; }
		bool operator()(openvdb::FloatTree::LeafNodeType::ChildAllCIter&) { return true; }

		openvdb::FloatGrid::Ptr m_target_solid_sdf;
		openvdb::FloatGrid::Ptr m_source_solid_sdf;
	};
}
void FLIP_vdb::update_solid_sdf(std::vector<openvdb::FloatGrid::Ptr> &moving_solids, openvdb::FloatGrid::Ptr solid_sdf, openvdb::points::PointDataGrid::Ptr particles)
{
	using  namespace openvdb::tools::local_util;
	for (auto& solidsdfptr : moving_solids) {
		for (auto leafiter = solidsdfptr->tree().cbeginLeaf(); leafiter; ++leafiter) {
			auto source_ipos = leafiter->origin();
			auto source_wpos = solidsdfptr->indexToWorld(source_ipos);
			auto target_ipos = openvdb::Coord(floorVec3(solid_sdf->worldToIndex(source_wpos)));
			solid_sdf->tree().touchLeaf(target_ipos);
		}
		/*auto toucher = solid_leaf_tile_toucher(m_solid_sdf, solidsdfptr);
		solidsdfptr->tree().visit(toucher);*/
	}
	//also retrieve all possible liquids trapped inside solid
	tbb::concurrent_vector<openvdb::Coord> neib_liquid_origins;
	auto collect_liquid_origins = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		for (auto& solidsdfptr : moving_solids) {
			auto axr{ solidsdfptr->getConstUnsafeAccessor() };
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
					}//end kk
				}//end jj
			}//end ii
		}//end for all solids
	};//end collect liquid origin
	auto partman = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(particles->tree());
	partman.foreach(collect_liquid_origins);

	//touch those liquid leafs as well
	for (auto coord : neib_liquid_origins) {
		solid_sdf->tree().touchLeaf(coord);
	}

	auto update_solid_op = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		for (const auto& solidsdfptr : moving_solids) {
			auto source_solid_axr{ solidsdfptr->getConstAccessor() };

			//get the sdf from the moving solids
			for (auto offset = 0; offset < leaf.SIZE; ++offset) {
				float current_sdf = leaf.getValue(offset);
				auto current_Ipos = leaf.offsetToGlobalCoord(offset);
				auto current_wpos = solid_sdf->indexToWorld(current_Ipos);
				//interpolate the value at the
				float source_sdf = openvdb::tools::BoxSampler::sample(
					source_solid_axr, solidsdfptr->worldToIndex(current_wpos));
				leaf.setValueOn(offset, std::min(current_sdf, source_sdf));
			}//end for target solid voxel
		}//end for all moving solids
	};//end update solid operator

	auto solid_leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(solid_sdf->tree());

	solid_leafman.foreach(update_solid_op);

	
}




void FLIP_vdb::emit_liquid(
	openvdb::points::PointDataGrid::Ptr in_out_particles,
	openvdb::FloatGrid::Ptr sdf,
	openvdb::Vec3fGrid::Ptr vel,
	float vx, float vy, float vz)
{
	using  namespace openvdb::tools::local_util;
	
	//alias
	auto in_sdf = sdf;
	auto in_vel = vel;
	bool use_vel_volume = false;
	//the input is assumed to be narrowband boxes
	auto inputbbox = openvdb::CoordBBox();
	auto is_valid = in_sdf->tree().evalLeafBoundingBox(inputbbox);
	if (!is_valid) {
		return;
	}
	if(vel!=nullptr)
	{
		use_vel_volume = true;
	}
	//bounding box in the world coordinates
	auto worldinputbbox = openvdb::BBoxd(in_sdf->indexToWorld(inputbbox.min()), 
		in_sdf->indexToWorld(inputbbox.max()));
	

	//map the bounding box to the particle index space
	auto loopbegin = floorVec3(in_out_particles->worldToIndex(worldinputbbox.min())) - openvdb::Vec3i(8);
	auto loopend = ceilVec3(in_out_particles->worldToIndex(worldinputbbox.max())) + openvdb::Vec3i(8);

	auto tempgrid = openvdb::BoolGrid::create();
	tempgrid->setTransform(in_out_particles->transformPtr());
	auto templeaf =  tempgrid->tree().touchLeaf(Coord(loopbegin));

	loopbegin = templeaf->origin().asVec3i()/8;
	templeaf = tempgrid->tree().touchLeaf(Coord(loopend));
	loopend = templeaf->origin().asVec3i()/8;

	using point_leaf_t = openvdb::points::PointDataTree::LeafNodeType;
	tbb::concurrent_hash_map<openvdb::Coord, point_leaf_t*, tbbcoordhash> target_point_leaves;

	auto pispace_range = tbb::blocked_range3d<int, int, int>(loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(), loopend.z());

	float dx = in_out_particles->transform().voxelSize()[0];
	printf("%f\n",dx);
	auto mark_active_point_leaves = [&](const tbb::blocked_range3d<int,int,int>& r) {
		auto solid_axr{ in_sdf->getConstAccessor() };


		for (int i = r.pages().begin(); i < r.pages().end(); i +=1) {
			for (int j = r.rows().begin(); j < r.rows().end(); j +=1) {
				for (int k = r.cols().begin(); k < r.cols().end(); k +=1) {
					//test if any of the corner of this box touches the 
					int has_solid = 0;
					int i8 = i * 8, j8 = j * 8, k8 = k * 8;
					for (int ii = 0; ii <= 8 && !has_solid; ii += 8) {
						for (int jj = 0; jj <= 8 && !has_solid; jj += 8) {
							for (int kk = 0; kk <= 8 && !has_solid; kk += 8) {
								auto particle_idx_pos = openvdb::Vec3i(i8 + ii, j8 + jj, k8 + kk);
								has_solid = (openvdb::tools::BoxSampler::sample(solid_axr, in_sdf->worldToIndex(in_out_particles->indexToWorld(particle_idx_pos))) < 0);
							}
						}
					}

					if (has_solid) {
						auto floor_leaf = in_out_particles->tree().probeLeaf(openvdb::Coord(i8,j8,k8));
						if (floor_leaf) {
							target_point_leaves.insert(std::make_pair(floor_leaf->origin(), floor_leaf));
						}
						else {
							auto newleaff = new openvdb::points::PointDataTree::LeafNodeType(openvdb::Coord(i8, j8, k8), 0, true);
							auto new_origin = newleaff->origin();
							bool success = target_point_leaves.insert(std::make_pair(newleaff->origin(), newleaff));
							if (!success) {
								delete newleaff;
							}
						}//end else found existing particle leaf
					}//end if touch solid
				}//loop k
			}//loop j
		}//loop i
	};//end mark active point elaves
	tbb::parallel_for(pispace_range, mark_active_point_leaves);

	//for parallel process
	std::vector<point_leaf_t*> leafarray;
	for (auto& i : target_point_leaves) {
		leafarray.push_back(i.second);
	}
	//printf("touched_leafs:%zd\n", leafarray.size());

	std::atomic<size_t> added_particles{ 0 };

	auto pnamepair = FLIP_vdb::position_attribute::attributeType();
	auto m_position_attribute_descriptor = openvdb::points::AttributeSet::Descriptor::create(pnamepair);
	
	auto vnamepair = FLIP_vdb::velocity_attribute::attributeType();
	auto m_pv_attribute_descriptor = m_position_attribute_descriptor->duplicateAppend("v", vnamepair);

	//use existing descriptors if we already have particles
	if (!in_out_particles->empty()) {
		for (auto it = in_out_particles->tree().cbeginLeaf(); it; ++it) {
			if (it->getLastValue() != 0) {
				//a non-empty leaf with attributes
				m_pv_attribute_descriptor = it->attributeSet().descriptorPtr();
				break;
			}
		}
	}

	//seed each leaf
	auto seed_leaf = [&](const tbb::blocked_range<size_t>& r) {
		if(use_vel_volume) 
		{
			auto in_sdf_axr{ in_sdf->getConstAccessor() };
			auto in_vel_axr{ in_vel->getConstAccessor() };
			float sdf_threshold = -dx * 0.55;

			std::random_device device;
			std::mt19937 generator(/*seed=*/device());
			std::uniform_real_distribution<> distribution(-0.5, 0.5);

			for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
				auto& leaf = *leafarray[ileaf];
				std::vector<openvdb::Vec3f> new_pos, new_vel;
				std::vector<openvdb::PointDataIndex32> new_idxend;

				//check if the last element is zero
				//if so, it's a new leaf
				if (leaf.getLastValue() == 0) {
					//initialize the attribute descriptor
					auto local_pv_descriptor = m_pv_attribute_descriptor;
					leaf.initializeAttributes(m_position_attribute_descriptor, 0);
					leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
				}

				//reader
				const openvdb::points::AttributeArray& positionArray =
					leaf.attributeArray("P");
				// Extract the velocity attribute from the leaf by name (v is velocity).
				const openvdb::points::AttributeArray& velocityArray =
					leaf.attributeArray("v");
				auto p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
				auto v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

				openvdb::Index32 emitted_particle = 0;
				for (auto offset = 0; offset < leaf.SIZE; ++offset) {
					openvdb::Index32 idxbegin = 0;
					if (offset != 0) {
						idxbegin = leaf.getValue(offset - 1);
					}
					openvdb::Index32 idxend = leaf.getValue(offset);


					//used to indicate sub_voxel(8 cells) occupancy by particles
					unsigned char sub_voxel_occupancy = 0;
					
					//first emit original particles
					openvdb::Index32 this_voxel_emitted = 0;
					for (auto idx = idxbegin; idx < idxend; ++idx) {
						auto p = p_handle_ptr->get(idx);
						unsigned char sv_pos = ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
						//only emit uniformly, otherwise skip it
						if (sub_voxel_occupancy & (1 << sv_pos)) {
							//that bit is active, there is already an particle
							//skip it
							continue;
						}
						sub_voxel_occupancy |= (1 << sv_pos);
						new_pos.push_back(p);
						new_vel.push_back(v_handle_ptr->get(idx));
						emitted_particle++;
						this_voxel_emitted++;
					}

					//emit to fill the sdf
					auto voxelwpos = in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
					auto voxelipos = in_sdf->worldToIndex(voxelwpos);
					if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < dx) {
						const int max_emit_trial = 16;
						for (int trial = 0; this_voxel_emitted < 8 && trial<max_emit_trial; trial++) {
							openvdb::Vec3d particle_pipos{ distribution(generator) ,distribution(generator) ,distribution(generator) };
							auto& p = particle_pipos;
							unsigned char sv_pos = ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
							//only emit uniformly, otherwise skip it
							if (sub_voxel_occupancy & (1 << sv_pos)) {
								//that bit is active, there is already an particle
								//skip it
								continue;
							}
							auto particlewpos = in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
							auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
							if (openvdb::tools::BoxSampler::sample(in_sdf_axr, particle_sdfipos) < sdf_threshold) {
								sub_voxel_occupancy |= 1 << sv_pos;
								new_pos.push_back(particle_pipos);
								new_vel.push_back(openvdb::tools::BoxSampler::sample(in_vel_axr,particle_sdfipos));
								emitted_particle++;
								this_voxel_emitted++;
								added_particles++;
							}
						}//end for 16 emit trials
					}//if voxel inside emitter

					new_idxend.push_back(emitted_particle);
				}//end emit original particles and new particles

				//set the new index ends, attributes
				//initialize the attribute descriptor
				auto local_pv_descriptor = m_pv_attribute_descriptor;
				leaf.initializeAttributes(m_position_attribute_descriptor, emitted_particle);
				leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
				leaf.setOffsets(new_idxend);

				//set the positions and velocities
				//get the new attribute arrays
				openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
				openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

				// Create read handles for position and velocity
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

				for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
					posWHandle.set(*iter, new_pos[*iter]);
					vWHandle.set(*iter, new_vel[*iter]);
				}//end for all on voxels
			}//end for range leaf
		} 
		else {
			auto in_sdf_axr{ in_sdf->getConstAccessor() };
			float sdf_threshold = -dx * 0.55;

			std::random_device device;
			std::mt19937 generator(/*seed=*/device());
			std::uniform_real_distribution<> distribution(-0.5, 0.5);

			for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
				auto& leaf = *leafarray[ileaf];
				std::vector<openvdb::Vec3f> new_pos, new_vel;
				std::vector<openvdb::PointDataIndex32> new_idxend;

				//check if the last element is zero
				//if so, it's a new leaf
				if (leaf.getLastValue() == 0) {
					//initialize the attribute descriptor
					auto local_pv_descriptor = m_pv_attribute_descriptor;
					leaf.initializeAttributes(m_position_attribute_descriptor, 0);
					leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
				}

				//reader
				const openvdb::points::AttributeArray& positionArray =
					leaf.attributeArray("P");
				// Extract the velocity attribute from the leaf by name (v is velocity).
				const openvdb::points::AttributeArray& velocityArray =
					leaf.attributeArray("v");
				auto p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec>::create(positionArray);
				auto v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec>::create(velocityArray);

				openvdb::Index32 emitted_particle = 0;
				for (auto offset = 0; offset < leaf.SIZE; ++offset) {
					openvdb::Index32 idxbegin = 0;
					if (offset != 0) {
						idxbegin = leaf.getValue(offset - 1);
					}
					openvdb::Index32 idxend = leaf.getValue(offset);


					//used to indicate sub_voxel(8 cells) occupancy by particles
					unsigned char sub_voxel_occupancy = 0;
					
					//first emit original particles
					openvdb::Index32 this_voxel_emitted = 0;
					for (auto idx = idxbegin; idx < idxend; ++idx) {
						auto p = p_handle_ptr->get(idx);
						unsigned char sv_pos = ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
						//only emit uniformly, otherwise skip it
						if (sub_voxel_occupancy & (1 << sv_pos)) {
							//that bit is active, there is already an particle
							//skip it
							continue;
						}
						sub_voxel_occupancy |= (1 << sv_pos);
						new_pos.push_back(p);
						new_vel.push_back(v_handle_ptr->get(idx));
						emitted_particle++;
						this_voxel_emitted++;
					}

					//emit to fill the sdf
					auto voxelwpos = in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
					auto voxelipos = in_sdf->worldToIndex(voxelwpos);
					if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < dx) {
						const int max_emit_trial = 16;
						for (int trial = 0; this_voxel_emitted < 8 && trial<max_emit_trial; trial++) {
							openvdb::Vec3d particle_pipos{ distribution(generator) ,distribution(generator) ,distribution(generator) };
							auto& p = particle_pipos;
							unsigned char sv_pos = ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
							//only emit uniformly, otherwise skip it
							if (sub_voxel_occupancy & (1 << sv_pos)) {
								//that bit is active, there is already an particle
								//skip it
								continue;
							}
							auto particlewpos = in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
							auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
							if (openvdb::tools::BoxSampler::sample(in_sdf_axr, particle_sdfipos) < sdf_threshold) {
								sub_voxel_occupancy |= 1 << sv_pos;
								new_pos.push_back(particle_pipos);
								new_vel.push_back(openvdb::Vec3d(vx, vy, vz));
								emitted_particle++;
								this_voxel_emitted++;
								added_particles++;
							}
						}//end for 16 emit trials
					}//if voxel inside emitter

					new_idxend.push_back(emitted_particle);
				}//end emit original particles and new particles

				//set the new index ends, attributes
				//initialize the attribute descriptor
				auto local_pv_descriptor = m_pv_attribute_descriptor;
				leaf.initializeAttributes(m_position_attribute_descriptor, emitted_particle);
				leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
				leaf.setOffsets(new_idxend);

				//set the positions and velocities
				//get the new attribute arrays
				openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
				openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

				// Create read handles for position and velocity
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> posWHandle(posarray);
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> vWHandle(varray);

				for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
					posWHandle.set(*iter, new_pos[*iter]);
					vWHandle.set(*iter, new_vel[*iter]);
				}//end for all on voxels
			}//end for range leaf
		}
	};//end seed particle 

	tbb::parallel_for(tbb::blocked_range<size_t>(0, leafarray.size()), seed_leaf);
	//printf("added particles:%zd\n", size_t(added_particles));
	//printf("original leaf count:%d\n", m_particles->tree().leafCount());
	for (auto leaf : leafarray) {
		if (leaf != in_out_particles->tree().probeLeaf(leaf->origin())) {
			in_out_particles->tree().addLeaf(leaf);
		}
	}
	//openvdb::io::File("dbg.vdb").write({ m_particles });
	//printf("new leafcount:%d\n", m_particles->tree().leafCount());
	in_out_particles->pruneGrid();
}




// void FLIP_vdb::update_solid_sdf()
// {
// 	using  namespace openvdb::tools::local_util;
	
// 	//touch all nodes in the m_solids that could potentially have solids
// 	for (auto& solidsdfptr : m_moving_solids) {
// 		for (auto leafiter = solidsdfptr->tree().cbeginLeaf(); leafiter; ++leafiter) {
// 			auto source_ipos = leafiter->origin();
// 			auto source_wpos = solidsdfptr->indexToWorld(source_ipos);
// 			auto target_ipos = openvdb::Coord(floorVec3(m_solid_sdf->worldToIndex(source_wpos)));
// 			m_solid_sdf->tree().touchLeaf(target_ipos);
// 		}
// 		/*auto toucher = solid_leaf_tile_toucher(m_solid_sdf, solidsdfptr);
// 		solidsdfptr->tree().visit(toucher);*/
// 	}

// 	//also retrieve all possible liquids trapped inside solid
// 	tbb::concurrent_vector<openvdb::Coord> neib_liquid_origins;

// 	auto collect_liquid_origins = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		for (auto& solidsdfptr : m_moving_solids) {
// 			auto axr{ solidsdfptr->getConstUnsafeAccessor() };
// 			for (int ii = 0; ii <= 8; ii += 8) {
// 				for (int jj = 0; jj <= 8; jj += 8) {
// 					for (int kk = 0; kk <= 8; kk += 8) {
// 						auto at_coord = leaf.origin().offsetBy(ii, jj, kk);
// 						auto wpos = m_particles->indexToWorld(at_coord);
// 						auto ipos = solidsdfptr->worldToIndex(wpos);
// 						if (openvdb::tools::BoxSampler::sample(axr, ipos) < 0) {
// 							neib_liquid_origins.push_back(leaf.origin());
// 							return;
// 						}
// 					}//end kk
// 				}//end jj
// 			}//end ii
// 		}//end for all solids
// 	};//end collect liquid origin
// 	auto partman = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(m_particles->tree());
// 	partman.foreach(collect_liquid_origins);

// 	//touch those liquid leafs as well
// 	for (auto coord : neib_liquid_origins) {
// 		m_solid_sdf->tree().touchLeaf(coord);
// 	}


// 	auto update_solid_op = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		for (const auto& solidsdfptr : m_moving_solids) {
// 			auto source_solid_axr{ solidsdfptr->getConstAccessor() };

// 			//get the sdf from the moving solids
// 			for (auto offset = 0; offset < leaf.SIZE; ++offset) {
// 				float current_sdf = leaf.getValue(offset);
// 				auto current_Ipos = leaf.offsetToGlobalCoord(offset);
// 				auto current_wpos = m_solid_sdf->indexToWorld(current_Ipos);
// 				//interpolate the value at the
// 				float source_sdf = openvdb::tools::BoxSampler::sample(
// 					source_solid_axr, solidsdfptr->worldToIndex(current_wpos));
// 				leaf.setValueOn(offset, std::min(current_sdf, source_sdf));
// 			}//end for target solid voxel
// 		}//end for all moving solids
// 	};//end update solid operator

// 	auto solid_leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_solid_sdf->tree());

// 	solid_leafman.foreach(update_solid_op);

// 	//merge the influence of the solid sdf
// 	for (auto iter = m_domain_solid_sdf->tree().cbeginLeaf(); iter; ++iter) {
// 		auto* leaf = m_solid_sdf->tree().touchLeaf(iter->origin());
// 		for (auto viter = iter->cbeginValueOn(); viter != iter->cendValueOn(); ++viter) {
// 			leaf->setValueOn(viter.offset(),
// 				std::min(leaf->getValue(viter.offset()), viter.getValue()));
// 		}
// 	}
// }
void FLIP_vdb::calculate_face_weights(
	openvdb::Vec3fGrid::Ptr face_weight, 
	openvdb::FloatGrid::Ptr liquid_sdf,
	openvdb::FloatGrid::Ptr solid_sdf)
{
	face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 1.0f });
	face_weight->setTree(std::make_shared<openvdb::Vec3fTree>(
		liquid_sdf->tree(), openvdb::Vec3f{ 1.0f }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(face_weight->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	face_weight->setName("Face_Weights");
	face_weight->setTransform(liquid_sdf->transformPtr());
	face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	auto set_face_weight_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto solid_axr{ solid_sdf->getConstAccessor() };
		//solid sdf
		float ssdf[2][2][2];
		openvdb::Vec3f uvwweight{ 1.f };
		for (auto offset = 0; offset < leaf.SIZE; offset++) {
			if (leaf.isValueMaskOff(offset)) {
				continue;
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (int kk = 0; kk < 2; kk++) {
						ssdf[ii][jj][kk] = solid_axr.getValue(leaf.offsetToGlobalCoord(offset) + openvdb::Coord{ ii,jj,kk });
					}
				}
			}//end retrieve eight solid sdfs on the voxel corners

			//fraction_inside(bl,br,tl,tr)
			//tl-----tr
			//|      |
			//|      |
			//bl-----br

			//look from positive x direction

			//   ^z
			//(0,0,1)------(0,1,1)
			//   |            |
			//(0,0,0)------(0,1,0)>y
			//uweight
			uvwweight[0] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][1][0],
				ssdf[0][0][1],
				ssdf[0][1][1]);
			uvwweight[0] = std::max(0.f, std::min(uvwweight[0], 1.f));

			//look from positive y direction
			//   ^x              
			//(1,0,0)------(1,0,1)
			//   |            |
			//(0,0,0)------(0,0,1)>z
			//vweight
			uvwweight[1] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][0][1],
				ssdf[1][0][0],
				ssdf[1][0][1]);
			uvwweight[1] = std::max(0.f, std::min(uvwweight[1], 1.f));

			//look from positive z direction
			//   ^y              
			//(0,1,0)------(1,1,0)
			//   |            |
			//(0,0,0)------(1,0,0)>x
			//wweight
			uvwweight[2] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[1][0][0],
				ssdf[0][1][0],
				ssdf[1][1][0]);
			uvwweight[2] = std::max(0.f, std::min(uvwweight[2], 1.f));
			leaf.setValueOn(offset, uvwweight);
		}//end for all offset
	};//end set face weight op

	auto leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(face_weight->tree());
	leafman.foreach(set_face_weight_op);
}
void FLIP_vdb::calculate_face_weights()
{
	m_face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 1.0f });
	m_face_weight->setTree(std::make_shared<openvdb::Vec3fTree>(
		m_liquid_sdf->tree(), openvdb::Vec3f{ 1.0f }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(m_face_weight->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	m_face_weight->setName("Face_Weights");
	m_face_weight->setTransform(m_liquid_sdf->transformPtr());
	m_face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	auto set_face_weight_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto solid_axr{ m_solid_sdf->getConstAccessor() };
		//solid sdf
		float ssdf[2][2][2];
		openvdb::Vec3f uvwweight{ 1.f };
		for (auto offset = 0; offset < leaf.SIZE; offset++) {
			if (leaf.isValueMaskOff(offset)) {
				continue;
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (int kk = 0; kk < 2; kk++) {
						ssdf[ii][jj][kk] = solid_axr.getValue(leaf.offsetToGlobalCoord(offset) + openvdb::Coord{ ii,jj,kk });
					}
				}
			}//end retrieve eight solid sdfs on the voxel corners

			//fraction_inside(bl,br,tl,tr)
			//tl-----tr
			//|      |
			//|      |
			//bl-----br

			//look from positive x direction

			//   ^z
			//(0,0,1)------(0,1,1)
			//   |            |
			//(0,0,0)------(0,1,0)>y
			//uweight
			uvwweight[0] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][1][0],
				ssdf[0][0][1],
				ssdf[0][1][1]);
			uvwweight[0] = std::max(0.f, std::min(uvwweight[0], 1.f));

			//look from positive y direction
			//   ^x              
			//(1,0,0)------(1,0,1)
			//   |            |
			//(0,0,0)------(0,0,1)>z
			//vweight
			uvwweight[1] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][0][1],
				ssdf[1][0][0],
				ssdf[1][0][1]);
			uvwweight[1] = std::max(0.f, std::min(uvwweight[1], 1.f));

			//look from positive z direction
			//   ^y              
			//(0,1,0)------(1,1,0)
			//   |            |
			//(0,0,0)------(1,0,0)>x
			//wweight
			uvwweight[2] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[1][0][0],
				ssdf[0][1][0],
				ssdf[1][1][0]);
			uvwweight[2] = std::max(0.f, std::min(uvwweight[2], 1.f));
			leaf.setValueOn(offset, uvwweight);
		}//end for all offset
	};//end set face weight op

	auto leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_face_weight->tree());
	leafman.foreach(set_face_weight_op);
}

void FLIP_vdb::clamp_liquid_phi_in_solids(openvdb::FloatGrid::Ptr liquid_sdf,
										  openvdb::FloatGrid::Ptr solid_sdf,
										  openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
										  float dx)
{
	openvdb::tools::dilateActiveValues(liquid_sdf->tree(), 
	1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	auto correct_liquid_phi_in_solid = [&](openvdb::FloatTree::LeafNodeType& leaf, 
	openvdb::Index leafpos) {
		//detech if there is solid
		if (solid_sdf->tree().probeConstLeaf(leaf.origin())) {
			auto const_solid_axr{ solid_sdf->getConstAccessor() };
			auto shift{ openvdb::Vec3R{0.5} };

			for (auto offset = 0; offset < leaf.SIZE; offset++) {
				if (leaf.isValueMaskOff(offset)) {
					continue;
				}
				auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
					solid_sdf->tree(), leaf.offsetToGlobalCoord(offset).asVec3d() + shift);

				if (voxel_solid_sdf<0 && leaf.getValue(offset) < (-voxel_solid_sdf)) {
					leaf.setValueOn(offset, -voxel_solid_sdf);
				}
			}//end for all voxel
		}//end there is solid in this leaf
	};//end correct_liquid_phi_in_solid

	auto phi_manager = openvdb::tree::LeafManager<openvdb::FloatTree>(liquid_sdf->tree());

	phi_manager.foreach(correct_liquid_phi_in_solid);

	pushed_out_liquid_sdf = liquid_sdf->deepCopy();

	//this is to be called by a solid manager
	auto immerse_liquid_into_solid = [&](openvdb::FloatTree::LeafNodeType& leaf, 
	openvdb::Index leafpos) {
		//detech if there is solid
		if (solid_sdf->tree().probeConstLeaf(leaf.origin())) {
			auto pushed_out_liquid_axr{ pushed_out_liquid_sdf->getConstAccessor() };
			auto const_solid_axr{ solid_sdf->getConstAccessor() };
			auto shift{ openvdb::Vec3R{0.5} };

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
						}
						else {
							test_coord[component]--;
						}
						found_liquid_neib |= (pushed_out_liquid_axr.getValue(test_coord) < 0);
					}//end for 6 direction

					//mark this voxel as liquid
					if (found_liquid_neib) {
						float current_sdf = iter.getValue();
						iter.setValue(current_sdf - dx * 1);
						//iter.setValue(-m_dx * 0.01);
						//iter.setValueOn();
					}
				}//end if this voxel is inside solid

			}//end for all voxel
		}//end there is solid in this leaf
	};//end immerse_liquid_into_solid

	phi_manager.foreach(immerse_liquid_into_solid);
}
void FLIP_vdb::clamp_liquid_phi_in_solids()
{
	//some particles can come very close to the solid boundary
	//in that case the liquid phi inside the solid may be marked as negative
	//we loop over the active solid leafs
	//and correct those liquid phi that are too small
	openvdb::tools::dilateActiveValues(m_liquid_sdf->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	auto correct_liquid_phi_in_solid = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		//detech if there is solid
		if (m_solid_sdf->tree().probeConstLeaf(leaf.origin())) {
			auto const_solid_axr{ m_solid_sdf->getConstAccessor() };
			auto shift{ openvdb::Vec3R{0.5} };

			for (auto offset = 0; offset < leaf.SIZE; offset++) {
				if (leaf.isValueMaskOff(offset)) {
					continue;
				}
				auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
					m_solid_sdf->tree(), leaf.offsetToGlobalCoord(offset).asVec3d() + shift);

				if (voxel_solid_sdf<0 && leaf.getValue(offset) < (-voxel_solid_sdf)) {
					leaf.setValueOn(offset, -voxel_solid_sdf);
				}
			}//end for all voxel
		}//end there is solid in this leaf
	};//end correct_liquid_phi_in_solid

	auto phi_manager = openvdb::tree::LeafManager<openvdb::FloatTree>(m_liquid_sdf->tree());

	phi_manager.foreach(correct_liquid_phi_in_solid);

	m_pushed_out_liquid_sdf = m_liquid_sdf->deepCopy();
	printf("immerse liquid\n");

	//this is to be called by a solid manager
	auto immerse_liquid_into_solid = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		//detech if there is solid
		if (m_solid_sdf->tree().probeConstLeaf(leaf.origin())) {
			auto pushed_out_liquid_axr{ m_pushed_out_liquid_sdf->getConstAccessor() };
			auto const_solid_axr{ m_solid_sdf->getConstAccessor() };
			auto shift{ openvdb::Vec3R{0.5} };

			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
					m_solid_sdf->tree(), leaf.offsetToGlobalCoord(iter.offset()).asVec3d() + shift);

				if (voxel_solid_sdf < 0) {
					bool found_liquid_neib = false;
					for (int i_neib = 0; i_neib < 6 && !found_liquid_neib; i_neib++) {
						int component = i_neib / 2;
						int positive_dir = (i_neib % 2 == 0);

						auto test_coord = iter.getCoord();

						if (positive_dir) {
							test_coord[component]++;
						}
						else {
							test_coord[component]--;
						}
						found_liquid_neib |= (pushed_out_liquid_axr.getValue(test_coord) < 0);
					}//end for 6 direction

					//mark this voxel as liquid
					if (found_liquid_neib) {
						float current_sdf = iter.getValue();
						iter.setValue(current_sdf - m_dx * 1);
						//iter.setValue(-m_dx * 0.01);
						//iter.setValueOn();
					}
				}//end if this voxel is inside solid

			}//end for all voxel
		}//end there is solid in this leaf
	};//end immerse_liquid_into_solid

	phi_manager.foreach(immerse_liquid_into_solid);
}

void FLIP_vdb::set_solid_velocity()
{
	m_solid_velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0 });
	m_solid_velocity->setTransform(m_voxel_center_transform);
	m_solid_velocity->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	m_solid_velocity->setName("Solid_Velocity");

	for (auto iter = m_solid_sdf->tree().cbeginLeaf(); iter; ++iter) {
		m_solid_velocity->tree().touchLeaf(iter->origin());
	}

	auto solid_vel_set_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto weight_axr{ m_face_weight->getConstAccessor() };
		for (auto offset = 0; offset < leaf.SIZE; ++offset) {
			if (weight_axr.getValue(leaf.offsetToGlobalCoord(offset)) != openvdb::Vec3f{ 1 }) {
				//has solid component
				auto gcoord = leaf.offsetToGlobalCoord(offset);
				auto sampled_vel = openvdb::tools::BoxSampler::sample(
					m_boundary_velocity_volume->tree(),
					m_boundary_velocity_volume->worldToIndex(
						m_solid_velocity->indexToWorld(gcoord)));
				leaf.setValueOn(offset, sampled_vel);
			}
		}

	};//end solid vel set op

	auto leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_solid_velocity->tree());

	leafman.foreach(solid_vel_set_op);

	auto set_moving_solid_vel = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		//check if it is close to a moving objects
		//if so, set its velocity to the object's velocity, because the velocity volume may
		//set the incorrect solid velocity
		for (auto solid : m_moving_solids) {
			auto solid_axr(solid->getConstAccessor());

			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				auto wpos = m_solid_velocity->indexToWorld(iter.getCoord());
				auto ipos = solid->worldToIndex(wpos);
				if (openvdb::tools::BoxSampler::sample(solid_axr,ipos) <=2*m_dx) {
					iter.setValue(openvdb::Vec3f(0, 0, 0));
				}
			}
		}
	};
	leafman.foreach(set_moving_solid_vel);
}

namespace {
	//return the total number of degree of freedom
	//and set the degree of freedom tree
	openvdb::Int32 fill_idx_tree_from_liquid_sdf(openvdb::Int32Tree::Ptr dof_tree, openvdb::FloatGrid::Ptr liquid_sdf_tree) {

		auto dof_leafman = openvdb::tree::LeafManager<openvdb::Int32Tree>(*dof_tree);

		//first count how many dof in each leaf
		//then assign the global dof id
		std::vector<openvdb::Int32> dof_end_in_each_leaf;
		dof_end_in_each_leaf.assign(liquid_sdf_tree->tree().leafCount(), 0);

		//mark the active state of dofs
		auto mark_dof_active_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
			auto* sdf_leaf = liquid_sdf_tree->tree().probeConstLeaf(leaf.origin());
			leaf.setValueMask(sdf_leaf->getValueMask());
			for (auto iter = leaf.cbeginValueOn(); iter != leaf.cendValueOn(); ++iter) {
				//mark the position where liquid sdf non-negative as inactive
				if (sdf_leaf->getValue(iter.offset()) >= 0) {
					leaf.setValueOff(iter.offset());
				}
			}
		};//end mark_dof_active_op
		dof_leafman.foreach(mark_dof_active_op);

		auto leaf_active_dof_counter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
			dof_end_in_each_leaf[leafpos] = leaf.onVoxelCount();
		};//end leaf active dof counter
		dof_leafman.foreach(leaf_active_dof_counter);

		//scan through all leaves to determine
		for (size_t i = 1; i < dof_end_in_each_leaf.size(); i++) {
			dof_end_in_each_leaf[i] += dof_end_in_each_leaf[i - 1];
		}

		auto set_dof_id = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
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
		dof_leafman.foreach(set_dof_id);

		//return the total number of degree of freedom
		return *dof_end_in_each_leaf.crbegin();
	}//end fill idx tree from liquid sdf


}
void FLIP_vdb::apply_pressure_gradient(
	openvdb::FloatGrid::Ptr liquid_sdf,
	openvdb::FloatGrid::Ptr solid_sdf,
	openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
	openvdb::FloatGrid::Ptr pressure,
	openvdb::Vec3fGrid::Ptr face_weight,
	openvdb::Vec3fGrid::Ptr velocity,
	openvdb::Vec3fGrid::Ptr solid_velocity,
	float dx,float in_dt)
{
	Eigen::Matrix<float, 4, 4> invmat;
	float invdx2 = 1.0f / (dx * dx);
	invmat.col(0) = invdx2 * Eigen::Vector4f{ 1.0 / 8.0, 0, 0, 0 };
	invmat.col(1) = invdx2 * Eigen::Vector4f{ 0, 1.0f / 3.0f ,0 ,-1.0f / 6.0f };
	invmat.col(2) = invdx2 * Eigen::Vector4f{ 0, 0, 1.0f / 3.0f ,-1.0f / 6.0f };
	invmat.col(3) = invdx2 * Eigen::Vector4f{ 0, -1.0f / 6.0f, -1.0f / 6.0f, 1.0f / 4.0f };

	Eigen::Matrix<float, 4, 12> mT;
	for (int i = -1; i <=1; i++) {
		float fi = float(i);
		mT.block<1, 4>(0, 4 * (i + 1)) = Eigen::Vector4f{ fi,fi,fi,fi };
		mT.block<1, 4>(1, 4 * (i + 1)) = Eigen::Vector4f{ 0,1,1,0 };
		mT.block<1, 4>(2, 4 * (i + 1)) = Eigen::Vector4f{ 0,0,1,1 };
		mT.block<1, 4>(3, 4 * (i + 1)) = Eigen::Vector4f{ 1,1,1,1 };
	}
	mT = mT * dx;

	//given the solved pressure, update the velocity
	//this is to be used by the velocity(post_pressure) leaf manager
	auto velocity_update_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, 
	openvdb::Index leafpos) {
		auto phi_axr{ liquid_sdf->getConstAccessor() };
		auto true_phi_axr{ pushed_out_liquid_sdf->getConstAccessor() };
		auto weight_axr{ face_weight->getConstAccessor() };
		auto solid_vel_axr{ solid_velocity->getConstAccessor() };
		auto solid_sdf_axr{ solid_sdf->getConstAccessor() };
		auto pressure_axr{ pressure->getConstAccessor() };
		//auto update_axr{ m_velocity_update->getAccessor() };

		

		for (auto offset = 0; offset < leaf.SIZE; offset++) {
			if (leaf.isValueMaskOff(offset)) {
				continue;
			}
			//we are looking at a velocity sample previously obtained from p2g
			//possibly with extrapolation

			//set this velocity as off unless we have an update
			//on any component of its velocity
			bool has_any_update = false;
			auto gcoord = leaf.offsetToGlobalCoord(offset);
			auto original_vel = leaf.getValue(offset);
			auto solid_vel = solid_vel_axr.getValue(gcoord);

			//face_weight=0 in solid
			auto face_weights = weight_axr.getValue(gcoord);
			auto vel_update = 0*original_vel;
			//three velocity channel
			for (int ic = 0; ic < 3; ic++) {
				auto lower_gcoord = gcoord;
				lower_gcoord[ic] -= 1;
				if (face_weights[ic] > 0) {
					//this face has liquid component
					//does it have any dof on its side?
					auto phi_this = phi_axr.getValue(gcoord);
					auto phi_below = phi_axr.getValue(lower_gcoord);
					float p_this = pressure_axr.getValue(gcoord);
					float p_below = pressure_axr.getValue(lower_gcoord);

					bool update_this_velocity = false;
					if (phi_this < 0 && phi_below < 0) {
						update_this_velocity = true;
					}
					else {
						if (phi_this > 0 && phi_below > 0) {
							update_this_velocity = false;
						} // end if all outside liquid
						else {
							//one of them is inside the liquid, one is outside
							if (phi_this >= 0) {
								//this point is outside the liquid, possibly free air
								if (openvdb::tools::BoxSampler::sample(solid_sdf_axr, gcoord.asVec3d()+openvdb::Vec3d(0.5)) > 0) {
									//if so, set the pressure to be ghost value
									if (true_phi_axr.getValue(lower_gcoord) < 0) {
										update_this_velocity = true;
									}
									//p_this = p_below / std::min(phi_below, - m_dx*1e-3f) * phi_this;
								}
							}

							if (phi_below >= 0) {
								//this point is outside the liquid, possibly free air
								if (openvdb::tools::BoxSampler::sample(solid_sdf_axr, lower_gcoord.asVec3d() + openvdb::Vec3d(0.5)) > 0) {
									//if so, set the pressure to be ghost value
									if (true_phi_axr.getValue(gcoord) < 0) {
										update_this_velocity = true;
									}
									//p_below = p_this / std::min(phi_this, - m_dx * 1e-3f) * phi_below;
								}
							}
						} //end else all outside liquid
					}//end all inside liquid

					if (update_this_velocity) {
						float theta = 1;
						if (phi_this >= 0 || phi_below >= 0) {
							theta = fraction_inside(phi_below, phi_this);
							if (theta < 0.02f) theta = 0.02f;
						}
							
						original_vel[ic] -= in_dt * (float)(p_this - p_below) / dx / theta;
						vel_update[ic] = in_dt * (float)(p_this - p_below) / dx / theta;
						if (face_weights[ic] < 1) {
							//use the voxel center solid sdf gradient
							openvdb::Vec3f grad_solid{ 0 };
							openvdb::Vec3f evalpos{ 0.5 };

							//evaluate the four sdf on this face
							Eigen::VectorXf neib_sdf = Eigen::VectorXf::Zero(12);


							//four_sdf[0] = solid_sdf_axr.getValue(gcoord);
							//calculate the in-plane solid normal
							//then deduce the off-plane solid normal 
							for (int iclevel = -1; iclevel <= 1; iclevel++) {
								int ic4 = (iclevel + 1) * 4;
								switch (ic) {
								case 0:
									//   ^z
									//(0,0,1)------(0,1,1)
									//   |            |
									//(0,0,0)------(0,1,0)>y
									//u
									neib_sdf[0 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 0, 0));
									neib_sdf[1 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 1, 0));
									neib_sdf[2 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 1, 1));
									neib_sdf[3 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 0, 1));
									break;
								case 1:
									//   ^x              
									//(1,0,0)------(1,0,1)
									//   |            |
									//(0,0,0)------(0,0,1)>z
									//v
									neib_sdf[0 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, iclevel, 0));
									neib_sdf[1 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, iclevel, 1));
									neib_sdf[2 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, iclevel, 1));
									neib_sdf[3 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, iclevel, 0));
									break;
								case 2:
									//   ^y              
									//(0,1,0)------(1,1,0)
									//   |            |
									//(0,0,0)------(1,0,0)>x
									//w
									neib_sdf[0 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, 0, iclevel));
									neib_sdf[1 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, 0, iclevel));
									neib_sdf[2 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, 1, iclevel));
									neib_sdf[3 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, 1, iclevel));
									break;
								}
							}
							

							Eigen::Vector4f abcd = invmat * mT * neib_sdf;
							float normal_a = std::max(-1.f, std::min(1.f, abcd[0]));

							//make sure on the normal direction, the velocity has the same component

							openvdb::Vec3f facev{ 0,0,0 };
							grad_solid[ic] = normal_a;
							facev[ic] = original_vel[ic];
							if (1||(original_vel[ic]-solid_vel[ic]) * normal_a < 0) {
								facev -= grad_solid * grad_solid.dot(facev);
								facev += grad_solid * grad_solid.dot(solid_vel);
							}
							original_vel[ic] = facev[ic];
							/*float penetrating = grad_solid[ic] * original_vel[ic];
								original_vel[ic] -= penetrating;
								original_vel[ic] += grad_solid[ic] * solid_vel[ic];
							}*/

							//mix the solid velocity and fluid velocity
							float solid_fraction = (1 - face_weights[ic])*0.8;
							original_vel[ic] = (1-solid_fraction) * original_vel[ic] + (solid_fraction) * solid_vel[ic];
						}
						has_any_update = true;
					}//end if any dofs on two sides
				}//end if face_weight[ic]>0
				else {
					//this face is inside solid
					//just let it be the solid velocity
					//original_vel[ic] = solid_vel[ic];
					//has_any_update = true;
				}//end else face_weight[ic]>0
			}//end for three component

			if (!has_any_update) {
				leaf.setValueOff(offset, openvdb::Vec3f{ 0,0,0 });
				//update_axr.setValueOff(leaf.offsetToGlobalCoord(offset));
			}
			else {
				leaf.setValueOn(offset, original_vel);
				//update_axr.setValue(leaf.offsetToGlobalCoord(offset), vel_update);
			}
		}//end for all voxels
	};//end velocity_update_op

	auto vel_leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity->tree());
	vel_leafman.foreach(velocity_update_op);
}
void FLIP_vdb::apply_pressure_gradient(float in_dt)
{
	//m_velocity_update = m_velocity->deepCopy();
	//m_velocity_update->setName("Velocity_Update");



	Eigen::Matrix<float, 4, 4> invmat;
	float invdx2 = 1.0f / (m_dx * m_dx);
	invmat.col(0) = invdx2 * Eigen::Vector4f{ 1.0 / 8.0, 0, 0, 0 };
	invmat.col(1) = invdx2 * Eigen::Vector4f{ 0, 1.0f / 3.0f ,0 ,-1.0f / 6.0f };
	invmat.col(2) = invdx2 * Eigen::Vector4f{ 0, 0, 1.0f / 3.0f ,-1.0f / 6.0f };
	invmat.col(3) = invdx2 * Eigen::Vector4f{ 0, -1.0f / 6.0f, -1.0f / 6.0f, 1.0f / 4.0f };

	Eigen::Matrix<float, 4, 12> mT;
	for (int i = -1; i <=1; i++) {
		float fi = float(i);
		mT.block<1, 4>(0, 4 * (i + 1)) = Eigen::Vector4f{ fi,fi,fi,fi };
		mT.block<1, 4>(1, 4 * (i + 1)) = Eigen::Vector4f{ 0,1,1,0 };
		mT.block<1, 4>(2, 4 * (i + 1)) = Eigen::Vector4f{ 0,0,1,1 };
		mT.block<1, 4>(3, 4 * (i + 1)) = Eigen::Vector4f{ 1,1,1,1 };
	}
	mT = mT * m_dx;

	//given the solved pressure, update the velocity
	//this is to be used by the velocity(post_pressure) leaf manager
	auto velocity_update_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto phi_axr{ m_liquid_sdf->getConstAccessor() };
		auto true_phi_axr{ m_pushed_out_liquid_sdf->getConstAccessor() };
		auto weight_axr{ m_face_weight->getConstAccessor() };
		auto solid_vel_axr{ m_solid_velocity->getConstAccessor() };
		auto solid_sdf_axr{ m_solid_sdf->getConstAccessor() };
		auto pressure_axr{ m_pressure->getConstAccessor() };
		//auto update_axr{ m_velocity_update->getAccessor() };


		for (auto offset = 0; offset < leaf.SIZE; offset++) {
			if (leaf.isValueMaskOff(offset)) {
				continue;
			}
			//we are looking at a velocity sample previously obtained from p2g
			//possibly with extrapolation

			//set this velocity as off unless we have an update
			//on any component of its velocity
			bool has_any_update = false;
			auto gcoord = leaf.offsetToGlobalCoord(offset);
			auto original_vel = leaf.getValue(offset);
			auto solid_vel = solid_vel_axr.getValue(gcoord);

			//face_weight=0 in solid
			auto face_weights = weight_axr.getValue(gcoord);
			auto vel_update = 0*original_vel;
			//three velocity channel
			for (int ic = 0; ic < 3; ic++) {
				auto lower_gcoord = gcoord;
				lower_gcoord[ic] -= 1;
				if (face_weights[ic] > 0) {
					//this face has liquid component
					//does it have any dof on its side?
					auto phi_this = phi_axr.getValue(gcoord);
					auto phi_below = phi_axr.getValue(lower_gcoord);
					float p_this = pressure_axr.getValue(gcoord);
					float p_below = pressure_axr.getValue(lower_gcoord);

					bool update_this_velocity = false;
					if (phi_this < 0 && phi_below < 0) {
						update_this_velocity = true;
					}
					else {
						if (phi_this > 0 && phi_below > 0) {
							update_this_velocity = false;
						} // end if all outside liquid
						else {
							//one of them is inside the liquid, one is outside
							if (phi_this >= 0) {
								//this point is outside the liquid, possibly free air
								if (openvdb::tools::BoxSampler::sample(solid_sdf_axr, gcoord.asVec3d()+openvdb::Vec3d(0.5)) > 0) {
									//if so, set the pressure to be ghost value
									if (true_phi_axr.getValue(lower_gcoord) < 0) {
										update_this_velocity = true;
									}
									//p_this = p_below / std::min(phi_below, - m_dx*1e-3f) * phi_this;
								}
							}

							if (phi_below >= 0) {
								//this point is outside the liquid, possibly free air
								if (openvdb::tools::BoxSampler::sample(solid_sdf_axr, lower_gcoord.asVec3d() + openvdb::Vec3d(0.5)) > 0) {
									//if so, set the pressure to be ghost value
									if (true_phi_axr.getValue(gcoord) < 0) {
										update_this_velocity = true;
									}
									//p_below = p_this / std::min(phi_this, - m_dx * 1e-3f) * phi_below;
								}
							}
						} //end else all outside liquid
					}//end all inside liquid

					if (update_this_velocity) {
						float theta = 1;
						if (phi_this >= 0 || phi_below >= 0) {
							theta = fraction_inside(phi_below, phi_this);
							if (theta < 0.02f) theta = 0.02f;
						}
							
						original_vel[ic] -= in_dt * (float)(p_this - p_below) / m_dx / theta;
						vel_update[ic] = in_dt * (float)(p_this - p_below) / m_dx / theta;
						if (face_weights[ic] < 1) {
							//use the voxel center solid sdf gradient
							openvdb::Vec3f grad_solid{ 0 };
							openvdb::Vec3f evalpos{ 0.5 };

							//evaluate the four sdf on this face
							Eigen::VectorXf neib_sdf = Eigen::VectorXf::Zero(12);


							//four_sdf[0] = solid_sdf_axr.getValue(gcoord);
							//calculate the in-plane solid normal
							//then deduce the off-plane solid normal 
							for (int iclevel = -1; iclevel <= 1; iclevel++) {
								int ic4 = (iclevel + 1) * 4;
								switch (ic) {
								case 0:
									//   ^z
									//(0,0,1)------(0,1,1)
									//   |            |
									//(0,0,0)------(0,1,0)>y
									//u
									neib_sdf[0 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 0, 0));
									neib_sdf[1 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 1, 0));
									neib_sdf[2 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 1, 1));
									neib_sdf[3 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(iclevel, 0, 1));
									break;
								case 1:
									//   ^x              
									//(1,0,0)------(1,0,1)
									//   |            |
									//(0,0,0)------(0,0,1)>z
									//v
									neib_sdf[0 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, iclevel, 0));
									neib_sdf[1 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, iclevel, 1));
									neib_sdf[2 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, iclevel, 1));
									neib_sdf[3 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, iclevel, 0));
									break;
								case 2:
									//   ^y              
									//(0,1,0)------(1,1,0)
									//   |            |
									//(0,0,0)------(1,0,0)>x
									//w
									neib_sdf[0 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, 0, iclevel));
									neib_sdf[1 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, 0, iclevel));
									neib_sdf[2 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(1, 1, iclevel));
									neib_sdf[3 + ic4] = solid_sdf_axr.getValue(gcoord.offsetBy(0, 1, iclevel));
									break;
								}
							}
							

							Eigen::Vector4f abcd = invmat * mT * neib_sdf;
							float normal_a = std::max(-1.f, std::min(1.f, abcd[0]));

							//make sure on the normal direction, the velocity has the same component

							openvdb::Vec3f facev{ 0,0,0 };
							grad_solid[ic] = normal_a;
							facev[ic] = original_vel[ic];
							if (1||(original_vel[ic]-solid_vel[ic]) * normal_a < 0) {
								facev -= grad_solid * grad_solid.dot(facev);
								facev += grad_solid * grad_solid.dot(solid_vel);
							}
							original_vel[ic] = facev[ic];
							/*float penetrating = grad_solid[ic] * original_vel[ic];
								original_vel[ic] -= penetrating;
								original_vel[ic] += grad_solid[ic] * solid_vel[ic];
							}*/

							//mix the solid velocity and fluid velocity
							float solid_fraction = (1 - face_weights[ic])*0.8;
							original_vel[ic] = (1-solid_fraction) * original_vel[ic] + (solid_fraction) * solid_vel[ic];
						}
						has_any_update = true;
					}//end if any dofs on two sides
				}//end if face_weight[ic]>0
				else {
					//this face is inside solid
					//just let it be the solid velocity
					//original_vel[ic] = solid_vel[ic];
					//has_any_update = true;
				}//end else face_weight[ic]>0
			}//end for three component

			if (!has_any_update) {
				leaf.setValueOff(offset, openvdb::Vec3f{ 0,0,0 });
				//update_axr.setValueOff(leaf.offsetToGlobalCoord(offset));
			}
			else {
				leaf.setValueOn(offset, original_vel);
				//update_axr.setValue(leaf.offsetToGlobalCoord(offset), vel_update);
			}
		}//end for all voxels
	};//end velocity_update_op

	auto vel_leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_velocity->tree());
	vel_leafman.foreach(velocity_update_op);
}
void FLIP_vdb::solve_pressure_simd(
	openvdb::FloatGrid::Ptr liquid_sdf,
	openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
	openvdb::FloatGrid::Ptr rhsgrid,
	openvdb::FloatGrid::Ptr curr_pressure,
	openvdb::Vec3fGrid::Ptr face_weight,
	openvdb::Vec3fGrid::Ptr velocity,
	openvdb::Vec3fGrid::Ptr solid_velocity,
	float dt, float dx)
{
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/buildlevel").start();
	auto simd_solver = simd_vdb_poisson(liquid_sdf, 
	pushed_out_liquid_sdf, 
	face_weight, velocity, 
	solid_velocity, 
	dt, dx);

	simd_solver.construct_levels();
	simd_solver.build_rhs();
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/buildlevel").stop();

	

	auto pressure = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
	pressure->setName("Pressure");

	//use the previous pressure as warm start
	auto set_warm_pressure = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto old_pressure_axr{ curr_pressure->getConstAccessor() };
		auto* new_pressure_leaf = pressure->tree().probeLeaf(leaf.origin());

		for (auto iter = new_pressure_leaf->beginValueOn(); iter; ++iter) {
			float old_pressure = old_pressure_axr.getValue(iter.getCoord());
			if (std::isfinite(old_pressure)) {
				iter.setValue(old_pressure);
			}

		}
	};//end set_warm_pressure

	//simd_solver.m_laplacian_with_levels[0]->m_dof_leafmanager->foreach(set_warm_pressure);

	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/simdpcg").start();
	simd_solver.pcg_solve(pressure, 1e-7);
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/simdpcg").stop();

	curr_pressure.swap(pressure);
	// CSim::TimerMan::timer("Sim.step/vdbflip/pressure/updatevel").start();
	// apply_pressure_gradient(dt);
	// CSim::TimerMan::timer("Sim.step/vdbflip/pressure/updatevel").stop();

	//simd_solver.build_rhs();
	rhsgrid = simd_solver.m_rhs;
	//m_rhsgrid = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
	//simd_solver.m_laplacian_with_levels[0]->set_grid_constant_assume_topo(m_rhsgrid, 1);
	//simd_solver.m_laplacian_with_levels[0]->Laplacian_apply_assume_topo(m_rhsgrid, m_rhsgrid->deepCopy());
	//m_rhsgrid = simd_solver.m_laplacian_with_levels[0]->m_Neg_z_entry;
	rhsgrid->setName("RHS");
}
void FLIP_vdb::solve_pressure_simd(float dt)
{
	//test construct levels
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/buildlevel").start();
	auto simd_solver = simd_vdb_poisson(m_liquid_sdf, 
	m_pushed_out_liquid_sdf, 
	m_face_weight, m_velocity, 
	m_solid_velocity, 
	dt, m_dx);

	simd_solver.construct_levels();
	simd_solver.build_rhs();
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/buildlevel").stop();

	

	auto pressure = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
	pressure->setName("Pressure");

	//use the previous pressure as warm start
	auto set_warm_pressure = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto old_pressure_axr{ m_pressure->getConstAccessor() };
		auto* new_pressure_leaf = pressure->tree().probeLeaf(leaf.origin());

		for (auto iter = new_pressure_leaf->beginValueOn(); iter; ++iter) {
			float old_pressure = old_pressure_axr.getValue(iter.getCoord());
			if (std::isfinite(old_pressure)) {
				iter.setValue(old_pressure);
			}

		}
	};//end set_warm_pressure

	//simd_solver.m_laplacian_with_levels[0]->m_dof_leafmanager->foreach(set_warm_pressure);

	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/simdpcg").start();
	simd_solver.pcg_solve(pressure, 1e-7);
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/simdpcg").stop();

	m_pressure.swap(pressure);
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/updatevel").start();
	apply_pressure_gradient(dt);
	CSim::TimerMan::timer("Sim.step/vdbflip/pressure/updatevel").stop();

	//simd_solver.build_rhs();
	m_rhsgrid = simd_solver.m_rhs;
	//m_rhsgrid = simd_solver.m_laplacian_with_levels[0]->get_zero_vec_grid();
	//simd_solver.m_laplacian_with_levels[0]->set_grid_constant_assume_topo(m_rhsgrid, 1);
	//simd_solver.m_laplacian_with_levels[0]->Laplacian_apply_assume_topo(m_rhsgrid, m_rhsgrid->deepCopy());
	//m_rhsgrid = simd_solver.m_laplacian_with_levels[0]->m_Neg_z_entry;
	m_rhsgrid->setName("RHS");

	//simd_solver.symmetry_test(0);
}//end solve poisson simd

void FLIP_vdb::field_add_vector(openvdb::Vec3fGrid::Ptr velocity_field,
	openvdb::Vec3fGrid::Ptr face_weight,
	float x, float y, float z, float dt)
{
	
	auto add_gravity = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto face_weight_axr{ face_weight->getConstAccessor() };
		for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
			if (face_weight_axr.getValue(iter.getCoord())[1] > 0) {
				iter.modifyValue([&](openvdb::Vec3f& v) {
					v[0] += dt * x;
					v[1] += dt * y; 
					v[2] += dt * z;
					});
			}
		}
		
	};
	auto velman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity_field->tree());
	velman.foreach(add_gravity);
}
void FLIP_vdb::apply_body_force(float dt)
{
	m_acceleration_fields = m_velocity->deepCopy();
	openvdb::Vec3f acc{ 0 };
	auto normacc = acc.length();
	auto set_acceleration_field = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		leaf.fill(openvdb::Vec3f(0, 0, 0));
		for (auto propeller : m_propeller_sdf) {
			auto propaxr{ propeller->getConstAccessor() };
			bool propeller_not_found = true;
			for (int ii = 0; ii <= 8 && propeller_not_found; ii += 4) {
				for (int jj = 0; jj <= 8 && propeller_not_found; jj += 4) {
					for (int kk = 0; kk <= 8 && propeller_not_found; kk += 4) {
						auto wpos = m_acceleration_fields->indexToWorld(leaf.origin().offsetBy(ii, jj, kk));
						auto ipos = propeller->worldToIndex(wpos);
						if (openvdb::tools::BoxSampler::sample(propaxr,ipos)<0) {
							propeller_not_found = false;
						}
					}//end kk
				}//end jj
			}//end ii

			if (!propeller_not_found) {
				//there is propeller in this region
				//write the acceleration field
				std::random_device device;
				std::mt19937 generator(/*seed=*/device());
				std::normal_distribution<> distribution(0, 0.7);
				for (auto iter = leaf.beginValueOn(); iter; ++iter) {
					auto wpos = m_acceleration_fields->indexToWorld(iter.getCoord());
					auto ipos = propeller->worldToIndex(wpos);
					if (openvdb::tools::BoxSampler::sample(propaxr, ipos) < 0) {
						iter.setValue(acc + openvdb::Vec3f(normacc*distribution(device)));
					}
				}
			}
		}//end for all propeller
	};
	auto accman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_acceleration_fields->tree());
	accman.foreach(set_acceleration_field);


	auto add_gravity = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto face_weight_axr{ m_face_weight->getConstAccessor() };
		for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
			if (face_weight_axr.getValue(iter.getCoord())[1] > 0) {
				iter.modifyValue([dt](openvdb::Vec3f& v) {v[1] -= dt * 9.81; });
			}
		}
		auto accaxr{ m_acceleration_fields->getConstAccessor() };
		for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
			if (face_weight_axr.getValue(iter.getCoord())[1] > 0) {
				iter.modifyValue([&](openvdb::Vec3f& v) {v += dt * accaxr.getValue(iter.getCoord()); });
			}
		}
	};

	auto velman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_velocity->tree());
	velman.foreach(add_gravity);
}

float FLIP_vdb::cfl()
{
	std::vector<float> max_v_per_leaf;
	max_v_per_leaf.assign(m_velocity->tree().leafCount(), 0);

	auto find_per_leaf_max_vel = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
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
	};// end find per leaf max vel

	auto leafman = openvdb::tree::LeafManager <openvdb::Vec3fTree>(m_velocity->tree());
	leafman.foreach(find_per_leaf_max_vel);

	float max_v = 0;
	if (max_v_per_leaf.empty()) {
		return std::numeric_limits<float>::max() / 2;
	}
	max_v = max_v_per_leaf[0];

	//dont take the maximum number
	//take the 90%?
	int nleaf = max_v_per_leaf.size();
	int top90 = nleaf * 99 / 100;
	std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + nleaf - 1, max_v_per_leaf.end());
	std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + top90, max_v_per_leaf.end());

	/*for (const auto& v : max_v_per_leaf) {
		if (v > max_v) {
			max_v = v;
		}
	}*/
	max_v = max_v_per_leaf[nleaf-1];
	printf("max velocity component:%f\n", max_v_per_leaf.back());
	printf("cfl velocity:%f\n", max_v);
	return m_dx / (std::abs(max_v) + 1e-6f);
}

float FLIP_vdb::cfl_and_regularize_velocity()
{
	std::vector<float> max_v_per_leaf;
	max_v_per_leaf.assign(m_velocity->tree().leafCount(), 0);

	if (max_v_per_leaf.empty()) {
		return std::numeric_limits<float>::max()/2;
	}

	auto find_per_leaf_max_vel = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
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
	};// end find per leaf max vel

	auto leafman = openvdb::tree::LeafManager <openvdb::Vec3fTree>(m_velocity->tree());
	leafman.foreach(find_per_leaf_max_vel);

	float max_v = 0;
	
	
	max_v = max_v_per_leaf[0];

	//dont take the maximum number
	//take the 90%?
	int nleaf = max_v_per_leaf.size();
	int top90 = nleaf * 99 / 100;
	std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + nleaf - 1, max_v_per_leaf.end());
	std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + top90, max_v_per_leaf.end());

	/*for (const auto& v : max_v_per_leaf) {
		if (v > max_v) {
			max_v = v;
		}
	}*/
	max_v = max_v_per_leaf[top90];
	float max_allowed_vel = max_v * 2.0f;

	auto regularize_vel = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		float max_v = 0;
		auto old_vel_leaf = m_velocity_after_p2g->tree().probeLeaf(leaf.origin());
		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
			openvdb::Vec3f current_val = iter.getValue();
			for (int ic = 0; ic < 3; ic++) {
				if (current_val[ic] > max_allowed_vel) {
					current_val[ic] = max_allowed_vel;
				}
				if (current_val[ic] < -max_allowed_vel) {
					current_val[ic] = -max_allowed_vel;
				}
			}
			iter.setValue(current_val);
		}//end for all voxel

		if (old_vel_leaf) {
			for (auto iter = old_vel_leaf->beginValueOn(); iter; ++iter) {
				openvdb::Vec3f current_val = iter.getValue();
				for (int ic = 0; ic < 3; ic++) {
					if (current_val[ic] > max_allowed_vel) {
						current_val[ic] = max_allowed_vel;
					}
					if (current_val[ic] < -max_allowed_vel) {
						current_val[ic] = -max_allowed_vel;
					}
				}
				iter.setValue(current_val);
			}//end for all voxel
		}//end old velocity leaf

		max_v_per_leaf[leafpos] = max_v;
	};// end regularize_vel

	leafman.foreach(regularize_vel);
	return cfl();
}

//identify boundary particle voxels
//and only output n layer of particles with in this band
//this will reduce the output particle size
//this also generate the shrinked sdf object
// openvdb::points::PointDataGrid::Ptr FLIP_vdb::narrow_band_particles()
// {
// 	using point_leaf_t = openvdb::points::PointDataTree::LeafNodeType;
// 	//auto narrow_leaves = tbb::concurrent_hash_map<openvdb::Coord, point_leaf_t*>();

// 	//prepare the particle voxel mask
// 	//liquidsdf is one ring larger than point datagrid
// 	//this will only keep the extra one ring
// 	auto point_voxel_mask = m_pushed_out_liquid_sdf->deepCopy();

// 	auto keep_one_layer_mask = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		auto point_axr{ m_particles->getConstAccessor() };
// 		auto pointleaf = m_particles->tree().probeConstLeaf(leaf.origin());
// 		auto maskaxr{ point_voxel_mask->getAccessor() };
// 		//there is no boundary in this sdf leaf
// 		if (pointleaf && pointleaf->isValueMaskOn()) {
// 			leaf.setValuesOff();
// 			return;
// 		}
// 		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
// 			if (point_axr.isValueOn(iter.getCoord())) {
// 				iter.setValueOff();
// 			}
// 		}

// 		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
// 			//if all its neighbor is inside liquid
// 			//it is still surrounded by particles
// 			bool all_in_liquid = true;
// 			auto itercoord = iter.getCoord();
// 			for (int ii = -1; ii < 2 && all_in_liquid; ii++) {
// 				for (int jj = -1; jj < 2 && all_in_liquid; jj++) {
// 					for (int kk = -1; kk < 2 && all_in_liquid; kk++) {
// 						all_in_liquid &= 
// 							maskaxr.getValue(itercoord.offsetBy(ii,jj,kk)) < 0;
// 					}
// 				}
// 			}
// 			if (all_in_liquid) {
// 				iter.setValueOff();
// 				continue;
// 			}
			
// 			/*if (itercoord[0] < m_domain_index_begin[0]||itercoord[0]>=m_domain_index_end[0]||
// 				itercoord[1] < m_domain_index_begin[1] || itercoord[1] >= m_domain_index_end[1] || 
// 				itercoord[2] < m_domain_index_begin[2] || itercoord[2] >= m_domain_index_end[2]) {
// 				iter.setValueOff();
// 			}*/
// 		}
// 	};
// 	auto maskman = openvdb::tree::LeafManager<openvdb::FloatTree>(point_voxel_mask->tree());
// 	maskman.foreach(keep_one_layer_mask);
// 	printf("keep_one_layer_mask_done\n");
// 	auto smaller_mask = point_voxel_mask->deepCopy();
// 	auto small_maskman = openvdb::tree::LeafManager<openvdb::FloatTree>(smaller_mask->tree());

// 	int nlayer = 2;
// 	//dilate the mask
// 	openvdb::tools::dilateVoxels(maskman, nlayer, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
// 	openvdb::tools::dilateVoxels(small_maskman, nlayer - 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
// 	printf("dilate_voxel_mask_done\n");
// 	//use the mask to shrink the sdf so that it only shows the interior liquid
// 	//to be used by the shrinked sdf manager
// 	auto shrink_sdf_op = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
// 		auto smaller_mask_axr{ smaller_mask->getConstAccessor() };
// 		auto smaller_mask_leaf= smaller_mask->tree().probeConstLeaf(leaf.origin()) ;

// 		if (leaf.isValueMaskOn()) {
// 			//this leaf has no voxel near the free surface
// 			//it is interior liquid
// 			//set it off if it's quasi uniform
// 			bool all_negative = true;
// 			for (auto iter = leaf.beginValueAll(); iter; ++iter) {
// 				if (iter.getValue() > 0) {
// 					all_negative = false;
// 					break;
// 				}
// 			}
// 			if (all_negative) {
// 				leaf.fill(-m_particle_radius, true);
// 			}

// 			//return;
// 		}

// 		for (auto iter = smaller_mask_leaf->beginValueOn(); iter; ++iter) {
// 			if (leaf.isValueMaskOn(iter.offset())) {
// 				leaf.setValueOn(iter.offset(), m_dx * 0.5);
// 			}
// 		}
// 	};

// 	m_shrinked_liquid_sdf = m_pushed_out_liquid_sdf->deepCopy();
// 	m_shrinked_liquid_sdf->setName("Interior_SDF");
// 	auto sdfman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_shrinked_liquid_sdf->tree());
// 	sdfman.foreach(shrink_sdf_op);
// 	printf("shrink_sdf_done\n");
// 	openvdb::tools::pruneLevelSet(m_shrinked_liquid_sdf->tree(), 0.5f * m_dx, -0.5f * m_dx);
// 	//to be used by particle leaf manager
// 	//detects leaves with particles that are near the border
// 	//either it's not full
// 	//or its boundary voxels detects empty neighbors
// 	auto mark_point_on = [&](point_leaf_t& leaf, openvdb::Index leafpos) {
// 		auto mask_axr{ point_voxel_mask->getConstAccessor() };
// 		auto maskleaf = point_voxel_mask->tree().probeConstLeaf(leaf.origin());

// 		if (!maskleaf) {
// 			return;
// 		}
// 		leaf.setValueMask(maskleaf->getValueMask());
// 		if (leaf.isValueMaskOff()) {
// 			return;
// 		}

// 		std::vector<openvdb::Vec3f> pos;
// 		std::vector<openvdb::Vec3f> vel;
// 		std::vector<openvdb::Index> new_index;
// 		int emitted_particle = 0;
// 		auto& positionarray = leaf.attributeArray("P");
// 		auto phandle = openvdb::points::AttributeHandle<openvdb::Vec3f, PositionCodec>(positionarray);

// 		auto& velarray = leaf.attributeArray("v");
// 		auto vhandle = openvdb::points::AttributeHandle<openvdb::Vec3f, VelocityCodec>(velarray);
// 		for (auto iter = leaf.cbeginValueAll(); iter; ++iter) {
// 			if (!iter.isValueOn()) {
// 				new_index.push_back(emitted_particle);
// 			}
// 			else {
// 				int bidx = 0;
// 				if (iter.offset() != 0) {
// 					bidx = leaf.getValue(iter.offset() - 1);
// 				}
// 				int eidx = iter.getValue();
// 				for (auto idx = bidx; idx < eidx; ++idx) {
// 					pos.push_back(phandle.get(idx));
// 					vel.push_back(vhandle.get(idx));
// 					emitted_particle++;
// 				}
// 				new_index.push_back(emitted_particle);
// 			}
// 		}//end for all voxels in this leaf

// 		for (auto offset = 0; offset < leaf.SIZE; ++offset) {
// 			leaf.setOffsetOnly(offset, new_index[offset]);
// 		}


// 		//particle emit complete, replace the original attribute set
// 		leaf.initializeAttributes(pos_descriptor(), pos.size());
// 		leaf.appendAttribute(leaf.attributeSet().descriptor(), m_pv_attribute_descriptor, 1);

// 		auto& new_positionarray = leaf.attributeArray("P");
// 		auto p_write_handle = openvdb::points::AttributeWriteHandle<openvdb::Vec3f, PositionCodec>(new_positionarray);

// 		auto& new_velarray = leaf.attributeArray("v");
// 		auto v_write_handle = openvdb::points::AttributeWriteHandle<openvdb::Vec3f, VelocityCodec>(new_velarray);

// 		int new_emit_counter = 0;
// 		for (auto piter = leaf.beginIndexOn(); piter; ++piter) {
// 			p_write_handle.set(*piter, pos[new_emit_counter]);
// 			v_write_handle.set(*piter, vel[new_emit_counter]);
// 			new_emit_counter++;
// 		}
		
// 		//finally set on and off mask
// 		for (auto offset = 0; offset < leaf.SIZE; ++offset) {
// 			int bidx = 0;
// 			if (offset != 0) {
// 				bidx = leaf.getValue(offset - 1);
// 			}
// 			int eidx = leaf.getValue(offset);
// 			if (bidx == eidx) {
// 				leaf.setValueOff(offset);
// 			}
// 		}
// 	};//end collect leaves with holes
	
// 	auto result = m_particles->deepCopy();
// 	auto resultman = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(result->tree());
// 	resultman.foreach(mark_point_on);

// 	printf("mark_point_on_done\n");
// 	for (auto iter = result->tree().beginLeaf(); iter; ++iter) {
// 		if (iter->isValueMaskOff()) {
// 			delete result->tree().stealNode<point_leaf_t>(iter->origin(), 0, false);
// 		}
// 	}
// 	printf("remove empty node done\n");
// 	return result;
// }
