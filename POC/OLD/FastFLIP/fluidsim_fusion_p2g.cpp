#include "fluidsim.h"
#include "Sparse_buffer.h"
#include "sparse_matrix.h"
#include "Timer.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "volumeMeshTools.h"
#include "Eigen/Eigen"
#include <stdlib.h>
#include <memory>
#include <immintrin.h>
/// <summary>
// A fushion kernel that transform the velocity and liquid phi information
// to the grid at the same time
// Each voxel holds the corner position in float packed in 4 float
// packed in the following way:
// VX0: xyz0 V1: xyz0
//
// The particles are also sorted according to each bulks
// particle data: 
// P0: xyz0uvw0 P2: xyz0uvw0
// Hence the difference in space can be easily carried out by sse instruction
//
// The loop over 27 neighborhood is unrolled
// The loop generate 27 weights for a particle
// those weights will be used to transfer the uvw and phi variable
/// </summary>

namespace {
	//sort the particles into buckets for each bulk
	struct particle_sorter_body_t {
		//the initial constructor
		particle_sorter_body_t(
			const std::vector< minimum_FLIP_particle>& in_minimum_FLIP_particles,
			size_t number_of_bulks,
			std::shared_ptr<sparse_fluid8x8x8> sparse_bulk,
			float in_dx
		) :m_unsorted_particles(in_minimum_FLIP_particles) {
			m_sparse_bulk = sparse_bulk;
			m_bin_bulk_particle.resize(number_of_bulks);
			m_dx = in_dx;
		};

		//the split constructor
		particle_sorter_body_t(particle_sorter_body_t& x, tbb::split) :m_unsorted_particles(x.m_unsorted_particles) {
			m_sparse_bulk = x.m_sparse_bulk;
			m_bin_bulk_particle.resize(x.m_bin_bulk_particle.size());
			m_dx = x.m_dx;
		}

		void operator()(const tbb::blocked_range<size_t>& r) {
			//r is the range of particles
			std::vector<LosTopos::Vec3i> influencing_bulk_ijk;
			std::vector<int> extrabulk[3];
			for (auto i = r.begin(); i != r.end(); i++) {
				//identify which bulk idx this particle belongs to
				//it is possible that the fluid may be out of the domain
				//hence delete it completely if it is beyond [bmin,bmax]
				const auto& p = m_unsorted_particles[i].pos;
				const auto& vel = m_unsorted_particles[i].vel;
				bool is_in_all_bulks = true;
				for (int j = 0; j < 3; j++) {
					if (p[j] < (*m_sparse_bulk).bmin[j]) {
						is_in_all_bulks = false;
						break;
					}
					if (p[j] > (*m_sparse_bulk).bmax[j]) {
						is_in_all_bulks = false;
						break;
					}
				}

				if (is_in_all_bulks) {

					extrabulk[0].clear(); extrabulk[0].push_back(0);
					extrabulk[1].clear(); extrabulk[1].push_back(0);
					extrabulk[2].clear(); extrabulk[2].push_back(0);

					influencing_bulk_ijk.assign(8, LosTopos::Vec3i{ -1,-1,-1 });
					LosTopos::Vec3f pos = p - (*m_sparse_bulk).bmin;
					LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i((int)floor(pos[0] / (*m_sparse_bulk).bulk_size),
						(int)floor(pos[1] / (*m_sparse_bulk).bulk_size),
						(int)floor(pos[2] / (*m_sparse_bulk).bulk_size));
					LosTopos::Vec3i voxel_ijk = LosTopos::floor(pos / m_dx) - bulk_ijk * 8;


					for (int j = 0; j < 3; j++) {
						if (voxel_ijk[j] == 0) {
							extrabulk[j].push_back(-1);
						}
						else if (voxel_ijk[j] == 7) {
							extrabulk[j].push_back(1);
						}
					}//end test three neib direction

					for (const auto& xx : extrabulk[0]) {
						for (const auto& yy : extrabulk[1]) {
							for (const auto& zz : extrabulk[2]) {
								LosTopos::Vec3i extra_bulk_ijk = bulk_ijk + LosTopos::Vec3i{ xx,yy,zz };
								auto found_bulk = m_sparse_bulk->index_mapping.find(extra_bulk_ijk);
								if (found_bulk != m_sparse_bulk->index_mapping.end()) {
									auto idx = found_bulk->second;
									m_bin_bulk_particle[idx].push_back(i);
								}
								//auto idx = (*m_sparse_bulk).get_bulk_index(extra_bulk_ijk[0], extra_bulk_ijk[1], extra_bulk_ijk[2]);
								////m_bin_bulk_particle[idx].push_back(pos4_vel4_t(p, vel));
							}
						}
					}

				}//end if the particle is in all bulks
			}//end for all particles in this range
		};//end void operator()

		void join(const particle_sorter_body_t& other) {
			tbb::parallel_for((size_t)0, m_bin_bulk_particle.size(), [&](size_t bulk_idx) {
				size_t old_size = m_bin_bulk_particle[bulk_idx].size();
				m_bin_bulk_particle[bulk_idx].resize(old_size + other.m_bin_bulk_particle[bulk_idx].size());
				std::copy(other.m_bin_bulk_particle[bulk_idx].begin(),
					other.m_bin_bulk_particle[bulk_idx].end(),
					m_bin_bulk_particle[bulk_idx].begin() + old_size);
				});
		}

		//reference to particles
		const std::vector<minimum_FLIP_particle>& m_unsorted_particles;

		//reduction result
		//[index of bulk][posvel influencing this bulk]
		//note that the beginning of each bin may not be properly aligned with 32bit location
		std::vector<std::vector<int>> m_bin_bulk_particle;
		std::shared_ptr<sparse_fluid8x8x8> m_sparse_bulk;
		float m_dx;
	};




	//p2g bulk collection body
	struct p2g_body_t {
		p2g_body_t(const std::vector<std::vector<int>>& in_bin_pidx,
			std::shared_ptr<sparse_fluid8x8x8> in_sparse_bulk,
			const std::vector<minimum_FLIP_particle>& in_particles,
			float in_dx, float in_particle_radius) :
			m_bin_pidx(in_bin_pidx),
			m_particles(in_particles),
			m_dx(in_dx), m_particle_radius(in_particle_radius) {
			m_sparse_bulk = in_sparse_bulk;
			set_offsets();
		};

		void set_offsets() {
			//u:  (0, 0.5, 0.5)
			//v:  (0.5, 0, 0.5)
			//w:  (0.5, 0.5, 0)
			//phi:(0.5, 0.5, 0.5)
			//note we use setr so the 
			//R0 R1 R2 R3 correspond to the 0,1,2,3 argument
			//when later extracting using _mm_store_ps, we get float[0] = arg0 correctly
			//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_store.htm
			//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_set.htm
			__m128 xpack = _mm_setr_ps(0.0f, 0.5f, 0.5f, 0.5f);
			__m128 ypack = _mm_setr_ps(0.5f, 0.0f, 0.5f, 0.5f);
			__m128 zpack = _mm_setr_ps(0.5f, 0.5f, 0.0f, 0.5f);
			loop_order.resize(27, LosTopos::Vec3i{ 0,0,0 });
			for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
				int ijk = ivoxel;
				int basex = ijk / 9; ijk -= 9 * basex;
				int basey = ijk / 3; ijk -= 3 * basey;
				int basez = ijk;
				//becomes -1 -> 1
				basex -= 1; basey -= 1; basez -= 1;

				loop_order[ivoxel] = LosTopos::Vec3i{ basex,basey,basez };

				//broadcast four float as the base
				__m128 basex4 = _mm_set_ps1(float(basex));
				__m128 basey4 = _mm_set_ps1(float(basey));
				__m128 basez4 = _mm_set_ps1(float(basez));

				x_offset_to_center_voxel_origin[ivoxel] = _mm_add_ps(basex4, xpack);
				y_offset_to_center_voxel_origin[ivoxel] = _mm_add_ps(basey4, ypack);
				z_offset_to_center_voxel_origin[ivoxel] = _mm_add_ps(basez4, zpack);
			}
		};




		void operator()(const tbb::blocked_range<size_t>& bulk_range) const {
			const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
			__m128 x_dist_particle_to_sample[27];
			__m128 y_dist_particle_to_sample[27];
			__m128 z_dist_particle_to_sample[27];
			const __m128 float1x4 = _mm_set_ps1(float(1));
			const __m128 float0x4 = _mm_set_ps1(float(0));

			//local u,v,w,phi, sum of weight[u,v,w,phi]
			chunck3D<float, 8> u, v, w, phi, wtu, wtv, wtw, wtphi;

			for (size_t bulkidx = bulk_range.begin(); bulkidx < bulk_range.end(); bulkidx++) {

				//clear the temporary 
				u.data.assign(512, 0.f); wtu.data.assign(512, 0.f);
				v.data.assign(512, 0.f); wtv.data.assign(512, 0.f);
				w.data.assign(512, 0.f); wtw.data.assign(512, 0.f);
				phi.data.assign(512, 0.f); wtphi.data.assign(512, 0.f);

				//the bulk of data to write to
				auto& the_bulk = m_sparse_bulk->fluid_bulk[bulkidx];
				const LosTopos::Vec3i& bulk_corner_ijk = the_bulk.tile_corner;

				//prepare the particle data to process
				size_t np = m_bin_pidx[bulkidx].size();

				//aligned_posvel.resize(np);

				////retrieve the particle data
				//auto getparti = [&](size_t pidx) {
				//	for (int j = 0; j < 3; j++) {
				//		aligned_posvel[pidx].pos[j] = m_particles[m_bin_pidx[bulkidx][pidx]].pos[j];
				//		aligned_posvel[pidx].vel[j] = m_particles[m_bin_pidx[bulkidx][pidx]].vel[j];
				//	}
				//	aligned_posvel[pidx].pos[3] = 0;
				//	aligned_posvel[pidx].vel[3] = 0;
				//};

				//for (size_t i = 0; i < np; i++) {
				//	getparti(i);
				//}

				//scatter the particle information to the actual bulk
				for (size_t i_p = 0; i_p < np; i_p++) {
					const auto ppos = m_particles[m_bin_pidx[bulkidx][i_p]].pos;
					const auto pvel = m_particles[m_bin_pidx[bulkidx][i_p]].vel;
					//calculate the weights at the 27 voxels around p
					//note some voxels are outside the current bulk and its weights is not use at all

					//each voxel has four evaluation point: u,v,w,phi
					//their offset to the origin of the voxel is:
					//u:  (0, 0.5, 0.5)
					//v:  (0.5, 0, 0.5)
					//w:  (0.5, 0.5, 0)
					//phi:(0.5, 0.5, 0.5)

					/*particle distance to the origin of its voxel*/
					float coord_x_in_dx = (ppos[0] - m_sparse_bulk->bmin[0]) / m_dx;
					float coord_y_in_dx = (ppos[1] - m_sparse_bulk->bmin[1]) / m_dx;
					float coord_z_in_dx = (ppos[2] - m_sparse_bulk->bmin[2]) / m_dx;

					//coord_[x,y,z] is assumed to be positive
					//the following operation then becomes flooring
					int voxel_x_id = int(coord_x_in_dx);
					int voxel_y_id = int(coord_y_in_dx);
					int voxel_z_id = int(coord_z_in_dx);

					//now coord_[x,y,z]_in_dx should be between (0,1)
					coord_x_in_dx -= voxel_x_id;
					coord_y_in_dx -= voxel_y_id;
					coord_z_in_dx -= voxel_z_id;

					//broadcast the variables
					__m128 particle_x = _mm_set_ps1(coord_x_in_dx);
					__m128 particle_y = _mm_set_ps1(coord_y_in_dx);
					__m128 particle_z = _mm_set_ps1(coord_z_in_dx);

					//scatter

					for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
						//calculate the distance
						//arg(A,B): ret A-B
						// the absolute value trick: abs_mask: 01111111..32bit..1111 x 4
						// _mm_and_ps(abs_mask(), v);
						x_dist_particle_to_sample[ivoxel] = _mm_and_ps(absmask, _mm_sub_ps(x_offset_to_center_voxel_origin[ivoxel], particle_x));
						y_dist_particle_to_sample[ivoxel] = _mm_and_ps(absmask, _mm_sub_ps(y_offset_to_center_voxel_origin[ivoxel], particle_y));
						z_dist_particle_to_sample[ivoxel] = _mm_and_ps(absmask, _mm_sub_ps(z_offset_to_center_voxel_origin[ivoxel], particle_z));

						//transfer the distance to weight at the 27 voxels
						//(1-dist)
						//the far points now becomes negative
						x_dist_particle_to_sample[ivoxel] = _mm_sub_ps(float1x4, x_dist_particle_to_sample[ivoxel]);
						y_dist_particle_to_sample[ivoxel] = _mm_sub_ps(float1x4, y_dist_particle_to_sample[ivoxel]);
						z_dist_particle_to_sample[ivoxel] = _mm_sub_ps(float1x4, z_dist_particle_to_sample[ivoxel]);

						//turn everything positive or zero
						//now the dist_to_sample is actually the component-wise weight on the voxel
						//time to multiply them together
						x_dist_particle_to_sample[ivoxel] = _mm_max_ps(float0x4, x_dist_particle_to_sample[ivoxel]);
						y_dist_particle_to_sample[ivoxel] = _mm_max_ps(float0x4, y_dist_particle_to_sample[ivoxel]);
						z_dist_particle_to_sample[ivoxel] = _mm_max_ps(float0x4, z_dist_particle_to_sample[ivoxel]);

						//turn them into weights reduce to x
						x_dist_particle_to_sample[ivoxel] = _mm_mul_ps(x_dist_particle_to_sample[ivoxel], y_dist_particle_to_sample[ivoxel]);
						x_dist_particle_to_sample[ivoxel] = _mm_mul_ps(x_dist_particle_to_sample[ivoxel], z_dist_particle_to_sample[ivoxel]);
					}

					//at this point
					//x_dist_particle_to_sample contains the weights of this particle to the u,v,w,phi sampling point
					//write to the actual result

					voxel_x_id -= the_bulk.tile_corner[0];
					voxel_y_id -= the_bulk.tile_corner[1];
					voxel_z_id -= the_bulk.tile_corner[2];
					float total_weight[4] = { 0,0,0,0 };
					float inbox_counter = 0;
					for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
						//-1,0,1 offsets
						LosTopos::Vec3i idx3 = loop_order[ivoxel];
						idx3[0] += voxel_x_id;
						idx3[1] += voxel_y_id;
						idx3[2] += voxel_z_id;


						//channel
						bool in_bulk = true;
						for (int i_c = 0; i_c < 3; i_c++) {
							if (idx3[i_c] < 0 || idx3[i_c] >= 8) {
								in_bulk = false;
								break;
							}
						}

						if (in_bulk) {
							inbox_counter++;
							//unpack the four weights
							alignas(16) float packed_weight[4];
							_mm_store_ps(packed_weight, x_dist_particle_to_sample[ivoxel]);
							for (int temp = 0; temp < 4; temp++) {
								total_weight[temp] += packed_weight[temp];
							}
							int mem_offset = idx3[0] + 8 * (idx3[1] + 8 * idx3[2]);

							u.data[mem_offset] += packed_weight[0] * pvel[0];
							wtu.data[mem_offset] += packed_weight[0];

							v.data[mem_offset] += packed_weight[1] * pvel[1];
							wtv.data[mem_offset] += packed_weight[1];

							w.data[mem_offset] += packed_weight[2] * pvel[2];
							wtw.data[mem_offset] += packed_weight[2];

							phi.data[mem_offset] -= packed_weight[3] * m_particle_radius;
							wtphi.data[mem_offset] += packed_weight[3];
						}
					}//end write to the influenced voxel
				}//end actual p2g for this particle


				//composing
				for (size_t i_flat_voxel = 0; i_flat_voxel < 512; i_flat_voxel++) {
					//write to the actual data u
					if (wtu.data[i_flat_voxel] > 0) {
						the_bulk.u.data[i_flat_voxel] = u.data[i_flat_voxel] / wtu.data[i_flat_voxel];
						the_bulk.u_valid.data[i_flat_voxel] = 1;
					}
					else {
						the_bulk.u.data[i_flat_voxel] = 0;
						the_bulk.u_valid.data[i_flat_voxel] = 0;
					}
					//write to the actual data v
					if (wtv.data[i_flat_voxel] > 0) {
						the_bulk.v.data[i_flat_voxel] = v.data[i_flat_voxel] / wtv.data[i_flat_voxel];
						the_bulk.v_valid.data[i_flat_voxel] = 1;
					}
					else {
						the_bulk.v.data[i_flat_voxel] = 0;
						the_bulk.v_valid.data[i_flat_voxel] = 0;
					}

					//write to the actual data w
					if (wtw.data[i_flat_voxel] > 0) {
						the_bulk.w.data[i_flat_voxel] = w.data[i_flat_voxel] / wtw.data[i_flat_voxel];
						the_bulk.w_valid.data[i_flat_voxel] = 1;
					}
					else {
						the_bulk.w.data[i_flat_voxel] = 0;
						the_bulk.w_valid.data[i_flat_voxel] = 0;
					}

					//write to the phi
					//0.5*dx + 2.0*res[gidx];
					the_bulk.liquid_phi.data[i_flat_voxel] = 0.5 * m_dx + 2.0 * phi.data[i_flat_voxel] / (1.0e-6f + wtphi.data[i_flat_voxel]);
				}
				//printf("p2g for bulk %d done\n", bulkidx);
			}//end for each bulk
		}//end operator()


		//each voxel has four evaluation point: u,v,w,phi
		//their offset to the origin of the voxel is:
		//u:  (0, 0.5, 0.5)
		//v:  (0.5, 0, 0.5)
		//w:  (0.5, 0.5, 0)
		//phi:(0.5, 0.5, 0.5)
		//additionally, the neighbor voxels have offsets to the center voxel
		//the 27 offsets in each channel are ordered according their positions relative to
		//the minimum of the 27 voxel
		//the 000 voxel has (-1,-1,-1) offset to the center voxel
		//the 001 voxel has (-1,-1, 1) offset to the center voxel
		//...
		//the 222 voxel has (1, 1, 1) offset to the center voxel
		//the inner loop is ordered by uvwphi
		//the offsets here have unit of dx
		alignas(32) __m128 x_offset_to_center_voxel_origin[27];
		alignas(32) __m128 y_offset_to_center_voxel_origin[27];
		alignas(32) __m128 z_offset_to_center_voxel_origin[27];


		std::shared_ptr<sparse_fluid8x8x8> m_sparse_bulk;
		const std::vector<minimum_FLIP_particle>& m_particles;
		const std::vector<std::vector<int>>& m_bin_pidx;
		std::vector<LosTopos::Vec3i> loop_order;
		const float m_dx;
		const float m_particle_radius;
		//dangerous raw pointer to store all the positions and velocities of particles
		//associated with a bulk
		//it will be re-sized for different bulk.

	};
}

void FluidSim::fusion_p2g_liquid_phi()
{
	auto sort_body = particle_sorter_body_t{ particles,eulerian_fluids->fluid_bulk.size(),eulerian_fluids,dx };
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, particles.size()), sort_body);
	//sort_body(tbb::blocked_range<size_t>(0, particles.size()));

	auto p2gbody = p2g_body_t(sort_body.m_bin_bulk_particle, eulerian_fluids, particles, dx, particle_radius);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, eulerian_fluids->fluid_bulk.size()), p2gbody);
	//p2gbody(tbb::blocked_range<size_t>(0, eulerian_fluids->fluid_bulk.size()));
	//extend liquids slightly into solids
	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).fluid_bulk.size(),
		(size_t)1,
		[&](size_t index)
		{
			for (uint i = 0; i < (*eulerian_fluids).n_perbulk; i++)
			{
				LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
				if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0.5 * dx)
				{
					float solid_phi = 0.125 * ((*eulerian_fluids).solid_phi(index, ijk[0], ijk[1], ijk[2])
						+ (*eulerian_fluids).solid_phi(index, ijk[0] + 1, ijk[1], ijk[2])
						+ (*eulerian_fluids).solid_phi(index, ijk[0], ijk[1] + 1, ijk[2])
						+ (*eulerian_fluids).solid_phi(index, ijk[0] + 1, ijk[1] + 1, ijk[2])
						+ (*eulerian_fluids).solid_phi(index, ijk[0], ijk[1], ijk[2] + 1)
						+ (*eulerian_fluids).solid_phi(index, ijk[0] + 1, ijk[1], ijk[2] + 1)
						+ (*eulerian_fluids).solid_phi(index, ijk[0], ijk[1] + 1, ijk[2] + 1)
						+ (*eulerian_fluids).solid_phi(index, ijk[0] + 1, ijk[1] + 1, ijk[2] + 1));
					if (solid_phi < 0)
					{
						(*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) = -0.5 * dx;
					}
				}
			}

		});
}