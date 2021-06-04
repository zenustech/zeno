#include "Sparse_buffer.h"
#include "tbb/blocked_range3d.h"
#include <functional>

template<> void sparse_fluid_3D<8>::initialize_bulks(
	std::vector<minimum_FLIP_particle>& particles,
	double cell_size,
	bool use_hard_bminmax,
	LosTopos::Vec3f hard_bmin,
	LosTopos::Vec3f hard_bmax,
	bool reserve_potential_bulks,
	std::function<bool(const LosTopos::Vec3i& bulk_ijk)> worth)
{
	int N = 8;
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks").start();
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead1").start();
	//fluid_bulk.resize(0);
	m_index3_of_non_empty_bulks.resize(0);
	index_mapping.clear();
	h = (float)cell_size;
	uint num_particles = (uint)particles.size();

	if (use_hard_bminmax) {
		//determine from caller
		bmin = hard_bmin;
		bmax = hard_bmax;
	}
	else {
		//determine from the extend of the liquid particles
		bmin = particles[0].pos;
		bmax = particles[0].pos;
		for (uint i = 1; i < num_particles; i++)
		{
			bmin = min_union(bmin, particles[i].pos);
			bmax = max_union(bmax, particles[i].pos);
		}
		bmin -= LosTopos::Vec3f(24 * h, 24 * h, 24 * h);
		bmax += LosTopos::Vec3f(24 * h, 24 * h, 24 * h);
	}
	//move bmin to align with h
	LosTopos::Vec3i bmin_i = LosTopos::floor(bmin / h);
	bmin = h * LosTopos::Vec3f(bmin_i);

	bulk_size = (float)N * h;
	ni = (uint)ceil((bmax[0] - bmin[0]) / bulk_size);
	nj = (uint)ceil((bmax[1] - bmin[1]) / bulk_size);
	nk = (uint)ceil((bmax[2] - bmin[2]) / bulk_size);
	std::cout << "bmin: " << bmin << std::endl;
	std::cout << "bmax: " << bmax << std::endl;
	std::cout << " bulk_size " << bulk_size << std::endl;

	bmax = bmin + bulk_size * LosTopos::Vec3f((float)ni, (float)nj, (float)nk);
	std::vector<char> buffer;
	buffer.resize(ni * nj * nk);
	buffer.assign(ni * nj * nk, 0);
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead1").stop();

    //CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead2").start();
	

	tbb::parallel_for((uint)0, (uint)num_particles, (uint)1, [&](uint i)
//		for (uint i=0;i<num_particles;i++)
		{
			//disgard particles that are beyond the domain
			for (int j = 0; j < 3; j++) {
				if (particles[i].pos[j] < bmin[j]) {
					return;
				}
				if (particles[i].pos[j] >= bmax[j]) {
					return;
				}
			}

			LosTopos::Vec3f pos = particles[i].pos - bmin;

			

			LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i((int)floor(pos[0] / bulk_size),
				(int)floor(pos[1] / bulk_size),
				(int)floor(pos[2] / bulk_size));

			if (bulk_ijk[0] >= 0 && bulk_ijk[0] < ni) {
				if (bulk_ijk[1] >= 0 && bulk_ijk[1] < nj) {
					if (bulk_ijk[2] >= 0 && bulk_ijk[2] < nk) {
						size_t idx = bulk_ijk[0] + ni * bulk_ijk[1] + ni * nj * bulk_ijk[2];
						buffer[idx] = 1;
					}
				}
			}
		});
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead2").stop();

    //the above step activates where the fluid particles present
	//however there are some places where there are currently no particles, but will be seeced with particles
	//those are boundary bulks
	//the boundary bulks also need to be activated
	if (reserve_potential_bulks) {
		tbb::parallel_for(tbb::blocked_range3d<uint>(0, ni, 0, nj, 0, nk),
			[&](const tbb::blocked_range3d<uint>& R) {
				for (auto i = R.pages().begin(); i != R.pages().end(); ++i) {
					for (auto j = R.rows().begin(); j != R.rows().end(); ++j) {
						for (auto k = R.cols().begin(); k != R.cols().end(); k++) {
							LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(i, j, k);
							if (worth(bulk_ijk)) {
								size_t idx = bulk_ijk[0] + ni * bulk_ijk[1] + ni * nj * bulk_ijk[2];
								buffer[idx] = 1;
							}
						}
					}
				}
			});
	}
	printf("found additional bulk complete\n");



	for (int buffer_id = 0; buffer_id < buffer.size(); buffer_id++)
	{
		if (buffer[buffer_id] == 1)
		{
			int ii = buffer_id % ni, jj = (buffer_id % (ni * nj)) / ni, kk = buffer_id / (ni * nj);
			LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(ii, jj, kk);
			if (index_mapping.find(bulk_ijk) == index_mapping.end())
			{
				/*index_mapping[bulk_ijk] = fluid_bulk.size();
				fluid_bulk.push_back(fluid_Tile<N>(LosTopos::Vec3i(N*ii, N*jj, N*kk), (uint)fluid_bulk.size()));*/
				index_mapping[bulk_ijk] = m_index3_of_non_empty_bulks.size();
				m_index3_of_non_empty_bulks.push_back(LosTopos::Vec3i(N * ii, N * jj, N * kk));
			}
		}
	}

	//int non_boundary_bulks;

	for (int i = 0; i < 1; i++)
	{
		//non_boundary_bulks = fluid_bulk.size();
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks").start();
		if (!reserve_potential_bulks) {
			//when reserving potential bulks, it is assumed the external function will
			//handle the extension
			extend_fluidbulks();
		}
		
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks").stop();
	}

	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/push_fluid").start();
	fluid_bulk.resize(m_index3_of_non_empty_bulks.size());
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/push_fluid").stop();
	for (int i = 0; i < fluid_bulk.size(); i++) {
		fluid_bulk[i].tile_corner = m_index3_of_non_empty_bulks[i];
		fluid_bulk[i].bulk_index = i;
	}
	std::cout << "indexmap info: bucket count: " << index_mapping.bucket_count() << std::endl;
	std::cout << "elements: " << index_mapping.size() << std::endl;
	//std::cout<<non_boundary_bulks<<std::endl;
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/identify_boundary").start();
	for (int i = 0; i < fluid_bulk.size(); i++)
	{
		LosTopos::Vec3i bulk_ijk = fluid_bulk[i].tile_corner / N;
		if (index_mapping.find(bulk_ijk + LosTopos::Vec3i(-1, 0, 0)) == index_mapping.end()
			|| index_mapping.find(bulk_ijk + LosTopos::Vec3i(1, 0, 0)) == index_mapping.end()
			|| index_mapping.find(bulk_ijk + LosTopos::Vec3i(0, 1, 0)) == index_mapping.end()
			|| index_mapping.find(bulk_ijk + LosTopos::Vec3i(0, -1, 0)) == index_mapping.end()
			|| index_mapping.find(bulk_ijk + LosTopos::Vec3i(0, 0, -1)) == index_mapping.end()
			|| index_mapping.find(bulk_ijk + LosTopos::Vec3i(0, 0, 1)) == index_mapping.end())
		{
			fluid_bulk[i].is_boundary = true;
		}

	}
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/identify_boundary").stop();
	n_bulks = (uint)fluid_bulk.size();
	std::cout << "num of bulks:" << fluid_bulk.size() << std::endl;
	std::cout << "bmin:" << bmin[0] << " " << bmin[1] << " " << bmin[2] << std::endl;
	std::cout << "bmax:" << bmax[0] << " " << bmax[1] << " " << bmax[2] << std::endl;
	std::cout << "dimension:" << ni << " " << nj << " " << nk << std::endl;
	//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks").stop();
}
