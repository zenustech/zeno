#include "fluidsim.h"
#include "Sparse_buffer.h"
#include "sparse_matrix.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "volumeMeshTools.h"
#include "Eigen/Eigen"
#include "amg2.h"
#include "morton_encoding.h"
#include <openvdb/tools/Interpolation.h>
#include "SimOptions.h"
FluidSim::FluidSim(BPS3D* sim, bool sync_with_bem) : m_bps(sim), sync_with_bem(sync_with_bem) {
    eulerian_fluids = std::make_shared<sparse_fluid8x8x8>(sparse_fluid8x8x8{});
    resample_field = std::make_shared<sparse_fluid8x8x8>(sparse_fluid8x8x8{});
}
void FluidSim::initialize(double _dx)
{
	dx = _dx;
	particle_radius = (float)(dx * 1.01*sqrt(3.0)/2.0);
	total_frame = 0;
}
float FluidSim::cfl()
{
	float max_vel = 0;
    std::vector<float> max_vels;
	max_vels.resize((*eulerian_fluids).n_bulks);
	tbb::parallel_for((size_t)0,
					  (size_t)(*eulerian_fluids).n_bulks,
					  (size_t)1,
					  [&](size_t index)
	{
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			
			max_vels[index] = std::max(std::fabs((*eulerian_fluids).fluid_bulk[index].u.data[i]),max_vels[index]);
		}
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			
			max_vels[index] = std::max(std::fabs((*eulerian_fluids).fluid_bulk[index].v.data[i]),max_vels[index]);
		}
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			
			max_vels[index] = std::max(std::fabs((*eulerian_fluids).fluid_bulk[index].w.data[i]),max_vels[index]);
		}
	});
	for(uint i=0;i<max_vels.size();i++)
	{
		max_vel = std::max(max_vel,max_vels[i]);
	}
	std::cout<<"max vel:"<<max_vel<<std::endl;
	float dt = dx / max_vel;

    std::vector<float> temp;
	max_vels.swap(temp);
	return dt;
}

void FluidSim::set_liquid(const LosTopos::Vec3f &bmin, const LosTopos::Vec3f &bmax, std::function<float(const LosTopos::Vec3f&)> phi)
{
	float gap = 0.5*dx;
	std::cout<<"gap between particles: "<<gap<<std::endl;
	std::cout<<"bmin: "<<bmin[0]<<" "<<bmin[1]<<" "<<bmin[2]<<std::endl;
	std::cout<<"bmax: "<<bmax[0]<<" "<<bmax[1]<<" "<<bmax[2]<<std::endl;
	LosTopos::Vec3i nijk = LosTopos::ceil((bmax-bmin)/gap);
	std::cout<<"particle dimensions: "<<nijk[0]<<" "<<nijk[1]<<" "<<nijk[2]<<std::endl;
    //particles.reserve(particles.size()+nijk[0] * nijk[1] * nijk[2] * 0.7);
	/*for (int k=0;k<nijk[2];k++)for(int j=0;j<nijk[1];j++)for(int i=0;i<nijk[0];i++)
	{
		LosTopos::Vec3f pos = LosTopos::Vec3f(i, j, k)*gap + bmin;
		if(phi(pos)<=0 && pos[0]>=bmin[0] && pos[1]>=bmin[1] && pos[2]>=bmin[2]
			&& pos[0]<=bmax[0] && pos[1]<=bmax[1] && pos[2]<=bmax[2])
			particles.push_back(minimum_FLIP_particle(pos,LosTopos::Vec3f(0,0,0)));
	}*/



    /*
        |-----------|
        |   x   x   |
        |           |
        |   x   x   | 
        |-----------|
    
        |-----dx----|
    */

    //try parallel build
    std::vector<minimum_FLIP_particle> addition_particles=tbb::parallel_reduce(
        tbb::blocked_range3d<int>(0,nijk[2],0,nijk[1],0,nijk[0]),
        std::vector<minimum_FLIP_particle>{},
        [&](const tbb::blocked_range3d<int>& r, const std::vector<minimum_FLIP_particle>& init) {
            std::vector<minimum_FLIP_particle> result = init;
            // emitter jitter
            std::random_device rd;  //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution<> distrib(-10, 10);

            for (auto pageit = r.pages().begin(); pageit != r.pages().end(); ++pageit) {
                for (auto rowit = r.rows().begin(); rowit != r.rows().end(); ++rowit) {
                    for (auto colit = r.cols().begin(); colit != r.cols().end(); ++colit) {
						LosTopos::Vec3f pos = LosTopos::Vec3f(colit, rowit, pageit) * gap + bmin +  LosTopos::Vec3f(0.5f, 0.5f, 0.5f) * gap;

                        pos[0] += (float(distrib(gen)) * 0.045) * gap;
                        pos[1] += (float(distrib(gen)) * 0.045) * gap;
                        pos[2] += (float(distrib(gen)) * 0.045) * gap;
						if (phi(pos) <= 0 && pos[0] >= bmin[0] && pos[1] >= bmin[1] && pos[2] >= bmin[2]
							&& pos[0] < bmax[0] && pos[1] < bmax[1] && pos[2] < bmax[2])
							result.push_back(minimum_FLIP_particle(pos, LosTopos::Vec3f(0, 0, 0)));
                    }
                }
            }
            return result;
        },
        [&](const std::vector<minimum_FLIP_particle>& a, const std::vector<minimum_FLIP_particle>& b) {
            std::vector<minimum_FLIP_particle> result;
            result.reserve(a.size() + b.size());
            result.insert(result.end(), a.begin(), a.end());
            result.insert(result.end(), b.begin(), b.end());
            return result;
        }
        );
    particles.reserve(particles.size() + addition_particles.size());
    particles.insert(particles.end(), addition_particles.begin(), addition_particles.end());
}
void FluidSim::init_domain()
{
    (*eulerian_fluids).initialize_bulks(particles, dx);//, /*use_hard_bminmax*/ true, m_flip_options.m_exterior_boundary_min, m_flip_options.m_exterior_boundary_max);
}
void FluidSim::emitFluids(float dt, float(*phi)(const LosTopos::Vec3f&))
{
	for(int k = 0;k<emitters.size();k++)
	{
		boxEmitter emitter = emitters[k];
		int slices = floor(dt*mag(emitter.vel) / dx)*2 + 1;
		LosTopos::Vec3f b = emitter.bmax - emitter.bmin;
		int w = floor(b[0] / dx * 2.0); int h = floor(b[1] / dx * 2.0);
		int n = particles.size();
		particles.resize(particles.size() + slices*w*h);
		int totalCnt = 0;
		for (int i = 0;i < slices;i++)
		{
			int cnt = 0;
			for(int jj=0;jj<h;jj++)for(int ii=0;ii<w;ii++)
			{
				LosTopos::Vec3f pos = emitter.bmin + LosTopos::Vec3f(((float)ii+0.5)/(float)w*b[0],((float)jj+0.5)/(float)h*b[1],0) + (float)i / (float)slices*dt*emitter.vel;
				cnt++;
				particles[n + totalCnt] = minimum_FLIP_particle(pos, emitter.vel);
				totalCnt++;
			}
		}
		printf("emitted %d particles\n", totalCnt);
	}
	std::vector<minimum_FLIP_particle> newParticles;
	newParticles.resize(particles.size());
	int cnt = 0;
	for (size_t i = 0;i < particles.size();i++)
	{
		LosTopos::Vec3f pos = particles[i].pos;
		if (phi(pos) >= 0 && pos[0] >= regionMin[0] && pos[1] >= regionMin[1] && pos[2] >= regionMin[2]
			&& pos[0] <= regionMax[0] && pos[1] <= regionMax[1] && pos[2] <= regionMax[2])
		{
			newParticles[cnt++] = particles[i];
		}
	}
	newParticles.resize(cnt);
	particles.swap(newParticles);
    std::vector<minimum_FLIP_particle> temp;
	newParticles.swap(temp);

}
bool FluidSim::isIsolatedParticle(LosTopos::Vec3f &pos) {
    float phi = (*eulerian_fluids).get_liquid_phi(pos);
//    LosTopos::Vec3f pos2 = pos - (*eulerian_fluids).bmin;
//    LosTopos::Vec3i ijk = LosTopos::Vec3i(pos2/dx);
//    int bulkidx=(*eulerian_fluids).find_bulk(ijk[0],ijk[1],ijk[2]);
//    int cnt=0;
//    for(int k=ijk[2]-1;k<=ijk[2]+1;k++)
//        for(int j=ijk[1]-1;j<=ijk[1]+1;j++)
//            for(int i=ijk[0]-1;i<=ijk[0]+1;i++)
//            {
//
//                if((*eulerian_fluids).liquid_phi(bulkidx,i,j,k)<0)
//                {
//                    cnt++;
//                }
//            }
    return phi>0;
}
void FluidSim::advect_particles(float dt)
{
	//particles are already sorted to maximize RAM hit rate
	tbb::parallel_for((size_t)0,
					  (size_t)particles.size(),
					  (size_t)1,
					  [&](size_t index)
	{

		LosTopos::Vec3f pos = particles[index].pos;
        pos += dt*(*eulerian_fluids).get_velocity(pos);

//        if((*eulerian_fluids).isValidVel(pos)&&(!(*eulerian_fluids).isIsolated(pos))) {
//
//            pos = trace_rk3(particles[index].pos, dt);
//
//        } else{
//            particles[index].vel +=  0.5f*dt*LosTopos::Vec3f(0,-9.8,0);
//            pos += dt*particles[index].vel;
//            particles[index].vel +=  0.5f*dt*LosTopos::Vec3f(0,-9.8,0);
//        }


        float phi_val = (*eulerian_fluids).get_solid_phi(pos);
        if (phi_val < 0) {
            LosTopos::Vec3f grad;
            grad = (*eulerian_fluids).get_grad_solid(pos);
            if (mag(grad) > 0)
                normalize(grad);
            pos -= phi_val * grad;
//            if (pos[0] != pos[0] || pos[1] != pos[1] || pos[2] != pos[2]) std::cout << phi_val << " " << grad << std::endl;
        }
        particles[index].pos = pos;
	});

}
void FluidSim::resolveParticleBoundaryCollision()
{
	//particles are already sorted to maximize RAM hit rate
	tbb::parallel_for((size_t)0,
		(size_t)particles.size(),
		(size_t)1,
		[&](size_t index)
	{
		LosTopos::Vec3f pos = particles[index].pos;
		//check boundaries and project exterior particles back in
		float phi_val = (*eulerian_fluids).get_solid_phi(pos);
		if (phi_val < 0) {
			LosTopos::Vec3f grad;
			grad = (*eulerian_fluids).get_grad_solid(particles[index].pos);
			if (mag(grad) > 0)
				normalize(grad);
			pos -= phi_val * grad;
		}
	});
}
void FluidSim::FLIP_advection(float dt)
{
	//for all eulerian bulks, u_coef.zero, u_delta.zero;
	tbb::parallel_for((size_t)0,
					  (size_t)(*eulerian_fluids).n_bulks,
					  (size_t)1,
					  [&](size_t index)
	{
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].u_delta.data[i] = 0;
		}
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].v_delta.data[i] = 0;
		}
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].w_delta.data[i] = 0;
		}
	});

	//compute delta,
	//for all eulerian bulks, compute u_delta
	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].u_delta.data[i] =
				(*eulerian_fluids).fluid_bulk[index].u.data[i] -
				(*eulerian_fluids).fluid_bulk[index].u_save.data[i];
		}
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].v_delta.data[i] =
				(*eulerian_fluids).fluid_bulk[index].v.data[i] -
				(*eulerian_fluids).fluid_bulk[index].v_save.data[i];
		}
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].w_delta.data[i] =
				(*eulerian_fluids).fluid_bulk[index].w.data[i] -
				(*eulerian_fluids).fluid_bulk[index].w_save.data[i];
		}
	});

	//for each particle, p.vel = alpha*Interp(U) + (1-alpha)*(U_p + Interp(dU))
	particle_interpolate(0.01);
	//move particle
	float t = 0;
	float substep = dt;
	while(t < dt) {

		if(t + substep > dt)
			substep = dt - t;
		advect_particles(substep);
		t+=substep;
	}

	//

}
void FluidSim::reorder_particles()
{
    //try atomic
//    float *res =
//    std::vector<std::vector<minimum_FLIP_particle>> particle_reorder;
//    particle_reorder.resize((*eulerian_fluids).n_bulks);
//    for (uint i=0;i<particle_reorder.size();i++)
//    {
//        particle_reorder[i].resize(0);
//    }
//
//    for (uint i=0;i<particles.size();i++)
//    {
//        LosTopos::Vec3f pos = particles[i].pos - (*eulerian_fluids).bmin;
//        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos/(*eulerian_fluids).bulk_size);
//        uint64 idx = (*eulerian_fluids).get_bulk_index(bulk_ijk[0],bulk_ijk[1],bulk_ijk[2]);
//        particle_reorder[idx].push_back(particles[i]);
//    }
//    particles.resize(0);
//    particles.shrink_to_fit();
//    for (uint i=0;i<particle_reorder.size();i++)
//    {
//        for (uint ii=0;ii<particle_reorder[i].size();ii++)
//        {
//            particles.push_back(particle_reorder[i][ii]);
//        }
//    }
//    particle_reorder=std::vector<std::vector<minimum_FLIP_particle>>();
//
//    particle_bulks.resize(0);
//    particle_bulks.resize((*eulerian_fluids).n_bulks);
//    std::vector<std::vector<int64>> particle_bulk_idx;
//    particle_bulk_idx.resize(particles.size());
//
//    for (uint i=0;i<particles.size();i++)
//
//    {
//        particle_bulk_idx[i].reserve(8);
//    };
//
//    tbb::parallel_for((size_t)0, (size_t)particle_bulk_idx.size(),
//                      (size_t)1, [&](size_t p)
//                      {
//                          LosTopos::Vec3f pos = particles[p].pos-(*eulerian_fluids).bmin;
//                          int i = floor(pos[0]/dx);
//                          int j = floor(pos[1]/dx);
//                          int k = floor(pos[2]/dx);
//                          std::unordered_map<int64, int64> bulks_this_particle_is_assigned_to;
//                          bulks_this_particle_is_assigned_to.clear();
//                          int64 n=0;
//                          //a particle's index can be assigned to different bulks
//                          //but can only be assigned to one bulk once.
//                          for (int kk=k-1;kk<=k+1;kk++)
//                              for (int jj=j-1;jj<=j+1;jj++)
//                                  for (int ii=i-1;ii<=i+1;ii++)
//                                  {
//                                      int64 bulk_index = (*eulerian_fluids).find_bulk(ii,jj,kk);
//                                      if (bulks_this_particle_is_assigned_to.find(bulk_index)==bulks_this_particle_is_assigned_to.end())
//                                      {
//                                          //particle_bulks[bulk_index].push_back(p);
//                                          particle_bulk_idx[p].push_back(bulk_index);
//                                          bulks_this_particle_is_assigned_to[bulk_index] = n;
//                                          n++;
//                                      }
//                                  }
//                          bulks_this_particle_is_assigned_to.clear();
//                      });
//
//    for (size_t i=0;i<particles.size();i++)
//    {
//        for (int ii=0;ii<particle_bulk_idx[i].size();ii++)
//        {
//            particle_bulks[particle_bulk_idx[i][ii]].push_back(i);
//        }
//
//    }
//
//    particle_bulk_idx.resize(0);
//    particle_bulk_idx.shrink_to_fit();
}


void FluidSim::sort_particle_by_bulks_reducer:: operator()(const tbb::blocked_range<size_t>& r) {
    //r is the range of particles
    for (auto i = r.begin(); i != r.end(); i++) {
        //identify which bulk idx this particle belongs to
        //it is possible that the fluid may be out of the domain
        //hence delete it completely if it is beyond [bmin,bmax]
        const auto& p = m_unsorted_particles[i].pos;

        bool is_in_all_bulks = true;
        for (int j = 0; j < 3; j++) {
            if (p[j] <= (*m_sparse_bulk).bmin[j]) {
                is_in_all_bulks = false;
                break;
            }
            if (p[j] >= (*m_sparse_bulk).bmax[j]) {
                is_in_all_bulks = false;
                break;
            }
        }

        if (is_in_all_bulks) {
            LosTopos::Vec3f pos = p - (*m_sparse_bulk).bmin;
            LosTopos::Vec3i global_ijk = LosTopos::floor(pos/dx);
            LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i((int)floor(pos[0] / (*m_sparse_bulk).bulk_size),
                (int)floor(pos[1] / (*m_sparse_bulk).bulk_size),
                (int)floor(pos[2] / (*m_sparse_bulk).bulk_size));
            LosTopos::Vec3i local_ijk = global_ijk - bulk_ijk * 8;
            //LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(LosTopos::floor(pos / (*m_sparse_bulk).bulk_size));
            if ((*m_sparse_bulk).index_mapping.find(LosTopos::Vec3i{ bulk_ijk[0], bulk_ijk[1], bulk_ijk[2] })
                !=m_sparse_bulk->index_mapping.end()) {
                uint64 idx = (*m_sparse_bulk).get_bulk_index(bulk_ijk[0], bulk_ijk[1], bulk_ijk[2]);

                bool in_solid = false;
                bool any_solid = m_sparse_bulk->fluid_bulk[idx].any_solid(local_ijk);
                if (any_solid) {
                    float phi_value = 10;
                    for (int j = 0; j < mesh_vec.size(); j++)
                    {
                        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*mesh_vec[j].sdf);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(p[0], p[1], p[2]));
                        phi_value = std::min(phi_value, sdf_value);
                    }
                    if (phi_value < 0) {
                        in_solid = true;
                    }
                }

                if ((m_sparse_bulk->fluid_bulk[idx].eight_solid(local_ijk)==0)) {
                    if (!in_solid) {
                        m_bin_bulk_particle[idx].push_back(i);
                    } 
                }
                else {
                    //printf("particle %d cleaned because inside solid\n", i);
                }
            }
            
        }
    }
}


void FluidSim::assign_particle_to_bulks()
{
    tbb::atomic<size_t> full_solid_counter{0};
    //mark the voxels that are purely inside solid
    tbb::parallel_for(size_t(0), eulerian_fluids->fluid_bulk.size(), [&](size_t bulkidx) {
        eulerian_fluids->fluid_bulk[bulkidx].eight_solid.data.assign(512, (unsigned char)0);
        eulerian_fluids->fluid_bulk[bulkidx].any_solid.data.assign(512, (unsigned char)0);
        for (int i = 0; i < eulerian_fluids->loop_order.size(); i++) {
            const auto& ijk = eulerian_fluids->loop_order[i];
            unsigned char all_solid = 1;
            unsigned char any_solid = 0;
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0; jj < 2 ; jj++) {
                    for (int kk = 0; kk < 2 ; kk++) {
                        if (!(eulerian_fluids->solid_phi(bulkidx, ijk[0] + ii, ijk[1] + jj, ijk[2] + kk) < 0)) {
                            all_solid = 0;
                        }
                        else {
                            any_solid = 1;
                        }
                    }
                }
            }

            eulerian_fluids->fluid_bulk[bulkidx].eight_solid(ijk) = all_solid;
            eulerian_fluids->fluid_bulk[bulkidx].any_solid(ijk) = any_solid;
            if (all_solid) {
                full_solid_counter++;
            }
        }
        });
    std::cout<< "all solid voxel count: "<< full_solid_counter<<std::endl;



    //for each bulk, find what particles are in it
    sort_particle_by_bulks_reducer reducer(
        particles,
        eulerian_fluids->fluid_bulk.size(),
        eulerian_fluids, dx, mesh_vec);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, particles.size()), reducer);
    printf("sort complete\n");
    size_t re_count = 0;
    size_t empty_bulk_count = 0;
    particle_bulks = reducer.m_bin_bulk_particle;
    for (int i = 0; i < particle_bulks.size(); i++) {
        re_count += particle_bulks[i].size();
        if (particle_bulks[i].size() == 0) {
            empty_bulk_count++;
            //printf("empty bulk, bulkid:%d\n", i);
            //std::cout << "empty bulk tile corner " << eulerian_fluids->fluid_bulk[i].tile_corner << std::endl;
        }
    }
    //printf("recount:%d, original count:%d empty bulk count:%d\n", re_count, particles.size(),empty_bulk_count);

    //std::vector<minimum_FLIP_particle> old_minimum_FLIP_particles = particles;
    ////re-order the original FLIP particle array
    //tbb::parallel_for(size_t(0), particle_bulks.size(), [&](size_t bulk_idx) {
    //    size_t pidx_begin = 0;
    //    for (size_t i = 0; i < bulk_idx; i++ ) {
    //        pidx_begin += particle_bulks[i].size();
    //    }
    //    for (size_t i = 0; i < particle_bulks[bulk_idx].size(); i++) {
    //        particles[pidx_begin + i] = old_minimum_FLIP_particles[particle_bulks[bulk_idx][i]];
    //    }
    //    });
}
void FluidSim::particle_to_grid()
{
	//particle to grid
	//for all eulerian bulk u.zero;
	tbb::parallel_for((size_t)0,
					  (size_t)(*eulerian_fluids).n_bulks,
					  (size_t)1,
					  [&](size_t index)
	{
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].u.data[i] = 0;
			(*eulerian_fluids).fluid_bulk[index].v.data[i] = 0;
			(*eulerian_fluids).fluid_bulk[index].w.data[i] = 0;
			(*eulerian_fluids).fluid_bulk[index].u_coef.data[i] = 1e-4;
			(*eulerian_fluids).fluid_bulk[index].v_coef.data[i] = 1e-4;
			(*eulerian_fluids).fluid_bulk[index].w_coef.data[i] = 1e-4;
		}

	});
	//try atomic
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].global_index.data[i] = index*512 + i;
                          }

                      });

    int n = (*eulerian_fluids).n_bulks*(*eulerian_fluids).n_perbulk;
    float *resu = new float[n];
    float *resv = new float[n];
    float *resw = new float[n];
    float *reswu = new float[n];
    float *reswv = new float[n];
    float *resww = new float[n];
    memset(resu,0, sizeof(float)*n);
    memset(resv,0, sizeof(float)*n);
    memset(resw,0, sizeof(float)*n);
    memset(reswu,0,sizeof(float)*n);
    memset(reswv,0,sizeof(float)*n);
    memset(resww,0,sizeof(float)*n);

    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        LosTopos::Vec3f pos = particles[i].pos
                    - ((*eulerian_fluids).bmin + dx * LosTopos::Vec3f(0.0, 0.5, 0.5));
        LosTopos::Vec3f vel = particles[i].vel;
        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos / (8 * (*eulerian_fluids).h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos / (*eulerian_fluids).h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk = particle_local_ijk[2]; kk <= particle_local_ijk[2]+1; kk++)
            for (int jj = particle_local_ijk[1]; jj <= particle_local_ijk[1]+1; jj++)
                for (int ii = particle_local_ijk[0]; ii <= particle_local_ijk[0]+1; ii++) {
                    int bulkidx = (*eulerian_fluids).find_bulk(particle_ijk[0], particle_ijk[1], particle_ijk[2]);
                    int mem_idx = (*eulerian_fluids).global_index(bulkidx, ii, jj, kk);

                    LosTopos::Vec3f sample_pos_u
                            = dx * LosTopos::Vec3f(8*bulk_ijk + LosTopos::Vec3i(ii, jj, kk));

                    float weight0 = FluidSim::compute_weight(sample_pos_u, pos, dx);
                    #pragma omp atomic
                    resu[mem_idx] += (float) (weight0 * vel[0]);
                    #pragma omp atomic
                    reswu[mem_idx] += (float)weight0;

                }
    }



    #pragma omp parallel for
    for(int i=0;i<particles.size();i++)
    {
        LosTopos::Vec3f pos = particles[i].pos
                    - ((*eulerian_fluids).bmin+dx*LosTopos::Vec3f(0.5,0.0,0.5));
        LosTopos::Vec3f vel = particles[i].vel;
        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos/(8*(*eulerian_fluids).h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/(*eulerian_fluids).h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
            for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
                for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
                {
                    int bulkidx = (*eulerian_fluids).find_bulk(particle_ijk[0],particle_ijk[1],particle_ijk[2]);
                    int mem_idx = (*eulerian_fluids).global_index(bulkidx,ii,jj,kk);

                    LosTopos::Vec3f sample_pos_v
                            = dx*LosTopos::Vec3f(8*bulk_ijk+LosTopos::Vec3i(ii,jj,kk));

                    float weight1 = FluidSim::compute_weight(sample_pos_v,pos,dx);
                    #pragma omp atomic
                    resv[mem_idx] += (float)(weight1*vel[1]);
                    #pragma omp atomic
                    reswv[mem_idx] += (float)weight1;


                }
    }


#pragma omp parallel for
    for(int i=0;i<particles.size();i++)
    {
        LosTopos::Vec3f pos = particles[i].pos
                    - ((*eulerian_fluids).bmin+dx*LosTopos::Vec3f(0.5,0.5,0.0));
        LosTopos::Vec3f vel = particles[i].vel;
        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos/(8*(*eulerian_fluids).h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/(*eulerian_fluids).h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
            for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
                for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
                {
                    int bulkidx = (*eulerian_fluids).find_bulk(particle_ijk[0],particle_ijk[1],particle_ijk[2]);
                    int mem_idx = (*eulerian_fluids).global_index(bulkidx,ii,jj,kk);

                    LosTopos::Vec3f sample_pos_w
                            = dx*LosTopos::Vec3f(8*bulk_ijk+LosTopos::Vec3i(ii,jj,kk));

                    float weight2 = FluidSim::compute_weight(sample_pos_w,pos,dx);
#pragma omp atomic
                    resw[mem_idx] += (float)(weight2*vel[2]);
#pragma omp atomic
                    resww[mem_idx] += (float)weight2;


                }
    }






	//spread particle velocity to grid
//	tbb::parallel_for((size_t)0,
//		(size_t)particle_bulks.size(),
//		(size_t)1,
//		[&](size_t index)
//	{
//
//		for (uint i=0;i<particle_bulks[index].size();i++)
//		{
//			LosTopos::Vec3f pos = particles[particle_bulks[index][i]].pos
//				- (*eulerian_fluids).bmin;
//			LosTopos::Vec3f vel = particles[particle_bulks[index][i]].vel;
//			LosTopos::Vec3i bulk_ijk = (*eulerian_fluids).fluid_bulk[index].tile_corner;
//			LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/(*eulerian_fluids).h);
//			LosTopos::Vec3i particle_local_ijk = particle_ijk - bulk_ijk;
//			for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
//			for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
//			for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
//			{
//				if (kk>=0&&kk<(*eulerian_fluids).tile_n
//					&&jj>=0&&jj<(*eulerian_fluids).tile_n
//					&&ii>=0&&ii<(*eulerian_fluids).tile_n)
//				{
//					LosTopos::Vec3f sample_pos_u
//						= dx*LosTopos::Vec3f(bulk_ijk+LosTopos::Vec3i(ii,jj,kk))+dx*LosTopos::Vec3f(0.0,0.5,0.5);
//					LosTopos::Vec3f sample_pos_v
//						= dx*LosTopos::Vec3f(bulk_ijk+LosTopos::Vec3i(ii,jj,kk))+dx*LosTopos::Vec3f(0.5,0.0,0.5);
//					LosTopos::Vec3f sample_pos_w
//						= dx*LosTopos::Vec3f(bulk_ijk+LosTopos::Vec3i(ii,jj,kk))+dx*LosTopos::Vec3f(0.5,0.5,0.0);
//
//					float weight0 = FluidSim::compute_weight(sample_pos_u,pos,dx);
//					float weight1 = FluidSim::compute_weight(sample_pos_v,pos,dx);
//					float weight2 = FluidSim::compute_weight(sample_pos_w,pos,dx);
//					(*eulerian_fluids).u(index,ii,jj,kk) += weight0*vel[0];
//					(*eulerian_fluids).u_coef(index,ii,jj,kk) += weight0;
//
//					(*eulerian_fluids).v(index,ii,jj,kk) += weight1*vel[1];
//					(*eulerian_fluids).v_coef(index,ii,jj,kk) += weight1;
//
//					(*eulerian_fluids).w(index,ii,jj,kk) += weight2*vel[2];
//					(*eulerian_fluids).w_coef(index,ii,jj,kk) += weight2;
//				}
//			}
//		}
//
//	});
//
//	//divide_weight(u,u_coef);
//	//divide_weight(v,v_coef);
//	//divide_weight(w,w_coef);
	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
		    int gidx = (*eulerian_fluids).fluid_bulk[index].global_index.data[i];
		    if(reswu[gidx]>0){
                (*eulerian_fluids).fluid_bulk[index].u.data[i] = resu[gidx]/reswu[gidx];
                (*eulerian_fluids).fluid_bulk[index].u_valid.data[i] = 1;
		    }else
            {
                (*eulerian_fluids).fluid_bulk[index].u.data[i] = 0;
                (*eulerian_fluids).fluid_bulk[index].u_valid.data[i] = 0;
            }
            if(reswv[gidx]>0) {
                (*eulerian_fluids).fluid_bulk[index].v.data[i] = resv[gidx] / reswv[gidx];
                (*eulerian_fluids).fluid_bulk[index].v_valid.data[i] = 1;
            }
            else
            {
                (*eulerian_fluids).fluid_bulk[index].v.data[i] = 0;
                (*eulerian_fluids).fluid_bulk[index].v_valid.data[i] = 0;
            }
            if(resww[gidx]>0) {
                (*eulerian_fluids).fluid_bulk[index].w.data[i] = resw[gidx] / resww[gidx];
                (*eulerian_fluids).fluid_bulk[index].w_valid.data[i] = 1;
            }
            else
            {
                (*eulerian_fluids).fluid_bulk[index].w.data[i] = 0;
                (*eulerian_fluids).fluid_bulk[index].w_valid.data[i] = 0;
            }

		}

	});




    delete []resu ;
    delete []resv ;
    delete []resw ;
    delete []reswu;
    delete []reswv;
    delete []resww;
}
void FluidSim::compute_phi()
{



    //make use of vdb
//    openvdb::FloatGrid::Ptr levelset = vdbToolsWapper::particleToLevelset(particles, particle_radius, 0.5*dx);
//    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*levelset);
//
    //try atomic
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].global_index.data[i] = index*512 + i;
                          }

                      });

    int n = (*eulerian_fluids).n_bulks*(*eulerian_fluids).n_perbulk;
    float *res = new float[n];
    float *resw = new float[n];
    memset(res,0, sizeof(float)*n);
    memset(resw,0, sizeof(float)*n);
//    tbb::parallel_for((size_t)0,
//                      (size_t)(*eulerian_fluids).n_bulks,
//                      (size_t)1,
//                      [&](size_t index)
//                      {
//                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
//                          {
//                              int gidx = (*eulerian_fluids).fluid_bulk[index].global_index.data[i];
//                              res[gidx] = 3.0f*dx;
//                              (*eulerian_fluids).fluid_bulk[index].liquid_phi.data[i] = 3.0f*dx;
//
//                          }
//
//                      });
#pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        LosTopos::Vec3f pos = particles[i].pos
                    - ((*eulerian_fluids).bmin + dx * LosTopos::Vec3f(0.5, 0.5, 0.5));

        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos / (8 * (*eulerian_fluids).h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos / (*eulerian_fluids).h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk = particle_local_ijk[2]-1; kk <= particle_local_ijk[2]+1; kk++)
            for (int jj = particle_local_ijk[1]-1; jj <= particle_local_ijk[1]+1; jj++)
                for (int ii = particle_local_ijk[0]-1; ii <= particle_local_ijk[0]+1; ii++) {
                    int bulkidx = (*eulerian_fluids).find_bulk(particle_ijk[0], particle_ijk[1], particle_ijk[2]);
                    int mem_idx = (*eulerian_fluids).global_index(bulkidx, ii, jj, kk);

                    LosTopos::Vec3f sample_pos
                            = dx * LosTopos::Vec3f(8*bulk_ijk + LosTopos::Vec3i(ii, jj, kk));

                    float weight0 = FluidSim::compute_weight(sample_pos, pos, dx);
#pragma omp atomic
//                    res[mem_idx] += (float) (weight0 * (-particle_radius));
                    res[mem_idx] -= weight0 * particle_radius;
#pragma omp atomic
                    resw[mem_idx] += (float) weight0;


                }
    }
#pragma omp parallel for
    for(int i=0;i<n;i++)
    {

        res[i] = res[i] /(1e-6+resw[i]);
        //std::cout<<res[i]<<std::endl;

    }
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              int gidx = (*eulerian_fluids).fluid_bulk[index].global_index.data[i];

//                              if (resw[gidx] == 0) (*eulerian_fluids).fluid_bulk[index].liquid_phi.data[i] = 3.0 * dx;
//                              else (*eulerian_fluids).fluid_bulk[index].liquid_phi.data[i] = res[gidx];
                              (*eulerian_fluids).fluid_bulk[index].liquid_phi.data[i] = 0.5*dx + 2.0*res[gidx];


                          }

                      });
//    tbb::parallel_for((size_t)0,
//                      (size_t)(*eulerian_fluids).n_bulks,
//                      (size_t)1,
//                      [&](size_t index)
//                      {
//                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
//                          {
//                              int gidx = (*eulerian_fluids).fluid_bulk[index].global_index.data[i];
//                              if(res[gidx]==0)
//                              {
//                                  float value = dx;
//                                  LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
//                                  for(int kk = ijk[2]-1;kk<=ijk[2]+1;kk++)for(int jj = ijk[1]-1;jj<=ijk[1]+1;jj++)for(int ii = ijk[0]-1; ii<=ijk[0]+1; ii++)
//                                          {
//                                              if((*eulerian_fluids).liquid_phi(index, ii,jj,kk)<0)
//                                              {
//                                                  value = min(value, dx+(*eulerian_fluids).liquid_phi(index, ii,jj,kk));
//                                              }
//                                          }
//
//
//                              }
//
//                          }
//
//                      });




//    tbb::parallel_for((size_t)0,
//                      (size_t)(*eulerian_fluids).fluid_bulk.size(),
//                      (size_t)1,
//                      [&](size_t index)
//                      {
//                          for (uint i=0;i<(*eulerian_fluids).n_perbulk;i++)
//                          {
//                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
//                              LosTopos::Vec3i bulk_ijk = (*eulerian_fluids).fluid_bulk[index].tile_corner;
//                              LosTopos::Vec3f sample_pos = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f(bulk_ijk+ijk)+dx*LosTopos::Vec3f(0.5,0.5,0.5);
//                              float sdf_value = box_sampler.wsSample(openvdb::Vec3R(sample_pos[0], sample_pos[1], sample_pos[2]));
//                              (*eulerian_fluids).liquid_phi(index, ijk[0],ijk[1],ijk[2]) = 3.0*dx;
//                              if(sdf_value<3.0*dx)
//                                  (*eulerian_fluids).liquid_phi(index, ijk[0],ijk[1],ijk[2]) = sdf_value;
//                              (*eulerian_fluids).liquid_phi(index, ijk[0],ijk[1],ijk[2]) += particle_radius;
//
//
//                          }
//
//                      });
//
//    for(int i=0;i<particles.size();i++)
//    {
//        LosTopos::Vec3f pos = particles[i].pos
//                    - (*eulerian_fluids).bmin;
//        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos/(8*(*eulerian_fluids).h));
//        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/(*eulerian_fluids).h);
//        LosTopos::Vec3i particle_local_ijk = particle_ijk - bulk_ijk;
//        for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
//            for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
//                for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
//                {
//                    int bulkidx = (*eulerian_fluids).find_bulk(particle_ijk[0],particle_ijk[1],particle_ijk[2]);
//                    int mem_idx = (*eulerian_fluids).global_index(bulkidx,ii,jj,kk);
//                    if(mem_idx>=0&&mem_idx<n)
//                    {
//                        LosTopos::Vec3f sample_pos = dx*LosTopos::Vec3f(bulk_ijk+LosTopos::Vec3i(ii,jj,kk))+dx*LosTopos::Vec3f(0.5,0.5,0.5);
//                        {
//                            float liquid_phi = res[mem_idx];
//                            float test = dist(sample_pos, pos) - particle_radius;
//                            if(test<liquid_phi) {
//                                res[mem_idx] = test;
//                            }
//                        }
//
//
//                    }
//
//                }
//    }
//    tbb::parallel_for((size_t)0,
//                      (size_t)(*eulerian_fluids).n_bulks,
//                      (size_t)1,
//                      [&](size_t index)
//                      {
//                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
//                          {
//                              int gidx = (*eulerian_fluids).fluid_bulk[index].global_index.data[i];
//                              (*eulerian_fluids).fluid_bulk[index].liquid_phi.data[i] = res[gidx];
//
//                          }
//
//                      });


//	//provided all particles sorted in bulks
//	//for all bulks in parallel, for all particles
//	//in this bulk, spread their quantities to
//	//the grid of this bulk.
//	tbb::parallel_for((size_t)0,
//		              (size_t)particle_bulks.size(),
//					  (size_t)1,
//					  [&](size_t index)
//	{
//
//		for (uint i=0;i<particle_bulks[index].size();i++)
//		{
//			LosTopos::Vec3f pos = particles[particle_bulks[index][i]].pos
//				- (*eulerian_fluids).bmin;
//			LosTopos::Vec3i bulk_ijk = (*eulerian_fluids).fluid_bulk[index].tile_corner;
//			LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/(*eulerian_fluids).h);
//			LosTopos::Vec3i particle_local_ijk = particle_ijk - bulk_ijk;
//			for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
//			for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
//			for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
//			{
//				if (kk>=0&&kk<(*eulerian_fluids).tile_n
//				  &&jj>=0&&jj<(*eulerian_fluids).tile_n
//				  &&ii>=0&&ii<(*eulerian_fluids).tile_n)
//				{
//					float liquid_phi = (*eulerian_fluids).liquid_phi(index,ii,jj,kk);
//					LosTopos::Vec3f sample_pos = dx*LosTopos::Vec3f(bulk_ijk+LosTopos::Vec3i(ii,jj,kk))+dx*LosTopos::Vec3f(0.5,0.5,0.5);
//					float test_val = dist(sample_pos,pos)-particle_radius;
//					if(test_val<liquid_phi)
//					{
//						(*eulerian_fluids).liquid_phi(index,ii,jj,kk) = test_val;
//					}
//				}
//			}
//		}
//
//	});

    //extend liquids slightly into solids
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).fluid_bulk.size(),
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (uint i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                              if((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])<0.5*dx)
                              {
                                  float solid_phi = 0.125*((*eulerian_fluids).solid_phi(index,ijk[0],ijk[1],ijk[2])
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0]+1,ijk[1],ijk[2])
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0],ijk[1]+1,ijk[2])
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0]+1,ijk[1]+1,ijk[2])
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0],ijk[1],ijk[2]+1)
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0]+1,ijk[1],ijk[2]+1)
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0],ijk[1]+1,ijk[2]+1)
                                                           +(*eulerian_fluids).solid_phi(index,ijk[0]+1,ijk[1]+1,ijk[2]+1));
                                  if (solid_phi<0)
                                  {
                                      (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]) = -0.5*dx;
                                  }
                              }
                          }

                      });
    delete []res;
    delete []resw;
}
void FluidSim::set_boundary(float (* phi)(const LosTopos::Vec3f &)) {
    // first update moving solids position
    for (int j = 0; j < mesh_vec.size(); j++)
    {
        mesh_vec[j].updateSolid(total_frame);
    }
    tbb::parallel_for((uint)0,(*eulerian_fluids).n_bulks,(uint)1,[&](uint b) {
        for (int i = 0; i < (*eulerian_fluids).loop_order.size(); i++) {
            LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
            LosTopos::Vec3f pos = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[b].tile_corner + ijk);
            float phi_value = phi(pos);
            for (int j = 0; j < mesh_vec.size(); j++)
            {
                openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*mesh_vec[j].sdf);
                float sdf_value = box_sampler.wsSample(openvdb::Vec3R(pos[0], pos[1], pos[2]));
                phi_value = std::min(phi_value, sdf_value);
            }
            (*eulerian_fluids).solid_phi(b,ijk[0], ijk[1]+1,ijk[2]) = phi_value;
        }
    });
}

void FluidSim::set_boundary(std::function<float(const LosTopos::Vec3f&)> phi)
{
	// first update moving solids position
	for (int j = 0; j < mesh_vec.size(); j++)
	{
		mesh_vec[j].updateSolid(total_frame);
	}

    //general phi
	tbb::parallel_for(tbb::blocked_range<uint>((uint)0, (*eulerian_fluids).n_bulks), [&](const tbb::blocked_range<uint>& r) {
		for (auto b = r.begin(); b != r.end(); b++) {
			for (int i = 0; i < (*eulerian_fluids).loop_order.size(); i++) {
				LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
				LosTopos::Vec3f pos = (*eulerian_fluids).bmin + dx * LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[b].tile_corner + ijk);
				float phi_value = phi(pos);
				(*eulerian_fluids).fluid_bulk[b].solid_phi(ijk) = phi_value;
			}//end for voxel in bulk b
		}//for bulk b in range 

		});

    //secondary phi
	tbb::parallel_for(tbb::blocked_range<uint>((uint)0, (*eulerian_fluids).n_bulks), [&](const tbb::blocked_range<uint>& r) {
		for (int j = 0; j < mesh_vec.size(); j++)
		{
			//openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*mesh_vec[j].sdf);
            auto constaxr = mesh_vec[j].sdf->getConstAccessor();
            auto box_sampler = openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, 
                openvdb::tools::BoxSampler>{ constaxr, mesh_vec[j].sdf->transform() };

			for (auto b = r.begin(); b != r.end(); b++) {

				for (int i = 0; i < (*eulerian_fluids).loop_order.size(); i++) {
					LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
					LosTopos::Vec3f pos = (*eulerian_fluids).bmin + dx * LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[b].tile_corner + ijk);

                    //read 
                    float phi_value = (*eulerian_fluids).fluid_bulk[b].solid_phi(ijk);

					float sdf_value = box_sampler.wsSample(openvdb::Vec3R(pos[0], pos[1], pos[2]));

					if (sdf_value < 0) {
						sdf_value = -0.5 * dx;
					}
					if (sdf_value > 0) {
						sdf_value = 0.5 * dx;
					}

                    //compare
					phi_value = std::min(phi_value, sdf_value);

                    //store
                    (*eulerian_fluids).fluid_bulk[b].solid_phi(ijk) = phi_value;
				}//end for voxel in bulk b
			}//for bulk b in range 
		}//end for object j
		});
}

void FluidSim::particle_to_grid(sparse_fluid8x8x8 &_eulerian_fluid, std::vector<minimum_FLIP_particle> &_particles, float _dx)
{
    tbb::parallel_for((size_t)0,
                      (size_t)_eulerian_fluid.n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                          {
                              _eulerian_fluid.fluid_bulk[index].u.data[i] = 0;
                              _eulerian_fluid.fluid_bulk[index].v.data[i] = 0;
                              _eulerian_fluid.fluid_bulk[index].w.data[i] = 0;
                              _eulerian_fluid.fluid_bulk[index].u_coef.data[i] = 1e-4;
                              _eulerian_fluid.fluid_bulk[index].v_coef.data[i] = 1e-4;
                              _eulerian_fluid.fluid_bulk[index].w_coef.data[i] = 1e-4;
                          }

                      });
    //try atomic
    tbb::parallel_for((size_t)0,
                      (size_t)_eulerian_fluid.n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i = 0;i<_eulerian_fluid.n_perbulk;i++)
                          {
                              _eulerian_fluid.fluid_bulk[index].global_index.data[i] = index*512 + i;
                          }

                      });

    int n = _eulerian_fluid.n_bulks*_eulerian_fluid.n_perbulk;
    float *resu = new float[n];
    float *resv = new float[n];
    float *resw = new float[n];
    float *reswu = new float[n];
    float *reswv = new float[n];
    float *resww = new float[n];
    memset(resu,0, sizeof(float)*n);
    memset(resv,0, sizeof(float)*n);
    memset(resw,0, sizeof(float)*n);
    memset(reswu,0,sizeof(float)*n);
    memset(reswv,0,sizeof(float)*n);
    memset(resww,0,sizeof(float)*n);
#pragma omp parallel for
    for (int i = 0; i < _particles.size(); i++) {
        LosTopos::Vec3f pos = _particles[i].pos
                           - (_eulerian_fluid.bmin + _dx * LosTopos::Vec3f(0.0, 0.5, 0.5));
        LosTopos::Vec3f vel = _particles[i].vel;
        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos / (8 * _eulerian_fluid.h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos / _eulerian_fluid.h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk = particle_local_ijk[2]-1; kk <= particle_local_ijk[2]+1; kk++)
            for (int jj = particle_local_ijk[1]-1; jj <= particle_local_ijk[1]+1; jj++)
                for (int ii = particle_local_ijk[0]-1; ii <= particle_local_ijk[0]+1; ii++) {
                    int bulkidx = _eulerian_fluid.find_bulk(particle_ijk[0], particle_ijk[1], particle_ijk[2]);
                    int mem_idx = _eulerian_fluid.global_index(bulkidx, ii, jj, kk);

                    LosTopos::Vec3f sample_pos_u
                            = _dx * LosTopos::Vec3f(8*bulk_ijk + LosTopos::Vec3i(ii, jj, kk));

                    float weight0 = FluidSim::compute_weight(sample_pos_u, pos, _dx);
#pragma omp atomic
                    resu[mem_idx] += (float) (weight0 * vel[0]); 
#pragma omp atomic
                    reswu[mem_idx] += (float)weight0;


                }
    }

#pragma omp parallel for
    for(int i=0;i<_particles.size();i++)
    {
        LosTopos::Vec3f pos = _particles[i].pos
                           - (_eulerian_fluid.bmin+_dx*LosTopos::Vec3f(0.5,0.0,0.5));
        LosTopos::Vec3f vel = _particles[i].vel;
        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos/(8*_eulerian_fluid.h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/_eulerian_fluid.h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
            for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
                for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
                {
                    int bulkidx = _eulerian_fluid.find_bulk(particle_ijk[0],particle_ijk[1],particle_ijk[2]);
                    int mem_idx = _eulerian_fluid.global_index(bulkidx,ii,jj,kk);

                    LosTopos::Vec3f sample_pos_v
                            = _dx*LosTopos::Vec3f(8*bulk_ijk+LosTopos::Vec3i(ii,jj,kk));

                    float weight1 = FluidSim::compute_weight(sample_pos_v,pos,_dx);
#pragma omp atomic
                    resv[mem_idx] += (float)(weight1*vel[1]);
#pragma omp atomic
                    reswv[mem_idx] += (float)weight1;


                }
    }
#pragma omp parallel for
    for(int i=0;i<_particles.size();i++)
    {
        LosTopos::Vec3f pos = _particles[i].pos
                           - (_eulerian_fluid.bmin+_dx*LosTopos::Vec3f(0.5,0.5,0.0));
        LosTopos::Vec3f vel = _particles[i].vel;
        LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(pos/(8*_eulerian_fluid.h));
        LosTopos::Vec3i particle_ijk = LosTopos::Vec3i(pos/_eulerian_fluid.h);
        LosTopos::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
        for (int kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
            for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
                for (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
                {
                    int bulkidx = _eulerian_fluid.find_bulk(particle_ijk[0],particle_ijk[1],particle_ijk[2]);
                    int mem_idx = _eulerian_fluid.global_index(bulkidx,ii,jj,kk);

                    LosTopos::Vec3f sample_pos_w
                            = _dx*LosTopos::Vec3f(8*bulk_ijk+LosTopos::Vec3i(ii,jj,kk));

                    float weight2 = FluidSim::compute_weight(sample_pos_w,pos,_dx);
#pragma omp atomic
                    resw[mem_idx] += (float)(weight2*vel[2]);
#pragma omp atomic
                    resww[mem_idx] += (float)weight2;


                }
    }


    tbb::parallel_for((size_t)0,
                      (size_t)_eulerian_fluid.n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                          {
                              int gidx = _eulerian_fluid.fluid_bulk[index].global_index.data[i];
                              if(reswu[gidx]>0){
                                  _eulerian_fluid.fluid_bulk[index].u.data[i] = resu[gidx]/reswu[gidx];
                                  _eulerian_fluid.fluid_bulk[index].u_valid.data[i] = 1;
                              }else
                              {
                                  _eulerian_fluid.fluid_bulk[index].u.data[i] = 0;
                                  _eulerian_fluid.fluid_bulk[index].u_valid.data[i] = 0;
                              }
                              if(reswv[gidx]>0) {
                                  _eulerian_fluid.fluid_bulk[index].v.data[i] = resv[gidx] / reswv[gidx];
                                  _eulerian_fluid.fluid_bulk[index].v_valid.data[i] = 1;
                              }
                              else
                              {
                                  _eulerian_fluid.fluid_bulk[index].v.data[i] = 0;
                                  _eulerian_fluid.fluid_bulk[index].v_valid.data[i] = 0;
                              }
                              if(resww[gidx]>0) {
                                  _eulerian_fluid.fluid_bulk[index].w.data[i] = resw[gidx] / resww[gidx];
                                  _eulerian_fluid.fluid_bulk[index].w_valid.data[i] = 1;
                              }
                              else
                              {
                                  _eulerian_fluid.fluid_bulk[index].w.data[i] = 0;
                                  _eulerian_fluid.fluid_bulk[index].w_valid.data[i] = 0;
                              }

                          }

                      });



    delete []resu ;
    delete []resv ;
    delete []resw ;
    delete []reswu;
    delete []reswv;
    delete []resww;
}




void FluidSim::advance(float dt, std::function<float(const LosTopos::Vec3f&)> phi)
{
    float t = 0;
    while (t < dt) {
        cfl_dt = 4.0 * cfl();
        float substep = cfl_dt;
        if (t + substep > dt)
            substep = dt - t;
        printf("Taking substep of size %f (to %0.3f%% of the frame)\n", substep, 100 * (t + substep) / dt);

        printf("FLIP advection\n");
        FLIP_advection(substep);

        printf("handle_boundary\n");
        handle_boundary_layer();

        //printf("reinitialize bulks\n");        
        (*eulerian_fluids).initialize_bulks(particles, dx, /*use_hard_boundary?*/true, m_flip_options.m_exterior_boundary_min, m_flip_options.m_exterior_boundary_max);

        
        set_boundary(phi);

        //requires solid_phi
        printf("p2g\n");
        fusion_p2g_liquid_phi();
       /* particle_to_grid();
        compute_phi();*/


        //Compute finite-volume type face area weight for each velocity sample.
        //requires solid_phi
        compute_weights();

        assign_boundary_layer_solid_velocities();


        //        extrapolate(4);
        //printf("extrapolate\n");
        extrapolate(4);

        tbb::parallel_for((size_t)0,
            (size_t)(*eulerian_fluids).n_bulks,
            (size_t)1,
            [&](size_t index)
            {
                for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
                {
                    (*eulerian_fluids).fluid_bulk[index].u_save.data[i]
                        = (*eulerian_fluids).fluid_bulk[index].u.data[i];
                    (*eulerian_fluids).fluid_bulk[index].v_save.data[i]
                        = (*eulerian_fluids).fluid_bulk[index].v.data[i];
                    (*eulerian_fluids).fluid_bulk[index].w_save.data[i]
                        = (*eulerian_fluids).fluid_bulk[index].w.data[i];
                }

            });
        
        //printf("add gravity\n");
        add_force(substep);
        printf(" Pressure projection\n");
        project(1.0);
        //Pressure projection only produces valid velocities in faces with non-zero associated face area.
        //Because the advection step may interpolate from these invalid faces,
        //we must extrapolate velocities from the fluid domain into these invalid faces.
        //printf(" Extrapolation\n");
        extrapolate(20);
        //For extrapolated velocities, replace the normal component with
        //that of the object.
        //printf(" Constrain boundary velocities\n");
        //constrain_velocity();
        t += substep;
    }
    total_frame++;
}
void FluidSim::add_force(float dt)
{
	tbb::parallel_for((uint)0, (uint)(*eulerian_fluids).n_bulks, (uint)1,[&](uint b){
		for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[b].v.data[i] -= dt*9.81f;
		}
	});
}
void FluidSim::compute_weights()
{
	tbb::parallel_for((uint)0,(*eulerian_fluids).n_bulks,(uint)1,[&](uint b)
	{
		for (int i=0;i<(*eulerian_fluids).loop_order.size();i++)
		{
			LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
			LosTopos::Vec3i bulk_corner = (*eulerian_fluids).fluid_bulk[b].tile_corner;

			(*eulerian_fluids).u_weight(b, ijk[0], ijk[1], ijk[2])
				= 1 - fraction_inside((*eulerian_fluids).solid_phi(b,ijk[0], ijk[1], ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0], ijk[1]+1,ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0], ijk[1], ijk[2]+1),
				(*eulerian_fluids).solid_phi(b,ijk[0], ijk[1]+1,ijk[2]+1));
			(*eulerian_fluids).u_weight(b,ijk[0],ijk[1],ijk[2])
				= LosTopos:: clamp((*eulerian_fluids).u_weight(b,ijk[0],ijk[1],ijk[2]),
				0.0f,1.0f);


			(*eulerian_fluids).v_weight(b, ijk[0], ijk[1], ijk[2])
				= 1 - fraction_inside((*eulerian_fluids).solid_phi(b,ijk[0], ijk[1], ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0], ijk[1],ijk[2]+1),
				(*eulerian_fluids).solid_phi(b,ijk[0]+1, ijk[1], ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0]+1, ijk[1],ijk[2]+1));
			(*eulerian_fluids).v_weight(b,ijk[0],ijk[1],ijk[2])
				= LosTopos:: clamp((*eulerian_fluids).v_weight(b,ijk[0],ijk[1],ijk[2]),
				0.0f,1.0f);


			(*eulerian_fluids).w_weight(b, ijk[0], ijk[1], ijk[2])
				= 1 - fraction_inside((*eulerian_fluids).solid_phi(b,ijk[0], ijk[1], ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0] + 1, ijk[1],ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0], ijk[1] + 1, ijk[2]),
				(*eulerian_fluids).solid_phi(b,ijk[0]+1, ijk[1]+1,ijk[2]));
			(*eulerian_fluids).w_weight(b,ijk[0],ijk[1],ijk[2])
				= LosTopos:: clamp((*eulerian_fluids).w_weight(b,ijk[0],ijk[1],ijk[2]),
				0.0f,1.0f);


		}

	});
}



LosTopos::Vec3f FluidSim::trace_rk3(const LosTopos::Vec3f& position, float dt)
{
//	float c1 = 0.22222222222*dt, c2 = 0.33333333333 * dt, c3 = 0.44444444444 * dt;
//	LosTopos::Vec3f input = position;
//	LosTopos::Vec3f velocity1 = (*eulerian_fluids).get_velocity(input);
//	LosTopos::Vec3f midp1 = input + ((float)(0.5*dt))*velocity1;
//	LosTopos::Vec3f velocity2 = (*eulerian_fluids).get_velocity(midp1);
//	LosTopos::Vec3f midp2 = input + ((float)(0.75*dt))*velocity2;
//	LosTopos::Vec3f velocity3 = (*eulerian_fluids).get_velocity(midp2);
//	//velocity = get_velocity(input + 0.5f*dt*velocity);
//	//input += dt*velocity;
//	input = input + c1*velocity1 + c2*velocity2 + c3*velocity3;
    LosTopos::Vec3f input = position;
    LosTopos::Vec3f vel1  = (*eulerian_fluids).get_velocity(input);
    LosTopos::Vec3f pos1  = input + 0.5f*dt*vel1;
    LosTopos::Vec3f vel2  = (*eulerian_fluids).get_velocity(pos1);
    LosTopos::Vec3f pos2  = input + 0.5f*dt*vel2;
    LosTopos::Vec3f vel3  = (*eulerian_fluids).get_velocity(pos2);
    LosTopos::Vec3f pos3  = input + dt*vel3;
    LosTopos::Vec3f vel4  = (*eulerian_fluids).get_velocity(pos3);

    input = input + 1.0f/6.0f*dt*(vel1 + 2.0f*vel2 + 2.0f*vel3 + vel4);
	return input;
}
void FluidSim::particle_interpolate(float alpha)
{
	//p.v = alpha * Interp(v) + (1-alpha)*(p.v + Interp(dv));
	tbb::parallel_for((size_t)0,
					  (size_t)particles.size(),
					  (size_t)1,
					  [&](size_t index)
	{
		LosTopos::Vec3f pos = particles[index].pos;
		LosTopos::Vec3f pv  = particles[index].vel;
		LosTopos::Vec3f v  = (*eulerian_fluids).get_velocity(pos);
		LosTopos::Vec3f dv = (*eulerian_fluids).get_delta_vel(pos);
		particles[index].vel = alpha * v + (1.0f-alpha)*(pv + dv);
	});

}


void FluidSim::constrain_velocity() {

    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].u_delta.data[i]
                                      =(*eulerian_fluids).fluid_bulk[index].u.data[i];
                              (*eulerian_fluids).fluid_bulk[index].v_delta.data[i]
                                      =(*eulerian_fluids).fluid_bulk[index].v.data[i];
                              (*eulerian_fluids).fluid_bulk[index].w_delta.data[i]
                                      =(*eulerian_fluids).fluid_bulk[index].w.data[i];
                          }

                      });

    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                              LosTopos::Vec3f pos((*eulerian_fluids).fluid_bulk[index].tile_corner+ijk);
                              if ((*eulerian_fluids).u_weight(index,ijk[0],ijk[1],ijk[2])==0)
                              {
                                  LosTopos::Vec3f posu = (pos + LosTopos::Vec3f(0,0.5,0.5))*dx+(*eulerian_fluids).bmin;
                                  LosTopos::Vec3f velu = (*eulerian_fluids).get_velocity(posu);
                                  LosTopos::Vec3f normalu(0,0,0);
                                  normalu = (*eulerian_fluids).get_grad_solid(posu);
                                  normalize(normalu);
                                  float perp_componentu = dot(velu,normalu);
                                  float solid_perp_componentu = (eulerian_fluids->fluid_bulk[index].u_solid(ijk)) * normalu[0];
                                  velu -= perp_componentu*normalu;
                                  velu += solid_perp_componentu * normalu;
                                  (*eulerian_fluids).u_delta(index,ijk[0],ijk[1],ijk[2]) = velu[0];
                              }
                              if ((*eulerian_fluids).v_weight(index,ijk[0],ijk[1],ijk[2])==0)
                              {
                                  LosTopos::Vec3f posv = (pos + LosTopos::Vec3f(0.5,0,0.5))*dx+(*eulerian_fluids).bmin;
                                  LosTopos::Vec3f velv = (*eulerian_fluids).get_velocity(posv);
                                  LosTopos::Vec3f normalv(0,0,0);
                                  normalv = (*eulerian_fluids).get_grad_solid(posv);
                                  normalize(normalv);
                                  float perp_componentv = dot(velv,normalv);
                                  float solid_perp_componentv = (eulerian_fluids->fluid_bulk[index].v_solid(ijk)) * normalv[0];
                                  velv -= perp_componentv*normalv;
                                  velv += solid_perp_componentv * normalv;
                                  (*eulerian_fluids).v_delta(index,ijk[0],ijk[1],ijk[2]) = velv[1];
                              }
                              if ((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2])==0)
                              {
                                  LosTopos::Vec3f posw = (pos + LosTopos::Vec3f(0.5,0.5,0))*dx+(*eulerian_fluids).bmin;
                                  LosTopos::Vec3f velw = (*eulerian_fluids).get_velocity(posw);
                                  LosTopos::Vec3f normalw(0,0,0);
                                  normalw = (*eulerian_fluids).get_grad_solid(posw);
                                  normalize(normalw);
                                  float perp_componentw = dot(velw,normalw);
                                  float solid_perp_componentw = (eulerian_fluids->fluid_bulk[index].w_solid(ijk)) * normalw[0];
                                  velw -= perp_componentw*normalw;
                                  velw += solid_perp_componentw* normalw;
                                  (*eulerian_fluids).w_delta(index,ijk[0],ijk[1],ijk[2]) = velw[2];
                              }
                          }
                      });



    //update
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].u.data[i]
                                      =(*eulerian_fluids).fluid_bulk[index].u_delta.data[i];
                              (*eulerian_fluids).fluid_bulk[index].v.data[i]
                                      =(*eulerian_fluids).fluid_bulk[index].v_delta.data[i];
                              (*eulerian_fluids).fluid_bulk[index].w.data[i]
                                      =(*eulerian_fluids).fluid_bulk[index].w_delta.data[i];
                          }

                      });

}
void FluidSim::solve_pressure_parallel_build(float dt)
{
    std::vector<LosTopos::Vec3i> Bulk_ijk;
	Bulk_ijk.resize((*eulerian_fluids).n_bulks);
	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		Bulk_ijk[index] = (*eulerian_fluids).fluid_bulk[index].tile_corner / 8;
		for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].global_index.data[i] = index*512 + i;
			(*eulerian_fluids).fluid_bulk[index].pressure.data[i] = 0.0;
		}

	});
	Dofs.resize(512*(*eulerian_fluids).n_bulks);

	std::cout << "PPE unkowns:" << Dofs.size() << std::endl;
	matrix.resize(Dofs.size());
	rhs.resize(Dofs.size());
	matrix.zero();
	rhs.assign(rhs.size(), 0);
	Dofs.assign(Dofs.size(), 0);
	std::cout << "begin assemble" << std::endl;
	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
			float centre_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]);
			if (centre_phi<0)
			{
				uint Dof_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
				//right neighbour
				float term = (*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2])*dt / LosTopos:: sqr(dx);

				float right_phi = (*eulerian_fluids).liquid_phi(index, ijk[0] + 1, ijk[1], ijk[2]);
				if (right_phi < 0) {
					matrix.add_to_element(Dof_idx,
						Dof_idx, term);
					matrix.add_to_element(Dof_idx,
						(*eulerian_fluids).global_index(index, ijk[0] + 1, ijk[1], ijk[2]),
						-term);
				}
				else {
					float theta = fraction_inside(centre_phi, right_phi);
					if (theta < 0.01f) theta = 0.01f;
					matrix.add_to_element(Dof_idx,
						Dof_idx, term / theta);
				}
				rhs[Dof_idx] -= (*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2])*
					(*eulerian_fluids).u(index, ijk[0] + 1, ijk[1], ijk[2]) / dx;

				//left neighbour
				term = (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2])*dt / LosTopos:: sqr(dx);

				float left_phi = (*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]);
				if (left_phi < 0) {
					matrix.add_to_element(Dof_idx,
						Dof_idx, term);
					matrix.add_to_element(Dof_idx,
						(*eulerian_fluids).global_index(index, ijk[0] - 1, ijk[1], ijk[2]),
						-term);
				}
				else {
					float theta = fraction_inside(centre_phi, left_phi);
					if (theta < 0.01f) theta = 0.01f;
					matrix.add_to_element(Dof_idx,
						Dof_idx, term / theta);
				}
				rhs[Dof_idx] += (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2])*
					(*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) / dx;

				//top neighbour
				term = (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) * dt / LosTopos:: sqr(dx);
				float top_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] + 1, ijk[2]);
				if (top_phi < 0) {
					matrix.add_to_element(Dof_idx,
						Dof_idx, term);
					matrix.add_to_element(Dof_idx,
						(*eulerian_fluids).global_index(index, ijk[0], ijk[1] + 1, ijk[2]), -term);
				}
				else {
					float theta = fraction_inside(centre_phi, top_phi);
					if (theta < 0.01f) theta = 0.01f;
					matrix.add_to_element(Dof_idx,
						Dof_idx, term / theta);
				}
				rhs[Dof_idx] -= (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2])*
					(*eulerian_fluids).v(index, ijk[0], ijk[1] + 1, ijk[2]) / dx;

				//bottom neighbour
				term = (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]) * dt / LosTopos:: sqr(dx);
				float bot_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]);
				if (bot_phi < 0) {
					matrix.add_to_element(Dof_idx,
						Dof_idx, term);
					matrix.add_to_element(Dof_idx,
						(*eulerian_fluids).global_index(index, ijk[0], ijk[1] - 1, ijk[2]), -term);
				}
				else {
					float theta = fraction_inside(centre_phi, bot_phi);
					if (theta < 0.01f) theta = 0.01f;
					matrix.add_to_element(Dof_idx,
						Dof_idx, term / theta);
				}
				rhs[Dof_idx] += (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2])*
					(*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) / dx;


				//far neighbour
				term = (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1) * dt / LosTopos:: sqr(dx);
				float far_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] + 1);
				if (far_phi < 0) {
					matrix.add_to_element(Dof_idx,
						Dof_idx, term);
					matrix.add_to_element(Dof_idx,
						(*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2] + 1), -term);
				}
				else {
					float theta = fraction_inside(centre_phi, far_phi);
					if (theta < 0.01f) theta = 0.01f;
					matrix.add_to_element(Dof_idx,
						Dof_idx, term / theta);
				}
				rhs[Dof_idx] -= (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1)*
					(*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2] + 1) / dx;

				//near neighbour
				term = (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]) * dt / LosTopos:: sqr(dx);
				float near_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1);
				if (near_phi < 0) {
					matrix.add_to_element(Dof_idx,
						Dof_idx, term);
					matrix.add_to_element(Dof_idx,
						(*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2] - 1), -term);
				}
				else {
					float theta = fraction_inside(centre_phi, near_phi);
					if (theta < 0.01f) theta = 0.01f;
					matrix.add_to_element(Dof_idx,
						Dof_idx, term / theta);
				}
				rhs[Dof_idx] += (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2])*
					(*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) / dx;

			}

		}
	});

    
	std::cout << "assign matrix done" << std::endl;
	LosTopos::Vec3i nijk = LosTopos::Vec3i(((*eulerian_fluids).bmax - (*eulerian_fluids).bmin) / dx);
	float tolerance;
	int iterations;
	bool success = AMGPCGSolveSparseParallelBuild(matrix, rhs, Dofs,
		1e-12f, 100, tolerance, iterations, Bulk_ijk);
	printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
	if (!success) {
		printf("WARNING: Pressure solve failed!************************************************\n");
	}

	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
			if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2])<0)
			{
				(*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2])
					= Dofs[(*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2])];
			}
			else
			{
				(*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) = 0;
			}

		}
	});


	//u = u- grad p;
	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			(*eulerian_fluids).fluid_bulk[index].u_valid.data[i] = 0;
			(*eulerian_fluids).fluid_bulk[index].v_valid.data[i] = 0;
			(*eulerian_fluids).fluid_bulk[index].w_valid.data[i] = 0;

		}
	});


	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
			if ((*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2])>0
				&& ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2])<0
					|| (*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2])<0))
			{
				float theta = 1;
				if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) >= 0
					|| (*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) >= 0)
				{
					theta = fraction_inside((*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]),
						(*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]));
				}
				if (theta < 0.01) theta = 0.01;
				(*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) -=
					dt*(((*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) -
						(*eulerian_fluids).pressure(index, ijk[0] - 1, ijk[1], ijk[2])) / dx / theta);
				(*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
				LosTopos::Vec3f sample_pos_u
					= (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.0, 0.5, 0.5);
				if ((*eulerian_fluids).get_liquid_phi(sample_pos_u) > 0)//particularly important for numerical stability
				{
					(*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = 0;
					(*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
				}

			}
			else
			{
				(*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = 0;
				(*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
			}

			if ((*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2])>0
				&& ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2])<0
					|| (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2])<0))
			{
				float theta = 1;
				if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) >= 0
					|| (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) >= 0)
				{
					theta = fraction_inside((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]),
						(*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]));
				}
				if (theta < 0.01) theta = 0.01;
				(*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) -=
					dt*(((*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) -
						(*eulerian_fluids).pressure(index, ijk[0], ijk[1] - 1, ijk[2])) / dx / theta);
				(*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
				LosTopos::Vec3f sample_pos_v
					= (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.5, 0.0, 0.5);
				if ((*eulerian_fluids).get_liquid_phi(sample_pos_v) > 0)//particularly important for numerical stability
				{
					(*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = 0;
					(*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
				}
			}
			else
			{
				(*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = 0;
				(*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
			}

			if ((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2])>0
				&& ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2])<0
					|| (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1)<0))
			{
				float theta = 1;
				if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) >= 0
					|| (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) >= 0)
				{
					theta = fraction_inside((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1),
						(*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]));
				}
				if (theta < 0.01) theta = 0.01;
				(*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) -=
					dt*(((*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) -
						(*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2] - 1)) / dx / theta);
				(*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
				LosTopos::Vec3f sample_pos_w
					= (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.5, 0.5, 0.0);
				if ((*eulerian_fluids).get_liquid_phi(sample_pos_w) > 0)//particularly important for numerical stability
				{
					(*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = 0;
					(*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
				}
			}
			else
			{
				(*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = 0;
				(*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
			}

		}
	});

	tbb::parallel_for((size_t)0,
		(size_t)(*eulerian_fluids).n_bulks,
		(size_t)1,
		[&](size_t index)
	{
		for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
		{
			if ((*eulerian_fluids).fluid_bulk[index].u_valid.data[i] == 0)
			{
				(*eulerian_fluids).fluid_bulk[index].u.data[i] = 0;
			}
			if ((*eulerian_fluids).fluid_bulk[index].v_valid.data[i] == 0)
			{
				(*eulerian_fluids).fluid_bulk[index].v.data[i] = 0;
			}
			if ((*eulerian_fluids).fluid_bulk[index].v_valid.data[i] == 0)
			{
				(*eulerian_fluids).fluid_bulk[index].v.data[i] = 0;
			}

		}
	});
}

void FluidSim::bem_boundaryvel()
{
    int n = (*eulerian_fluids).n_bulks*(*eulerian_fluids).n_perbulk;
    solid_upos.resize(n);
    solid_vpos.resize(n);
    solid_wpos.resize(n);
    solid_uweight.resize(n);
    solid_vweight.resize(n);
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i = 0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                              uint idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);

                              LosTopos::Vec3f sample_pos_u_0
                                      = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.0, 0.5, 0.5);
                              solid_upos[idx] = sample_pos_u_0;
                              solid_uweight[idx] = (1.0f - (*eulerian_fluids).u_weight(index, ijk[0], ijk[1],ijk[2]));

                              LosTopos::Vec3f sample_pos_v_0
                                      = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.5, 0.0, 0.5);
                              solid_vpos[idx] = sample_pos_v_0;
                              solid_vweight[idx] = (1.0f - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1],ijk[2]));

                              LosTopos::Vec3f sample_pos_w_0
                                      = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.5, 0.5, 0.0);
                              solid_wpos[idx] = sample_pos_w_0;
                          }
                      });
    std::vector<bool> out_is_in_liquid;
    bps()->extract_velocities_at_positions(solid_upos, solid_u, out_is_in_liquid);
    bps()->extract_velocities_at_positions(solid_vpos, solid_v, out_is_in_liquid);
    bps()->extract_velocities_at_positions(solid_wpos, solid_w, out_is_in_liquid);
    std::cout << "extract velocity from BEM finished!" << std::endl;
}

void FluidSim::const_velocity_volume(const std::vector<LosTopos::Vec3f>& in_pos, std::vector<LosTopos::Vec3f>& out_vel, LosTopos::Vec3f vel = LosTopos::Vec3f{ 1,0,0 })
{
    out_vel.resize(in_pos.size(), vel);
}

void FluidSim::handle_boundary_layer()
{
    set_sea_level_from_BEM();
    printf("get sea_level from bem complete\n");
    //This function is expected to execute after the advection procedure
    //where FLIP particles move to new positions, potentially leaving gaps at boundaries.
    
    bool additionally_initialize_bulks = true;
    double bulk_size = eulerian_fluids->bulk_size;

    auto any_corner_in_boundary_layer = [&](const LosTopos::Vec3i& bulk_ijk) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    LosTopos::Vec3d corner_position = LosTopos::Vec3d(m_flip_options.m_exterior_boundary_min) +
                        bulk_size * LosTopos::Vec3d(bulk_ijk + LosTopos::Vec3i{ i,j,k });
                    if (in_boundary_volume(corner_position)) {
                        return true;
                    }
                }
            }
        }

        //none of the corner is in the boundary volume;
        return false;
    };

    if (additionally_initialize_bulks) {
        //determine if a bulk is worth reserving
        //it happens if any of the bulk corner is in the boundary zone
        eulerian_fluids->initialize_bulks(
            particles, dx, 
            /*use hard min max*/true, m_flip_options.m_exterior_boundary_min, m_flip_options.m_exterior_boundary_max, 
           /*reserve worthy bulks*/ true, any_corner_in_boundary_layer);
    }

    std::function<float(const LosTopos::Vec3f&)>  intrinsic_boundary = [](const LosTopos::Vec3f& x) {return float(0); };
    set_boundary(intrinsic_boundary);
    //The FLIP domain has interior bounding box that hold all the FLIP particles
    //It also hold another exterior bounding box which set the bmin and bmax of the fluid bulks
    //1. All FLIP particles are assigned to fluid bulks within the exterior bounding box
    assign_particle_to_bulks();
    //after this step, all fluids within the exterior bounding box are recorded here

    //2. FlIP particles that are outside of the exterior bounding box are removed
    //this is implicitly accomplished by the bin-sorting algorithm

    //3. For fluid bulks that have any voxels outside of the interior bounding box and any
    //   voxels below the sea-level (waterline or exterior fluid volume(BEM)), they are
    //   marked as boundary layer bulks.

	std::vector<size_t> index_of_boundary_bulks;
    std::vector<size_t> index_of_normal_bulks;
	for (size_t bulkidx = 0; bulkidx < eulerian_fluids->fluid_bulk.size(); bulkidx++) {
		auto corner_voxel_idx3 = eulerian_fluids->fluid_bulk[bulkidx].tile_corner;
		if (bulk_contain_boundary(corner_voxel_idx3 / 8)) {
			index_of_boundary_bulks.push_back(bulkidx);
        }
        else {
            index_of_normal_bulks.push_back(bulkidx);
        }
	}
    
    size_t n_boundary_bulks = index_of_boundary_bulks.size();

    //4. For boundary layer bulks, the number of particles at each voxel are calculated.
    //   Each voxel within the interior bounding box keeps 8 FLIP particles
    constexpr int N = 8;
    std::vector<chunck3D<uint32_t, N>> particle_per_voxel_at_boundary_bulks;
    particle_per_voxel_at_boundary_bulks.resize(n_boundary_bulks);
   

    //set counter to zero
    for (auto& x: particle_per_voxel_at_boundary_bulks) {
        x.data.assign(x.data.size(), 0);
    }



    //assign the value of each voxel
    const auto& bmin = eulerian_fluids->bmin;
    std::vector<std::vector<int>> proposed_pid_at_boundary_bulks;
    proposed_pid_at_boundary_bulks.resize(n_boundary_bulks);

    auto in_interior_bbox = [&](const LosTopos::Vec3f& pos) {
        if (!below_waterline_or_sealevel(pos)) {
            return false;
        }
        for (int i = 0; i < 3; i++) {
            if (pos[i] > m_flip_options.m_inner_boundary_max[i]) {
                return false;
            }
            if (pos[i] < m_flip_options.m_inner_boundary_min[i]) {
                return false;
            }
        }
        return true;
    };

    tbb::parallel_for(size_t(0), n_boundary_bulks, [&](size_t boundary_bulk_idx) {
        
        size_t original_bulk_idx = index_of_boundary_bulks[boundary_bulk_idx];
        const auto& bulk_size = eulerian_fluids->bulk_size;
            //the particle it contains
        for (const auto& pidx : particle_bulks[original_bulk_idx]) {
            //pidx is the particle index in "particles"
            LosTopos::Vec3i at_voxel = LosTopos::floor((particles[pidx].pos-bmin)/dx);
            at_voxel -= eulerian_fluids->fluid_bulk[original_bulk_idx].tile_corner;

            //only keep the particles near the boundary and below the waterline
            if (!in_interior_bbox(particles[pidx].pos)) {
                continue;
            }

            if (particle_per_voxel_at_boundary_bulks[boundary_bulk_idx](at_voxel[0], at_voxel[1], at_voxel[2]) < 16) {
                proposed_pid_at_boundary_bulks[boundary_bulk_idx].push_back(pidx);
                particle_per_voxel_at_boundary_bulks[boundary_bulk_idx](at_voxel[0], at_voxel[1], at_voxel[2])++;
            }
        }
        });
    

    std::vector<std::vector<minimum_FLIP_particle>> proposed_emitted_particles;
    proposed_emitted_particles.resize(n_boundary_bulks);
    



    //seed particles in those voxels that are in the interior of the liquid
    //but has insufficient FLIP particles
    tbb::parallel_for(size_t(0), n_boundary_bulks, [&](size_t boundary_bulk_idx) {
        size_t original_bulk_idx = index_of_boundary_bulks[boundary_bulk_idx];
        const auto& tile_corner = eulerian_fluids->fluid_bulk[original_bulk_idx].tile_corner;
        LosTopos::Vec3f tile_corner_pos = bmin + LosTopos::Vec3f(tile_corner) * dx;
        LosTopos::Vec3f corner_voxel_center_pos = tile_corner_pos + LosTopos::Vec3f(0.5, 0.5, 0.5) * dx;
        

        // emitter jitter
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distrib(-10, 10);

        //voxel position
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    LosTopos::Vec3f voxel_center = corner_voxel_center_pos + LosTopos::Vec3f(LosTopos::Vec3i(i, j, k)) * dx;
                    if (in_interior_bbox(tile_corner_pos + LosTopos::Vec3f(LosTopos::Vec3i(i, j, k)) * dx)) {
                        //try to fill this voxel
                        for (uint32_t n_filled = particle_per_voxel_at_boundary_bulks[boundary_bulk_idx](i, j, k); n_filled < 8; n_filled++) {
                            LosTopos::Vec3f new_pos = voxel_center;
                            //x
                            //if ((n_filled & uint32_t(1))!=0) {
                                new_pos[0] += (float(distrib(gen)) * 0.045) * dx;
                            //}
                            //y
                            //if ((n_filled & uint32_t(2))!=0) {
                                new_pos[1] += ( float(distrib(gen)) * 0.045) * dx;
                            //}
                            //z
                            //if ((n_filled & uint32_t(4))!=0) {
                                new_pos[2] += (float(distrib(gen)) * 0.045) * dx;
                            //}
                            if (in_interior_bbox(new_pos)) {
                                proposed_emitted_particles[boundary_bulk_idx].push_back(minimum_FLIP_particle(new_pos, LosTopos::Vec3f{ 0,0,0 }));
                            }
                        }
                    }
                }//end k
            }//end j
        }//end i 
        });

    /***************************modify emitted particle velocity begin ****************************/
    //use a reference to access the required particles
    std::vector<minimum_FLIP_particle*> ptrs_to_emitted_particles;
    size_t emitted_boundary_particle_count = 0;
    
    for (const auto& bin : proposed_emitted_particles) {
        emitted_boundary_particle_count += bin.size();
    }

    ptrs_to_emitted_particles.reserve(emitted_boundary_particle_count);

    for (auto& bin : proposed_emitted_particles) {
        for (size_t i = 0; i < bin.size(); i++) {
            ptrs_to_emitted_particles.push_back(&(bin[i]));
        }
    }

    std::vector<LosTopos::Vec3f> emitted_particle_pos;
    emitted_particle_pos.reserve(emitted_boundary_particle_count);
    
    for (auto ptr : ptrs_to_emitted_particles) {
        emitted_particle_pos.push_back(ptr->pos);
    }
    printf("%d particles emitted\n", emitted_particle_pos.size());
    std::vector<LosTopos::Vec3f> emitted_particle_vel;
    {
        std::vector<bool> dummy;
        LosTopos::Vec3f flowv;
        flowv[0] = Options::doubleValue("tank_flow_x");
        flowv[0] = Options::doubleValue("tank_flow_y");
        flowv[2] = Options::doubleValue("tank_flow_z");
        //const_velocity_volume(emitted_particle_pos, emitted_particle_vel, flowv);
        //bps()->extract_velocities_at_positions(emitted_particle_pos, emitted_particle_vel, dummy);
        auto flipmin = m_flip_options.m_inner_boundary_min;
        auto flipmax = m_flip_options.m_inner_boundary_max;
        bps()->extract_velocities_from_volume_vel_cache(
            flipmin,flipmax, dx, 4, 2,
            emitted_particle_pos, emitted_particle_vel);
    }
    printf("extract vel done\n");
    for (size_t i = 0; i < emitted_particle_pos.size(); i++) {
        ptrs_to_emitted_particles[i]->vel = emitted_particle_vel[i];
    }
    /***************************modify emitted particle velocity end   ****************************/

    //combine all sources of particles for the boundary layer bulks 
    tbb::parallel_for(size_t(0), n_boundary_bulks, [&](size_t boundary_bulk_idx) {
        
        const auto& pids = proposed_pid_at_boundary_bulks[boundary_bulk_idx];
        proposed_emitted_particles[boundary_bulk_idx].reserve(
            proposed_emitted_particles[boundary_bulk_idx].size() 
            + pids.size());
        //add the original particles in the boundary volume
        for (const auto& pid : pids) {
            proposed_emitted_particles[boundary_bulk_idx].push_back(particles[pid]);
        }
        });
    //printf("collect boundary layer bulks particle done\n");

    //the begin and end particle index of the final newly-arranged FLIP arrays
    std::vector<std::pair<uint32_t, uint32_t>> normal_bulk_new_particle_be;
    std::vector<std::pair<uint32_t, uint32_t>> boundary_bulk_new_particle_be;

    normal_bulk_new_particle_be.reserve(index_of_normal_bulks.size());
    boundary_bulk_new_particle_be.reserve(index_of_boundary_bulks.size());
    uint32_t new_particle_counter = 0;
    //set be for normal bulks
    for (const auto& original_bulk_idx : index_of_normal_bulks) {
        normal_bulk_new_particle_be.push_back(std::make_pair(new_particle_counter, new_particle_counter + particle_bulks[original_bulk_idx].size()));
        new_particle_counter += particle_bulks[original_bulk_idx].size();
    }
    //set be for boundary bulks
    for (const auto& bin : proposed_emitted_particles) {
        boundary_bulk_new_particle_be.push_back(std::make_pair(new_particle_counter, new_particle_counter + bin.size()));
        new_particle_counter += bin.size();
    }

    //write to the new FLIP particles array
    //std::vector<minimum_FLIP_particle> new_particles;
    m_new_particles.resize(new_particle_counter);
    
    //copy the old normal FLIP data
    tbb::parallel_for(size_t(0), normal_bulk_new_particle_be.size(), [&](size_t normal_bulk_idx) {
        const auto& pbe = normal_bulk_new_particle_be[normal_bulk_idx];
        for (auto i = pbe.first; i < pbe.second; i++) {
            //copy the old flip data position and velocity to the new flip
            uint32_t old_particle_idx = particle_bulks[index_of_normal_bulks[normal_bulk_idx]][i-pbe.first];
            m_new_particles[i].pos = particles[old_particle_idx].pos;
            m_new_particles[i].vel = particles[old_particle_idx].vel;
        }
        });

    //copy the boundary bulks new particles
    tbb::parallel_for(size_t(0), proposed_emitted_particles.size(), [&](size_t boundary_bulk_idx) {
        const auto& pbe = boundary_bulk_new_particle_be[boundary_bulk_idx];
        for (auto i = pbe.first; i < pbe.second; i++) {
            m_new_particles[i].pos = proposed_emitted_particles[boundary_bulk_idx][i - pbe.first].pos;
            m_new_particles[i].vel = proposed_emitted_particles[boundary_bulk_idx][i - pbe.first].vel;
        }
        });

    printf(" old particle count:%d new particle count:%d\n", particles.size(), new_particle_counter);
    particles = m_new_particles;
    //5. The solid velocities on the interior bounding box boundary in the boundary layer bulks
    //   and the new FLIP particles are assigned with new velocities determined either by constant
    //   or external program (BEM).
    //   It is likely the bulk created here is not really used for future simulation
    //   The velocity is not handled here
}

void FluidSim::seed_and_remove_boundary_layer_particles()
{
    
}

void FluidSim::assign_boundary_layer_solid_velocities()
{
    //for [u,v,w]_weight<1 meaning it has solid velocity component
    //the solid velocities at those positions will be assigned

    //for simplicity, any voxels whose any of the 8 corners is inside the solid
    //will be pulled out to form the vector of query voxel id
    //then its id will be converted to positions and the velocity will be queried

    //[bulkidx][idx of solid voxel]
    using index3_bin_t = std::vector<std::vector<LosTopos::Vec3i>>;
    index3_bin_t solid_voxel_bin;
    solid_voxel_bin.resize(eulerian_fluids->fluid_bulk.size());

	tbb::parallel_for((size_t)0, eulerian_fluids->fluid_bulk.size(), [&](size_t bulkidx) {
		for (int i = 0; i < eulerian_fluids->n_perbulk; i++) {
			const auto& ijk = eulerian_fluids->loop_order[i];
			bool is_solid = false;
            //if any of the eight corner is solid
            //it is likely that 
			for (int ii = 0; ii < 2 && (!is_solid); ii++) {
				for (int jj = 0; jj < 2 && (!is_solid); jj++) {
					for (int kk = 0; kk < 2 && (!is_solid); kk++) {
						if (eulerian_fluids->solid_phi(bulkidx, ijk[0] + ii, ijk[1] + jj, ijk[2] + kk) < 0) {
							is_solid = true;
						}
					}//end kk
				}//end jj
			}//end ii
            
            

            if (is_solid) {
                //also check if there is any liquid
                //only query solid velocity when any of the six neighbors are liquid voxels
                bool has_liquid_neighbor_hood = false;
                for (int ii = -1; ii <= 1&&(!has_liquid_neighbor_hood); ii++) {
                    for (int jj = -1; jj <= 1 && (!has_liquid_neighbor_hood); jj++) {
                        for (int kk = -1; kk <= 1 && (!has_liquid_neighbor_hood); kk++) {
                            has_liquid_neighbor_hood = eulerian_fluids->liquid_phi(bulkidx, ijk[0] + ii, ijk[1] + jj, ijk[2] + kk) < 0;
                        }
                    }
                }
                if (has_liquid_neighbor_hood) {
                    solid_voxel_bin[bulkidx].push_back(ijk + eulerian_fluids->fluid_bulk[bulkidx].tile_corner);
                }
            }
		}//end loop over all voxel in a bulk
		});//end for all bulks

    //the solid voxels in each bulk is found
    //the next step is to combine them together to find where to query the velocity

    std::vector<LosTopos::Vec3i> solid_voxel_index3;

    size_t solid_voxel_count = 0;
    for (const auto& bin : solid_voxel_bin) {
        solid_voxel_count += bin.size();
    }

    solid_voxel_index3.reserve(solid_voxel_count);

    for (const auto& bin : solid_voxel_bin) {
        solid_voxel_index3.insert(solid_voxel_index3.end(), bin.begin(), bin.end());
    }

    //transform the voxel index3 to voxel center positions
    //for simplicity the velocity is measured at the voxel center
    std::vector<LosTopos::Vec3f> solid_voxel_center_pos;
    solid_voxel_center_pos.resize(solid_voxel_index3.size(), eulerian_fluids->bmin + LosTopos::Vec3f{0.5,0.5,0.5}*dx);

    tbb::parallel_for(size_t(0), solid_voxel_count, [&](size_t idx) {
        solid_voxel_center_pos[idx] += LosTopos::Vec3f(solid_voxel_index3[idx]) * dx;
        });

    std::vector<LosTopos::Vec3f> velocity;

    //this can later be adjusted to support BEM velocity interpolation
    std::vector<bool> dummy;
    {
        LosTopos::Vec3f flowv;
        flowv[0] = Options::doubleValue("tank_flow_x");
        flowv[0] = Options::doubleValue("tank_flow_y");
        flowv[2] = Options::doubleValue("tank_flow_z");
        //const_velocity_volume(solid_voxel_center_pos, velocity, flowv);
        //bps()->extract_velocities_at_positions(solid_voxel_center_pos, velocity, dummy);
        auto flipmin = m_flip_options.m_inner_boundary_min;
        auto flipmax = m_flip_options.m_inner_boundary_max;
        bps()->extract_velocities_from_volume_vel_cache(
            flipmin,flipmax, dx, 4,2,
            solid_voxel_center_pos, 
            velocity);
        //printf("query solid voxel number:%d\n", solid_voxel_center_pos.size());
    }
    

	//assign the velocity to the voxel solid velocity
	tbb::parallel_for(size_t(0), solid_voxel_count, [&](size_t idx) {
		LosTopos::Vec3i bulk_ijk = solid_voxel_index3[idx] / 8;
		LosTopos::Vec3i voxel_local_ijk = solid_voxel_index3[idx] - bulk_ijk * 8;
		auto bulkidx = eulerian_fluids->get_bulk_index(bulk_ijk[0], bulk_ijk[1], bulk_ijk[2]);
		const auto& v = velocity[idx];
		eulerian_fluids->u_solid(bulkidx, voxel_local_ijk[0], voxel_local_ijk[1], voxel_local_ijk[2]) = v[0];
		eulerian_fluids->v_solid(bulkidx, voxel_local_ijk[0], voxel_local_ijk[1], voxel_local_ijk[2]) = v[1];
		eulerian_fluids->w_solid(bulkidx, voxel_local_ijk[0], voxel_local_ijk[1], voxel_local_ijk[2]) = v[2];
		});//end for each query voxel

    //overwrite the solid velocity for moving solid boundary
    tbb::parallel_for(size_t(0), solid_voxel_count, [&](size_t idx) {
        LosTopos::Vec3i bulk_ijk = solid_voxel_index3[idx] / 8;
        LosTopos::Vec3i voxel_local_ijk = solid_voxel_index3[idx] - bulk_ijk * 8;
        auto bulkidx = eulerian_fluids->get_bulk_index(bulk_ijk[0], bulk_ijk[1], bulk_ijk[2]);
        LosTopos::Vec3f pos = eulerian_fluids->bmin + dx * LosTopos::Vec3f(eulerian_fluids->fluid_bulk[bulkidx].tile_corner + voxel_local_ijk);
        for (int j = 0; j < mesh_vec.size(); j++)
        {
            openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*mesh_vec[j].sdf);
            float sdf_value = box_sampler.wsSample(openvdb::Vec3R(pos[0], pos[1], pos[2]));
            LosTopos::Vec3f solid_vel = mesh_vec[j].vel_func(total_frame);
            if (sdf_value < 0) {
                eulerian_fluids->u_solid(bulkidx, voxel_local_ijk[0], voxel_local_ijk[1], voxel_local_ijk[2]) = solid_vel[0];
                eulerian_fluids->v_solid(bulkidx, voxel_local_ijk[0], voxel_local_ijk[1], voxel_local_ijk[2]) = solid_vel[1];
                eulerian_fluids->w_solid(bulkidx, voxel_local_ijk[0], voxel_local_ijk[1], voxel_local_ijk[2]) = solid_vel[2];
            }
        }
    });//end for each query voxel



    printf("assign solid vel done\n");
    ////brute force constvel
    //tbb::parallel_for(size_t(0), eulerian_fluids->fluid_bulk.size(), [&](size_t bidx) {
    //    for (int i = 0; i < 512; i++) {
    //        eulerian_fluids->fluid_bulk[bidx].u_solid.data[i] = Options::doubleValue("tank_flow_x");
    //        eulerian_fluids->fluid_bulk[bidx].v_solid.data[i] = Options::doubleValue("tank_flow_y");
    //        eulerian_fluids->fluid_bulk[bidx].w_solid.data[i] = Options::doubleValue("tank_flow_z");
    //        }
    //    });
}

bool FluidSim::in_boundary_volume(const LosTopos::Vec3d& pos)
{
    //inside the exterior volume
    bool inside_exterior_bbox = true;
    for (int i = 0; i < 3; i++) {
        if (pos[i] > m_flip_options.m_exterior_boundary_max[i]) {
            inside_exterior_bbox = false;
            break;
        }
        if (pos[i] < m_flip_options.m_exterior_boundary_min[i]) {
            inside_exterior_bbox = false;
            break;
        }
    }

    if (!inside_exterior_bbox) {
        return false;
    }

    //test outside the interior bounding box
    bool outside_interior_bbox = false;
    for (int i = 0; i < 3; i++) {
        if (pos[i] > m_flip_options.m_inner_boundary_max[i]) {
            outside_interior_bbox = true;
            break;
        }
        if (pos[i] < m_flip_options.m_inner_boundary_min[i]) {
            outside_interior_bbox = true;
            break;
        }
    }
    
    if (!outside_interior_bbox) {
        return false;
    }

    //test the waterline
    if (m_flip_options.m_use_waterline_for_boundary_layer) {
        if (pos[1] > m_flip_options.m_waterline) {
            return false;
        }
    }
    else {
        float sea_level = get_sea_level(LosTopos::Vec3f(pos));
        if (pos[1] > sea_level) {
            return false;
        }
    }

    //in the boundary layer and below the waterline
    return true;
}

float FluidSim::get_sea_level(const LosTopos::Vec3f& pos)
{
    if (sea_level_at_voxel_xz.empty()) {
        return 0.f;
    }

    //interpolate from the four sea level samples

    float grid_dx = m_flip_options.m_sea_level_gridsize;
    LosTopos::Vec3f indexf3 = (pos - m_flip_options.m_exterior_boundary_min) / grid_dx;
    LosTopos::Vec3i index3 = LosTopos::floor(indexf3);
    indexf3 -= LosTopos::Vec3f(index3);

    int i = LosTopos::clamp(index3[0], 0, (int)sea_level_at_voxel_xz.size()-1);
    int j = LosTopos::clamp(index3[2], 0, (int)sea_level_at_voxel_xz[i].size()-1);
    int i1 = i + 1;
    int j1 = j + 1;
    if (i1 >= sea_level_at_voxel_xz.size()) {
        i1--;
    }
    if (j1 >= (int)sea_level_at_voxel_xz[i].size()) {
        j1--;
    }
    float f11 = sea_level_at_voxel_xz[i][j];
    float f12 = sea_level_at_voxel_xz[i][j1];
    float f21 = sea_level_at_voxel_xz[i1][j];
    float f22 = sea_level_at_voxel_xz[i1][j1];

    float a21 = f21 - f11;
    float a12 = f12 - f11;
    float a22 = f22 + f11 - (f21 + f12);

    float sea_level = f11 + indexf3[0] * a21 + indexf3[2] * (a12 + a22 * indexf3[0]);
    return sea_level;
}

bool FluidSim::below_waterline_or_sealevel(const LosTopos::Vec3f& pos)
{
    float sea_level;
    if (m_flip_options.m_use_waterline_for_boundary_layer) {
        sea_level = m_flip_options.m_waterline;
    }
    else {
        sea_level = get_sea_level(pos);
    }
    
    return pos[1] < sea_level;
}

void FluidSim::set_sea_level_from_BEM()
{
    m_flip_options.m_sea_level_gridsize = dx*4;

    float grid_size = m_flip_options.m_sea_level_gridsize;
    float ext_x = m_flip_options.m_exterior_boundary_max[0] - m_flip_options.m_exterior_boundary_min[0];
    float ext_z = m_flip_options.m_exterior_boundary_max[2] - m_flip_options.m_exterior_boundary_min[2];
    float inner_minx = m_flip_options.m_inner_boundary_min[0];
    float inner_maxx = m_flip_options.m_inner_boundary_max[0];
    float inner_minz = m_flip_options.m_inner_boundary_min[2];
    float inner_maxz = m_flip_options.m_inner_boundary_max[2];
    int sea_nx = int(std::ceil(ext_x / grid_size));
    int sea_nz = int(std::ceil(ext_z / grid_size));

    sea_level_at_voxel_xz.resize(sea_nx);
    for (auto& z : sea_level_at_voxel_xz) {
        z.resize(sea_nz);
    }
    std::vector<LosTopos::Vec3f> bottom_points;
    bottom_points.reserve(sea_nx * sea_nz);
    LosTopos::Vec3f corner = m_flip_options.m_exterior_boundary_min;
    //corner[0] += 0.5f * grid_size;
    //corner[2] += 0.5f * grid_size;
    for (int i = 0; i < sea_nx; i++) {
        for (int j = 0; j < sea_nz; j++) {
            bottom_points.push_back(LosTopos::Vec3f{i* grid_size,0.f,j* grid_size }+corner);
        }
    }
    std::vector<float> sea_level;
    m_bps->extract_sea_level(sea_level, bottom_points);

    FILE* outfile;
    outfile = fopen("sea_level.txt", "w");
    for (int i = 0; i < sea_nx; i++) {
        for (int j = 0; j < sea_nz; j++) {
            sea_level_at_voxel_xz[i][j] = sea_level[i*sea_nz+j]-0.5*dx;
            //fprintf(outfile, "%e ", sea_level_at_voxel_xz[i][j]);
        }
        //fprintf(outfile, "\n");
    }

    auto avr = [&](float x, float xp, float xm, float xu, float xd) {
        return (4 * x + xp + xm + xu + xd) * 0.125f;
    };

    auto temp_sea_grid = sea_level_at_voxel_xz;
    //run a smoothing convolution
    for (int smooth_count = 0; smooth_count < 0; smooth_count++) {
        for (int i = 1; i < sea_nx - 1; i++) {
            for (int j = 1; j < sea_nz - 1; j++) {
                auto& x = temp_sea_grid;
                auto& x_new = sea_level_at_voxel_xz;
                x_new[i][j] = avr(x[i][j], x[i + 1][j], x[i - 1][j], x[i][j + 1], x[i][j - 1]);
            }
        }
        std::swap(temp_sea_grid, sea_level_at_voxel_xz);
    }
    

    for (int i = 0; i < sea_nx; i++) {
        for (int j = 0; j < sea_nz; j++) {
            fprintf(outfile, "%e ", sea_level_at_voxel_xz[i][j]);
        }
        fprintf(outfile, "\n");
    }

    fclose(outfile);
}

bool FluidSim::bulk_contain_boundary(const LosTopos::Vec3i& bulk_ijk)
{
	float bulk_size = eulerian_fluids->bulk_size;
	//if any of the eight corner voxel is outside of the inner bounding box

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				LosTopos::Vec3f corner_position = LosTopos::Vec3f(m_flip_options.m_exterior_boundary_min) +
					bulk_size * LosTopos::Vec3f(bulk_ijk + LosTopos::Vec3i{ i,j,k });

				//test outside the interior bounding box
				for (int i = 0; i < 3; i++) {
					if (corner_position[i] > m_flip_options.m_inner_boundary_max[i]) {
						return true;
					}
					if (corner_position[i] < m_flip_options.m_inner_boundary_min[i]) {
						return true;
					}
				}//end for three component of the corner

			}
		}
	}

	//none of the corner is outside of the inner bounding box
    //so it does not contain boundary
	return false;
}

void FluidSim::resampleVelocity(std::vector<minimum_FLIP_particle>& _particles, float _dx, std::vector<minimum_FLIP_particle>& _resample_pos)
{
    resample_field->initialize_bulks(_particles, _dx);
    particle_to_grid(*resample_field, _particles, _dx);
    extrapolate(*resample_field, 4);
    tbb::parallel_for((size_t)0,
        (size_t)_resample_pos.size(),
        (size_t)1,
        [&](size_t index)
        {
            _resample_pos[index].vel = resample_field->get_velocity(_resample_pos[index].pos);
        });
}

void FluidSim::solve_pressure(float dt)
{
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].global_index.data[i] = 0;
                              (*eulerian_fluids).fluid_bulk[index].pressure.data[i] = 0.0;
                          }

                      });
    Dofs.resize(0);
    std::vector<int> isolated_cell_index;
    std::vector<LosTopos::Vec3i> Dof_ijk;
    Dof_ijk.resize(0);
    for(size_t index=0; index<(*eulerian_fluids).n_bulks;index++)
    {
        for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
        {
            LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
            if ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])<0)
            {
                (*eulerian_fluids).global_index(index,ijk[0],ijk[1],ijk[2])
                        =Dofs.size();

                float t = 1e-2;
                float right = std::abs((*eulerian_fluids).u_weight(index,ijk[0]+1,ijk[1],ijk[2]));
                float left = std::abs((*eulerian_fluids).u_weight(index,ijk[0],ijk[1],ijk[2]));
                float top = std::abs((*eulerian_fluids).v_weight(index,ijk[0],ijk[1]+1,ijk[2]));
                float bot = std::abs((*eulerian_fluids).v_weight(index,ijk[0],ijk[1],ijk[2]));
                float far_ = std::abs((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2]+1));
                float near_ = std::abs((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2]));
                if (right < t && left < t && top < t && bot < t && far_ < t && near_ < t)
                {
                    isolated_cell_index.push_back(Dofs.size());
                }

                Dofs.push_back(0);
                Dof_ijk.push_back((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk);
            }
        }
    }
    std::cout << "numbers of isolated cells: " << isolated_cell_index.size() << std::endl;
    std::cout << "PPE unkowns:" << Dofs.size() << std::endl;
    matrix.resize(Dofs.size());
    rhs.resize(Dofs.size());
    matrix.zero();
    rhs.assign(rhs.size(), 0);
    Dofs.assign(Dofs.size(), 0);
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                              float centre_phi = (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]);

                              if (centre_phi<0)
                              {
                                  uint Dof_idx = (*eulerian_fluids).global_index(index,ijk[0],ijk[1],ijk[2]);

                                  //right neighbour
                                  float term = (*eulerian_fluids).u_weight(index,ijk[0]+1,ijk[1],ijk[2])*dt/LosTopos:: sqr(dx);

                                  float right_phi = (*eulerian_fluids).liquid_phi(index,ijk[0]+1,ijk[1],ijk[2]);
                                  if(right_phi < 0) {
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term);
                                      matrix.add_to_element(Dof_idx,
                                                            (*eulerian_fluids).global_index(index,ijk[0]+1,ijk[1],ijk[2]),
                                                            -term);
                                  }
                                  else {
                                      float theta = fraction_inside(centre_phi, right_phi);
                                      if(theta < 0.01f) theta = 0.01f;
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term/theta);
                                  }
                                  if (!sync_with_bem) {
                                      rhs[Dof_idx] -= ((*eulerian_fluids).u_weight(index,ijk[0]+1,ijk[1],ijk[2])*
                                                       (*eulerian_fluids).u(index,ijk[0]+1,ijk[1],ijk[2]) +
                                                       (1.0f - (*eulerian_fluids).u_weight(index, ijk[0]+1, ijk[1],ijk[2]))*(*eulerian_fluids).u_solid(index, ijk[0] + 1, ijk[1], ijk[2]))/ dx;
                                  } else {
                                      // TODO x + 1
                                      LosTopos::Vec3i u_ijk(ijk[0] + 1, ijk[1], ijk[2]);
                                      uint u_idx = (*eulerian_fluids).global_index(index, u_ijk[0], u_ijk[1], u_ijk[2]);
                                      LosTopos::Vec3f vel = solid_u[u_idx];
                                      float u_component = vel[0];
                                      rhs[Dof_idx] -= ((*eulerian_fluids).u_weight(index,ijk[0]+1,ijk[1],ijk[2])*
                                                       (*eulerian_fluids).u(index,ijk[0]+1,ijk[1],ijk[2]) +
                                                       (1.0 - (*eulerian_fluids).u_weight(index, ijk[0]+1, ijk[1],ijk[2]))*u_component)/ dx;
                                  }


                                  //left neighbour
                                  term = (*eulerian_fluids).u_weight(index,ijk[0],ijk[1],ijk[2])*dt/LosTopos:: sqr(dx);

                                  float left_phi = (*eulerian_fluids).liquid_phi(index,ijk[0]-1,ijk[1],ijk[2]);
                                  if(left_phi < 0) {
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term);
                                      matrix.add_to_element(Dof_idx,
                                                            (*eulerian_fluids).global_index(index,ijk[0]-1,ijk[1],ijk[2]),
                                                            -term);
                                  }
                                  else {
                                      float theta = fraction_inside(centre_phi, left_phi);
                                      if(theta < 0.01f) theta = 0.01f;
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term/theta);
                                  }
                                  if (!sync_with_bem) {
                                      rhs[Dof_idx] += ((*eulerian_fluids).u_weight(index,ijk[0],ijk[1],ijk[2])*
                                                       (*eulerian_fluids).u(index,ijk[0],ijk[1],ijk[2]) +
                                                       (1.0 - (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]))*(*eulerian_fluids).u_solid(index, ijk[0], ijk[1], ijk[2]))/ dx;
                                  } else {
                                      // TODO x
                                      LosTopos::Vec3i u_ijk(ijk[0], ijk[1], ijk[2]);
                                      uint u_idx = (*eulerian_fluids).global_index(index, u_ijk[0], u_ijk[1], u_ijk[2]);
                                      LosTopos::Vec3f vel = solid_u[u_idx];
                                      float u_component = vel[0];
                                      rhs[Dof_idx] += ((*eulerian_fluids).u_weight(index,ijk[0],ijk[1],ijk[2])*
                                                       (*eulerian_fluids).u(index,ijk[0],ijk[1],ijk[2]) +
                                                       (1.0 - (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]))*u_component)/ dx;
                                  }

                                  //top neighbour
                                  term = (*eulerian_fluids).v_weight(index,ijk[0],ijk[1]+1,ijk[2]) * dt / LosTopos:: sqr(dx);
                                  float top_phi = (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1]+1,ijk[2]);
                                  if(top_phi < 0) {
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term);
                                      matrix.add_to_element(Dof_idx,
                                                            (*eulerian_fluids).global_index(index,ijk[0],ijk[1]+1,ijk[2]), -term);
                                  }
                                  else {
                                      float theta = fraction_inside(centre_phi, top_phi);
                                      if(theta < 0.01f) theta = 0.01f;
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term/theta);
                                  }
                                  if (!sync_with_bem) {
                                      rhs[Dof_idx] -= ((*eulerian_fluids).v_weight(index,ijk[0],ijk[1]+1,ijk[2])*
                                                       (*eulerian_fluids).v(index,ijk[0],ijk[1]+1,ijk[2]) +
                                                       (1.0- (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]))*(*eulerian_fluids).v_solid(index, ijk[0], ijk[1] + 1, ijk[2]))/ dx;
                                  } else {
                                      // TODO: y + 1
                                      LosTopos::Vec3i v_ijk(ijk[0], ijk[1] + 1, ijk[2]);
                                      uint v_idx = (*eulerian_fluids).global_index(index, v_ijk[0], v_ijk[1], v_ijk[2]);
                                      LosTopos::Vec3f vel = solid_v[v_idx];
                                      float v_component = vel[1];
                                      rhs[Dof_idx] -= ((*eulerian_fluids).v_weight(index,ijk[0],ijk[1]+1,ijk[2])*
                                                       (*eulerian_fluids).v(index,ijk[0],ijk[1]+1,ijk[2]) +
                                                       (1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]))*v_component)/ dx;
                                  }

                                  //bottom neighbour
                                  term = (*eulerian_fluids).v_weight(index,ijk[0],ijk[1],ijk[2]) * dt / LosTopos:: sqr(dx);
                                  float bot_phi = (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1]-1,ijk[2]);
                                  if(bot_phi < 0) {
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term);
                                      matrix.add_to_element(Dof_idx,
                                                            (*eulerian_fluids).global_index(index,ijk[0],ijk[1]-1,ijk[2]), -term);
                                  }
                                  else {
                                      float theta = fraction_inside(centre_phi, bot_phi);
                                      if(theta < 0.01f) theta = 0.01f;
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term/theta);
                                  }
                                  if (!sync_with_bem) {
                                      rhs[Dof_idx] += ((*eulerian_fluids).v_weight(index,ijk[0],ijk[1],ijk[2])*
                                                       (*eulerian_fluids).v(index,ijk[0],ijk[1],ijk[2])
                                                       +(1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]))*(*eulerian_fluids).v_solid(index, ijk[0], ijk[1], ijk[2]))/ dx;
                                  } else {
                                      //TODO y
                                      LosTopos::Vec3i v_ijk(ijk[0], ijk[1], ijk[2]);
                                      uint v_idx = (*eulerian_fluids).global_index(index, v_ijk[0], v_ijk[1], v_ijk[2]);
                                      LosTopos::Vec3f vel = solid_v[v_idx];
                                      float v_component = vel[1];
                                      rhs[Dof_idx] += ((*eulerian_fluids).v_weight(index,ijk[0],ijk[1],ijk[2])*
                                                       (*eulerian_fluids).v(index,ijk[0],ijk[1],ijk[2])
                                                       +(1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]))*v_component)/ dx;
                                  }

                                  //far neighbour
                                  term = (*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2]+1) * dt / LosTopos:: sqr(dx);
                                  float far_phi = (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]+1);
                                  if(far_phi < 0) {
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term);
                                      matrix.add_to_element(Dof_idx,
                                                            (*eulerian_fluids).global_index(index,ijk[0],ijk[1],ijk[2]+1), -term);
                                  }
                                  else {
                                      float theta = fraction_inside(centre_phi, far_phi);
                                      if(theta < 0.01f) theta = 0.01f;
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term/theta);
                                  }
                                  if (!sync_with_bem) {
                                      rhs[Dof_idx] -= ((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2]+1)*
                                                       (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2]+1)
                                                       + (1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1))*(*eulerian_fluids).w_solid(index, ijk[0], ijk[1], ijk[2] + 1))/ dx;
                                  } else {
                                      // TODO w + 1
                                      LosTopos::Vec3i w_ijk(ijk[0], ijk[1], ijk[2] + 1);
                                      uint w_idx = (*eulerian_fluids).global_index(index, w_ijk[0], w_ijk[1], w_ijk[2]);
                                      LosTopos::Vec3f vel = solid_w[w_idx];
                                      float w_component = vel[2];
                                      rhs[Dof_idx] -= ((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2]+1)*
                                                       (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2]+1)
                                                       + (1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1))*w_component)/ dx;
                                  }

                                  //near neighbour
                                  term = (*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2]) * dt / LosTopos:: sqr(dx);
                                  float near_phi = (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]-1);
                                  if(near_phi < 0) {
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term);
                                      matrix.add_to_element(Dof_idx,
                                                            (*eulerian_fluids).global_index(index,ijk[0],ijk[1],ijk[2]-1), -term);
                                  }
                                  else {
                                      float theta = fraction_inside(centre_phi, near_phi);
                                      if(theta < 0.01f) theta = 0.01f;
                                      matrix.add_to_element(Dof_idx,
                                                            Dof_idx, term/theta);
                                  }
                                  if (!sync_with_bem) {
                                      rhs[Dof_idx] += ((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2])*
                                                       (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2])
                                                       +(1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]))*(*eulerian_fluids).w_solid(index, ijk[0], ijk[1], ijk[2]))/ dx;
                                  } else {
                                      //TODO: w
                                      LosTopos::Vec3i w_ijk(ijk[0], ijk[1], ijk[2]);
                                      uint w_idx = (*eulerian_fluids).global_index(index, w_ijk[0], w_ijk[1], w_ijk[2]);
                                      LosTopos::Vec3f vel = solid_w[w_idx];
                                      float w_component = vel[2];
                                      rhs[Dof_idx] += ((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2])*
                                                       (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2])
                                                       +(1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]))*w_component)/ dx;
                                  }

                              }

                          }
                      });
    // modify isolated cell RHS
    for (int i = 0; i < isolated_cell_index.size(); i++)
        rhs[isolated_cell_index[i]] = 0;

    std::cout<<"assign matrix done"<<std::endl;
    LosTopos::Vec3i nijk = LosTopos::Vec3i(((*eulerian_fluids).bmax - (*eulerian_fluids).bmin)/dx);
    float tolerance;
    int iterations;
    {
        auto tend_eigen = std::chrono::steady_clock::now();
        bool success = Libo::AMGPCGSolveSparse(matrix, rhs, Dofs, Dof_ijk,
            1e-9f, 100, tolerance, iterations, nijk[0], nijk[1], nijk[2]);
        auto tend_amg = std::chrono::steady_clock::now();
        std::chrono::duration<double> amgseconds = tend_amg - tend_eigen;
        std::cout << "amg elapsed time: " << amgseconds.count() << "s\n";

        printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
        if (!success) {
            printf("WARNING: Pressure solve failed!************************************************\n");
        }
    }
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                              if ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])<0)
                              {
                                  (*eulerian_fluids).pressure(index,ijk[0],ijk[1],ijk[2])
                                          = Dofs[(*eulerian_fluids).global_index(index,ijk[0],ijk[1],ijk[2])];
                              }
                              else
                              {
                                  (*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) = 0;
                              }

                          }
                      });


    //u = u- grad p;
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].u_valid.data[i] = 0;
                              (*eulerian_fluids).fluid_bulk[index].v_valid.data[i] = 0;
                              (*eulerian_fluids).fluid_bulk[index].w_valid.data[i] = 0;

                          }
                      });


    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                              if((*eulerian_fluids).u_weight(index,ijk[0],ijk[1],ijk[2])>0
                                 && ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])<0
                                     || (*eulerian_fluids).liquid_phi(index,ijk[0]-1,ijk[1],ijk[2])<0))
                              {
                                  float theta = 1;
                                  if ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])>=0
                                      ||(*eulerian_fluids).liquid_phi(index,ijk[0]-1,ijk[1],ijk[2])>=0)
                                  {
                                      theta = fraction_inside((*eulerian_fluids).liquid_phi(index,ijk[0]-1,ijk[1],ijk[2]),
                                                              (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]));
                                  }
                                  if (theta < 0.01) theta = 0.01;
                                  (*eulerian_fluids).u(index,ijk[0],ijk[1],ijk[2]) -=
                                          dt*(((*eulerian_fluids).pressure(index,ijk[0],ijk[1],ijk[2])-
                                               (*eulerian_fluids).pressure(index,ijk[0]-1,ijk[1],ijk[2]))/dx / theta);
                                  (*eulerian_fluids).u_valid(index,ijk[0],ijk[1],ijk[2]) = 1;
                                  LosTopos::Vec3f sample_pos_u
                                          = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.0, 0.5, 0.5);
                                  if ((*eulerian_fluids).get_liquid_phi(sample_pos_u) > 0)//particularly important for numerical stability
                                  {
//                                      (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = 0;
                                      (*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                                  }

                              }
                              else
                              {
                                  if (!sync_with_bem) {
                                      (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = (*eulerian_fluids).u_solid(index, ijk[0], ijk[1], ijk[2]);
                                  } else {
                                      // TODO
                                      uint u_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
                                      LosTopos::Vec3f vel = solid_u[u_idx];
                                      float u_component = vel[0];
                                      (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = u_component;
                                  }
                                  //this could be both air voxel or solid voxel
                                  //air voxel needs extrapolation
                                  if (eulerian_fluids->u_weight(index, ijk[0], ijk[1], ijk[2]) < 1) {
                                      (*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                                  }
                                  else {
                                      (*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                                  }
                                  
                              }

                              if((*eulerian_fluids).v_weight(index,ijk[0],ijk[1],ijk[2])>0
                                 && ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])<0
                                     || (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1]-1,ijk[2])<0))
                              {
                                  float theta = 1;
                                  if ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])>=0
                                      ||(*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1]-1,ijk[2])>=0)
                                  {
                                      theta = fraction_inside((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1]-1,ijk[2]),
                                                              (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]));
                                  }
                                  if (theta < 0.01) theta = 0.01;
                                  (*eulerian_fluids).v(index,ijk[0],ijk[1],ijk[2]) -=
                                          dt*(((*eulerian_fluids).pressure(index,ijk[0],ijk[1],ijk[2])-
                                               (*eulerian_fluids).pressure(index,ijk[0],ijk[1]-1,ijk[2]))/dx / theta);
                                  (*eulerian_fluids).v_valid(index,ijk[0],ijk[1],ijk[2]) = 1;
                                  LosTopos::Vec3f sample_pos_v
                                          = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.5, 0.0, 0.5);
                                  if ((*eulerian_fluids).get_liquid_phi(sample_pos_v) > 0)//particularly important for numerical stability
                                  {
//                                      (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = 0;
                                      (*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                                  }
                              }
                              else
                              {
                                  if (!sync_with_bem) {
                                      (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = (*eulerian_fluids).v_solid(index, ijk[0], ijk[1], ijk[2]);
                                  } else {
                                      // TODO
                                      uint v_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
                                      LosTopos::Vec3f vel = solid_v[v_idx];
                                      float v_component = vel[1];
                                      (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = v_component;
                                  }
                                  if (eulerian_fluids->v_weight(index, ijk[0], ijk[1], ijk[2]) < 1) {
                                      (*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                                  }
                                  else {
                                      (*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                                  }
                                  
                              }

                              if((*eulerian_fluids).w_weight(index,ijk[0],ijk[1],ijk[2])>0
                                 && ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])<0
                                     ||  (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]-1)<0))
                              {
                                  float theta = 1;
                                  if ((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2])>=0
                                      ||(*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]-1)>=0)
                                  {
                                      theta = fraction_inside((*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]-1),
                                                              (*eulerian_fluids).liquid_phi(index,ijk[0],ijk[1],ijk[2]));
                                  }
                                  if (theta < 0.01) theta = 0.01;
                                  (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2]) -=
                                          dt*(((*eulerian_fluids).pressure(index,ijk[0],ijk[1],ijk[2])-
                                               (*eulerian_fluids).pressure(index,ijk[0],ijk[1],ijk[2]-1))/dx/theta);
                                  (*eulerian_fluids).w_valid(index,ijk[0],ijk[1],ijk[2]) = 1;
                                  LosTopos::Vec3f sample_pos_w
                                          = (*eulerian_fluids).bmin + dx*LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx*LosTopos::Vec3f(0.5, 0.5, 0.0);
                                  if ((*eulerian_fluids).get_liquid_phi(sample_pos_w) > 0)//particularly important for numerical stability
                                  {
//                                      (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = 0;
                                      (*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                                  }
                              }
                              else
                              {
                                  if (!sync_with_bem) {
                                      (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = (*eulerian_fluids).w_solid(index, ijk[0], ijk[1], ijk[2]);
                                  } else {
                                      // TODO
                                      uint w_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
                                      LosTopos::Vec3f vel = solid_w[w_idx];
                                      float w_component = vel[2];
                                      (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = w_component;
                                  }
                                  if (eulerian_fluids->w_weight(index, ijk[0], ijk[1], ijk[2]) < 1) {
                                      (*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                                  }
                                  else {
                                      (*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                                  }
                                  
                              }

                          }
                      });

    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              if ((*eulerian_fluids).fluid_bulk[index].u_valid.data[i]==0)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].u.data[i]=0;
                              }
                              if ((*eulerian_fluids).fluid_bulk[index].v_valid.data[i]==0)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].v.data[i]=0;
                              }
                              if ((*eulerian_fluids).fluid_bulk[index].w_valid.data[i]==0)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].w.data[i]=0;
                              }

                          }
                      });

}
void FluidSim::solve_pressure_morton(float dt)
{
    tbb::parallel_for((size_t)0,
        (size_t)(*eulerian_fluids).n_bulks,
        (size_t)1,
        [&](size_t index)
        {
            for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
            {
                (*eulerian_fluids).fluid_bulk[index].global_index.data[i] = 0;
                (*eulerian_fluids).fluid_bulk[index].pressure.data[i] = 0.0;
            }

        });
    Dofs.resize(0);
    std::vector<int> isolated_cell_index;
    std::vector<LosTopos::Vec3i> Dof_ijk;
    Dof_ijk.resize(0);


    //1. discoverid-- unordered tag for each liquid voxel
    //2. mortoncode
    //3. index in a sorted morton code
    //4. ijk
    // 4->1, 4->2 easy
    // the final dof should be ordered by 3
    // so when constructing the matrix, 4->3 to assign the matrix is important
    // when discovering the liquid voxel, it is impossible to know the order in morton code
    // hence it's nice to associate each morton code with the bulk and storage index
    // after all voxels are discovered, sort the morton code, and loop over the sorted morton code
    // to set the global index of that voxel
    using morton_pair_t = std::pair<uint64_t, std::pair<uint64_t,uint32_t>>;
    tbb::concurrent_vector<morton_pair_t> normal_morton_bulk_storageidx;
    tbb::concurrent_vector<std::pair<size_t, uint32_t>> isolated_cell_bulk_storageidx;

    auto collect_mortons_for_bulk = [&](const tbb::blocked_range<size_t>& bulk_range) {
        for (size_t index = bulk_range.begin(); index < bulk_range.end(); index++) {
            for (uint32_t i = 0; i < eulerian_fluids->n_perbulk; i++) {
                LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0)
                {
                    //global voxel ijk
                    LosTopos::Vec3i gijk = ijk + (*eulerian_fluids).fluid_bulk[index].tile_corner + ijk;
                    uint64_t morton = morton_encode::encode(gijk[0], gijk[1], gijk[2]);
                    normal_morton_bulk_storageidx.push_back(std::make_pair(morton, std::make_pair(index, i)));

                    //test isolation
                    float t = 1e-2;
                    float right = std::abs((*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2]));
                    float left = std::abs((*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]));
                    float top = std::abs((*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]));
                    float bot = std::abs((*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]));
                    float far_ = std::abs((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1));
                    float near_ = std::abs((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]));

                    if (right < t && left < t && top < t && bot < t && far_ < t && near_ < t)
                    {
                        //tbb::concurrent_hash_map<uint64_t, std::pair<size_t, uint32_t>>::accessor axr;
                        //isolated_cell_morton2bulk_storageidx.insert(axr, std::make_pair(morton, std::make_pair(index, i)));
                        isolated_cell_bulk_storageidx.push_back(std::make_pair(index, i));
                    }
                    //Dofs.push_back(0);
                    //Dof_ijk.push_back((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk);
                }
            }//end for all voxels

        }//end for bulk range
    };

    tbb::parallel_for(tbb::blocked_range<size_t>(0, eulerian_fluids->fluid_bulk.size()), collect_mortons_for_bulk);

    //sort the morton encoding for all fluid voxels

    auto morton_comp = [](const morton_pair_t& a, const morton_pair_t& b) {
        return a.first < b.first;
    };
    tbb::parallel_sort(normal_morton_bulk_storageidx.begin(), normal_morton_bulk_storageidx.end(), morton_comp);
    //use the sorted morton codes to generate the dofidx and dofidx of the isolated cells

    //(morton, (bulk, storageidx))
    //assign the 
    isolated_cell_index.reserve(isolated_cell_bulk_storageidx.size());
    Dof_ijk.reserve(normal_morton_bulk_storageidx.size());
    Dofs.assign(normal_morton_bulk_storageidx.size(), 0);

    for (size_t i = 0; i < normal_morton_bulk_storageidx.size(); i++) {
        size_t at_bulk = normal_morton_bulk_storageidx[i].second.first;
        size_t at_storage_idx = normal_morton_bulk_storageidx[i].second.second;
        auto ijk = eulerian_fluids->loop_order[at_storage_idx];
        eulerian_fluids->fluid_bulk[at_bulk].global_index(ijk[0],ijk[1],ijk[2]) = i;

        //coordinates
        Dof_ijk.push_back(eulerian_fluids->fluid_bulk[at_bulk].tile_corner + eulerian_fluids->loop_order[at_storage_idx]);
    }

    //isolated cells
    for (size_t i = 0; i < isolated_cell_bulk_storageidx.size(); i++) {
        size_t at_bulk = isolated_cell_bulk_storageidx[i].first;
        size_t at_storage_idx = isolated_cell_bulk_storageidx[i].second;
        auto ijk = eulerian_fluids->loop_order[at_storage_idx];

        auto dofidx = eulerian_fluids->fluid_bulk[at_bulk].global_index(ijk[0],ijk[1],ijk[2]);
        isolated_cell_index.push_back(dofidx);
    }


    //for (size_t index = 0; index < (*eulerian_fluids).n_bulks; index++)
    //{
    //    for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
    //    {
    //        LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
    //        if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0)
    //        {
    //            /*(*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2])
    //                = Dofs.size();*/

    //            float t = 1e-2;
    //            float right = std::abs((*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2]));
    //            float left = std::abs((*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]));
    //            float top = std::abs((*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]));
    //            float bot = std::abs((*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]));
    //            float far_ = std::abs((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1));
    //            float near_ = std::abs((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]));
    //            if (right < t && left < t && top < t && bot < t && far_ < t && near_ < t)
    //            {
    //                isolated_cell_index.push_back(Dofs.size());
    //            }

    //           /* Dofs.push_back(0);
    //            Dof_ijk.push_back((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk);*/
    //        }
    //    }
    //}

    std::cout << "numbers of isolated cells: " << isolated_cell_index.size() << std::endl;
    std::cout << "PPE unkowns:" << Dofs.size() << std::endl;
    matrix.resize(Dofs.size());
    rhs.resize(Dofs.size());
    matrix.zero();
    rhs.assign(rhs.size(), 0);
    Dofs.assign(Dofs.size(), 0);

    tbb::parallel_for((size_t)0,
        (size_t)(*eulerian_fluids).n_bulks,
        (size_t)1,
        [&](size_t index)
        {
            for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
            {
                LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                float centre_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]);

                if (centre_phi < 0)
                {
                    uint Dof_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);

                    //right neighbour
                    float term = (*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) * dt / LosTopos::sqr(dx);

                    float right_phi = (*eulerian_fluids).liquid_phi(index, ijk[0] + 1, ijk[1], ijk[2]);
                    if (right_phi < 0) {
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term);
                        matrix.add_to_element(Dof_idx,
                            (*eulerian_fluids).global_index(index, ijk[0] + 1, ijk[1], ijk[2]),
                            -term);
                    }
                    else {
                        float theta = fraction_inside(centre_phi, right_phi);
                        if (theta < 0.01f) theta = 0.01f;
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term / theta);
                    }
                    if (!sync_with_bem) {
                        rhs[Dof_idx] -= ((*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) *
                            (*eulerian_fluids).u(index, ijk[0] + 1, ijk[1], ijk[2]) +
                            (1.0f - (*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2])) * (*eulerian_fluids).u_solid(index, ijk[0] + 1, ijk[1], ijk[2])) / dx;
                    }
                    else {
                        // TODO x + 1
                        LosTopos::Vec3i u_ijk(ijk[0] + 1, ijk[1], ijk[2]);
                        uint u_idx = (*eulerian_fluids).global_index(index, u_ijk[0], u_ijk[1], u_ijk[2]);
                        LosTopos::Vec3f vel = solid_u[u_idx];
                        float u_component = vel[0];
                        rhs[Dof_idx] -= ((*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) *
                            (*eulerian_fluids).u(index, ijk[0] + 1, ijk[1], ijk[2]) +
                            (1.0 - (*eulerian_fluids).u_weight(index, ijk[0] + 1, ijk[1], ijk[2])) * u_component) / dx;
                    }


                    //left neighbour
                    term = (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]) * dt / LosTopos::sqr(dx);

                    float left_phi = (*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]);
                    if (left_phi < 0) {
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term);
                        matrix.add_to_element(Dof_idx,
                            (*eulerian_fluids).global_index(index, ijk[0] - 1, ijk[1], ijk[2]),
                            -term);
                    }
                    else {
                        float theta = fraction_inside(centre_phi, left_phi);
                        if (theta < 0.01f) theta = 0.01f;
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term / theta);
                    }
                    if (!sync_with_bem) {
                        rhs[Dof_idx] += ((*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]) *
                            (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) +
                            (1.0 - (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2])) * (*eulerian_fluids).u_solid(index, ijk[0], ijk[1], ijk[2])) / dx;
                    }
                    else {
                        // TODO x
                        LosTopos::Vec3i u_ijk(ijk[0], ijk[1], ijk[2]);
                        uint u_idx = (*eulerian_fluids).global_index(index, u_ijk[0], u_ijk[1], u_ijk[2]);
                        LosTopos::Vec3f vel = solid_u[u_idx];
                        float u_component = vel[0];
                        rhs[Dof_idx] += ((*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]) *
                            (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) +
                            (1.0 - (*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2])) * u_component) / dx;
                    }

                    //top neighbour
                    term = (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) * dt / LosTopos::sqr(dx);
                    float top_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] + 1, ijk[2]);
                    if (top_phi < 0) {
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term);
                        matrix.add_to_element(Dof_idx,
                            (*eulerian_fluids).global_index(index, ijk[0], ijk[1] + 1, ijk[2]), -term);
                    }
                    else {
                        float theta = fraction_inside(centre_phi, top_phi);
                        if (theta < 0.01f) theta = 0.01f;
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term / theta);
                    }
                    if (!sync_with_bem) {
                        rhs[Dof_idx] -= ((*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) *
                            (*eulerian_fluids).v(index, ijk[0], ijk[1] + 1, ijk[2]) +
                            (1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2])) * (*eulerian_fluids).v_solid(index, ijk[0], ijk[1] + 1, ijk[2])) / dx;
                    }
                    else {
                        // TODO: y + 1
                        LosTopos::Vec3i v_ijk(ijk[0], ijk[1] + 1, ijk[2]);
                        uint v_idx = (*eulerian_fluids).global_index(index, v_ijk[0], v_ijk[1], v_ijk[2]);
                        LosTopos::Vec3f vel = solid_v[v_idx];
                        float v_component = vel[1];
                        rhs[Dof_idx] -= ((*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) *
                            (*eulerian_fluids).v(index, ijk[0], ijk[1] + 1, ijk[2]) +
                            (1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1] + 1, ijk[2])) * v_component) / dx;
                    }

                    //bottom neighbour
                    term = (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]) * dt / LosTopos::sqr(dx);
                    float bot_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]);
                    if (bot_phi < 0) {
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term);
                        matrix.add_to_element(Dof_idx,
                            (*eulerian_fluids).global_index(index, ijk[0], ijk[1] - 1, ijk[2]), -term);
                    }
                    else {
                        float theta = fraction_inside(centre_phi, bot_phi);
                        if (theta < 0.01f) theta = 0.01f;
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term / theta);
                    }
                    if (!sync_with_bem) {
                        rhs[Dof_idx] += ((*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]) *
                            (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2])
                            + (1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2])) * (*eulerian_fluids).v_solid(index, ijk[0], ijk[1], ijk[2])) / dx;
                    }
                    else {
                        //TODO y
                        LosTopos::Vec3i v_ijk(ijk[0], ijk[1], ijk[2]);
                        uint v_idx = (*eulerian_fluids).global_index(index, v_ijk[0], v_ijk[1], v_ijk[2]);
                        LosTopos::Vec3f vel = solid_v[v_idx];
                        float v_component = vel[1];
                        rhs[Dof_idx] += ((*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]) *
                            (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2])
                            + (1.0 - (*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2])) * v_component) / dx;
                    }

                    //far neighbour
                    term = (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1) * dt / LosTopos::sqr(dx);
                    float far_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] + 1);
                    if (far_phi < 0) {
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term);
                        matrix.add_to_element(Dof_idx,
                            (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2] + 1), -term);
                    }
                    else {
                        float theta = fraction_inside(centre_phi, far_phi);
                        if (theta < 0.01f) theta = 0.01f;
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term / theta);
                    }
                    if (!sync_with_bem) {
                        rhs[Dof_idx] -= ((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1) *
                            (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2] + 1)
                            + (1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1)) * (*eulerian_fluids).w_solid(index, ijk[0], ijk[1], ijk[2] + 1)) / dx;
                    }
                    else {
                        // TODO w + 1
                        LosTopos::Vec3i w_ijk(ijk[0], ijk[1], ijk[2] + 1);
                        uint w_idx = (*eulerian_fluids).global_index(index, w_ijk[0], w_ijk[1], w_ijk[2]);
                        LosTopos::Vec3f vel = solid_w[w_idx];
                        float w_component = vel[2];
                        rhs[Dof_idx] -= ((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1) *
                            (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2] + 1)
                            + (1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2] + 1)) * w_component) / dx;
                    }

                    //near neighbour
                    term = (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]) * dt / LosTopos::sqr(dx);
                    float near_phi = (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1);
                    if (near_phi < 0) {
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term);
                        matrix.add_to_element(Dof_idx,
                            (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2] - 1), -term);
                    }
                    else {
                        float theta = fraction_inside(centre_phi, near_phi);
                        if (theta < 0.01f) theta = 0.01f;
                        matrix.add_to_element(Dof_idx,
                            Dof_idx, term / theta);
                    }
                    if (!sync_with_bem) {
                        rhs[Dof_idx] += ((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]) *
                            (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2])
                            + (1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2])) * (*eulerian_fluids).w_solid(index, ijk[0], ijk[1], ijk[2])) / dx;
                    }
                    else {
                        //TODO: w
                        LosTopos::Vec3i w_ijk(ijk[0], ijk[1], ijk[2]);
                        uint w_idx = (*eulerian_fluids).global_index(index, w_ijk[0], w_ijk[1], w_ijk[2]);
                        LosTopos::Vec3f vel = solid_w[w_idx];
                        float w_component = vel[2];
                        rhs[Dof_idx] += ((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]) *
                            (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2])
                            + (1.0 - (*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2])) * w_component) / dx;
                    }

                }

            }
        });

    // modify isolated cell RHS
    for (int i = 0; i < isolated_cell_index.size(); i++)
        rhs[isolated_cell_index[i]] = 0;

    std::cout << "assign matrix done" << std::endl;
    LosTopos::Vec3i nijk = LosTopos::Vec3i(((*eulerian_fluids).bmax - (*eulerian_fluids).bmin) / dx);
    float tolerance;
    int iterations;
    {
        auto tend_eigen = std::chrono::steady_clock::now();
        bool success = Libo::AMGPCGSolveSparse(matrix, rhs, Dofs, Dof_ijk,
            1e-9f, 100, tolerance, iterations, nijk[0], nijk[1], nijk[2]);
        auto tend_amg = std::chrono::steady_clock::now();
        std::chrono::duration<double> amgseconds = tend_amg - tend_eigen;
        std::cout << "amg elapsed time: " << amgseconds.count() << "s\n";

        printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
        if (!success) {
            printf("WARNING: Pressure solve failed!************************************************\n");
        }
    }
    tbb::parallel_for((size_t)0,
        (size_t)(*eulerian_fluids).n_bulks,
        (size_t)1,
        [&](size_t index)
        {
            for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
            {
                LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0)
                {
                    (*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2])
                        = Dofs[(*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2])];
                }
                else
                {
                    (*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) = 0;
                }

            }
        });


    //u = u- grad p;
    tbb::parallel_for((size_t)0,
        (size_t)(*eulerian_fluids).n_bulks,
        (size_t)1,
        [&](size_t index)
        {
            for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
            {
                (*eulerian_fluids).fluid_bulk[index].u_valid.data[i] = 0;
                (*eulerian_fluids).fluid_bulk[index].v_valid.data[i] = 0;
                (*eulerian_fluids).fluid_bulk[index].w_valid.data[i] = 0;

            }
        });


    tbb::parallel_for((size_t)0,
        (size_t)(*eulerian_fluids).n_bulks,
        (size_t)1,
        [&](size_t index)
        {
            for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
            {
                LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];
                if ((*eulerian_fluids).u_weight(index, ijk[0], ijk[1], ijk[2]) > 0
                    && ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0
                        || (*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) < 0))
                {
                    float theta = 1;
                    if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) >= 0
                        || (*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) >= 0)
                    {
                        theta = fraction_inside((*eulerian_fluids).liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]),
                            (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]));
                    }
                    if (theta < 0.01) theta = 0.01;
                    (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) -=
                        dt * (((*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) -
                            (*eulerian_fluids).pressure(index, ijk[0] - 1, ijk[1], ijk[2])) / dx / theta);
                    (*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                    LosTopos::Vec3f sample_pos_u
                        = (*eulerian_fluids).bmin + dx * LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx * LosTopos::Vec3f(0.0, 0.5, 0.5);
                    if ((*eulerian_fluids).get_liquid_phi(sample_pos_u) > 0)//particularly important for numerical stability
                    {
                        //                                      (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = 0;
                        (*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                    }

                }
                else
                {
                    if (!sync_with_bem) {
                        (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = (*eulerian_fluids).u_solid(index, ijk[0], ijk[1], ijk[2]);
                    }
                    else {
                        // TODO
                        uint u_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
                        LosTopos::Vec3f vel = solid_u[u_idx];
                        float u_component = vel[0];
                        (*eulerian_fluids).u(index, ijk[0], ijk[1], ijk[2]) = u_component;
                    }
                    (*eulerian_fluids).u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                }

                if ((*eulerian_fluids).v_weight(index, ijk[0], ijk[1], ijk[2]) > 0
                    && ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0
                        || (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) < 0))
                {
                    float theta = 1;
                    if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) >= 0
                        || (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) >= 0)
                    {
                        theta = fraction_inside((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]),
                            (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]));
                    }
                    if (theta < 0.01) theta = 0.01;
                    (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) -=
                        dt * (((*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) -
                            (*eulerian_fluids).pressure(index, ijk[0], ijk[1] - 1, ijk[2])) / dx / theta);
                    (*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                    LosTopos::Vec3f sample_pos_v
                        = (*eulerian_fluids).bmin + dx * LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx * LosTopos::Vec3f(0.5, 0.0, 0.5);
                    if ((*eulerian_fluids).get_liquid_phi(sample_pos_v) > 0)//particularly important for numerical stability
                    {
                        //                                      (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = 0;
                        (*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                    }
                }
                else
                {
                    if (!sync_with_bem) {
                        (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = (*eulerian_fluids).v_solid(index, ijk[0], ijk[1], ijk[2]);
                    }
                    else {
                        // TODO
                        uint v_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
                        LosTopos::Vec3f vel = solid_v[v_idx];
                        float v_component = vel[1];
                        (*eulerian_fluids).v(index, ijk[0], ijk[1], ijk[2]) = v_component;
                    }
                    (*eulerian_fluids).v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                }

                if ((*eulerian_fluids).w_weight(index, ijk[0], ijk[1], ijk[2]) > 0
                    && ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0
                        || (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) < 0))
                {
                    float theta = 1;
                    if ((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]) >= 0
                        || (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) >= 0)
                    {
                        theta = fraction_inside((*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1),
                            (*eulerian_fluids).liquid_phi(index, ijk[0], ijk[1], ijk[2]));
                    }
                    if (theta < 0.01) theta = 0.01;
                    (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) -=
                        dt * (((*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2]) -
                            (*eulerian_fluids).pressure(index, ijk[0], ijk[1], ijk[2] - 1)) / dx / theta);
                    (*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                    LosTopos::Vec3f sample_pos_w
                        = (*eulerian_fluids).bmin + dx * LosTopos::Vec3f((*eulerian_fluids).fluid_bulk[index].tile_corner + ijk) + dx * LosTopos::Vec3f(0.5, 0.5, 0.0);
                    if ((*eulerian_fluids).get_liquid_phi(sample_pos_w) > 0)//particularly important for numerical stability
                    {
                        //                                      (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = 0;
                        (*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
                    }
                }
                else
                {
                    if (!sync_with_bem) {
                        (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = (*eulerian_fluids).w_solid(index, ijk[0], ijk[1], ijk[2]);
                    }
                    else {
                        // TODO
                        uint w_idx = (*eulerian_fluids).global_index(index, ijk[0], ijk[1], ijk[2]);
                        LosTopos::Vec3f vel = solid_w[w_idx];
                        float w_component = vel[2];
                        (*eulerian_fluids).w(index, ijk[0], ijk[1], ijk[2]) = w_component;
                    }
                    (*eulerian_fluids).w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                }

            }
        });

    tbb::parallel_for((size_t)0,
        (size_t)(*eulerian_fluids).n_bulks,
        (size_t)1,
        [&](size_t index)
        {
            for (int i = 0; i < (*eulerian_fluids).n_perbulk; i++)
            {
                if ((*eulerian_fluids).fluid_bulk[index].u_valid.data[i] == 0)
                {
                    (*eulerian_fluids).fluid_bulk[index].u.data[i] = 0;
                }
                if ((*eulerian_fluids).fluid_bulk[index].v_valid.data[i] == 0)
                {
                    (*eulerian_fluids).fluid_bulk[index].v.data[i] = 0;
                }
                if ((*eulerian_fluids).fluid_bulk[index].w_valid.data[i] == 0)
                {
                    (*eulerian_fluids).fluid_bulk[index].w.data[i] = 0;
                }

            }
        });

}
void FluidSim::project(float dt)
{
	//Compute finite-volume type face area weight for each velocity sample.
	//compute_weights();

	/*if (sync_with_bem)
	    bem_boundaryvel();*/

	//Set up and solve the variational pressure solve.
	solve_pressure(dt);
}

void FluidSim::extrapolate(sparse_fluid8x8x8 &_eulerian_fluid, int times) {
    tbb::parallel_for((size_t)0,
                      (size_t)_eulerian_fluid.n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                          {
                              _eulerian_fluid.fluid_bulk[index].u_delta.data[i] =
                                      _eulerian_fluid.fluid_bulk[index].u.data[i];
                              _eulerian_fluid.fluid_bulk[index].v_delta.data[i] =
                                      _eulerian_fluid.fluid_bulk[index].v.data[i];
                              _eulerian_fluid.fluid_bulk[index].w_delta.data[i] =
                                      _eulerian_fluid.fluid_bulk[index].w.data[i];
                          }

                      });
    //extrapolate u
    for (int layer=0;layer<times;layer++)
    {
        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  _eulerian_fluid.fluid_bulk[index].old_valid.data[i] =
                                          _eulerian_fluid.fluid_bulk[index].u_valid.data[i];
                              }

                          });

        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {

                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  LosTopos::Vec3i ijk = _eulerian_fluid.loop_order[i];

                                  if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2])!=1)
                                  {
                                      int count = 0;
                                      float sum = 0;

                                      if (_eulerian_fluid.old_valid(index,ijk[0]+1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.u(index,ijk[0]+1,ijk[1],ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0]-1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.u(index,ijk[0]-1,ijk[1],ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1]+1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.u(index,ijk[0],ijk[1]+1,ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1]-1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.u(index,ijk[0],ijk[1]-1,ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2]+1)==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.u(index,ijk[0],ijk[1],ijk[2]+1);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2]-1)==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.u(index,ijk[0],ijk[1],ijk[2]-1);
                                      }

                                      if(count>0)
                                      {
                                          _eulerian_fluid.u_valid(index,ijk[0],ijk[1],ijk[2])=1;
                                          _eulerian_fluid.u_delta(index,ijk[0],ijk[1],ijk[2]) =
                                                  sum/(float)count;
                                      }
                                  }
                              }
                          });
        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  _eulerian_fluid.fluid_bulk[index].u.data[i] =
                                          _eulerian_fluid.fluid_bulk[index].u_delta.data[i];
                              }

                          });

    }



    //extrapolate v
    for (int layer=0;layer<times;layer++)
    {
        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  _eulerian_fluid.fluid_bulk[index].old_valid.data[i] =
                                          _eulerian_fluid.fluid_bulk[index].v_valid.data[i];
                              }

                          });

        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {

                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  LosTopos::Vec3i ijk = _eulerian_fluid.loop_order[i];

                                  if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2])!=1)
                                  {
                                      int count = 0;
                                      float sum = 0;

                                      if (_eulerian_fluid.old_valid(index,ijk[0]+1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.v(index,ijk[0]+1,ijk[1],ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0]-1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.v(index,ijk[0]-1,ijk[1],ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1]+1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.v(index,ijk[0],ijk[1]+1,ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1]-1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.v(index,ijk[0],ijk[1]-1,ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2]+1)==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.v(index,ijk[0],ijk[1],ijk[2]+1);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2]-1)==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.v(index,ijk[0],ijk[1],ijk[2]-1);
                                      }

                                      if(count>0)
                                      {
                                          _eulerian_fluid.v_valid(index,ijk[0],ijk[1],ijk[2])=1;
                                          _eulerian_fluid.v_delta(index,ijk[0],ijk[1],ijk[2]) =
                                                  sum/(float)count;
                                      }
                                  }
                              }
                          });
        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  _eulerian_fluid.fluid_bulk[index].v.data[i] =
                                          _eulerian_fluid.fluid_bulk[index].v_delta.data[i];
                              }

                          });

    }



    //extrapolate w
    for (int layer=0;layer<times;layer++)
    {
        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  _eulerian_fluid.fluid_bulk[index].old_valid.data[i] =
                                          _eulerian_fluid.fluid_bulk[index].w_valid.data[i];
                              }

                          });

        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {

                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  LosTopos::Vec3i ijk = _eulerian_fluid.loop_order[i];

                                  if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2])!=1)
                                  {
                                      int count = 0;
                                      float sum = 0;

                                      if (_eulerian_fluid.old_valid(index,ijk[0]+1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.w(index,ijk[0]+1,ijk[1],ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0]-1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.w(index,ijk[0]-1,ijk[1],ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1]+1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.w(index,ijk[0],ijk[1]+1,ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1]-1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.w(index,ijk[0],ijk[1]-1,ijk[2]);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2]+1)==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.w(index,ijk[0],ijk[1],ijk[2]+1);
                                      }
                                      if (_eulerian_fluid.old_valid(index,ijk[0],ijk[1],ijk[2]-1)==1)
                                      {
                                          count++;
                                          sum += _eulerian_fluid.w(index,ijk[0],ijk[1],ijk[2]-1);
                                      }

                                      if(count>0)
                                      {
                                          _eulerian_fluid.w_valid(index,ijk[0],ijk[1],ijk[2])=1;
                                          _eulerian_fluid.w_delta(index,ijk[0],ijk[1],ijk[2]) =
                                                  sum/(float)count;
                                      }
                                  }
                              }
                          });
        tbb::parallel_for((size_t)0,
                          (size_t)_eulerian_fluid.n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<_eulerian_fluid.n_perbulk;i++)
                              {
                                  _eulerian_fluid.fluid_bulk[index].w.data[i] =
                                          _eulerian_fluid.fluid_bulk[index].w_delta.data[i];
                              }

                          });

    }
}

void FluidSim::extrapolate(int times)
{
    tbb::parallel_for((size_t)0,
                      (size_t)(*eulerian_fluids).n_bulks,
                      (size_t)1,
                      [&](size_t index)
                      {
                          for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                          {
                              (*eulerian_fluids).fluid_bulk[index].u_delta.data[i] =
                                      (*eulerian_fluids).fluid_bulk[index].u.data[i];
                              (*eulerian_fluids).fluid_bulk[index].v_delta.data[i] =
                                      (*eulerian_fluids).fluid_bulk[index].v.data[i];
                              (*eulerian_fluids).fluid_bulk[index].w_delta.data[i] =
                                      (*eulerian_fluids).fluid_bulk[index].w.data[i];
                          }

                      });
    //extrapolate u
    for (int layer=0;layer<times;layer++)
    {
        //tbb::atomic<size_t> new_valid_u{ 0 };
        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].old_valid.data[i] =
                                          (*eulerian_fluids).fluid_bulk[index].u_valid.data[i];
                              }

                          });

        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {

                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];

                                  if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2])!=1)
                                  {
                                      int count = 0;
                                      float sum = 0;

                                      if ((*eulerian_fluids).old_valid(index,ijk[0]+1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).u(index,ijk[0]+1,ijk[1],ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0]-1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).u(index,ijk[0]-1,ijk[1],ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1]+1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).u(index,ijk[0],ijk[1]+1,ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1]-1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).u(index,ijk[0],ijk[1]-1,ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2]+1)==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).u(index,ijk[0],ijk[1],ijk[2]+1);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2]-1)==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).u(index,ijk[0],ijk[1],ijk[2]-1);
                                      }

                                      if(count>0)
                                      {
                                          //new_valid_u++;
                                          (*eulerian_fluids).u_valid(index,ijk[0],ijk[1],ijk[2])=1;
                                          (*eulerian_fluids).u_delta(index,ijk[0],ijk[1],ijk[2]) =
                                                  sum/(float)count;
                                      }
                                  }
                              }
                          });

        //std::cout << "new_valid_u counter: " << new_valid_u << std::endl;
        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].u.data[i] =
                                          (*eulerian_fluids).fluid_bulk[index].u_delta.data[i];
                              }

                          });

    }



    //extrapolate v
    for (int layer=0;layer<times;layer++)
    {
        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].old_valid.data[i] =
                                          (*eulerian_fluids).fluid_bulk[index].v_valid.data[i];
                              }

                          });

        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {

                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];

                                  if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2])!=1)
                                  {
                                      int count = 0;
                                      float sum = 0;

                                      if ((*eulerian_fluids).old_valid(index,ijk[0]+1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).v(index,ijk[0]+1,ijk[1],ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0]-1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).v(index,ijk[0]-1,ijk[1],ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1]+1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).v(index,ijk[0],ijk[1]+1,ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1]-1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).v(index,ijk[0],ijk[1]-1,ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2]+1)==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).v(index,ijk[0],ijk[1],ijk[2]+1);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2]-1)==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).v(index,ijk[0],ijk[1],ijk[2]-1);
                                      }

                                      if(count>0)
                                      {
                                          (*eulerian_fluids).v_valid(index,ijk[0],ijk[1],ijk[2])=1;
                                          (*eulerian_fluids).v_delta(index,ijk[0],ijk[1],ijk[2]) =
                                                  sum/(float)count;
                                      }
                                  }
                              }
                          });
        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].v.data[i] =
                                          (*eulerian_fluids).fluid_bulk[index].v_delta.data[i];
                              }

                          });

    }



    //extrapolate w
    for (int layer=0;layer<times;layer++)
    {
        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].old_valid.data[i] =
                                          (*eulerian_fluids).fluid_bulk[index].w_valid.data[i];
                              }

                          });

        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {

                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  LosTopos::Vec3i ijk = (*eulerian_fluids).loop_order[i];

                                  if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2])!=1)
                                  {
                                      int count = 0;
                                      float sum = 0;

                                      if ((*eulerian_fluids).old_valid(index,ijk[0]+1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).w(index,ijk[0]+1,ijk[1],ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0]-1,ijk[1],ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).w(index,ijk[0]-1,ijk[1],ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1]+1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).w(index,ijk[0],ijk[1]+1,ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1]-1,ijk[2])==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).w(index,ijk[0],ijk[1]-1,ijk[2]);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2]+1)==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2]+1);
                                      }
                                      if ((*eulerian_fluids).old_valid(index,ijk[0],ijk[1],ijk[2]-1)==1)
                                      {
                                          count++;
                                          sum += (*eulerian_fluids).w(index,ijk[0],ijk[1],ijk[2]-1);
                                      }

                                      if(count>0)
                                      {
                                          (*eulerian_fluids).w_valid(index,ijk[0],ijk[1],ijk[2])=1;
                                          (*eulerian_fluids).w_delta(index,ijk[0],ijk[1],ijk[2]) =
                                                  sum/(float)count;
                                      }
                                  }
                              }
                          });
        tbb::parallel_for((size_t)0,
                          (size_t)(*eulerian_fluids).n_bulks,
                          (size_t)1,
                          [&](size_t index)
                          {
                              for (int i=0;i<(*eulerian_fluids).n_perbulk;i++)
                              {
                                  (*eulerian_fluids).fluid_bulk[index].w.data[i] =
                                          (*eulerian_fluids).fluid_bulk[index].w_delta.data[i];
                              }

                          });

    }



}
