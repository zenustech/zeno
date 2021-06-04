#pragma once
#ifndef SPARSE_BUFFER_H
#define SPARSE_BUFFER_H


#include <vector>
#include <unordered_map>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cfloat>
#include "util.h"
#include "tile.h"
#include "vec.h"
#include "tbb/parallel_for.h"
#include "zxxtypedefs.h"
#include "Timer.h"
#include "FLIP_particle.h"
struct iSpaceHasher
{
    unsigned int operator()(const LosTopos::Vec3i &k) const {
        return  ((k[0]+128)%128) + 128*(((k[1]+128)%128) + 128*((k[2]+128)%128));
    }
};
struct iequal_to
{
    bool operator()(const LosTopos::Vec3i&a, const LosTopos::Vec3i &b)const{return ((a[0]==b[0])&&(a[1]==b[1])&&(a[2]==b[2]));}
};
//using namespace std;

template<int N>
struct sparse_fluid_3D{
	float h, bulk_size;
	int tile_n;
	uint n_bulks,n_perbulk;
	std::vector<LosTopos::Vec3i> m_index3_of_non_empty_bulks;
	std::vector<fluid_Tile<N>> fluid_bulk;
	std::unordered_map<LosTopos::Vec3i,uint64, iSpaceHasher, iequal_to> index_mapping;//maps an bulk's physical coord to bulk array
	//indexMapper index_mapping;
	std::vector<LosTopos::Vec3i> loop_order;
	uint ni, nj, nk;
	LosTopos::Vec3f bmin, bmax;
	
	void extend_fluidbulks()
	{
		//std::vector<fluid_Tile<N>> new_fluid_bulk;
		//new_fluid_bulk.resize(0);
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks/first").start();
		for (int i=0;i<fluid_bulk.size();i++)
		{
			//new_fluid_bulk.push_back(fluid_Tile<N>(fluid_bulk[i].tile_corner,fluid_bulk[i].bulk_index));
		}
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks/first").stop();
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks/second").start();
		size_t original_bulk_size = m_index3_of_non_empty_bulks.size();
		for (int i=0;i< original_bulk_size;i++)
		{
			LosTopos::Vec3i bulk_ijk = m_index3_of_non_empty_bulks[i]/N;
			for (int kk = bulk_ijk[2]-1; kk<=bulk_ijk[2]+1;kk++)
				for (int jj = bulk_ijk[1]-1; jj<=bulk_ijk[1]+1;jj++)
					for (int ii = bulk_ijk[0]-1; ii<=bulk_ijk[0]+1;ii++)
					{
						if(ii>=0&&ii<(int)ni&&jj>=0&&jj<(int)nj&&kk>=0&&kk<(int)nk)
						{

							//uint64 idx = kk*nj*ni + jj*ni + ii;
							LosTopos::Vec3i alc_ijk(ii, jj, kk);
							if (index_mapping.find(alc_ijk) == index_mapping.end())
							{
								//index_mapping[alc_ijk] = new_fluid_bulk.size();
								//new_fluid_bulk.push_back(fluid_Tile<N>(LosTopos::Vec3i(ii*N, jj*N, kk*N), (uint)new_fluid_bulk.size()));
								index_mapping[alc_ijk] = m_index3_of_non_empty_bulks.size();
								m_index3_of_non_empty_bulks.push_back(LosTopos::Vec3i(ii * N, jj * N, kk * N));
							}
						}
					}
		}
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks/second").stop();
		//fluid_bulk.resize(0);
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks/third").start();
		//for (int i=0;i<new_fluid_bulk.size();i++)
		{
			//fluid_bulk.push_back(fluid_Tile<N>(new_fluid_bulk[i].tile_corner,new_fluid_bulk[i].bulk_index));
		}
		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks/third").stop();
	}

	void initialize_bulks(
			std::vector<minimum_FLIP_particle> &particles,
			double cell_size, 
			bool use_hard_bminmax=false,
			LosTopos::Vec3f hard_bmin = LosTopos::Vec3f{ -1,-1,-1 },
			LosTopos::Vec3f hard_bmax = LosTopos::Vec3f{ 1, 1, 1 },
			bool reserve_potential_bulks=false,
			std::function<bool(const LosTopos::Vec3i& bulk_ijk)> worth = [](const LosTopos::Vec3i& bulk_ijk) {return false; });
	//void initialize_bulks(
	//	std::vector<minimum_FLIP_particle> &particles,
	//	double cell_size, 
	//	bool use_hard_bminmax=false,
	//	LosTopos::Vec3f hard_bmin = LosTopos::Vec3f{ -1,-1,-1 },
	//	LosTopos::Vec3f hard_bmax = LosTopos::Vec3f{ 1, 1, 1 })
	//{
	//	CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks").start();
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead1").start();
	//		//fluid_bulk.resize(0);
	//		m_index3_of_non_empty_bulks.resize(0);
	//		index_mapping.clear();
	//		h=(float)cell_size;
	//		uint num_particles = (uint)particles.size();

	//		if (use_hard_bminmax) {
	//			//determine from caller
	//			bmin = hard_bmin;
	//			bmax = hard_bmax;
	//		}
	//		else {
	//			//determine from the extend of the liquid particles
	//			bmin = particles[0].pos;
	//			bmax = particles[0].pos;
	//			for (uint i = 1; i < num_particles; i++)
	//			{
	//				bmin = min_union(bmin, particles[i].pos);
	//				bmax = max_union(bmax, particles[i].pos);
	//			}
	//			bmin -= LosTopos::Vec3f(24 * h, 24 * h, 24 * h);
	//			bmax += LosTopos::Vec3f(24 * h, 24 * h, 24 * h);
	//		}

	//		bulk_size = (float)N*h;
	//		ni = (uint)ceil((bmax[0]-bmin[0])/bulk_size);
	//		nj = (uint)ceil((bmax[1]-bmin[1])/bulk_size);
	//		nk = (uint)ceil((bmax[2]-bmin[2])/bulk_size);

	//		LosTopos::Vec3i bmin_i = LosTopos::floor(bmin/h);
	//		bmin = h*LosTopos::Vec3f(bmin_i);
	//		bmax = bmin + bulk_size * LosTopos::Vec3f((float)ni, (float)nj, (float)nk);
	//		std::vector<char> buffer;
	//		buffer.resize(ni*nj*nk);
	//		buffer.assign(ni*nj*nk, 0);
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead1").stop();
	//		printf("1 nijk: %d %d %d nparticles%d\n",ni,nj,nk,num_particles);
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead2").start();
	//		tbb::parallel_for((uint)0, (uint) num_particles, (uint)1, [&](uint i)
	//			//for (uint i=0;i<num_particles;i++)
	//		{
	//			LosTopos::Vec3f pos = particles[i].pos - bmin;
	//			LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i((int)floor(pos[0]/bulk_size),
	//				(int)floor(pos[1]/bulk_size),
	//				(int)floor(pos[2]/bulk_size));
	//			size_t idx = bulk_ijk[0] + ni*bulk_ijk[1] + ni*nj*bulk_ijk[2];
	//			buffer[idx] = 1;
	//		});
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/overhead2").stop();
	//		printf("2\n");
	//		for (int buffer_id = 0;buffer_id < buffer.size();buffer_id++)
	//		{
	//			if (buffer[buffer_id] == 1)
	//			{
	//				int ii = buffer_id % ni, jj = (buffer_id % (ni*nj)) / ni, kk = buffer_id / (ni*nj);
	//				LosTopos::Vec3i bulk_ijk = LosTopos::Vec3i(ii, jj, kk);
	//				if (index_mapping.find(bulk_ijk) == index_mapping.end())
	//				{
	//					/*index_mapping[bulk_ijk] = fluid_bulk.size();
	//					fluid_bulk.push_back(fluid_Tile<N>(LosTopos::Vec3i(N*ii, N*jj, N*kk), (uint)fluid_bulk.size()));*/
	//					index_mapping[bulk_ijk] = m_index3_of_non_empty_bulks.size();
	//					m_index3_of_non_empty_bulks.push_back(LosTopos::Vec3i(N * ii, N * jj, N * kk));
	//				}
	//			}
	//		}
	//		printf("3\n");
	//		//int non_boundary_bulks;
	//		
	//		for(int i=0;i<2;i++)
	//		{
	//			//non_boundary_bulks = fluid_bulk.size();
	//			//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks").start();
	//			extend_fluidbulks();
	//			//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/extend_fluidbulks").stop();
	//		}
	//		printf("4\n");
	//		//transform the recorded index to really allocated memory
	//		int incremental_bulk_number = m_index3_of_non_empty_bulks.size() - fluid_bulk.size();
	//		int original_fluid_bulk_size = fluid_bulk.size();

	//		/*if (incremental_bulk_number > 0) {
	//			for (int i = 0; i < incremental_bulk_number > 0; i++) {
	//				const auto& new_bulk_corner_ijk = m_index3_of_non_empty_bulks[i + original_fluid_bulk_size];
	//				fluid_bulk.push_back(fluid_Tile<N>(new_bulk_corner_ijk, (uint)fluid_bulk.size()));
	//			}
	//		}
	//		else {
	//			while (fluid_bulk.size() > m_index3_of_non_empty_bulks.size()) {
	//				fluid_bulk.pop_back();
	//			}
	//		}*/
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/push_fluid").start();
	//		fluid_bulk.resize(m_index3_of_non_empty_bulks.size());
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/push_fluid").stop();
	//		for (int i = 0; i < fluid_bulk.size(); i++) {
	//			fluid_bulk[i].tile_corner = m_index3_of_non_empty_bulks[i];
	//			fluid_bulk[i].bulk_index = i;
	//		}
	//		
	//		//std::cout<<non_boundary_bulks<<std::endl;
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/identify_boundary").start();
	//		for(int i=0;i<fluid_bulk.size();i++)
	//		{
	//			LosTopos::Vec3i bulk_ijk = fluid_bulk[i].tile_corner/N;
	//			if(index_mapping.find(bulk_ijk+LosTopos::Vec3i(-1,0,0))==index_mapping.end()
	//			 ||index_mapping.find(bulk_ijk+LosTopos::Vec3i(1,0,0))==index_mapping.end()
	//			 ||index_mapping.find(bulk_ijk+LosTopos::Vec3i(0,1,0))==index_mapping.end()
	//			 ||index_mapping.find(bulk_ijk+LosTopos::Vec3i(0,-1,0))==index_mapping.end()
	//			 ||index_mapping.find(bulk_ijk+LosTopos::Vec3i(0,0,-1))==index_mapping.end()
	//			 ||index_mapping.find(bulk_ijk+LosTopos::Vec3i(0,0,1))==index_mapping.end())
	//			{
	//				fluid_bulk[i].is_boundary = true;
	//			}

	//		}
	//		//CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks/identify_boundary").stop();
	//		n_bulks = (uint)fluid_bulk.size();
	//		std::cout<<"num of bulks:"<<fluid_bulk.size()<<std::endl;
	//		std::cout<<"bmin:"<<bmin[0]<<" "<<bmin[1]<<" "<<bmin[2]<<std::endl;
	//		std::cout<<"bmax:"<<bmax[0]<<" "<<bmax[1]<<" "<<bmax[2]<<std::endl;
	//		std::cout<<"dimension:"<<ni<<" "<<nj<<" "<<nk<<std::endl;
	//	CSim::TimerMan::timer("Sim.step/FLIP/initialize_builks").stop();
	//}


	sparse_fluid_3D()
	{
		n_perbulk = N*N*N;
		fluid_bulk.resize(0);
		tile_n = N;
		chunck3D<char,N> is_there;
		loop_order.resize(0);
		for(int k=1;k<N-1;k++)for(int j=1;j<N-1;j++)for (int i=1;i<N-1;i++)
		{
			is_there(i,j,k) = 1;
			loop_order.push_back(LosTopos::Vec3i(i,j,k));
		}
		for (int k=0;k<N;k++)for(int j=0;j<N;j++)
		{
			
			if (is_there(0,j,k)==0)
			{
				is_there(0,j,k) = 1;
				loop_order.push_back(LosTopos::Vec3i(0,j,k));
			}
			
		}//i==0
		for (int k=0;k<N;k++)for(int j=0;j<N;j++)
		{
			if (is_there(N-1,j,k)==0)
			{
				is_there(N-1,j,k)=1;
				loop_order.push_back(LosTopos::Vec3i(N-1,j,k));
			}
			
		}//i==N-1
		for (int k=0;k<N;k++)for(int i=0;i<N;i++)
		{
			if(is_there(i,0,k)==0)
			{
				is_there(i,0,k)=1;
				loop_order.push_back(LosTopos::Vec3i(i,0,k));
			}
			
		}//j==0
		for(int k=0;k<N;k++)for(int i=0;i<N;i++)
		{
			if(is_there(i,N-1,k)==0)
			{
				is_there(i,N-1,k)=1;
				loop_order.push_back(LosTopos::Vec3i(i,N-1,k));
			}
		}//j==N-1
		for (int j=0;j<N;j++)for(int i=0;i<N;i++)
		{
			if(is_there(i,j,0)==0)
			{
				is_there(i,j,0)=1;
				loop_order.push_back(LosTopos::Vec3i(i,j,0));
			}
		}//k=0
		for (int j=0;j<N;j++)for(int i=0;i<N;i++)
		{
			if(is_there(i,j,N-1)==0)
			{
				is_there(i,j,N-1)=1;
				loop_order.push_back(LosTopos::Vec3i(i,j,N-1));
			}
		}//k=0

		assert(loop_order.size()==N*N*N);
	}
	~sparse_fluid_3D(){fluid_bulk.resize(0);}
	void clear()
	{
		fluid_bulk.resize(0);
	}
	int64 find_bulk(int i,int j, int k)
	{
		int I = i/N, J=j/N, K=k/N;
		int64 bulk_index;
		if (index_mapping.find(LosTopos::Vec3i(I,J,K))==index_mapping.end())
		{
			bulk_index = -1;
		}
		else
		{
			bulk_index = (int64)index_mapping[LosTopos::Vec3i(I,J,K)];
		}
		return bulk_index;
	}
	int64 find_bulk(int index, int i,int j, int k)
	{
		int I = (fluid_bulk[index].tile_corner[0] + i)/N;
		int J = (fluid_bulk[index].tile_corner[1] + j)/N;
		int K = (fluid_bulk[index].tile_corner[2] + k)/N;
		int64 bulk_index;
		if (index_mapping.find(LosTopos::Vec3i(I,J,K))==index_mapping.end())
		{
			bulk_index = -1;
		}
		else
		{
			bulk_index = (int64)index_mapping[LosTopos::Vec3i(I,J,K)];
		}
		return bulk_index;
	}
	float &
		omega_x(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_x(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_x((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_x(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_x(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_x(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_x((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_x(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_y(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_y(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_y((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_y(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_y(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_y(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_y((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_y(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_z(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_z(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_z((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_z(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_z(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_z(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_z((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_z(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_x_save(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_x_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_x_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_x_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_x_save(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_x_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_x_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_x_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_y_save(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_y_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_y_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_y_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_y_save(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_y_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_y_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_y_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_z_save(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_z_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_z_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_z_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_z_save(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_z_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_z_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_z_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_x_delta(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_x_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_x_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_x_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_x_delta(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_x_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_x_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_x_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_y_delta(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_y_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_y_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_y_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_y_delta(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_y_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_y_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_y_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		omega_z_delta(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_z_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_z_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_z_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		omega_z_delta(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).omega_z_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].omega_z_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].omega_z_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}


	float &
		psi_x(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).psi_x(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].psi_x((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].psi_x(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		psi_x(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).psi_x(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].psi_x((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].psi_x(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		psi_y(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).psi_y(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].psi_y((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].psi_y(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		psi_y(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).psi_y(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].psi_y((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].psi_y(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		psi_z(int64 bulk_index, int i,int j,int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).psi_z(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].psi_z((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].psi_z(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		psi_z(int64 bulk_index, int i,int j,int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).psi_z(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].psi_z((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].psi_z(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		u(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		u(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		u_save(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		u_save(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		u_delta(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		u_delta(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}


	float &
		u_coef(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_coef(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_coef((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_coef(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		u_coef(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_coef(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_coef((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_coef(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		u_extrapolate(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_extrapolate(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_extrapolate((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_extrapolate(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		u_extrapolate(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_extrapolate(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_extrapolate((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_extrapolate(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}


	float &
		v(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		v(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		v_save(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		v_save(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		v_delta(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		v_delta(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}


	float &
		v_coef(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_coef(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_coef((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_coef(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		v_coef(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_coef(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_coef((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_coef(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		v_extrapolate(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_extrapolate(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_extrapolate((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_extrapolate(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		v_extrapolate(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_extrapolate(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_extrapolate((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_extrapolate(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		w(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		w(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		w_save(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		w_save(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_save(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_save((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_save(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		w_delta(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		w_delta(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_delta(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_delta((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_delta(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}


	float &
		w_coef(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_coef(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_coef((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_coef(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		w_coef(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_coef(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_coef((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_coef(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		w_extrapolate(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_extrapolate(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_extrapolate((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_extrapolate(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		w_extrapolate(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_extrapolate(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_extrapolate((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_extrapolate(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}


	float &
		liquid_phi(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).liquid_phi(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].liquid_phi((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].liquid_phi(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		liquid_phi(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).liquid_phi(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].liquid_phi((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].liquid_phi(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}




	float &
		solid_phi(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).solid_phi(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].solid_phi((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].solid_phi(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	float const &
		solid_phi(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).solid_phi(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].solid_phi((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].solid_phi(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	float &
		u_weight(int64 bulk_index, int i, int j, int k)
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_weight(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].u_weight((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_weight(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float const &
		u_weight(int64 bulk_index, int i, int j, int k) const
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_weight(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].u_weight((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_weight(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float &
		v_weight(int64 bulk_index, int i, int j, int k)
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_weight(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].v_weight((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_weight(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float const &
		v_weight(int64 bulk_index, int i, int j, int k) const
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_weight(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].v_weight((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_weight(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float &
		w_weight(int64 bulk_index, int i, int j, int k)
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_weight(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].w_weight((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_weight(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float const &
		w_weight(int64 bulk_index, int i, int j, int k) const
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_weight(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].w_weight((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_weight(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}

	float &
		u_solid(int64 bulk_index, int i, int j, int k)
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_solid(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].u_solid((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_solid(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float const &
		u_solid(int64 bulk_index, int i, int j, int k) const
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_solid(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].u_solid((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_solid(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float &
		v_solid(int64 bulk_index, int i, int j, int k)
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_solid(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].v_solid((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_solid(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float const &
		v_solid(int64 bulk_index, int i, int j, int k) const
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_solid(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].v_solid((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_solid(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float &
		w_solid(int64 bulk_index, int i, int j, int k)
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_solid(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].w_solid((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_solid(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}
	float const &
		w_solid(int64 bulk_index, int i, int j, int k) const
	{
		if (i >= 0 && i<N && j >= 0 && j<N && k >= 0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_solid(i, j, k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
				fluid_bulk[bulk_index].tile_corner[1] + j,
				fluid_bulk[bulk_index].tile_corner[2] + k);
			if (bulk_index2 != -1)
			{
				return fluid_bulk[bulk_index2].w_solid((i + N) % N, (j + N) % N, (k + N) % N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_solid(std::min(std::max(0, i), N - 1),
					std::min(std::max(0, j), N - 1),
					std::min(std::max(0, k), N - 1));
			}
		}
	}

	char &
		u_valid(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	char const &
		u_valid(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).u_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].u_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].u_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	char &
		v_valid(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	char const &
		v_valid(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).v_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].v_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].v_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	char &
		w_valid(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	char const &
		w_valid(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).w_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].w_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].w_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	char &
		old_valid(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).old_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].old_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].old_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	char const &
		old_valid(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).old_valid(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].old_valid((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].old_valid(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	double &
		pressure(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).pressure(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].pressure((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].pressure(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	double const &
		pressure(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).pressure(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].pressure((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].pressure(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}

	uint &
		global_index(int64 bulk_index, int i,int j, int k)
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).global_index(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].global_index((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].global_index(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	uint const &
		global_index(int64 bulk_index, int i,int j, int k) const
	{
		if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
		{
			return (fluid_bulk[bulk_index]).global_index(i,j,k);
		}
		else
		{
			int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0]+i,
				fluid_bulk[bulk_index].tile_corner[1]+j,
				fluid_bulk[bulk_index].tile_corner[2]+k);
			if (bulk_index2!=-1)
			{
				return fluid_bulk[bulk_index2].global_index((i+N)%N,(j+N)%N,(k+N)%N);
			}
			else
			{
				return fluid_bulk[bulk_index].global_index(std::min(std::max(0,i),N-1),
					std::min(std::max(0,j),N-1),
					std::min(std::max(0,k),N-1));
			}
		}
	}
	void linear_coef(LosTopos::Vec3f &pos, int &i, int &j, int &k, float &fx, float &fy, float &fz)
	{
		i = (int)floor(pos[0]/h);
		j = (int)floor(pos[1]/h);
		k = (int)floor(pos[2]/h);
		fx= pos[0]/h - (float)i;
		fy= pos[1]/h - (float)j;
		fz= pos[2]/h - (float)k;
	}

    inline bool isValid_u(LosTopos::Vec3f &pos)
    {
        LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0,0.5*h,0.5*h) - bmin;
        int i,j,k;
        float fx,fy,fz;
        linear_coef(local_pos,i,j,k,fx,fy,fz);
        int64 bulk_idx = find_bulk(i,j,k);
        if(bulk_idx!=-1)
        {
            int local_i = i%N;
            int local_j = j%N;
            int local_k = k%N;
            char v000 = u_valid(bulk_idx,local_i,local_j,local_k);
            char v001 = u_valid(bulk_idx,local_i+1,local_j,local_k);
            char v010 = u_valid(bulk_idx,local_i,local_j+1,local_k);
            char v011 = u_valid(bulk_idx,local_i+1,local_j+1,local_k);
            char v100 = u_valid(bulk_idx,local_i,local_j,local_k+1);
            char v101 = u_valid(bulk_idx,local_i+1,local_j,local_k+1);
            char v110 = u_valid(bulk_idx,local_i,local_j+1,local_k+1);
            char v111 = u_valid(bulk_idx,local_i+1,local_j+1,local_k+1);
            return v000&&v001&&v010&&v011&&v100&&v101&&v110&&v111;
        }
        else
        {
            return false;
        }
    }
    inline bool isValid_v(LosTopos::Vec3f &pos)
    {
        LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.0,0.5*h) - bmin;
        int i,j,k;
        float fx,fy,fz;
        linear_coef(local_pos,i,j,k,fx,fy,fz);
        int64 bulk_idx = find_bulk(i,j,k);
        if(bulk_idx!=-1)
        {
            int local_i = i%N;
            int local_j = j%N;
            int local_k = k%N;
            char v000 = v_valid(bulk_idx,local_i,local_j,local_k);
            char v001 = v_valid(bulk_idx,local_i+1,local_j,local_k);
            char v010 = v_valid(bulk_idx,local_i,local_j+1,local_k);
            char v011 = v_valid(bulk_idx,local_i+1,local_j+1,local_k);
            char v100 = v_valid(bulk_idx,local_i,local_j,local_k+1);
            char v101 = v_valid(bulk_idx,local_i+1,local_j,local_k+1);
            char v110 = v_valid(bulk_idx,local_i,local_j+1,local_k+1);
            char v111 = v_valid(bulk_idx,local_i+1,local_j+1,local_k+1);
            return v000&&v001&&v010&&v011&&v100&&v101&&v110&&v111;
        }
        else
        {
            return false;
        }
    }
    inline bool isValid_w(LosTopos::Vec3f &pos)
    {
        LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.5*h,0.0) - bmin;
        int i,j,k;
        float fx,fy,fz;
        linear_coef(local_pos,i,j,k,fx,fy,fz);
        int64 bulk_idx = find_bulk(i,j,k);
        if(bulk_idx!=-1)
        {
            int local_i = i%N;
            int local_j = j%N;
            int local_k = k%N;
            char v000 = w_valid(bulk_idx,local_i,local_j,local_k);
            char v001 = w_valid(bulk_idx,local_i+1,local_j,local_k);
            char v010 = w_valid(bulk_idx,local_i,local_j+1,local_k);
            char v011 = w_valid(bulk_idx,local_i+1,local_j+1,local_k);
            char v100 = w_valid(bulk_idx,local_i,local_j,local_k+1);
            char v101 = w_valid(bulk_idx,local_i+1,local_j,local_k+1);
            char v110 = w_valid(bulk_idx,local_i,local_j+1,local_k+1);
            char v111 = w_valid(bulk_idx,local_i+1,local_j+1,local_k+1);
            return v000&&v001&&v010&&v011&&v100&&v101&&v110&&v111;
        }
        else
        {
            return false;
        }
    }
    bool isValidVel(LosTopos::Vec3f &pos)
    {
	    return isValid_u(pos)&&isValid_v(pos)&&isValid_w(pos);
    }
    bool isIsolated(LosTopos::Vec3f &pos)
    {
        LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.5*h,0.5*h) - bmin;
        int i,j,k;
        float fx,fy,fz;
        linear_coef(local_pos,i,j,k,fx,fy,fz);
        int64 bulk_idx = find_bulk(i,j,k);
        if(bulk_idx!=-1)
        {
            int local_i = i%N;
            int local_j = j%N;
            int local_k = k%N;
            for(int kk=local_k-1;kk<=local_k+1;kk++)for(int jj=local_j-1;jj<=local_j+1;jj++)for(int ii=local_i-1;ii<=local_i+1;ii++)
                    {
                if(liquid_phi(bulk_idx,ii,jj,kk)<0)
                    return false;
                    }
            return true;
        }else{
            return true;
        }
    }
	float get_u(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.f,0.5f*h,0.5f*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = u(bulk_idx,local_i,local_j,local_k);
			float v001 = u(bulk_idx,local_i+1,local_j,local_k);
			float v010 = u(bulk_idx,local_i,local_j+1,local_k);
			float v011 = u(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = u(bulk_idx,local_i,local_j,local_k+1);
			float v101 = u(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = u(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = u(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_v(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5f*h,0.f,0.5f*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = v(bulk_idx,local_i,local_j,local_k);
			float v001 = v(bulk_idx,local_i+1,local_j,local_k);
			float v010 = v(bulk_idx,local_i,local_j+1,local_k);
			float v011 = v(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = v(bulk_idx,local_i,local_j,local_k+1);
			float v101 = v(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = v(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = v(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_w(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5f*h,0.5f*h,0.f) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = w(bulk_idx,local_i,local_j,local_k);
			float v001 = w(bulk_idx,local_i+1,local_j,local_k);
			float v010 = w(bulk_idx,local_i,local_j+1,local_k);
			float v011 = w(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = w(bulk_idx,local_i,local_j,local_k+1);
			float v101 = w(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = w(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = w(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_du(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0,0.5*h,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = u_delta(bulk_idx,local_i,local_j,local_k);
			float v001 = u_delta(bulk_idx,local_i+1,local_j,local_k);
			float v010 = u_delta(bulk_idx,local_i,local_j+1,local_k);
			float v011 = u_delta(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = u_delta(bulk_idx,local_i,local_j,local_k+1);
			float v101 = u_delta(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = u_delta(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = u_delta(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_dv(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = v_delta(bulk_idx,local_i,local_j,local_k);
			float v001 = v_delta(bulk_idx,local_i+1,local_j,local_k);
			float v010 = v_delta(bulk_idx,local_i,local_j+1,local_k);
			float v011 = v_delta(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = v_delta(bulk_idx,local_i,local_j,local_k+1);
			float v101 = v_delta(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = v_delta(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = v_delta(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_dw(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.5*h,0) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = w_delta(bulk_idx,local_i,local_j,local_k);
			float v001 = w_delta(bulk_idx,local_i+1,local_j,local_k);
			float v010 = w_delta(bulk_idx,local_i,local_j+1,local_k);
			float v011 = w_delta(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = w_delta(bulk_idx,local_i,local_j,local_k+1);
			float v101 = w_delta(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = w_delta(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = w_delta(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}

	float get_omega_x(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.0*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_x(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_x(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_x(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_x(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_x(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_x(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_x(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_x(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_omega_y(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.5*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_y(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_y(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_y(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_y(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_y(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_y(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_y(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_y(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_omega_z(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.0*h,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_z(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_z(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_z(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_z(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_z(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_z(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_z(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_z(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_d_omega_x(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.0*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_x_delta(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_x_delta(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_x_delta(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_x_delta(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_x_delta(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_x_delta(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_x_delta(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_x_delta(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_d_omega_y(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.5*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_y_delta(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_y_delta(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_y_delta(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_y_delta(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_y_delta(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_y_delta(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_y_delta(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_y_delta(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_d_omega_z(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.0*h,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_z_delta(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_z_delta(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_z_delta(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_z_delta(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_z_delta(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_z_delta(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_z_delta(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_z_delta(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}

	float get_s_omega_x(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.0*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_x_save(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_x_save(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_x_save(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_x_save(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_x_save(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_x_save(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_x_save(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_x_save(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_s_omega_y(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.5*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_y_save(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_y_save(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_y_save(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_y_save(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_y_save(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_y_save(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_y_save(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_y_save(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_s_omega_z(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.0*h,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = omega_z_save(bulk_idx,local_i,local_j,local_k);
			float v001 = omega_z_save(bulk_idx,local_i+1,local_j,local_k);
			float v010 = omega_z_save(bulk_idx,local_i,local_j+1,local_k);
			float v011 = omega_z_save(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = omega_z_save(bulk_idx,local_i,local_j,local_k+1);
			float v101 = omega_z_save(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = omega_z_save(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = omega_z_save(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float M4_kernel(float rho)
	{
		if(rho>=2)
			return 0;
		if(rho>=1)
			return 0.5*(2-rho)*(2-rho)*(1-rho);
		return 1 - 2.5*rho*rho+1.5*rho*rho*rho;
	}
	float M4_weight(LosTopos::Vec3f & pos, LosTopos::Vec3f & grid)
	{
		float w = 1.0;
		for (int i=0;i<3;i++)
		{
			w *= M4_kernel(fabs((pos[i]-grid[i])/h));
		}
		return w;
	}
	float get_omega_x_M4(LosTopos::Vec3f & pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.0*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			float sum = 0;
			for (int kk=k-2;kk<=k+2;kk++)
			{
				for(int jj=j-2;jj<=j+2;jj++)
					for(int ii=i-2;ii<=i+2;ii++)
					{
						float value00 = omega_x(bulk_idx,ii,jj,kk);
						float weight = M4_weight(local_pos,LosTopos::Vec3f(ii,jj,kk)*h);
						sum += value00*weight;
					}
			}
			return sum;
		}
		else
		{
			return 0;
		}
	}
	float get_omega_y_M4(LosTopos::Vec3f & pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.5*h,0.0*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			float sum = 0;
			for (int kk=k-2;kk<=k+2;kk++)
			{
				for(int jj=j-2;jj<=j+2;jj++)
					for(int ii=i-2;ii<=i+2;ii++)
					{
						float value00 = omega_y(bulk_idx,ii,jj,kk);
						float weight = M4_weight(local_pos,LosTopos::Vec3f(ii,jj,kk)*h);
						sum += value00*weight;
					}
			}
			return sum;
		}
		else
		{
			return 0;
		}
	}
	float get_omega_z_M4(LosTopos::Vec3f & pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.0*h,0.0*h,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			float sum = 0;
			for (int kk=k-2;kk<=k+2;kk++)
			{
				for(int jj=j-2;jj<=j+2;jj++)
					for(int ii=i-2;ii<=i+2;ii++)
					{
						float value00 = omega_z(bulk_idx,ii,jj,kk);
						float weight = M4_weight(local_pos,LosTopos::Vec3f(ii,jj,kk)*h);
						sum += value00*weight;
					}
			}
			return sum;
		}
		else
		{
			return 0;
		}
	}
	LosTopos::Vec3f get_vorticity_M4(LosTopos::Vec3f &pos)
	{
		return LosTopos::Vec3f(get_omega_x_M4(pos), get_omega_y_M4(pos), get_omega_z_M4(pos));
	}
	LosTopos::Vec3f get_velocity(LosTopos::Vec3f &pos)
	{
		return LosTopos::Vec3f(get_u(pos), get_v(pos),get_w(pos));
	}
	LosTopos::Vec3f get_vorticity(LosTopos::Vec3f &pos)
	{
		return LosTopos::Vec3f(get_omega_x(pos), get_omega_y(pos),get_omega_z(pos));
	}
	LosTopos::Vec3f get_dvorticity(LosTopos::Vec3f &pos)
	{
		return LosTopos::Vec3f(get_d_omega_x(pos), get_d_omega_y(pos),get_d_omega_z(pos));
	}
	LosTopos::Vec3f get_svorticity(LosTopos::Vec3f &pos)
	{
		return LosTopos::Vec3f(get_s_omega_x(pos), get_s_omega_y(pos),get_s_omega_z(pos));
	}
	float get_liquid_phi(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5f*h,0.5f*h,0.5f*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = liquid_phi(bulk_idx,local_i,local_j,local_k);
			float v001 = liquid_phi(bulk_idx,local_i+1,local_j,local_k);
			float v010 = liquid_phi(bulk_idx,local_i,local_j+1,local_k);
			float v011 = liquid_phi(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = liquid_phi(bulk_idx,local_i,local_j,local_k+1);
			float v101 = liquid_phi(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = liquid_phi(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = liquid_phi(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	float get_solid_phi(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = solid_phi(bulk_idx,local_i,local_j,local_k);
			float v001 = solid_phi(bulk_idx,local_i+1,local_j,local_k);
			float v010 = solid_phi(bulk_idx,local_i,local_j+1,local_k);
			float v011 = solid_phi(bulk_idx,local_i+1,local_j+1,local_k);
			float v100 = solid_phi(bulk_idx,local_i,local_j,local_k+1);
			float v101 = solid_phi(bulk_idx,local_i+1,local_j,local_k+1);
			float v110 = solid_phi(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = solid_phi(bulk_idx,local_i+1,local_j+1,local_k+1);
			return LosTopos::trilerp(v000,v001,v010,v011,v100,v101,v110,v111,fx,fy,fz);
		}
		else
		{
			return 0;
		}
	}
	LosTopos::Vec3f get_delta_vel(LosTopos::Vec3f &pos)
	{
		return LosTopos::Vec3f(get_du(pos), get_dv(pos),get_dw(pos));
	}
	LosTopos::Vec3f get_grad_solid(LosTopos::Vec3f &pos)
	{
		LosTopos::Vec3f local_pos = pos-LosTopos::Vec3f(0.5*h,0.5*h,0.5*h) - bmin;
		int i,j,k;
		float fx,fy,fz;
		linear_coef(local_pos,i,j,k,fx,fy,fz);
		int64 bulk_idx = find_bulk(i,j,k);
		if(bulk_idx!=-1)
		{
			int local_i = i%N;
			int local_j = j%N;
			int local_k = k%N;
			float v000 = solid_phi(bulk_idx,local_i,local_j,local_k);
			float v100 = solid_phi(bulk_idx,local_i+1,local_j,local_k);
			float v010 = solid_phi(bulk_idx,local_i,local_j+1,local_k);
			float v110 = solid_phi(bulk_idx,local_i+1,local_j+1,local_k);
			float v001 = solid_phi(bulk_idx,local_i,local_j,local_k+1);
			float v101 = solid_phi(bulk_idx,local_i+1,local_j,local_k+1);
			float v011 = solid_phi(bulk_idx,local_i,local_j+1,local_k+1);
			float v111 = solid_phi(bulk_idx,local_i+1,local_j+1,local_k+1);

			float ddx00 = (v100 - v000);
			float ddx10 = (v110 - v010);
			float ddx01 = (v101 - v001);
			float ddx11 = (v111 - v011);
			float dv_dx = LosTopos::bilerp(ddx00,ddx10,ddx01,ddx11, fy,fz);

			float ddy00 = (v010 - v000);
			float ddy10 = (v110 - v100);
			float ddy01 = (v011 - v001);
			float ddy11 = (v111 - v101);
			float dv_dy = LosTopos::bilerp(ddy00,ddy10,ddy01,ddy11, fx,fz);

			float ddz00 = (v001 - v000);
			float ddz10 = (v101 - v100);
			float ddz01 = (v011 - v010);
			float ddz11 = (v111 - v110);
			float dv_dz = LosTopos::bilerp(ddz00,ddz10,ddz01,ddz11, fx,fy);

			return LosTopos::Vec3f(dv_dx,dv_dy,dv_dz);
		}
		else
		{
			return LosTopos::Vec3f(0,0,0);
		}
	}
	uint64 get_bulk_index(int i, int j, int k)
	{
		//int64 idx = i+j*ni+k*ni*nj;
		return index_mapping[LosTopos::Vec3i(i,j,k)];
	}
	void write_bulk_obj(std::string file_path, int frame)
	{
		std::cout<<"ouput bulks"<<std::endl;
		std::ostringstream strout;
		strout<<file_path<<"/bulk_"<<frame<<".obj";

		std::string filename = strout.str();
		std::ofstream outfile(filename.c_str());

		for (unsigned int i=0;i<n_bulks;i++)
		{
			LosTopos::Vec3f c_pos = LosTopos::Vec3f(fluid_bulk[i].tile_corner)*h+bmin;
			outfile<<"v"<<" "<<c_pos[0]<<" "<<c_pos[1]<<" "<<c_pos[2]<<std::endl;
			outfile<<"v"<<" "<<c_pos[0]+bulk_size<<" "<<c_pos[1]<<" "<<c_pos[2]<<std::endl;
			outfile<<"v"<<" "<<c_pos[0]+bulk_size<<" "<<c_pos[1]+bulk_size<<" "<<c_pos[2]<<std::endl;
			outfile<<"v"<<" "<<c_pos[0]<<" "<<c_pos[1]+bulk_size<<" "<<c_pos[2]<<std::endl;

			outfile<<"v"<<" "<<c_pos[0]<<" "<<c_pos[1]<<" "<<c_pos[2]+bulk_size<<std::endl;
			outfile<<"v"<<" "<<c_pos[0]+bulk_size<<" "<<c_pos[1]<<" "<<c_pos[2]+bulk_size<<std::endl;
			outfile<<"v"<<" "<<c_pos[0]+bulk_size<<" "<<c_pos[1]+bulk_size<<" "<<c_pos[2]+bulk_size<<std::endl;
			outfile<<"v"<<" "<<c_pos[0]<<" "<<c_pos[1]+bulk_size<<" "<<c_pos[2]+bulk_size<<std::endl;

		}
		for (unsigned int i=0;i<n_bulks;i++)
		{
			uint off_set = i*8;
			outfile<<"f"<<" "<<4+off_set<<" "<<3+off_set<<" "<<2+off_set<<" "<<1+off_set<<std::endl;
			outfile<<"f"<<" "<<8+off_set<<" "<<7+off_set<<" "<<6+off_set<<" "<<5+off_set<<std::endl;
			outfile<<"f"<<" "<<4+off_set<<" "<<3+off_set<<" "<<7+off_set<<" "<<8+off_set<<std::endl;
			outfile<<"f"<<" "<<1+off_set<<" "<<2+off_set<<" "<<6+off_set<<" "<<5+off_set<<std::endl;
		}
		
	}
};
typedef sparse_fluid_3D<8> sparse_fluid8x8x8;
typedef sparse_fluid_3D<6> sparse_fluid6x6x6;

#endif