#ifndef _geo_level_gen_h_
#define _geo_level_gen_h_


#include <iostream>
#include <vector>
#include <unordered_map>
#include "tbb/tbb.h"
#include "util.h"
#include "vec.h"
#include <cmath>
#include "sparse_matrix.h"
#include "blas_wrapper.h"

using namespace std;

template<class T>
struct levelGen{
	void generateRP(const FixedSparseMatrix<T> &A,
		FixedSparseMatrix<T> &R,
		FixedSparseMatrix<T> &P,
		int ni, int nj, int nk);



	void generateLevelsGalerkinCoarsening(vector<FixedSparseMatrix<T> *> &A_L,
		vector<FixedSparseMatrix<T> *> &R_L,
		vector<FixedSparseMatrix<T> *> &P_L,
		vector<FLUID::Vec3i>                 &S_L,
		int & total_level,
		//given
		FixedSparseMatrix<T> &A,
		int ni,int nj,int nk);

	void generateRPCompressed(const FixedSparseMatrix<T> &A,
		FixedSparseMatrix<T> &R,
		FixedSparseMatrix<T> &P,
		vector<bool>         &pattern,
		vector<char>         &mask,
		vector<int>          &index_table,
		vector<char>         &mask_this_level,
		vector<int>          &idxTable_this_level,
		int ni, int nj, int nk)
	{
		generatePattern(mask,index_table,pattern,ni,nj,nk);
		int nni = ceil((float)ni/2.0);
		int nnj = ceil((float)nj/2.0);
		int nnk = ceil((float)nk/2.0);
		SparseMatrix<T> r;
		SparseMatrix<T> p;
		p.resize(A.n);
		p.zero();
		//generate index table for this level
		mask_this_level.resize(nni*nnj*nnk);
		idxTable_this_level.resize(nni*nnj*nnk);
		mask_this_level.assign(mask_this_level.size(),0);
		idxTable_this_level.assign(idxTable_this_level.size(),0);
		int compute_num = ni*nj*nk;
		int slice = ni*nj;
		tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
		{
			int k = thread_idx/slice;
			int j = (thread_idx%slice)/ni;
			int i = thread_idx%ni;

			{
				int ii = i/2, jj = j/2, kk = k/2;
				int index = ii + nni*jj + nni*nnj*kk;
				int upper_idx = i + j*ni + ni*nj*k;
				if(mask[upper_idx] == 1) 
				{
					mask_this_level[index]=1;
				}
			}
		});
		idxTable_this_level[0] = mask_this_level[0]-1;
		for (int i=1;i<idxTable_this_level.size();i++)
		{
			idxTable_this_level[i] = idxTable_this_level[i-1]+mask_this_level[i];
		}
		int unknowns = idxTable_this_level[idxTable_this_level.size()-1]+1;
		assert(unknowns>0);
		r.resize(unknowns);
		r.zero();


		for(int k=0;k<nnk;k++)for(int j=0;j<nnj;j++)for (int i=0;i<nni;i++)
		{
			unsigned int index = (k*nnj +j)*nni + i;
			for(int kk=0;kk<=1;kk++)for(int jj=0;jj<=1;jj++)for(int ii=0;ii<=1;ii++)
			{
				int iii = i*2 + ii;
				int jjj = j*2 + jj;
				int kkk = k*2 + kk;
				if(iii<ni && jjj<nj && kkk<nk)
				{
					unsigned int index2 = (kkk*nj + jjj)*ni + iii;
					if (mask[index2]==1 && mask_this_level[index]==1)
					{
						r.set_element(idxTable_this_level[index], 
							index_table[index2], (T)0.125);
						p.set_element(index_table[index2],
							idxTable_this_level[index], 1.0);
					}

				}
			}
		}


		R.construct_from_matrix(r);
		P.construct_from_matrix(p);
		r.clear();
		p.clear();

	}

	void generatePattern(vector<char>         &mask,
		vector<int>          &index_table,
		vector<bool>         &pattern,
		int ni, int nj, int nk)
	{
		pattern.resize(index_table[index_table.size()-1]+1);
		pattern.assign(pattern.size(),false);
		int compute_num = ni*nj*nk;
		int slice = ni*nj;
		tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
		{
			int k = thread_idx/slice;
			int j = (thread_idx%slice)/ni;
			int i = thread_idx%ni;
			if((i+j+k)%2 == 0)
			{
				int index = k*nj*ni + j*ni + i;
				if(mask[index]==1)
				{
					pattern[index_table[index]] = true;
				}
			}


		});
	}

	void generateLevelsGalerkinCoarseningCompressed(vector<FixedSparseMatrix<T> *> &A_L,
		vector<FixedSparseMatrix<T> *> &R_L,
		vector<FixedSparseMatrix<T> *> &P_L,
		vector<vector<bool>*>           &b_L,
		int & total_level,
		//given
		FixedSparseMatrix<T> &A,
		vector<char>         &mask,
		vector<int>          &index_table,
		int ni,int nj,int nk){
			cout<<"building levels ...... "<<endl;
			vector<FLUID::Vec3i> S_L;
			vector<char>          mask0;
			vector<int>           idxTable0;
			A_L.resize(0);
			R_L.resize(0);
			P_L.resize(0);
			b_L.resize(0);
			total_level = 1;
			A_L.push_back(&A);
			mask0 = mask;
			idxTable0 = index_table;
			S_L.push_back(FLUID::Vec3i(ni,nj,nk));
			b_L.push_back(new vector<bool>);
			int nni = ni, nnj = nj, nnk = nk;
			unsigned int unknowns = A.n;
			while (unknowns > 4096)
			{
				A_L.push_back(new FixedSparseMatrix<T>);
				R_L.push_back(new FixedSparseMatrix<T>);
				P_L.push_back(new FixedSparseMatrix<T>);
				nni = ceil((float)nni/2.0);
				nnj = ceil((float)nnj/2.0);
				nnk = ceil((float)nnk/2.0);

				S_L.push_back(FLUID::Vec3i(nni,nnj,nnk));
				b_L.push_back(new vector<bool>);

				int i = total_level - 1;
				vector<char> mask_temp;
				vector<int> idxTable_temp;
				//printf("generating R and P\n");
				generateRPCompressed(*(A_L[i]), *(R_L[i]),*(P_L[i]), *(b_L[i]),
					mask0, idxTable0,
					mask_temp, idxTable_temp,
					S_L[i].v[0], S_L[i].v[1],S_L[i].v[2]);
				//printf("generating R and P done!\n");
				//printf("%d,%d,%d\n",A_L[i]->n, P_L[i]->n, R_L[i]->n);
				FixedSparseMatrix<T> temp;
				multiplyMat(*(A_L[i]),*(P_L[i]),temp,1.0);
				multiplyMat(*(R_L[i]),temp, *(A_L[i+1]),0.5);
				//printf("multiply matrix done\n");
				temp.resize(0);
				temp.clear();
				mask0 = mask_temp;
				idxTable0 = idxTable_temp;
				idxTable_temp.resize(0);idxTable_temp.shrink_to_fit();
				mask_temp.resize(0);    mask_temp.shrink_to_fit();


				unknowns = A_L[i+1]->n;
				total_level++;
			}


			generatePattern(mask0, idxTable0,
				*(b_L[total_level-1]), 
				S_L[total_level-1].v[0], 
				S_L[total_level-1].v[1], 
				S_L[total_level-1].v[2]);




			cout<<"build levels done"<<endl;
	}







	void generateRPSparse(const FixedSparseMatrix<T> &A,
		FixedSparseMatrix<T> &R,
		FixedSparseMatrix<T> &P,
		vector<bool>         &pattern,
		vector<FLUID::Vec3i>		 &Dof_ijk_fine,
		vector<FLUID::Vec3i>	     &Dof_ijk_coarse,
		int ni, int nj, int nk)
	{
		generatePatternSparse(Dof_ijk_fine, pattern);
		int nni = ceil((float)ni / 2.0);
		int nnj = ceil((float)nj / 2.0);
		int nnk = ceil((float)nk / 2.0);
		SparseMatrix<T> r;
		SparseMatrix<T> p;
		p.resize(A.n);
		p.zero();
		Dof_ijk_coarse.resize(0);
		std::unordered_map<unsigned int, unsigned int> index_mapping;
		//generate index table for this level
		for (unsigned int idx_f = 0;idx_f<Dof_ijk_fine.size();idx_f++)
		{
			FLUID::Vec3i ijk = Dof_ijk_fine[idx_f];
			int coarse_i = ijk[0] / 2, coarse_j = ijk[1] / 2, coarse_k = ijk[2] / 2;
			unsigned int idx = coarse_i + coarse_j*nni + coarse_k*nni*nnj;
			if (index_mapping.find(idx) == index_mapping.end())
			{
				unsigned int idx_c = Dof_ijk_coarse.size();
				index_mapping[idx] = idx_c;
				p.set_element(idx_f, idx_c, 1.0f);
				//r.set_element(idx_c,idx_f,0.25f);
				Dof_ijk_coarse.push_back(FLUID::Vec3i(coarse_i, coarse_j, coarse_k));
			}
			else
			{
				unsigned int idx_c = index_mapping[idx];
				p.set_element(idx_f, idx_c, 1.0f);
				//r.set_element(idx_c,idx_f,0.25f);
			}
		}
		r.resize(Dof_ijk_coarse.size());

		for (unsigned int idx_f = 0;idx_f<Dof_ijk_fine.size();idx_f++)
		{
			FLUID::Vec3i ijk = Dof_ijk_fine[idx_f];
			int coarse_i = ijk[0] / 2, coarse_j = ijk[1] / 2, coarse_k = ijk[2] / 2;
			unsigned int idx = coarse_i + coarse_j*nni + coarse_k*nni*nnj;
			unsigned int idx_c = index_mapping[idx];
			r.set_element(idx_c, idx_f, 0.125f);
		}



		R.construct_from_matrix(r);
		P.construct_from_matrix(p);
		r.clear();
		p.clear();

	}

	void generatePatternSparse
	(vector<FLUID::Vec3i>       &Dof_ijk,
		vector<bool>         &pattern)
	{
		pattern.resize(Dof_ijk.size());
		tbb::parallel_for((size_t)0,
			(size_t)Dof_ijk.size(),
			(size_t)1,
			[&](size_t index)
		{
			FLUID::Vec3i ijk = Dof_ijk[index];
			if ((ijk[0] + ijk[1] + ijk[2]) % 2 == 1)
			{
				pattern[index] = true;
			}
			else
			{
				pattern[index] = false;
			}
		});
	}

	void generateLevelsGalerkinCoarseningSparse
	(vector<FixedSparseMatrix<T> *> &A_L,
		vector<FixedSparseMatrix<T> *> &R_L,
		vector<FixedSparseMatrix<T> *> &P_L,
		vector<vector<bool>*>           &b_L,
		int & total_level,
		//given
		FixedSparseMatrix<T> &A,
		vector<FLUID::Vec3i> & Dof_ijk,
		int ni, int nj, int nk) {
		cout << "building levels ...... " << endl;
		vector<FLUID::Vec3i> Dof_ijk_fine;
		vector<FLUID::Vec3i> Dof_ijk_coarse;
		vector<FLUID::Vec3i> S_L;
		Dof_ijk_fine = Dof_ijk;
		A_L.resize(0);
		R_L.resize(0);
		P_L.resize(0);
		b_L.resize(0);
		total_level = 1;
		A_L.push_back(&A);
		S_L.push_back(FLUID::Vec3i(ni, nj, nk));
		b_L.push_back(new vector<bool>);
		int nni = ni, nnj = nj, nnk = nk;
		unsigned int unknowns = A.n;
		while (unknowns > 4096)
		{
			A_L.push_back(new FixedSparseMatrix<T>);
			R_L.push_back(new FixedSparseMatrix<T>);
			P_L.push_back(new FixedSparseMatrix<T>);
			nni = ceil((float)nni / 2.0);
			nnj = ceil((float)nnj / 2.0);
			nnk = ceil((float)nnk / 2.0);

			S_L.push_back(FLUID::Vec3i(nni, nnj, nnk));
			b_L.push_back(new vector<bool>);

			int i = total_level - 1;
			//printf("generating R and P\n");
			generateRPSparse(*(A_L[i]), *(R_L[i]), *(P_L[i]), *(b_L[i]),
				Dof_ijk_fine, Dof_ijk_coarse,
				S_L[i].v[0], S_L[i].v[1], S_L[i].v[2]);
			Dof_ijk_fine.resize(0); Dof_ijk_fine.shrink_to_fit();
			Dof_ijk_fine = Dof_ijk_coarse;
			//printf("generating R and P done!\n");
			//printf("%d,%d,%d\n",A_L[i]->n, P_L[i]->n, R_L[i]->n);
			FixedSparseMatrix<T> temp;
			multiplyMat(*(A_L[i]), *(P_L[i]), temp, 1.0);
			multiplyMat(*(R_L[i]), temp, *(A_L[i + 1]), 0.5);
			//printf("multiply matrix done\n");
			temp.resize(0);
			temp.clear();


			unknowns = A_L[i + 1]->n;
			total_level++;
		}


		generatePatternSparse(Dof_ijk_fine,
			*(b_L[total_level - 1]));

		Dof_ijk_fine.resize(0);
		Dof_ijk_fine.shrink_to_fit();


		cout << "build levels done" << endl;
	}




	/////////parallel build amg//////////////////////////////////////////////
	void generateRPSparseParallelBuild(const FixedSparseMatrix<T> &A,
		FixedSparseMatrix<T> &R,
		FixedSparseMatrix<T> &P,
		vector<bool>         &pattern,
		vector<FLUID::Vec3i>		 &Dof_ijk_fine,
		vector<FLUID::Vec3i>	     &Dof_ijk_coarse,
		int base,
		vector<FLUID::Vec3i> & Bulk_ijk)
	{
		generatePatternSparse(Dof_ijk_fine, pattern);
		SparseMatrix<T> r;
		SparseMatrix<T> p;
		p.resize(A.n);
		p.reserve(1);
		r.resize(Bulk_ijk.size()*base*base*base);
		r.reserve(8);
		Dof_ijk_coarse.resize(Bulk_ijk.size()*base*base*base);
		//construct R and P in parallel
		tbb::parallel_for((size_t)0,
			(size_t)Bulk_ijk.size(),
			(size_t)1,
			[&](size_t index)
		{
			for (int kk = 0;kk < base;kk++)for (int jj = 0;jj < base;jj++)for (int ii = 0;ii < base;ii++)
			{
				size_t coarse_idx = index * base*base*base + kk*base*base + jj*base + ii;
				Dof_ijk_coarse[coarse_idx] = base*Bulk_ijk[index] + FLUID::Vec3i(ii, jj, kk);
				for (int c = 0;c < 8;c++)
				{
					int iii = c % 2, jjj = (c%4)/2, kkk = c / 4;
					int base2 = base * 2;
					size_t fine_idx = index * base2 * base2 * base2 + (kk*2+kkk) * base2 * base2 + (jj*2+jjj)*base2 + (ii*2+iii);
					p.set_element(fine_idx, coarse_idx, 1.0);
					r.set_element(coarse_idx, fine_idx, 0.125);
				}
			}
		});



		R.construct_from_matrix(r);
		P.construct_from_matrix(p);
		r.clear();
		p.clear();

	}

	void generatePatternSparseParallelBuild
	(vector<FLUID::Vec3i>       &Dof_ijk,
		vector<bool>         &pattern)
	{
		pattern.resize(Dof_ijk.size());
		tbb::parallel_for((size_t)0,
			(size_t)Dof_ijk.size(),
			(size_t)1,
			[&](size_t index)
		{
			FLUID::Vec3i ijk = Dof_ijk[index];
			if ((ijk[0] + ijk[1] + ijk[2]) % 2 == 1)
			{
				pattern[index] = true;
			}
			else
			{
				pattern[index] = false;
			}
		});
	}

	void generateLevelsGalerkinCoarseningSparseParallelBuild
	(vector<FixedSparseMatrix<T> *> &A_L,
		vector<FixedSparseMatrix<T> *> &R_L,
		vector<FixedSparseMatrix<T> *> &P_L,
		vector<vector<bool>*>           &b_L,
		int & total_level,
		//given
		FixedSparseMatrix<T> &A,
		vector<FLUID::Vec3i> & Bulk_ijk) {
		cout << "building levels ...... " << endl;
		vector<FLUID::Vec3i> Dof_ijk_fine;
		vector<FLUID::Vec3i> Dof_ijk_coarse;
		vector<FLUID::Vec3i> S_L;
		Dof_ijk_fine.resize(Bulk_ijk.size() * 512);
		tbb::parallel_for((size_t)0,
			(size_t)Bulk_ijk.size(),
			(size_t)1,
			[&](size_t index)
		{
			for (int i = 0;i < 512;i++)
			{
				size_t g_idx = 512 * index + i;
				int ii = i % 8, jj = (i % 64) / 8, kk = i / 64;
				Dof_ijk_fine[g_idx] = 8*Bulk_ijk[index] + FLUID::Vec3i(ii, jj, kk);
			}
		});
		A_L.resize(0);
		S_L.resize(0);
		R_L.resize(0);
		P_L.resize(0);
		b_L.resize(0);
		total_level = 1;
		A_L.push_back(&A);
		b_L.push_back(new vector<bool>);
		S_L.push_back(FLUID::Vec3i(0, 0, 0));
		unsigned int unknowns = A.n;
		//parallel build
		int base = 8;
		// TODO: temp
		vector<FLUID::Vec3i> temp;
		for (int iter = 0;iter < 3;iter++)
		{
			base /= 2;
			if (unknowns <= 4096) continue;
			A_L.push_back(new FixedSparseMatrix<T>);
			R_L.push_back(new FixedSparseMatrix<T>);
			P_L.push_back(new FixedSparseMatrix<T>);
			S_L.push_back(FLUID::Vec3i(0));
			b_L.push_back(new vector<bool>);
			int i = total_level - 1;
			generateRPSparseParallelBuild(*(A_L[i]), *(R_L[i]), *(P_L[i]), *(b_L[i]), Dof_ijk_fine, Dof_ijk_coarse,base, Bulk_ijk);
//			Dof_ijk_fine.swap(vector<FLUID::Vec3i>());
			Dof_ijk_fine.swap(temp);
			Dof_ijk_fine = Dof_ijk_coarse;
			FixedSparseMatrix<T> temp;
			multiplyMat(*(A_L[i]), *(P_L[i]), temp, 1.0);
			multiplyMat(*(R_L[i]), temp, *(A_L[i + 1]), 0.5);
			temp.resize(0);
			temp.clear();
			unknowns = A_L[i + 1]->n;
			total_level++;
		}
		int nni, nnj, nnk;
		for (int i = 0;i < Dof_ijk_fine.size();i++)
		{
			nni = max(Bulk_ijk[i][0] + 1, nni);
			nnj = max(Bulk_ijk[i][1] + 1, nnj);
			nnk = max(Bulk_ijk[i][2] + 1, nnk);
		}
		S_L[S_L.size() - 1] = FLUID::Vec3i(nni, nnj, nnk);
		//sequential build
		while (unknowns > 4096)
		{
			A_L.push_back(new FixedSparseMatrix<T>);
			R_L.push_back(new FixedSparseMatrix<T>);
			P_L.push_back(new FixedSparseMatrix<T>);
			nni = ceil((float)nni / 2.0);
			nnj = ceil((float)nnj / 2.0);
			nnk = ceil((float)nnk / 2.0);

			S_L.push_back(FLUID::Vec3i(nni, nnj, nnk));
			b_L.push_back(new vector<bool>);

			int i = total_level - 1;
			//printf("generating R and P\n");
			generateRPSparse(*(A_L[i]), *(R_L[i]), *(P_L[i]), *(b_L[i]),
				Dof_ijk_fine, Dof_ijk_coarse,
				S_L[i].v[0], S_L[i].v[1], S_L[i].v[2]);
			Dof_ijk_fine.resize(0); Dof_ijk_fine.shrink_to_fit();
			Dof_ijk_fine = Dof_ijk_coarse;
			//printf("generating R and P done!\n");
			//printf("%d,%d,%d\n",A_L[i]->n, P_L[i]->n, R_L[i]->n);
			FixedSparseMatrix<T> temp;
			multiplyMat(*(A_L[i]), *(P_L[i]), temp, 1.0);
			multiplyMat(*(R_L[i]), temp, *(A_L[i + 1]), 0.5);
			//printf("multiply matrix done\n");
			temp.resize(0);
			temp.clear();


			unknowns = A_L[i + 1]->n;
			total_level++;
		}


		generatePatternSparse(Dof_ijk_fine,
			*(b_L[total_level - 1]));

		Dof_ijk_fine.resize(0);
		Dof_ijk_fine.shrink_to_fit();


		cout << "build levels done" << endl;
	}




























};




#endif