#pragma once
//improving implementation of zxx's ampcg
#include "sparse_matrix.h"
#include "vec.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_scan.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/atomic.h"
#include "tbb/concurrent_vector.h"
#include "blas_wrapper_independent.h"
#include "util.h"
#include "AlgebraicMultigrid.h"
#include "morton_encoding.h"
namespace Libo {

	//This is a class dedicated for the restriction and prolongation operator
	template <typename T = float>
	struct sparse_const_entry_matrix {

		void clear() {
			m_scalar = T(1);
			m_nrows = 0;
			m_ncols = 0;
			init();
		}

		sparse_const_entry_matrix() {
			clear();
		};

		sparse_const_entry_matrix(uint64_t in_nrows, uint64_t in_ncols, const T& in_scalar = T(1)) {
			m_nrows = in_nrows;
			m_ncols = in_ncols;
			m_scalar = in_scalar;
			init();
		};

		void init() {
			m_non_zero_cols.resize(m_nrows, std::vector<uint64_t>{});
			m_rowbegin.resize(m_nrows + 1);
			m_flat_cols.clear();
			m_space_for_rhs.resize(m_nrows, 0);
			m_is_compressed = false;
			m_prefer_compressed = true;
		};

		void set_nrows(uint64_t in_nrows) {
			m_nrows = in_nrows;
			init();
		}

		void make_compressed() {
			//turn the std vector of non zeros 
			//into offsets of rows, and list of non zero cols
			m_rowbegin.resize(m_nrows + 1, 0);
			/*m_rowbegin[m_nrows] = tbb::parallel_scan(
				tbb::blocked_range<uint64_t>(uint64_t(0), (uint64_t)m_non_zero_cols.size()),
				uint64_t(0),
				[&](const tbb::blocked_range<uint64_t>& r, uint64_t sum, bool is_final_scan) {
					uint64_t temp = sum;
					for (uint64_t i = r.begin(); i < r.end(); ++i) {
						if (is_final_scan) {
							m_rowbegin[i] = temp;
						}
						temp = temp + m_non_zero_cols[i].size();
					}
					return temp;
				},
				[](uint64_t left, uint64_t right) {
					return left + right;
				}
				);*/
			uint64_t counter = 0;
			for (int i = 0; i < m_non_zero_cols.size(); i++) {
				m_rowbegin[i] = counter;
				counter += m_non_zero_cols[i].size();
			}
			m_rowbegin[m_nrows] = counter;


			//copy the column indices
			m_flat_cols.resize(m_rowbegin[m_nrows], 0);
			tbb::parallel_for((uint64_t)0, m_nrows, [&](uint64_t i_row) {
				//copy the column idx
				std::copy(m_non_zero_cols[i_row].begin(), m_non_zero_cols[i_row].end(), m_flat_cols.begin() + m_rowbegin[i_row]);
				});

			m_is_compressed = true;
		}

		//multiply by vector
		void mulv(const std::vector<T>& rhs, std::vector<T>& out_lhs) {
			if (rhs.size() != m_ncols) {
				printf("mulv rhs size!= m_ncols exit\n");
				exit(-1);
			}
			out_lhs.assign(m_nrows, T(0));
			if (m_nrows < m_ncols) {
				//scale lhs is cheaper
				mulv_scale_lhs(rhs, out_lhs);
			}
			else {
				mulv_scale_rhs(rhs, out_lhs);
			}
		}

		void mulv_scale_rhs(const std::vector<T>& rhs, std::vector<T>& out_lhs) {
			if (m_scalar == T(1)) {
				mulv_no_scale(rhs, out_lhs);
			}
			else {
				m_space_for_rhs = rhs;
				tbb::parallel_for((size_t)0, rhs.size(), [&](size_t i) {
					m_space_for_rhs[i] *= m_scalar;
					});
				mulv_no_scale(m_space_for_rhs, out_lhs);
			}


		}//end mulv scale rhs first

		void mulv_scale_lhs(const std::vector<T>& rhs, std::vector<T>& out_lhs) {
			mulv_no_scale(rhs, out_lhs);
			if (m_scalar == T(1)) {
				return;
			}
			tbb::parallel_for((size_t)0, out_lhs.size(), [&](size_t i) {
				out_lhs[i] *= m_scalar;
				});
		}//end mulv scale lhs first

		void mulv_no_scale(const std::vector<T>& rhs, std::vector<T>& out_lhs) {
			if (m_prefer_compressed && m_is_compressed) {
				mulv_no_scale_compressed(rhs, out_lhs);
			}

			tbb::parallel_for((uint64_t)0, m_nrows, [&](uint64_t i)
				{
					out_lhs[i] = 0;
					for (const auto& col : m_non_zero_cols[i]) {
						out_lhs[i] += rhs[col];
					}
				});
		}//end mulv noscale

		void mulv_no_scale_compressed(const std::vector<T>& rhs, std::vector<T>& out_lhs) {
			tbb::parallel_for((uint64_t)0, m_nrows, [&](uint64_t i)
				{
					out_lhs[i] = 0;
					for (uint64_t coliter = m_rowbegin[i]; coliter < m_rowbegin[i + 1]; coliter++) {
						out_lhs[i] += rhs[m_flat_cols[coliter]];
					}
				});
		}//end mulv noscale compressed

		void set_element(uint64_t i, uint64_t j) {
			if (i >= m_non_zero_cols.size()) {
				printf("set element i>= reserved row number\n");
				exit(-1);
			}
			m_is_compressed = false;
			for (uint64_t k = 0; k < m_non_zero_cols[i].size(); ++k) {
				if (m_non_zero_cols[i][k] == j) {
					return;
				}
				else if (m_non_zero_cols[i][k] > j) {
					LosTopos::insert(m_non_zero_cols[i], k, j);
					return;
				}
			}
			m_non_zero_cols[i].push_back(j);
		}//end set element


		void set_estimated_non_zero_cols(uint64_t in_ncols) {
			for (auto& row : m_non_zero_cols) {
				row.reserve(in_ncols);
			}
		}

		void postmulmat(const FixedSparseMatrix<T>& A, FixedSparseMatrix<T>& C, T scale, uint64_t estimated_nz_cols) {
			//A * B(self)=C
			//needs parallel
			C.clear();
			SparseMatrix<T> c;
			c.resize(A.n);
			c.zero();
			for (auto& i : c.index) {
				i.reserve(estimated_nz_cols);
			}
			for (auto& i : c.value) {
				i.reserve(estimated_nz_cols);
			}

			tbb::parallel_for((uint64_t)0, (uint64_t)c.n, (uint64_t)1, [&](uint64_t i)
				//for (uint64_t i=0;i<c.n;++i)
				{
					for (uint64_t j = A.rowstart[i]; j < A.rowstart[i + 1]; ++j)
					{
						uint64_t k = A.colindex[j];
						T A_ik = A.value[j];

						if (m_prefer_compressed && m_is_compressed) {
							//compressed storage
							for (uint64_t kkk = m_rowbegin[k]; kkk < m_rowbegin[k + 1]; ++kkk)
							{
								c.add_to_element(i, m_flat_cols[kkk], A_ik);
							}
						}
						else {
							//uncompressed storage
							for (uint64_t kkk = 0; kkk < m_non_zero_cols[k].size(); kkk++) {
								c.add_to_element(i, m_non_zero_cols[k][kkk], A_ik);
							}
						}


					}
				});

			C.construct_from_matrix(c);
			c.clear();

			T combined_scalar = m_scalar * scale;
			if (combined_scalar != T(1)) {
				tbb::parallel_for((uint64_t)0, (uint64_t)C.value.size(), [&](uint64_t i) {
					C.value[i] *= combined_scalar;
					});
			}
		}//end post multiply matrix

		void premulmat(const FixedSparseMatrix<T>& B, FixedSparseMatrix<T>& C, T scale, uint64_t estimated_nz_cols) {
			//A(self) * B=C
			//needs parallel
			C.clear();
			SparseMatrix<T> c;
			c.resize(m_non_zero_cols.size());
			c.zero();
			for (uint64_t i = 0; i < m_nrows; i++) {
				c.index[i].reserve(estimated_nz_cols);
				c.value[i].reserve(estimated_nz_cols);
			}

			tbb::parallel_for((uint64_t)0, (uint64_t)c.n, (uint64_t)1, [&](uint64_t i)
				{
					if (m_prefer_compressed && m_is_compressed) {
						for (uint64_t j = m_rowbegin[i]; j < m_rowbegin[i + 1]; j++) {
							uint64_t k = m_flat_cols[j];
							for (uint64_t kkk = B.rowstart[k]; kkk < B.rowstart[k + 1]; kkk++) {
								c.add_to_element(i, B.colindex[kkk], B.value[kkk]);
							}
						}
					}
					else {
						for (uint64_t j = 0; j < m_non_zero_cols[i].size(); j++) {
							uint64_t k = m_non_zero_cols[i][j];
							for (uint64_t kkk = B.rowstart[k]; kkk < B.rowstart[k + 1]; kkk++) {
								c.add_to_element(i, B.colindex[kkk], B.value[kkk]);
							}
						}
					}
				});

			C.construct_from_matrix(c);
			c.clear();

			T combined_scale = m_scalar * scale;
			tbb::parallel_for(uint64_t(0), (uint64_t)C.value.size(), [&](uint64_t i) {
				C.value[i] *= combined_scale;
				});

		}//end pre multiply matrix

		std::vector<T> m_space_for_rhs;

		//size: [nrows][non zero cols at ith row]
		std::vector<std::vector<uint64_t>> m_non_zero_cols;
		std::vector<uint64_t> m_flat_cols;
		std::vector<uint64_t> m_rowbegin;
		uint64_t m_nrows;
		uint64_t m_ncols;
		T m_scalar;
		bool m_is_compressed;
		bool m_prefer_compressed;
	};
	//end class sparse_const_entry_matrix

	template<class T>
	void restriction(sparse_const_entry_matrix<T>& R,
		const FixedSparseMatrix<T>& A,
		const std::vector<T>& x,
		const std::vector<T>& b_curr,
		std::vector<T>& b_next)
	{
		b_next.assign(b_next.size(), 0);
		std::vector<T> r = b_curr;
		multiply_and_subtract(A, x, r);
		R.mulv(r, b_next);
		//multiply(R, r, b_next);
		r.resize(0);
	}
	template<class T>
	void prolongatoin(sparse_const_entry_matrix<T>& P,
		const std::vector<T>& x_curr,
		std::vector<T>& x_next)
	{
		std::vector<T> xx;
		xx.resize(x_next.size());
		xx.assign(xx.size(), 0);
		P.mulv(x_curr, xx);
		//multiply(P, x_curr, xx);//xx = P*x_curr;
		BLAS::add_scaled(T(1.0), xx, x_next);
		xx.resize(0);
	}

	template<class T>
	void amgVCycleCompressed(std::vector<FixedSparseMatrix<T>*>& A_L,
		std::vector<sparse_const_entry_matrix<T>*>& R_L,
		std::vector<sparse_const_entry_matrix<T>*>& P_L,
		std::vector<std::vector<bool>*>& p_L,
		std::vector<T>& x,
		const std::vector<T>& b)
	{
		int total_level = A_L.size();
		std::vector<std::vector<T>> x_L;
		std::vector<std::vector<T>> b_L;
		x_L.resize(total_level);
		b_L.resize(total_level);
		b_L[0] = b;
		x_L[0] = x;
		for (int i = 1; i < total_level; i++)
		{
			int unknowns = A_L[i]->n;
			x_L[i].resize(unknowns);
			x_L[i].assign(x_L[i].size(), 0);
			b_L[i].resize(unknowns);
			b_L[i].assign(b_L[i].size(), 0);
		}
		for (int i = 0; i < total_level - 1; i++)
		{
			//printf("level: %d, RBGS\n", i);
			RBGS_with_pattern_tbb_range(*(A_L[i]), b_L[i], x_L[i], *(p_L[i]), 4);
			//printf("level: %d, restriction\n", i);
			Libo::restriction(*(R_L[i]), *(A_L[i]), x_L[i], b_L[i], b_L[i + 1]);
		}
		int i = total_level - 1;
		//printf("level: %d, top solve\n", i);
		RBGS_with_pattern_tbb_range(*(A_L[i]), b_L[i], x_L[i], *(p_L[i]), 40);

		for (int i = total_level - 2; i >= 0; i--)
		{
			//printf("level: %d, prolongation\n", i);
			Libo::prolongatoin(*(P_L[i]), x_L[i + 1], x_L[i]);
			//printf("level: %d, RBGS\n", i);
			RBGS_with_pattern_tbb_range(*(A_L[i]), b_L[i], x_L[i], *(p_L[i]), 4);
		}
		x = x_L[0];

		for (int i = 0; i < total_level; i++)
		{

			x_L[i].resize(0);
			x_L[i].shrink_to_fit();
			b_L[i].resize(0);
			b_L[i].shrink_to_fit();
		}
	}

	template<class T>
	void amgPrecondCompressed(std::vector<FixedSparseMatrix<T>*>& A_L,
		std::vector<sparse_const_entry_matrix<T>*>& R_L,
		std::vector<sparse_const_entry_matrix<T>*>& P_L,
		std::vector<std::vector<bool>* >& p_L,
		std::vector<T>& x,
		const std::vector<T>& b)
	{
		//printf("preconditioning begin\n");
		x.resize(b.size());
		x.assign(x.size(), 0);
		amgVCycleCompressed(A_L, R_L, P_L, p_L, x, b);
		//printf("preconditioning finished\n");
	}

	template <typename T = float>
	void generatePatternSparse(
		const std::vector<LosTopos::Vec3i>& Dof_ijk,
		std::vector<bool>& pattern)
	{
		pattern.resize(Dof_ijk.size());
		tbb::parallel_for((size_t)0,
			(size_t)Dof_ijk.size(),
			(size_t)1,
			[&](size_t index)
			{
				LosTopos::Vec3i ijk = Dof_ijk[index];
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

	template <typename T = float>
	void genRP_parallel_hashmap(
		const FixedSparseMatrix<T>& A,
		Libo::sparse_const_entry_matrix<T>& r,
		Libo::sparse_const_entry_matrix<T>& p,
		std::vector<bool>& pattern,
		std::vector<LosTopos::Vec3i>& Dof_ijk_fine,
		std::vector<LosTopos::Vec3i>& Dof_ijk_coarse,
		const int ni, const int nj, const int nk) {
		Libo::generatePatternSparse(Dof_ijk_fine, pattern);
		int nni = ceil((float)ni / 2.0);
		int nnj = ceil((float)nj / 2.0);
		int nnk = ceil((float)nk / 2.0);

		//prolongation, input: ndof_corase, output: ndof_fine
		//p by default has m_scalar 1 so there is no need to set it
		p.clear();
		p.set_nrows(A.n);
		//each row of prolongation seems to have ony one coarse input
		//hence just reserve one element
		p.set_estimated_non_zero_cols(1);

		Dof_ijk_coarse.resize(0);
		tbb::atomic<uint64_t> coarse_dof_counter{ 0 };
		tbb::atomic<uint64_t> coarse_morton_counter{ 0 };
		tbb::concurrent_hash_map<uint64_t, uint64_t> coarse_grididx2dofidx;
		tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>> coarse_dofidx2finedofidx;
		tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>> coarse_morton2finedofidx;

		tbb::concurrent_vector<std::pair<LosTopos::Vec3i, uint64_t>> coarse_paired_ijk_morton;
		tbb::concurrent_vector<std::pair<LosTopos::Vec3i, uint64_t>> coarse_paired_ijk_dofidx;
		size_t est_n_coarse_dof = Dof_ijk_fine.size() / (size_t)8;


		bool use_morton_for_coarse_dof = false;

		
		auto process_range_fine_dof = [&](const tbb::blocked_range<uint64_t>& fine_dof_range) {
			tbb::concurrent_hash_map<uint64_t, uint64_t>::accessor axr;
			tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>>::accessor axr_list_of_finedof;
			tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>>::accessor axr_morton_2_finedof;
			for (uint64_t fine_dof_idx = fine_dof_range.begin(); fine_dof_idx < fine_dof_range.end(); fine_dof_idx++) {
				
				const LosTopos::Vec3i& fine_ijk = Dof_ijk_fine[fine_dof_idx];
				int coarse_i = fine_ijk[0] / 2, coarse_j = fine_ijk[1] / 2, coarse_k = fine_ijk[2] / 2;


				//associate each coarse morton code with fine dof
				if (use_morton_for_coarse_dof) {
					uint64_t coarse_morton_code = morton_encode::encode(coarse_i, coarse_j, coarse_k);
					
					if (coarse_morton2finedofidx.insert(axr_morton_2_finedof, std::make_pair(coarse_morton_code, std::vector<uint64_t>{}))) {
						//first time this morton code is accessed
						coarse_morton_counter++;
						coarse_paired_ijk_morton.push_back(
							std::make_pair(LosTopos::Vec3i{ coarse_i,coarse_j,coarse_k }, coarse_morton_code));		
					}
					axr_morton_2_finedof->second.push_back(fine_dof_idx);
					axr_morton_2_finedof.release();
				}
				else {
					uint64_t coarse_grididx = coarse_i + nni * (coarse_j + coarse_k * nnj);

					uint64_t current_coarse_dof_idx = 0;

					//try to see if the coarse grid idx is already visited
					//return true if new
					if (coarse_grididx2dofidx.insert(axr, std::make_pair(coarse_grididx, 0))) {
						//this coarse grid is not yet accessed
						//link it with the current coarse grid dof, and increment the atomic variable
						//so other threads know updated dof
						current_coarse_dof_idx = coarse_dof_counter.fetch_and_increment();
						axr->second = current_coarse_dof_idx;

						coarse_paired_ijk_dofidx.push_back(
							std::make_pair(LosTopos::Vec3i{ coarse_i,coarse_j,coarse_k },
								current_coarse_dof_idx));
					}
					else {
						//this coarse grid is already accessed, 
						current_coarse_dof_idx = axr->second;

					}
					axr.release();

					if (!use_morton_for_coarse_dof) {
						p.set_element(fine_dof_idx, current_coarse_dof_idx);
					}

					//add the fine dof idx to the list of coarse dof idx
					coarse_dofidx2finedofidx.insert(axr_list_of_finedof, std::make_pair(current_coarse_dof_idx, std::vector<uint64_t>{}));
					axr_list_of_finedof->second.push_back(fine_dof_idx);
					axr_list_of_finedof.release();
				}
			}//end for this blocked range
		};//end process range find dof

		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, (uint64_t)Dof_ijk_fine.size()), process_range_fine_dof);

		//printf("coarsedof:%d coarsemorton count:%d  coarsemortonsize:%d morton mapsize:%d\n", coarse_dof_counter, coarse_morton_counter, coarse_paired_ijk_morton.size(), coarse_morton2finedofidx.size());
		//printf("coarsedofcounter:%d grididx2dofidx %d, coarse_paired_ijk_dofidx%d\n", coarse_dof_counter, coarse_grididx2dofidx.size(),coarse_paired_ijk_dofidx.size());
		
		if (use_morton_for_coarse_dof) {
			//we need to sort the coarse dofs according to their morton code
			auto morton_comp = [](const std::pair<LosTopos::Vec3i, uint64_t>& a,
				const std::pair<LosTopos::Vec3i, uint64_t>& b) {
					return a.second < b.second;
			};
			
			tbb::parallel_sort(coarse_paired_ijk_morton.begin(), coarse_paired_ijk_morton.end(), morton_comp);

			//build the p 
			for (uint64_t coarse_dof = 0; coarse_dof < coarse_paired_ijk_morton.size(); coarse_dof++) {
				tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>>::const_accessor temp_axr;
				if (coarse_morton2finedofidx.find(temp_axr, coarse_paired_ijk_morton[coarse_dof].second)) {
					for (uint64_t fine_dofidx : temp_axr->second) {
						p.set_element(fine_dofidx, coarse_dof);
					}
				}
				else {
					printf("coarse morton not found\n");
					exit(-1);
				}
			}//end for coarse dof
		}

		if (use_morton_for_coarse_dof) {
			p.m_ncols = coarse_paired_ijk_morton.size();
		}
		else {
			p.m_ncols = coarse_dof_counter;
		}
		
		//p.make_compressed();

		/**************************************************************/
		//restriction operator
		//input fine grid value, output: coarse_grid value


		r.clear();
		if (use_morton_for_coarse_dof) {
			r.set_nrows(coarse_paired_ijk_morton.size());
		}
		else {
			r.set_nrows(coarse_dof_counter);
		}
		
		//on average, each coarse level output get input from 8 fine grid
		r.set_estimated_non_zero_cols(8);




		if (use_morton_for_coarse_dof) {
			auto process_range_morton_coarse = [&](uint64_t coarse_dof) {
				uint64_t morton = coarse_paired_ijk_morton[coarse_dof].second;
				tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>>::const_accessor axr;
				//get the list of fine grid dof idx
				if (coarse_morton2finedofidx.find(axr, morton)) {
					for (const auto& fine_dof_idx : axr->second) {
						r.set_element(coarse_dof, fine_dof_idx);
					}
				}
				else {
					printf("a coarse dof is not found\n");
					exit(-1);
				}
			};
			tbb::parallel_for((uint64_t)0, r.m_nrows, process_range_morton_coarse);
		}
		else {
			auto process_range_coarse_dof = [&](uint64_t coarse_dof_idx) {
				tbb::concurrent_hash_map<uint64_t, std::vector<uint64_t>>::const_accessor axr;
				//get the list of fine grid dof idx
				if (coarse_dofidx2finedofidx.find(axr, coarse_dof_idx)) {
					for (const auto& fine_dof_idx : axr->second) {
						r.set_element(coarse_dof_idx, fine_dof_idx);
					}
				}
				else {
					printf("a coarse dof is not found\n");
					exit(-1);
				}
			};// end process range coarse dof
			tbb::parallel_for((uint64_t)0, r.m_nrows, process_range_coarse_dof);
		}


		r.m_ncols = Dof_ijk_fine.size();
		//r.make_compressed();
		r.m_scalar = T(0.125);

		//re-order the coarse level ijk
		Dof_ijk_coarse.resize(r.m_nrows);

		if (use_morton_for_coarse_dof) {
			for (uint64_t i = 0; i < r.m_nrows; i++) {
				Dof_ijk_coarse[i] = coarse_paired_ijk_morton[i].first;
			}
		}
		else {
			tbb::parallel_for((uint64_t)0, r.m_nrows, [&](uint64_t iter_unordered_coarse_ijk) {
				uint64_t ordered_ijk = coarse_paired_ijk_dofidx[iter_unordered_coarse_ijk].second;
				Dof_ijk_coarse[ordered_ijk] = coarse_paired_ijk_dofidx[iter_unordered_coarse_ijk].first;
				});
		}

	}//end genRP_parallel_hashmap

	template <typename T = float>
	void genAMGlevels(
		//matrix
		std::vector<FixedSparseMatrix<T>*>& out_A_L,
		//restriction operator
		std::vector<Libo::sparse_const_entry_matrix<T>*>& out_R_L,
		//prolongation operator
		std::vector<Libo::sparse_const_entry_matrix<T>*>& out_P_L,
		//red black pattern
		std::vector<std::vector<bool>*>& out_b_L,
		int& total_level,
		//given
		FixedSparseMatrix<T>& A,
		const std::vector<LosTopos::Vec3i>& Dof_ijk,
		int ni, int nj, int nk) {

		std::cout << "building levels ...... " << std::endl;
		std::vector<LosTopos::Vec3i> Dof_ijk_fine;
		std::vector<LosTopos::Vec3i> Dof_ijk_coarse;
		std::vector<LosTopos::Vec3i> S_L;
		Dof_ijk_fine = Dof_ijk;
		out_A_L.resize(0);
		out_R_L.resize(0);
		out_P_L.resize(0);
		out_b_L.resize(0);
		total_level = 1;
		out_A_L.push_back(&A);
		S_L.push_back(LosTopos::Vec3i(ni, nj, nk));
		out_b_L.push_back(new std::vector<bool>);
		int nni = ni, nnj = nj, nnk = nk;
		uint64_t unknowns = A.n;
		while (unknowns > 1000)
		{
			out_A_L.push_back(new FixedSparseMatrix<T>);
			out_R_L.push_back(new sparse_const_entry_matrix<T>);
			out_P_L.push_back(new sparse_const_entry_matrix<T>);
			nni = ceil((float)nni / 2.0);
			nnj = ceil((float)nnj / 2.0);
			nnk = ceil((float)nnk / 2.0);

			S_L.push_back(LosTopos::Vec3i(nni, nnj, nnk));
			out_b_L.push_back(new std::vector<bool>);

			int i = total_level - 1;
			Libo::genRP_parallel_hashmap(*(out_A_L[i]), *(out_R_L[i]), *(out_P_L[i]), *(out_b_L[i]),
				Dof_ijk_fine, Dof_ijk_coarse,
				S_L[i].v[0], S_L[i].v[1], S_L[i].v[2]);

			Dof_ijk_fine.resize(0); Dof_ijk_fine.shrink_to_fit();
			Dof_ijk_fine = Dof_ijk_coarse;
			//printf("generating R and P done!\n");
			//printf("%d,%d,%d\n",A_L[i]->n, P_L[i]->n, R_L[i]->n);
			FixedSparseMatrix<T> temp;

			//multiplyMat(*(A_L[i]), *(P_L[i]), temp, T(1.0));
			//multiplyMat(*(R_L[i]), temp, *(A_L[i + 1]), T(0.5));

			//out_P_L[i]->postmulmat(*(out_A_L[i]), temp, T(1), /*estimated_nz_cols*/ 7);
			//out_R_L[i]->premulmat(temp, *(out_A_L[i + 1]), T(0.5), 7);

			out_R_L[i]->premulmat(*(out_A_L[i]), temp, T(0.5), 7);
			out_P_L[i]->postmulmat(temp, *(out_A_L[i + 1]), T(1), /*estimated_nz_cols*/ 7);


			//printf("multiply matrix done\n");
			temp.resize(0);
			temp.clear();


			unknowns = out_A_L[i + 1]->n;
			total_level++;
		}


		Libo::generatePatternSparse(Dof_ijk_fine,
			*(out_b_L[total_level - 1]));

		Dof_ijk_fine.resize(0);
		Dof_ijk_fine.shrink_to_fit();

		std::cout << "build levels done" << std::endl;

	};//end genAMGlevels

	template<class T>
	bool AMGPCGSolveSparse(const SparseMatrix<T>& matrix,
		const std::vector<T>& rhs,
		std::vector<T>& result,
		std::vector<LosTopos::Vec3i>& Dof_ijk,
		T tolerance_factor,
		int max_iterations,
		T& residual_out,
		int& iterations_out,
		int ni, int nj, int nk)
	{
		FixedSparseMatrix<T> fixed_matrix;
		fixed_matrix.construct_from_matrix(matrix);
		std::vector<FixedSparseMatrix<T>*> A_L;
		std::vector<Libo::sparse_const_entry_matrix<T>*> R_L;
		std::vector<Libo::sparse_const_entry_matrix<T>*> P_L;
		std::vector<std::vector<bool>*>          p_L;
		std::vector<T>                      m, z, s, r;
		int total_level;
		Libo::genAMGlevels(A_L, R_L, P_L, p_L, total_level, fixed_matrix, Dof_ijk, ni, nj, nk);

		uint64_t n = matrix.n;
		if (m.size() != n) { m.resize(n); s.resize(n); z.resize(n); r.resize(n); }
		LosTopos::zero(result);
		r = rhs;
		residual_out = BLAS::abs_max(r);
		if (residual_out == 0) {
			iterations_out = 0;
			for (int i = 0; i < total_level; i++)
			{
				A_L[i]->clear();
				if (i != 0) {
					delete A_L[i];
				}
				delete p_L[i];
			}
			for (int i = 0; i < total_level - 1; i++)
			{

				R_L[i]->clear();
				P_L[i]->clear();
				delete R_L[i];
				delete P_L[i];
			}

			return true;
		}
		double tol = tolerance_factor * residual_out;
		tol = std::max(tol, 1e-6);

		Libo::amgPrecondCompressed(A_L, R_L, P_L, p_L, z, r);

		//cout<<"first precond done"<<endl;
		double rho = BLAS::dot(z, r);
		if (rho == 0 || rho != rho) {
			for (int i = 0; i < total_level; i++)
			{
				A_L[i]->clear();
				if (i != 0) {
					delete A_L[i];
				}
				delete p_L[i];
			}
			for (int i = 0; i < total_level - 1; i++)
			{

				R_L[i]->clear();
				P_L[i]->clear();
				delete R_L[i];
				delete P_L[i];
			}
			iterations_out = 0;
			return false;
		}

		s = z;

		int iteration;
		for (iteration = 0; iteration < max_iterations; ++iteration) {
			multiply(fixed_matrix, s, z);
			//printf("multiply done\n");
			T alpha = rho / BLAS::dot(s, z);
			//printf("%d,%d,%d,%d\n",s.size(),z.size(),r.size(),result.size());
			BLAS::add_scaled(alpha, s, result);
			BLAS::add_scaled(-alpha, z, r);
			residual_out = BLAS::abs_max(r);
			std::cout << "iteration number: " << iteration << ", residual: " << residual_out << std::endl;

			if (residual_out <= tol) {
				iterations_out = iteration + 1;

				for (int i = 0; i < total_level; i++)
				{
					A_L[i]->clear();
					if (i != 0) {
						delete A_L[i];
					}
					delete p_L[i];
				}
				for (int i = 0; i < total_level - 1; i++)
				{

					R_L[i]->clear();
					P_L[i]->clear();
					delete R_L[i];
					delete P_L[i];
				}

				return true;
			}
			Libo::amgPrecondCompressed(A_L, R_L, P_L, p_L, z, r);
			//cout<<"second precond done"<<endl;
			T rho_new = BLAS::dot(z, r);
			T beta = rho_new / rho;
			BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
			rho = rho_new;
		}
		iterations_out = iteration;
		for (int i = 0; i < total_level; i++)
		{
			A_L[i]->clear();
			if (i != 0) {
				delete A_L[i];
			}
			delete p_L[i];
		}
		for (int i = 0; i < total_level - 1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();
			delete R_L[i];
			delete P_L[i];
		}
	}//end amgpcgsolvesparse

}//end namespace Libo
