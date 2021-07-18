#ifndef _AMG_H_
#define _AMG_H_

#include <iostream>
#include <vector>
#include "tbb/tbb.h"
#include "util.h"
#include "vec.h"
#include <cmath>
#include "sparse_matrix.h"
#include "blas_wrapper.h"
#include "GeometricLevelGen.h"
#include "pcg_solver.h"
/*
given A_L, R_L, P_L, b,compute x using
Multigrid Cycles.

*/
using namespace std;
using namespace BLAS;
template<class T>
void RBGS(const FixedSparseMatrix<T> &A, 
	const vector<T> &b,
	vector<T> &x, 
	int ni, int nj, int nk, int iternum)
{

	for (int iter=0;iter<iternum;iter++)
	{
		size_t num = ni*nj*nk;
		size_t slice = ni*nj;
		tbb::parallel_for((size_t)0, num, (size_t)1, [&](size_t thread_idx){
			int k = thread_idx/slice;
			int j = (thread_idx%slice)/ni;
			int i = thread_idx%ni;
			if(k<nk && j<nj && i<ni)
			{
				if ((i+j+k)%2 == 1)
				{
					unsigned int index = (unsigned int)thread_idx;
					T sum = 0;
					T diag= 0;
					for (int ii=A.rowstart[index];ii<A.rowstart[index+1];ii++)
					{
						if(A.colindex[ii]!=index)//none diagonal terms
						{
							sum += A.value[ii]*x[A.colindex[ii]];
						}
						else//record diagonal value A(i,i)
						{
							diag = A.value[ii];
						}
					}//A(i,:)*x for off-diag terms
					if(diag!=0)
					{
						x[index] = (b[index] - sum)/diag;
					}
					else
					{
						x[index] = 0;
					}
				}
			}

		});

		tbb::parallel_for((size_t)0, num, (size_t)1, [&](size_t thread_idx){
			int k = thread_idx/slice;
			int j = (thread_idx%slice)/ni;
			int i = thread_idx%ni;
			if(k<nk && j<nj && i<ni)
			{
				if ((i+j+k)%2 == 0)
				{
					unsigned int index = (unsigned int)thread_idx;
					T sum = 0;
					T diag= 0;
					for (int ii=A.rowstart[index];ii<A.rowstart[index+1];ii++)
					{
						if(A.colindex[ii]!=index)//none diagonal terms
						{
							sum += A.value[ii]*x[A.colindex[ii]];
						}
						else//record diagonal value A(i,i)
						{
							diag = A.value[ii];
						}
					}//A(i,:)*x for off-diag terms
					if(diag!=0)
					{
						x[index] = (b[index] - sum)/diag;
					}
					else
					{
						x[index] = 0;
					}
				}
			}

		});
	}
}

template<class T>
void restriction(const FixedSparseMatrix<T> &R,
	const FixedSparseMatrix<T> &A,
	const vector<T>            &x,
	const vector<T>            &b_curr,
	vector<T>                  &b_next)
{
	b_next.assign(b_next.size(),0);
	vector<T> r = b_curr;
	multiply_and_subtract(A,x,r);
	multiply(R,r,b_next);
	r.resize(0);
}
template<class T>
void prolongatoin(const FixedSparseMatrix<T> &P,
	const vector<T>            &x_curr,
	vector<T>                  &x_next)
{
	vector<T> xx;
	xx.resize(x_next.size());
	xx.assign(xx.size(),0);
	multiply(P,x_curr,xx);//xx = P*x_curr;
	add_scaled(1.0,xx,x_next);
	xx.resize(0);
}
template<class T>
void amgVCycle(vector<FixedSparseMatrix<T> *> &A_L,
	vector<FixedSparseMatrix<T> *> &R_L,
	vector<FixedSparseMatrix<T> *> &P_L,
	vector<FLUID::Vec3i>                  &S_L,
	vector<T>                      &x,
	const vector<T>                &b)
{
	int total_level = A_L.size();
	vector<vector<T>> x_L;
	vector<vector<T>> b_L;
	x_L.resize(total_level);
	b_L.resize(total_level);
	b_L[0] = b;
	x_L[0] = x;
	for(int i=1;i<total_level;i++)
	{
		int unknowns = S_L[i].v[0]*S_L[i].v[1]*S_L[i].v[2];
		x_L[i].resize(unknowns);
		x_L[i].assign(x_L[i].size(),0);
		b_L[i].resize(unknowns);
		b_L[i].assign(b_L[i].size(),0);
	}

	for (int i=0;i<total_level-1;i++)
	{
		RBGS(*(A_L[i]),b_L[i],x_L[i],S_L[i].v[0],S_L[i].v[1],S_L[i].v[2],4);
		restriction(*(R_L[i]),*(A_L[i]),x_L[i],b_L[i],b_L[i+1]);
	}
	int i = total_level-1;
	RBGS(*(A_L[i]),b_L[i],x_L[i],S_L[i].v[0],S_L[i].v[1],S_L[i].v[2],200);
	for (int i=total_level-2;i>=0;i--)
	{
		prolongatoin(*(P_L[i]),x_L[i+1],x_L[i]);
		RBGS(*(A_L[i]),b_L[i],x_L[i],S_L[i].v[0],S_L[i].v[1],S_L[i].v[2],4);
	}
	x = x_L[0];

	for(int i=0;i<total_level;i++)
	{

		x_L[i].resize(0);
		x_L[i].shrink_to_fit();
		b_L[i].resize(0);
		b_L[i].shrink_to_fit();
	}
}
template<class T>
void amgPrecond(vector<FixedSparseMatrix<T> *> &A_L,
	vector<FixedSparseMatrix<T> *> &R_L,
	vector<FixedSparseMatrix<T> *> &P_L,
	vector<FLUID::Vec3i>                  &S_L,
	vector<T>                      &x,
	const vector<T>                &b)
{
	x.resize(b.size());
	x.assign(x.size(),0);
	amgVCycle(A_L,R_L,P_L,S_L,x,b);
}
template<class T>
bool AMGPCGSolvePrebuilt(const FixedSparseMatrix<T> &fixed_matrix,
	const std::vector<T> &rhs,
	std::vector<T> &result,
	vector<FixedSparseMatrix<T> *> &A_L,
	vector<FixedSparseMatrix<T> *> &R_L,
	vector<FixedSparseMatrix<T> *> &P_L,
	vector<FLUID::Vec3i>                  &S_L,
	const int total_level,
	T tolerance_factor,
	int max_iterations,
	T &residual_out,
	int &iterations_out,
	int ni, int nj, int nk)
{
	vector<T>                      m, z, s, r;
	unsigned int n = ni*nj*nk;
	if (m.size() != n) { m.resize(n); s.resize(n); z.resize(n); r.resize(n); }
	zero(result);
	r = rhs;
	residual_out = BLAS::abs_max(r);
	if (residual_out == 0) {
		iterations_out = 0;
		return true;
	}
	double tol = tolerance_factor*residual_out;

	amgPrecond(A_L, R_L, P_L, S_L, z, r);
	double rho = BLAS::dot(z, r);
	if (rho == 0 || rho != rho) {
		iterations_out = 0;
		return false;
	}

	s = z;

	int iteration;
	for (iteration = 0; iteration<max_iterations; ++iteration) {
		multiply(fixed_matrix, s, z);
		double alpha = rho / BLAS::dot(s, z);
		BLAS::add_scaled(alpha, s, result);
		BLAS::add_scaled(-alpha, z, r);
		residual_out = BLAS::abs_max(r);
		if (residual_out <= tol) {
			iterations_out = iteration + 1;
			return true;
		}
		amgPrecond(A_L, R_L, P_L, S_L, z, r);
		double rho_new = BLAS::dot(z, r);
		double beta = rho_new / rho;
		BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
		rho = rho_new;
	}
	iterations_out = iteration;
	return false;
}
template<class T>
bool AMGPCGSolve(const SparseMatrix<T> &matrix, 
	const std::vector<T> &rhs, 
	std::vector<T> &result, 
	T tolerance_factor,
	int max_iterations,
	T &residual_out, 
	int &iterations_out,
	int ni, int nj, int nk) 
{
	FixedSparseMatrix<T> fixed_matrix;
	fixed_matrix.construct_from_matrix(matrix);
	vector<FixedSparseMatrix<T> *> A_L;
	vector<FixedSparseMatrix<T> *> R_L;
	vector<FixedSparseMatrix<T> *> P_L;
	vector<FLUID::Vec3i>                  S_L;
	vector<T>                      m,z,s,r;
	int total_level;
	levelGen<T> amg_levelGen;
	amg_levelGen.generateLevelsGalerkinCoarsening(A_L,R_L,P_L,S_L,total_level,fixed_matrix,ni,nj,nk);

	unsigned int n=matrix.n;
	if(m.size()!=n){ m.resize(n); s.resize(n); z.resize(n); r.resize(n); }
	zero(result);
	r=rhs;
	residual_out=BLAS::abs_max(r);
	if(residual_out==0) {
		iterations_out=0;


		for (int i=0; i<total_level; i++)
		{
			A_L[i]->clear();
		}
		for (int i=0; i<total_level-1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}

		return true;
	}
	double tol=tolerance_factor*residual_out;

	amgPrecond(A_L,R_L,P_L,S_L,z,r);
	double rho=BLAS::dot(z, r);
	if(rho==0 || rho!=rho) {
		for (int i=0; i<total_level; i++)
		{
			A_L[i]->clear();

		}
		for (int i=0; i<total_level-1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}
		iterations_out=0;
		return false;
	}

	s=z;

	int iteration;
	for(iteration=0; iteration<max_iterations; ++iteration){
		multiply(fixed_matrix, s, z);
		double alpha=rho/BLAS::dot(s, z);
		BLAS::add_scaled(alpha, s, result);
		BLAS::add_scaled(-alpha, z, r);
		residual_out=BLAS::abs_max(r);
		if(residual_out<=tol) {
			iterations_out=iteration+1;

			for (int i=0; i<total_level; i++)
			{
				A_L[i]->clear();

			}
			for (int i=0; i<total_level-1; i++)
			{

				R_L[i]->clear();
				P_L[i]->clear();

			}

			return true; 
		}
		amgPrecond(A_L,R_L,P_L,S_L,z,r);
		double rho_new=BLAS::dot(z, r);
		double beta=rho_new/rho;
		BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
		rho=rho_new;
	}
	iterations_out=iteration;
	for (int i=0; i<total_level; i++)
	{
		A_L[i]->clear();

	}
	for (int i=0; i<total_level-1; i++)
	{

		R_L[i]->clear();
		P_L[i]->clear();

	}
	return false;
}













template<class T>
void RBGS_with_pattern(const FixedSparseMatrix<T> &A, 
	const vector<T> &b,
	vector<T> &x, 
	vector<bool> & pattern,
	int iternum)
{

	for (int iter=0;iter<iternum;iter++)
	{
		size_t num = x.size();

		tbb::parallel_for((size_t)0, num, (size_t)1, [&](size_t thread_idx){



			if(pattern[thread_idx]==true)
			{
				T sum = 0;
				T diag= 0;
				for (int ii=A.rowstart[thread_idx];ii<A.rowstart[thread_idx+1];ii++)
				{
					if(A.colindex[ii]!=thread_idx)//none diagonal terms
					{
						sum += A.value[ii]*x[A.colindex[ii]];
					}
					else//record diagonal value A(i,i)
					{
						diag = A.value[ii];
					}
				}//A(i,:)*x for off-diag terms
				if(diag!=0)
				{
					x[thread_idx] = (b[thread_idx] - sum)/diag;
				}
				else
				{
					x[thread_idx] = 0;
				}
			}

		});

		tbb::parallel_for((size_t)0, num, (size_t)1, [&](size_t thread_idx){



			if(pattern[thread_idx]==false)
			{
				T sum = 0;
				T diag= 0;
				for (int ii=A.rowstart[thread_idx];ii<A.rowstart[thread_idx+1];ii++)
				{
					if(A.colindex[ii]!=thread_idx)//none diagonal terms
					{
						sum += A.value[ii]*x[A.colindex[ii]];
					}
					else//record diagonal value A(i,i)
					{
						diag = A.value[ii];
					}
				}//A(i,:)*x for off-diag terms
				if(diag!=0)
				{
					x[thread_idx] = (b[thread_idx] - sum)/diag;
				}
				else
				{
					x[thread_idx] = 0;
				}
			}

		});
	}
}







template<class T>
void amgVCycleCompressed(vector<FixedSparseMatrix<T> *> &A_L,
	vector<FixedSparseMatrix<T> *> &R_L,
	vector<FixedSparseMatrix<T> *> &P_L,
	vector<vector<bool> *>         &p_L,
	vector<T>                      &x,
	const vector<T>                &b)
{
	int total_level = A_L.size();
	vector<vector<T>> x_L;
	vector<vector<T>> b_L;
	x_L.resize(total_level);
	b_L.resize(total_level);
	b_L[0] = b;
	x_L[0] = x;
	for(int i=1;i<total_level;i++)
	{
		int unknowns = A_L[i]->n;
		x_L[i].resize(unknowns);
		x_L[i].assign(x_L[i].size(),0);
		b_L[i].resize(unknowns);
		b_L[i].assign(b_L[i].size(),0);
	}

	for (int i=0;i<total_level-1;i++)
	{
		//printf("level: %d, RBGS\n", i);
		RBGS_with_pattern(*(A_L[i]),b_L[i],x_L[i],*(p_L[i]),4);
		//printf("level: %d, restriction\n", i);
		restriction(*(R_L[i]),*(A_L[i]),x_L[i],b_L[i],b_L[i+1]);
	}
	int i = total_level-1;
	//printf("level: %d, top solve\n", i);
	RBGS_with_pattern(*(A_L[i]),b_L[i],x_L[i],*(p_L[i]),500);

	for (int i=total_level-2;i>=0;i--)
	{
		//printf("level: %d, prolongation\n", i);
		prolongatoin(*(P_L[i]),x_L[i+1],x_L[i]);
		//printf("level: %d, RBGS\n", i);
		RBGS_with_pattern(*(A_L[i]),b_L[i],x_L[i],*(p_L[i]),4);
	}
	x = x_L[0];

	for(int i=0;i<total_level;i++)
	{

		x_L[i].resize(0);
		x_L[i].shrink_to_fit();
		b_L[i].resize(0);
		b_L[i].shrink_to_fit();
	}
}
template<class T>
void amgPrecondCompressed(vector<FixedSparseMatrix<T> *> &A_L,
	vector<FixedSparseMatrix<T> *> &R_L,
	vector<FixedSparseMatrix<T> *> &P_L,
	vector<vector<bool> * >        &p_L,
	vector<T>                      &x,
	const vector<T>                &b)
{
	//printf("preconditioning begin\n");
	x.resize(b.size());
	x.assign(x.size(),0);
	amgVCycleCompressed(A_L,R_L,P_L,p_L,x,b);
	//printf("preconditioning finished\n");
}
template<class T>
bool AMGPCGSolveCompressed(const SparseMatrix<T> &matrix, 
	const std::vector<T> &rhs, 
	std::vector<T> &result, 
	vector<char> &mask,
	vector<int>  &index_table,
	T tolerance_factor,
	int max_iterations,
	T &residual_out, 
	int &iterations_out,
	int ni, int nj, int nk) 
{
	FixedSparseMatrix<T> fixed_matrix;
	fixed_matrix.construct_from_matrix(matrix);
	vector<FixedSparseMatrix<T> *> A_L;
	vector<FixedSparseMatrix<T> *> R_L;
	vector<FixedSparseMatrix<T> *> P_L;
	vector<vector<bool>*>          p_L;
	vector<T>                      m,z,s,r;
	int total_level;
	levelGen<T> amg_levelGen;
	amg_levelGen.generateLevelsGalerkinCoarseningCompressed(A_L,R_L,P_L,p_L,
		total_level,fixed_matrix,mask,index_table,ni,nj,nk);

	unsigned int n=matrix.n;
	if(m.size()!=n){ m.resize(n); s.resize(n); z.resize(n); r.resize(n); }
	zero(result);
	r=rhs;
	residual_out=BLAS::abs_max(r);
	if(residual_out==0) {
		iterations_out=0;


		for (int i=0; i<total_level; i++)
		{
			A_L[i]->clear();
		}
		for (int i=0; i<total_level-1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}

		return true;
	}
	double tol=tolerance_factor*residual_out;

	amgPrecondCompressed(A_L,R_L,P_L,p_L,z,r);
	//cout<<"first precond done"<<endl;
	double rho=BLAS::dot(z, r);
	if(rho==0 || rho!=rho) {
		for (int i=0; i<total_level; i++)
		{
			A_L[i]->clear();

		}
		for (int i=0; i<total_level-1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}
		iterations_out=0;
		return false;
	}

	s=z;

	int iteration;
	for(iteration=0; iteration<max_iterations; ++iteration){
		multiply(fixed_matrix, s, z);
		//printf("multiply done\n");
		double alpha=rho/BLAS::dot(s, z);
		//printf("%d,%d,%d,%d\n",s.size(),z.size(),r.size(),result.size());
		BLAS::add_scaled(alpha, s, result);
		BLAS::add_scaled(-alpha, z, r);
		residual_out=BLAS::abs_max(r);

		if(residual_out<=tol) {
			iterations_out=iteration+1;

			for (int i=0; i<total_level; i++)
			{
				A_L[i]->clear();

			}
			for (int i=0; i<total_level-1; i++)
			{

				R_L[i]->clear();
				P_L[i]->clear();

			}

			return true; 
		}
		amgPrecondCompressed(A_L,R_L,P_L,p_L,z,r);
		//cout<<"second precond done"<<endl;
		double rho_new=BLAS::dot(z, r);
		double beta=rho_new/rho;
		BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
		rho=rho_new;
	}
	iterations_out=iteration;
	for (int i=0; i<total_level; i++)
	{
		A_L[i]->clear();

	}
	for (int i=0; i<total_level-1; i++)
	{

		R_L[i]->clear();
		P_L[i]->clear();

	}
	return false;
}

template<class T>
bool AMGPCGSolveSparse(const SparseMatrix<T> &matrix, 
	const std::vector<T> &rhs, 
	std::vector<T> &result, 
	vector<FLUID::Vec3i> &Dof_ijk,
	T tolerance_factor,
	int max_iterations,
	T &residual_out, 
	int &iterations_out,
	int ni, int nj, int nk,
    bool PURE_NEUMANN)
{
	FixedSparseMatrix<T> fixed_matrix;
	fixed_matrix.construct_from_matrix(matrix);
	vector<FixedSparseMatrix<T> *> A_L;
	vector<FixedSparseMatrix<T> *> R_L;
	vector<FixedSparseMatrix<T> *> P_L;
	vector<vector<bool>*>          p_L;
	vector<T>                      m,z,s,r,v;
	int total_level;
	levelGen<T> amg_levelGen;
	amg_levelGen.generateLevelsGalerkinCoarseningSparse
		(A_L,R_L,P_L,p_L,total_level,fixed_matrix,Dof_ijk,ni,nj,nk);

	unsigned int n=matrix.n;
	if(m.size()!=n){ m.resize(n); s.resize(n); z.resize(n); r.resize(n); if (PURE_NEUMANN) { v.resize(n); } }
	zero(result);
	r=rhs;
    if (PURE_NEUMANN) {
        double mean = BLAS::mean(r);
        v = r;
        BLAS::subtractConst(v, mean);
    }
    residual_out = PURE_NEUMANN ? BLAS::abs_max(v) : BLAS::abs_max(r);
	if(residual_out==0) {
		iterations_out=0;


		for (int i=0; i<total_level; i++)
		{
			A_L[i]->clear();
		}
		for (int i=0; i<total_level-1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}

		return true;
	}
	double tol=tolerance_factor*residual_out;
    if (PURE_NEUMANN) r = v;
	amgPrecondCompressed(A_L,R_L,P_L,p_L,z,r);
	//cout<<"first precond done"<<endl;
	double rho=BLAS::dot(z, r);
	if(rho==0 || rho!=rho) {
		for (int i=0; i<total_level; i++)
		{
			A_L[i]->clear();

		}
		for (int i=0; i<total_level-1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}
		iterations_out=0;
		return false;
	}

	s=z;

	int iteration;
	for(iteration=0; iteration<max_iterations; ++iteration){
		multiply(fixed_matrix, s, z);
		//printf("multiply done\n");
		double alpha=rho/BLAS::dot(s, z);
		//printf("%d,%d,%d,%d\n",s.size(),z.size(),r.size(),result.size());
		BLAS::add_scaled(alpha, s, result);
		BLAS::add_scaled(-alpha, z, r);
        if (PURE_NEUMANN) {
            double mean = BLAS::mean(r);
            v = r;
            BLAS::subtractConst(v, mean);
        }
        residual_out = PURE_NEUMANN ? BLAS::abs_max(v) : BLAS::abs_max(r);

		if(residual_out<=tol) {
			iterations_out=iteration+1;

			for (int i=0; i<total_level; i++)
			{
				A_L[i]->clear();

			}
			for (int i=0; i<total_level-1; i++)
			{

				R_L[i]->clear();
				P_L[i]->clear();

			}

			return true; 
		}
        if (PURE_NEUMANN) r = v;
		amgPrecondCompressed(A_L,R_L,P_L,p_L,z,r);
		//cout<<"second precond done"<<endl;
		double rho_new=BLAS::dot(z, r);
		double beta=rho_new/rho;
		BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
		rho=rho_new;
	}
	iterations_out=iteration;
	for (int i=0; i<total_level; i++)
	{
		A_L[i]->clear();

	}
	for (int i=0; i<total_level-1; i++)
	{

		R_L[i]->clear();
		P_L[i]->clear();

	}
	return false;
}

template<class T>
bool AMGPCGSolveSparseParallelBuild(const SparseMatrix<T> &matrix,
	const std::vector<T> &rhs,
	std::vector<T> &result,
	T tolerance_factor,
	int max_iterations,
	T &residual_out,
	int &iterations_out,
	vector<FLUID::Vec3i> Bulk_ijk)
{
	FixedSparseMatrix<T> fixed_matrix;
	fixed_matrix.construct_from_matrix(matrix);
	vector<FixedSparseMatrix<T> *> A_L;
	vector<FixedSparseMatrix<T> *> R_L;
	vector<FixedSparseMatrix<T> *> P_L;
	vector<vector<bool>*>          p_L;
	vector<T>                      m, z, s, r;
	int total_level;
	levelGen<T> amg_levelGen;
	amg_levelGen.generateLevelsGalerkinCoarseningSparseParallelBuild
	(A_L, R_L, P_L, p_L, total_level, fixed_matrix, Bulk_ijk);

	unsigned int n = matrix.n;
	if (m.size() != n) { m.resize(n); s.resize(n); z.resize(n); r.resize(n); }
	zero(result);
	r = rhs;
	residual_out = BLAS::abs_max(r);
	if (residual_out == 0) {
		iterations_out = 0;


		for (int i = 0; i<total_level; i++)
		{
			A_L[i]->clear();
		}
		for (int i = 0; i<total_level - 1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}

		return true;
	}
	double tol = tolerance_factor*residual_out;

	amgPrecondCompressed(A_L, R_L, P_L, p_L, z, r);
	//cout<<"first precond done"<<endl;
	double rho = BLAS::dot(z, r);
	if (rho == 0 || rho != rho) {
		for (int i = 0; i<total_level; i++)
		{
			A_L[i]->clear();

		}
		for (int i = 0; i<total_level - 1; i++)
		{

			R_L[i]->clear();
			P_L[i]->clear();

		}
		iterations_out = 0;
		return false;
	}

	s = z;

	int iteration;
	for (iteration = 0; iteration<max_iterations; ++iteration) {
		multiply(fixed_matrix, s, z);
		//printf("multiply done\n");
		double alpha = rho / BLAS::dot(s, z);
		//printf("%d,%d,%d,%d\n",s.size(),z.size(),r.size(),result.size());
		BLAS::add_scaled(alpha, s, result);
		BLAS::add_scaled(-alpha, z, r);
		residual_out = BLAS::abs_max(r);

		if (residual_out <= tol) {
			iterations_out = iteration + 1;

			for (int i = 0; i<total_level; i++)
			{
				A_L[i]->clear();

			}
			for (int i = 0; i<total_level - 1; i++)
			{

				R_L[i]->clear();
				P_L[i]->clear();

			}

			return true;
		}
		amgPrecondCompressed(A_L, R_L, P_L, p_L, z, r);
		//cout<<"second precond done"<<endl;
		double rho_new = BLAS::dot(z, r);
		double beta = rho_new / rho;
		BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
		rho = rho_new;
	}
	iterations_out = iteration;
	for (int i = 0; i<total_level; i++)
	{
		A_L[i]->clear();

	}
	for (int i = 0; i<total_level - 1; i++)
	{

		R_L[i]->clear();
		P_L[i]->clear();

	}
	return false;
}

#endif