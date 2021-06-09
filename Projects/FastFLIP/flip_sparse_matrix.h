#pragma once
#ifndef FLIP_SPARSE_MATRIX_H
#define FLIP_SPARSE_MATRIX_H
//#define TBB_PREVIEW_SERIAL_SUBSET 1
#include <iostream>
#include <vector>
#include "tbb/parallel_for.h"
#include "util.h"
#include "zxxtypedefs.h"
//============================================================================
// Dynamic compressed sparse row matrix.

template<class T>
struct SparseMatrix
{
	unsigned int n; // dimension
	std::vector<std::vector<unsigned int> > index; // for each row, a list of all column indices (sorted)
	std::vector<std::vector<T> > value; // values corresponding to index
	~SparseMatrix()
	{
		clear();
	}
	explicit SparseMatrix(unsigned int n_=0, unsigned int expected_nonzeros_per_row=7)
		: n(n_), index(n_), value(n_)
	{
		for(unsigned int i=0; i<n; ++i){
			index[i].reserve(expected_nonzeros_per_row);
			value[i].reserve(expected_nonzeros_per_row);
		}
	}

	void clear(void)
	{
		n=0;
		for (int i=0;i<index.size();i++)
		{
			index[i].resize(0);
			index[i].shrink_to_fit();
		}
		for (int i=0;i<value.size();i++)
		{
			value[i].resize(0);
			value[i].shrink_to_fit();
		}
		index.resize(0); index.shrink_to_fit();
		value.resize(0); value.shrink_to_fit();
	}
	void reserve(int x)
	{
		for (unsigned int i = 0; i<n; ++i) {
			index[i].reserve(x);
			index[i].resize(0);
			value[i].reserve(x);
			value[i].resize(0);
		}
	}
	void zero(void)
	{
		for(unsigned int i=0; i<n; ++i){
			index[i].resize(0);
			value[i].resize(0);
		}
	}

	void resize(int n_)
	{
		n=n_;
		index.resize(n);
		value.resize(n);
	}

	T operator()(unsigned int i, unsigned int j) const
	{
		for(unsigned int k=0; k<index[i].size(); ++k){
			if(index[i][k]==j) return value[i][k];
			else if(index[i][k]>j) return 0;
		}
		return 0;
	}

	void set_element(unsigned int i, unsigned int j, T new_value)
	{
		unsigned int k=0;
		for(; k<index[i].size(); ++k){
			if(index[i][k]==j){
				value[i][k]=new_value;
				return;
			}else if(index[i][k]>j){
				LosTopos::insert(index[i], k, j);
				LosTopos::insert(value[i], k, new_value);
				return;
			}
		}
		index[i].push_back(j);
		value[i].push_back(new_value);
	}

	void add_to_element(unsigned int i, unsigned int j, T increment_value)
	{
		unsigned int k=0;
		for(; k<index[i].size(); ++k){
			if(index[i][k]==j){
				value[i][k]+=increment_value;
				return;
			}else if(index[i][k]>j){
				LosTopos::insert(index[i], k, j);
				LosTopos::insert(value[i], k, increment_value);
				return;
			}
		}
		index[i].push_back(j);
		value[i].push_back(increment_value);
	}

	// assumes indices is already sorted
	void add_sparse_row(unsigned int i, const std::vector<unsigned int> &indices, const std::vector<T> &values)
	{
		unsigned int j=0, k=0;
		while(j<indices.size() && k<index[i].size()){
			if(index[i][k]<indices[j]){
				++k;
			}else if(index[i][k]>indices[j]){
				LosTopos::insert(index[i], k, indices[j]);
				LosTopos::insert(value[i], k, values[j]);
				++j;
			}else{
				value[i][k]+=values[j];
				++j;
				++k;
			}
		}
		for(;j<indices.size(); ++j){
			index[i].push_back(indices[j]);
			value[i].push_back(values[j]);
		}
	}

	void add_sparse_row(unsigned int i, const std::vector<unsigned int> &indices, T multiplier, const std::vector<T> &values)
	{
		assert(i<n);
		unsigned int j=0, k=0;
		while(j<indices.size() && k<index[i].size()){
			if(index[i][k]<indices[j]){
				++k;
			}else if(index[i][k]>indices[j]){
				LosTopos::insert(index[i], k, indices[j]);
				LosTopos::insert(value[i], k, multiplier*values[j]);
				++j;
			}else{
				value[i][k]+=multiplier*values[j];
				++j;
				++k;
			}
		}
		for(;j<indices.size(); ++j){
			index[i].push_back(indices[j]);
			value[i].push_back(multiplier*values[j]);
		}
	}

	// assumes matrix has symmetric structure - so the indices in row i tell us which columns to delete i from
	void symmetric_remove_row_and_column(unsigned int i)
	{
		for(unsigned int a=0; a<index[i].size(); ++a){
			unsigned int j=index[i][a]; // 
			for(unsigned int b=0; b<index[j].size(); ++b){
				if(index[j][b]==i){
					LosTopos::erase(index[j], b);
					LosTopos::erase(value[j], b);
					break;
				}
			}
		}
		index[i].resize(0);
		value[i].resize(0);
	}

	void write_matlab(std::ostream &output, const char *variable_name)
	{
		output<<variable_name<<"=sparse([";
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int j=0; j<index[i].size(); ++j){
				output<<i+1<<" ";
			}
		}
		output<<"],...\n  [";
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int j=0; j<index[i].size(); ++j){
				output<<index[i][j]+1<<" ";
			}
		}
		output<<"],...\n  [";
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int j=0; j<value[i].size(); ++j){
				output<<value[i][j]<<" ";
			}
		}
		output<<"], "<<n<<", "<<n<<");"<<std::endl;
	}
};


template<class T>
void multiply_sparse_matrices_with_diagonal_weighting(const SparseMatrix<T> &A, const std::vector<T> &diagD,
	const SparseMatrix<T> &B, SparseMatrix<T> &C)
{
	//assert(A.n==B.m);
	assert(diagD.size()==A.n);
	C.resize(A.n);
	C.zero();
	for(unsigned int i=0; i<A.n; ++i){
		for(unsigned int p=0; p<A.index[i].size(); ++p){
			unsigned int k=A.index[i][p];
			C.add_sparse_row(i, B.index[k], A.value[i][p]*diagD[k], B.value[k]);
		}
	}
}



typedef SparseMatrix<float> SparseMatrixf;
typedef SparseMatrix<double> SparseMatrixd;

// perform result=matrix*x
template<class T>
void multiply(const SparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result)
{
	//needs parallel
	//assert(matrix.n==x.size());
	result.resize(matrix.n);
	//for(unsigned int i=0; i<matrix.n; ++i){
	//   result[i]=0;
	//   for(unsigned int j=0; j<matrix.index[i].size(); ++j){
	//      result[i]+=matrix.value[i][j]*x[matrix.index[i][j]];
	//   }
	//}
	int num = matrix.n;
	tbb::parallel_for(0,num,1,[&](int i)
	{
		result[i] = 0;
		for(unsigned int j=0; j<matrix.index[i].size(); ++j){
			result[i]+=matrix.value[i][j]*x[matrix.index[i][j]];
		}
	});
}

// perform result=result-matrix*x
template<class T>
void multiply_and_subtract(const SparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result)
{
	//needs parallel
	assert(matrix.n==x.size());
	result.resize(matrix.n);
	//for(unsigned int i=0; i<matrix.n; ++i){
	// for(unsigned int j=0; j<matrix.index[i].size(); ++j){
	//  result[i]-=matrix.value[i][j]*x[matrix.index[i][j]];
	// }
	//}
	int num = matrix.n;
	tbb::parallel_for(0,num,1,[&](int i)
	{
		result[i] = 0;
		for(unsigned int j=0; j<matrix.index[i].size(); ++j){
			result[i]-=matrix.value[i][j]*x[matrix.index[i][j]];
		}
	});
}

//============================================================================
// Fixed version of SparseMatrix. This is not a good structure for dynamically
// modifying the matrix, but can be significantly faster for matrix-vector
// multiplies due to better data locality.

template<class T=float>
struct FixedSparseMatrix
{
	unsigned int n; // dimension
	std::vector<T> value; // nonzero values row by row
	std::vector<unsigned int> colindex; // corresponding column indices
	std::vector<unsigned int> rowstart; // where each row starts in value and colindex (and last entry is one past the end, the number of nonzeros)

	explicit FixedSparseMatrix(unsigned int n_=0)
		: n(n_), value(0), colindex(0), rowstart(n_+1)
	{}
	~FixedSparseMatrix()
	{
		clear();
	}
	void clear(void)
	{
		n=0;
		value.resize(0); value.shrink_to_fit();
		colindex.resize(0); colindex.shrink_to_fit();
		rowstart.resize(0); rowstart.shrink_to_fit();
	}

	void resize(int n_)
	{
		n=n_;
		rowstart.resize(n+1);
	}

	void construct_from_matrix(const SparseMatrix<T> &matrix)
	{
		resize(matrix.n);
		rowstart[0]=0;
		for(unsigned int i=0; i<n; ++i){
			rowstart[i+1]=rowstart[i]+matrix.index[i].size();
		}
		value.resize(rowstart[n]);
		colindex.resize(rowstart[n]);
		unsigned int j=0;
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int k=0; k<matrix.index[i].size(); ++k){
				value[j]=matrix.value[i][k];
				colindex[j]=matrix.index[i][k];
				++j;
			}
		}
	}

	void write_matlab(std::ostream &output, const char *variable_name)
	{
		output<<variable_name<<"=sparse([";
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int j=rowstart[i]; j<rowstart[i+1]; ++j){
				output<<i+1<<" ";
			}
		}
		output<<"],...\n  [";
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int j=rowstart[i]; j<rowstart[i+1]; ++j){
				output<<colindex[j]+1<<" ";
			}
		}
		output<<"],...\n  [";
		for(unsigned int i=0; i<n; ++i){
			for(unsigned int j=rowstart[i]; j<rowstart[i+1]; ++j){
				output<<value[j]<<" ";
			}
		}
		output<<"], "<<n<<", "<<n<<");"<<std::endl;
	}
};

typedef FixedSparseMatrix<float> FixedSparseMatrixf;
typedef FixedSparseMatrix<double> FixedSparseMatrixd;

// perform result=matrix*x
template<class T>
void multiply(const FixedSparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result)
{
	//needs parallel
	//assert(matrix.n==x.size());
	result.resize(matrix.n);
	//for(unsigned int i=0; i<matrix.n; ++i){
	//   result[i]=0;
	//   for(unsigned int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
	//      result[i]+=matrix.value[j]*x[matrix.colindex[j]];
	//   }
	//}

	int num = matrix.n;
	tbb::parallel_for(0,num,1,[&](int i)
	{
		result[i]=0;
		for(unsigned int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
			result[i]+=matrix.value[j]*x[matrix.colindex[j]];
		}

	});


}


// perform C=scale*A*B
template<class T>
void multiplyMat(const FixedSparseMatrix<T> &A, const FixedSparseMatrix<T> &B, FixedSparseMatrix<T> &C, T scale)
{
	//needs parallel
	C.clear();
	SparseMatrix<T> c; 
	c.resize(A.n);
	c.zero();
	tbb::parallel_for((unsigned int)0,(unsigned int)c.n,(unsigned int)1,[&](unsigned int i)
		//for (unsigned int i=0;i<c.n;++i)
	{
		for (unsigned int j=A.rowstart[i]; j<A.rowstart[i+1]; ++j)
		{
			unsigned int k = A.colindex[j];
			T A_ik = A.value[j];
			for (unsigned int kkk=B.rowstart[k];kkk<B.rowstart[k+1];++kkk)
			{
				c.add_to_element(i,B.colindex[kkk],scale*A_ik*B.value[kkk]);
			}

		}

	});

	//for(unsigned int i=0; i<matrix.n; ++i){
	//   result[i]=0;
	//   for(unsigned int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
	//      result[i]+=matrix.value[j]*x[matrix.colindex[j]];
	//   }
	//}

	C.construct_from_matrix(c);
	c.clear();


}


// perform A = coef*B'
template<class T>
void transposeMat(const FixedSparseMatrix<T> &B, FixedSparseMatrix<T> &A, T coef)
{
	A.clear();

	//needs parallel
	SparseMatrix<T> a; 
	unsigned int max_col = 0;
	//find how many columns does B have
	for (unsigned int i=0; i<B.n;++i)
	{
		for (unsigned int j=B.rowstart[i];j<B.rowstart[i+1];++j)
		{
			if (B.colindex[j]>max_col)
			{
				max_col = B.colindex[j];
			}

		}

	}
	a.resize(max_col+1);
	a.zero();

	for (unsigned int i=0; i<B.n;++i)
	{
		for (unsigned int j=B.rowstart[i];j<B.rowstart[i+1];++j)
		{
			T val = B.value[j];
			unsigned int ii = B.colindex[j];
			unsigned int jj = i;
			a.set_element(ii,jj,coef*val);
		}
	}



	//for(unsigned int i=0; i<matrix.n; ++i){
	//   result[i]=0;
	//   for(unsigned int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
	//      result[i]+=matrix.value[j]*x[matrix.colindex[j]];
	//   }
	//}

	A.construct_from_matrix(a);
	a.clear();


}

// perform result=result-matrix*x
template<class T>
void multiply_and_subtract(const FixedSparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result)
{
	//needs parallel
	assert(matrix.n==x.size());
	//result.resize(matrix.n);
	//for(unsigned int i=0; i<matrix.n; ++i){
	//   for(unsigned int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
	//      result[i]-=matrix.value[j]*x[matrix.colindex[j]];
	//   }
	//}
	int num = matrix.n;
	tbb::parallel_for(0,num,1,[&](int i)
	{
		//result[i]=0;
		for(unsigned int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
			result[i]-=matrix.value[j]*x[matrix.colindex[j]];
		}

	});
}

#endif
