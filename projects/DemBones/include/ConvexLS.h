///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#ifndef DEM_BONES_CONVEX_LS
#define DEM_BONES_CONVEX_LS

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "Indexing.h"

namespace Dem
{

/** @class ConvexLS ConvexLS.h "DemBones/ConvexLS.h"
	@brief Linear least squares solver with non-negativity constraint and optional affinity constraint
	@details Solve: 
	@f{eqnarray*}{
		min &||Ax-b||^2 \\ 
		\mbox{Subject to: } & x(0).. x(n-1) \geq 0, \\
		\mbox{(optional) } & x(0) +.. + x(n-1) = 1 
	@f}
	The solver implements active set method to handle non-negativity constraint and QR decomposition to handle affinity constraint.

	@b _Scalar is the floating-point data type.
*/
template<class _Scalar>
class ConvexLS {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	using MatrixX=Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorX=Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;

	/** Constructor, just call init()
		@param[in] maxSize is the maximum size of the unknown @f$ x @f$ if the affinity constraint is imposed.
	*/
	ConvexLS(int maxSize=1) {
		q2.resize(0);
		init(maxSize);
	}
	
	/** Init matrices @f$ Q @f$ in the QR decomposition used for affinity constraint
		@param[in] maxSize is the maximum size of the unknown @f$ x @f$ if the affinity constraint is imposed.
	*/
	void init(int maxSize) {
		int curN=(int)q2.size()+1;
		if (curN<maxSize) {
			q2.resize(maxSize-1);
			#pragma omp parallel for
			for (int n=curN-1; n<maxSize-1; n++)
				q2[n]=MatrixX(VectorX::Constant(n+2, _Scalar(1)).householderQr().householderQ()).rightCols(n+1);
		}
	}
	
	/** Solve the least squares problem
		@param[in] aTa is the cross product matrix @f$ A^TA @f$
		@param[in] aTb is the vector @f$ A^Tb @f$
		@param[in, out] x is the by-reference output and it is also the init solution (if @b warmStart == @c true)
		@param[in] affine=true will impose affinity constraint
		@param[in] warmStart=true will initialize the solution by @b x
	*/
	void solve(const MatrixX& aTa, const VectorX& aTb, VectorX& x, bool affine, bool warmStart=false) {
		int n=int(aTa.cols());

		if (!warmStart) x=VectorX::Constant(n, _Scalar(1)/n);

		Eigen::ArrayXi idx(n);
		int np=0;
		for (int i=0; i<n; i++)
			if (x(i)>0) idx[np++]=i; else idx[n-i+np-1]=i;

		VectorX p;

		for (int rep=0; rep<n; rep++) {
			solveP(aTa, aTb, x, idx, np, affine, p);

			if ((indexing_vector(x, idx.head(np))+indexing_vector(p, idx.head(np))).minCoeff()>=0) {
				x+=p;
				if (np==n) break;
				Eigen::Index iMax;
				(indexing_vector(aTb, idx.tail(n-np))-indexing_row(aTa, idx.tail(n-np))*x).maxCoeff(&iMax);
				std::swap(idx[iMax+np], idx[np]);
				np++;
			} else {
				_Scalar alpha;
				int iMin=-1;
				for (int i=0; i<np; i++)
					if (p(idx[i])<0) {
						if ((iMin==-1)||(x(idx[i])<-alpha*p(idx[i]))) {
							alpha=-x(idx[i])/p(idx[i]);
							iMin=i;
						}
					}
				x+=alpha*p;
				_Scalar eps=std::abs(x(idx[iMin]));
				x(idx[iMin])=0;
				for (int i=0; i<np; i++)
					if (x(idx[i])<=eps) std::swap(idx[i--], idx[--np]);
			}
			if (affine) x/=x.sum();
		}
	}
	
private:
	//! Store @f$ Q @f$ matrices in QR decompositions, except the first column. q2.size()==maxSize-1 (of x), q2[n].size()==(n+2)*(n+1)
	std::vector<MatrixX, Eigen::aligned_allocator<MatrixX>> q2;
	
	/** Solve the gradient
		@param[in] aTa is the cross product matrix @f$ A^TA @f$
		@param[in] aTb is the vector @f$ A^Tb @f$
		@param[in] x is the current solution
		@param[in] idx indicates the current active set, @p idx(0).. @p idx(np-1) are passive (free) variables
		@param[in] np is the size of the active set
		@param[in] zeroSum=true will impose zer-sum of gradient
		@param[output] p is the by-reference negative gradient output
	*/
	void solveP(const MatrixX& aTa, const VectorX& aTb, const VectorX& x, const Eigen::ArrayXi& idx, int np, bool zeroSum, VectorX& p) {
		VectorX z;
		p.setZero(aTb.size());
		if (!zeroSum) {
			z=	indexing_row_col(aTa, idx.head(np), idx.head(np)).colPivHouseholderQr().solve( //A
				indexing_vector(aTb, idx.head(np))-indexing_row(aTa, idx.head(np))*x);         //b
			for (int ip=0; ip<np; ip++) p(idx[ip])=z(ip);
		} else if (np>1) {
			z=q2[np-2]*(	                                                                                                       //Re-project
				(q2[np-2].transpose()*indexing_row_col(aTa, idx.head(np), idx.head(np))*q2[np-2]).colPivHouseholderQr().solve( //A
				q2[np-2].transpose()*(indexing_vector(aTb, idx.head(np))-indexing_row(aTa, idx.head(np))*x) ));                //b
			for (int ip=0; ip<np; ip++) p(idx[ip])=z(ip);
		}
	}
};

}

#endif
