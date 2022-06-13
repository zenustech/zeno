///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#ifndef DEM_BONES_EXT
#define DEM_BONES_EXT

#include "DemBones.h"

#include <Eigen/Geometry> 

#ifndef DEM_BONES_MAT_BLOCKS
#include "MatBlocks.h"
#define DEM_BONES_DEM_BONES_EXT_MAT_BLOCKS_UNDEFINED
#endif

namespace Dem
{

/**  @class DemBonesExt DemBonesExt.h "DemBones/DemBonesExt.h"
	@brief Extended class to handle hierarchical skeleton with local rotations/translations and bind matrices

	@details Call computeRTB() to get local rotations/translations and bind matrices after skinning decomposition is done and other data is set.

	@b _Scalar is the floating-point data type. @b _AniMeshScalar is the floating-point data type of mesh sequence #v.
*/
template<class _Scalar, class _AniMeshScalar>
class DemBonesExt: public DemBones<_Scalar, _AniMeshScalar> {
public:	
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	using MatrixX=Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Matrix4=Eigen::Matrix<_Scalar, 4, 4>;
	using Matrix3=Eigen::Matrix<_Scalar, 3, 3>;
	using VectorX=Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;
	using Vector4=Eigen::Matrix<_Scalar, 4, 1>;
	using Vector3=Eigen::Matrix<_Scalar, 3, 1>;
	using SparseMatrix=Eigen::SparseMatrix<_Scalar>;
	using Triplet=Eigen::Triplet<_Scalar>;

	using DemBones<_Scalar, _AniMeshScalar>::nIters;
	using DemBones<_Scalar, _AniMeshScalar>::nInitIters;
	using DemBones<_Scalar, _AniMeshScalar>::nTransIters;
	using DemBones<_Scalar, _AniMeshScalar>::transAffine;
	using DemBones<_Scalar, _AniMeshScalar>::transAffineNorm;
	using DemBones<_Scalar, _AniMeshScalar>::nWeightsIters;
	using DemBones<_Scalar, _AniMeshScalar>::nnz;
	using DemBones<_Scalar, _AniMeshScalar>::weightsSmooth;	
	using DemBones<_Scalar, _AniMeshScalar>::weightsSmoothStep;
	using DemBones<_Scalar, _AniMeshScalar>::weightEps;

	using DemBones<_Scalar, _AniMeshScalar>::nV;
	using DemBones<_Scalar, _AniMeshScalar>::nB;
	using DemBones<_Scalar, _AniMeshScalar>::nS;
	using DemBones<_Scalar, _AniMeshScalar>::nF;
	using DemBones<_Scalar, _AniMeshScalar>::fStart;
	using DemBones<_Scalar, _AniMeshScalar>::subjectID;
	using DemBones<_Scalar, _AniMeshScalar>::u;
	using DemBones<_Scalar, _AniMeshScalar>::w;
	using DemBones<_Scalar, _AniMeshScalar>::lockW;
	using DemBones<_Scalar, _AniMeshScalar>::m;
	using DemBones<_Scalar, _AniMeshScalar>::lockM;
	using DemBones<_Scalar, _AniMeshScalar>::v;
	using DemBones<_Scalar, _AniMeshScalar>::fv;

	using DemBones<_Scalar, _AniMeshScalar>::iter;
	using DemBones<_Scalar, _AniMeshScalar>::iterTransformations;
	using DemBones<_Scalar, _AniMeshScalar>:: iterWeights;

	//! Timestamps for bone transformations #m, [@c size] = #nS, #fTime(@p k) is the timestamp of frame @p k
	Eigen::VectorXd fTime;

	//! Name of bones, [@c size] = #nB, #boneName(@p j) is the name bone of @p j
	std::vector<std::string> boneName;

	//! Parent bone index, [@c size] = #nB, #parent(@p j) is the index of parent bone of @p j, #parent(@p j) = -1 if @p j has no parent.
	Eigen::VectorXi parent;

	//! Original bind pre-matrix, [@c size] = [4*#nS, 4*#nB], #bind.@a block(4*@p s, 4*@p j, 4, 4) is the global bind matrix of bone @p j on subject @p s at the rest pose
	MatrixX bind;

	//! Inverse pre-multiplication matrices, [@c size] = [4*#nS, 4*#nB], #preMulInv.@a block(4*@p s, 4*@p j, 4, 4) is the inverse of pre-local transformation of bone @p j on subject @p s 
	MatrixX preMulInv;

	//! Rotation order, [@c size] = [3*#nS, #nB], #rotOrder.@a col(@p j).@a segment<3>(3*@p s) is the rotation order of bone @p j on subject @p s, 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1, 2} is @c XYZ order  
	Eigen::MatrixXi rotOrder;

	//! Orientations of bones,  [@c size] = [3*#nS, #nB], @p orient.@a col(@p j).@a segment<3>(3*@p s) is the(@c rx, @c ry, @c rz) orientation of bone @p j in degree
	MatrixX orient;

	//! Bind transformation update, 0=keep original, 1=set translations to p-norm centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and group joints
	int bindUpdate;

	/** @brief Constructor and setting default parameters
	*/
	DemBonesExt(): bindUpdate(0) {
		clear();
	}

	/** @brief Clear all data
	*/
	void clear() {
		fTime.resize(0);
		boneName.resize(0);
		parent.resize(0);
		bind.resize(0, 0);
		preMulInv.resize(0, 0);
		rotOrder.resize(0, 0);
		orient.resize(0, 0);
		DemBones<_Scalar, _AniMeshScalar>::clear();
	}

	/** @brief Local rotations, translations and global bind matrices of a subject
		@details Required all data in the base class: #u, #fv, #nV, #v, #nF, #fStart, #subjectID, #nS, #m, #w, #nB

		This function will initialize missing attributes:
		- #parent: -1 vector (if no joint grouping) or parent to a root, [@c size] = #nB
		- #preMulInv: 4*4 identity matrix blocks, [@c size] = [4*#nS, 4*#nB]
		- #rotOrder: {0, 1, 2} vector blocks, [@c size] = [3*#nS, #nB]
		- #orient: 0 matrix, [@c size] = [3*#nS, #nB]

		@param[in] s is the subject index
		@param[out] lr is the [3*@p nFr, #nB] by-reference output local rotations, @p lr.@a col(@p j).segment<3>(3*@p k) is the (@c rx, @c ry, @c rz) of bone @p j at frame @p k
		@param[out] lt is the [3*@p nFr, #nB] by-reference output local translations, @p lt.@a col(@p j).segment<3>(3*@p k) is the (@c tx, @c ty, @c tz) of bone @p j at frame @p k
		@param[out] gb is the [4, 4*#nB] by-reference output global bind matrices, @p gb.@a block(0, 4*@p j, 4, 4) is the bind matrix of bone j
		@param[out] lbr is the [3, #nB] by-reference output local rotations at bind pose @p lbr.@a col(@p j).segment<3>(3*@p k) is the (@c rx, @c ry, @c rz) of bone @p j
		@param[out] lbt is the [3, #nB] by-reference output local translations at bind pose, @p lbt.@a col(@p j).segment<3>(3*@p k) is the (@c tx, @c ty, @c tz) of bone @p j
		@param[in] degreeRot=true will output rotations in degree, otherwise output in radian
	*/
	void computeRTB(int s, MatrixX& lr, MatrixX& lt, MatrixX& gb, MatrixX& lbr, MatrixX& lbt, bool degreeRot=true) {
		computeBind(s, gb);

		if (parent.size()==0) {
			if (bindUpdate==2) {
				int root=computeRoot();
				parent=Eigen::VectorXi::Constant(nB, root);
				parent(root)=-1;
			} else parent=Eigen::VectorXi::Constant(nB, -1);
		}
		if (preMulInv.size()==0) preMulInv=MatrixX::Identity(4, 4).replicate(nS, nB);
		if (rotOrder.size()==0) rotOrder=Eigen::Vector3i(0, 1, 2).replicate(nS, nB);
		if (orient.size()==0) orient=MatrixX::Zero(3*nS, nB);

		int nFs=fStart(s+1)-fStart(s);
		lr.resize(nFs*3, nB);
		lt.resize(nFs*3, nB);
		lbr.resize(3, nB);
		lbt.resize(3, nB);

		MatrixX lm(4*nFs, 4*nB);
		#pragma omp parallel for
		for (int j=0; j<nB; j++) {
			Eigen::Vector3i ro=rotOrder.col(j).template segment<3>(s*3);

			Vector3 ov=orient.vec3(s, j)*EIGEN_PI/180;
			Matrix3 invOM=Matrix3(Eigen::AngleAxis<_Scalar>(ov(ro(2)), Vector3::Unit(ro(2))))*
				Eigen::AngleAxis<_Scalar>(ov(ro(1)), Vector3::Unit(ro(1)))*
				Eigen::AngleAxis<_Scalar>(ov(ro(0)), Vector3::Unit(ro(0)));
			invOM.transposeInPlace();

			Matrix4 lb;
			if (parent(j)==-1) lb=preMulInv.blk4(s, j)*gb.blk4(0, j);
			else lb=preMulInv.blk4(s, j)*gb.blk4(0, parent(j)).inverse()*gb.blk4(0, j);

			Vector3 curRot=Vector3::Zero();
			toRot(invOM*lb.template topLeftCorner<3, 3>(), curRot, ro);
			lbr.col(j)=curRot;
			lbt.col(j)=lb.template topRightCorner<3, 1>();

			Matrix4 lm;
			for (int k=0; k<nFs; k++) {
				if (parent(j)==-1) lm=preMulInv.blk4(s, j)*m.blk4(k+fStart(s), j)*gb.blk4(0, j);
				else lm=preMulInv.blk4(s, j)*(m.blk4(k+fStart(s), parent(j))*gb.blk4(0, parent(j))).inverse()*m.blk4(k+fStart(s), j)*gb.blk4(0, j);
				toRot(invOM*lm.template topLeftCorner<3, 3>(), curRot, ro);
				lr.vec3(k, j)=curRot;
				lt.vec3(k, j)=lm.template topRightCorner<3, 1>();
			}
		}

		if (degreeRot) {
			lr*=180/EIGEN_PI;
			lbr*=180/EIGEN_PI;
		}
	}
	
private:
	/** p-norm centroids (using #transAffineNorm) and rotations to identity
		@param s is the subject index
		@param b is the [4, 4*#nB] by-reference output global bind matrices, #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
	*/
	void computeCentroids(int s, MatrixX& b) {
		MatrixX c=MatrixX::Zero(4, nB);
		for (int i=0; i<nV; i++)
			for (typename SparseMatrix::InnerIterator it(w, i); it; ++it)
				c.col(it.row())+=pow(it.value(), transAffineNorm)*u.vec3(s, i).homogeneous();
		for (int j=0; j<nB; j++)
			if ((c(3, j)!=0)&&(lockM(j)==0)) b.transVec(0, j)=c.col(j).template head<3>()/c(3, j);
	}

	/** Global bind pose
		@param s is the subject index
		@param bindUpdate is the type of bind pose update, 0=keep original, 1 or 2=set translations to p-norm centroids (using #transAffineNorm) and rotations to identity
		@param b is the the [4, 4*#nB] by-reference output global bind matrices, #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
	*/
	void computeBind(int s, MatrixX& b) {
		if (bind.size()==0) {
			lockM=Eigen::VectorXi::Zero(nB);
			bind.resize(nS*4, nB*4);
			for (int k=0; k<nS; k++) {
				b=MatrixX::Identity(4, 4).replicate(1, nB);
				computeCentroids(k, b);
				bind.block(4*k, 0, 4, 4*nB)=b;
			}
		}
		
		b=bind.block(4*s, 0, 4, 4*nB);
		if (bindUpdate>=1) computeCentroids(s, b);
	}

	/** Root joint
	*/
	int computeRoot() {
		VectorX err(nB);
		#pragma omp parallel for
		for (int j=0; j<nB; j++) {
			double ej=0;
			for (int i=0; i<nV; i++)
				for (int k=0; k<nF; k++) ej+=(m.rotMat(k, j)*u.vec3(subjectID(k), i)+m.transVec(k, j)-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
			err(j)=ej;
		}
		int rj;
		err.minCoeff(&rj);
		return rj;
	}

	/** Euler angles from rotation matrix
		@param rMat is the 3*3 rotation matrix
		@param curRot is the input current Euler angles, it is also the by-reference output closet Euler angles correspond to @p rMat
		@param ro is the rotation order, 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1, 2} is @c XYZ order  
		@param eps is the epsilon
	*/
	void toRot(const Matrix3& rMat, Vector3& curRot, const Eigen::Vector3i& ro, _Scalar eps=_Scalar(1e-10)) {
		Vector3 r0=rMat.eulerAngles(ro(2), ro(1), ro(0)).reverse();
		_Scalar gMin=(r0-curRot).squaredNorm();
		Vector3 rMin=r0;
		Vector3 r;
		Matrix3 tmpMat;
		for (int fx=-1; fx<=1; fx+=2)
			for (_Scalar sx=-2*EIGEN_PI; sx<2.1*EIGEN_PI; sx+=EIGEN_PI) {
				r(0)=fx*r0(0)+sx;
				for (int fy=-1; fy<=1; fy+=2)
					for (_Scalar sy=-2*EIGEN_PI; sy<2.1*EIGEN_PI; sy+=EIGEN_PI) {
						r(1)=fy*r0(1)+sy;
						for (int fz=-1; fz<=1; fz+=2)
							for (_Scalar sz=-2*EIGEN_PI; sz<2.1*EIGEN_PI; sz+=EIGEN_PI) {
								r(2)=fz*r0(2)+sz;
								tmpMat=Matrix3(Eigen::AngleAxis<_Scalar>(r(ro(2)), Vector3::Unit(ro(2))))*
									Eigen::AngleAxis<_Scalar>(r(ro(1)), Vector3::Unit(ro(1)))*
									Eigen::AngleAxis<_Scalar>(r(ro(0)), Vector3::Unit(ro(0)));
								if ((tmpMat-rMat).squaredNorm()<eps) {
									_Scalar tmp=(r-curRot).squaredNorm();
									if (tmp<gMin) {
										gMin=tmp;
										rMin=r;
									}
								}
							}
					}
			}
		curRot=rMin;
	}
};

}

#ifdef DEM_BONES_DEM_BONES_EXT_MAT_BLOCKS_UNDEFINED
#undef blk4
#undef rotMat
#undef transVec
#undef vec3
#undef DEM_BONES_MAT_BLOCKS
#endif

#undef rotMatFromEuler

#endif
