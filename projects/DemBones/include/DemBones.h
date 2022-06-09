///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#ifndef DEM_BONES_DEM_BONES
#define DEM_BONES_DEM_BONES

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <algorithm>
#include <queue>
#include <set>
#include "ConvexLS.h"

#ifndef DEM_BONES_MAT_BLOCKS
#include "MatBlocks.h"
#define DEM_BONES_DEM_BONES_MAT_BLOCKS_UNDEFINED
#endif

namespace Dem
{

/** @mainpage Overview
	Main elements:
	- @ref DemBones : base class with the core solver using relative bone transformations DemBones::m
	- @ref DemBonesExt : extended class to handle hierarchical skeleton with local rotations/translations and bind matrices
	- DemBones/MatBlocks.h: macros to access sub-blocks of packing transformation/position matrices for convenience

	Include DemBones/DemBonesExt.h (or DemBones/DemBones.h) with optional DemBones/MatBlocks.h then follow these steps to use the library:
	-# Load required data in the base class:
		- Rest shapes: DemBones::u, DemBones::fv, DemBones::nV
		- Sequence: DemBones::v, DemBones::nF, DemBones::fStart, DemBones::subjectID, DemBones::nS
		- Number of bones DemBones::nB
	-# Load optional data in the base class:
		- Skinning weights DemBones::w and weights soft-lock DemBones::lockW
		- Bone transformations DemBones::m and bones hard-lock DemBones::lockM
	-# [@c optional] Set parameters in the base class: 
		- DemBones::nIters
		- DemBones::nInitIters
		- DemBones::nTransIters, DemBones::transAffine, DemBones::transAffineNorm
		- DemBones::nWeightsIters, DemBones::nnz, DemBones::weightsSmooth, DemBones::weightsSmoothStep, DemBones::weightEps
	-# [@c optional] Setup extended class:
		- Load data: DemBonesExt::parent, DemBonesExt::preMulInv, DemBonesExt::rotOrder, DemBonesExt::orient, DemBonesExt::bind
		- Set parameter DemBonesExt::bindUpdate
	-# [@c optional] Override callback functions (cb...) in the base class @ref DemBones
	-# Call decomposition function DemBones::compute(), DemBones::computeWeights(), DemBones::computeTranformations(), or DemBones::init()
	-# [@c optional] Get local transformations/bind poses with DemBonesExt::computeRTB() 
*/

/** @class DemBones DemBones.h "DemBones/DemBones.h"
	@brief Smooth skinning decomposition with rigid bones and sparse, convex weights
	
	@details Setup the required data, parameters, and call either compute(), computeWeights(), computeTranformations(), or init().
	
	Callback functions and read-only values can be used to report progress and stop on convergence: cbInitSplitBegin(), cbInitSplitEnd(), 
	cbIterBegin(), cbIterEnd(), cbWeightsBegin(), cbWeightsEnd(), cbTranformationsBegin(), cbTransformationsEnd(), cbTransformationsIterBegin(),
	cbTransformationsIterEnd(), cbWeightsIterBegin(), cbWeightsIterEnd(), rmse(), #iter, #iterTransformations, #iterWeights.

	@b _Scalar is the floating-point data type. @b _AniMeshScalar is the floating-point data type of mesh sequence #v.
*/
template<class _Scalar, class _AniMeshScalar>
class DemBones {
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

	//! [@c parameter] Number of global iterations, @c default = 30
	int nIters;

	//! [@c parameter] Number of clustering update iterations in the initalization, @c default = 10
	int nInitIters;

	//! [@c parameter] Number of bone transformations update iterations per global iteration, @c default = 5
	int nTransIters;
	//! [@c parameter] Translations affinity soft constraint, @c default = 10.0
	_Scalar transAffine;
	//! [@c parameter] p-norm for bone translations affinity soft constraint, @c default = 4.0
	_Scalar transAffineNorm;
	
	//! [@c parameter] Number of weights update iterations per global iteration, @c default = 3
	int nWeightsIters;
	//! [@c parameter] Number of non-zero weights per vertex, @c default = 8
	int nnz;
	//! [@c parameter] Weights smoothness soft constraint, @c default = 1e-4
	_Scalar weightsSmooth;	
	//! [@c parameter] Step size for the weights smoothness soft constraint, @c default = 1.0
	_Scalar weightsSmoothStep;
	//! [@c parameter] Epsilon for weights solver, @c default = 1e-15
	_Scalar weightEps;
	
	/** @brief Constructor and setting default parameters
	*/
	DemBones():	nIters(30), nInitIters(10),
			nTransIters(5),	transAffine(_Scalar(10)), transAffineNorm(_Scalar(4)),
			nWeightsIters(3), nnz(8), weightsSmooth(_Scalar(1e-4)), weightsSmoothStep(_Scalar(1)),
			weightEps(_Scalar(1e-15)),
			iter(_iter), iterTransformations(_iterTransformations), iterWeights(_iterWeights) {
		clear();
	}
	
	//! Number of vertices, typically indexed by @p i
	int nV;
	//! Number of bones, typically indexed by @p j
	int nB;					
	//! Number of subjects, typically indexed by @p s
	int nS;		
	//! Number of total frames, typically indexed by @p k, #nF = #fStart(#nS)
	int nF;	

	//! Start frame indices, @c size = #nS+1, #fStart(@p s), #fStart(@p s+1) are data frames for subject @p s
	Eigen::VectorXi fStart;
	//! Subject index of the frame, @c size = #nF, #subjectID(@p k)=@p s, where #fStart(@p s) <= @p k < #fStart(<tt>s</tt>+1)
	Eigen::VectorXi subjectID;

	//! Geometry at the rest poses, @c size = [3*#nS, #nV], #u.@a col(@p i).@a segment(3*@p s, 3) is the rest pose of vertex @p i of subject @p s
	MatrixX u;

	//! Skinning weights, @c size = [#nB, #nV], #w.@a col(@p i) are the skinning weights of vertex @p i, #w(@p j, @p i) is the influence of bone @p j to vertex @p i
	SparseMatrix w;

	//! Skinning weights lock control, @c size = #nV, #lockW(@p i) is the amount of input skinning weights will be kept for vertex @p i, where 0 (no lock) <= #lockW(@p i) <= 1 (full lock)
	VectorX lockW;

	/** @brief Bone transformations, @c size = [4*#nF*4, 4*#nB], #m.@a blk4(@p k, @p j) is the 4*4 relative transformation matrix of bone @p j at frame @p k
		@details Note that the transformations are relative, that is #m.@a blk4(@p k, @p j) brings the global transformation of bone @p j from the rest pose to the pose at frame @p k.
	*/
	MatrixX m;

	//! Bone transformation lock control, @c size = #nB, #lockM(@p j) is the amount of input transformations will be kept for bone @p j, where #lockM(@p j) = 0 (no lock) or 1 (lock)
	Eigen::VectorXi lockM;

	//! Animated mesh sequence, @c size = [3*#nF, #nV], #v.@a col(@p i).@a segment(3*@p k, 3) is the position of vertex @p i at frame @p k
	Eigen::Matrix<_AniMeshScalar, Eigen::Dynamic, Eigen::Dynamic> v;
	
	//! Mesh topology, @c size=[<tt>number of polygons</tt>], #fv[@p p] is the vector of vertex indices of polygon @p p
	std::vector<std::vector<int>> fv;

	//! [<tt>zero indexed</tt>, <tt>read only</tt>] Current global iteration number that can be used for callback functions
	const int& iter;

	//! [<tt>zero indexed</tt>, <tt>read only</tt>] Current bone transformations update iteration number that can be used for callback functions
	const int& iterTransformations;

	//! [<tt>zero indexed</tt>, <tt>read only</tt>] Current weights update iteration number that can be used for callback functions
	const int& iterWeights;
	
	/** @brief Clear all data
	*/
	void clear() {
		nV=nB=nS=nF=0;
		fStart.resize(0);
		subjectID.resize(0);
		u.resize(0, 0);
		w.resize(0, 0);
		lockW.resize(0);
		m.resize(0, 0);
		lockM.resize(0);
		v.resize(0, 0);
		fv.resize(0);
		modelSize=-1;
		laplacian.resize(0, 0);
	}

	/** @brief Initialize missing skinning weights and/or bone transformations
		@details Depending on the status of #w and #m, this function will:
			- Both #w and #m are already set: do nothing
			- Only one in #w or #m is missing (zero size): initialize missing matrix, i.e. #w (or #m)
			- Both #w and #m are missing (zero size): initialize both with rigid skinning using approximately #nB bones, i.e. values of #w are 0 or 1.
			LBG-VQ clustering is peformed using mesh sequence #v, rest pose geometries #u and topology #fv.
			@b Note: as the initialization does not use exactly #nB bones, the value of #nB could be changed when both #w and #m are missing.
			
		This function is called at the begining of every compute update functions as a safeguard.
	*/
	void init() {
		if (modelSize<0) modelSize=sqrt((u-(u.rowwise().sum()/nV).replicate(1, nV)).squaredNorm()/nV/nS);
		if (laplacian.cols()!=nV) computeSmoothSolver();

		if (((int)w.rows()!=nB)||((int)w.cols()!=nV)) { //No skinning weight
			if (((int)m.rows()!=nF*4)||((int)m.cols()!=nB*4)) { //No transformation
				int targetNB=nB;
				//LBG-VQ
				nB=1;
				label=Eigen::VectorXi::Zero(nV);
				computeTransFromLabel();
				std::cout << "computeTransFromLabel" << std::endl;
				std::cout << label.transpose() << std::endl;

				bool cont=true;
				while (cont) {
					cbInitSplitBegin();
					int prev=nB;
					split(targetNB, 3);
					std::cout << "split" << std::endl;
					std::cout << label.transpose() << std::endl;
					for (int rep=0; rep<nInitIters; rep++) {
						computeTransFromLabel();
						computeLabel();
						std::cout << "computeLabel" << std::endl;
						std::cout << label.transpose() << std::endl;
						pruneBones(3);
					}
					cont=(nB<targetNB)&&(nB>prev);
					cbInitSplitEnd();
				}
				lockM=Eigen::VectorXi::Zero(nB);
				labelToWeights();
			} else initWeights(); //Has transformations
		} else { //Has skinning weights
			if (((int)m.rows()!=nF*4)||((int)m.cols()!=nB*4)) { //No transformation
				m=Matrix4::Identity().replicate(nF, nB);
				lockM=Eigen::VectorXi::Zero(nB);
			}
		}

		if (lockW.size()!=nV) lockW=VectorX::Zero(nV);
		if (lockM.size()!=nB) lockM=Eigen::VectorXi::Zero(nB);
	}

	/** @brief Update bone transformations by running #nTransIters iterations with #transAffine and #transAffineNorm regularizers
		@details Required input data:
			- Rest shapes: #u, #fv, #nV
			- Sequence: #v, #nF, #fStart, #subjectID, #nS
			- Number of bones: #nB

		Optional input data:
			- Skinning weights: #w, #lockW
			- Bone transformations: #m, #lockM

		Output: #m. Missing #w and/or #m (with zero size) will be initialized by init().
	*/
	void computeTranformations() {
		if (nTransIters==0) return;

		init();
		cbTranformationsBegin();

		compute_vuT();
		compute_uuT();

		for (_iterTransformations=0; _iterTransformations<nTransIters; _iterTransformations++) {
			cbTransformationsIterBegin();
			#pragma omp parallel for
			for (int k=0; k<nF; k++)
				for (int j=0; j<nB; j++) 
					if (lockM(j)==0) {
						Matrix4 qpT=vuT.blk4(k, j);
						for (int it=uuT.outerIdx(j); it<uuT.outerIdx(j+1); it++)
							if (uuT.innerIdx(it)!=j) qpT-=m.blk4(k, uuT.innerIdx(it))*uuT.val.blk4(subjectID(k), it);
						qpT2m(qpT, k, j);
					}
			if (cbTransformationsIterEnd()) return;
		}
		
		cbTransformationsEnd();
	}

	/** @brief Update skinning weights by running #nWeightsIters iterations with #weightsSmooth and #weightsSmoothStep regularizers
		@details Required input data:
			- Rest shapes: #u, #fv, #nV
			- Sequence: #v, #nF, #fStart, #subjectID, #nS
			- Number of bones: #nB

		Optional input data:
			- Skinning weights: #w, #lockW
			- Bone transformations: #m, #lockM

		Output: #w. Missing #w and/or #m (with zero size) will be initialized by init().

	*/
	void computeWeights() {
		if (nWeightsIters==0) return;
		
		init();
		cbWeightsBegin();
		
		compute_mTm();

		aTb=MatrixX::Zero(nB, nV);
		wSolver.init(nnz);
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> trip;
		trip.reserve(nV*nnz);

		for (_iterWeights=0; _iterWeights<nWeightsIters; _iterWeights++) {
			cbWeightsIterBegin();

			compute_ws();
			compute_aTb();

			double reg_scale=pow(modelSize, 2)*nF;

			trip.clear();
			#pragma omp parallel for
			for (int i=0; i<nV; i++) {
				MatrixX aTai;
				compute_aTa(i, aTai);
				aTai=(1-lockW(i))*(aTai/reg_scale+weightsSmooth*MatrixX::Identity(nB, nB))+lockW(i)*MatrixX::Identity(nB, nB);
				VectorX aTbi=(1-lockW(i))*(aTb.col(i)/reg_scale+weightsSmooth*ws.col(i))+lockW(i)*w.col(i);

				VectorX x=(1-lockW(i))*ws.col(i)+lockW(i)*w.col(i);
				Eigen::ArrayXi idx=Eigen::ArrayXi::LinSpaced(nB, 0, nB-1);
				std::sort(idx.data(), idx.data()+nB, [&x](int i1, int i2) { return x(i1)>x(i2); });
				int nnzi=std::min(nnz, nB);
				while (x(idx(nnzi-1))<weightEps) nnzi--;

				VectorX x0=w.col(i).toDense().cwiseMax(0.0);
				x=indexing_vector(x0, idx.head(nnzi));
				_Scalar s=x.sum();
				if (s>_Scalar(0.1)) x/=s; else x=VectorX::Constant(nnzi, _Scalar(1)/nnzi);

				wSolver.solve(indexing_row_col(aTai, idx.head(nnzi), idx.head(nnzi)), indexing_vector(aTbi, idx.head(nnzi)), x, true, true);

				#pragma omp critical
				for (int j=0; j<nnzi; j++)
					if (x(j)!=0) trip.push_back(Triplet(idx[j], i, x(j)));
			}

			w.resize(nB, nV);
			w.setFromTriplets(trip.begin(), trip.end());
			
			if (cbWeightsIterEnd()) return;
		}
		
		cbWeightsEnd();
	}

	/** @brief Skinning decomposition by #nIters iterations of alternative updating weights and bone transformations
		@details Required input data:
			- Rest shapes: #u, #fv, #nV
			- Sequence: #v, #nF, #fStart, #subjectID, #nS
			- Number of bones: #nB

		Optional input data:
			- Skinning weights: #w
			- Bone transformations: #m

		Output: #w, #m. Missing #w and/or #m (with zero size) will be initialized by init().
	*/
	void compute() {
		init();

		for (_iter=0; _iter<nIters; _iter++) {
			cbIterBegin();
			computeTranformations();
			computeWeights();
			if (cbIterEnd()) break;
		}
	}

	//! @return Root mean squared reconstruction error
	_Scalar rmse() {
		_Scalar e=0;
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			_Scalar ei=0;
			Matrix4 mki;
			for (int k=0; k<nF; k++) {
				mki.setZero();
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) mki+=it.value()*m.blk4(k, it.row());
				ei+=(mki.template topLeftCorner<3, 3>()*u.vec3(subjectID(k), i)+mki.template topRightCorner<3, 1>()-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
			}
			#pragma omp atomic
			e+=ei;
		}
		return std::sqrt(e/nF/nV);
	}

	//! Callback function invoked before each spliting of bone clusters in initialization
	virtual void cbInitSplitBegin() {}
	//! Callback function invoked after each spliting of bone clusters in initialization
	virtual void cbInitSplitEnd() {}

	//! Callback function invoked before each global iteration update
	virtual void cbIterBegin() {}
	//! Callback function invoked after each global iteration update, stop iteration if return true
	virtual bool cbIterEnd() { return false; }

	//! Callback function invoked before each skinning weights update
	virtual void cbWeightsBegin() {}
	//! Callback function invoked after each skinning weights update
	virtual void cbWeightsEnd() {}

	//! Callback function invoked before each bone transformations update
	virtual void cbTranformationsBegin() {}
	//! Callback function invoked after each bone transformations update
	virtual void cbTransformationsEnd() {}

	//! Callback function invoked before each local bone transformations update iteration
	virtual void cbTransformationsIterBegin() {}
	//! Callback function invoked after each local bone transformations update iteration, stop iteration if return true
	virtual bool cbTransformationsIterEnd() { return false; }

	//! Callback function invoked before each local weights update iteration
	virtual void cbWeightsIterBegin() {}
	//! Callback function invoked after each local weights update iteration, stop iteration if return true
	virtual bool cbWeightsIterEnd() { return false; }

private:
	int _iter, _iterTransformations, _iterWeights;

	/** Best rigid transformation from covariance matrix
		@param _qpT is the 4*4 covariance matrix
		@param k is the frame number
		@param j is the bone index
	*/
	void qpT2m(const Matrix4& _qpT, int k, int j) {
		if (_qpT(3, 3)!=0) {
			Matrix4 qpT=_qpT/_qpT(3, 3);
			Eigen::JacobiSVD<Matrix3> svd(qpT.template topLeftCorner<3, 3>()-qpT.template topRightCorner<3, 1>()*qpT.template bottomLeftCorner<1, 3>(), Eigen::ComputeFullU|Eigen::ComputeFullV);
			Matrix3 d=Matrix3::Identity();
			d(2, 2)=(svd.matrixU()*svd.matrixV().transpose()).determinant();
			m.rotMat(k, j)=svd.matrixU()*d*svd.matrixV().transpose();
			m.transVec(k, j)=qpT.template topRightCorner<3, 1>()-m.rotMat(k, j)*qpT.template bottomLeftCorner<1, 3>().transpose();
		}
	}

	/** Fitting error
		@param i is the vertex index
		@param j is the bone index
	*/
	_Scalar errorVtxBone(int i, int j, bool par=true) {
		_Scalar e=0;
		#pragma omp parallel for if(par)
		for (int k=0; k<nF; k++)
			#pragma omp atomic
			e+=(m.rotMat(k, j)*u.vec3(subjectID(k), i)+m.transVec(k, j)-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
		return e;
	}

	//! label(i) is the index of the bone associated with vertex i
	Eigen::VectorXi label;

	//! Comparator for heap with smallest values on top
	struct TripletLess {
		bool operator() (const Triplet& t1, const Triplet& t2) {
			return t1.value()>t2.value();
		}
	};

	/** Update labels of vertices
	*/
	void computeLabel() {
		VectorX ei(nV);
		Eigen::VectorXi seed=Eigen::VectorXi::Constant(nB, -1);
		VectorX gMin(nB);
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			int j=label(i);
			if (j!=-1) {
				ei(i)=errorVtxBone(i, j, false);
				if ((seed(j)==-1)||(ei(i)<gMin(j))) {
					#pragma omp critical
					if ((seed(j)==-1)||(ei(i)<gMin(j))) {
						gMin(j)=ei(i);
						seed(j)=i;
					}
				}
			}
		}

		std::priority_queue<Triplet, std::vector<Triplet, Eigen::aligned_allocator<Triplet>>, TripletLess> heap;
		for (int j=0; j<nB; j++) if (seed(j)!=-1) heap.push(Triplet(j, seed(j), ei(seed(j))));

		if (laplacian.cols()!=nV) computeSmoothSolver();

		std::vector<bool> dirty(nV, true);
		while (!heap.empty()) {
			Triplet top=heap.top();
			heap.pop();
			int i=(int)top.col();
			int j=(int)top.row();
			if (dirty[i]) {
				label(i)=j;
				ei(i)=top.value();
				dirty[i]=false;
				for (typename SparseMatrix::InnerIterator it(laplacian, i); it; ++it) {
					int i2=(int)it.row();
					if (dirty[i2]) {
						double tmp=(label(i2)==j)?ei(i2):errorVtxBone(i2, j);
						heap.push(Triplet(j, i2, tmp));
					}
				}
			}
		}

		#pragma omp parallel for
		for (int i=0; i<nV; i++) 
			if (label(i)==-1) {
				_Scalar gMin;
				for (int j=0; j<nB; j++) {
					_Scalar ej=errorVtxBone(i, j, false);
					if ((label(i)==-1)||(gMin>ej)) {
						gMin=ej;
						label(i)=j;
					}
				}
			}
	}

	/** Update bone transformation from label
	*/
	void computeTransFromLabel() {
		m=Matrix4::Identity().replicate(nF, nB);
		#pragma omp parallel for
		for (int k=0; k<nF; k++) {
			MatrixX qpT=MatrixX::Zero(4, 4*nB);
			for (int i=0; i<nV; i++) 
				if (label(i)!=-1) 
				qpT.blk4(0, label(i))+=Vector4(v.vec3(k, i).template cast<_Scalar>().homogeneous())*u.vec3(subjectID(k), i).homogeneous().transpose();
			for (int j=0; j<nB; j++) qpT2m(qpT.blk4(0, j), k, j);
		}
	}

	/** Set matrix w from label
	*/
	void labelToWeights() {
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> trip(nV);
		for (int i=0; i<nV; i++) trip[i]=Triplet(label(i), i, _Scalar(1));
		w.resize(nB, nV);
		w.setFromTriplets(trip.begin(), trip.end());
		lockW=VectorX::Zero(nV);
	}

	/** Split bone clusters
		@param maxB is the maximum number of bones
		@param threshold*2 is the minimum size of the bone cluster to be splited 
	*/
	void split(int maxB, int threshold) {
		//Centroids
		MatrixX cu=MatrixX::Zero(3*nS, nB);
		Eigen::VectorXi s=Eigen::VectorXi::Zero(nB);
		for (int i=0; i<nV; i++) {
			cu.col(label(i))+=u.col(i);
			s(label(i))++;
		}
		for (int j=0; j<nB; j++) if (s(j)!=0) cu.col(j)/=_Scalar(s(j));

		//Distance to centroid & error
		VectorX d(nV), e(nV);
		VectorX minD=VectorX::Constant(nB, std::numeric_limits<_Scalar>::max());
		VectorX minE=VectorX::Constant(nB, std::numeric_limits<_Scalar>::max());
		VectorX ce=VectorX::Zero(nB);

		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			int j=label(i);
			d(i)=(u.col(i)-cu.col(j)).norm();
			e(i)=sqrt(errorVtxBone(i, j, false));
			if (d(i)<minD(j)) {
				#pragma omp critical
				minD(j)=std::min(minD(j), d(i));
			}
			if (e(i)<minE(j)) {
				#pragma omp critical
				minE(j)=std::min(minE(j), e(i));
			}
			#pragma omp atomic
			ce(j)+=e(i);
		}

		//Seed
		Eigen::VectorXi seed=Eigen::VectorXi::Constant(nB, -1);
		VectorX gMax(nB);

		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			int j=label(i);
			double tmp=abs((e(i)-minE(j))*(d(i)-minD(j)));

			if ((seed(j)==-1)||(tmp>gMax(j))) {
				#pragma omp critical
				if ((seed(j)==-1)||(tmp>gMax(j))) {
					gMax(j)=tmp;
					seed(j)=i;
				}
			}
		}

		int countID=nB;
		_Scalar avgErr=ce.sum()/nB;
		for (int j=0; j<nB; j++)
			if ((countID<maxB)&&(s(j)>threshold*2)&&(ce(j)>avgErr/100)) {
				int newLabel=countID++;
				int i=seed(j);
				for (typename SparseMatrix::InnerIterator it(laplacian, i); it; ++it) label(it.row())=newLabel;
			}
		nB=countID;
	}

	/** Remove bones with small number of associated vertices
		@param threshold is the minimum number of vertices assigned to a bone
	*/
	void pruneBones(int threshold) {
		Eigen::VectorXi s=Eigen::VectorXi::Zero(nB);
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			#pragma omp atomic
			s(label(i))++;
		}

		Eigen::VectorXi newID(nB);
		int countID=0;
		for (int j=0; j<nB; j++)
			if (s(j)<threshold) newID(j)=-1; else newID(j)=countID++;

		if (countID==nB) return;

		for (int j=0; j<nB; j++)
			if (newID(j)!=-1) m.template middleCols<4>(newID(j)*4)=m.template middleCols<4>(j*4);

		#pragma omp parallel for
		for (int i=0; i<nV; i++) label(i)=newID(label(i));

		nB=countID;
		m.conservativeResize(nF*4, nB*4);
		computeLabel();
	}

	/** Initialize skinning weights with rigid bind to the best bone
	*/
	void initWeights() {
		label=Eigen::VectorXi::Constant(nV, -1);
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			_Scalar gMin;
			for (int j=0; j<nB; j++) {
				_Scalar ej=errorVtxBone(i, j, false);
				if ((label(i)==-1)||(gMin>ej)) {
					gMin=ej;
					label(i)=j;
				}
			}
		}
		computeLabel();
		labelToWeights();
	}

	//! vuT.blk4(k, j) = \sum_{i=0}^{nV-1}  w(j, i)*v.vec3(k, i).homogeneous()*u.vec3(subjectID(k), i).homogeneous()^T
	MatrixX vuT;

	/** Pre-compute vuT with bone translations affinity soft constraint
	*/
	void compute_vuT() {
		vuT=MatrixX::Zero(nF*4, nB*4);
		#pragma omp parallel for
		for (int k=0; k<nF; k++) {
			MatrixX vuTp=MatrixX::Zero(4, nB*4);
			for (int i=0; i<nV; i++)
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) {
					Matrix4 tmp=Vector4(v.vec3(k, i).template cast<_Scalar>().homogeneous())*u.vec3(subjectID(k), i).homogeneous().transpose();
					vuT.blk4(k, it.row())+=it.value()*tmp;
					vuTp.blk4(0, it.row())+=pow(it.value(), transAffineNorm)*tmp;
				}
			for (int j=0; j<nB; j++)
				if (vuTp(3, j*4+3)!=0)
					vuT.blk4(k, j)+=(transAffine*vuT(k*4+3, j*4+3)/vuTp(3, j*4+3))*vuTp.blk4(0, j);
		}
	}
	
	//! uuT is a sparse block matrix, uuT(j, k).block<4, 4>(s*4, 0) = \sum{i=0}{nV-1} w(j, i)*w(k, i)*u.col(i).segment<3>(s*3).homogeneous().transpose()*u.col(i).segment<3>(s*3).homogeneous()
	struct SparseMatrixBlock {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		MatrixX val;
		Eigen::VectorXi innerIdx, outerIdx;
	} uuT;

	/** Pre-compute uuT for bone transformations update
	*/
	void compute_uuT() {
		Eigen::MatrixXi pos=Eigen::MatrixXi::Constant(nB, nB, -1);
		#pragma omp parallel for
		for (int i=0; i<nV; i++)
			for (typename SparseMatrix::InnerIterator it(w, i); it; ++it)
				for (typename SparseMatrix::InnerIterator jt(w, i); jt; ++jt)
					pos(it.row(), jt.row())=1;

		uuT.outerIdx.resize(nB+1);
		uuT.innerIdx.resize(nB*nB);
		int nnz=0;
		for (int j=0; j<nB; j++) {
			uuT.outerIdx(j)=nnz;
			for (int i=0; i<nB; i++)
				if (pos(i, j)!=-1) {
					uuT.innerIdx(nnz)=i;
					pos(i, j)=nnz++;
				}
		}
		uuT.outerIdx(nB)=nnz;
		uuT.innerIdx.conservativeResize(nnz);
		uuT.val=MatrixX::Zero(nS*4, nnz*4);

		#pragma omp parallel for
		for (int i=0; i<nV; i++)
			for (typename SparseMatrix::InnerIterator it(w, i); it; ++it)
				for (typename SparseMatrix::InnerIterator jt(w, i); jt; ++jt)
					if (it.row()>=jt.row()) {
						double _w=it.value()*jt.value();
						MatrixX _uuT(4*nS, 4);
						Vector4 _u;
						for (int s=0; s<nS; s++) {
							_u=u.vec3(s, i).homogeneous();
							_uuT.blk4(s, 0)=_w*_u*_u.transpose();
						}
						int p=pos(it.row(), jt.row())*4;
						for (int c=0; c<4; c++)
							for (int r=0; r<4*nS; r++)
								#pragma omp atomic
								uuT.val(r, p+c)+=_uuT(r, c);
					}

		for (int i=0; i<nB; i++)
			for (int j=i+1; j<nB; j++)
				if (pos(i, j)!=-1)
					uuT.val.middleCols(pos(i, j)*4, 4)=uuT.val.middleCols(pos(j, i)*4, 4);
	}



	//! mTm.size = (4*nS*nB, 4*nB), where mTm.block<4, 4>(s*nB+i, j) = \sum_{k=fStart(s)}^{fStart(s+1)-1} m.block<3, 4>(k*4, i*4)^T*m.block<3, 4>(k*4, j*4)
	MatrixX mTm;

	/** Pre-compute mTm for weights update
	*/
	void compute_mTm() {
		Eigen::MatrixXi idx(2, nB*(nB+1)/2);
		int nPairs=0;
		for (int i=0; i<nB; i++)
			for (int j=i; j<nB; j++) {
				idx(0, nPairs)=i;
				idx(1, nPairs)=j;
				nPairs++;
			}

		mTm=MatrixX::Zero(nS*nB*4, nB*4);
		#pragma omp parallel for
		for (int p=0; p<nPairs; p++) {
			int i=idx(0, p);
			int j=idx(1, p);
			for (int k=0; k<nF; k++)
				mTm.blk4(subjectID(k)*nB+i, j)+=m.blk4(k, i).template topRows<3>().transpose()*m.blk4(k, j).template topRows<3>();
			if (i!=j) for (int s=0; s<nS; s++) mTm.blk4(s*nB+j, i)=mTm.blk4(s*nB+i, j);
		}
	}

	//! aTb.col(i) is the A^Tb for vertex i, where A.size = (3*nF, nB), A.col(j).segment<3>(f*3) is the transformed position of vertex i by bone j at frame f, b = v.col(i).
	MatrixX aTb;

	/** Pre-compute aTb for weights update
	*/
	void compute_aTb() {
		#pragma omp parallel for
		for (int i=0; i<nV; i++)
			for (int j=0; j<nB; j++)
				if ((aTb(j, i)==0)&&(ws(j, i)>weightEps))
					for (int k=0; k<nF; k++)
						aTb(j, i)+=v.vec3(k, i).template cast<_Scalar>().dot(m.blk4(k, j).template topRows<3>()*u.vec3(subjectID(k), i).homogeneous());
	}

	//! Size of the model=RMS distance to centroid
	_Scalar modelSize;
	
	//! Laplacian matrix
	SparseMatrix laplacian;

	//! LU factorization of Laplacian
	Eigen::SparseLU<SparseMatrix> smoothSolver;

	/** Pre-compute Laplacian and LU factorization
	*/
	void computeSmoothSolver() {
		int nFV=(int)fv.size();

		_Scalar epsDis=0;
		for (int f=0; f<nFV; f++) {
			int nf=(int)fv[f].size();
			for (int g=0; g<nf; g++) {
				int i=fv[f][g];
				int j=fv[f][(g+1)%nf];
				epsDis+=(u.col(i)-u.col(j)).norm();
			}
		}
		epsDis=epsDis*weightEps/(_Scalar)nS;

		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> triplet;
		VectorX d=VectorX::Zero(nV);
		std::vector<std::set<int>> isComputed(nV);

		#pragma omp parallel for
		for (int f=0; f<nFV; f++) {
			int nf=(int)fv[f].size();
			for (int g=0; g<nf; g++) {
				int i=fv[f][g];
				int j=fv[f][(g+1)%nf];

				bool needCompute=false;
				#pragma omp critical 
				if (isComputed[i].find(j)==isComputed[i].end()) {
					needCompute=true;
					isComputed[i].insert(j);
					isComputed[j].insert(i);
				}

				if (needCompute) {
					double val=0;
					for (int s=0; s<nS; s++) {
						double du=(u.vec3(s, i)-u.vec3(s, j)).norm();
						for (int k=fStart(s); k<fStart(s+1); k++)
							val+=pow((v.vec3(k, i).template cast<_Scalar>()-v.vec3(k, j).template cast<_Scalar>()).norm()-du, 2);
					}
					val=1/(sqrt(val/nF)+epsDis);

					#pragma omp critical
					triplet.push_back(Triplet(i, j, -val));
					#pragma omp atomic
					d(i)+=val;

					#pragma omp critical
					triplet.push_back(Triplet(j, i, -val));
					#pragma omp atomic
					d(j)+=val;
				}
			}
		}

		for (int i=0; i<nV; i++)
			triplet.push_back(Triplet(i, i, d(i)));

		laplacian.resize(nV, nV);
		laplacian.setFromTriplets(triplet.begin(), triplet.end());

		for (int i=0; i<nV; i++)
			if (d(i)!=0) laplacian.row(i)/=d(i);

		laplacian=weightsSmoothStep*laplacian+SparseMatrix((VectorX::Ones(nV)).asDiagonal());
		smoothSolver.compute(laplacian);
	}

	//! Smoothed skinning weights
	MatrixX ws;

	/** Implicit skinning weights Laplacian smoothing
	*/
	void compute_ws() {
		ws=w.transpose();
		#pragma omp parallel for
		for (int j=0; j<nB; j++) ws.col(j)=smoothSolver.solve(ws.col(j));
		ws.transposeInPlace();

		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			ws.col(i)=ws.col(i).cwiseMax(0.0);
			_Scalar si=ws.col(i).sum();
			if (si<_Scalar(0.1)) ws.col(i)=VectorX::Constant(nB, _Scalar(1)/nB); else ws.col(i)/=si;
		}
	}

	//! Per-vertex weights solver
	ConvexLS<_Scalar> wSolver;

	/** Pre-compute aTa for weights update on one vertex
		@param i is the vertex index.
		@param aTa is the by-reference output of A^TA for vertex i, where A.size = (3*nF, nB), A.col(j).segment<3>(f*3) is the transformed position of vertex i by bone j at frame f.
	*/
	void compute_aTa(int i, MatrixX& aTa) {
		aTa=MatrixX::Zero(nB, nB);
		for (int j1=0; j1<nB; j1++)
			for (int j2=j1; j2<nB; j2++) {
				for (int s=0; s<nS; s++) aTa(j1, j2)+=u.vec3(s, i).homogeneous().dot(mTm.blk4(s*nB+j1, j2)*u.vec3(s, i).homogeneous());
				if (j1!=j2) aTa(j2, j1)=aTa(j1, j2);
			}
	}
};
	
}

#ifdef DEM_BONES_DEM_BONES_MAT_BLOCKS_UNDEFINED
#undef blk4
#undef rotMat
#undef transVec
#undef vec3
#undef DEM_BONES_MAT_BLOCKS
#endif

#endif
