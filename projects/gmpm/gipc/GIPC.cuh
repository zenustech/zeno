#pragma once
#ifndef _GIPC_H_
#define _GIPC_H_
#include "mlbvh.cuh"
#include "zensim/container/Bvh.hpp"
#include "device_fem_data.cuh"

#include "PCG_SOLVER.cuh"

using lbvh_t = zs::LBvh<3, 32, int, zs::f64>;
using bv_t = typename lbvh_t::Box;

class GIPC {
public:
	double3* _vertexes;
	double3* _rest_vertexes;
	uint3* _faces;
	uint2* _edges;
	uint32_t* _surfVerts;

	// zs 
	void retrieveSurfaces();
	void retrievePositions();
	void retrieveDirections();
	tiles_t surfaces, surfEdges, surfVerts;
	tiles_t *p_verts, *p_eles;
	dtiles_t *p_vtemp, *p_etemp;
	lbvh_t triBvh, edgeBvh;

	double3* _moveDir;
	lbvh_f bvh_f;
	lbvh_e bvh_e;

	PCG_Data pcg_data;

	int4* _collisonPairs;
	int4* _ccd_collisonPairs;
	uint32_t* _cpNum;
	int* _MatIndex;
	uint32_t* _close_cpNum;

	uint32_t* _environment_collisionPair;

	uint32_t* _closeConstraintID;
	double* _closeConstraintVal;

	int4* _closeMConstraintID;
	double* _closeMConstraintVal;

	uint32_t* _gpNum;
	uint32_t* _close_gpNum;
	//uint32_t* _cpNum;
	uint32_t h_cpNum[5];
	uint32_t h_ccd_cpNum;
	uint32_t h_gpNum;

	uint32_t h_close_cpNum;
	uint32_t h_close_gpNum;

	double Kappa;
	double dHat;
	double bboxDiagSize2;
	double dTol;
	double IPC_dt;
	double Step;
	double meanMass;
	double meanVolumn;
	double3* _groundNormal;
	double* _groundOffset;

	// for friction
	double* lambda_lastH_scalar;
	double2* distCoord;
	__GEIGEN__::Matrix3x2d* tanBasis;
	int4* _collisonPairs_lastH;
	uint32_t h_cpNum_last[5];
	int* _MatIndex_last;

	double* lambda_lastH_scalar_gd;
	uint32_t* _collisonPairs_lastH_gd;
	uint32_t h_gpNum_last;

	uint32_t vertexNum;
	uint32_t surf_vertexNum;
	uint32_t edge_Num;
	uint32_t surface_Num;
	uint32_t tetrahedraNum;

	BHessian BH;
	AABB SceneSize;
	int MAX_COLLITION_PAIRS_NUM;
	int MAX_CCD_COLLITION_PAIRS_NUM;

	double RestNHEnergy;
public:
	GIPC() {}
	~GIPC();
	void MALLOC_DEVICE_MEM();

	void tempMalloc_closeConstraint();
	void tempFree_closeConstraint();

	void FREE_DEVICE_MEM();
	void initBVH();
	void init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer);

	void buildCP();
	void buildFullCP(const double& alpha);
	void buildBVH();

	AABB* calcuMaxSceneSize();

	void buildBVH_FULLCCD(const double& alpha);


	void GroundCollisionDetect();
	void calBarrierHessian();
	void calBarrierGradient(double3* _gradient);
	// void calFrictionHessian(device_TetraData& TetMesh);
	// void calFrictionGradient(double3* _gradient, device_TetraData& TetMesh);

	void calculateMovingDirection(device_TetraData& TetMesh);
	void computeGradientAndHessian(device_TetraData& TetMesh);
	void computeGroundGradientAndHessian(double3* _gradient);
	void computeGroundGradient(double3* _gradient);

	double computeEnergy(device_TetraData& TetMesh);

	double Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh);

	double ground_largestFeasibleStepSize(double slackness, double* mqueue);

	double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);

	double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);

	double cfl_largestSpeed(double* mqueue);

	bool lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha);
	void postLineSearch(device_TetraData& TetMesh, double alpha);

	bool checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh);
	bool isIntersected(device_TetraData& TetMesh);
	bool checkGroundIntersection();

	void computeCloseGroundVal();
	void computeSelfCloseVal();

	bool checkCloseGroundVal();
	bool checkSelfCloseVal();

	double2 minMaxGroundDist();
	double2 minMaxSelfDist();

	void updateVelocities(device_TetraData& TetMesh);
	void updateBoundary(device_TetraData& TetMesh, double alpha);
	void updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha);
	void computeXTilta(device_TetraData& TetMesh);

	void initKappa(device_TetraData& TetMesh);
	void suggestKappa(double& kappa);
	void upperBoundKappa(double& kappa);
	int solve_subIP(device_TetraData& TetMesh);
	void IPC_Solver(device_TetraData& TetMesh);
	void sortMesh(device_TetraData& TetMesh);
	// void buildFrictionSets();
};

#endif