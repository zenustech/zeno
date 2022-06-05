#pragma once
#ifndef _EIGEN_DATA_H_
#define _EIGEN_DATA_H_
namespace __GEIGEN__ {

	struct Vector6 {
		double v[6];
	};

	struct Vector9 {
		double v[9];
	};

	struct Vector12 {
		double v[12];
	};

	struct Matrix3x3d {
		double m[3][3];
	};

	struct Matrix3x6d {
		double m[3][6];
	};

	struct Matrix6x3d {
		double m[6][3];
	};

	struct Matrix2x2d {
		double m[2][2];
	};

	struct Matrix3x2d {
		double m[3][2];
	};

	struct Matrix9x9d {
		double m[9][9];
	};

	struct Matrix12x12d {
		double m[12][12];
	};

	struct Matrix9x2d {
		double m[9][2];
	};

	struct Matrix6x2d {
		double m[6][2];
	};

	struct Matrix12x2d {
		double m[12][2];
	};

	struct Matrix9x12d {
		double m[9][12];
	};

	//struct Matrix9x12d_v {
	//	double m[108];
	//};

	struct Matrix12x9d {
		double m[12][9];
	};

	struct Matrix12x6d {
		double m[12][6];
	};

	struct Matrix6x12d {
		double m[6][12];
	};

	struct Matrix6x6d {
		double m[6][6];
	};

	struct Matrix12x4d {
		double m[12][4];
	};

	struct Matrix9x4d {
		double m[9][4];
	};

	struct Matrix6x9d {
		double m[6][9];
	};

	struct Matrix9x6d {
		double m[9][6];
	};

	struct Matrix2x3d {
		double m[2][3];
	};
}

#endif // !_EIGEN_DATA_H_

