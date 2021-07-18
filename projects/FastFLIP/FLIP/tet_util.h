#ifndef TET_UTIL_H
#define TET_UTIL_H

#include "vec.h"
#include "util.h"
struct matrix3x3d
{
	vector<double> v;
	matrix3x3d(){ v.resize(9);}
	matrix3x3d(double a, double b, double c,
			   double d, double e, double f,
			   double g, double h, double i)
	{
		v.resize(9);
		v[0]=a;v[1]=b;v[2]=c;
		v[3]=d;v[4]=e;v[5]=f;
		v[6]=g;v[7]=h;v[8]=i;
	}
	matrix3x3d(const matrix3x3d & m){v = m.v;}
	~matrix3x3d(){ v.resize(0); };
	const double& operator()(int i,int j) const
	{
		return v[i*3+j];
	}
	double& operator()(int i,int j)
	{
		return v[i*3+j];
	}
	const double& operator[](int i) const{
		return v[i];
	}
	double& operator[](int i)
	{
		return v[i];
	}
	matrix3x3d inverse()
	{
		matrix3x3d a;
		double a0 = (v[4]*v[8]-v[7]*v[5]);
		double a1 = (v[5]*v[6]-v[3]*v[8]);
		double a2 = (v[3]*v[7]-v[4]*v[6]);
		double det = v[0] * a0 + v[1]*a1 + v[2]*a2;
		double invdet = 1.0/det;
		a[0] = a0*invdet;
		a[1] = (v[2]*v[7]-v[1]*v[8])*invdet;
		a[2] = (v[1]*v[5]-v[2]*v[4])*invdet;
		a[3] = a1*invdet;
		a[4] = (v[0]*v[8]-v[2]*v[6])*invdet;
		a[5] = (v[3]*v[2]-v[0]*v[5])*invdet;
		a[6] = a2*invdet;
		a[7] = (v[6]*v[1]-v[0]*v[7])*invdet;
		a[8] = (v[0]*v[4]-v[3]*v[1])*invdet;

		return a;
	}
	Vec3f multply(const Vec3d & x)
	{
		Vec3f r;
		r[0] = v[0]*x[0]+v[1]*x[1]+v[2]*x[2];
		r[1] = v[3]*x[0]+v[4]*x[1]+v[5]*x[2];
		r[2] = v[6]*x[0]+v[7]*x[1]+v[8]*x[2];
		return r;
	}

};


void get_circumcenter(Vec3f &v0, Vec3f &v1,Vec3f &v2,Vec3f &v3, Vec3f &c, float &r)
{
	Vec3d b;
	b[0] = v0[0]*v0[0] - v3[0]*v3[0] + v0[1]*v0[1] - v3[1]*v3[1] + v0[2]*v0[2] - v3[2]*v3[2];
	b[1] = v1[0]*v1[0] - v3[0]*v3[0] + v1[1]*v1[1] - v3[1]*v3[1] + v1[2]*v1[2] - v3[2]*v3[2];
	b[2] = v2[0]*v2[0] - v3[0]*v3[0] + v2[1]*v2[1] - v3[1]*v3[1] + v2[2]*v2[2] - v3[2]*v3[2];

	matrix3x3d A(2.0*(v0[0]-v3[0]), 2.0*(v0[1]-v3[1]),2.0*(v0[2]-v3[2]),
				 2*(v1[0]-v3[0]), 2*(v1[1]-v3[1]),2*(v1[2]-v3[2]),
				 2*(v2[0]-v3[0]), 2*(v2[1]-v3[1]),2*(v2[2]-v3[2]));

	c = (A.inverse()).multply(b);
	r = dist(v0,c);
}



#endif