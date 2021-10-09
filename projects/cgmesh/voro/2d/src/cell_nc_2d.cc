// Voro++, a cell-based Voronoi library
//
// Authors  : Chris H. Rycroft (LBL / UC Berkeley)
//            Cody Robert Dance (UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file cell_nc_2d.cc
 * \brief Function implementations for the non-convex 2D Voronoi classes. */

#include "cell_nc_2d.hh"

namespace voro {

void voronoicell_nonconvex_neighbor_2d::init(double xmin,double xmax,double ymin,double ymax) {
	nonconvex=exclude=false;
	init_base(xmin,xmax,ymin,ymax);
	*ne=-3;ne[1]=-2;ne[2]=-4;ne[3]=-1;
}

void voronoicell_nonconvex_base_2d::init_nonconvex_base(double xmin,double xmax,double ymin,double ymax,double wx0,double wy0,double wx1,double wy1) {
	xmin*=2;xmax*=2;ymin*=2;ymax*=2;
	int f0=face(xmin,xmax,ymin,ymax,wx0,wy0),
	    f1=face(xmin,xmax,ymin,ymax,wx1,wy1);

	*pts=0;pts[1]=0;
	pts[2]=wx0;pts[3]=wy0;p=4;
	if(f0!=f1||wx0*wy1<wx1*wy0) {
		do {
			if(f0>1) {
				if(f0==2) {pts[p++]=xmin;pts[p++]=ymin;}
				else {pts[p++]=xmax;pts[p++]=ymin;}
			} else {
				if(f0==0) {pts[p++]=xmax;pts[p++]=ymax;}
				else {pts[p++]=xmin;pts[p++]=ymax;}
			}
			f0++;f0&=3;
		} while(f0!=f1);
	}
	pts[p++]=wx1;pts[p++]=wy1;
	p>>=1;

	int i,*q=ed;
	*(q++)=1;*(q++)=p-1;
	for(i=1;i<p-1;i++) {*(q++)=i+1;*(q++)=i-1;}
	*(q++)=0;*q=p-2;

	exclude=true;
	if(wx0*wy1>wx1*wy0) nonconvex=false;
	else {
		nonconvex=true;
		if(wx0*wx1+wy0*wy1>0) {
			*reg=wy0+wy1;reg[1]=-wx0-wx1;
		} else {
			*reg=wx1-wx0;reg[1]=wy1-wy0;
		}
	}
	reg[2]=wx0;reg[3]=wy0;
	reg[4]=wx1;reg[5]=wy1;
}

void voronoicell_nonconvex_neighbor_2d::init_nonconvex(double xmin,double xmax,double ymin,double ymax,double wx0,double wy0,double wx1,double wy1) {
	init_nonconvex_base(xmin,xmax,ymin,ymax,wx0,wy0,wx1,wy1);
	*ne=-5;
	for(int i=1;i<p-1;i++) ne[i]=-99;
	ne[p-1]=-5;
}

/** Cuts the Voronoi cell by a particle whose center is at a separation of
 * (x,y) from the cell center. The value of rsq should be initially set to
 * \f$x^2+y^2\f$.
 * \param[in] (x,y) the normal vector to the plane.
 * \param[in] rsq the distance along this vector of the plane.
 * \return False if the plane cut deleted the cell entirely, true otherwise. */
template<class vc_class>
bool voronoicell_nonconvex_base_2d::nplane_nonconvex(vc_class &vc,double x,double y,double rsq,int p_id) {
	int up=0,*edd;
	double u,rx,ry,sx,sy;

	if(x*(*reg)+y*reg[1]<0) {edd=ed;rx=reg[2];ry=reg[3];sx=*reg;sy=reg[1];}
	else {edd=ed+1;rx=reg[4];ry=reg[5];sx=-*reg;sy=-reg[1];}

	up=*edd;u=pos(x,y,rsq,up);
	while(u<tolerance) {
		up=edd[2*up];if(up==0) return true;
		if(pts[2*up]*sx+pts[2*up+1]*sy>0&&rx*pts[2*up]+ry*pts[2*up+1]>0) return true;
		u=pos(x,y,rsq,up);
	}

	return nplane_cut(vc,x,y,rsq,p_id,u,up);
}

inline int voronoicell_nonconvex_base_2d::face(double xmin,double xmax,double ymin,double ymax,double &wx,double &wy) {
	if(wy>0) {
		if(xmin*wy>ymax*wx) {wy*=xmin/wx;wx=xmin;return 2;}
		if(xmax*wy>ymax*wx) {wx*=ymax/wy;wy=ymax;return 1;}
		wy*=xmax/wx;wx=xmax;return 0;
	}
	if(xmax*wy>ymin*wx) {wy*=xmax/wx;wx=xmax;return 0;}
	if(xmin*wy>ymin*wx) {wx*=ymin/wy;wy=ymin;return 3;}
	wy*=xmin/wx;wx=xmin;return 2;
}

void voronoicell_nonconvex_neighbor_2d::neighbors(vector<int> &v) {
	v.resize(p);
	for(int i=0;i<p;i++) v[i]=ne[i];
}

// Explicit instantiation
template bool voronoicell_nonconvex_base_2d::nplane_nonconvex(voronoicell_nonconvex_2d&,double,double,double,int);
template bool voronoicell_nonconvex_base_2d::nplane_nonconvex(voronoicell_nonconvex_neighbor_2d&,double,double,double,int);

}
