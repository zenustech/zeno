// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file c_loops_2d.cc
 * \brief Function implementations for the 2D loop classes. */

#include "c_loops_2d.hh"

namespace voro {

/** Initializes a c_loop_subset_2d object to scan over all particles within a
 * given circle.
 * \param[in] (vx,vy) the position vector of the center of the circle.
 * \param[in] r the radius of the circle.
 * \param[in] bounds_test whether to do detailed bounds checking. If this is
 *                        false then the class will loop over all particles in
 *                        blocks that overlap the given circle. If it is true,
 *                        the particle will only loop over the particles which
 *                        actually lie within the circle.
 * \return True if there is any valid point to loop over, false otherwise. */
void c_loop_subset_2d::setup_circle(double vx,double vy,double r,bool bounds_test) {
	if(bounds_test) {mode=circle;v0=vx;v1=vy;v2=r*r;} else mode=no_check;
	ai=step_int((vx-ax-r)*xsp);
	bi=step_int((vx-ax+r)*xsp);
	aj=step_int((vy-ay-r)*ysp);
	bj=step_int((vy-ay+r)*ysp);
	setup_common();
}

/** Initializes the class to loop over all particles in a rectangular subgrid
 * of blocks.
 * \param[in] (ai_,bi_) the subgrid range in the x-direction, inclusive of both
 *                      ends.
 * \param[in] (aj_,bj_) the subgrid range in the y-direction, inclusive of both
 *                      ends.
 * \return True if there is any valid point to loop over, false otherwise. */
void c_loop_subset_2d::setup_intbox(int ai_,int bi_,int aj_,int bj_) {
	ai=ai_;bi=bi_;aj=aj_;bj=bj_;
	mode=no_check;
	setup_common();
}

/** Sets up all of the common constants used for the loop.
 * \return True if there is any valid point to loop over, false otherwise. */
void c_loop_subset_2d::setup_common() {
	if(!xperiodic) {
		if(ai<0) {ai=0;if(bi<0) bi=0;}
		if(bi>=nx) {bi=nx-1;if(ai>=nx) ai=nx-1;}
	}
	if(!yperiodic) {
		if(aj<0) {aj=0;if(bj<0) bj=0;}
		if(bj>=ny) {bj=ny-1;if(aj>=ny) aj=ny-1;}
	}
	ci=ai;cj=aj;
	di=i=step_mod(ci,nx);apx=px=step_div(ci,nx)*sx;
	dj=j=step_mod(cj,ny);apy=py=step_div(cj,ny)*sy;
	inc1=nx+di-step_mod(bi,nx);
	ij=di+nx*dj;
	q=0;
}

/** Starts the loop by finding the first particle within the container to
 * consider.
 * \return True if there is any particle to consider, false otherwise. */
bool c_loop_subset_2d::start() {
	while(co[ij]==0) {if(!next_block()) return false;}
	while(mode!=no_check&&out_of_bounds()) {
		q++;
		while(q>=co[ij]) {q=0;if(!next_block()) return false;}
	}
	return true;
}

/** Initializes the class to loop over all particles in a rectangular box.
 * \param[in] (xmin,xmax) the minimum and maximum x coordinates of the box.
 * \param[in] (ymin,ymax) the minimum and maximum y coordinates of the box.
 * \param[in] bounds_test whether to do detailed bounds checking. If this is
 *                        false then the class will loop over all particles in
 *                        blocks that overlap the given box. If it is true, the
 *                        particle will only loop over the particles which
 *                        actually lie within the box.
 * \return True if there is any valid point to loop over, false otherwise. */
void c_loop_subset_2d::setup_box(double xmin,double xmax,double ymin,double ymax,bool bounds_test) {
	if(bounds_test) {mode=rectangle;v0=xmin;v1=xmax;v2=ymin;v3=ymax;} else mode=no_check;
	ai=step_int((xmin-ax)*xsp);
	bi=step_int((xmax-ax)*xsp);
	aj=step_int((ymin-ay)*ysp);
	bj=step_int((ymax-ay)*ysp);
	setup_common();
}

/** Computes whether the current point is out of bounds, relative to the
 * current loop setup.
 * \return True if the point is out of bounds, false otherwise. */
bool c_loop_subset_2d::out_of_bounds() {
	double *pp(p[ij]+ps*q);
	if(mode==circle) {
		double fx(*pp+px-v0),fy(pp[1]+py-v1);
		return fx*fx+fy*fy>v2;
	} else {
		double f(*pp+px);if(f<v0||f>v1) return true;
		f=pp[1]+py;return f<v2||f>v3;
	}
}

/** Returns the next block to be tested in a loop, and updates the periodicity
 * vector if necessary. */
bool c_loop_subset_2d::next_block() {
	if(i<bi) {
		i++;
		if(ci<nx-1) {ci++;ij++;} else {ci=0;ij+=1-nx;px+=sx;}
		return true;
	} else if(j<bj) {
		i=ai;ci=di;px=apx;j++;
		if(cj<ny-1) {cj++;ij+=inc1;} else {cj=0;ij+=inc1-nxy;py+=sy;}
		return true;
	} else return false;
}

/** Extends the memory available for storing the ordering. */
void particle_order::add_ordering_memory() {
	int *no=new int[size<<2],*nop=no,*opp=o;
	while(opp<op) *(nop++)=*(opp++);
	delete [] o;
	size<<=1;o=no;op=nop;
}

}
